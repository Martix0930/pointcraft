"""LOD2 mesh → target SHELL occupancy + deterministic semantics (M0-3).

Per decision D2 the building target is a **shell**: only the LOD2 *surface* is
voxelized (no interior fill). We sample each triangle uniformly into world-space
points (barycentric sampling, same idea as the legacy
`LOD2Rasterizer.colored_point_samples`, but geometry-only — no texture needed),
map the samples onto the SAME `VoxelGrid` used for the partial input, and merge.

Semantics are assigned **deterministically from face orientation** (no learning):
    |n_z| >= roof_nz  → near-horizontal surface → roof   (label 3)
    otherwise          → near-vertical surface  → facade (label 4)
Per-voxel label is the majority vote of the samples landing in it; ties go to the
lower label id (roof) for determinism.

Known limitation: a building's *bottom* face is near-horizontal too, so it is
labelled `roof` under this rule. Real PLATEAU LOD2 buildings rarely expose a
separate ground-level bottom face; this is acceptable for M0 and documented in
the session log. Terrain/vegetation labels (1/5) are out of scope until a terrain
source is paired in.

Pure numpy (+ optional parse_obj for loading); no learning.
"""
from __future__ import annotations

import glob
import logging
import os

import numpy as np

from ..voxelization import VoxelGrid

log = logging.getLogger(__name__)

ROOF_LABEL = 3
FACADE_LABEL = 4
#: Candidate labels in tie-break priority order (argmax of equal votes -> roof).
_LABELS = np.array([ROOF_LABEL, FACADE_LABEL], dtype=np.int64)


def _sample_triangle(tri: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly sample `n` points inside a triangle (3,3) -> (n,3) world XYZ."""
    u1 = rng.random(n)
    u2 = rng.random(n)
    r1 = np.sqrt(u1)
    b0 = (1.0 - r1)[:, None]
    b1 = (r1 * (1.0 - u2))[:, None]
    b2 = (r1 * u2)[:, None]
    return b0 * tri[0] + b1 * tri[1] + b2 * tri[2]


def voxelize_target(
    verts: np.ndarray,
    faces,
    grid: VoxelGrid,
    *,
    roof_nz: float = 0.7,
    spacing: float | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxelize a LOD2 mesh surface (shell) into target occupancy + semantics.

    Args:
        verts:    (V, 3) float mesh vertices (world XYZ, absolute z).
        faces:    iterable of (v_idx_tuple3, vt_idx_tuple3, material) as returned
                  by `pointcraft.data.lod2.parse_obj` (only v_idx is used).
        grid:     the shared VoxelGrid (same instance as the partial input).
        roof_nz:  |normalized n_z| threshold separating roof from facade.
        spacing:  surface sample spacing in meters (default voxel_size/2, so each
                  voxel-sized surface patch is hit by multiple samples).
        seed:     RNG seed (deterministic output).

    Returns:
        coords_target: (M, 3) int32 unique voxel indices, sorted.
        occ_target:    (M,)  uint8, all 1 (only occupied shell voxels are stored).
        sem_target:    (M,)  int64 semantic label per voxel (roof=3 / facade=4).
    """
    verts = np.asarray(verts, dtype=np.float64)
    if spacing is None:
        spacing = grid.voxel_size / 2.0
    rng = np.random.default_rng(seed)

    pts_chunks: list[np.ndarray] = []
    lab_chunks: list[np.ndarray] = []
    for vi, _vti, _mat in faces:
        tri = verts[list(vi)]
        cross = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = float(np.linalg.norm(cross))
        if norm <= 0.0:
            continue  # degenerate triangle
        area = 0.5 * norm
        nz = abs(cross[2] / norm)
        label = ROOF_LABEL if nz >= roof_nz else FACADE_LABEL
        n = int(np.clip(np.ceil(area / (spacing * spacing)), 1, 200_000))
        p = _sample_triangle(tri, n, rng)
        pts_chunks.append(p)
        lab_chunks.append(np.full(n, label, dtype=np.int64))

    if not pts_chunks:
        return (
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0,), dtype=np.uint8),
            np.zeros((0,), dtype=np.int64),
        )

    pts = np.concatenate(pts_chunks, axis=0)
    labels = np.concatenate(lab_chunks, axis=0)

    idx = grid.world_to_index(pts)
    inb = grid.in_bounds(idx)
    n_drop = int((~inb).sum())
    if n_drop:
        log.info("target: dropped %d/%d out-of-range surface samples", n_drop, len(pts))
    idx = idx[inb]
    labels = labels[inb]

    if idx.shape[0] == 0:
        return (
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0,), dtype=np.uint8),
            np.zeros((0,), dtype=np.int64),
        )

    coords, inverse = np.unique(idx, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    m = coords.shape[0]

    votes = np.zeros((m, _LABELS.shape[0]), dtype=np.int64)
    for col, lab in enumerate(_LABELS):
        sel = labels == lab
        if sel.any():
            np.add.at(votes[:, col], inverse[sel], 1)
    sem_target = _LABELS[np.argmax(votes, axis=1)].astype(np.int64)

    occ_target = np.ones(m, dtype=np.uint8)
    log.info(
        "target: %d surface samples -> %d shell voxels (roof=%d, facade=%d)",
        idx.shape[0],
        m,
        int((sem_target == ROOF_LABEL).sum()),
        int((sem_target == FACADE_LABEL).sum()),
    )
    return coords.astype(np.int32), occ_target, sem_target


def _triangulate_ring(ring: np.ndarray):
    """Fan-triangulate a planar polygon ring (n>=3, may be closed) -> list of (3,3).

    Skips the degenerate triangle formed by a closing duplicate vertex via the
    caller's area check.
    """
    v0 = ring[0]
    return [np.stack([v0, ring[i], ring[i + 1]]) for i in range(1, ring.shape[0] - 1)]


def voxelize_citygml_target(
    surfaces,
    grid: VoxelGrid,
    *,
    spacing: float | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxelize CityGML typed LOD2 surfaces (shell, D2) into target occ + semantics.

    Unlike :func:`voxelize_target` (OBJ + face-normal heuristic), semantics come
    **directly from the CityGML surface type** carried by ``surfaces`` (D5): roof
    (3) / facade (4) / ground (1), with no geometry inference — so a building base
    is ``ground``, never a mislabelled roof.

    Args:
        surfaces: a ``pointcraft.data.citygml.TypedSurfaces`` (or any object with
                  ``.polygons`` list of ``(n_i,3)`` rings and aligned ``.labels``
                  int array), already reprojected to the grid CRS (EPSG:6677).
        grid:     the shared ``VoxelGrid`` (same instance as the partial input).
        spacing:  surface sample spacing in meters (default ``voxel_size/2``).
        seed:     RNG seed (deterministic output).

    Returns:
        coords_target: (M, 3) int32 unique voxel indices, sorted.
        occ_target:    (M,)  uint8, all 1 (occupied shell voxels only).
        sem_target:    (M,)  int64 semantic label per voxel (from surface type;
                       per-voxel majority vote of samples, ties -> lower label id).
    """
    if spacing is None:
        spacing = grid.voxel_size / 2.0
    rng = np.random.default_rng(seed)

    pts_chunks: list[np.ndarray] = []
    lab_chunks: list[np.ndarray] = []
    for ring, label in zip(surfaces.polygons, np.asarray(surfaces.labels)):
        ring = np.asarray(ring, dtype=np.float64)
        if ring.shape[0] < 3:
            continue
        label = int(label)
        for tri in _triangulate_ring(ring):
            cross = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            norm = float(np.linalg.norm(cross))
            if norm <= 0.0:
                continue  # degenerate (incl. closing-duplicate) triangle
            area = 0.5 * norm
            n = int(np.clip(np.ceil(area / (spacing * spacing)), 1, 200_000))
            pts_chunks.append(_sample_triangle(tri, n, rng))
            lab_chunks.append(np.full(n, label, dtype=np.int64))

    if not pts_chunks:
        return (
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0,), dtype=np.uint8),
            np.zeros((0,), dtype=np.int64),
        )

    pts = np.concatenate(pts_chunks, axis=0)
    labels = np.concatenate(lab_chunks, axis=0)

    idx = grid.world_to_index(pts)
    inb = grid.in_bounds(idx)
    n_drop = int((~inb).sum())
    if n_drop:
        log.info("citygml target: dropped %d/%d out-of-range samples", n_drop, len(pts))
    idx = idx[inb]
    labels = labels[inb]
    if idx.shape[0] == 0:
        return (
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0,), dtype=np.uint8),
            np.zeros((0,), dtype=np.int64),
        )

    coords, inverse = np.unique(idx, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    m = coords.shape[0]

    # Vote per voxel among the label ids actually present; ascending label order
    # makes argmax break ties toward the LOWER id (deterministic).
    present = np.unique(labels)  # sorted ascending
    votes = np.zeros((m, present.shape[0]), dtype=np.int64)
    for col, lab in enumerate(present):
        sel = labels == lab
        np.add.at(votes[:, col], inverse[sel], 1)
    sem_target = present[np.argmax(votes, axis=1)].astype(np.int64)

    occ_target = np.ones(m, dtype=np.uint8)
    counts = {int(l): int((sem_target == l).sum()) for l in present}
    log.info(
        "citygml target: %d samples -> %d shell voxels by label %s",
        idx.shape[0], m, counts,
    )
    return coords.astype(np.int32), occ_target, sem_target


def load_lod2_meshes(tile_dirs_or_objs) -> tuple[np.ndarray, list]:
    """Load + merge one or more LOD2 OBJ meshes into (verts, faces).

    Accepts a list of tile directories (uses the first *.obj in each) and/or
    direct *.obj file paths. Vertex indices are offset so faces remain valid
    after concatenation. Reuses `pointcraft.data.lod2.parse_obj`.
    """
    from .lod2 import parse_obj

    if isinstance(tile_dirs_or_objs, (str, os.PathLike)):
        tile_dirs_or_objs = [tile_dirs_or_objs]

    all_verts: list[np.ndarray] = []
    all_faces: list = []
    offset = 0
    for entry in tile_dirs_or_objs:
        if os.path.isdir(entry):
            objs = sorted(glob.glob(os.path.join(entry, "*.obj")))
            if not objs:
                continue
            obj_path = objs[0]
        else:
            obj_path = entry
        verts, _uvs, faces, _mtllib = parse_obj(obj_path)
        all_verts.append(verts)
        for vi, vti, mat in faces:
            all_faces.append((tuple(i + offset for i in vi), vti, mat))
        offset += len(verts)

    if not all_verts:
        return np.zeros((0, 3), dtype=np.float64), []
    return np.concatenate(all_verts, axis=0), all_faces
