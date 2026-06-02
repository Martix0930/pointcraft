"""Masks + .npz sample writer per docs/02_DATA_CONTRACT.md (M0-4).

Brings the partial input and the CityGML/LOD2 target (both on the SAME
`VoxelGrid`) together into one on-disk training sample:

  * `compute_masks` — derive the required observed / unobserved masks (D4/D6):
        unobserved = occupied_target ∧ ¬observed   (= 1 - observed here, since
                     every stored target voxel is occupied)
    The "observed" test is **class-aware** (D6, `dataset_version v0.2`):
      - horizontal surfaces (roof/ground): observed if a partial voxel sits at the
        same `(i,j)` within `|Δk| <= z_tol` — corrects the sub-voxel z-quantization
        that otherwise flags ~55 % of seen roofs "unobserved".
      - vertical surfaces (facade): observed only on an EXACT partial hit that is a
        *genuine mid-wall* cell (>= `wall_margin` voxels from the column's lowest
        and highest target voxel), excluding ground/roof points that merely clip the
        bottom/top wall voxel. This keeps the facade completion region honest
        (~30 % of facade carries real aerial signal on the Tokyo tile).
    With `sem_target=None` it falls back to the legacy exact set-membership
    (`dataset_version v0.1`).
  * `build_metadata` — assemble the metadata block from the grid + provenance.
  * `write_sample_npz` — write every contract field with the exact dtype/shape,
        plus metadata as a JSON string (no pickle → loadable with numpy.load
        without allow_pickle).

Pure numpy + json; no learning.
"""
from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np

from ..voxelization import VoxelGrid
from .partial import FEATURE_LAYOUT_V01

DATASET_VERSION = "v0.2"

#: Default class-aware mask parameters (D6). Labels per docs/02_DATA_CONTRACT.md.
HORIZONTAL_LABELS = (1, 3)   # ground, roof
VERTICAL_LABELS = (4,)       # facade
DEFAULT_Z_TOL = 1            # voxels, horizontal surfaces
DEFAULT_WALL_MARGIN = 2      # voxels, facade genuine-mid-wall band


def _ravel(coords: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """Flatten (N,3) in-bounds voxel indices to unique int64 keys for set ops."""
    c = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    sj, sk = int(shape[1]), int(shape[2])
    return c[:, 0] * (sj * sk) + c[:, 1] * sk + c[:, 2]


def compute_masks(
    coords_target: np.ndarray,
    coords_partial: np.ndarray,
    grid: VoxelGrid,
    *,
    sem_target: np.ndarray | None = None,
    z_tol: int = DEFAULT_Z_TOL,
    horizontal_labels: Sequence[int] = HORIZONTAL_LABELS,
    vertical_labels: Sequence[int] = VERTICAL_LABELS,
    wall_margin: int = DEFAULT_WALL_MARGIN,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-target-voxel observed / unobserved masks (uint8), aligned to
    `coords_target` rows. Both coord arrays must hold in-bounds indices on `grid`.

    If `sem_target` is None: legacy exact set-membership (v0.1). Otherwise the
    class-aware rule (v0.2, D6) above is used; pass `z_tol=0, wall_margin=0` to
    reduce it back toward exact while keeping the per-class structure.
    """
    shape = grid.shape
    key_t = _ravel(coords_target, shape)
    key_p = _ravel(coords_partial, shape)
    exact = np.isin(key_t, key_p)

    if sem_target is None:
        observed = exact.astype(np.uint8)
        return observed, (1 - observed).astype(np.uint8)

    coords_target = np.asarray(coords_target, dtype=np.int64).reshape(-1, 3)
    sem = np.asarray(sem_target).reshape(-1)
    sj, sk = int(shape[1]), int(shape[2])
    i, j, k = coords_target[:, 0], coords_target[:, 1], coords_target[:, 2]
    observed = np.zeros(coords_target.shape[0], dtype=bool)

    # --- horizontal surfaces (roof/ground): z-tolerance about the same column ---
    hmask = np.isin(sem, list(horizontal_labels))
    obs_h = exact & hmask
    for d in range(1, int(z_tol) + 1):
        for sgn in (-1, 1):
            shifted = i * (sj * sk) + j * sk + (k + sgn * d)
            obs_h |= hmask & np.isin(shifted, key_p)
    observed |= obs_h

    # --- vertical surfaces (facade): exact hit AND genuine mid-wall ---
    vmask = np.isin(sem, list(vertical_labels))
    if vmask.any():
        col = i * (10 ** 7) + j  # per-(i,j) column key
        ucol, inv = np.unique(col, return_inverse=True)
        col_base = np.full(ucol.shape[0], 1 << 30, dtype=np.int64)
        col_top = np.full(ucol.shape[0], -(1 << 30), dtype=np.int64)
        np.minimum.at(col_base, inv, k)
        np.maximum.at(col_top, inv, k)
        midwall = (k >= col_base[inv] + wall_margin) & (k <= col_top[inv] - wall_margin)
        observed |= exact & vmask & midwall

    observed = observed.astype(np.uint8)
    return observed, (1 - observed).astype(np.uint8)


def build_metadata(
    grid: VoxelGrid,
    *,
    tile_id: str,
    crs: str,
    source_files: Sequence[str],
    dataset_version: str = DATASET_VERSION,
    feature_layout: Sequence[str] = tuple(FEATURE_LAYOUT_V01),
) -> dict[str, Any]:
    """Assemble the metadata block (all contract keys) from grid + provenance."""
    origin = grid.origin.astype(float)
    shape = grid.shape.astype(int)
    maxc = origin + shape * grid.voxel_size
    return {
        "tile_id": str(tile_id),
        "voxel_size": float(grid.voxel_size),
        "origin": origin.tolist(),
        "bounds": [
            float(origin[0]), float(origin[1]), float(origin[2]),
            float(maxc[0]), float(maxc[1]), float(maxc[2]),
        ],
        "grid_shape": shape.tolist(),
        "crs": str(crs),
        "source_files": [str(s) for s in source_files],
        "dataset_version": str(dataset_version),
        "feature_layout": [str(f) for f in feature_layout],
    }


def write_sample_npz(
    path: str,
    *,
    coords_partial: np.ndarray,
    feats_partial: np.ndarray,
    coords_target: np.ndarray,
    occ_target: np.ndarray,
    sem_target: np.ndarray,
    observed_mask: np.ndarray,
    unobserved_mask: np.ndarray,
    metadata: dict[str, Any],
) -> str:
    """Write one paired voxel sample to `path` (.npz) per the data contract.

    All arrays are cast to their contract dtypes. `metadata` is stored as a 0-d
    string array of JSON (key ``metadata``); load with
    ``json.loads(str(np.load(path)["metadata"]))``.
    """
    np.savez_compressed(
        path,
        coords_partial=np.asarray(coords_partial, dtype=np.int32).reshape(-1, 3),
        feats_partial=np.asarray(feats_partial, dtype=np.float32).reshape(
            -1, len(metadata["feature_layout"])
        ),
        coords_target=np.asarray(coords_target, dtype=np.int32).reshape(-1, 3),
        occ_target=np.asarray(occ_target, dtype=np.uint8).reshape(-1),
        sem_target=np.asarray(sem_target, dtype=np.int64).reshape(-1),
        observed_mask=np.asarray(observed_mask, dtype=np.uint8).reshape(-1),
        unobserved_mask=np.asarray(unobserved_mask, dtype=np.uint8).reshape(-1),
        metadata=np.array(json.dumps(metadata, ensure_ascii=False)),
    )
    return path


def load_sample_metadata(npz) -> dict[str, Any]:
    """Decode the metadata JSON from a loaded .npz (or its ``metadata`` entry)."""
    entry = npz["metadata"] if hasattr(npz, "__getitem__") else npz
    return json.loads(str(entry))
