"""Tests for masks + .npz writer (M0-4), end-to-end on the cube fixture.

Combines partial (LiDAR) + target (LOD2 shell) on ONE shared grid, derives the
required masks (D4), writes an .npz, and reloads it with plain numpy.load to
assert the data contract (fields, dtypes, shapes, metadata keys).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from pointcraft.data import (
    FEATURE_LAYOUT_V01,
    build_metadata,
    compute_masks,
    load_lod2_meshes,
    voxelize_partial,
    voxelize_target,
    write_sample_npz,
)
from pointcraft.voxelization import VoxelGrid

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "test_data" / "m0_data_pairing"
BOUNDS = [0.0, 0.0, 0.0, 4.0, 4.0, 4.0]
VOXEL_SIZE = 1.0


def test_class_aware_masks_v02():
    """D6 / v0.2: z-tolerance rescues a quantization-straddled roof; a facade is
    observed only on an exact mid-wall hit (ground/roof-clip hits excluded)."""
    grid = VoxelGrid.from_bounds([0, 0, 0, 10, 1, 10], 1.0)
    # One building column (i=2): ground@k0, facade k1..5, roof@k6.
    ct = np.array(
        [[2, 0, 0], [2, 0, 1], [2, 0, 2], [2, 0, 3], [2, 0, 4], [2, 0, 5], [2, 0, 6]],
        dtype=np.int32,
    )
    sem = np.array([1, 4, 4, 4, 4, 4, 3], dtype=np.int64)  # ground/facade*5/roof
    # Partial: exact mid-wall facade hit (k3); exact base facade hit (k1, a
    # ground-clip); a roof point one voxel above the roof (k7, z-quantization).
    cp = np.array([[2, 0, 3], [2, 0, 1], [2, 0, 7]], dtype=np.int32)

    def row(i, j, k):
        return int(np.where((ct == [i, j, k]).all(1))[0][0])

    obs_v02, unobs = compute_masks(ct, cp, grid, sem_target=sem)
    obs_exact, _ = compute_masks(ct, cp, grid)  # sem=None -> legacy exact

    # col_base=0, col_top=6, wall_margin=2 -> mid-wall facade is k in [2,4].
    assert obs_v02[row(2, 0, 3)] == 1            # mid-wall facade, exact hit
    assert obs_v02[row(2, 0, 1)] == 0            # base facade hit excluded (genuine)
    assert obs_v02[row(2, 0, 6)] == 1            # roof rescued by z-tol (point at k7)
    assert (obs_v02 + unobs == 1).all()          # complementary

    # Legacy exact differs exactly where the v0.2 rule intervenes:
    assert obs_exact[row(2, 0, 1)] == 1          # exact hit counts under v0.1
    assert obs_exact[row(2, 0, 6)] == 0          # no exact roof hit under v0.1

CONTRACT_FIELDS = {
    "coords_partial": np.int32,
    "feats_partial": np.float32,
    "coords_target": np.int32,
    "occ_target": np.uint8,
    "sem_target": np.int64,
    "observed_mask": np.uint8,
    "unobserved_mask": np.uint8,
}
META_KEYS = {
    "tile_id", "voxel_size", "origin", "bounds", "grid_shape", "crs",
    "source_files", "dataset_version", "feature_layout",
}


def _lidar_xyz():
    xyz = []
    with open(FIX / "tiny_lidar_points.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            xyz.append([float(row["x"]), float(row["y"]), float(row["z"])])
    return np.array(xyz, dtype=np.float64)


@pytest.fixture
def grid():
    return VoxelGrid.from_bounds(BOUNDS, VOXEL_SIZE)


@pytest.fixture
def pieces(grid):
    cp, fp = voxelize_partial(_lidar_xyz(), grid)
    verts, faces = load_lod2_meshes(str(FIX / "tiny_lod2_cube.obj"))
    ct, occ, sem = voxelize_target(verts, faces, grid, seed=0)
    obs, unobs = compute_masks(ct, cp, grid)
    return dict(cp=cp, fp=fp, ct=ct, occ=occ, sem=sem, obs=obs, unobs=unobs)


def test_masks_are_complementary_and_aligned(pieces):
    obs, unobs, ct = pieces["obs"], pieces["unobs"], pieces["ct"]
    assert obs.shape == (ct.shape[0],) and unobs.shape == (ct.shape[0],)
    assert np.array_equal(obs + unobs, np.ones(ct.shape[0], dtype=np.uint8))


def test_observed_includes_roof_unobserved_includes_midfacade(pieces):
    ct, obs, unobs, sem = pieces["ct"], pieces["obs"], pieces["unobs"], pieces["sem"]
    k = ct[:, 2]
    # roof at k=3 was seen by the aerial partial -> observed
    assert (obs[(k == 3)] == 1).any()
    # mid-height facade (k=1,2) was NOT seen -> unobserved (the completion region)
    midfacade = np.isin(k, [1, 2])
    assert midfacade.any() and (unobs[midfacade] == 1).all()


def test_masks_match_bruteforce_membership(grid, pieces):
    # independent check: observed iff the exact target voxel appears in partial
    ct, cp, obs = pieces["ct"], pieces["cp"], pieces["obs"]
    partial_set = {tuple(r) for r in cp.tolist()}
    brute = np.array([tuple(r) in partial_set for r in ct.tolist()], dtype=np.uint8)
    assert np.array_equal(obs, brute)


def test_write_and_reload_roundtrip(tmp_path, grid, pieces):
    meta = build_metadata(
        grid, tile_id="tiny_synthetic_0", crs="LOCAL_SYNTHETIC",
        source_files=["tiny_lidar_points.csv", "tiny_lod2_cube.obj"],
    )
    out = str(tmp_path / "sample.npz")
    write_sample_npz(
        out,
        coords_partial=pieces["cp"], feats_partial=pieces["fp"],
        coords_target=pieces["ct"], occ_target=pieces["occ"],
        sem_target=pieces["sem"], observed_mask=pieces["obs"],
        unobserved_mask=pieces["unobs"], metadata=meta,
    )
    # reload WITHOUT allow_pickle -> proves no pickled objects
    z = np.load(out)
    for field, dtype in CONTRACT_FIELDS.items():
        assert field in z.files, f"missing field {field}"
        assert z[field].dtype == dtype, f"{field} dtype {z[field].dtype} != {dtype}"
    # metadata decodes from JSON and has all keys with sane values
    md = json.loads(str(z["metadata"]))
    assert META_KEYS.issubset(md.keys())
    assert md["voxel_size"] == VOXEL_SIZE
    assert md["grid_shape"] == [4, 4, 4]
    assert md["origin"] == [0.0, 0.0, 0.0]
    assert md["dataset_version"] == "v0.2"
    assert md["feature_layout"] == FEATURE_LAYOUT_V01
    # shapes are mutually consistent
    M = z["coords_target"].shape[0]
    assert z["occ_target"].shape == (M,)
    assert z["sem_target"].shape == (M,)
    assert z["observed_mask"].shape == (M,)
    assert z["feats_partial"].shape == (z["coords_partial"].shape[0], len(FEATURE_LAYOUT_V01))
