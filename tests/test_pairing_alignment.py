"""M0 Phase D regression tests: grid-equality, building alignment, end-to-end.

These guard the data-pairing invariants the contract depends on:
  * partial & target live on the SAME grid (shared origin/voxel_size/grid_shape);
  * a building's target footprint overlaps its observed roof voxels;
  * the end-to-end run_m0 pipeline emits a numpy-loadable .npz with every field.

(world↔index round-trip is covered in tests/test_voxel_grid.py.)
"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from pointcraft.data import (
    compute_masks,
    load_lod2_meshes,
    voxelize_partial,
    voxelize_target,
)
from pointcraft.voxelization import VoxelGrid

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "test_data" / "m0_data_pairing"
BOUNDS = [0.0, 0.0, 0.0, 4.0, 4.0, 4.0]
VOXEL_SIZE = 1.0


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
def paired(grid):
    cp, fp = voxelize_partial(_lidar_xyz(), grid)
    verts, faces = load_lod2_meshes(str(FIX / "tiny_lod2_cube.obj"))
    ct, occ, sem = voxelize_target(verts, faces, grid, seed=0)
    obs, unobs = compute_masks(ct, cp, grid)
    return dict(grid=grid, cp=cp, ct=ct, sem=sem, obs=obs)


def test_grid_equality_partial_and_target(paired):
    # Both coordinate sets must be valid indices on the one shared grid.
    g = paired["grid"]
    cp, ct = paired["cp"], paired["ct"]
    # Both coordinate sets are valid indices on the one shared grid_shape,
    # i.e. 0 <= idx < shape elementwise (in_bounds asserts both halves).
    assert g.in_bounds(cp).all()
    assert g.in_bounds(ct).all()
    assert (cp.min(axis=0) >= 0).all() and (cp.max(axis=0) < g.shape).all()
    assert (ct.min(axis=0) >= 0).all() and (ct.max(axis=0) < g.shape).all()


def test_building_footprint_overlaps_observed_roof(paired):
    cp, ct, sem, obs = paired["cp"], paired["ct"], paired["sem"], paired["obs"]
    # observed roof voxels of the LiDAR partial (its roof sits at k=3)
    partial_roof_ij = {tuple(r) for r in cp[cp[:, 2] == 3][:, :2].tolist()}
    # target building roof footprint (semantic roof)
    target_roof_ij = {tuple(r) for r in ct[sem == 3][:, :2].tolist()}
    assert partial_roof_ij, "fixture must have observed roof voxels"
    assert target_roof_ij & partial_roof_ij, "target roof must overlap observed roof"
    # and those overlapping roof voxels are flagged observed in the mask
    assert (obs[(sem == 3)] == 1).any()


def test_end_to_end_run_m0_fixture_writes_loadable_npz(tmp_path):
    spec = importlib.util.spec_from_file_location("run_m0", REPO / "scripts" / "run_m0.py")
    run_m0 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_m0)
    out = tmp_path / "sample.npz"
    rc = run_m0.main(["--fixture", "--out", str(out)])
    assert rc == 0 and out.exists()

    z = np.load(out)  # no allow_pickle
    for field in ("coords_partial", "feats_partial", "coords_target", "occ_target",
                  "sem_target", "observed_mask", "unobserved_mask", "metadata"):
        assert field in z.files
    md = json.loads(str(z["metadata"]))
    assert md["dataset_version"] == "v0.2"
    assert md["feature_layout"] == ["height", "point_count"]
    # partial recovers both roof (k=3) and ground (k=0) — the grid-extent fix
    kp = z["coords_partial"][:, 2]
    assert set(np.unique(kp).tolist()) == {0, 3}
