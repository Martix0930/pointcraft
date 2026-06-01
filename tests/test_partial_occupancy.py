"""Tests for LiDAR → partial occupancy (M0-2).

Driven by the committed tiny fixture test_data/m0_data_pairing/, whose voxel
indices are hand-checkable (origin [0,0,0], voxel_size 1.0, grid 4x4x4). The
fixture is an aerial observation: roof points at z≈3 and ground points at z≈0,
with the cube's facades (z=1,2) deliberately UNobserved.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from pointcraft.data import FEATURE_LAYOUT_V01, voxelize_partial
from pointcraft.voxelization import VoxelGrid

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "test_data" / "m0_data_pairing"
BOUNDS = [0.0, 0.0, 0.0, 4.0, 4.0, 4.0]
VOXEL_SIZE = 1.0


def _load_lidar_xyz() -> np.ndarray:
    xyz = []
    with open(FIX / "tiny_lidar_points.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            xyz.append([float(row["x"]), float(row["y"]), float(row["z"])])
    return np.array(xyz, dtype=np.float64)


@pytest.fixture
def grid():
    return VoxelGrid.from_bounds(BOUNDS, VOXEL_SIZE)


@pytest.fixture
def partial(grid):
    return voxelize_partial(_load_lidar_xyz(), grid)


def test_outputs_have_contract_dtypes_and_shapes(partial):
    coords, feats = partial
    assert coords.dtype == np.int32 and coords.ndim == 2 and coords.shape[1] == 3
    assert feats.dtype == np.float32 and feats.shape[1] == len(FEATURE_LAYOUT_V01)
    assert coords.shape[0] == feats.shape[0]


def test_coords_are_unique_and_in_bounds(grid, partial):
    coords, _ = partial
    assert grid.in_bounds(coords).all()
    # no duplicate voxels
    assert np.unique(coords, axis=0).shape[0] == coords.shape[0]


def test_expected_voxel_count_and_point_count_conservation(partial):
    coords, feats = partial
    # 13 fixture points -> 12 occupied voxels (two roof points share (2,2,3)).
    assert coords.shape[0] == 12
    point_count = feats[:, FEATURE_LAYOUT_V01.index("point_count")]
    assert point_count.sum() == 13  # every input point accounted for


def test_roof_observed_ground_observed_facades_not(partial):
    coords, _ = partial
    k = coords[:, 2]
    assert (k == 3).any(), "roof voxels (k=3) must be observed"
    assert (k == 0).any(), "ground voxels (k=0) must be observed"
    # aerial LiDAR never sees the cube's facades -> no occupied voxel at k in {1,2}
    assert not np.isin(k, [1, 2]).any(), "facade voxels must be UNobserved"


def test_merged_voxel_height_is_mean_z(partial):
    coords, feats = partial
    hcol = FEATURE_LAYOUT_V01.index("height")
    pccol = FEATURE_LAYOUT_V01.index("point_count")
    # voxel (2,2,3) merges two roof points, both at z=3.0
    row = np.where((coords == [2, 2, 3]).all(axis=1))[0]
    assert row.size == 1
    assert feats[row[0], pccol] == 2.0
    assert feats[row[0], hcol] == pytest.approx(3.0)
    # a ground voxel sits at mean z ≈ 0.0
    g = np.where((coords == [0, 0, 0]).all(axis=1))[0]
    assert feats[g[0], hcol] == pytest.approx(0.0)


def test_empty_input_returns_empty_contract_arrays(grid):
    coords, feats = voxelize_partial(np.zeros((0, 3)), grid)
    assert coords.shape == (0, 3) and coords.dtype == np.int32
    assert feats.shape == (0, len(FEATURE_LAYOUT_V01)) and feats.dtype == np.float32


def test_out_of_range_points_are_dropped(grid):
    pts = np.array([[2.0, 2.0, 2.0], [99.0, 99.0, 99.0], [-5.0, 0.0, 0.0]])
    coords, feats = voxelize_partial(pts, grid)
    assert coords.shape[0] == 1
    assert coords[0].tolist() == [2, 2, 2]
