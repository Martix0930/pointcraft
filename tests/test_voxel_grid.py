"""Tests for the shared voxel grid utility (M0-1).

Uses the committed tiny fixtures in test_data/m0_voxel_grid/ so expected results
are hand-checkable. No heavy data, no learning.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from pointcraft.voxelization import VoxelGrid

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "test_data" / "m0_voxel_grid"


def _load_bounds():
    with open(FIX / "tiny_bounds.json", encoding="utf-8") as f:
        return json.load(f)


def _load_points():
    ids, xyz = [], []
    with open(FIX / "tiny_points.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids.append(int(row["id"]))
            xyz.append([float(row["x"]), float(row["y"]), float(row["z"])])
    return ids, np.array(xyz, dtype=np.float64)


def _load_expected():
    with open(FIX / "expected_indices.json", encoding="utf-8") as f:
        d = json.load(f)["indices"]
    return {int(k): v for k, v in d.items()}


@pytest.fixture
def grid():
    b = _load_bounds()
    return VoxelGrid.from_bounds(b["bounds"], b["voxel_size"])


def test_grid_shape_and_origin_match_fixture(grid):
    b = _load_bounds()
    assert grid.shape.tolist() == b["grid_shape"]
    assert grid.origin.tolist() == b["origin"]
    assert grid.voxel_size == b["voxel_size"]
    assert grid.num_voxels == int(np.prod(b["grid_shape"]))


def test_world_to_index_matches_expected(grid):
    ids, xyz = _load_points()
    expected = _load_expected()
    idx = grid.world_to_index(xyz)
    for row, pid in enumerate(ids):
        assert idx[row].tolist() == expected[pid], f"point id={pid}"


def test_boundary_coordinate_maps_up(grid):
    # A coordinate exactly on an integer boundary floors to the higher voxel.
    idx = grid.world_to_index([[1.0, 0.0, 0.0]])
    assert idx[0].tolist() == [1, 0, 0]


def test_center_roundtrip_is_stable(grid):
    # index -> center -> index returns the original index for all in-bounds voxels.
    ii, jj, kk = np.meshgrid(
        np.arange(grid.shape[0]),
        np.arange(grid.shape[1]),
        np.arange(grid.shape[2]),
        indexing="ij",
    )
    idx = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)
    centers = grid.index_to_center(idx)
    back = grid.world_to_index(centers)
    assert np.array_equal(back, idx)


def test_in_bounds_mask(grid):
    idx = np.array([[0, 0, 0], [3, 3, 3], [4, 0, 0], [-1, 0, 0], [0, 0, 4]])
    mask = grid.in_bounds(idx)
    assert mask.tolist() == [True, True, False, False, False]


def test_corner_is_origin_offset(grid):
    corner = grid.index_to_corner([[2, 1, 0]])
    assert corner[0].tolist() == [2.0, 1.0, 0.0]


def test_from_bounds_rejects_bad_voxel_size():
    with pytest.raises(ValueError):
        VoxelGrid.from_bounds([0, 0, 0, 1, 1, 1], 0.0)
