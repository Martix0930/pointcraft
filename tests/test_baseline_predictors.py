"""Unit tests for the M1 deterministic predictors (B1 extrusion, B2 volume fill).

Tiny hand-constructed scenes with checkable voxel sets; contract-format output.
"""
import numpy as np

from pointcraft.baseline import (
    estimate_ground_k,
    footprint_volume_fill,
    naive_roof_extrusion,
)
from pointcraft.voxelization import VoxelGrid


def _grid(shape=(5, 5, 10)):
    return VoxelGrid(origin=np.zeros(3), voxel_size=1.0, shape=np.asarray(shape, np.int64))


def _as_set(coords):
    return {tuple(int(v) for v in row) for row in coords}


# --- B1 naive roof extrusion --------------------------------------------------
def test_b1_extrudes_building_column_to_ground():
    grid = _grid()
    # five ground returns at k=0 + one roof return at (2,2,5).
    partial = np.array(
        [[0, 0, 0], [4, 4, 0], [0, 4, 0], [4, 0, 0], [1, 1, 0], [2, 2, 5]],
        dtype=np.int32,
    )
    assert estimate_ground_k(partial, grid) == 0
    pred = naive_roof_extrusion(partial, grid)
    # building column (2,2) extruded k=0..5; flat-ground columns (height<3) skipped.
    assert _as_set(pred) == {(2, 2, k) for k in range(6)}


def test_b1_contract_format_and_inbounds():
    grid = _grid()
    partial = np.array([[1, 1, 0], [2, 2, 7], [3, 3, 0]], dtype=np.int32)
    pred = naive_roof_extrusion(partial, grid)
    assert pred.dtype == np.int32 and pred.ndim == 2 and pred.shape[1] == 3
    assert grid.in_bounds(pred).all()
    assert len(_as_set(pred)) == pred.shape[0]  # unique


def test_b1_empty_input():
    grid = _grid()
    pred = naive_roof_extrusion(np.zeros((0, 3), np.int32), grid)
    assert pred.shape == (0, 3)


def test_b1_respects_explicit_ground_and_threshold():
    grid = _grid()
    partial = np.array([[2, 2, 6]], dtype=np.int32)
    # explicit ground_k=4, top=6 -> height 2 < min_building_height 3 -> skipped.
    assert naive_roof_extrusion(partial, grid, ground_k=4).shape[0] == 0
    # lower threshold -> extruded k=4..6.
    pred = naive_roof_extrusion(partial, grid, ground_k=4, min_building_height=2)
    assert _as_set(pred) == {(2, 2, 4), (2, 2, 5), (2, 2, 6)}


# --- B2 footprint volume fill -------------------------------------------------
def _footprint_3x3():
    """3x3 footprint block (i,j in {1,2,3}); each column has base k=2, top k=5."""
    cols = [(i, j) for i in (1, 2, 3) for j in (1, 2, 3)]
    ct = np.array([[i, j, k] for (i, j) in cols for k in (2, 5)], dtype=np.int32)
    return ct


def test_b2_shell_roof_walls_and_hollow_interior():
    grid = _grid()
    ct = _footprint_3x3()
    pred = footprint_volume_fill(ct, grid, mode="shell")
    s = _as_set(pred)
    # 8 perimeter columns filled k=2..5 (32) + hollow centre (2,2) only k=2,5.
    assert len(s) == 34
    # centre column is hollow: no wall voxels between base and top.
    assert (2, 2, 3) not in s and (2, 2, 4) not in s
    assert (2, 2, 2) in s and (2, 2, 5) in s
    # a perimeter column is solid base->top.
    assert all((1, 1, k) in s for k in range(2, 6))


def test_b2_solid_fills_every_column():
    grid = _grid()
    ct = _footprint_3x3()
    pred = footprint_volume_fill(ct, grid, mode="solid")
    # all 9 columns filled k=2..5 -> 36 voxels including the centre.
    assert _as_set(pred) == {(i, j, k) for i in (1, 2, 3) for j in (1, 2, 3) for k in range(2, 6)}


def test_b2_contract_format_and_empty():
    grid = _grid()
    pred = footprint_volume_fill(_footprint_3x3(), grid, mode="shell")
    assert pred.dtype == np.int32 and pred.shape[1] == 3
    assert grid.in_bounds(pred).all()
    assert footprint_volume_fill(np.zeros((0, 3), np.int32), grid).shape == (0, 3)
