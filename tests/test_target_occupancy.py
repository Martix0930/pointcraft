"""Tests for LOD2 shell → target occupancy + semantics (M0-3).

Driven by the committed cube fixture test_data/m0_data_pairing/tiny_lod2_cube.obj
(x,y∈[1,3], z∈[0,3]) on the same 4x4x4 / voxel_size 1.0 grid as the partial test.
The shell must include the facades (z=1,2) that the aerial partial never observed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pointcraft.data import (
    FACADE_LABEL,
    ROOF_LABEL,
    load_lod2_meshes,
    voxelize_target,
)
from pointcraft.voxelization import VoxelGrid

REPO = Path(__file__).resolve().parents[1]
OBJ = REPO / "test_data" / "m0_data_pairing" / "tiny_lod2_cube.obj"
BOUNDS = [0.0, 0.0, 0.0, 4.0, 4.0, 4.0]
VOXEL_SIZE = 1.0


@pytest.fixture
def grid():
    return VoxelGrid.from_bounds(BOUNDS, VOXEL_SIZE)


@pytest.fixture
def target(grid):
    verts, faces = load_lod2_meshes(str(OBJ))
    return voxelize_target(verts, faces, grid, seed=0)


def test_contract_dtypes_and_shapes(target):
    coords, occ, sem = target
    assert coords.dtype == np.int32 and coords.shape[1] == 3
    assert occ.dtype == np.uint8 and occ.shape == (coords.shape[0],)
    assert sem.dtype == np.int64 and sem.shape == (coords.shape[0],)


def test_occ_all_one_and_coords_unique_in_bounds(grid, target):
    coords, occ, _ = target
    assert (occ == 1).all()
    assert grid.in_bounds(coords).all()
    assert np.unique(coords, axis=0).shape[0] == coords.shape[0]


def test_only_roof_and_facade_labels(target):
    _, _, sem = target
    assert np.isin(sem, [ROOF_LABEL, FACADE_LABEL]).all()


def test_roof_present_at_top_and_facade_present(target):
    coords, _, sem = target
    k = coords[:, 2]
    # roof of the cube is at z=3 -> voxel layer k=3
    roof_top = (sem == ROOF_LABEL) & (k == 3)
    assert roof_top.any(), "cube roof (label 3 at k=3) must exist"
    assert (sem == FACADE_LABEL).any(), "cube walls (facade label 4) must exist"


def test_facade_fills_mid_height_unobserved_by_aerial(target):
    # The aerial partial only saw roof (k=3) + ground (k=0). The target shell must
    # additionally contain facade voxels at the mid heights k=1 and k=2 — exactly
    # the completion region M4 cares about.
    coords, _, sem = target
    k = coords[:, 2]
    facade_mid = (sem == FACADE_LABEL) & np.isin(k, [1, 2])
    assert facade_mid.any(), "facade voxels must fill mid-height layers k=1,2"


def test_deterministic_across_runs(grid):
    verts, faces = load_lod2_meshes(str(OBJ))
    a = voxelize_target(verts, faces, grid, seed=0)
    b = voxelize_target(verts, faces, grid, seed=0)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    assert np.array_equal(a[2], b[2])


def test_empty_mesh_returns_empty_contract_arrays(grid):
    coords, occ, sem = voxelize_target(np.zeros((0, 3)), [], grid)
    assert coords.shape == (0, 3) and coords.dtype == np.int32
    assert occ.shape == (0,) and occ.dtype == np.uint8
    assert sem.shape == (0,) and sem.dtype == np.int64
