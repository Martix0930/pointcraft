"""B3 deterministic shell extraction (morphological boundary). Pure numpy."""
import numpy as np

from pointcraft.baseline import morphological_boundary
from pointcraft.voxelization import VoxelGrid


def _grid(n):
    return VoxelGrid(origin=np.zeros(3), voxel_size=1.0, shape=np.array([n, n, n]))


def test_solid_cube_only_center_is_interior():
    # a 3x3x3 solid: every voxel is surface except the fully-surrounded centre.
    g = np.arange(3)
    ii, jj, kk = np.meshgrid(g, g, g, indexing="ij")
    cube = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1).astype(np.int32)
    shell = morphological_boundary(cube, _grid(5))
    assert shell.shape[0] == 26  # 27 - 1 centre
    assert not (shell == np.array([1, 1, 1])).all(axis=1).any()  # centre removed


def test_grid_edge_voxel_is_boundary():
    # a voxel touching the grid edge has an out-of-grid neighbour -> surface.
    g = np.arange(3)
    ii, jj, kk = np.meshgrid(g, g, g, indexing="ij")
    cube = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1).astype(np.int32)
    # exact-fit grid (shape 3): the centre's neighbours are all in-grid & present,
    # so it is still the only interior voxel.
    shell = morphological_boundary(cube, _grid(3))
    assert shell.shape[0] == 26


def test_empty_and_dtype():
    out = morphological_boundary(np.zeros((0, 3), np.int32), _grid(4))
    assert out.shape == (0, 3) and out.dtype == np.int32


def test_single_voxel_is_its_own_boundary():
    out = morphological_boundary(np.array([[2, 2, 2]], np.int32), _grid(5))
    assert out.shape == (1, 3)
