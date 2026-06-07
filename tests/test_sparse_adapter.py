"""M2 Phase B: M0 contract ↔ spconv SparseConvTensor adapter round-trips losslessly.

Guarded with importorskip so the suite stays green in the torch-free global env;
these run in the M2 `.venv` (torch + spconv-cu126). See data/sparse.py and
docs/07_GOTCHAS.md (M2 Phase 0).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("spconv.pytorch")

from pointcraft.data.sparse import (  # noqa: E402
    coords_from_sparse_tensor,
    occupancy_logits_to_coords,
    to_sparse_tensor,
)
from pointcraft.voxelization import VoxelGrid  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _grid():
    return VoxelGrid(origin=np.zeros(3), voxel_size=1.0, shape=np.array([8, 6, 5]))


def test_grid_metadata_and_features_preserved():
    grid = _grid()
    coords = np.array([[0, 0, 0], [7, 5, 4], [3, 2, 1], [1, 1, 1]], dtype=np.int32)
    feats = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    x = to_sparse_tensor(coords, feats, grid, device=DEVICE)
    # grid geometry survives the conversion (spatial_shape == grid.shape)
    assert [int(s) for s in x.spatial_shape] == [8, 6, 5]
    assert int(x.batch_size) == 1
    # features survive unchanged
    assert np.allclose(x.features.detach().cpu().numpy(), feats)


def test_indices_are_batch_ijk_and_roundtrip():
    grid = _grid()
    coords = np.array([[0, 0, 0], [1, 2, 3], [7, 5, 4]], dtype=np.int32)
    feats = np.ones((3, 2), dtype=np.float32)
    x = to_sparse_tensor(coords, feats, grid, device=DEVICE)
    idx = x.indices.detach().cpu().numpy()
    assert idx.shape == (3, 4)
    assert (idx[:, 0] == 0).all()              # batch column prepended
    assert np.array_equal(idx[:, 1:4], coords)  # (i,j,k) order preserved
    # recovery drops the batch column and matches the input exactly
    rec = coords_from_sparse_tensor(x)
    assert rec.dtype == np.int32
    assert np.array_equal(rec, coords)


def test_batch_index_and_per_batch_recovery():
    grid = _grid()
    coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.int32)
    feats = np.ones((3, 2), dtype=np.float32)
    bidx = np.array([0, 1, 0], dtype=np.int32)
    x = to_sparse_tensor(coords, feats, grid, batch_size=2, batch_index=bidx, device=DEVICE)
    assert int(x.batch_size) == 2
    rec0 = coords_from_sparse_tensor(x, batch=0)
    rec1 = coords_from_sparse_tensor(x, batch=1)
    assert np.array_equal(rec0, coords[[0, 2]])
    assert np.array_equal(rec1, coords[[1]])


def test_logits_to_coords_threshold():
    coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.int32)
    logits = torch.tensor([2.0, -1.0, 0.5])  # >0 → occupied: rows 0 and 2
    out = occupancy_logits_to_coords(coords, logits, threshold=0.0)
    assert np.array_equal(out, coords[[0, 2]])
    # numpy logits accepted too
    out2 = occupancy_logits_to_coords(coords, np.array([-5.0, 9.0, -0.1]), threshold=0.0)
    assert np.array_equal(out2, coords[[1]])
