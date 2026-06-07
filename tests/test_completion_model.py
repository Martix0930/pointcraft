"""M2 Phase C/D: completion U-Net forward/backward + overfit data builders.

Guarded with importorskip (torch + spconv) → runs in the M2 `.venv` only.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("spconv.pytorch")

from pointcraft.data.sparse import to_sparse_tensor  # noqa: E402
from pointcraft.metrics.evaluate import Sample  # noqa: E402
from pointcraft.models.completion_unet import OccupancyCompletionUNet  # noqa: E402
from pointcraft.train.overfit import (  # noqa: E402
    build_candidate_support,
    build_features,
    build_labels,
)
from pointcraft.voxelization import VoxelGrid  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _dense_support(n_side=16):
    """A small solid block of voxels (dense neighbourhoods for sparse convs).

    Must stay large enough that two stride-2 downsamples don't collapse the grid
    below the kernel size (≥ ~8 per axis), else spconv's gemm errors.
    """
    g = np.arange(n_side)
    ii, jj, kk = np.meshgrid(g, g, g, indexing="ij")
    coords = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1).astype(np.int32)
    grid = VoxelGrid(origin=np.zeros(3), voxel_size=1.0, shape=np.array([n_side] * 3))
    return coords, grid


def test_unet_forward_backward_and_order():
    coords, grid = _dense_support(16)
    feats = np.random.RandomState(0).randn(coords.shape[0], 5).astype(np.float32)
    x = to_sparse_tensor(coords, feats, grid, device=DEVICE)
    model = OccupancyCompletionUNet(in_channels=5, base=8).to(DEVICE)

    out = model(x)
    # one logit per input voxel, aligned with the input support order
    assert out.features.shape == (coords.shape[0], 1)
    assert torch.equal(out.indices, x.indices)

    # gradients flow to params
    loss = out.features.float().pow(2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)


def test_semantic_ready_head_width():
    """Head out-channels widen for a future semantic head without backbone changes."""
    coords, grid = _dense_support(12)
    feats = np.zeros((coords.shape[0], 5), dtype=np.float32)
    x = to_sparse_tensor(coords, feats, grid, device=DEVICE)
    model = OccupancyCompletionUNet(in_channels=5, base=8, out_channels=4).to(DEVICE)
    out = model(x)
    assert out.features.shape == (coords.shape[0], 4)


def _toy_sample():
    """Tiny synthetic contract Sample: one 'building' column + ground patch."""
    I = J = 5
    K = 8
    grid = VoxelGrid(origin=np.zeros(3), voxel_size=1.0, shape=np.array([I, J, K]))
    # partial: a roof voxel at (2,2,6) and some ground returns at k=0
    coords_partial = np.array(
        [[2, 2, 6], [1, 1, 0], [2, 2, 0], [3, 3, 0]], dtype=np.int32
    )
    feats_partial = np.tile(np.array([[2.0, 1.0]], np.float32), (4, 1))
    # target shell: a 3-voxel column (ground..roof) at (2,2)
    coords_target = np.array([[2, 2, 0], [2, 2, 3], [2, 2, 6]], dtype=np.int32)
    occ = np.ones(3, np.uint8)
    sem = np.array([1, 4, 3], np.int64)  # ground, facade, roof
    obs = np.array([1, 0, 1], np.uint8)
    unobs = 1 - obs
    return Sample(coords_partial, feats_partial, coords_target, occ, sem, obs, unobs,
                  grid, {"grid_shape": [I, J, K], "voxel_size": 1.0})


def test_overfit_builders_are_input_only_and_cover_target():
    s = _toy_sample()
    support = build_candidate_support(s)  # uses no target
    feats = build_features(s, support)
    labels = build_labels(s, support)
    assert support.dtype == np.int32 and support.shape[1] == 3
    assert feats.shape == (support.shape[0], 5)
    # labels are 1 exactly where a support voxel is a target voxel
    def keys(c):
        c = c.astype(np.int64); return c[:, 0] * (5 * 8) + c[:, 1] * 8 + c[:, 2]
    kt = set(keys(s.coords_target).tolist())
    expect = np.array([1.0 if k in kt else 0.0 for k in keys(support).tolist()], np.float32)
    assert np.array_equal(labels, expect)
    # the extruded column should cover the unobserved mid-facade target voxel (2,2,3)
    assert (support == np.array([2, 2, 3], np.int32)).all(axis=1).any()
    # 'observed' feature channel matches partial membership
    kp = set(keys(s.coords_partial).tolist())
    obs_expect = np.array([1.0 if k in kp else 0.0 for k in keys(support).tolist()], np.float32)
    assert np.array_equal(feats[:, 0], obs_expect)
