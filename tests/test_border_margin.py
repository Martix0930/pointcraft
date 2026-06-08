"""M2 G0: border ignore-margin keep-mask + evaluate exclusion. Pure numpy."""
import numpy as np

from pointcraft.metrics import border_keep_mask, evaluate
from pointcraft.metrics.evaluate import Sample
from pointcraft.voxelization import VoxelGrid


def _grid(I=10, J=10, K=6):
    return VoxelGrid(origin=np.zeros(3), voxel_size=1.0, shape=np.array([I, J, K]))


def test_keep_mask_drops_only_the_band():
    g = _grid(10, 10, 6)
    coords = np.array([[0, 5, 0], [1, 5, 0], [2, 5, 0], [5, 5, 0],
                       [9, 5, 0], [5, 0, 0], [5, 9, 0]], dtype=np.int32)
    keep = border_keep_mask(coords, g, margin=2)
    # i or j within 2 of an edge (0,1,8,9) -> dropped; (2,5) and (5,5) kept
    assert keep.tolist() == [False, False, True, True, False, False, False]
    # margin 0 keeps everything
    assert border_keep_mask(coords, g, 0).all()


def _sample(coords_target, sem, grid):
    occ = np.ones(len(coords_target), np.uint8)
    obs = np.zeros(len(coords_target), np.uint8)   # all unobserved (simple)
    return Sample(
        coords_partial=np.zeros((0, 3), np.int32),
        feats_partial=np.zeros((0, 2), np.float32),
        coords_target=coords_target, occ_target=occ, sem_target=sem,
        observed_mask=obs, unobserved_mask=1 - obs, grid=grid,
        metadata={"grid_shape": list(map(int, grid.shape)), "voxel_size": 1.0},
    )


def test_evaluate_border_margin_excludes_band_both_sides():
    g = _grid(10, 10, 6)
    # target: one interior voxel (5,5,3) + one border voxel (0,5,3)
    ct = np.array([[5, 5, 3], [0, 5, 3]], dtype=np.int32)
    sem = np.array([4, 4], np.int64)
    s = _sample(ct, sem, g)
    cut = {"strict": s.unobserved_mask.copy()}

    # prediction hits only the interior target voxel
    pred = np.array([[5, 5, 3]], dtype=np.int32)

    # no margin: target has 2 voxels, pred 1 -> recall 1/2
    r0 = evaluate(pred, s, cutoffs=cut, border_margin=0)
    assert r0["completion"]["recall"] == 0.5

    # margin 2: the border target voxel (0,5,3) is excluded -> target has 1, pred 1
    #           -> perfect score within the kept region
    r1 = evaluate(pred, s, cutoffs=cut, border_margin=2)
    assert r1["completion"]["recall"] == 1.0
    assert r1["completion"]["tp"] == 1 and r1["completion"]["fn"] == 0


def test_evaluate_border_margin_drops_border_false_positive():
    g = _grid(10, 10, 6)
    ct = np.array([[5, 5, 3]], dtype=np.int32)
    s = _sample(ct, np.array([4], np.int64), g)
    cut = {"strict": s.unobserved_mask.copy()}
    # prediction: the correct interior voxel + a spurious border voxel
    pred = np.array([[5, 5, 3], [0, 0, 3]], dtype=np.int32)
    r0 = evaluate(pred, s, cutoffs=cut, border_margin=0)
    assert r0["completion"]["fp"] == 1            # border FP counts without margin
    r1 = evaluate(pred, s, cutoffs=cut, border_margin=2)
    assert r1["completion"]["fp"] == 0            # excluded with margin
