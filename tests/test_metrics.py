"""Unit tests for the shared metrics module (M1, Phase B).

Hand-checkable IoU on tiny synthetic grids + the multi-cutoff contract (D8):
strict reproduces the stored v0.2 mask, and unobserved-observed counts are
monotone strict >= mid >= tolerant as the facade line is relaxed.
"""
import os
import tempfile

import numpy as np
import pytest

from pointcraft.data import compute_masks, write_sample_npz, build_metadata
from pointcraft.metrics import (
    build_cutoff_masks,
    evaluate,
    load_sample,
    occupancy_scores,
    per_class_recall,
    unobserved_scores,
)
from pointcraft.voxelization import VoxelGrid


def _grid(shape=(5, 5, 10)):
    return VoxelGrid(
        origin=np.zeros(3),
        voxel_size=1.0,
        shape=np.asarray(shape, dtype=np.int64),
    )


def test_occupancy_iou_hand_checked():
    grid = _grid()
    target = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.int32)
    pred = np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0]], dtype=np.int32)
    # tp={(0,0,0),(1,0,0)}=2, fp={(3,0,0)}=1, fn={(2,0,0)}=1
    s = occupancy_scores(pred, target, grid)
    assert (s.tp, s.fp, s.fn) == (2, 1, 1)
    assert s.iou == pytest.approx(2 / 4)
    assert s.precision == pytest.approx(2 / 3)
    assert s.recall == pytest.approx(2 / 3)


def test_occupancy_dedup_and_empty():
    grid = _grid()
    target = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.int32)  # dup
    pred = np.zeros((0, 3), dtype=np.int32)
    s = occupancy_scores(pred, target, grid)
    assert (s.tp, s.fp, s.fn) == (0, 0, 2)  # 2 unique target voxels, nothing pred
    assert s.iou == 0.0


# --- a tiny scene shared by the cutoff + unobserved tests ---------------------
# facade column at (2,2), k=0..5 (label 4); a roof voxel (0,0,5) label 3; a ground
# voxel (4,4,0) label 1. col_base=0, col_top=5, wall_margin=2 -> midwall k in [2,3].
def _scene():
    grid = _grid()
    col = [[2, 2, k] for k in range(6)]
    coords_target = np.array(col + [[0, 0, 5], [4, 4, 0]], dtype=np.int32)
    sem_target = np.array([4] * 6 + [3, 1], dtype=np.int64)
    coords_partial = np.array(
        [
            [2, 2, 2],  # exact midwall facade hit -> observed in ALL cutoffs
            [1, 2, 3],  # XY-neighbour at midwall k=3 -> observed in mid + tolerant
            [3, 2, 0],  # XY-neighbour at k=0 (not midwall) -> only tolerant
            [0, 0, 4],  # one below the roof -> z_tol=1 observes roof in ALL cutoffs
        ],
        dtype=np.int32,
    )
    return grid, coords_target, sem_target, coords_partial


def test_strict_cutoff_reproduces_stored_mask():
    grid, ct, sem, cp = _scene()
    _obs, stored = compute_masks(ct, cp, grid, sem_target=sem)  # v0.2 defaults
    masks = build_cutoff_masks(ct, cp, sem, grid)
    assert np.array_equal(masks["strict"], stored)


def test_cutoffs_monotone_and_hand_counted():
    grid, ct, sem, cp = _scene()
    masks = build_cutoff_masks(ct, cp, sem, grid)
    facade = sem == 4
    # facade voxels still UNobserved per cutoff (6 total facade voxels):
    #   strict   -> only k=2 observed -> 5 unobserved
    #   mid      -> k=2,3 observed     -> 4 unobserved
    #   tolerant -> k=0,2,3 observed   -> 3 unobserved
    f_unobs = {n: int(masks[n][facade].sum()) for n in ("strict", "mid", "tolerant")}
    assert f_unobs == {"strict": 5, "mid": 4, "tolerant": 3}
    # relaxing the line never hides more: strict >= mid >= tolerant overall.
    tot = {n: int(masks[n].sum()) for n in ("strict", "mid", "tolerant")}
    assert tot["strict"] >= tot["mid"] >= tot["tolerant"]
    # roof is observed (z_tol), ground never observed -> unobserved in every cutoff.
    roof, ground = sem == 3, sem == 1
    for n in masks:
        assert masks[n][roof].sum() == 0
        assert masks[n][ground].sum() == 1


def test_unobserved_scores_penalise_hallucination():
    grid, ct, sem, cp = _scene()
    _obs, unobs = compute_masks(ct, cp, grid, sem_target=sem)  # strict
    n_unobs = int(unobs.sum())  # facade(5) + ground(1) = 6

    # Perfect prediction (== target): every unobserved target voxel recovered,
    # observed voxels are excluded from the region -> IoU 1.0, no fp.
    s_perfect = unobserved_scores(ct, ct, unobs, cp, grid)
    assert (s_perfect.tp, s_perfect.fp, s_perfect.fn) == (n_unobs, 0, 0)
    assert s_perfect.iou == pytest.approx(1.0)

    # Add a voxel in never-seen free space -> a false positive in the region.
    hallucinated = np.concatenate([ct, np.array([[4, 4, 9]], dtype=np.int32)])
    s_hall = unobserved_scores(hallucinated, ct, unobs, cp, grid)
    assert s_hall.fp == 1
    assert s_hall.iou == pytest.approx(n_unobs / (n_unobs + 1))


def test_per_class_recall_hand_checked():
    grid, ct, sem, cp = _scene()
    # Predict only the roof voxel + 3 of the 6 facade voxels.
    pred = np.array([[0, 0, 5], [2, 2, 0], [2, 2, 1], [2, 2, 2]], dtype=np.int32)
    pcr = per_class_recall(pred, ct, sem, grid)
    assert pcr[3] == pytest.approx(1.0)       # roof: 1/1
    assert pcr[4] == pytest.approx(3 / 6)     # facade: 3/6
    assert pcr[1] == pytest.approx(0.0)       # ground: 0/1


def test_evaluate_end_to_end_keys_and_consistency():
    grid, ct, sem, cp = _scene()
    sample = load_sample(_write_tmp_sample(grid, ct, sem, cp))
    out = evaluate(ct, sample)  # perfect prediction
    assert set(out) == {"completion", "per_class_recall", "unobserved"}
    assert set(out["unobserved"]) == {"strict", "mid", "tolerant"}
    assert out["completion"]["iou"] == pytest.approx(1.0)
    # perfect prediction recovers the unobserved region under every cutoff
    for c in out["unobserved"].values():
        assert c["iou"] == pytest.approx(1.0)
    # custom cutoff set is honoured (entry point takes a SET of definitions)
    out2 = evaluate(ct, sample, cutoffs={"only": sample.unobserved_mask})
    assert set(out2["unobserved"]) == {"only"}


def _write_tmp_sample(grid, ct, sem, cp):
    feats = np.zeros((cp.shape[0], 2), dtype=np.float32)
    occ = np.ones(ct.shape[0], dtype=np.uint8)
    obs, unobs = compute_masks(ct, cp, grid, sem_target=sem)
    meta = build_metadata(
        grid, tile_id="t", crs="LOCAL", source_files=["synthetic"]
    )
    path = os.path.join(tempfile.mkdtemp(), "s.npz")
    write_sample_npz(
        path,
        coords_partial=cp, feats_partial=feats,
        coords_target=ct, occ_target=occ, sem_target=sem,
        observed_mask=obs, unobserved_mask=unobs, metadata=meta,
    )
    return path
