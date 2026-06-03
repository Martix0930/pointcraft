"""Occupancy completion metrics on the M0 voxel contract (M1, shared with M2+).

Predictions and targets are **sparse sets of occupied voxel indices** (the data
contract stores occupied voxels only). All scoring reduces to set algebra on
int64 raveled keys:

    tp = |pred ∩ target|,  fp = |pred \\ target|,  fn = |target \\ pred|
    IoU       = tp / (tp + fp + fn)
    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)

The headline M4 number is the **unobserved-region IoU**: IoU restricted to the
region the input never saw. That region is the whole grid **minus** the observed
voxels (the partial input plus the target voxels flagged observed for the chosen
cutoff). Restricting to it — rather than to the unobserved *target* voxels alone —
keeps false positives in the metric, so a predictor that hallucinates occupancy in
unseen free space is penalised, not merely scored on recall (D8).

Pure numpy; no learning, no heavy deps.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..voxelization import VoxelGrid


def _ravel(coords: np.ndarray, grid: VoxelGrid) -> np.ndarray:
    """(N,3) voxel indices -> unique int64 keys, matching data.sample convention."""
    c = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    sj, sk = int(grid.shape[1]), int(grid.shape[2])
    return c[:, 0] * (sj * sk) + c[:, 1] * sk + c[:, 2]


@dataclass(frozen=True)
class Scores:
    """Occupancy scores for one (pred, target) comparison over some region."""

    iou: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int

    def as_dict(self) -> dict:
        return asdict(self)


def _scores_from_keys(pred_keys: np.ndarray, target_keys: np.ndarray) -> Scores:
    """Set-algebra scores from two arrays of (not necessarily unique) int64 keys."""
    pred = np.unique(np.asarray(pred_keys, dtype=np.int64))
    target = np.unique(np.asarray(target_keys, dtype=np.int64))
    tp = int(np.isin(pred, target, assume_unique=True).sum())
    fp = int(pred.size - tp)
    fn = int(target.size - tp)
    union = tp + fp + fn
    iou = tp / union if union else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return Scores(iou=iou, precision=precision, recall=recall, tp=tp, fp=fp, fn=fn)


def occupancy_scores(
    pred_coords: np.ndarray,
    target_coords: np.ndarray,
    grid: VoxelGrid,
) -> Scores:
    """Overall completion scores: predicted occupancy vs the full target occupancy."""
    return _scores_from_keys(_ravel(pred_coords, grid), _ravel(target_coords, grid))


def unobserved_scores(
    pred_coords: np.ndarray,
    target_coords: np.ndarray,
    unobserved_mask: np.ndarray,
    coords_partial: np.ndarray,
    grid: VoxelGrid,
) -> Scores:
    """Scores restricted to the unobserved region (D8).

    The observed region excluded from both sides is the union of the partial-input
    voxels and the target voxels flagged observed (``unobserved_mask == 0``); what
    remains of the target is exactly the unobserved target voxels, and what remains
    of the prediction is everything it placed in never-seen space.
    """
    target_keys = _ravel(target_coords, grid)
    unobs = np.asarray(unobserved_mask).reshape(-1).astype(bool)
    if unobs.shape[0] != target_keys.shape[0]:
        raise ValueError(
            f"unobserved_mask ({unobs.shape[0]}) must align with target "
            f"({target_keys.shape[0]})"
        )
    observed_target_keys = target_keys[~unobs]
    exclude = np.union1d(_ravel(coords_partial, grid), observed_target_keys)

    pred_keys = _ravel(pred_coords, grid)
    pred_unobs = pred_keys[~np.isin(pred_keys, exclude)]
    target_unobs = target_keys[unobs]
    return _scores_from_keys(pred_unobs, target_unobs)


def per_class_recall(
    pred_coords: np.ndarray,
    target_coords: np.ndarray,
    sem_target: np.ndarray,
    grid: VoxelGrid,
    *,
    classes=(1, 3, 4),
) -> dict[int, float]:
    """Per-class occupancy **recall** (cheap, semantics-free on the prediction side).

    For an occupancy-only predictor false positives are not class-attributable, so
    we report recall per target class — of the target voxels of class ``c``, the
    fraction the prediction covered. Facade (4) is the class that matters for M4.
    """
    target_keys = _ravel(target_coords, grid)
    sem = np.asarray(sem_target).reshape(-1)
    pred = np.unique(_ravel(pred_coords, grid))
    out: dict[int, float] = {}
    for c in classes:
        sel = sem == c
        n = int(sel.sum())
        if n == 0:
            continue
        hit = int(np.isin(target_keys[sel], pred).sum())
        out[int(c)] = hit / n
    return out
