"""Metrics: occupancy completion IoU + multi-cutoff unobserved-region IoU.

Shared across milestones (D8): M1 scores its deterministic baselines here and
M2/M3/M4 import the same functions so numbers are directly comparable. The
unobserved-region IoU is reported under several mask cutoffs (strict / mid /
tolerant) — the M1↔M4 contract for the headline metric.

See docs/03_EXPERIMENT_PROTOCOL.md and docs/06_DECISIONS.md (D6/D7/D8).
"""
from .cutoffs import CUTOFFS, build_cutoff_masks
from .evaluate import Sample, evaluate, load_sample
from .occupancy import (
    Scores,
    border_keep_mask,
    occupancy_scores,
    per_class_recall,
    unobserved_scores,
)

__all__ = [
    "Scores",
    "occupancy_scores",
    "unobserved_scores",
    "per_class_recall",
    "border_keep_mask",
    "CUTOFFS",
    "build_cutoff_masks",
    "Sample",
    "load_sample",
    "evaluate",
]
