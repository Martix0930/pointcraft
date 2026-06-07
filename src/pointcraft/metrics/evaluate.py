"""Top-level evaluation entry point: score a prediction under several cutoffs.

`evaluate` is the shared scoring surface for M1 (the deterministic baselines) and
for M2+/M4 (the learned models), so every milestone's numbers are computed
identically (D8). It reports:

  * ``completion`` — overall occupancy IoU / precision / recall.
  * ``per_class_recall`` — recall for ground(1) / roof(3) / facade(4).
  * ``unobserved`` — the unobserved-region scores **once per mask cutoff** (the
    M4 multi-definition sensitivity requirement).

`load_sample` rebuilds the grid + arrays from a contract `.npz` so callers do not
re-derive the grid geometry by hand.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data import grid_from_metadata, load_sample_metadata
from ..voxelization import VoxelGrid
from .cutoffs import build_cutoff_masks
from .occupancy import occupancy_scores, per_class_recall, unobserved_scores


@dataclass(frozen=True)
class Sample:
    """A loaded M0 contract sample plus its reconstructed `VoxelGrid`."""

    coords_partial: np.ndarray
    feats_partial: np.ndarray
    coords_target: np.ndarray
    occ_target: np.ndarray
    sem_target: np.ndarray
    observed_mask: np.ndarray
    unobserved_mask: np.ndarray
    grid: VoxelGrid
    metadata: dict


def load_sample(npz_path: str) -> Sample:
    """Load a contract `.npz` into a `Sample` (arrays + grid from metadata)."""
    d = np.load(npz_path)
    meta = load_sample_metadata(d)
    grid = grid_from_metadata(meta)
    return Sample(
        coords_partial=d["coords_partial"],
        feats_partial=d["feats_partial"],
        coords_target=d["coords_target"],
        occ_target=d["occ_target"],
        sem_target=d["sem_target"],
        observed_mask=d["observed_mask"],
        unobserved_mask=d["unobserved_mask"],
        grid=grid,
        metadata=meta,
    )


def evaluate(
    pred_coords: np.ndarray,
    sample: Sample,
    *,
    cutoffs: dict[str, np.ndarray] | None = None,
) -> dict:
    """Score `pred_coords` against `sample` under each unobserved-mask cutoff.

    Args:
        pred_coords: (P, 3) predicted occupied voxel indices (contract convention).
        sample:      a loaded `Sample`.
        cutoffs:     ``{name: unobserved_mask}``. If None, the three default cutoffs
                     (strict/mid/tolerant) are built from the sample.

    Returns a JSON-serialisable dict:
        {
          "completion": {...overall occupancy scores...},
          "per_class_recall": {1: .., 3: .., 4: ..},
          "unobserved": {cutoff_name: {...scores...}, ...},
        }
    """
    if cutoffs is None:
        cutoffs = build_cutoff_masks(
            sample.coords_target,
            sample.coords_partial,
            sample.sem_target,
            sample.grid,
        )

    completion = occupancy_scores(pred_coords, sample.coords_target, sample.grid)
    pcr = per_class_recall(
        pred_coords, sample.coords_target, sample.sem_target, sample.grid
    )

    unobserved: dict[str, dict] = {}
    for name, mask in cutoffs.items():
        unobserved[name] = unobserved_scores(
            pred_coords,
            sample.coords_target,
            mask,
            sample.coords_partial,
            sample.grid,
        ).as_dict()

    return {
        "completion": completion.as_dict(),
        "per_class_recall": pcr,
        "unobserved": unobserved,
    }
