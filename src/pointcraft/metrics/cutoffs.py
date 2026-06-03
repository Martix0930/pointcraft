"""Mask cutoffs for the multi-definition unobserved-region metric (M1↔M4, D8).

M4's headline must prove "beats M1 baseline" under several observation lines, not
one. `observed`/`unobserved` is a task-oriented choice, not a physical measurement
(D6/D7): the LiDAR physically grazes ~67 % of facade within 1 m, but the strict
v0.2 mask counts only ~35 % observed to preserve a genuine completion region.

This module turns one contract sample into the **three cutoff masks** so every
milestone scores its prediction on the same definitions:

  * ``strict``   — v0.2 stored rule: facade observed only on an exact genuine
                   mid-wall hit (~35 % facade observed). Reproduces the `.npz`
                   ``unobserved_mask`` bit-for-bit.
  * ``mid``      — facade observed within XY ±1 but still mid-wall only.
  * ``tolerant`` — facade observed within XY ±1, mid-wall requirement dropped
                   (~67 %, the physical-grazing line).

Roof/ground always use the z±1 tolerance (D6) across all cutoffs; only the facade
rule moves. Built on `pointcraft.data.compute_masks` so there is a single mask
implementation.
"""
from __future__ import annotations

import numpy as np

from ..data import compute_masks
from ..voxelization import VoxelGrid

#: cutoff name -> facade parameters passed to compute_masks (roof/ground fixed).
CUTOFFS: dict[str, dict] = {
    "strict": dict(xy_tol=0, wall_margin=2),
    "mid": dict(xy_tol=1, wall_margin=2),
    "tolerant": dict(xy_tol=1, wall_margin=0),
}


def build_cutoff_masks(
    coords_target: np.ndarray,
    coords_partial: np.ndarray,
    sem_target: np.ndarray,
    grid: VoxelGrid,
    *,
    cutoffs: dict[str, dict] = CUTOFFS,
) -> dict[str, np.ndarray]:
    """Return ``{cutoff_name: unobserved_mask (uint8, aligned to coords_target)}``.

    The ``strict`` cutoff (defaults) equals the stored v0.2 ``unobserved_mask``.
    """
    out: dict[str, np.ndarray] = {}
    for name, params in cutoffs.items():
        _observed, unobserved = compute_masks(
            coords_target,
            coords_partial,
            grid,
            sem_target=sem_target,
            **params,
        )
        out[name] = np.asarray(unobserved, dtype=np.uint8)
    return out
