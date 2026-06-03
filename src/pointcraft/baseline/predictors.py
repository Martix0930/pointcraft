"""Deterministic (no-NN) completion predictors for the M1 baseline.

These emit a prediction in the M0 **data-contract** format — an ``(N, 3)`` int32
array of occupied voxel indices on the sample's shared ``VoxelGrid`` — so the shared
metrics module scores them exactly like a learned model (M2+).

Two baselines (see ``tasks/M1_deterministic_baseline/EXECUTION_PLAN.md``):

  * **B1 — naive roof extrusion** (`naive_roof_extrusion`): the observation-only
    *floor*. For every observed building column, drop a solid vertical column from
    the top LiDAR return down to an estimated ground plane. It recovers facade only
    where the wall sits **directly under the roof footprint** (vertical columns); it
    cannot recover setbacks/insets, so its unobserved-region IoU is expected to be
    low — that low number is the honest floor M2 must beat. Uses **no target
    information**.

  * **B2 — rule-based footprint volume fill** (`footprint_volume_fill`, in
    `volume.py`): an *upper reference* that peeks at the CityGML target footprint —
    not a fair predictor.

Borrows only the *extrusion idea* from the legacy 2.5D `baseline/stages.py`; it does
not reuse that module's height-map / colouring logic. Pure numpy; no learning.
"""
from __future__ import annotations

import numpy as np

from ..voxelization import VoxelGrid

#: Default min height (voxels = metres at 1 m) above ground for a column to count
#: as a "building" worth extruding (vs flat ground / low clutter).
DEFAULT_MIN_BUILDING_HEIGHT = 3
#: Default percentile of per-column minima used to estimate the global ground k.
DEFAULT_GROUND_PERCENTILE = 25.0


def _column_keys(coords: np.ndarray, grid: VoxelGrid) -> np.ndarray:
    """(N,3) indices -> per-(i,j) column key int64 (k collapsed)."""
    c = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    return c[:, 0] * int(grid.shape[1]) + c[:, 1]


def estimate_ground_k(
    coords_partial: np.ndarray,
    grid: VoxelGrid,
    *,
    ground_percentile: float = DEFAULT_GROUND_PERCENTILE,
) -> int:
    """Estimate a single global ground voxel-k from the LiDAR (observation only).

    Per-column minima are the lowest return in each column; their low percentile is
    the terrain floor, robust to building-interior columns (whose only return is a
    high roof) which sit in the upper tail. Assumes a roughly flat tile (true for
    the Tokyo-Station tile: per-column min-k ~17–18 over a 222-voxel grid).
    """
    coords_partial = np.asarray(coords_partial, dtype=np.int64).reshape(-1, 3)
    if coords_partial.shape[0] == 0:
        return 0
    col = _column_keys(coords_partial, grid)
    ucol, inv = np.unique(col, return_inverse=True)
    col_min = np.full(ucol.shape[0], 1 << 30, dtype=np.int64)
    np.minimum.at(col_min, inv, coords_partial[:, 2])
    return int(round(float(np.percentile(col_min, ground_percentile))))


def _extrude(cols_i: np.ndarray, cols_j: np.ndarray, k_lo, k_hi: np.ndarray):
    """Solid-fill each column (i,j) over k in [k_lo, k_hi] -> (M,3) int indices.

    `k_hi` is per-column; `k_lo` is a shared scalar floor or a per-column array.
    Vectorised ragged range.
    """
    cols_i = np.asarray(cols_i, dtype=np.int64)
    cols_j = np.asarray(cols_j, dtype=np.int64)
    k_hi = np.asarray(k_hi, dtype=np.int64)
    k_lo = np.broadcast_to(np.asarray(k_lo, dtype=np.int64), k_hi.shape)
    heights = (k_hi - k_lo + 1)
    keep = heights > 0
    cols_i, cols_j, heights, k_lo = cols_i[keep], cols_j[keep], heights[keep], k_lo[keep]
    if heights.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    total = int(heights.sum())
    starts = np.cumsum(heights) - heights
    within = np.arange(total) - np.repeat(starts, heights)
    kk = within + np.repeat(k_lo, heights)
    ii = np.repeat(cols_i, heights)
    jj = np.repeat(cols_j, heights)
    return np.column_stack([ii, jj, kk]).astype(np.int32)


def naive_roof_extrusion(
    coords_partial: np.ndarray,
    grid: VoxelGrid,
    *,
    ground_k: int | None = None,
    min_building_height: int = DEFAULT_MIN_BUILDING_HEIGHT,
    ground_percentile: float = DEFAULT_GROUND_PERCENTILE,
) -> np.ndarray:
    """B1 — naive roof extrusion (observation-only completion floor).

    For each observed column whose top return is at least `min_building_height`
    voxels above the estimated ground, fill a solid vertical column from the ground
    plane up to that top. Returns predicted occupied voxel indices `(M, 3)` int32,
    deduplicated and sorted, all in-bounds on `grid`. **Uses no target data.**

    Args:
        coords_partial:      `(N,3)` observed voxel indices (the partial input).
        grid:                the sample's shared `VoxelGrid`.
        ground_k:            explicit ground voxel-k; if None it is estimated from
                             the LiDAR via `estimate_ground_k`.
        min_building_height: min top-above-ground (voxels) to treat a column as a
                             building worth extruding.
        ground_percentile:   percentile for the ground estimate when `ground_k` is
                             None.
    """
    coords_partial = np.asarray(coords_partial, dtype=np.int64).reshape(-1, 3)
    if coords_partial.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.int32)
    if ground_k is None:
        ground_k = estimate_ground_k(
            coords_partial, grid, ground_percentile=ground_percentile
        )

    col = _column_keys(coords_partial, grid)
    ucol, inv = np.unique(col, return_inverse=True)
    col_top = np.full(ucol.shape[0], -(1 << 30), dtype=np.int64)
    np.maximum.at(col_top, inv, coords_partial[:, 2])

    # building columns: top is high enough above ground.
    is_building = col_top >= ground_k + int(min_building_height)
    sj = int(grid.shape[1])
    cols_i = ucol[is_building] // sj
    cols_j = ucol[is_building] % sj
    tops = col_top[is_building]

    pred = _extrude(cols_i, cols_j, int(ground_k), tops)
    if pred.shape[0] == 0:
        return pred
    pred = np.unique(pred, axis=0)
    return pred.astype(np.int32)
