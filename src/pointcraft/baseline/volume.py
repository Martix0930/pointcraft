"""B2 — rule-based footprint volume fill (M1 upper-reference predictor).

⚠ **B2 is NOT a fair predictor.** It reads the **target footprint** (which columns
the CityGML shell occupies) and each footprint column's height range, then
reconstructs building geometry by rule. It answers *"if you even knew the
footprint, how far could deterministic rules go?"* — an upper reference that brackets
the M2 headroom from above, while B1 (`predictors.naive_roof_extrusion`,
observation-only) brackets it from below.

Two modes:
  * ``shell`` (default) — reconstruct the LOD2 **shell** (D2): a roof cap on every
    footprint column, full-height walls on the footprint **perimeter** columns, and
    a base/ground voxel on every column. Matches the shell target representation, so
    this is the meaningful ceiling.
  * ``solid`` — fill every footprint column base→top solidly (the cruder volume
    reference; over-predicts the hollow interior).

Heights/base come from the target (the footprint *is* the peek); no LiDAR is used.
Pure numpy; no learning.
"""
from __future__ import annotations

import numpy as np

from ..voxelization import VoxelGrid
from .predictors import _extrude


def footprint_volume_fill(
    coords_target: np.ndarray,
    grid: VoxelGrid,
    *,
    mode: str = "shell",
) -> np.ndarray:
    """B2 — reconstruct building geometry from the target footprint + heights.

    Args:
        coords_target: `(M,3)` target voxel indices (the footprint peek).
        grid:          the sample's shared `VoxelGrid`.
        mode:          ``"shell"`` (roof cap + perimeter walls + base) or
                       ``"solid"`` (fill every footprint column base→top).

    Returns predicted occupied voxel indices `(P,3)` int32, deduplicated and sorted.
    """
    ct = np.asarray(coords_target, dtype=np.int64).reshape(-1, 3)
    if ct.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.int32)

    si, sj = int(grid.shape[0]), int(grid.shape[1])
    col = ct[:, 0] * sj + ct[:, 1]
    ucol, inv = np.unique(col, return_inverse=True)
    col_top = np.full(ucol.shape[0], -(1 << 30), dtype=np.int64)
    col_base = np.full(ucol.shape[0], 1 << 30, dtype=np.int64)
    np.maximum.at(col_top, inv, ct[:, 2])
    np.minimum.at(col_base, inv, ct[:, 2])
    cols_i = ucol // sj
    cols_j = ucol % sj

    if mode == "solid":
        # extrude each footprint column over its own [base, top].
        pred = _extrude(cols_i, cols_j, col_base, col_top)
        return np.unique(pred, axis=0).astype(np.int32)

    if mode != "shell":
        raise ValueError(f"mode must be 'shell' or 'solid', got {mode!r}")

    # --- shell: roof cap + base everywhere; full walls on perimeter columns ---
    fp = np.zeros((si, sj), dtype=bool)
    fp[cols_i, cols_j] = True
    # interior = footprint cell whose 4 neighbours are all footprint cells.
    interior = (
        fp
        & np.roll(fp, 1, 0) & np.roll(fp, -1, 0)
        & np.roll(fp, 1, 1) & np.roll(fp, -1, 1)
    )
    # np.roll wraps at the array edge; a grid-border footprint cell is perimeter.
    interior[0, :] = interior[-1, :] = interior[:, 0] = interior[:, -1] = False
    is_perim = ~interior[cols_i, cols_j]

    # roof cap (top) + base (ground) on every footprint column.
    caps = np.column_stack([cols_i, cols_j, col_top]).astype(np.int32)
    bases = np.column_stack([cols_i, cols_j, col_base]).astype(np.int32)
    # full-height walls (base->top) on perimeter columns only.
    walls = _extrude(
        cols_i[is_perim], cols_j[is_perim], col_base[is_perim], col_top[is_perim]
    )
    pred = np.concatenate([caps, bases, walls], axis=0)
    return np.unique(pred, axis=0).astype(np.int32)
