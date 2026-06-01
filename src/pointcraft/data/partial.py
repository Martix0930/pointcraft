"""LiDAR → partial voxel occupancy on the shared VoxelGrid (M0-2).

Maps LiDAR points onto a `VoxelGrid`, drops out-of-range points (logged),
merges duplicate points that fall in the same voxel, and emits a sparse
representation: voxel coordinates + per-voxel features.

feature_layout `v0.1` (see docs/02_DATA_CONTRACT.md):
    ["height", "point_count"]
    - height      = mean world-z of the points merged into the voxel
                    (mean, not max — robust to the LiDAR outliers seen in Phase B)
    - point_count = number of points merged into the voxel

Pure numpy; no learning. LAS reading is an optional thin helper (laspy imported
lazily) so tests can run on plain arrays / CSV fixtures without laspy.
"""
from __future__ import annotations

import logging

import numpy as np

from ..voxelization import VoxelGrid

log = logging.getLogger(__name__)

#: Feature column names/order for dataset_version v0.1. The returned
#: ``feats_partial`` has one column per entry, in this order.
FEATURE_LAYOUT_V01 = ["height", "point_count"]


def voxelize_partial(
    points_xyz: np.ndarray, grid: VoxelGrid
) -> tuple[np.ndarray, np.ndarray]:
    """Voxelize LiDAR points into partial occupancy on ``grid``.

    Args:
        points_xyz: (P, 3) world XYZ of LiDAR points (meters, absolute z).
        grid:       the shared VoxelGrid (same instance used for the target).

    Returns:
        coords_partial: (N, 3) int32 unique voxel indices (i, j, k), sorted.
        feats_partial:  (N, C) float32 per-voxel features, columns per
                        ``FEATURE_LAYOUT_V01`` = [height, point_count].

    Out-of-range points are dropped (count logged). Points sharing a voxel are
    merged: occupancy is the OR (the voxel is present once), ``height`` is the
    mean z, and ``point_count`` is how many points were merged.
    """
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    idx = grid.world_to_index(pts)  # int64 (P, 3)

    inb = grid.in_bounds(idx)
    n_drop = int((~inb).sum())
    if n_drop:
        log.info("partial: dropped %d/%d out-of-range points", n_drop, len(pts))
    idx = idx[inb]
    z = pts[inb, 2]

    if idx.shape[0] == 0:
        return (
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0, len(FEATURE_LAYOUT_V01)), dtype=np.float32),
        )

    # Unique voxels + inverse map for segmented aggregation.
    coords, inverse = np.unique(idx, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)  # numpy version-proof shape
    n = coords.shape[0]

    point_count = np.zeros(n, dtype=np.float64)
    np.add.at(point_count, inverse, 1.0)

    z_sum = np.zeros(n, dtype=np.float64)
    np.add.at(z_sum, inverse, z)
    height = z_sum / point_count

    feats = np.stack([height, point_count], axis=1).astype(np.float32)
    log.info("partial: %d points -> %d occupied voxels", idx.shape[0], n)
    return coords.astype(np.int32), feats


def load_las_xyz(path: str) -> np.ndarray:
    """Read a LAS/LAZ file's point coordinates as (P, 3) float64 world XYZ.

    Thin wrapper over laspy (imported lazily — only needed for real-data runs,
    not for fixture-driven tests). Classification/intensity are intentionally
    not returned: the v0.1 feature layout does not use them.
    """
    import laspy

    las = laspy.read(path)
    return np.stack(
        [np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)], axis=1
    ).astype(np.float64)
