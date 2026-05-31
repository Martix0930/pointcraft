"""Shared voxel grid: the single source of grid geometry used for BOTH the
partial (LiDAR) and target (LOD2) voxelization in M0.

Conventions (see docs/02_DATA_CONTRACT.md):
    - world axes: x=east, y=north, z=up (meters)
    - voxel index:  idx = floor((world_xyz - origin) / voxel_size)
    - origin is the MIN corner of voxel (0, 0, 0)
    - voxel center = origin + (idx + 0.5) * voxel_size
    - grid_shape   = ceil((bounds_max - origin) / voxel_size)
    - index order (i, j, k) <-> (x, y, z)

This module is pure geometry: numpy only, no I/O, no learning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class VoxelGrid:
    """A regular axis-aligned voxel grid.

    Attributes:
        origin:     (3,) float64 — world coord of the min corner of voxel (0,0,0).
        voxel_size: float        — edge length of a voxel in world units (meters).
        shape:      (3,) int64   — (I, J, K) grid dimensions in voxels.
    """

    origin: np.ndarray
    voxel_size: float
    shape: np.ndarray

    # -- construction ----------------------------------------------------------

    @classmethod
    def from_bounds(cls, bounds: Sequence[float], voxel_size: float) -> "VoxelGrid":
        """Build a grid covering `bounds` = [xmin, ymin, zmin, xmax, ymax, zmax].

        origin = bounds_min; shape = ceil((bounds_max - origin) / voxel_size),
        clamped to at least 1 along each axis.
        """
        b = np.asarray(bounds, dtype=np.float64).reshape(6)
        if voxel_size <= 0:
            raise ValueError(f"voxel_size must be > 0, got {voxel_size}")
        origin = b[:3].copy()
        extent = b[3:] - origin
        if np.any(extent < 0):
            raise ValueError(f"bounds max < min: {bounds}")
        shape = np.maximum(np.ceil(extent / voxel_size).astype(np.int64), 1)
        return cls(origin=origin, voxel_size=float(voxel_size), shape=shape)

    # -- transforms ------------------------------------------------------------

    def world_to_index(self, points: np.ndarray) -> np.ndarray:
        """(N,3) world XYZ -> (N,3) int64 voxel indices via floor. No clipping."""
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        idx = np.floor((p - self.origin) / self.voxel_size)
        return idx.astype(np.int64)

    def index_to_center(self, idx: np.ndarray) -> np.ndarray:
        """(N,3) voxel indices -> (N,3) float64 world coords of voxel centers."""
        i = np.asarray(idx, dtype=np.float64).reshape(-1, 3)
        return self.origin + (i + 0.5) * self.voxel_size

    def index_to_corner(self, idx: np.ndarray) -> np.ndarray:
        """(N,3) voxel indices -> (N,3) float64 world coords of voxel min corners."""
        i = np.asarray(idx, dtype=np.float64).reshape(-1, 3)
        return self.origin + i * self.voxel_size

    # -- queries ---------------------------------------------------------------

    def in_bounds(self, idx: np.ndarray) -> np.ndarray:
        """(N,3) voxel indices -> (N,) bool mask of indices within [0, shape)."""
        i = np.asarray(idx, dtype=np.int64).reshape(-1, 3)
        return np.all((i >= 0) & (i < self.shape), axis=1)

    @property
    def num_voxels(self) -> int:
        return int(np.prod(self.shape))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        o = ", ".join(f"{v:g}" for v in self.origin)
        s = ", ".join(str(int(v)) for v in self.shape)
        return f"VoxelGrid(origin=({o}), voxel_size={self.voxel_size:g}, shape=({s}))"
