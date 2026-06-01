"""Data: LiDAR → partial occupancy, LOD2/mesh → target, and .npz sample I/O.

See docs/02_DATA_CONTRACT.md and tasks/M0_data_pairing/. (M0)
"""
from .partial import FEATURE_LAYOUT_V01, load_las_xyz, voxelize_partial
from .target import (
    FACADE_LABEL,
    ROOF_LABEL,
    load_lod2_meshes,
    voxelize_target,
)

__all__ = [
    "voxelize_partial",
    "load_las_xyz",
    "FEATURE_LAYOUT_V01",
    "voxelize_target",
    "load_lod2_meshes",
    "ROOF_LABEL",
    "FACADE_LABEL",
]
