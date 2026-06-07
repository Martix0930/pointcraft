"""Data: LiDAR → partial occupancy, LOD2/mesh → target, and .npz sample I/O.

See docs/02_DATA_CONTRACT.md and tasks/M0_data_pairing/. (M0)
"""
from .citygml import (
    GROUND_LABEL,
    SURFACE_TYPE_TO_LABEL,
    TypedSurfaces,
    load_citygml,
    parse_citygml,
)
from .partial import FEATURE_LAYOUT_V01, load_las_xyz, voxelize_partial
from .sample import (
    DATASET_VERSION,
    build_metadata,
    compute_masks,
    grid_from_metadata,
    load_sample_metadata,
    write_sample_npz,
)
from .target import (
    FACADE_LABEL,
    ROOF_LABEL,
    load_lod2_meshes,
    voxelize_citygml_target,
    voxelize_target,
)

__all__ = [
    "voxelize_partial",
    "load_las_xyz",
    "FEATURE_LAYOUT_V01",
    "parse_citygml",
    "load_citygml",
    "TypedSurfaces",
    "GROUND_LABEL",
    "SURFACE_TYPE_TO_LABEL",
    "voxelize_target",
    "voxelize_citygml_target",
    "load_lod2_meshes",
    "ROOF_LABEL",
    "FACADE_LABEL",
    "compute_masks",
    "build_metadata",
    "grid_from_metadata",
    "write_sample_npz",
    "load_sample_metadata",
    "DATASET_VERSION",
]
