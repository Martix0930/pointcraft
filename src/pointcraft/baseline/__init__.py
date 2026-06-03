"""Baseline: the legacy deterministic LiDAR -> voxel -> Minecraft pipeline.

This is the **M1 deterministic baseline** (formerly the repo-root ``pointcraft``
package). It is preserved here as a reference/floor for learned models. It mixes
data loading, voxelization, LOD2 fusion, block mapping, and schematic export as
``Stage`` objects driven by ``pointcraft.pipeline``. New M0+ research code should
live in the dedicated subpackages (``data``, ``voxelization``, ``mc_export``,
``models`` ...), not here.

Heavy dependencies (laspy, mcschematic, pyvista) are imported lazily by the
submodules, so importing ``pointcraft`` does not pull them in.

The M1 deterministic *predictors* (the new no-NN completion baselines scored on the
M0 contract) live in ``predictors`` (B1 naive roof extrusion) and ``volume`` (B2
footprint volume fill); they are pure numpy and safe to import directly.
"""
from .predictors import (
    estimate_ground_k,
    naive_roof_extrusion,
)
from .volume import footprint_volume_fill

__all__ = [
    "naive_roof_extrusion",
    "estimate_ground_k",
    "footprint_volume_fill",
]
