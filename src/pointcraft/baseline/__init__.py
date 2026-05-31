"""Baseline: the legacy deterministic LiDAR -> voxel -> Minecraft pipeline.

This is the **M1 deterministic baseline** (formerly the repo-root ``pointcraft``
package). It is preserved here as a reference/floor for learned models. It mixes
data loading, voxelization, LOD2 fusion, block mapping, and schematic export as
``Stage`` objects driven by ``pointcraft.pipeline``. New M0+ research code should
live in the dedicated subpackages (``data``, ``voxelization``, ``mc_export``,
``models`` ...), not here.

Heavy dependencies (laspy, mcschematic, pyvista) are imported lazily by the
submodules, so importing ``pointcraft`` does not pull them in.
"""
