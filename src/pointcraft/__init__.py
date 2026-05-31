"""PointCraft — Aerial-to-Embodied Semantic Scene Completion.

This is the single importable ``pointcraft`` package. Research code lives in the
subpackages (``data``, ``voxelization``, ``models``, ``losses``, ``metrics``,
``mc_export``, ``utils``). The legacy deterministic pipeline (the former repo-root
``pointcraft`` package) has been merged into ``pointcraft.baseline`` and serves as
the M1 baseline. See ``CLAUDE.md`` and ``docs/06_DECISIONS.md``.

Heavy dependencies are imported lazily inside submodules, so ``import pointcraft``
stays cheap.
"""

__version__ = "0.1.0"
