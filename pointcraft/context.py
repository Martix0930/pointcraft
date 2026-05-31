"""Pipeline core: Context (data), Stage (op), Pipeline (sequencer)."""
from __future__ import annotations
import logging
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np

# Silence noisy-but-harmless warnings from nanmedian on all-NaN slices.
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in cast")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
log = logging.getLogger("pointcraft")


@dataclass
class Context:
    """Mutable data container passed between Stages.

    Stages produce / consume fields here. None means "not produced yet".
    """
    # Raw point data
    points: Optional[np.ndarray] = None          # (N, 3) float64 world coords
    classification: Optional[np.ndarray] = None  # (N,) uint8
    rgb: Optional[np.ndarray] = None             # (N, 3) uint16

    # Voxel grid (filled by Voxelize)
    top_y: Optional[np.ndarray] = None           # (H, W) int32 — top Y above base
    valid: Optional[np.ndarray] = None           # (H, W) bool
    support_count: Optional[np.ndarray] = None   # (H, W) int32
    ground_frac: Optional[np.ndarray] = None     # (H, W) float

    # Color (filled by Voxelize + SampleLOD2Color)
    las_rgb: Optional[np.ndarray] = None         # (H, W, 3) uint8 — mean color from LAS RGB (top-surface points only)
    has_las_rgb: Optional[np.ndarray] = None     # (H, W) bool — true if cell had any colored LAS points
    top_color: Optional[np.ndarray] = None       # (H, W, 3) uint8 — merged display color (LAS RGB)
    has_lod2: Optional[np.ndarray] = None        # (H, W) bool — true if LOD2 covered this cell
    lod2_top_y: Optional[np.ndarray] = None      # (H, W) int32 — LOD2 roof height above ground (blocks)

    # Semantics (filled by FuseLOD2Geometry)
    cell_kind: Optional[np.ndarray] = None       # (H, W) uint8 — 0=terrain, 1=building, 2=tree

    # Facade color (filled by SampleLOD2Facade): per-cell vertical color profile
    facade_color: Optional[np.ndarray] = None    # (H, W) object — (top_y+1, 3) uint8 per building cell, else None

    # Block IDs (filled by MapBlocks)
    block_top: Optional[np.ndarray] = None       # (H, W) object — MC block id strings
    block_side: Optional[np.ndarray] = None      # (H, W) object — MC block id strings
    block_column: Optional[np.ndarray] = None    # (H, W) object — per-height list[str] for buildings, else None

    # Grid geometry
    origin_x: float = 0.0
    origin_y: float = 0.0
    ground_z: float = 0.0
    cell_size: float = 1.0
    grid_w: int = 0
    grid_h: int = 0

    # Config + diagnostics
    config: dict = field(default_factory=dict)
    history: list = field(default_factory=list)


class Stage(ABC):
    """A pipeline step. Subclass and implement run()."""
    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def run(self, ctx: Context) -> Context: ...

    def __call__(self, ctx: Context) -> Context:
        t0 = time.time()
        log.info(f"--- {self.name} ---")
        ctx = self.run(ctx)
        dt = time.time() - t0
        log.info(f"    ({dt:.2f}s)")
        ctx.history.append({"stage": self.name, "seconds": dt})
        return ctx


class Pipeline:
    """Runs Stages in order on a Context."""
    def __init__(self, stages: List[Stage]):
        self.stages = list(stages)

    def run(self, ctx: Optional[Context] = None) -> Context:
        if ctx is None:
            ctx = Context()
        for s in self.stages:
            ctx = s(ctx)
        return ctx

    def remove_by_type(self, stage_type) -> "Pipeline":
        """Return a new Pipeline with all stages of given type removed."""
        return Pipeline([s for s in self.stages if not isinstance(s, stage_type)])
