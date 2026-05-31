"""All pipeline stages. Each is a Stage subclass that operates on a Context.

Adding a new stage = subclass Stage, implement run(ctx). Compose into Pipeline.
"""
from __future__ import annotations
import os
from typing import List, Optional

import numpy as np
import laspy
import mcschematic
from scipy import ndimage

from .context import Context, Stage, log
from .lod2 import LOD2Rasterizer
from .palette import BlockPalette


# ============================================================
# Input
# ============================================================

class LoadLas(Stage):
    """Read one or more LAS / LAZ files into Context, merging point clouds.

    Pass a single path or a list of paths. All tiles must share a CRS
    (here EPSG:6677); points are concatenated into one cloud so the grid
    spans the full extent regardless of source tile boundaries.
    """

    def __init__(self, path):
        self.paths = [path] if isinstance(path, str) else list(path)

    def run(self, ctx):
        pts_all, cls_all, rgb_all = [], [], []
        any_rgb = False
        for p in self.paths:
            las = laspy.read(p)
            pts = np.stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)], axis=1)
            cls = np.asarray(las.classification)
            pts_all.append(pts)
            cls_all.append(cls)
            if "red" in las.point_format.dimension_names:
                rgb_all.append(np.stack([np.asarray(las.red), np.asarray(las.green), np.asarray(las.blue)], axis=1))
                any_rgb = True
            else:
                rgb_all.append(None)
            log.info(f"    {os.path.basename(p)}: {len(pts):,} pts")

        ctx.points = np.concatenate(pts_all, axis=0)
        ctx.classification = np.concatenate(cls_all, axis=0)
        if any_rgb:
            # Fill tiles lacking RGB with zeros so the concat aligns with points.
            filled = [r if r is not None else np.zeros((len(c), 3), dtype=np.uint16)
                      for r, c in zip(rgb_all, cls_all)]
            ctx.rgb = np.concatenate(filled, axis=0)

        log.info(f"    loaded {len(ctx.points):,} points from {len(self.paths)} tile(s)")
        log.info(f"    bbox X[{ctx.points[:,0].min():.1f}..{ctx.points[:,0].max():.1f}] "
                 f"Y[{ctx.points[:,1].min():.1f}..{ctx.points[:,1].max():.1f}] "
                 f"Z[{ctx.points[:,2].min():.1f}..{ctx.points[:,2].max():.1f}]")
        return ctx


# ============================================================
# Point-level cleanup
# ============================================================

class DropNoiseClass(Stage):
    """Remove ASPRS noise classes (default: 7=low noise, 18=high noise)."""

    def __init__(self, classes=(7, 18)):
        self.classes = set(classes)

    def run(self, ctx):
        if ctx.classification is None:
            return ctx
        keep = ~np.isin(ctx.classification, list(self.classes))
        removed = int((~keep).sum())
        ctx.points = ctx.points[keep]
        ctx.classification = ctx.classification[keep]
        if ctx.rgb is not None:
            ctx.rgb = ctx.rgb[keep]
        log.info(f"    dropped {removed:,} ({100*removed/len(keep):.2f}%)")
        return ctx


class PercentileZClip(Stage):
    """Clip extreme Z outliers (birds, scan glitches)."""

    def __init__(self, p_lo=0.05, p_hi=99.95):
        self.p_lo, self.p_hi = p_lo, p_hi

    def run(self, ctx):
        z = ctx.points[:, 2]
        lo = np.percentile(z, self.p_lo)
        hi = np.percentile(z, self.p_hi)
        keep = (z >= lo) & (z <= hi)
        removed = int((~keep).sum())
        ctx.points = ctx.points[keep]
        ctx.classification = ctx.classification[keep]
        if ctx.rgb is not None:
            ctx.rgb = ctx.rgb[keep]
        log.info(f"    Z range kept: [{lo:.2f}, {hi:.2f}], dropped {removed:,}")
        return ctx


# ============================================================
# Voxelization
# ============================================================

class Voxelize(Stage):
    """Build 1m (configurable) 2.5D grid: top Z + supporting point count + ground fraction."""

    def __init__(self, cell_size=1.0, ground_class=2):
        self.cell_size = cell_size
        self.ground_class = ground_class

    def run(self, ctx):
        pts = ctx.points
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        ox, oy = x.min(), y.min()
        if (ctx.classification == self.ground_class).any():
            ground_z = float(np.median(z[ctx.classification == self.ground_class]))
        else:
            ground_z = float(z.min())

        ix = np.floor((x - ox) / self.cell_size).astype(np.int32)
        iy = np.floor((y - oy) / self.cell_size).astype(np.int32)
        W = int(ix.max()) + 1
        H = int(iy.max()) + 1
        flat = iy.astype(np.int64) * W + ix.astype(np.int64)

        top_z = np.full(W * H, -np.inf, dtype=np.float32)
        np.maximum.at(top_z, flat, z.astype(np.float32))
        top_z = top_z.reshape(H, W)

        support = np.zeros(W * H, dtype=np.int32)
        np.add.at(support, flat, 1)
        support = support.reshape(H, W)

        ground_count = np.zeros(W * H, dtype=np.int32)
        np.add.at(ground_count, flat, (ctx.classification == self.ground_class).astype(np.int32))
        ground_count = ground_count.reshape(H, W)
        ground_frac = np.where(support > 0, ground_count / np.maximum(support, 1), 0.0).astype(np.float32)

        valid = np.isfinite(top_z) & (support > 0)
        top_y = np.where(valid, np.round(top_z - ground_z), -1).astype(np.int32)
        top_y = np.clip(top_y, -1, 280)

        # Per-cell LAS RGB: average only points near the top surface (top_z - 1m .. top_z)
        # so we capture the visible color, not the column-averaged grey.
        las_rgb = np.zeros((H * W, 3), dtype=np.float32)
        las_rgb_count = np.zeros(H * W, dtype=np.int32)
        if ctx.rgb is not None:
            top_z_per_pt = top_z.ravel()[flat]  # broadcast top_z to each point's cell
            near_top = z >= (top_z_per_pt - 1.0)
            f_near = flat[near_top]
            r8 = (ctx.rgb[near_top, 0].astype(np.float32) / 65535.0 * 255.0)
            g8 = (ctx.rgb[near_top, 1].astype(np.float32) / 65535.0 * 255.0)
            b8 = (ctx.rgb[near_top, 2].astype(np.float32) / 65535.0 * 255.0)
            np.add.at(las_rgb[:, 0], f_near, r8)
            np.add.at(las_rgb[:, 1], f_near, g8)
            np.add.at(las_rgb[:, 2], f_near, b8)
            np.add.at(las_rgb_count, f_near, 1)
        las_rgb_count_safe = np.maximum(las_rgb_count, 1)
        las_rgb_mean = (las_rgb / las_rgb_count_safe[:, None]).clip(0, 255).astype(np.uint8)
        las_rgb_mean = las_rgb_mean.reshape(H, W, 3)
        has_las_rgb = (las_rgb_count > 0).reshape(H, W) & valid

        ctx.origin_x, ctx.origin_y, ctx.ground_z = float(ox), float(oy), ground_z
        ctx.cell_size = self.cell_size
        ctx.grid_w, ctx.grid_h = W, H
        ctx.top_y, ctx.valid, ctx.support_count = top_y, valid, support
        ctx.ground_frac = ground_frac
        ctx.las_rgb = las_rgb_mean
        ctx.has_las_rgb = has_las_rgb
        # Release point memory (no stage after voxelize needs raw points in v0.1)
        ctx.points = None
        ctx.rgb = None
        log.info(f"    grid {W}x{H}, {valid.sum():,} valid cells ({valid.mean()*100:.1f}%)")
        log.info(f"    support median={np.median(support[valid]):.1f}, "
                 f"p10={np.percentile(support[valid],10):.0f}, max={support.max()}")
        return ctx


# ============================================================
# Grid-level noise filters
# ============================================================

class MinSupportFilter(Stage):
    """Drop cells with too few supporting points."""

    def __init__(self, min_points=3):
        self.min_points = min_points

    def run(self, ctx):
        before = int(ctx.valid.sum())
        weak = ctx.valid & (ctx.support_count < self.min_points)
        ctx.valid &= ~weak
        ctx.top_y[weak] = -1
        log.info(f"    removed {int(weak.sum()):,} weak cells "
                 f"(<{self.min_points} pts), {before:,} -> {int(ctx.valid.sum()):,}")
        return ctx


class LocalHeightOutlier(Stage):
    """Snap or remove cells whose height deviates from local median.

    Kills "bumps on flat ground" caused by transient objects (people, cars, leaves).
    """

    def __init__(self, window=5, threshold=2.0, mode="snap"):
        assert mode in ("remove", "snap")
        self.window = window
        self.threshold = threshold
        self.mode = mode

    def run(self, ctx):
        h = ctx.top_y.astype(np.float32)
        h[~ctx.valid] = np.nan
        r = self.window // 2
        H, W = h.shape
        hpad = np.pad(h, r, mode="edge")
        stack = np.empty((self.window * self.window, H, W), dtype=np.float32)
        k = 0
        for dy in range(self.window):
            for dx in range(self.window):
                stack[k] = hpad[dy:dy+H, dx:dx+W]
                k += 1
        med = np.nanmedian(stack, axis=0)
        diff = np.abs(h - med)
        outlier = ctx.valid & np.isfinite(diff) & (diff > self.threshold)
        n_out = int(outlier.sum())
        if self.mode == "remove":
            ctx.valid &= ~outlier
            ctx.top_y[outlier] = -1
            log.info(f"    removed {n_out:,} height outliers (>{self.threshold}m)")
        else:
            new_h = np.where(outlier, np.round(med), ctx.top_y).astype(np.int32)
            new_h[~ctx.valid] = -1
            ctx.top_y = new_h
            log.info(f"    snapped {n_out:,} height outliers to local median (>{self.threshold}m)")
        return ctx


class FillSingleStepHoles(Stage):
    """Median-fill cells with >=4 valid 8-neighbors. Mirrors C++ reference behavior."""

    def __init__(self, iterations=1):
        self.iterations = iterations

    def run(self, ctx):
        H, W = ctx.top_y.shape
        for _ in range(self.iterations):
            tp = np.pad(ctx.top_y.astype(np.float32), 1, mode="constant", constant_values=np.nan)
            vp = np.pad(ctx.valid, 1, mode="constant", constant_values=False)
            tp[~vp] = np.nan
            stack = []
            for dy in (0, 1, 2):
                for dx in (0, 1, 2):
                    if dy == 1 and dx == 1:
                        continue
                    stack.append(tp[dy:dy+H, dx:dx+W])
            stack = np.stack(stack, axis=0)
            n_neighbors = np.isfinite(stack).sum(axis=0)
            med = np.nanmedian(stack, axis=0)
            fillable = (~ctx.valid) & (n_neighbors >= 4) & np.isfinite(med)
            ctx.top_y = np.where(fillable, np.round(med).astype(np.int32), ctx.top_y)
            ctx.valid |= fillable
            log.info(f"    filled {int(fillable.sum()):,} holes")
        return ctx


class MorphologicalClose(Stage):
    """2D morphological close on the valid mask. Smooths jagged edges, fills tiny gaps."""

    def __init__(self, iterations=1):
        self.iterations = iterations

    def run(self, ctx):
        before = int(ctx.valid.sum())
        closed = ndimage.binary_closing(ctx.valid, iterations=self.iterations, border_value=1)
        closed |= ctx.valid  # extensiveness guard
        added = closed & ~ctx.valid
        if added.any():
            h = ctx.top_y.astype(np.float32)
            h[~ctx.valid] = np.nan
            hpad = np.pad(h, 1, mode="edge")
            stack = np.stack([hpad[i:i+h.shape[0], j:j+h.shape[1]]
                              for i in range(3) for j in range(3)], axis=0)
            med = np.nanmedian(stack, axis=0)
            new_h = np.where(added & np.isfinite(med), np.round(med).astype(np.int32), ctx.top_y)
            ctx.top_y = new_h
        ctx.valid = closed
        log.info(f"    closed: {before:,} -> {int(ctx.valid.sum()):,} (+{int(added.sum()):,})")
        return ctx


class RemoveSmallComponents(Stage):
    """Drop connected components of valid cells smaller than min_size."""

    def __init__(self, min_size=4):
        self.min_size = min_size

    def run(self, ctx):
        labels, n = ndimage.label(ctx.valid, structure=np.ones((3, 3), bool))
        if n == 0:
            return ctx
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        keep_label = sizes >= self.min_size
        keep_mask = keep_label[labels]
        removed = int(ctx.valid.sum()) - int(keep_mask.sum())
        ctx.valid &= keep_mask
        ctx.top_y[~ctx.valid] = -1
        log.info(f"    {n} components, removed {removed:,} cells in components < {self.min_size}")
        return ctx


# ============================================================
# Color
# ============================================================

class SampleLOD2Color(Stage):
    """Top-down z-buffered rasterization of LOD2 mesh -> per-cell RGB + has_lod2 mask."""

    def __init__(self, tile_dirs: List[str]):
        self.tile_dirs = tile_dirs

    def run(self, ctx):
        rast = LOD2Rasterizer(self.tile_dirs)
        ppm = 1.0 / ctx.cell_size
        rgb, zbuf, valid_lod2 = rast.rasterize(
            ctx.origin_x, ctx.origin_y,
            ctx.grid_w * ctx.cell_size, ctx.grid_h * ctx.cell_size,
            ppm=ppm,
        )
        ctx.has_lod2 = valid_lod2
        # LOD2 roof height -> blocks above ground. zbuf is world Z (max per cell).
        lod2_top_y = np.full(valid_lod2.shape, -1, dtype=np.int32)
        h = np.round(zbuf - ctx.ground_z).astype(np.int32)
        h = np.clip(h, 0, 280)
        lod2_top_y[valid_lod2] = h[valid_lod2]
        ctx.lod2_top_y = lod2_top_y
        cov_grid = valid_lod2.mean() * 100
        cov_valid = (valid_lod2 & ctx.valid).sum() / max(ctx.valid.sum(), 1) * 100
        log.info(f"    LOD2 coverage: {cov_grid:.1f}% of grid, {cov_valid:.1f}% of valid cells")
        return ctx


# ============================================================
# Geometry / semantic fusion
# ============================================================

class FuseLOD2Geometry(Stage):
    """Fuse LOD2 building geometry with the point-cloud skeleton.

    Architecture:
      * LOD2-covered cells  -> geometry (height + footprint) comes from LOD2,
        the point-cloud top is discarded (removes the over-thick "shell").
        Labelled kind=1 (building).
      * Tall cells with NO LOD2 -> ambiguous (small building vs tree, since the
        LAS has no building/veg class). Split by appearance:
            - green top  OR  rough local surface  -> tree   (kind=2, kept)
            - otherwise (flat, grey/compact)       -> small building (removed)
      * Short cells -> terrain / low vegetation (kind=0, kept).

    Tree detection signals:
      * greenness: LAS RGB green channel dominates (primary; user's cue).
      * roughness: high local std of height = bumpy canopy vs flat roof.
    """

    def __init__(self, tall_threshold=3.0, rough_window=3, rough_threshold=1.5,
                 green_margin=4):
        self.tall_threshold = tall_threshold      # blocks above ground to be "tall"
        self.rough_window = rough_window
        self.rough_threshold = rough_threshold     # height std (m) => canopy
        self.green_margin = green_margin           # g - max(r,b) to count as green

    def _local_std(self, height, valid):
        h = height.astype(np.float32)
        h[~valid] = np.nan
        r = self.rough_window // 2
        H, W = h.shape
        hpad = np.pad(h, r, mode="edge")
        stack = []
        for dy in range(self.rough_window):
            for dx in range(self.rough_window):
                stack.append(hpad[dy:dy+H, dx:dx+W])
        stack = np.stack(stack, axis=0)
        return np.nanstd(stack, axis=0)

    def run(self, ctx):
        H, W = ctx.valid.shape
        has_lod2 = ctx.has_lod2 if ctx.has_lod2 is not None else np.zeros((H, W), bool)
        lod2_h = ctx.lod2_top_y if ctx.lod2_top_y is not None else np.full((H, W), -1, np.int32)
        kind = np.zeros((H, W), dtype=np.uint8)  # 0 terrain

        # --- 1. Buildings from LOD2: override geometry, drop point-cloud shell ---
        lod2_cells = has_lod2 & (lod2_h >= 0)
        ctx.top_y = np.where(lod2_cells, lod2_h, ctx.top_y).astype(np.int32)
        ctx.valid = ctx.valid | lod2_cells
        kind[lod2_cells] = 1
        n_building = int(lod2_cells.sum())

        # --- 2. Tall non-LOD2 cells: tree vs small building -------------------
        tall = ctx.valid & (~has_lod2) & (ctx.top_y >= self.tall_threshold)

        # greenness from LAS RGB top color
        green = np.zeros((H, W), dtype=bool)
        if ctx.las_rgb is not None:
            r = ctx.las_rgb[:, :, 0].astype(np.int16)
            g = ctx.las_rgb[:, :, 1].astype(np.int16)
            b = ctx.las_rgb[:, :, 2].astype(np.int16)
            green = (g - np.maximum(r, b)) >= self.green_margin

        rough = self._local_std(ctx.top_y, ctx.valid) >= self.rough_threshold

        is_tree = tall & (green | rough)
        is_small_building = tall & ~is_tree

        kind[is_tree] = 2
        # remove small non-LOD2 buildings (and the point-cloud shell remnants)
        ctx.valid &= ~is_small_building
        ctx.top_y[is_small_building] = -1

        ctx.cell_kind = kind
        n_tree = int(is_tree.sum())
        n_removed = int(is_small_building.sum())
        n_terrain = int((ctx.valid & (kind == 0)).sum())
        log.info(f"    building(LOD2)={n_building:,}, tree={n_tree:,} "
                 f"(green+rough), terrain={n_terrain:,}, removed small-bldg={n_removed:,}")
        return ctx


# ============================================================
# Facade color (3D)
# ============================================================

class SampleLOD2Facade(Stage):
    """Per-voxel facade color for LOD2 buildings via 3D nearest-color lookup.

    The top-down raster only yields a single (roof) color per cell, so walls
    have no detail. Here we sample the LOD2 textured mesh surface into colored
    3D points, then for each building voxel (x, y, height) take the nearest
    sampled color. Result: ctx.facade_color[gy, gx] = (top_y+1, 3) uint8.
    """

    def __init__(self, tile_dirs: List[str], spacing=0.4, max_radius=6.0,
                 smooth_radius=2.0):
        self.tile_dirs = tile_dirs
        self.spacing = spacing
        self.max_radius = max_radius
        self.smooth_radius = smooth_radius

    def run(self, ctx):
        from scipy.spatial import cKDTree
        if ctx.cell_kind is None:
            log.info("    no cell_kind; skipping")
            return ctx
        rast = LOD2Rasterizer(self.tile_dirs)
        pts, cols = rast.colored_point_samples(
            ctx.origin_x, ctx.origin_y,
            ctx.grid_w * ctx.cell_size, ctx.grid_h * ctx.cell_size,
            spacing=self.spacing,
        )
        if len(pts) == 0:
            log.info("    no facade samples")
            return ctx
        # Work in (x, y, height-above-ground) so it matches voxel centers.
        sample_xyz = pts.copy()
        sample_xyz[:, 2] = pts[:, 2] - ctx.ground_z
        tree = cKDTree(sample_xyz)

        H, W = ctx.valid.shape
        building = ctx.valid & (ctx.cell_kind == 1)
        gy_idx, gx_idx = np.where(building)
        facade = np.empty((H, W), dtype=object)

        # Build query points for all building voxels at once.
        query_pts = []
        spans = []  # (gy, gx, top, start_index)
        cs = ctx.cell_size
        for gy, gx in zip(gy_idx, gx_idx):
            top = int(ctx.top_y[gy, gx])
            if top < 0:
                continue
            wx = ctx.origin_x + (gx + 0.5) * cs
            wy = ctx.origin_y + (gy + 0.5) * cs
            heights = np.arange(top + 1, dtype=np.float32) + 0.5
            qp = np.column_stack([np.full(top + 1, wx), np.full(top + 1, wy), heights])
            spans.append((gy, gx, top, len(query_pts)))
            query_pts.append(qp)
        if not query_pts:
            ctx.facade_color = facade
            return ctx
        query = np.concatenate(query_pts, axis=0)
        # PLATEAU facade textures carry a ~1 m-period vertical window/mullion
        # pattern. At 1 block = 1 m each column aliases onto either a dark window
        # or a light mullion, producing vertical pinstripes. To kill the aliasing
        # we average ALL facade samples inside a ball wide enough to span a full
        # window period (smooth_radius), so every wall voxel gets the integrated
        # facade tone rather than one phase of the pattern.
        cols_f = cols.astype(np.float32)
        Nq = len(query)
        q_color = np.zeros((Nq, 3), dtype=np.uint8)
        q_ok = np.zeros(Nq, dtype=bool)
        neigh_lists = tree.query_ball_point(query, r=self.smooth_radius, workers=-1)
        # Fallback single-nearest for query points whose ball is empty.
        miss = np.array([len(nl) == 0 for nl in neigh_lists])
        for qi in np.nonzero(~miss)[0]:
            nl = neigh_lists[qi]
            q_color[qi] = cols_f[nl].mean(axis=0).astype(np.uint8)
            q_ok[qi] = True
        if miss.any():
            d1, i1 = tree.query(query[miss], k=1,
                                distance_upper_bound=self.max_radius)
            ok1 = np.isfinite(d1) & (i1 < len(cols))
            mi = np.nonzero(miss)[0]
            q_color[mi[ok1]] = cols[i1[ok1]]
            q_ok[mi[ok1]] = True

        n_hit = 0
        n_cols_filled = 0
        cursor = 0
        for gy, gx, top, _ in spans:
            m = top + 1
            ok = q_ok[cursor:cursor + m]
            seg_color = q_color[cursor:cursor + m]
            cursor += m
            n_hit += int(ok.sum())
            if not ok.any():
                # No facade anywhere in this column (interior / no-material roof):
                # leave as None so MapBlocks colors it from LAS.
                facade[gy, gx] = None
                continue
            prof = np.zeros((m, 3), dtype=np.uint8)
            prof[ok] = seg_color[ok]
            # Vertical nearest-fill: extend wall colors to voxels that missed.
            present = np.where(ok)[0]
            pos = np.arange(m)
            nearest = present[np.argmin(np.abs(present[None, :] - pos[:, None]), axis=1)]
            prof[~ok] = prof[nearest[~ok]]
            facade[gy, gx] = prof
            n_cols_filled += 1

        ctx.facade_color = facade
        tot_vox = len(query)
        log.info(f"    facade colored {n_hit:,}/{tot_vox:,} building voxels "
                 f"({100*n_hit/max(tot_vox,1):.1f}% direct hits), "
                 f"{n_cols_filled:,} columns gap-filled")
        return ctx


# ============================================================
# Block selection
# ============================================================

class MapBlocks(Stage):
    """Choose top-block + side-block per cell.

    Color comes from LAS RGB (the accurate source on this dataset); LOD2 is
    used only for geometry, never color. Cells are handled by semantic kind:
        kind 0 terrain  -> LAS color -> nearest palette block
        kind 1 building -> LAS roof color -> palette; sides = darkened roof color
        kind 2 tree     -> leaves on top, log in the column (ignores palette)
    Cells with no LAS color fall back to a neutral grey.
    """

    def __init__(self, palette: Optional[BlockPalette] = None,
                 fallback_color: tuple = (130, 130, 130),
                 side_darken: float = 0.65,
                 las_saturation_boost: float = 1.6,
                 tree_top: str = "minecraft:oak_leaves",
                 tree_side: str = "minecraft:oak_log"):
        self.palette = palette or BlockPalette()
        self.fallback_color = np.array(fallback_color, dtype=np.uint8)
        self.side_darken = side_darken
        self.las_saturation_boost = las_saturation_boost
        self.tree_top = tree_top
        self.tree_side = tree_side

    def _boost_saturation_las(self, rgb: np.ndarray) -> np.ndarray:
        """LAS RGB on this dataset is heavily desaturated. Boost a*/b* in LAB."""
        if self.las_saturation_boost == 1.0:
            return rgb
        from skimage import color as skcolor
        rgb01 = rgb.astype(np.float32) / 255.0
        lab = skcolor.rgb2lab(rgb01.reshape(-1, 1, 3)).reshape(-1, 3)
        lab[:, 1] *= self.las_saturation_boost
        lab[:, 2] *= self.las_saturation_boost
        rgb_out = skcolor.lab2rgb(lab.reshape(-1, 1, 3)).reshape(-1, 3)
        return (rgb_out * 255).clip(0, 255).astype(np.uint8)

    def run(self, ctx):
        H, W = ctx.valid.shape
        kind = ctx.cell_kind if ctx.cell_kind is not None else np.zeros((H, W), np.uint8)
        facade = ctx.facade_color

        # LAS color (boosted) for terrain & color-less fallbacks.
        color = np.tile(self.fallback_color, (H, W, 1))
        n_las = 0
        if ctx.las_rgb is not None and ctx.has_las_rgb is not None:
            mask = ctx.valid & ctx.has_las_rgb
            if mask.any():
                color[mask] = self._boost_saturation_las(ctx.las_rgb[mask])
                n_las = int(mask.sum())

        block_top = np.empty((H, W), dtype=object)
        block_side = np.empty((H, W), dtype=object)
        block_column = np.empty((H, W), dtype=object)
        block_top[:] = "minecraft:stone"
        block_side[:] = "minecraft:cobblestone"

        # Trees: leaves canopy on top, log trunk below (per-voxel column).
        tree = ctx.valid & (kind == 2)
        block_top[tree] = self.tree_top
        block_side[tree] = self.tree_side
        for gy, gx in zip(*np.where(tree)):
            top = int(ctx.top_y[gy, gx])
            if top < 0:
                continue
            m = top + 1
            canopy = max(1, min(m, round(m * 0.5)))  # top half is leaves
            col = [self.tree_side] * (m - canopy) + [self.tree_top] * canopy
            block_column[gy, gx] = col

        # Terrain + buildings: tops & default sides from LAS color (accurate,
        # spatially coherent -> no striped roofs). Buildings then get per-voxel
        # facade colors on their *sides* via block_column below.
        nontree = ctx.valid & (kind != 2)
        if nontree.any():
            colors = color[nontree]
            top_ids = self.palette.nearest(colors)
            side_colors = np.clip(colors.astype(np.float32) * self.side_darken, 0, 255).astype(np.uint8)
            side_ids = self.palette.nearest(side_colors)
            for (r, c), tid, sid in zip(zip(*np.where(nontree)), top_ids, side_ids):
                block_top[r, c] = tid
                block_side[r, c] = sid

        # Building facade detail -> per-voxel side colors (LOD2). Roof (top) stays LAS.
        n_facade = 0
        if facade is not None:
            building = ctx.valid & (kind == 1)
            gy_idx, gx_idx = np.where(building)
            all_colors, spans = [], []
            for gy, gx in zip(gy_idx, gx_idx):
                prof = facade[gy, gx]
                if prof is None:
                    continue
                spans.append((gy, gx, len(all_colors), len(prof)))
                all_colors.append(prof)
            if all_colors:
                stacked = np.concatenate(all_colors, axis=0)
                ids = self.palette.nearest(stacked)
                for gy, gx, start, n in spans:
                    block_column[gy, gx] = list(ids[start:start + n])
                    n_facade += 1

        ctx.block_top = block_top
        ctx.block_side = block_side
        ctx.block_column = block_column
        ctx.top_color = color
        n_total = int(ctx.valid.sum())
        log.info(f"    blocks for {n_total:,} cells: nontree={int(nontree.sum()):,}, "
                 f"facade-columns={n_facade:,}, tree={int(tree.sum()):,}; LAS-colored={n_las:,}")
        return ctx


# ============================================================
# Output
# ============================================================

class EmitSchem(Stage):
    """Build .schem using per-cell block_top + block_side from MapBlocks."""

    def __init__(self, out_dir: str, schem_name: str,
                 version=mcschematic.Version.JE_1_21_1,
                 default_column: str = "minecraft:cobblestone",
                 default_top: str = "minecraft:stone"):
        self.out_dir = out_dir
        self.schem_name = schem_name
        self.version = version
        self.default_column = default_column
        self.default_top = default_top

    def run(self, ctx):
        os.makedirs(self.out_dir, exist_ok=True)
        schem = mcschematic.MCSchematic()
        H, W = ctx.top_y.shape
        total = 0
        for gy in range(H):
            for gx in range(W):
                if not ctx.valid[gy, gx]:
                    continue
                top = int(ctx.top_y[gy, gx])
                if top < 0:
                    continue
                top_block = ctx.block_top[gy, gx] if ctx.block_top is not None else self.default_top
                side_block = ctx.block_side[gy, gx] if ctx.block_side is not None else self.default_column
                column = ctx.block_column[gy, gx] if ctx.block_column is not None else None
                mx = gx
                mz = (H - 1) - gy
                for my in range(0, top):
                    blk = column[my] if (column is not None and my < len(column)) else side_block
                    schem.setBlock((mx, my, mz), blk)
                    total += 1
                schem.setBlock((mx, top, mz), top_block)
                total += 1
        schem.save(self.out_dir, self.schem_name, self.version)
        path = os.path.join(self.out_dir, self.schem_name + ".schem")
        log.info(f"    wrote {total:,} blocks -> {path} ({os.path.getsize(path)/1024:.0f} KB)")
        return ctx
