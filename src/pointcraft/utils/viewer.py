"""Interactive voxel viewer (PyVista). Loads a saved Context snapshot.

Keys (also printed when launched):
    1   MC block color  (final output — what MC will actually show)
    2   LOD2 raw color  (pre-palette-quantization)
    3   Data source     (red=LOD2, green=ground, blue=fallback)
    4   Height heatmap
    5   Toggle column visibility (top-only / full)
    R   Reset camera
    Q   Quit
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pyvista as pv

from ..mc_export.palette import PALETTE


# ---------- Save / load context snapshot ----------

def save_context(ctx, path: str) -> None:
    """Save the parts of Context needed for rendering."""
    H, W = ctx.top_y.shape
    np.savez_compressed(
        path,
        top_y=ctx.top_y,
        valid=ctx.valid,
        ground_frac=ctx.ground_frac if ctx.ground_frac is not None else np.zeros((H, W), dtype=np.float32),
        top_color=ctx.top_color if ctx.top_color is not None else np.zeros((H, W, 3), dtype=np.uint8),
        has_lod2=ctx.has_lod2 if ctx.has_lod2 is not None else np.zeros((H, W), dtype=bool),
        las_rgb=ctx.las_rgb if ctx.las_rgb is not None else np.zeros((H, W, 3), dtype=np.uint8),
        has_las_rgb=ctx.has_las_rgb if ctx.has_las_rgb is not None else np.zeros((H, W), dtype=bool),
        block_top=np.array(ctx.block_top, dtype=object) if ctx.block_top is not None else np.empty((H, W), dtype=object),
        block_side=np.array(ctx.block_side, dtype=object) if ctx.block_side is not None else np.empty((H, W), dtype=object),
        block_column=np.array(ctx.block_column, dtype=object) if ctx.block_column is not None else np.empty((H, W), dtype=object),
        cell_kind=ctx.cell_kind if ctx.cell_kind is not None else np.zeros((H, W), dtype=np.uint8),
        cell_size=np.array([ctx.cell_size]),
    )
    print(f"[viewer] saved context -> {path}")


def load_context(path: str) -> dict:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


# ---------- Voxel expansion ----------

def expand_voxels(ctx_dict, include_columns: bool = True) -> dict:
    """Expand 2.5D grid -> per-voxel arrays for rendering."""
    valid = ctx_dict["valid"]
    top_y = ctx_dict["top_y"]
    H, W = top_y.shape
    pos_list, is_top, gx_list, gy_list = [], [], [], []
    for gy in range(H):
        for gx in range(W):
            if not valid[gy, gx]:
                continue
            ty = int(top_y[gy, gx])
            if ty < 0:
                continue
            mx, mz = gx, (H - 1) - gy
            if include_columns:
                for my in range(0, ty):
                    pos_list.append((mx, my, mz)); is_top.append(False)
                    gx_list.append(gx); gy_list.append(gy)
            pos_list.append((mx, ty, mz)); is_top.append(True)
            gx_list.append(gx); gy_list.append(gy)
    return {
        "pos":    np.array(pos_list, dtype=np.int32),
        "is_top": np.array(is_top, dtype=bool),
        "gx":     np.array(gx_list, dtype=np.int32),
        "gy":     np.array(gy_list, dtype=np.int32),
    }


# ---------- Color modes ----------

BLOCK_RGB = {bid: np.array(rgb, dtype=np.uint8) for bid, rgb in PALETTE}
# Supplemental colors for block IDs emitted but intentionally kept out of the
# color-matching palette (so they don't steal nearest-color assignments).
for bid, rgb in (("minecraft:stone", (125, 125, 125)),
                 ("minecraft:cobblestone", (122, 122, 122)),
                 ("minecraft:oak_log", (101, 76, 47)),
                 ("minecraft:spruce_log", (58, 42, 24))):
    BLOCK_RGB.setdefault(bid, np.array(rgb, dtype=np.uint8))


def colors_mc_blocks(ctx_dict, voxels):
    block_top = ctx_dict["block_top"]
    block_side = ctx_dict["block_side"]
    block_column = ctx_dict.get("block_column")
    N = len(voxels["pos"])
    out = np.full((N, 3), 128, dtype=np.uint8)
    gx, gy, is_top = voxels["gx"], voxels["gy"], voxels["is_top"]
    my = voxels["pos"][:, 1]
    magenta = np.array((200, 0, 200), dtype=np.uint8)
    for i in range(N):
        if is_top[i]:
            bid = block_top[gy[i], gx[i]]
        else:
            col = block_column[gy[i], gx[i]] if block_column is not None else None
            if col is not None and my[i] < len(col):
                bid = col[my[i]]
            else:
                bid = block_side[gy[i], gx[i]]
        out[i] = BLOCK_RGB.get(bid, magenta)
    return out


def colors_lod2_raw(ctx_dict, voxels):
    top_color = ctx_dict["top_color"]
    has_lod2 = ctx_dict["has_lod2"]
    N = len(voxels["pos"])
    out = np.full((N, 3), 80, dtype=np.uint8)
    gx, gy, is_top = voxels["gx"], voxels["gy"], voxels["is_top"]
    for i in range(N):
        if has_lod2[gy[i], gx[i]]:
            c = top_color[gy[i], gx[i]]
            if not is_top[i]:
                c = (c.astype(np.float32) * 0.65).clip(0, 255).astype(np.uint8)
            out[i] = c
    return out


def colors_source(ctx_dict, voxels):
    """red=LOD2, orange=LAS RGB, blue=fallback."""
    has_lod2 = ctx_dict["has_lod2"]
    has_las_rgb = ctx_dict.get("has_las_rgb")
    if has_las_rgb is None or has_las_rgb.shape != has_lod2.shape:
        has_las_rgb = np.zeros_like(has_lod2)
    N = len(voxels["pos"])
    out = np.zeros((N, 3), dtype=np.uint8)
    gx, gy = voxels["gx"], voxels["gy"]
    for i in range(N):
        if has_lod2[gy[i], gx[i]]:
            out[i] = (220, 50, 50)
        elif has_las_rgb[gy[i], gx[i]]:
            out[i] = (230, 150, 40)
        else:
            out[i] = (60, 90, 220)
    return out


def colors_semantic(ctx_dict, voxels):
    """terrain=tan, building=blue, tree=green."""
    kind = ctx_dict.get("cell_kind")
    N = len(voxels["pos"])
    out = np.full((N, 3), 150, dtype=np.uint8)
    if kind is None:
        return out
    gx, gy = voxels["gx"], voxels["gy"]
    palette = {0: (170, 150, 110), 1: (70, 110, 210), 2: (60, 170, 60)}
    for i in range(N):
        out[i] = palette.get(int(kind[gy[i], gx[i]]), (150, 150, 150))
    return out


def colors_height(ctx_dict, voxels):
    ys = voxels["pos"][:, 1].astype(np.float32)
    if ys.max() == ys.min():
        norm = np.zeros_like(ys)
    else:
        norm = (ys - ys.min()) / (ys.max() - ys.min())
    r = (norm * 255).astype(np.uint8)
    g = (np.where(norm < 0.5, norm * 2 * 255, (1 - (norm - 0.5) * 2) * 255)).astype(np.uint8)
    b = ((1 - norm) * 255).astype(np.uint8)
    return np.stack([r, g, b], axis=1)


COLOR_MODES = [
    ("MC block color",   colors_mc_blocks),
    ("LAS raw color",    colors_lod2_raw),
    ("Data source",      colors_source),
    ("Height heatmap",   colors_height),
    ("Semantic kind",    colors_semantic),
]


# ---------- Viewer ----------

class Viewer:
    def __init__(self, ctx_dict):
        self.ctx = ctx_dict
        print("[viewer] expanding voxels...")
        self.voxels = expand_voxels(ctx_dict, include_columns=True)
        print(f"[viewer] {len(self.voxels['pos']):,} voxels")
        self.mode_idx = 0
        self.show_columns = True
        self.plotter = pv.Plotter(window_size=(1280, 800))
        self.plotter.set_background("white")
        # Lock camera so the world's +Y is always "up". Rotation is restricted to
        #   horizontal orbit (azimuth, free) + vertical pitch (elevation, clamped).
        # The model can never roll or flip upside-down.
        self._min_elev = 3.0    # degrees above horizontal (never go under the model)
        self._max_elev = 89.0   # degrees (near top-down, never cross the zenith)
        self._locking = False
        try:
            self.plotter.enable_terrain_style(mouse_wheel_zooms=True)
        except Exception:
            pass
        # Terrain style alone still lets the view pass the zenith; enforce a hard
        # clamp on every interaction event.
        try:
            self.plotter.iren.add_observer("InteractionEvent", self._lock_camera)
            self.plotter.iren.add_observer("EndInteractionEvent", self._lock_camera)
        except Exception as e:
            print(f"[viewer] camera-lock observer unavailable: {e}")
        self.actor = None
        self._build()
        self._bind_keys()
        self._show_help()

    def _lock_camera(self, *args):
        """Force world-up = +Y and clamp camera elevation to [min, max]."""
        if self._locking:
            return
        self._locking = True
        try:
            cam = self.plotter.camera
            pos = np.array(cam.GetPosition(), dtype=float)
            foc = np.array(cam.GetFocalPoint(), dtype=float)
            vec = pos - foc
            r = float(np.linalg.norm(vec))
            if r > 1e-9:
                elev = np.degrees(np.arcsin(np.clip(vec[1] / r, -1.0, 1.0)))
                azim = np.degrees(np.arctan2(vec[2], vec[0]))
                elev_c = float(np.clip(elev, self._min_elev, self._max_elev))
                if abs(elev_c - elev) > 1e-6:
                    e, a = np.radians(elev_c), np.radians(azim)
                    new_vec = r * np.array([np.cos(e) * np.cos(a),
                                            np.sin(e),
                                            np.cos(e) * np.sin(a)])
                    cam.SetPosition(*(foc + new_vec))
            cam.SetViewUp(0.0, 1.0, 0.0)
        finally:
            self._locking = False

    def _build(self):
        if self.actor is not None:
            self.plotter.remove_actor(self.actor)
        mode_name, mode_fn = COLOR_MODES[self.mode_idx]
        mask = np.ones(len(self.voxels["pos"]), dtype=bool) if self.show_columns else self.voxels["is_top"]
        sub_vox = {k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in self.voxels.items()}
        colors = mode_fn(self.ctx, sub_vox)
        pts = self.voxels["pos"][mask].astype(np.float32) + 0.5
        cloud = pv.PolyData(pts)
        cloud["RGB"] = colors
        cube = pv.Cube(x_length=1.0, y_length=1.0, z_length=1.0)
        glyphs = cloud.glyph(geom=cube, scale=False, orient=False)
        self.actor = self.plotter.add_mesh(glyphs, scalars="RGB", rgb=True, show_edges=False)
        self.plotter.add_text(
            f"Mode: {mode_name}  (1-5 switch, C toggle columns, R reset, Q quit)",
            name="title", position="upper_edge", font_size=10, color="black",
        )

    def _bind_keys(self):
        for i, _ in enumerate(COLOR_MODES, start=1):
            self.plotter.add_key_event(str(i), lambda idx=i-1: self._set_mode(idx))
        self.plotter.add_key_event("c", self._toggle_columns)
        self.plotter.add_key_event("r", self.plotter.reset_camera)

    def _set_mode(self, idx):
        self.mode_idx = idx
        self._build()
        self.plotter.render()

    def _toggle_columns(self):
        self.show_columns = not self.show_columns
        self._build()
        self.plotter.render()

    def _show_help(self):
        print("=" * 50)
        for i, (name, _) in enumerate(COLOR_MODES, start=1):
            print(f"  {i} = {name}")
        print("  C = toggle column visibility")
        print("  R = reset camera")
        print("  Q = quit")
        print("=" * 50)

    def show(self):
        self.plotter.show()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "output/last_ctx.npz"
    print(f"[viewer] loading {path}")
    Viewer(load_context(path)).show()


if __name__ == "__main__":
    main()
