"""Offscreen render of a saved context snapshot to PNGs, for headless iteration.

Usage:
    python scripts/render_views.py [snapshot.npz] [mode_index]

Saves output/renders/mode{N}_{angle}.png for visual inspection without a GUI.
"""
from __future__ import annotations
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
import pyvista as pv

from pointcraft.utils.viewer import load_context, expand_voxels, COLOR_MODES


def render(ctx_dict, mode_idx, out_dir, tag):
    voxels = expand_voxels(ctx_dict, include_columns=True)
    name, fn = COLOR_MODES[mode_idx]
    colors = fn(ctx_dict, voxels)
    pts = voxels["pos"].astype(np.float32) + 0.5
    cloud = pv.PolyData(pts)
    cloud["RGB"] = colors
    cube = pv.Cube(x_length=1.0, y_length=1.0, z_length=1.0)
    glyphs = cloud.glyph(geom=cube, scale=False, orient=False)

    cx, cy, cz = pts[:, 0].mean(), pts[:, 1].mean(), pts[:, 2].mean()
    span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 2]))

    # Tallest point -> close-up target on a building facade.
    ti = int(np.argmax(pts[:, 1]))
    tx, ty, tz = pts[ti, 0], pts[ti, 1], pts[ti, 2]

    # Head-on facade view: stand directly in front of the tower (offset along one
    # horizontal axis only) at mid-height, pulled back ~tower height so the wall
    # is seen flat (no grazing-angle foreshortening that fakes vertical stripes).
    dist = max(ty * 1.2, 80.0)

    views = {
        "persp": ((cx + span * 0.7, cz + span * 0.8, cy + span * 1.1), (cx, cz * 0.3, cy)),
        "top":   ((cx, cz + span * 2.0, cy + 1), (cx, 0, cy)),
        "close": ((tx, ty * 0.5, tz + dist), (tx, ty * 0.5, tz)),
    }
    os.makedirs(out_dir, exist_ok=True)
    for vname, (pos, foc) in views.items():
        p = pv.Plotter(off_screen=True, window_size=(1280, 800))
        p.set_background("white")
        p.add_mesh(glyphs, scalars="RGB", rgb=True, show_edges=False)
        # PyVista uses Z-up by default; our world is Y-up -> set camera in (x, y=height, z) space.
        p.camera_position = [pos, foc, (0, 1, 0)]
        fpath = os.path.join(out_dir, f"mode{mode_idx+1}_{tag}_{vname}.png")
        p.screenshot(fpath)
        p.close()
        print(f"  wrote {fpath}")


def main():
    snap = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "output", "last_ctx.npz")
    ctx = load_context(snap)
    out_dir = os.path.join(REPO, "output", "renders")
    modes = [int(sys.argv[2]) - 1] if len(sys.argv) > 2 else [0, 4]
    for m in modes:
        render(ctx, m, out_dir, "v")


if __name__ == "__main__":
    main()
