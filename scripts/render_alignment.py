"""Render per-building alignment as 2D CROSS-SECTION images (no 3D software needed).

For reviewers who can't open .ply in a 3D viewer. A 3D scatter is cluttered and
occluded; a thin vertical slice through the building is far easier to judge.

For each representative building we cut two thin slabs through its centre and plot
them as 2D profiles (x or y horizontal, z = height):

    black = LiDAR points (the real observation)
    red   = LOD2 roof samples (the "answer" roof)
    blue  = LOD2 facade samples (the "answer" walls)

What to look for in each profile:
  * do the RED roof points sit right on top of the BLACK point tops?  -> roof aligned
  * are the BLUE walls in places where there are NO black points?      -> genuine
    aerial blind spot (good — that's what completion must fill)
  * red/blue floating with no black points anywhere near -> coverage gap (LOD2 has
    a building the LiDAR never sampled; suspect "answer").

Usage:
    python scripts/render_alignment.py        # run export_buildings.py first
"""
from __future__ import annotations

import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyvista as pv

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BDIR = os.path.join(REPO, "outputs", "m0", "align3d", "buildings")
OUTDIR = os.path.join(REPO, "outputs", "m0", "align3d", "renders")
SLAB = 2.0  # slab half-thickness (m)


def load_xyz(path):
    if not os.path.exists(path):
        return np.zeros((0, 3))
    return np.asarray(pv.read(path).points)


def pick_buildings():
    dirs = sorted(glob.glob(os.path.join(BDIR, "bldg_*")))
    if not dirs:
        raise SystemExit(f"no buildings in {BDIR}; run scripts/export_buildings.py first")
    # cross-sections only make sense where the cloud actually covers the building;
    # _SPARSEpts towers have no point support to slice against, so skip them here.
    dense = [d for d in dirs if "_SPARSEpts" not in os.path.basename(d)]
    return dense[:6]


def _slice(ax, pc, roof, fac, axis, c0, title):
    """axis=0 -> EW profile (x horiz); axis=1 -> NS profile (y horiz)."""
    other = 1 - axis
    def near(pts):
        if len(pts) == 0:
            return pts[:, axis] if len(pts) else np.array([]), np.array([])
        m = np.abs(pts[:, other] - c0) <= SLAB
        return pts[m, axis], pts[m, 2]
    rx, rz = near(roof)
    fx, fz = near(fac)
    px, pz = near(pc)
    ax.scatter(fx, fz, s=4, c="dodgerblue", alpha=0.5, label="LOD2 wall", zorder=1)
    ax.scatter(rx, rz, s=4, c="red", alpha=0.6, label="LOD2 roof", zorder=2)
    ax.scatter(px, pz, s=2, c="black", alpha=0.6, label="LiDAR pts", zorder=3)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(("x (east)" if axis == 0 else "y (north)") + " [m]")
    ax.set_ylabel("z height [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=7, loc="upper right", markerscale=2)


def render_one(d):
    name = os.path.basename(d)
    pc = load_xyz(os.path.join(d, "pointcloud.ply"))
    roof = load_xyz(os.path.join(d, "lod2_roof.ply"))
    fac = load_xyz(os.path.join(d, "lod2_facade.ply"))
    # centre = densest point column (a real building under the cloud), fallback mean
    if len(pc):
        cx, cy = pc[:, 0].mean(), pc[:, 1].mean()
    else:
        allp = np.vstack([a for a in (roof, fac) if len(a)])
        cx, cy = allp[:, 0].mean(), allp[:, 1].mean()

    # Clip ALL layers to the same XY box around the centre. LOD2 faces get sampled
    # whole even if only their edge touches the window, so they can spill far past
    # the point cloud — clip them back so we compare like-for-like.
    HALF = 20.0
    def boxclip(a):
        if len(a) == 0:
            return a
        m = (np.abs(a[:, 0] - cx) <= HALF) & (np.abs(a[:, 1] - cy) <= HALF)
        return a[m]
    pc, roof, fac = boxclip(pc), boxclip(roof), boxclip(fac)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    _slice(ax[0], pc, roof, fac, axis=0, c0=cy, title=f"{name}\nEW slice (|y-{cy:.0f}|<{SLAB}m)")
    _slice(ax[1], pc, roof, fac, axis=1, c0=cx, title=f"{name}\nNS slice (|x-{cx:.0f}|<{SLAB}m)")
    fig.tight_layout()
    out = os.path.join(OUTDIR, f"{name}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out, len(pc), len(roof), len(fac)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    for d in pick_buildings():
        out, npc, nr, nf = render_one(d)
        print(f"{os.path.basename(out):40s} pts={npc:,} roof={nr:,} facade={nf:,}")
    print(f"\nimages in {OUTDIR}")


if __name__ == "__main__":
    main()
