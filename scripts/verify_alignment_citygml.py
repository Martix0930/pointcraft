"""C2 alignment gate: CityGML LOD2 surfaces vs LiDAR on tile 09LD1874 (D5).

Re-verifies, now from the CityGML target (not OBJ), that:
  * LiDAR roof points sit on CityGML RoofSurface (datum + 6697->6677 reprojection
    both correct), and
  * the building base reads as GroundSurface (no longer a normal-heuristic roof).

Two outputs (both decisive on their own):
  1. Quantitative z-gate: over 2 m XY cells covering buildings, compare LiDAR
     top-z vs CityGML roof-z. Median |diff| near 0 => aligned (cf. the Phase B
     OBJ z-gate which gave median +0.32 m).
  2. Visual cross-sections: for a few dense buildings, thin EW/NS slabs plotting
     LiDAR (black) vs CityGML roof (red) / facade (blue) / ground (green).
     PNGs -> outputs/m0/align_citygml/ (git-ignored).

Run: python scripts/verify_alignment_citygml.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from pointcraft.data import parse_citygml  # noqa: E402
from pointcraft.data.citygml import GROUND_LABEL  # noqa: E402
from pointcraft.data.target import FACADE_LABEL, ROOF_LABEL, _sample_triangle  # noqa: E402

LAS = os.path.join(REPO, "data", "raw", "lidar", "09LD1874.las")
GRID_DIR = os.path.join(REPO, "data", "raw", "citygml", "udx", "bldg")
GRIDS = ["53394610", "53394611", "53394620", "53394621"]  # cover 09LD1874
OUTDIR = os.path.join(REPO, "outputs", "m0", "align_citygml")
SLAB = 2.0   # cross-section half-thickness (m)
HALF = 25.0  # cross-section XY half-window (m)


def load_las_xyz(path):
    import laspy

    las = laspy.read(path)
    return np.column_stack([las.x, las.y, las.z]).astype(np.float64)


def sample_surfaces(ts, labels_wanted, spacing=0.5, seed=0):
    """Fan-triangulate each typed ring and barycentric-sample -> (pts, labels)."""
    rng = np.random.default_rng(seed)
    pts, labs = [], []
    for ring, lab in zip(ts.polygons, ts.labels):
        if lab not in labels_wanted or ring.shape[0] < 3:
            continue
        v0 = ring[0]
        for i in range(1, ring.shape[0] - 1):
            tri = np.stack([v0, ring[i], ring[i + 1]])
            cross = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            norm = float(np.linalg.norm(cross))
            if norm <= 1e-9:
                continue
            area = 0.5 * norm
            n = int(np.clip(np.ceil(area / (spacing * spacing)), 1, 20_000))
            p = _sample_triangle(tri, n, rng)
            pts.append(p)
            labs.append(np.full(n, lab, dtype=np.int64))
    if not pts:
        return np.zeros((0, 3)), np.zeros((0,), dtype=np.int64)
    return np.concatenate(pts), np.concatenate(labs)


def z_gate(lidar, roof_pts, cell=2.0):
    """Compare LiDAR top-z vs CityGML roof-z per XY cell over the overlap."""
    if len(roof_pts) == 0:
        print("  z-gate: no roof samples in window"); return
    x0 = min(lidar[:, 0].min(), roof_pts[:, 0].min())
    y0 = min(lidar[:, 1].min(), roof_pts[:, 1].min())

    def topz(pts):
        ij = np.floor((pts[:, :2] - [x0, y0]) / cell).astype(np.int64)
        key = ij[:, 0].astype(np.int64) * 100000 + ij[:, 1]
        order = np.argsort(key)
        key, z = key[order], pts[order, 2]
        uniq, start = np.unique(key, return_index=True)
        tops = np.maximum.reduceat(z, start)
        return dict(zip(uniq.tolist(), tops.tolist()))

    lt, rt = topz(lidar), topz(roof_pts)
    common = set(lt) & set(rt)
    if not common:
        print("  z-gate: no shared cells"); return
    diff = np.array([lt[k] - rt[k] for k in common])
    print(f"  z-gate over {len(common)} shared 2 m cells (LiDAR_top - CityGML_roof):")
    print(f"    median {np.median(diff):+.2f} m | p10 {np.percentile(diff,10):+.2f} "
          f"| p90 {np.percentile(diff,90):+.2f} | mean {diff.mean():+.2f}")
    print(f"    |diff|<=1 m: {100*np.mean(np.abs(diff)<=1):.0f}%")


def render_buildings(lidar, ts, n_buildings=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUTDIR, exist_ok=True)
    # Pick building centres = densest LiDAR columns within the CityGML coverage.
    roof = ts.polygons  # use roof-surface centroids as building anchors
    roof_c = np.array([
        [p[:, 0].mean(), p[:, 1].mean()]
        for p, l in zip(ts.polygons, ts.labels) if l == ROOF_LABEL
    ])
    if len(roof_c) == 0:
        print("  no roof centroids to anchor cross-sections"); return
    # space anchors out (greedy) so we don't slice the same building 4x
    chosen = []
    for c in roof_c[np.argsort(-_local_pt_density(roof_c, lidar))]:
        if all(np.hypot(*(c - q)) > 40 for q in chosen):
            chosen.append(c)
        if len(chosen) >= n_buildings:
            break

    for bi, (cx, cy) in enumerate(chosen):
        def boxclip(a):
            if len(a) == 0:
                return a
            m = (np.abs(a[:, 0] - cx) <= HALF) & (np.abs(a[:, 1] - cy) <= HALF)
            return a[m]

        pc = boxclip(lidar)
        rp, _ = sample_surfaces_window(ts, {ROOF_LABEL}, cx, cy)
        fp, _ = sample_surfaces_window(ts, {FACADE_LABEL}, cx, cy)
        gp, _ = sample_surfaces_window(ts, {GROUND_LABEL}, cx, cy)

        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        _slice(ax[0], pc, rp, fp, gp, axis=0, c0=cy,
               title=f"bldg{bi} @({cx:.0f},{cy:.0f})  EW (|y-{cy:.0f}|<{SLAB}m)")
        _slice(ax[1], pc, rp, fp, gp, axis=1, c0=cx,
               title=f"bldg{bi} @({cx:.0f},{cy:.0f})  NS (|x-{cx:.0f}|<{SLAB}m)")
        fig.tight_layout()
        out = os.path.join(OUTDIR, f"bldg{bi}.png")
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"  {os.path.basename(out)}: LiDAR={len(pc):,} roof={len(rp):,} "
              f"facade={len(fp):,} ground={len(gp):,}")


def _local_pt_density(centres, lidar, r=15.0):
    """Rough LiDAR point count near each centre (coarse, for ranking)."""
    # bin lidar to a coarse grid and look up
    cell = 5.0
    x0, y0 = lidar[:, 0].min(), lidar[:, 1].min()
    ij = np.floor((lidar[:, :2] - [x0, y0]) / cell).astype(np.int64)
    from collections import Counter
    cnt = Counter(map(tuple, ij))
    out = np.zeros(len(centres))
    for k, c in enumerate(centres):
        i, j = np.floor((c - [x0, y0]) / cell).astype(np.int64)
        out[k] = sum(cnt.get((i + di, j + dj), 0)
                     for di in (-1, 0, 1) for dj in (-1, 0, 1))
    return out


def sample_surfaces_window(ts, labels, cx, cy, spacing=0.4):
    """Sample only surfaces whose ring centroid is within the HALF window."""
    sub_polys, sub_labs = [], []
    for ring, lab in zip(ts.polygons, ts.labels):
        if lab not in labels:
            continue
        if abs(ring[:, 0].mean() - cx) <= HALF and abs(ring[:, 1].mean() - cy) <= HALF:
            sub_polys.append(ring)
            sub_labs.append(lab)
    from types import SimpleNamespace
    sub = SimpleNamespace(polygons=sub_polys, labels=np.array(sub_labs, dtype=np.int64))
    return sample_surfaces(sub, labels, spacing=spacing)


def _slice(ax, pc, roof, fac, ground, axis, c0, title):
    other = 1 - axis

    def near(pts):
        if len(pts) == 0:
            return np.array([]), np.array([])
        m = np.abs(pts[:, other] - c0) <= SLAB
        return pts[m, axis], pts[m, 2]

    for pts, c, lab, z in [(fac, "dodgerblue", "CityGML wall", 1),
                           (ground, "limegreen", "CityGML ground", 2),
                           (roof, "red", "CityGML roof", 3)]:
        hx, hz = near(pts)
        ax.scatter(hx, hz, s=4, c=c, alpha=0.6, label=lab, zorder=z)
    px, pz = near(pc)
    ax.scatter(px, pz, s=2, c="black", alpha=0.6, label="LiDAR pts", zorder=4)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(("x (east)" if axis == 0 else "y (north)") + " [m]")
    ax.set_ylabel("z height [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=7, loc="upper right", markerscale=2)


def main():
    print(f"LAS: {LAS}")
    lidar = load_las_xyz(LAS)
    x0, y0 = lidar[:, 0].min(), lidar[:, 1].min()
    x1, y1 = lidar[:, 0].max(), lidar[:, 1].max()
    print(f"  {len(lidar):,} pts  bbox X[{x0:.0f},{x1:.0f}] Y[{y0:.0f},{y1:.0f}]")

    # Parse + merge the tile's CityGML grids, clip to LAS footprint.
    files = [os.path.join(GRID_DIR, f"{g}_bldg_6697_2_op.gml") for g in GRIDS]
    files = [f for f in files if os.path.exists(f)]
    from pointcraft.data import load_citygml
    ts = load_citygml(files)
    keep_polys, keep_labs = [], []
    for ring, lab in zip(ts.polygons, ts.labels):
        cx, cy = ring[:, 0].mean(), ring[:, 1].mean()
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            keep_polys.append(ring)
            keep_labs.append(lab)
    from types import SimpleNamespace
    ts = SimpleNamespace(polygons=keep_polys, labels=np.array(keep_labs, dtype=np.int64))
    vals, cnts = np.unique(ts.labels, return_counts=True)
    name = {ROOF_LABEL: "roof", FACADE_LABEL: "facade", GROUND_LABEL: "ground"}
    print(f"  CityGML surfaces inside tile: {len(keep_polys)} "
          f"{ {name.get(int(v),int(v)): int(c) for v,c in zip(vals,cnts)} }")

    print("\n[1] Quantitative z-gate:")
    roof_pts, _ = sample_surfaces(ts, {ROOF_LABEL}, spacing=0.5)
    z_gate(lidar, roof_pts)

    print("\n[2] Cross-section renders:")
    render_buildings(lidar, ts, n_buildings=4)
    print(f"\nPNGs in {OUTDIR}")


if __name__ == "__main__":
    main()
