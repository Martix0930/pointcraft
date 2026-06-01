"""Export PER-BUILDING PLY layers (varied heights) for manual alignment QA.

The whole-tile export (`export_alignment_3d.py`) dumps everything at once; here we
pick several REPRESENTATIVE buildings spanning the height range — skyscraper /
mid-rise / low — so the manual check isn't biased toward one complex landmark
(e.g. Tokyo Station). Each building gets its own folder with 3 overlaying PLYs:

    bldg_<rank>_h<height>m/ { pointcloud.ply, lod2_roof.ply, lod2_facade.ply }

Buildings are found by non-maximum-suppression on the LOD2 roof-height map (pick
the tallest column, clear a window around it, repeat) so picks are spatially
spread and span heights. Source geometry, no voxelization.

Usage:
    python scripts/export_buildings.py --config configs/tokyo_station.yaml
    # options: --n 8  --window 40  --lod2-spacing 0.5  --las-stride 1
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np

from pointcraft.data.target import _sample_triangle, load_lod2_meshes
from pointcraft.utils.config import load_config, resolve_path


def write_ply(path, xyz, rgb):
    n = len(xyz)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("r", "u1"), ("g", "u1"), ("b", "u1")])
    arr = np.empty(n, dt)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["r"], arr["g"], arr["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())


def height_colors(zv):
    if len(zv) == 0:
        return np.zeros((0, 3), np.uint8)
    try:
        import matplotlib.cm as cm
        t = (zv - zv.min()) / (np.ptp(zv) + 1e-9)
        return (cm.viridis(t)[:, :3] * 255).astype(np.uint8)
    except Exception:
        g = np.clip((zv - zv.min()) / (np.ptp(zv) + 1e-9) * 255, 0, 255).astype(np.uint8)
        return np.stack([g, g, g], axis=1)


def band(height_above_ground):
    h = height_above_ground
    if h >= 40:
        return "skyscraper"
    if h >= 25:
        return "tall"
    if h >= 12:
        return "midrise"
    return "low"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Per-building PLY export for 3D QA")
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default=os.path.join(REPO, "outputs", "m0", "align3d", "buildings"))
    ap.add_argument("--n", type=int, default=8, help="number of buildings to export")
    ap.add_argument("--window", type=float, default=40.0, help="crop window edge (m)")
    ap.add_argument("--lod2-spacing", type=float, default=0.5)
    ap.add_argument("--las-stride", type=int, default=1)
    ap.add_argument("--roof-nz", type=float, default=0.7)
    ap.add_argument("--min-pts", type=int, default=400,
                    help="min LiDAR points in a window to count as point-covered")
    ap.add_argument("--per-band", type=int, default=2,
                    help="max buildings per height band (skyscraper/tall/midrise/low)")
    args = ap.parse_args(argv)

    import laspy

    cfg = load_config(args.config)
    base = os.path.dirname(os.path.abspath(args.config))
    las_paths = [resolve_path(p, base) for p in (cfg.get("las") or [])]
    lod2_paths = [resolve_path(p, base) for p in (cfg.get("lod2_tiles") or [])]
    vs = float(cfg.get("cell_size") or 1.0)
    os.makedirs(args.outdir, exist_ok=True)

    # --- load LiDAR ---
    pts = []
    for p in las_paths:
        las = laspy.read(p)
        pts.append(np.stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)], axis=1))
    pts = np.concatenate(pts, axis=0).astype(np.float64)
    if args.las_stride > 1:
        pts = pts[::args.las_stride]
    x0a, y0a = pts[:, 0].min(), pts[:, 1].min()
    x1a, y1a = pts[:, 0].max(), pts[:, 1].max()
    ground_z = float(np.percentile(pts[:, 2], 5))  # rough ground level for height labels

    # --- load LOD2 + keep faces overlapping the LiDAR extent ---
    verts, faces = load_lod2_meshes(lod2_paths)
    kept = []
    for vi, _vti, _mat in faces:
        tri = verts[list(vi)]
        if (tri[:, 0].max() < x0a or tri[:, 0].min() > x1a or
                tri[:, 1].max() < y0a or tri[:, 1].min() > y1a):
            continue
        cross = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = float(np.linalg.norm(cross))
        if norm <= 0:
            continue
        nz = abs(cross[2] / norm)
        kept.append((tri, 0.5 * norm, nz))

    # --- roof-height map on a coarse (i,j) grid for peak picking ---
    I = int(np.ceil((x1a - x0a) / vs)) + 1
    J = int(np.ceil((y1a - y0a) / vs)) + 1
    topz = np.full((J, I), -np.inf)
    for tri, _area, _nz in kept:
        ii = np.clip(((tri[:, 0] - x0a) / vs).astype(int), 0, I - 1)
        jj = np.clip(((tri[:, 1] - y0a) / vs).astype(int), 0, J - 1)
        zc = tri[:, 2].max()
        for a, b in zip(jj, ii):
            if zc > topz[a, b]:
                topz[a, b] = zc

    # --- NMS peak picking, then quota PER HEIGHT BAND ---
    # Collect spatially-spread peaks (suppress a window after each), keep those
    # with some LiDAR support, then take up to --per-band from each height band so
    # the sample spans skyscraper / tall / midrise / low — not just the tallest.
    work = topz.copy()
    r = int(args.window / 2 / vs)
    half0 = args.window / 2.0
    cands = []  # (cx, cy, zc, height, band, npts)
    n_skipped = 0
    while True:
        flat = int(np.argmax(work))
        j, i = divmod(flat, I)
        zc = work[j, i]
        if not np.isfinite(zc):
            break
        cx = x0a + i * vs
        cy = y0a + j * vs
        work[max(0, j - r):j + r, max(0, i - r):i + r] = -np.inf
        npts = int(((np.abs(pts[:, 0] - cx) <= half0) & (np.abs(pts[:, 1] - cy) <= half0)).sum())
        if npts < args.min_pts:
            n_skipped += 1
            continue
        h = zc - ground_z
        cands.append((cx, cy, zc, h, band(h), npts))
    if n_skipped:
        print(f"(skipped {n_skipped} LOD2 peaks with <{args.min_pts} LiDAR pts — "
              f"no point support, likely outside the swath)")

    order = ["skyscraper", "tall", "midrise", "low"]
    by_band = {b: sorted([c for c in cands if c[4] == b], key=lambda c: -c[2])
               for b in order}
    picks = []
    for b in order:
        picks.extend(by_band[b][:args.per_band])
    # if a band was empty, top up with the next tallest leftovers to reach --n
    if len(picks) < args.n:
        chosen = {(c[0], c[1]) for c in picks}
        leftovers = sorted([c for c in cands if (c[0], c[1]) not in chosen],
                           key=lambda c: -c[2])
        picks.extend(leftovers[:args.n - len(picks)])
    picks.sort(key=lambda c: -c[2])
    print("band coverage: " + ", ".join(f"{b}={len(by_band[b])}" for b in order))

    rng = np.random.default_rng(0)
    half = args.window / 2.0
    print(f"picked {len(picks)} buildings (window {args.window:.0f} m, ground≈{ground_z:.1f} m):")
    for rank, (cx, cy, zc, h, bnd, npts) in enumerate(picks):
        sparse = "_SPARSEpts" if npts < 5000 else ""
        name = f"bldg_{rank:02d}_{bnd}_h{h:.0f}m{sparse}"
        d = os.path.join(args.outdir, name)
        os.makedirs(d, exist_ok=True)

        # local point cloud
        msk = (np.abs(pts[:, 0] - cx) <= half) & (np.abs(pts[:, 1] - cy) <= half)
        lp = pts[msk]
        write_ply(os.path.join(d, "pointcloud.ply"), lp, height_colors(lp[:, 2]))

        # local LOD2 roof/facade samples
        roof_c, fac_c = [], []
        for tri, area, nz in kept:
            if (tri[:, 0].max() < cx - half or tri[:, 0].min() > cx + half or
                    tri[:, 1].max() < cy - half or tri[:, 1].min() > cy + half):
                continue
            nsamp = int(np.clip(np.ceil(area / (args.lod2_spacing ** 2)), 1, 50_000))
            sp = _sample_triangle(tri, nsamp, rng)
            (roof_c if nz >= args.roof_nz else fac_c).append(sp)
        roof = np.concatenate(roof_c) if roof_c else np.zeros((0, 3))
        fac = np.concatenate(fac_c) if fac_c else np.zeros((0, 3))
        write_ply(os.path.join(d, "lod2_roof.ply"), roof,
                  np.tile(np.array([220, 40, 40], np.uint8), (len(roof), 1)))
        write_ply(os.path.join(d, "lod2_facade.ply"), fac,
                  np.tile(np.array([40, 90, 220], np.uint8), (len(fac), 1)))
        print(f"  {name:28s} center=({cx:.0f},{cy:.0f}) roofZ={zc:.1f}  "
              f"pts={len(lp):,} roof={len(roof):,} facade={len(fac):,}")

    print(f"\nEach folder = one building. Open its 3 PLYs together in CloudCompare.")
    print(f"'_SPARSEpts' = few LiDAR points (coverage-edge tower, not a clean "
          f"alignment sample). Heights span {picks[-1][3]:.0f}..{picks[0][3]:.0f} m; "
          f"output: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
