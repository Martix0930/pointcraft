"""Export RAW LiDAR + LOD2 surface samples as PLY layers for manual 3D checking.

For human alignment QA (M0). Writes separate coloured PLY point clouds so you can
toggle layers in CloudCompare / MeshLab and eyeball, building by building, whether
the LOD2 roofs sit on the LiDAR roof points and where the LOD2 walls land:

    pointcloud.ply   raw LiDAR, coloured by height (viridis)
    lod2_roof.ply     LOD2 near-horizontal surface samples (red)
    lod2_facade.ply   LOD2 near-vertical surface samples (blue)

This is the source geometry — NO voxelization — so it reflects the data itself,
not any M0 processing. Same CRS for both, so they overlay directly.

Usage:
    python scripts/export_alignment_3d.py --config configs/tokyo_station.yaml \
        --outdir outputs/m0/align3d
    # options: --las-stride N (downsample points), --lod2-spacing M (sample step),
    #          --full-lod2 (don't clip LOD2 to the LiDAR XY extent)
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


def write_ply(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write a binary_little_endian PLY with float xyz + uchar rgb."""
    n = len(xyz)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("r", "u1"), ("g", "u1"), ("b", "u1")])
    arr = np.empty(n, dt)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["r"], arr["g"], arr["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())


def height_colors(z: np.ndarray) -> np.ndarray:
    try:
        import matplotlib.cm as cm
        t = (z - z.min()) / (np.ptp(z) + 1e-9)
        return (cm.viridis(t)[:, :3] * 255).astype(np.uint8)
    except Exception:
        g = np.clip((z - z.min()) / (np.ptp(z) + 1e-9) * 255, 0, 255).astype(np.uint8)
        return np.stack([g, g, g], axis=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export LiDAR+LOD2 PLY layers for 3D QA")
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default=os.path.join(REPO, "outputs", "m0", "align3d"))
    ap.add_argument("--las-stride", type=int, default=1, help="keep every Nth LiDAR point")
    ap.add_argument("--lod2-spacing", type=float, default=0.6, help="LOD2 surface sample step (m)")
    ap.add_argument("--roof-nz", type=float, default=0.7)
    ap.add_argument("--full-lod2", action="store_true", help="don't clip LOD2 to LiDAR XY extent")
    args = ap.parse_args(argv)

    import laspy

    cfg = load_config(args.config)
    base = os.path.dirname(os.path.abspath(args.config))
    las_paths = [resolve_path(p, base) for p in (cfg.get("las") or [])]
    lod2_paths = [resolve_path(p, base) for p in (cfg.get("lod2_tiles") or [])]
    os.makedirs(args.outdir, exist_ok=True)

    # --- LiDAR point cloud (height-coloured) ---
    pts = []
    for p in las_paths:
        las = laspy.read(p)
        pts.append(np.stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)], axis=1))
    pts = np.concatenate(pts, axis=0).astype(np.float64)
    if args.las_stride > 1:
        pts = pts[::args.las_stride]
    write_ply(os.path.join(args.outdir, "pointcloud.ply"), pts, height_colors(pts[:, 2]))
    print(f"[pointcloud] {len(pts):,} pts -> pointcloud.ply")

    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    print(f"  LiDAR XY extent: x[{x0:.1f},{x1:.1f}] y[{y0:.1f},{y1:.1f}]")

    # --- LOD2 surface samples, split roof vs facade ---
    verts, faces = load_lod2_meshes(lod2_paths)
    rng = np.random.default_rng(0)
    spacing = args.lod2_spacing
    roof_chunks, fac_chunks = [], []
    n_faces_used = 0
    for vi, _vti, _mat in faces:
        tri = verts[list(vi)]
        if not args.full_lod2:
            # skip faces whose XY bbox doesn't touch the LiDAR extent
            if (tri[:, 0].max() < x0 or tri[:, 0].min() > x1 or
                    tri[:, 1].max() < y0 or tri[:, 1].min() > y1):
                continue
        cross = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = float(np.linalg.norm(cross))
        if norm <= 0:
            continue
        area = 0.5 * norm
        nz = abs(cross[2] / norm)
        n = int(np.clip(np.ceil(area / (spacing * spacing)), 1, 200_000))
        p = _sample_triangle(tri, n, rng)
        (roof_chunks if nz >= args.roof_nz else fac_chunks).append(p)
        n_faces_used += 1

    roof = np.concatenate(roof_chunks, axis=0) if roof_chunks else np.zeros((0, 3))
    fac = np.concatenate(fac_chunks, axis=0) if fac_chunks else np.zeros((0, 3))
    write_ply(os.path.join(args.outdir, "lod2_roof.ply"), roof,
              np.tile(np.array([220, 40, 40], np.uint8), (len(roof), 1)))
    write_ply(os.path.join(args.outdir, "lod2_facade.ply"), fac,
              np.tile(np.array([40, 90, 220], np.uint8), (len(fac), 1)))
    print(f"[lod2]       {n_faces_used:,} faces "
          f"({'all' if args.full_lod2 else 'clipped to LiDAR extent'}) -> "
          f"roof {len(roof):,} pts (red), facade {len(fac):,} pts (blue)")
    print(f"\nOpen all 3 PLYs in CloudCompare/MeshLab (same CRS, they overlay).")
    print(f"Check per building: red roofs should cap the high LiDAR points;")
    print(f"blue walls should stand where the LiDAR is empty (the aerial blind spot).")
    print(f"Output dir: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
