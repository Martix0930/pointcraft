"""M0 end-to-end: raw LiDAR + LOD2 → one paired voxel sample (.npz).

One command, no manual steps. Builds the shared VoxelGrid from the LiDAR extent,
voxelizes the partial (LiDAR) and the target (LOD2 shell) on that SAME grid,
derives the observed/unobserved masks, and writes a contract-compliant .npz.

Usage:
    # real tile from a config (configs/tokyo_station.yaml):
    python scripts/run_m0.py --config configs/tokyo_station.yaml --out outputs/m0/tokyo.npz

    # explicit inputs:
    python scripts/run_m0.py --las a.las --lod2 dir1 dir2 \
        --out out.npz --tile-id foo --crs EPSG:6677 --voxel-size 1.0

    # tiny committed fixture (fast smoke, no large data):
    python scripts/run_m0.py --fixture --out outputs/m0/tiny.npz

Add --viz to also save a PNG sanity view next to the .npz.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np

from pointcraft.data import (
    build_metadata,
    compute_masks,
    load_citygml,
    load_las_xyz,
    load_lod2_meshes,
    voxelize_citygml_target,
    voxelize_partial,
    voxelize_target,
    write_sample_npz,
)
from pointcraft.utils.config import load_config, resolve_path
from pointcraft.voxelization import VoxelGrid

FIXTURE_DIR = os.path.join(REPO, "test_data", "m0_data_pairing")


def _load_points(path: str) -> np.ndarray:
    """Load (P,3) world XYZ from a .las/.laz or a .csv with x,y,z columns."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".las", ".laz"):
        return load_las_xyz(path)
    if ext == ".csv":
        xyz = []
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                xyz.append([float(row["x"]), float(row["y"]), float(row["z"])])
        return np.array(xyz, dtype=np.float64)
    raise ValueError(f"unsupported point file: {path}")


def _resolve_args(args: argparse.Namespace):
    """Return (las_paths, target_paths, tile_id, crs, voxel_size).

    ``target_paths`` are CityGML .gml files when ``--target citygml`` (default,
    D5) or OBJ dirs/files when ``--target obj`` (fallback).
    """
    if args.fixture:
        # The tiny fixture is an OBJ cube; force the OBJ target path for it.
        args.target = "obj"
        return (
            [os.path.join(FIXTURE_DIR, "tiny_lidar_points.csv")],
            [os.path.join(FIXTURE_DIR, "tiny_lod2_cube.obj")],
            args.tile_id or "tiny_synthetic_0",
            args.crs or "LOCAL_SYNTHETIC",
            args.voxel_size or 1.0,
        )
    las, crs, vsize, tile = args.las, args.crs, args.voxel_size, args.tile_id
    target = args.citygml if args.target == "citygml" else args.lod2
    if args.config:
        cfg = load_config(args.config)
        base = os.path.dirname(os.path.abspath(args.config))
        if not las:
            las = [resolve_path(p, base) for p in (cfg.get("las") or [])]
        if not target:
            key = "citygml_tiles" if args.target == "citygml" else "lod2_tiles"
            target = [resolve_path(p, base) for p in (cfg.get(key) or [])]
        crs = crs or cfg.get("crs")
        vsize = vsize or cfg.get("cell_size")
        tile = tile or cfg.get("name")
    if not las or not target:
        raise SystemExit(
            f"need --las and --{args.target} (or --config, or --fixture)"
        )
    return las, target, tile or "tile", crs or "UNKNOWN", float(vsize or 1.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M0 LiDAR+LOD2 → paired voxel .npz")
    ap.add_argument("--config", help="YAML config (las, citygml_tiles/lod2_tiles, crs, cell_size, name)")
    ap.add_argument("--las", nargs="+", help="LiDAR file(s) .las/.laz/.csv")
    ap.add_argument("--target", choices=["citygml", "obj"], default="citygml",
                    help="target source (default citygml, D5; obj = fallback)")
    ap.add_argument("--citygml", nargs="+", help="CityGML .gml file(s) (target=citygml)")
    ap.add_argument("--lod2", nargs="+", help="LOD2 tile dir(s) or .obj file(s) (target=obj)")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--tile-id")
    ap.add_argument("--crs")
    ap.add_argument("--voxel-size", type=float)
    ap.add_argument("--fixture", action="store_true", help="use the tiny committed fixture")
    ap.add_argument("--viz", action="store_true", help="also save a PNG sanity view")
    args = ap.parse_args(argv)

    las_paths, target_paths, tile_id, crs, voxel_size = _resolve_args(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    # --- partial (LiDAR) ---
    pts = np.concatenate([_load_points(p) for p in las_paths], axis=0)
    # Tile extent derives from LiDAR coverage (contract rule 4). Use an EXCLUSIVE
    # upper bound (floor+1 cells) so a point exactly on max() — e.g. an integer
    # roof z — still lands inside the grid rather than flooring one cell past it.
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    n_cells = np.floor((mx - mn) / voxel_size).astype(np.int64) + 1
    bounds = [*mn.tolist(), *(mn + n_cells * voxel_size).tolist()]
    grid = VoxelGrid.from_bounds(bounds, voxel_size)
    print(f"[grid]    {grid}  ({grid.num_voxels:,} voxels) from {len(pts):,} LiDAR pts")
    coords_p, feats_p = voxelize_partial(pts, grid)
    print(f"[partial] {coords_p.shape[0]:,} occupied voxels, feats {feats_p.shape}")

    # --- target (shell, D2) ---
    if args.target == "citygml":
        # Parse + merge the tile's CityGML grids, reprojected to the grid CRS,
        # then keep only surfaces whose ring centroid lands in the grid XY extent
        # (the grids cover more than one tile) before voxelizing.
        surfaces = load_citygml(target_paths)
        gx0, gy0 = grid.origin[0], grid.origin[1]
        gx1 = gx0 + grid.shape[0] * voxel_size
        gy1 = gy0 + grid.shape[1] * voxel_size
        keep_p, keep_l = [], []
        for ring, lab in zip(surfaces.polygons, surfaces.labels):
            cx, cy = ring[:, 0].mean(), ring[:, 1].mean()
            if gx0 <= cx <= gx1 and gy0 <= cy <= gy1:
                keep_p.append(ring)
                keep_l.append(lab)
        from types import SimpleNamespace
        surfaces = SimpleNamespace(
            polygons=keep_p, labels=np.array(keep_l, dtype=np.int64)
        )
        print(f"[citygml] {len(keep_p):,} surfaces inside tile (of {len(target_paths)} grids)")
        coords_t, occ_t, sem_t = voxelize_citygml_target(surfaces, grid)
        print(f"[target]  {coords_t.shape[0]:,} shell voxels "
              f"(roof {(sem_t == 3).sum():,}, facade {(sem_t == 4).sum():,}, "
              f"ground {(sem_t == 1).sum():,})")
    else:
        verts, faces = load_lod2_meshes(target_paths)
        coords_t, occ_t, sem_t = voxelize_target(verts, faces, grid)
        print(f"[target]  {coords_t.shape[0]:,} shell voxels "
              f"(roof {(sem_t == 3).sum():,}, facade {(sem_t == 4).sum():,})")

    # --- masks + write ---
    # Class-aware masks (D6, v0.2): z-tolerance for horizontal surfaces, genuine
    # mid-wall for facades. (OBJ target has no ground class but shares labels 3/4.)
    observed, unobserved = compute_masks(coords_t, coords_p, grid, sem_target=sem_t)
    print(f"[masks]   observed {int(observed.sum()):,}, "
          f"unobserved {int(unobserved.sum()):,} "
          f"({100*unobserved.mean():.1f}% of target = completion region)")
    src_files = [os.path.basename(p) for p in (*las_paths, *target_paths)]
    meta = build_metadata(grid, tile_id=tile_id, crs=crs, source_files=src_files)
    write_sample_npz(
        args.out,
        coords_partial=coords_p, feats_partial=feats_p,
        coords_target=coords_t, occ_target=occ_t, sem_target=sem_t,
        observed_mask=observed, unobserved_mask=unobserved, metadata=meta,
    )
    print(f"[write]   {args.out}  ({os.path.getsize(args.out)/1024:.0f} KB)")

    if args.viz:
        png = os.path.splitext(args.out)[0] + "_sanity.png"
        _sanity_view(grid, coords_p, feats_p, coords_t, sem_t, unobserved, png)
        print(f"[viz]     {png}")
    return 0


def _sanity_view(grid, coords_p, feats_p, coords_t, sem_t, unobserved, png):
    """Save a 1x3 matplotlib sanity figure (top-down partial/target + a slice)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover - optional dep
        print(f"[viz]     skipped (matplotlib unavailable: {e})")
        return
    I, J, K = (int(s) for s in grid.shape)

    def topdown_max(coords, weight):
        """Top-down (J,I) image, max of `weight` per column; empty cells -> NaN."""
        img = np.full((J, I), -np.inf)
        np.maximum.at(img, (coords[:, 1], coords[:, 0]), weight.astype(float))
        img[~np.isfinite(img)] = np.nan
        return img

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    im0 = ax[0].imshow(topdown_max(coords_p, feats_p[:, 0]), origin="lower", cmap="viridis")
    ax[0].set_title("partial: top-down max height")
    fig.colorbar(im0, ax=ax[0], fraction=0.046)

    # target top-down: highest voxel's semantic per column; empty -> NaN (white)
    sem_top = topdown_max(coords_t, sem_t)  # roof(3) wins ties as the higher layer
    im1 = ax[1].imshow(sem_top, origin="lower", cmap="Set1", vmin=3, vmax=4)
    ax[1].set_title("target: top-down semantic (roof=3 / facade=4)")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, ticks=[3, 4])

    # vertical slice through the column with the most target voxels -> facade fill.
    # Layers: 1=target-only(observed shell), 2=partial point, 3=UNobserved facade.
    jrow = int(np.bincount(coords_t[:, 1], minlength=J).argmax())
    sec = np.full((K, I), np.nan)
    mt = coords_t[:, 1] == jrow
    sec[coords_t[mt, 2], coords_t[mt, 0]] = 1
    munob = mt & (unobserved == 1)
    sec[coords_t[munob, 2], coords_t[munob, 0]] = 3
    mp = coords_p[:, 1] == jrow
    sec[coords_p[mp, 2], coords_p[mp, 0]] = 2
    im2 = ax[2].imshow(sec, origin="lower", cmap="Set1", vmin=1, vmax=3)
    ax[2].set_title(f"slice j={jrow}: shell=1, partial=2, UNobserved=3")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, ticks=[1, 2, 3])

    for a in ax:
        a.set_xlabel("i (x)")
    ax[0].set_ylabel("j (y)"); ax[2].set_ylabel("k (z)")
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
