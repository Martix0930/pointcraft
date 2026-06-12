"""exp_004 §2b pre-flight — MEASURE (do not guess) K and candidate-support for
high-nfr / high-height_std candidate tiles, in-memory, writing NO config and NO
.npz to the train set.

K (grid Z depth) and candidate support depend only on the LiDAR partial cloud:
the grid is built from the LiDAR extent and candidate_support() runs on
coords_partial. The LOD2/CityGML target is NOT needed, so this is cheap and
leaves no train-set artifacts behind.

HWM reference (proven trainable): 09LD1885 = 3,765,570 support voxels, K=259,
peak CUDA 3385 MB @ exp_003.

Usage:
    .venv/Scripts/python scripts/preflight_support.py 09LD1874 09LD1843 ...
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np

from pointcraft.baseline.predictors import candidate_support
from pointcraft.data import load_las_xyz, voxelize_partial
from pointcraft.voxelization import VoxelGrid

HWM_SUPPORT = 3_765_570   # 09LD1885, proven trainable @ 3385 MB
HWM_K = 259
VOXEL = 1.0
LIDAR_DIR = os.path.join(REPO, "data", "raw", "lidar")


def measure(tile_id: str) -> dict:
    las = os.path.join(LIDAR_DIR, f"{tile_id}.las")
    t0 = time.time()
    pts = load_las_xyz(las)
    # Same grid construction as run_m0 (exclusive upper bound).
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    n_cells = np.floor((mx - mn) / VOXEL).astype(np.int64) + 1
    bounds = [*mn.tolist(), *(mn + n_cells * VOXEL).tolist()]
    grid = VoxelGrid.from_bounds(bounds, VOXEL)
    coords_p, _ = voxelize_partial(pts, grid)
    sup = candidate_support(coords_p, grid)
    I, J, K = (int(s) for s in grid.shape)
    return {
        "tile_id": tile_id,
        "n_lidar": int(len(pts)),
        "I": I, "J": J, "K": K,
        "total_vox": I * J * K,
        "occ_partial": int(len(coords_p)),
        "support": int(len(sup)),
        "support_frac_of_hwm": len(sup) / HWM_SUPPORT,
        "fits_hwm": len(sup) <= HWM_SUPPORT,
        "secs": round(time.time() - t0, 1),
    }


def main() -> None:
    tiles = sys.argv[1:] or ["09LD1874", "09LD1843", "09LD2818", "09LD1876", "09LD1864"]
    rows = []
    for t in tiles:
        print(f"... measuring {t}", flush=True)
        rows.append(measure(t))

    print()
    print("=" * 92)
    print(f"PRE-FLIGHT  K/support  (HWM = 1885: {HWM_SUPPORT:,} support, K={HWM_K}, 3385 MB)")
    print("=" * 92)
    hdr = (f"{'tile':10s} {'K':>4s} {'total_vox':>13s} {'occ_part':>10s} "
           f"{'support':>11s} {'/HWM':>6s}  verdict")
    print(hdr)
    print("-" * 92)
    for r in sorted(rows, key=lambda x: x["support"]):
        v = "FITS" if r["fits_hwm"] else "** EXCEEDS HWM **"
        print(f"{r['tile_id']:10s} {r['K']:>4d} {r['total_vox']:>13,d} "
              f"{r['occ_partial']:>10,d} {r['support']:>11,d} "
              f"{r['support_frac_of_hwm']:>5.2f}x  {v}  ({r['secs']}s)")
    print("=" * 92)


if __name__ == "__main__":
    main()
