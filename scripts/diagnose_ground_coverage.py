#!/usr/bin/env python
"""diagnose_ground_coverage.py — QA sweep of ASPRS class-2 (ground) in the LiDAR.

For each LAS tile, report — read-only, no model, no voxelization:
  * class-2 (GROUND) point fraction,
  * the set of ASPRS classes present,
  * ground **XY coverage of the sensed footprint** at `--cell` m:
      ground_cov_footprint = (cells with >=1 class-2 return) / (cells with >=1 return)
    The complement is the **inter-building / under-roof holes** — aerial LiDAR cannot
    see ground beneath roofs, so the missing fraction is structured (densest in dense,
    articulated tiles), not random.
  * ground_cov_bbox = ground cells / all bbox cells (raw, for reference).

This is a **diagnostic / QA layer only** (M2 fork-1 follow-up): class-2 is authoritative
ground *where present*, used to validate the existing CityGML building-base ground and as
**sparse ground anchors**. It is deliberately NOT interpolated into a continuous DEM —
the occlusion holes are kept honest behind an explicit unknown mask rather than fabricated
by TIN/IDW. Any integration into the model is a later scale-up step under a NEW
dataset_version, and does not touch the D10 z_scale/above_ground feature semantics.

    python scripts/diagnose_ground_coverage.py --out outputs/g1/ground_class2_sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np

GROUND_CLASS = 2  # ASPRS standard: 2 = ground


def scan_tile(path: str, cell: float) -> dict:
    import laspy

    las = laspy.read(path)
    cls = np.asarray(las.classification)
    x = np.asarray(las.x)
    y = np.asarray(las.y)
    n = int(cls.size)
    present = sorted(int(v) for v in np.unique(cls))
    has_ground = GROUND_CLASS in present
    pct_ground = 100.0 * float((cls == GROUND_CLASS).sum()) / max(n, 1)

    xmin, ymin = float(x.min()), float(y.min())
    xmax, ymax = float(x.max()), float(y.max())
    J = max(int(np.floor((ymax - ymin) / cell)) + 1, 1)
    bbox_cells = max(int(np.floor((xmax - xmin) / cell)) + 1, 1) * J

    def n_cells(mask: np.ndarray) -> int:
        ix = np.floor((x[mask] - xmin) / cell).astype(np.int64)
        iy = np.floor((y[mask] - ymin) / cell).astype(np.int64)
        return int(np.unique(ix * J + iy).size)

    any_cells = n_cells(np.ones(n, dtype=bool))
    g_cells = n_cells(cls == GROUND_CLASS) if has_ground else 0
    return {
        "tile": os.path.splitext(os.path.basename(path))[0],
        "n_pts": n,
        "classes": "|".join(str(c) for c in present),
        "pct_ground": round(pct_ground, 1),
        "ground_cov_footprint": round(100.0 * g_cells / any_cells, 1) if any_cells else 0.0,
        "ground_cov_bbox": round(100.0 * g_cells / bbox_cells, 1),
        "ground_cells": g_cells,
        "any_cells": any_cells,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ASPRS class-2 ground QA sweep")
    ap.add_argument("--lidar-glob", default="data/raw/lidar/09LD*.las")
    ap.add_argument("--out", default="outputs/g1/ground_class2_sweep.csv")
    ap.add_argument("--cell", type=float, default=1.0, help="XY cell size (m); match the voxel size")
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(args.lidar_glob))
    if not paths:
        raise SystemExit(f"no LAS matched {args.lidar_glob!r}")
    print(f"sweeping {len(paths)} tiles (class-2 fraction + XY ground coverage @ {args.cell} m) ...",
          flush=True)
    rows = []
    for p in paths:
        try:
            rows.append(scan_tile(p, args.cell))
        except Exception as e:  # keep scanning the rest
            print(f"  ! {os.path.basename(p)}: {e}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    pcts = [r["pct_ground"] for r in rows]
    covf = [r["ground_cov_footprint"] for r in rows]
    no_ground = [r["tile"] for r in rows if str(GROUND_CLASS) not in r["classes"].split("|")]
    all_classes = sorted({int(c) for r in rows for c in r["classes"].split("|")})
    print(f"\n=== {len(rows)} tiles ===")
    print(f"class {GROUND_CLASS} present in {len(rows)-len(no_ground)}/{len(rows)} tiles"
          + (f"; MISSING: {no_ground}" if no_ground else " (ALL)"))
    print(f"class-{GROUND_CLASS} pct:  min {min(pcts)}  mean {round(float(np.mean(pcts)),1)}  max {max(pcts)}")
    print(f"ground XY coverage of footprint:  min {min(covf)}%  mean {round(float(np.mean(covf)),1)}%  max {max(covf)}%")
    print(f"union of classes across tiles: {all_classes}  (1=unclassified, 2=ground, 3=low veg)")
    print("\nworst-10 footprint coverage (most inter-building / under-roof holes):")
    for r in sorted(rows, key=lambda r: r["ground_cov_footprint"])[:10]:
        print(f"  {r['tile']}  cov_foot {r['ground_cov_footprint']:>5}%  pct_ground {r['pct_ground']:>5}%")
    print(f"\nwrote {args.out}  ({len(rows)} tiles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
