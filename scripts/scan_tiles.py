#!/usr/bin/env python
"""
scan_tiles.py — G1 zero-cost tile scan (PointCraft M2 fork-1).

Per candidate LAS tile, clip the on-hand CityGML surfaces to the LAS footprint
(centroid-in-bbox, same rule as the M0 Phase E coverage scan) and report, in ONE
read-only pass, three groups of coarse metrics:

  coverage   : surf_roof / surf_facade / surf_ground, coverage_ok
  density    : footprint_ratio, surf_per_ha, building_count, mean_nn_spacing_m
  complexity : height_std_m, non_flat_roof_ratio, roof_per_building

NOTHING here is voxelized, run through run_m0, or trained. It only reads LAS
*headers* (bbox + point count) and the already-parsed CityGML rings. Loading the
9 GMLs once is the only real cost (~seconds).

Why these columns (see G1_EXECUTION_SPEC.md §1):
  density   -> proxies the "candidate coverage too small" failure (dense blocks
               where B1 merges buildings and buries interior walls).
  complexity-> proxies the "hallucination" failure (setbacks / overhangs / pitched
               roofs the model gets wrong).
The held-out tile should be HIGH on both; train tiles regular (low), with a gradient.

After this runs: paste the printed table back; a human picks K=3..4 regular train
tiles + 1 hard held-out (disjoint).

------------------------------------------------------------------------------
TODO(repo) — confirm these 3 against the actual repo before running:
  (1) import path + return shape of parse_citygml / load_citygml
  (2) label int values for roof / wall(facade) / ground
  (3) LAS directory + glob (default data/raw/lidar/09LD*.las)
------------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from dataclasses import dataclass

import numpy as np

# ---- TODO(repo) (1): adjust to the real module path / return type ------------
# Expected: load_citygml(paths) -> object with
#   .polygons : list of (n,3) float arrays (rings, already reprojected to 6677)
#   .labels   : int64 array aligned to .polygons (surface-type code)
from pointcraft.data.citygml import load_citygml  # noqa: E402

# ---- TODO(repo) (2): set to the repo's surface-type label codes --------------
LABEL_ROOF = 3      # RoofSurface
LABEL_FACADE = 4    # WallSurface
LABEL_GROUND = 1    # GroundSurface
# ------------------------------------------------------------------------------

import laspy  # noqa: E402

NON_FLAT_NZ_THRESH = 0.9   # |n_z| < this  => "non-flat" roof face
MIN_ROOF_FOR_COVERAGE = 200


# ----------------------------- geometry helpers -------------------------------
def xy_area(ring: np.ndarray) -> float:
    """Shoelace area of a ring's XY projection."""
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def xy_centroid(ring: np.ndarray) -> np.ndarray:
    return ring[:, :2].mean(axis=0)


def newell_nz(ring: np.ndarray) -> float:
    """|n_z| of a 3D polygon's Newell normal (1 = horizontal/flat)."""
    a = ring
    b = np.roll(ring, -1, axis=0)
    nz = np.sum((a[:, 0] - b[:, 0]) * (a[:, 1] + b[:, 1]))
    nx = np.sum((a[:, 1] - b[:, 1]) * (a[:, 2] + b[:, 2]))
    ny = np.sum((a[:, 2] - b[:, 2]) * (a[:, 0] + b[:, 0]))
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    return abs(nz) / norm if norm > 0 else 0.0


def mean_nn_spacing(centroids: np.ndarray) -> float:
    """Mean nearest-neighbour distance among building centroids (brute force;
    building counts are small, so no scipy dependency)."""
    if len(centroids) < 2:
        return float("nan")
    d2 = ((centroids[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return float(np.sqrt(d2.min(axis=1)).mean())


# ----------------------------- per-ring cache ---------------------------------
@dataclass
class RingCache:
    cx: np.ndarray        # (N,) centroid x
    cy: np.ndarray        # (N,) centroid y
    label: np.ndarray     # (N,) int
    area: np.ndarray      # (N,) XY area
    meanz: np.ndarray     # (N,) ring mean z
    nz: np.ndarray        # (N,) |n_z|


def build_ring_cache(surfaces) -> RingCache:
    polys, labels = surfaces.polygons, np.asarray(surfaces.labels)
    n = len(polys)
    cx = np.empty(n); cy = np.empty(n); area = np.empty(n)
    meanz = np.empty(n); nz = np.empty(n)
    for i, ring in enumerate(polys):
        ring = np.asarray(ring, dtype=float)
        c = xy_centroid(ring)
        cx[i], cy[i] = c[0], c[1]
        area[i] = xy_area(ring)
        meanz[i] = ring[:, 2].mean()
        nz[i] = newell_nz(ring) if labels[i] == LABEL_ROOF else 1.0
    return RingCache(cx, cy, labels.astype(int), area, meanz, nz)


# ----------------------------- per-tile metrics -------------------------------
def las_bbox_and_count(path: str):
    """Read LAS header only (no point records): (xmin,ymin,xmax,ymax, npts)."""
    with laspy.open(path) as f:
        h = f.header
        return (float(h.x_min), float(h.y_min),
                float(h.x_max), float(h.y_max), int(h.point_count))


def scan_tile(path: str, rc: RingCache) -> dict:
    xmin, ymin, xmax, ymax, npts = las_bbox_and_count(path)
    inside = (rc.cx >= xmin) & (rc.cx <= xmax) & (rc.cy >= ymin) & (rc.cy <= ymax)

    is_roof = inside & (rc.label == LABEL_ROOF)
    is_fac = inside & (rc.label == LABEL_FACADE)
    is_grd = inside & (rc.label == LABEL_GROUND)

    n_roof, n_fac, n_grd = int(is_roof.sum()), int(is_fac.sum()), int(is_grd.sum())
    n_total = n_roof + n_fac + n_grd

    tile_area = max((xmax - xmin) * (ymax - ymin), 1e-9)
    footprint_ratio = float(rc.area[is_grd].sum() / tile_area)
    surf_per_ha = float(n_total / (tile_area / 1e4))
    building_count = n_grd  # GroundSurface proxy; LOD2 ground can be sparse
    grd_centroids = np.column_stack([rc.cx[is_grd], rc.cy[is_grd]])
    spacing = mean_nn_spacing(grd_centroids)

    roof_z = rc.meanz[is_roof]
    height_std = float(roof_z.std()) if n_roof > 1 else float("nan")
    non_flat = float((rc.nz[is_roof] < NON_FLAT_NZ_THRESH).mean()) if n_roof else float("nan")
    roof_per_bldg = float(n_roof / building_count) if building_count else float("nan")

    coverage_ok = (n_roof >= MIN_ROOF_FOR_COVERAGE) and (n_grd >= 1)

    return {
        "tile_id": os.path.splitext(os.path.basename(path))[0],
        "lidar_pts": npts,
        "surf_roof": n_roof, "surf_facade": n_fac, "surf_ground": n_grd,
        "footprint_ratio": round(footprint_ratio, 4),
        "surf_per_ha": round(surf_per_ha, 2),
        "building_count": building_count,
        "mean_nn_spacing_m": round(spacing, 2) if not math.isnan(spacing) else "",
        "height_std_m": round(height_std, 2) if not math.isnan(height_std) else "",
        "non_flat_roof_ratio": round(non_flat, 3) if not math.isnan(non_flat) else "",
        "roof_per_building": round(roof_per_bldg, 2) if not math.isnan(roof_per_bldg) else "",
        "coverage_ok": coverage_ok,
    }


# --------------------------------- main ---------------------------------------
def main():
    global NON_FLAT_NZ_THRESH, MIN_ROOF_FOR_COVERAGE
    ap = argparse.ArgumentParser()
    ap.add_argument("--lidar-glob", default="data/raw/lidar/09LD*.las",
                    help="TODO(repo) (3): LAS files to scan")
    ap.add_argument("--citygml-glob",
                    default="data/raw/citygml/udx/bldg/*_bldg_6697_2_op.gml",
                    help="on-hand LOD2 building GMLs (loaded once)")
    ap.add_argument("--out", default="outputs/g1/tile_scan.csv")
    ap.add_argument("--non-flat-thresh", type=float, default=NON_FLAT_NZ_THRESH)
    ap.add_argument("--min-roof", type=int, default=MIN_ROOF_FOR_COVERAGE)
    args = ap.parse_args()

    NON_FLAT_NZ_THRESH = args.non_flat_thresh
    MIN_ROOF_FOR_COVERAGE = args.min_roof

    gml_paths = sorted(glob.glob(args.citygml_glob))
    if not gml_paths:
        raise SystemExit(f"no CityGML matched {args.citygml_glob!r}")
    print(f"loading {len(gml_paths)} CityGML grids once ...", flush=True)
    surfaces = load_citygml(gml_paths)          # TODO(repo) (1)
    rc = build_ring_cache(surfaces)
    print(f"  {len(rc.label)} rings cached "
          f"(roof {(rc.label==LABEL_ROOF).sum()}, "
          f"facade {(rc.label==LABEL_FACADE).sum()}, "
          f"ground {(rc.label==LABEL_GROUND).sum()})", flush=True)

    las_paths = sorted(glob.glob(args.lidar_glob))
    if not las_paths:
        raise SystemExit(f"no LAS matched {args.lidar_glob!r}")

    rows = []
    for p in las_paths:
        try:
            rows.append(scan_tile(p, rc))
        except Exception as e:  # keep scanning the rest
            print(f"  ! {os.path.basename(p)}: {e}", flush=True)

    # rank: covered tiles first, hardest (density*complexity) on top
    def hardness(r):
        nf = r["non_flat_roof_ratio"] or 0
        hs = r["height_std_m"] or 0
        return (r["coverage_ok"], r["footprint_ratio"] * (nf + 0.0) + 0.01 * hs)
    rows.sort(key=hardness, reverse=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = list(rows[0].keys())
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # stdout table (only covered tiles, for the pick)
    print("\n=== covered tiles (hardest first) ===")
    hdr = ("tile_id  pts(M)  R/F/G            fp_ratio surf/ha bldg spacing "
           "h_std nonflat r/bldg")
    print(hdr)
    for r in rows:
        if not r["coverage_ok"]:
            continue
        print(f'{r["tile_id"]:8} {r["lidar_pts"]/1e6:5.2f}  '
              f'{r["surf_roof"]}/{r["surf_facade"]}/{r["surf_ground"]:<8} '
              f'{r["footprint_ratio"]:>7} {r["surf_per_ha"]:>7} '
              f'{r["building_count"]:>4} {str(r["mean_nn_spacing_m"]):>6} '
              f'{str(r["height_std_m"]):>5} {str(r["non_flat_roof_ratio"]):>6} '
              f'{str(r["roof_per_building"]):>6}')
    print(f"\nwrote {args.out}  ({sum(r['coverage_ok'] for r in rows)} covered "
          f"/ {len(rows)} scanned)")
    print("pick: K=3..4 regular (low fp_ratio & nonflat) for train, "
          "1 hard (high both) for held-out, disjoint.")


if __name__ == "__main__":
    main()
