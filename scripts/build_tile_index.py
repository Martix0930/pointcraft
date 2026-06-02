"""Build a LAS ↔ mesh ↔ CityGML alignment table.

Reads every LAS in data/raw/lidar/ (real header bbox, EPSG:6677), intersects it
with the LOD2 mesh grids in data/raw/citygml/mesh_index.csv, and writes a tidy
correspondence table so the two datasets line up at a glance:

    data/raw/tile_alignment.csv   one row per LAS: bbox + which mesh(es) it covers
                                  + whether that mesh has LOD2 CityGML.

Usage:  python scripts/build_tile_index.py
"""
from __future__ import annotations

import csv
import glob
import os

import laspy

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LIDAR = os.path.join(REPO, "data", "raw", "lidar")
MESH_IDX = os.path.join(REPO, "data", "raw", "citygml", "mesh_index.csv")
OUT = os.path.join(REPO, "data", "raw", "tile_alignment.csv")


def load_meshes():
    meshes = []
    with open(MESH_IDX, newline="") as f:
        for r in csv.DictReader(f):
            meshes.append((
                r["mesh_code"], r["citygml_has_lod2"] == "1",
                float(r["x_min"]), float(r["y_min"]),
                float(r["x_max"]), float(r["y_max"]),
            ))
    return meshes


def main():
    meshes = load_meshes()
    rows = []
    for p in sorted(glob.glob(os.path.join(LIDAR, "*.las"))):
        code = os.path.splitext(os.path.basename(p))[0]
        with laspy.open(p) as f:
            h = f.header
            x0, y0 = h.mins[0], h.mins[1]
            x1, y1 = h.maxs[0], h.maxs[1]
        hits = [(mc, l2) for (mc, l2, mx0, my0, mx1, my1) in meshes
                if not (x1 <= mx0 or x0 >= mx1 or y1 <= my0 or y0 >= my1)]
        mesh_codes = ";".join(mc for mc, _ in hits)
        lod2 = int(any(l2 for _, l2 in hits))
        rows.append([code, round(x0), round(y0), round(x1), round(y1), mesh_codes, lod2])

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["las_code", "x_min", "y_min", "x_max", "y_max",
                    "mesh_codes", "in_lod2_citygml"])
        w.writerows(rows)

    # summary: which mesh each LAS group falls in
    print(f"{len(rows)} LAS tiles -> {OUT}")
    by_mesh = {}
    for r in rows:
        for mc in r[5].split(";"):
            if mc:
                by_mesh.setdefault(mc, []).append(r[0])
    print("\nLAS coverage per mesh grid (LOD2 grids marked *):")
    lod2set = {mc for (mc, l2, *_) in meshes if l2}
    for mc in sorted(by_mesh):
        star = "*" if mc in lod2set else " "
        codes = by_mesh[mc]
        print(f"  {star}{mc}: {len(codes)} LAS  ({codes[0]}..{codes[-1]})")
    n_lod2 = sum(1 for r in rows if r[6] == 1)
    print(f"\n{n_lod2}/{len(rows)} LAS tiles fall in an LOD2-CityGML mesh.")


if __name__ == "__main__":
    main()
