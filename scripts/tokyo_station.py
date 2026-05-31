"""Build a Minecraft schematic of the Tokyo Station area from LiDAR + PLATEAU LOD2.

Usage:
    python scripts/tokyo_station.py                  # run pipeline, save snapshot, write .schem
    python scripts/tokyo_station.py --view           # also launch viewer
    python scripts/tokyo_station.py --no-schem       # skip schem export (fast iteration)
    python scripts/tokyo_station.py --view --no-schem

Data paths default to the example layout. Edit constants below or pass --las / --lod2.
"""
from __future__ import annotations
import argparse
import os
import sys

# Allow running from anywhere by adding repo root to sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from pointcraft import Context, Pipeline, save_context, load_context, Viewer
from pointcraft.stages import (
    LoadLas, DropNoiseClass, PercentileZClip,
    Voxelize, MinSupportFilter, LocalHeightOutlier,
    FillSingleStepHoles, MorphologicalClose, RemoveSmallComponents,
    SampleLOD2Color, FuseLOD2Geometry, SampleLOD2Facade, MapBlocks, EmitSchem,
)


# --- Defaults (override via CLI flags) ---------------------------------------
# Tokyo Station spans several adjacent LAS tiles; merge them so the full
# structure is reconstructed (a single tile clips half the station).
# Single original tile (LOD2 will supply full building geometry, so we no
# longer need to merge tiles just to recover clipped buildings).
DEFAULT_LAS = [
    r"D:\Desktop\实习\三维GIS\09LD1874.las",
]
DEFAULT_LOD2_TILES = [
    r"D:\Desktop\实习\三维GIS\LOD2\53394611",
    r"D:\Desktop\实习\三维GIS\LOD2\53394621",
]
DEFAULT_OUT = os.path.join(REPO, "output")
DEFAULT_NAME = "tokyo_station"


def build_pipeline(las_path, lod2_tiles, out_dir, schem_name):
    return Pipeline([
        LoadLas(las_path),
        DropNoiseClass(),
        PercentileZClip(p_lo=0.05, p_hi=99.95),
        Voxelize(cell_size=1.0),
        MinSupportFilter(min_points=3),
        LocalHeightOutlier(window=5, threshold=2.0, mode="snap"),
        FillSingleStepHoles(iterations=2),
        MorphologicalClose(iterations=1),
        RemoveSmallComponents(min_size=6),
        SampleLOD2Color(tile_dirs=lod2_tiles),
        FuseLOD2Geometry(),
        SampleLOD2Facade(tile_dirs=lod2_tiles),
        MapBlocks(),
        EmitSchem(out_dir, schem_name),
    ])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--las", nargs="*", default=DEFAULT_LAS)
    p.add_argument("--lod2", nargs="*", default=DEFAULT_LOD2_TILES)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--name", default=DEFAULT_NAME)
    p.add_argument("--view", action="store_true", help="Launch viewer after pipeline")
    p.add_argument("--no-schem", action="store_true", help="Skip schem export")
    args = p.parse_args()

    pipe = build_pipeline(args.las, args.lod2, args.out, args.name)
    if args.no_schem:
        pipe = pipe.remove_by_type(EmitSchem)

    ctx = pipe.run()

    snapshot_path = os.path.join(args.out, "last_ctx.npz")
    os.makedirs(args.out, exist_ok=True)
    save_context(ctx, snapshot_path)

    print()
    print("=" * 60)
    print("STAGE TIMING")
    for h in ctx.history:
        print(f"  {h['stage']:30s}  {h['seconds']:.2f}s")
    print("=" * 60)

    if args.view:
        Viewer(load_context(snapshot_path)).show()


if __name__ == "__main__":
    main()
