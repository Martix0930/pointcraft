"""Build a Minecraft schematic of the Tokyo Station area from LiDAR + PLATEAU LOD2.

Usage:
    python scripts/tokyo_station.py                  # run pipeline, save snapshot, write .schem
    python scripts/tokyo_station.py --view           # also launch viewer
    python scripts/tokyo_station.py --no-schem       # skip schem export (fast iteration)
    python scripts/tokyo_station.py --view --no-schem
    python scripts/tokyo_station.py --config configs/tokyo_station.yaml

Data paths / params come from a YAML config (default configs/tokyo_station.yaml).
Override individual values with --las / --lod2 / --out / --name.
"""
from __future__ import annotations
import argparse
import os
import sys

# Allow running from anywhere by adding the package source dir to sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from pointcraft.pipeline import Context, Pipeline
from pointcraft.utils.config import load_config, resolve_path
from pointcraft.utils.viewer import Viewer, save_context, load_context
from pointcraft.baseline.stages import (
    LoadLas, DropNoiseClass, PercentileZClip,
    Voxelize, MinSupportFilter, LocalHeightOutlier,
    FillSingleStepHoles, MorphologicalClose, RemoveSmallComponents,
    SampleLOD2Color, FuseLOD2Geometry, SampleLOD2Facade, MapBlocks, EmitSchem,
)


# Data paths / params now live in a YAML config (default below). Edit the config
# or override individual values with CLI flags. See configs/tokyo_station.yaml.
DEFAULT_CONFIG = os.path.join(REPO, "configs", "tokyo_station.yaml")


def build_pipeline(las_path, lod2_tiles, out_dir, schem_name, cell_size=1.0):
    return Pipeline([
        LoadLas(las_path),
        DropNoiseClass(),
        PercentileZClip(p_lo=0.05, p_hi=99.95),
        Voxelize(cell_size=cell_size),
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
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help="YAML data config (default: configs/tokyo_station.yaml)")
    p.add_argument("--las", nargs="*", default=None, help="Override config `las`")
    p.add_argument("--lod2", nargs="*", default=None, help="Override config `lod2_tiles`")
    p.add_argument("--out", default=None, help="Override config `output_dir`")
    p.add_argument("--name", default=None, help="Override config `name`")
    p.add_argument("--view", action="store_true", help="Launch viewer after pipeline")
    p.add_argument("--no-schem", action="store_true", help="Skip schem export")
    args = p.parse_args()

    cfg = load_config(args.config)
    # CLI flags override config values; paths resolve relative to the repo root.
    las = args.las if args.las is not None else cfg.get("las", [])
    lod2 = args.lod2 if args.lod2 is not None else cfg.get("lod2_tiles", [])
    name = args.name or cfg.get("name", "pointcraft")
    out_dir = resolve_path(args.out or cfg.get("output_dir", "output"), REPO)
    cell_size = float(cfg.get("cell_size", 1.0))
    print(f"[tokyo_station] config={args.config}  crs={cfg.get('crs', '?')}  "
          f"cell_size={cell_size}  las={len(las)} file(s)  lod2={len(lod2)} tile(s)")

    pipe = build_pipeline(las, lod2, out_dir, name, cell_size=cell_size)
    if args.no_schem:
        pipe = pipe.remove_by_type(EmitSchem)

    ctx = pipe.run()

    snapshot_path = os.path.join(out_dir, "last_ctx.npz")
    os.makedirs(out_dir, exist_ok=True)
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
