# M0 — Data Pairing — TASK SPEC

## Goal

Create **paired voxel training samples** from aerial LiDAR partial observation and
LOD2 (or mesh/CityGML) complete target. Output is the `.npz` format defined in
`docs/02_DATA_CONTRACT.md`, ready to be consumed by M2+ learning code.

This milestone is **data engineering only**. No learning.

## Context to read first

- `CLAUDE.md`
- `docs/00_PROJECT_BRIEF.md`, `docs/02_DATA_CONTRACT.md`
- Legacy pipeline in repo-root `pointcraft/` already has LiDAR loading,
  voxelization, and LOD2/LAS CRS alignment — **reuse, do not rewrite**.

## Deliverables

1. **Shared voxel grid utility** (`src/pointcraft/voxelization/`)
   - Define a grid from world bounds + `voxel_size` (origin, grid_shape).
   - `world_xyz -> voxel index` and back (centers/corners), per `02_DATA_CONTRACT`.
   - The same grid object is used for BOTH partial and target — single source.

2. **LiDAR → partial occupancy** (`src/pointcraft/data/`)
   - Load LiDAR points (reuse legacy loader), map to voxel indices on the shared
     grid, dedupe, produce `coords_partial` + `feats_partial`.
   - Implement the documented `feature_layout` (start minimal: e.g. height,
     point-count or intensity; expand later).

3. **LOD2 / mesh → target occupancy + semantics** (`src/pointcraft/data/`)
   - Voxelize LOD2 building solids (and terrain/veg as available) on the SAME grid
     → `coords_target`, `occ_target`, `sem_target`.
   - Assign placeholder semantic ids per `02_DATA_CONTRACT` label table.

4. **`.npz` sample writer** (`src/pointcraft/data/`)
   - Write all fields + `metadata` exactly per the data contract.
   - Optionally compute `observed_mask` / `unobserved_mask`.

5. **Sanity visualization / debug export**
   - A quick way to eyeball one sample (matplotlib slice, or reuse the legacy
     viewer, or a tiny Minecraft preview **for debug only**).

6. **Tests for voxel alignment** (`tests/`)
   - Partial and target share identical grid params.
   - Round-trip world↔index correctness.
   - Alignment regression: a building's target footprint overlaps its observed roof.

## Scope exclusions (do NOT do in M0)

- ❌ No neural network.
- ❌ No spconv / Minkowski / torch.
- ❌ No semantic *learning* (only deterministic label assignment from sources).
- ❌ No Minecraft export beyond an optional debug preview.
- ❌ No refactoring of the legacy `pointcraft/` package beyond what's needed to reuse it.

## Suggested layout

```
src/pointcraft/
├─ voxelization/   # grid definition + index transforms
├─ data/           # lidar->partial, lod2->target, npz writer
└─ utils/          # io / logging helpers
tests/             # alignment + round-trip tests
scripts/           # a small run_m0.py to build one sample end-to-end
```

## Definition of done

See `ACCEPTANCE.md`. In short: one small tile → one valid `.npz` on a shared grid,
loadable independently, with passing alignment tests and an updated session log.
