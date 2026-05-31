# m0_data_pairing fixture

A tiny synthetic "tile": one cube building plus a flat ground, with an aerial-style
LiDAR observation. For smoke-testing the M0 pairing pipeline (LiDAR → partial
occupancy, LOD2 → target occupancy) on the **same grid**.

## What it represents

- **Building**: an axis-aligned cube occupying `x∈[1,3]`, `y∈[1,3]`, `z∈[0,3]`
  (a ~2×2 footprint, 3 m tall), given as a closed mesh in `tiny_lod2_cube.obj`
  (this is the *complete* target geometry).
- **Aerial LiDAR** (`tiny_lidar_points.csv`): observes only what an airborne
  sensor would — the **roof** (points near `z≈3` over the footprint) and the
  **surrounding ground** (`z≈0`). The cube's **facades are NOT observed** — that
  unobserved volume is exactly what later milestones must complete.

## Grid / coordinate system (see `expected_metadata.json`)

- synthetic local frame, meters, `x`=east `y`=north `z`=up
- `origin`: `[0, 0, 0]`, `voxel_size`: `1.0`
- `bounds`: `[0, 0, 0, 4, 4, 4]`, `grid_shape`: `[4, 4, 4]`

## Files

- `tiny_lidar_points.csv` — columns `x,y,z,intensity,classification`
  (`classification`: `2`=ground, `6`=building/roof). Aerial partial observation.
- `tiny_lod2_cube.obj` — the complete cube building mesh (target geometry).
- `expected_metadata.json` — the `.npz` `metadata` block this tile should produce
  (`tile_id`, `voxel_size`, `origin`, `bounds`, `grid_shape`, `crs`,
  `source_files`, `dataset_version`).

## Notes / scope

This fixture intentionally specifies **inputs + high-level metadata only**, not the
exact per-voxel occupancy/semantics — voxelization and boundary handling are M0's
job to implement and test. The fixture lets a test assert: same grid for partial &
target, metadata correctness, and that the roof is observed while facades are not.

## Used by

- M0 data-pairing smoke tests in `tests/` (not yet implemented).
