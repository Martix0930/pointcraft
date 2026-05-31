# m0_voxel_grid fixture

Minimal points with **hand-computed** expected voxel indices, for testing the
shared voxel-grid utility (`src/pointcraft/voxelization/`, M0-1).

## What it represents

A handful of XYZ points placed near the origin and on voxel boundaries, so the
`floor` index convention (and boundary behavior) is unambiguous to verify.

## Grid (see `tiny_bounds.json`)

- coordinate system: synthetic local frame, meters, `x`=east `y`=north `z`=up
- `origin`: `[0, 0, 0]` (min corner of voxel `(0,0,0)`)
- `voxel_size`: `1.0`
- `bounds`: `[0, 0, 0, 4, 4, 4]`
- `grid_shape`: `[4, 4, 4]` = `ceil((bounds_max - origin) / voxel_size)`
- valid indices per axis: `0..3`

## Index convention

`idx = floor((xyz - origin) / voxel_size)`. A coordinate exactly on an integer
boundary (e.g. `x = 1.0`) maps **up** to the higher voxel (`floor(1.0) = 1`).

## Files

- `tiny_points.csv` — columns `id,x,y,z` (one point per row).
- `expected_indices.json` — `id -> [i, j, k]` expected voxel index per point.
- `tiny_bounds.json` — grid definition (`origin`, `voxel_size`, `bounds`, `grid_shape`).

## Used by

- voxel-grid round-trip / index tests in `tests/` (M0-1). Not yet implemented.
