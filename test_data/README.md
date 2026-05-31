# test_data/

Tiny, **committed** fixtures for unit tests and smoke runs. These are synthetic or
heavily reduced examples — small enough to live in git and to make expected
results checkable by hand.

## Policy (see also `CLAUDE.md` → "Test data policy")

- Only **tiny synthetic / heavily reduced** data here.
- **Never** commit real LiDAR / PLATEAU / CityGML / large OBJ / NPZ / checkpoints
  or generated training outputs — those stay outside git or in ignored local folders.
- Tests should prefer these fixtures over ad-hoc local paths.
- Every fixture subdir has a `README.md` stating: what it represents, the expected
  coordinate system, the expected voxel size / bounds, and which tests use it.

## Layout

```
test_data/
├─ m0_voxel_grid/      # points + expected voxel indices (voxel grid utility tests)
├─ m0_data_pairing/    # tiny aerial LiDAR + cube LOD2 + expected metadata
└─ smoke/              # placeholder for end-to-end smoke-run fixtures
```

## Conventions

- Coordinates are a **synthetic local frame** (not a real CRS), axes `x`(east),
  `y`(north), `z`(up), units = meters.
- Voxel index: `idx = floor((world_xyz - origin) / voxel_size)`; `origin` is the
  min corner of voxel `(0,0,0)`. (Matches `docs/02_DATA_CONTRACT.md`.)
