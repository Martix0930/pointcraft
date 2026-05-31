# 02 — Data Contract (M0 training sample format)

This defines the **planned** on-disk format for paired voxel training samples
produced by M0. One `.npz` file = one tile sample. This contract is the interface
between M0 (data pairing) and M2+ (learning).

> Status: **proposed**. May be revised during M0; log changes in `docs/06_DECISIONS.md`.

## `.npz` fields

| Field | Dtype | Shape | Meaning |
|-------|-------|-------|---------|
| `coords_partial` | int32 | `[N, 3]` | Voxel indices `(i, j, k)` that are **observed** (occupied by the partial LiDAR input). |
| `feats_partial`  | float32 | `[N, C]` | Per-observed-voxel features (see feature layout). `C` fixed per dataset version. |
| `coords_target`  | int32 | `[M, 3]` | Voxel indices for the **complete** target (the supervision set; typically the union of all target-occupied voxels). |
| `occ_target`     | uint8 | `[M]` | Occupancy label per `coords_target` entry (`1` occupied, `0` free if explicitly stored). |
| `sem_target`     | int64 | `[M]` | Semantic class id per `coords_target` entry (see label table). `-1` / `ignore_index` for unknown. |
| `observed_mask`  | uint8 | `[M]` | Optional. `1` if this target voxel was directly observed by the partial input. |
| `unobserved_mask`| uint8 | `[M]` | Optional. `1` if this target voxel was **never** observed (the completion region). Usually `1 - observed_mask` within occupied target. |
| `metadata`       | (see below) | — | Saved as a 0-d object array or sidecar JSON. |

Notes:
- Sparse storage (coordinate lists) is preferred over dense grids for memory.
- `feats_partial` column count `C` and column order are fixed by the dataset
  version string in `metadata` and documented in the feature layout below.
- `observed_mask` / `unobserved_mask` enable the M4 unobserved-region metrics; if
  not stored, they can be recomputed from `coords_partial` ∩ `coords_target`.

## `metadata` fields

| Key | Type | Meaning |
|-----|------|---------|
| `tile_id` | str | Unique tile identifier. |
| `voxel_size` | float | Edge length of a voxel in world units (meters). |
| `origin` | float[3] | World coordinate of voxel index `(0,0,0)` corner. |
| `bounds` | float[6] | `[xmin, ymin, zmin, xmax, ymax, zmax]` world bounds of the grid. |
| `grid_shape` | int[3] | `(I, J, K)` voxel-grid dimensions. |
| `crs` | str | Coordinate reference system (e.g. `EPSG:6677`). |
| `source_files` | list[str] | Paths/ids of source LiDAR + LOD2/CityGML files. |
| `dataset_version` | str | Schema/feature-layout version (e.g. `v0.1`). |
| `feature_layout` | list[str] | Names of the `C` feature columns, in order. |

## Coordinate convention

- **World axes**: `x` = easting, `y` = northing, `z` = up (height). Right-handed.
- A point's voxel index: `idx = floor((world_xyz - origin) / voxel_size)`.
- `origin` is the corner (minimum) of voxel `(0,0,0)`. Voxel center =
  `origin + (idx + 0.5) * voxel_size`.
- `grid_shape = ceil((bounds_max - origin) / voxel_size)`.
- Index order is `(i, j, k)` ↔ `(x, y, z)`. Document any transpose at the
  Minecraft-export boundary (MC uses its own axis convention) — keep training data
  in world convention; convert only at export.

## Voxel index convention

- Indices are non-negative integers within `[0, grid_shape)`.
- Out-of-range points are dropped during voxelization (logged as a count).
- Duplicate points mapping to the same voxel are merged (occupancy = OR; features
  aggregated, e.g. mean/min/max — specify in feature layout).

## Semantic label table (placeholder)

> Provisional. Finalize during M0 once LOD2 semantic granularity is confirmed.

| id | name | source |
|----|------|--------|
| 0 | free / empty | (only if storing free voxels) |
| 1 | ground | LiDAR class + LOD terrain |
| 2 | building | LOD2 building solid |
| 3 | roof | LOD2 roof surface |
| 4 | facade / wall | LOD2 wall surface |
| 5 | vegetation | LiDAR class (low/high veg) |
| 6 | road | OSM / LiDAR (optional) |
| 7 | water | optional |
| 255 | ignore / unknown | unlabeled |

## Alignment rules (LiDAR ↔ LOD2)

1. **Same CRS, no transform.** Verify LiDAR and LOD2 share the CRS (e.g. EPSG:6677)
   before pairing; if not, reproject LOD2 to the LiDAR CRS and log it.
2. **Shared grid.** `coords_partial` and `coords_target` MUST use the **same**
   `origin`, `voxel_size`, and `grid_shape`. The grid is defined once per tile and
   reused for both input and target.
3. **Ground reference.** Define `z` consistently (absolute elevation vs.
   height-above-ground). If using height-above-ground, store the ground model in
   `metadata` or document the offset.
4. **Tile extent.** Tile bounds derive from the LiDAR coverage; LOD2 is clipped to
   the same bounds.
5. **Sanity check.** A target building's footprint must overlap the observed roof
   voxels of the same building (alignment regression test in M0).
