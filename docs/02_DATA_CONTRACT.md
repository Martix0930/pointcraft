# 02 — Data Contract (M0 training sample format)

This defines the **planned** on-disk format for paired voxel training samples
produced by M0. One `.npz` file = one tile sample. This contract is the interface
between M0 (data pairing) and M2+ (learning).

> Status: **finalized for `dataset_version = v0.2`** (M0 decisions D1–D6 locked in
> `docs/06_DECISIONS.md`). The target source is **CityGML** (D5), not OBJ; building
> semantics come from CityGML surface types. The on-disk field set / feature layout
> are unchanged from v0.1, but the **observed/unobserved mask definition changed**
> (D6, class-aware) so the version is bumped to `v0.2`. Further schema or mask-rule
> changes require bumping `dataset_version`.

## `.npz` fields

| Field | Dtype | Shape | Meaning |
|-------|-------|-------|---------|
| `coords_partial` | int32 | `[N, 3]` | Voxel indices `(i, j, k)` that are **observed** (occupied by the partial LiDAR input). |
| `feats_partial`  | float32 | `[N, C]` | Per-observed-voxel features (see feature layout). `C` fixed per dataset version. |
| `coords_target`  | int32 | `[M, 3]` | Voxel indices for the **complete** target. Under the **shell** representation (D2) these are the CityGML LOD2 *surface* voxels — no interior fill. |
| `occ_target`     | uint8 | `[M]` | Occupancy label per `coords_target` entry. Every stored target voxel is occupied, so this is `1` for all `M` entries in v0.1 (free voxels are not stored). |
| `sem_target`     | int64 | `[M]` | Semantic class id per `coords_target` entry (see label table). `255` (`ignore_index`) for unknown. |
| `observed_mask`  | uint8 | `[M]` | **Required** (D4). `1` if this target voxel was directly observed by the partial input, under the **class-aware rule (D6, v0.2)** below. |
| `unobserved_mask`| uint8 | `[M]` | **Required** (D4). `1` if this target voxel was **never** observed (the completion region). Equals `occupied_target ∧ ¬observed`, i.e. `1 - observed_mask` over the occupied target. |
| `metadata`       | (see below) | — | Saved as a 0-d object array or sidecar JSON. |

Notes:
- Sparse storage (coordinate lists) is preferred over dense grids for memory.
- `feats_partial` column count `C` and column order are fixed by the dataset
  version string in `metadata` and documented in the feature layout below.
- `observed_mask` / `unobserved_mask` enable the M4 unobserved-region metrics.
  Per D4 they are **required** and computed in M0 while partial and target share one
  grid in a single pass — not recomputed downstream (a later intersection corrupts
  the headline metric on boundary near-misses).

**Observed rule (D6, `v0.2`, class-aware)** — `compute_masks` marks a target voxel
observed by its semantic class (`sem_target`):
- **Horizontal** (roof `3` / ground `1`): observed if a partial voxel exists at the
  same `(i,j)` within `|Δk| ≤ z_tol` (default `z_tol = 1`). Corrects the sub-voxel
  z-quantization that otherwise flags seen roofs "unobserved".
- **Vertical** (facade `4`): observed only on an **exact** partial hit in a *genuine
  mid-wall* cell — `≥ wall_margin` voxels from the column's lowest and highest target
  voxel (default `wall_margin = 2`) — excluding ground/roof points that merely clip
  the base/top wall voxel.
- Legacy exact set-membership (v0.1) remains available via `sem_target=None`.
- Reference numbers (tile 09LD1874): observed roof 70 % / facade ~35 % / ground 18 %;
  total unobserved 61 %. The facade figure reflects this LiDAR's real (sparse)
  oblique facade coverage, not an artifact (D6 research note).

### Feature layout (`dataset_version = v0.2`, unchanged from v0.1)

`feature_layout = ["height", "point_count"]` (`C = 2`):

| col | name | meaning | aggregation |
|-----|------|---------|-------------|
| 0 | `height` | representative world-z of the LiDAR points merged into the voxel | mean of point z |
| 1 | `point_count` | number of LiDAR points merged into the voxel | count |

Minimal and expandable: intensity, return number, and normals are deferred to a
later `dataset_version`, not added in v0.1.

## `metadata` fields

| Key | Type | Meaning |
|-----|------|---------|
| `tile_id` | str | Unique tile identifier. |
| `voxel_size` | float | Edge length of a voxel in world units (meters). **Default `1.0` (D1).** |
| `origin` | float[3] | World coordinate of voxel index `(0,0,0)` corner. |
| `bounds` | float[6] | `[xmin, ymin, zmin, xmax, ymax, zmax]` world bounds of the grid. |
| `grid_shape` | int[3] | `(I, J, K)` voxel-grid dimensions. |
| `crs` | str | Coordinate reference system of the grid and all stored coords. **`EPSG:6677`** (the LiDAR native CRS). The CityGML target is reprojected 6697→6677 before voxelization (D5; see Alignment rule 1). |
| `source_files` | list[str] | Paths/ids of source files: the LiDAR LAS sheet **and the CityGML grid file(s)** (D5). OBJ paths appear here only when the OBJ fallback is used instead of CityGML. |
| `dataset_version` | str | Schema / mask-rule version (current `v0.2`). |
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

## Semantic label table (`dataset_version = v0.2`, unchanged from v0.1)

> Finalized for v0.1 under the **shell** target (D2): a building is its surface
> voxels (`roof` + `facade`), so the building-*solid* class `2` is **unused**.
> Semantics are read **directly from CityGML surface types** (D5), not inferred
> from face geometry.

| id | name | source (v0.1 = CityGML surface type, D5) | v0.1 status |
|----|------|------------------------------------------|-------------|
| 0 | free / empty | (only if storing free voxels; v0.1 stores occupied only) | unused |
| 1 | ground | CityGML `bldg:GroundSurface` | active (new under D5) |
| 2 | building (solid interior) | LOD2 building solid | **unused** (shell target, D2) |
| 3 | roof | CityGML `bldg:RoofSurface` | active |
| 4 | facade / wall | CityGML `bldg:WallSurface` | active |
| 5 | vegetation | LiDAR class (low/high veg) | active if available |
| 6 | road | OSM / LiDAR | optional |
| 7 | water | optional | optional |
| 255 | ignore / unknown | unlabeled | `ignore_index` |

CityGML surface-type → label mapping (D5):
`bldg:RoofSurface → roof (3)`, `bldg:WallSurface → facade (4)`,
`bldg:GroundSurface → ground (1)`. Surfaces with no recognised type → `ignore (255)`.

## Alignment rules (LiDAR ↔ LOD2)

1. **CRS reconciliation (D5).** The grid and all stored coords are **EPSG:6677**
   (LiDAR native). CityGML is delivered in **EPSG:6697 (lat/lon)** and MUST be
   reprojected **6697→6677** before pairing; log the reprojection. The transform is
   horizontal-only — `z` (absolute elevation, D3) passes through unchanged and is
   re-verified on a real building (Phase C2 gate). (For the OBJ fallback, OBJ is
   already 6677, so no reprojection is needed.)
2. **Shared grid.** `coords_partial` and `coords_target` MUST use the **same**
   `origin`, `voxel_size`, and `grid_shape`. The grid is defined once per tile and
   reused for both input and target.
3. **Ground reference.** `z` is **absolute elevation** (D3), with a single shared
   origin for partial and target. No DTM / height-above-ground model in v0.1;
   adopting height-above-ground later means recomputing `z` and bumping
   `dataset_version` (document the offset in `metadata`).
4. **Tile extent.** Tile bounds derive from the LiDAR coverage; the (reprojected)
   CityGML target is clipped to the same bounds.
5. **Sanity check.** A target building's footprint must overlap the observed roof
   voxels of the same building (alignment regression test in M0).
