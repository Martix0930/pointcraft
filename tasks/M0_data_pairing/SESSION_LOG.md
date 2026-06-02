# M0 — Data Pairing — SESSION LOG

Append a dated entry at the end of every session. Keep the "Current status" and
"Next recommended prompt" blocks at the top up to date.

---

## Current status: **IN PROGRESS — re-targeting M0 to CityGML (EXECUTION_PLAN v2,
D5).** The OBJ-based pipeline (v0.1) is complete and runs end-to-end
(`scripts/run_m0.py`, 28 tests), but its alignment audit (entry below) showed the
masks are inflated by OBJ-geometry artifacts and the OBJ target lacks a ground
class. Per EXECUTION_PLAN v2 the M0 **target source is now CityGML** (D5):
semantics are read from surface types, and CityGML is reprojected 6697→6677.
Proof tile = **09LD1874** (Tokyo-Station core; grids 53394610/611/620/621; ~2061
CityGML surfaces; LiDAR 6.39 M pts; the tile `tokyo_station.yaml` already targets).
The originally-planned **09LD1848 was dropped at the Phase B gate** — its
`in_lod2_citygml=1` flag is only a bbox overlap; **0** CityGML buildings actually
fall in its footprint (gotcha logged in `docs/07_GOTCHAS.md`). Real data staged
under `data/raw/` (git-ignored). **Phases A, B done; Phase C1 done** (CityGML parser
`src/pointcraft/data/citygml.py`: typed LOD2 surfaces + 6697→6677 reprojection,
`pyproj` dep added; validated on real grids). GML confirmed: EPSG:6697, 3D, posList
axis order **lat,lon,z**. **Next = Phase C2**: re-verify CityGML↔LiDAR alignment on
09LD1874 (gate) before voxelization. OBJ pipeline stays as fallback/comparison.

M0 delivers `pointcraft.data`: `voxelize_partial` (LiDAR→partial), `voxelize_target`
(LOD2 shell→target+semantics), `compute_masks` (D4), `build_metadata` +
`write_sample_npz`; on the shared `pointcraft.voxelization.VoxelGrid`. Driven
end-to-end by `scripts/run_m0.py` (config / explicit / `--fixture`, `--viz`).
Contract finalized for `dataset_version=v0.1`. 28 tests passing.

## Known limitations (M0 v0.1)

- **Building bottom labelled roof.** Semantics come from face normal |n_z|; a
  building's near-horizontal bottom face is labelled roof(3). Rare/benign in
  PLATEAU LOD2; revisit if it pollutes metrics.
- **Sloped-roof threshold.** `roof_nz=0.7` splits roof/facade; steep roofs near
  the cutoff may flip. Tune per-source if needed.
- **No terrain/vegetation target.** Target is the LOD2 building shell only
  (classes 3/4). Ground(1)/veg(5) need a terrain source (DEM) — future version.
- **No per-face XY pre-cull.** `voxelize_target` samples all faces before
  clipping → ~18 s for the real tile. Fine for one tile; add a bbox reject if
  batching many tiles (M2+).
- **z aggregation.** `height` is mean z (robust); raw LiDAR has rare high
  outliers (Phase B). No DTM/height-above-ground in v0.1 (absolute z, D3).
- **Grid from points.** Callers building a grid from raw point min/max must use
  an exclusive upper bound (run_m0 does) or max-boundary points are dropped.

## Next recommended prompt for Claude Code

> Read `CLAUDE.md`, `docs/07_GOTCHAS.md`, `docs/02_DATA_CONTRACT.md`,
> `docs/06_DECISIONS.md` (incl. D5) and `tasks/M0_data_pairing/EXECUTION_PLAN.md`.
> Phases A, B, C1 are DONE. Proof tile is **09LD1874** (09LD1848 was dropped at the
> gate — see GOTCHAS). CityGML parser `pointcraft.data.parse_citygml` returns typed
> 6677 surfaces (roof/facade/ground), validated on real grids. Execute **Phase C2**
> only: re-verify CityGML↔LiDAR alignment on 09LD1874 — load the LAS, parse the
> tile's CityGML grids (610/611/620/621), clip surfaces to the LAS footprint, and
> render/compare EW–NS cross-sections (reuse `scripts/render_alignment.py` idiom)
> to confirm LiDAR roof points sit on CityGML roof surfaces and that ground reads as
> GroundSurface. This is a **gate** — if misaligned, fix reprojection/datum before
> C5 voxelization. Keep M0 scope (no NN, no real-data commits). Update this log.
>
> (Deferred: M1 baseline, once the CityGML-based `.npz` re-verifies alignment.)

---

## Session entries

### 2026-06-01 — repository cleanup (pre-M0)

**Moved** (former repo-root `pointcraft/` → single `src/pointcraft/` package):
- `context.py` → `src/pointcraft/pipeline.py`
- `lod2.py` → `src/pointcraft/data/lod2.py`
- `palette.py` → `src/pointcraft/mc_export/palette.py`
- `viewer.py` → `src/pointcraft/utils/viewer.py`
- `stages.py` → `src/pointcraft/baseline/stages.py` (kept cohesive = M1 baseline)

**Removed:**
- root `pointcraft/__init__.py` and the entire repo-root `pointcraft/` directory
  (it shadowed `src/pointcraft/` on import).

**Naming cleanup result:**
- code/package/import/path = lowercase `pointcraft`; human-facing prose = `PointCraft`.
- Added `docs/07_GOTCHAS.md`; logged merge + naming decisions in `docs/06_DECISIONS.md`.

**Import smoke test:** `pip install -e .` then
`python -c "import pointcraft; print(pointcraft.__file__)"` →
`...\src\pointcraft\__init__.py` ✓. All submodules import OK
(`pointcraft`, `.pipeline`, `.data.lod2`, `.mc_export.palette`, `.utils.viewer`,
`.baseline.stages`).

**Known issues:**
- `pytest` not installed (in `dev` optional extras); **no functional tests exist
  yet** — only the import smoke test was run. M0 should add the first real tests.
- Editable install required for bare `import pointcraft` from arbitrary CWDs;
  legacy scripts also self-add `REPO/src` to `sys.path`.

**Did NOT** implement any M0 functionality (per task scope).

### 2026-06-01 — added root-level `test_data/` fixtures (pre-M0)

- Created `test_data/` with tiny synthetic fixtures:
  - `m0_voxel_grid/` — `tiny_points.csv`, `tiny_bounds.json`, `expected_indices.json`
    (6 points, hand-computed voxel indices; origin `[0,0,0]`, voxel_size `1.0`, grid `4×4×4`).
  - `m0_data_pairing/` — `tiny_lidar_points.csv` (aerial: roof + ground, facades
    unobserved), `tiny_lod2_cube.obj` (cube building `[1,3]×[1,3]×[0,3]`),
    `expected_metadata.json`.
  - `smoke/` — placeholder README.
- Updated `.gitignore` to re-include only tiny `test_data/` fixtures (`.obj` via
  `!test_data/**/*.obj`); generated `.npz`/`.npy` remain ignored.
- Added "Test data policy" to `CLAUDE.md`; noted `test_data/` in `README.md` and
  `tests/README.md`; added test-data gotcha in `docs/07_GOTCHAS.md`.
- **M0 implementation has NOT started** — fixtures only.

### 2026-06-01 — M0-1 shared voxel grid utility

- Added `src/pointcraft/voxelization/grid.py` → `VoxelGrid` (frozen dataclass):
  `from_bounds(bounds, voxel_size)`, `world_to_index` (floor), `index_to_center`,
  `index_to_corner`, `in_bounds`, `num_voxels`. Exported from
  `pointcraft.voxelization`. Pure numpy; no I/O, no learning.
- Conventions match `docs/02_DATA_CONTRACT.md` (origin = min corner of voxel
  (0,0,0); boundary coords floor up; index order (i,j,k)↔(x,y,z)).
- Added `tests/test_voxel_grid.py` (7 tests, **all passing**) driven by
  `test_data/m0_voxel_grid/` fixtures: shape/origin match, world→index ==
  expected, boundary-maps-up, center round-trip, in-bounds mask, corner offset,
  bad-voxel-size raises.
- Also (separate commit) externalized baseline data paths to
  `configs/tokyo_station.yaml` + `pointcraft.utils.config`.
- Installed `pytest` (dev) to run tests. No neural network.

### 2026-06-01 — M0 Phase A: lock decisions + finalize data contract (docs only)

- Logged the four M0 decisions in `docs/06_DECISIONS.md`:
  - **D1** voxel edge length = 1.0 m.
  - **D2** building target geometry = **shell** (surface voxels only; class `2`
    building-solid is unused).
  - **D3** vertical reference = **absolute elevation**, shared origin, no DTM.
  - **D4** `observed_mask` / `unobserved_mask` **promoted to required** M0 fields.
- Finalized `docs/02_DATA_CONTRACT.md` for `dataset_version = v0.1`: status
  flipped from "proposed" to finalized; masks moved Optional → required with exact
  derivations; `occ_target` clarified (all-1, occupied-only); shell semantics for
  `coords_target`; `voxel_size` default 1.0; z = absolute elevation; semantic label
  table finalized (class 2 unused under shell); pinned
  `feature_layout = ["height", "point_count"]` (C=2).
- **No implementation code changed** in this phase (per Phase A scope).
- Next: **Phase B** — inventory legacy `pointcraft` pkg, inspect real data, and
  verify vertical (z) alignment on one building before writing voxelization code.

### 2026-06-01 — M0 Phase B: inventory, data inspection, z-alignment GATE (passed)

**B1 — legacy inventory** (read-only; no refactor):
- LiDAR loader: `baseline/stages.py::LoadLas` — `laspy.read`, returns `x,y,z`,
  `classification`, optional `red/green/blue`. **Reuse the laspy read idiom**;
  M0's loader is a thin new function (no merge/cleanup pipeline needed for one tile).
- Voxelization: `baseline/stages.py::Voxelize` is **2.5D** — a single top-Z height
  map `[H,W]` with `origin=(x.min,y.min)` and heights *relative to ground_z*. **Not
  reusable as-is for M0** (we need true 3D, absolute-z, shared `VoxelGrid`). Borrow
  only the flat-index aggregation trick (`np.maximum.at` / `np.add.at`).
- LOD2 parsing: `data/lod2.py::parse_obj` (verts/uvs/faces/mtllib) — **reuse as-is**.
  `LOD2Rasterizer.rasterize` is 2.5D top-down (not for 3D shell). But
  `LOD2Rasterizer.colored_point_samples` **uniformly samples the mesh surface into
  3D world points** — exactly what a shell voxelizer needs: sample surface → map to
  `VoxelGrid`. **Reuse/wrap this for C3 target shell voxelization.**
- Net: reuse `parse_obj` + surface-sampling + laspy idiom; **write new** 3D
  partial/target voxelizers on `VoxelGrid` (M0-2/M0-3).

**B2 — real data** (`configs/tokyo_station.yaml`):
- LAS `09LD1874.las`: **1,752,298 pts**, CRS **EPSG:6677** (header-confirmed),
  units meters. Bounds X[-6239.1,-6000.0] Y[-35400.0,-35168.8] (~239×231 m tile),
  Z[-14.95, 48.29].
- LAS `classification` = `{1: 1.03M (unclassified, incl. roofs), 2: 676k (ground),
  3: 43k (low veg)}`. **No ASPRS class 6 (building)** — confirms building semantics
  must come from LOD2, not LAS (matches C3 design + D2). v0.1 `feature_layout`
  (height, point_count) needs no class, so unaffected.
- LOD2 tiles `53394611` (77.5k v / 152k f) + `53394621` (44.4k v / 88.5k f), CRS
  EPSG:6677 (filename `_6677`). LOD2 extent is **much larger** than the LAS tile
  (X[-6464,-5219] Y[-36148,-34279], Z up to 213 m). → tile extent must be driven by
  LAS coverage; **LOD2 clipped to LAS bounds** (contract alignment rule 4).

**B3 — z-alignment GATE → PASSED ✓** (LAS roof pts vs LOD2 surface, 2 m cells over
the overlap region; 450 cells with both):
- `LAS_topZ − LOD2_topZ`: median **+0.32 m**, p10 −1.20, p90 +3.50 (positive tail =
  trees/objects LOD2 doesn't model). No systematic datum offset of meters/tens of m.
- Representative tall building: cell LOD2 roof = 39.54 m; LAS **p90 = 39.48 m**
  (Δ ≈ 0.06 m). Neighbouring tall cells diff ∈ [−1.6, 0] m around zero.
- **Conclusion:** LAS and LOD2 share the vertical datum (absolute elevation, D3
  holds); **no offset correction needed**. Practical note for M0-2: aggregate
  partial occupancy with a robust statistic — raw `max` is pulled up by outliers
  (one cell showed LAS max 48 m vs p90 39 m); `feature_layout.height` is defined as
  **mean** z (contract), which avoids this.
- Verified with a throwaway script (`parse_obj` + laspy); not committed.

**Phase B exit criterion met.** No implementation code added; no legacy refactor.
Ready for Phase C (M0-2: LiDAR → partial occupancy on the shared `VoxelGrid`).

### 2026-06-01 — M0-2: LiDAR → partial occupancy

- Added `src/pointcraft/data/partial.py`:
  - `voxelize_partial(points_xyz, grid) -> (coords_partial int32[N,3],
    feats_partial float32[N,2])`. Maps points via `VoxelGrid.world_to_index`,
    drops out-of-range (logged), merges duplicates with `np.unique(axis=0)` +
    `np.add.at` segmented aggregation.
  - `feature_layout v0.1 = ["height", "point_count"]`; `height` = **mean** z per
    voxel (robust to the LiDAR outliers Phase B flagged — raw max over-counts);
    `point_count` = points merged.
  - `load_las_xyz(path)` thin laspy wrapper (lazy import; tests don't need laspy).
  - Exported from `pointcraft.data` (`voxelize_partial`, `load_las_xyz`,
    `FEATURE_LAYOUT_V01`).
- Fixed `test_data/m0_data_pairing/expected_metadata.json`: `feature_layout` was
  the stale pre-M0 `["intensity","classification"]` → now matches the v0.1
  contract `["height","point_count"]`.
- Added `tests/test_partial_occupancy.py` (7 passing, fixture-driven): contract
  dtypes/shapes, unique+in-bounds coords, 13 pts→12 voxels & point_count
  conservation, roof(k=3)/ground(k=0) observed while facades(k∈{1,2}) UNobserved,
  merged-voxel height = mean z, empty input, out-of-range dropped.
- Real-LAS smoke (`09LD1874.las`): 1,752,298 pts → 96,264 occupied voxels on a
  240×232×64 grid; all in-bounds, unique, point_count sum == input, height ∈
  [-14.88, 48.18]. No neural network.
- Next: **M0-3** LOD2 shell → `coords_target`/`occ_target`/`sem_target` on the
  SAME grid — reuse `LOD2Rasterizer.colored_point_samples` (surface point
  sampling) to produce shell voxels (per Phase B reuse plan).

### 2026-06-01 — M0-3: LOD2 shell → target occupancy + semantics

- Added `src/pointcraft/data/target.py`:
  - `voxelize_target(verts, faces, grid) -> (coords_target int32[M,3],
    occ_target uint8[M] all-1, sem_target int64[M])`. Shell only (D2): each
    triangle is barycentric-sampled (spacing = voxel_size/2) into world points,
    mapped to the shared `VoxelGrid`, merged with `np.unique`.
  - Deterministic semantics from face orientation: `|n_z| >= roof_nz (0.7)` →
    roof(3), else facade(4); per-voxel label = majority vote of samples, ties →
    roof (lower id). Seeded RNG → deterministic (tested).
  - `load_lod2_meshes(tile_dirs_or_objs)` merges multiple OBJs (vertex-index
    offset), reusing `data.lod2.parse_obj`.
  - Exported `voxelize_target`, `load_lod2_meshes`, `ROOF_LABEL`, `FACADE_LABEL`.
- Tests `tests/test_target_occupancy.py` (7 passing): contract dtypes/shapes,
  occ all-1 + unique/in-bounds coords, labels ⊆ {3,4}, roof at k=3 + facade
  present, **facade fills mid-height k∈{1,2}** (the aerial-unobserved completion
  region), determinism, empty-mesh.
- Real-LOD2 smoke (`53394611`+`53394621`, clipped to LAS grid): 240,646 faces →
  74,850 shell voxels (roof 33,453 / facade 41,397), occ all-1, in-bounds,
  k∈[10,63] (LOD2 building bottoms z≈−4 to roofs ≈48 m; the 213 m towers fall
  outside the LAS XY extent and are correctly dropped). ~18 s for one tile.
- **Known limitations** (logged for later): (a) a building bottom face is
  near-horizontal so labelled roof — acceptable for M0, rare in PLATEAU LOD2;
  (b) sloped roofs near the |n_z|=0.7 boundary may flip roof/facade; (c) ~18 s
  because all faces are sampled before XY clipping — a per-face bbox pre-reject
  (cf. `colored_point_samples`) would speed run_m0 up; deferred (one-tile cost ok).
- Next: **M0-4** observed/unobserved masks (D4) + `.npz` writer (all contract
  fields + metadata), then sanity view + `scripts/run_m0.py`.

### 2026-06-01 — M0-4: observed/unobserved masks + .npz writer

- Added `src/pointcraft/data/sample.py`:
  - `compute_masks(coords_target, coords_partial, grid)` — observed =
    `coords_target ∈ coords_partial` (via int64 ravel keys on grid shape),
    unobserved = `1 - observed`; both uint8 aligned to `coords_target` (D4).
  - `build_metadata(grid, tile_id, crs, source_files, …)` — all 9 contract keys
    (`dataset_version` default `v0.1`, `feature_layout` default = partial layout).
  - `write_sample_npz(path, …)` — writes the 7 array fields at exact contract
    dtypes + `metadata` as a 0-d JSON string array → reloadable with plain
    `numpy.load` (no allow_pickle). `load_sample_metadata` decodes it back.
  - Exported the above from `pointcraft.data`.
- Tests `tests/test_sample_io.py` (4 passing, end-to-end on the cube fixture):
  masks complementary & row-aligned; observed⊇roof / unobserved⊇mid-facade;
  masks match brute-force set membership; write→`np.load` round-trip checks every
  contract field dtype, metadata JSON keys/values, and mutual shape consistency.
- Full suite now **25 passing**.
- Next: **C6** sanity visualization of a produced sample + **C7**
  `scripts/run_m0.py` (raw LAS + LOD2 → one `.npz`, end-to-end), then Phase D
  regression tests (grid-equality, alignment) + Phase E commit/handoff to M1.

### 2026-06-01 — C6 + C7: run_m0 end-to-end + sanity view

- Added `scripts/run_m0.py`: one command, raw LiDAR + LOD2 → one contract `.npz`.
  - Inputs via `--config` (reuses `configs/tokyo_station.yaml`), explicit
    `--las/--lod2`, or `--fixture` (tiny committed data, fast smoke). `_load_points`
    dispatches `.las/.laz` (laspy) vs `.csv` (fixture).
  - Builds the shared `VoxelGrid` from the **LiDAR extent** (contract rule 4);
    runs `voxelize_partial` + `voxelize_target` on that one grid, `compute_masks`,
    `build_metadata`, `write_sample_npz`. `--viz` saves a matplotlib sanity PNG.
  - **Grid-extent fix:** tile bounds use an EXCLUSIVE upper bound (`floor+1`
    cells), so a point exactly on `max()` (e.g. an integer roof z) lands inside
    the grid instead of flooring one cell past `ceil(extent/voxel)` and being
    dropped. (Fixture exposed this: roof k=3 points were being lost; partial went
    3→12 voxels after the fix.) `VoxelGrid.from_bounds` is unchanged — its
    exclusive-bounds semantics are correct; the script now feeds it correct
    bounds. Gotcha noted for any future grid-from-points caller.
- Real-tile end-to-end (`configs/tokyo_station.yaml`):
  grid 240×232×64; partial 96,264 voxels; target 74,850 shell voxels (roof
  33,453 / facade 41,397); masks observed 17,854 / **unobserved 56,996 (76.1 %)**
  — i.e. ~3/4 of the target (mostly facades) is never seen by the aerial LiDAR,
  which is exactly the completion region M4 targets. `.npz` 813 KB. Sanity PNG
  eyeballed (height map, semantic top-down, vertical slice) — looks correct.
- Generated outputs land in `outputs/` (git-ignored — `.npz` + `outputs/`).
- Next: **Phase D** regression tests (grid-equality of partial/target; building
  footprint/alignment) and **Phase E** handoff to M1.

### 2026-06-01 — Phase D + E: regression tests, M0 DONE, handoff to M1

- Added `tests/test_pairing_alignment.py` (3 tests): partial & target share one
  grid (`in_bounds` + `0<=idx<shape`); building target roof footprint overlaps
  the observed-roof voxels and they're flagged observed; `run_m0 --fixture`
  end-to-end writes a `numpy.load`-able `.npz` with every contract field +
  metadata, partial recovering roof(k=3) and ground(k=0). World↔index round-trip
  stays in `test_voxel_grid.py`. **Full suite: 28 passing.**
- **ACCEPTANCE re-checked — all met:** end-to-end one tile (run_m0), shared grid
  (tested), independently loadable npz with all fields/dtypes, complete metadata
  (`dataset_version=v0.1`), round-trip + grid-equality + alignment tests, sanity
  viz, masks stored (D4), session log updated, scope respected (no NN/torch/
  semantic learning), `scripts/run_m0.py` present.
- Status → **DONE (v0.1)**. Known limitations consolidated in the block above.
  **M1 is unblocked** — see the "Next recommended prompt" at the top.
- Commit sequence this session: docs(Phase A) → docs(Phase B) → feat partial →
  feat target → feat masks+writer → feat run_m0 → test(Phase D) → this handoff.

### 2026-06-01 — Alignment audit of the observed/unobserved masks (⚠ findings)

Triggered by a sanity question on the "76.1 % unobserved" figure. Dissected the
real-tile `.npz`; the headline number is **inflated by two artifacts**, not all
genuine blind spots:

- **Composition:** of 56,996 unobserved voxels, 30,345 are facade but **26,651
  are roof** — i.e. **80 % of roof voxels are flagged "not seen"**, which is
  physically wrong (aerial LiDAR sees roofs). So the masks over-state the
  completion region.
- **Height datum is fine.** On the 11,377 columns where both exist, partial-top
  minus LOD2-roof k: **median 0.0**, 74 % within ±1 voxel, no systematic offset
  (Phase-B z-gate holds). Mean +2.07 / 19.5 % of partial sits >2 above the roof =
  trees/rooftop clutter over the footprint (expected).
- **Artifact 1 — z quantization.** `compute_masks` matches voxels by EXACT
  (i,j,k). A roof surface at k=40 and a roof point at k=39 differ <1 m but miss.
  Allowing ±1 voxel tolerance: roof observed 20 %→35 %, overall unobserved
  76 %→60 %.
- **Artifact 2 — XY coverage gap (the real concern).** 36 % of LOD2 roof columns
  have NO LiDAR point at all (any height). The 3-colour XY map shows these
  concentrate at the **tile edges/corners** — LOD2 extent ≫ the LiDAR swath
  (Phase B noted this), so edge buildings have model geometry but no point
  support. Using those as supervision = teaching from unevidenced data.
- **Conclusion:** geometry/datum align well *inside the swath*; the masks need
  (a) a vertical tolerance in `compute_masks`, and (b) clipping the target to the
  LiDAR-covered footprint (or marking LOD2-only cells `ignore`, excluded from the
  M4 denominator). **Genuine blind region ≈ the facades (~40 % of target), not
  76 %.**

**Action taken:** added two QA exporters (PLY, git-ignored under
`outputs/m0/align3d/`): `scripts/export_alignment_3d.py` (whole tile: raw LiDAR
height-coloured + LOD2 roof-red/facade-blue) and `scripts/export_buildings.py`
(per-building, NMS + per-height-band quota so the sample spans skyscraper/tall/
midrise/low, not just one landmark; tags point-sparse coverage-edge towers
`_SPARSEpts`). Both are source geometry (no voxelization) for manual
CloudCompare/MeshLab review. Real tile: dense-coverage mid/low-rise are the clean
alignment checks; the ~160 m towers are point-sparse (swath-edge coverage gap).
Also `scripts/render_alignment.py` → renders dense buildings as 2D EW/NS
cross-section PNGs (black LiDAR / red LOD2 roof / blue LOD2 facade, all clipped to
one XY box) so a reviewer with no 3D viewer can judge alignment from images
(`outputs/m0/align3d/renders/`). Visual check on dense mid/low-rise: red roofs sit
on the black point tops, blue walls stand in the point-empty gaps — source data
aligns; the 76 % is inflated by exact-voxel matching + the coverage gap, as
diagnosed.
**No labelling logic changed yet** — `compute_masks`/`target` left as-is pending
the human verdict. Candidate fixes (z-tolerance, coverage clip) are on hold.
Once verified, log the decision in `docs/06_DECISIONS.md` and bump
`dataset_version` if the mask definition changes.

### 2026-06-03 — M0 Phase A (v2): adopt CityGML target (D5), finalize contract

Re-targeting M0 from OBJ to CityGML per `EXECUTION_PLAN.md` v2. **Docs only — no
`.py` changed** (Phase A scope).

- `docs/06_DECISIONS.md`:
  - Added **D5** — CityGML replaces OBJ as the M0 target; roof/facade/ground
    semantics read from CityGML surface types (`RoofSurface`/`WallSurface`/
    `GroundSurface`) instead of inferred from face `|n_z|`. Removes the
    building-base-→-roof mislabelling the alignment audit flagged and unlocks a
    `ground`(1) class. Requires a GML parser + **6697→6677** reprojection (z passes
    through). OBJ kept only as fallback/comparison. `dataset_version` stays `v0.1`.
  - Added a **correction entry** confirming there is no repo-root `pointcraft/`
    (legacy merged into `src/pointcraft/` and deleted) — new CityGML code goes in
    `src/pointcraft/data/`.
- `docs/02_DATA_CONTRACT.md` finalized for the CityGML target:
  - `crs` clarified = `EPSG:6677` for grid + stored coords; CityGML reprojected
    6697→6677 (Alignment rule 1 rewritten: horizontal-only, z re-verified Phase C2).
  - `source_files` target = the CityGML grid file(s) (OBJ only under fallback).
  - Semantic label table sourced from CityGML surface types; `ground`(1) now
    **active**; explicit surface-type→label mapping added; class `2` still unused
    (shell, D2). `feature_layout = ["height","point_count"]` and required masks (D4)
    unchanged.
- This SESSION_LOG: status → **IN PROGRESS**; tile = `09LD1848`; target = CityGML;
  next-recommended-prompt repointed to Phase B (tile-↔-CityGML gate + reuse
  inventory + GML probe).

**Stop point:** Phase A complete. Phase B is the tile-mapping gate — if `09LD1848`
has no on-hand CityGML grid in `tile_alignment.csv`, pick a tile that does before
any further work.

### 2026-06-03 — M0 Phase B (v2): tile gate + reuse inventory + GML probe (read-only)

Read/probe only — **no `.py` changed**.

**B1 — tile↔CityGML gate → PASSED ✓ (cleaner than feared).**
- `09LD1848` (`tile_alignment.csv`): `mesh_codes=53394622;53394632`,
  `in_lod2_citygml=1`. Bbox (6677) X[-4800,-4400] Y[-34500,-34200].
- On-hand bldg GMLs = the 3×3 block 53394600–622 (9 files in
  `data/raw/citygml/udx/bldg/*_bldg_6697_2_op.gml`). `53394622` **is present**;
  `53394632` is **not** (LOD1-only 30/31/32 row, never downloaded).
- Per `mesh_index.csv`, grid **53394622** extent X[-5324,-4146] Y[-35173,**-34191**]
  **fully contains** the 09LD1848 footprint (its north edge -34191 already passes the
  tile's north edge -34200). So `53394632` is only a bbox-overlap of the adjacent
  grid over a ~35 m north sliver that 53394622 also covers. **Net: 09LD1848 is
  effectively fully covered by the single on-hand `53394622` GML.**
- ⚠ Residual edge risk: under PLATEAU's "building belongs to one mesh cell by its
  representative point" rule, a few buildings in the north sliver could live in the
  un-downloaded `53394632` file. Minor; revisit only if the north edge looks bare in
  the C2 alignment check.

**B2 — reuse inventory** (`src/pointcraft/`, current layout):
- LAS load: `data.partial.load_las_xyz` (laspy wrapper) + `voxelize_partial`
  (LiDAR→partial on `VoxelGrid`, v0.1 layout) — **reuse as-is for C4** (no change).
- Shared grid: `voxelization.grid.VoxelGrid` (`from_bounds`, `world_to_index`,
  `index_to_center/corner`, `in_bounds`) — **reuse as-is for C3** (already done).
- Target plumbing: `data.target.voxelize_target` voxelizes a triangle mesh shell
  onto the grid but derives semantics from face `|n_z|` — **C5 will not reuse its
  labelling**; it can reuse the barycentric surface-sampling → `VoxelGrid` mechanic,
  fed by CityGML *typed* polygons (label comes from surface type, not normal).
- Masks + writer: `data.sample.compute_masks` / `build_metadata` /
  `write_sample_npz` / `load_sample_metadata` — **reuse as-is for C6/C7**.
- OBJ parser: `data.lod2.parse_obj` (+ `LOD2Rasterizer.colored_point_samples`) —
  fallback path only; also a **structural reference** for the new GML polygon
  extractor (both yield typed polygons to sample).
- Config: `utils.config` (`resolve_path` resolves config-relative paths) — reuse
  for `run_m0`/configs (C9). `pipeline.py` (Context/Stage/Pipeline) not needed in M0.
- Net new code for C1/C5: `data/citygml.py` (GML parse + reproject + typed
  surfaces) and a typed-surface shell voxelizer; everything else reuses existing.

**B3 — GML probe** (`53394622_bldg_6697_2_op.gml`, 34 MB):
- CRS declared `EPSG:6697`, `srsDimension="3"`. `gml:posList` coords are
  **lat lon z** order, e.g. `35.683… 139.779… 30.0995` → axis order is (Y=lat,
  X=lon, Z=elev). **C1 reprojection must respect this axis order** (pyproj
  `always_xy=False` for 6697, or swap before transform) — getting it wrong silently
  flips easting/northing. z is absolute elevation in metres (D3 holds; reprojection
  is horizontal-only, z passes through).
- LOD2 present. Tag counts: 3101 `bldg:Building`, 2044 `RoofSurface`,
  5320 `WallSurface`, **226 `GroundSurface`**, 213 `lod2Solid`,
  7619 `lod2MultiSurface`, 3101 `lod1Solid`.
- **GroundSurface is sparse (226 / 3101 buildings).** PLATEAU LOD2 buildings are
  mostly open-bottomed (no explicit base face). Consequence for D5: `ground`(1) from
  `GroundSurface` is the *building footprint base*, present only where modelled — it
  is **not** terrain ground (terrain still needs the DEM, deferred). Upside: open-
  bottomed buildings have no bottom face to mislabel, so the OBJ "base→roof" bug is
  gone regardless. Most target voxels will be roof(3)/facade(4) as before, now
  correctly typed at the source.

**Phase B exit met:** tile confirmed (53394622 on hand, covers tile), reuse map
written, GML structure + CRS + axis-order confirmed. No code added.
**Next: Phase C1** — `src/pointcraft/data/citygml.py` (parse typed LOD2 surfaces +
6697→6677 reprojection with correct lat/lon axis order; add `pyproj` dep). This is
the first step that touches `.py` / deps — pause for go-ahead before starting.

> ⚠ **Phase-B follow-up (same day): the 09LD1848 gate was over-optimistic.** The
> "53394622 covers the tile" conclusion above was a *bbox-overlap* judgement; once
> C1 could actually reproject and clip the surfaces (below), **0** of 53394622's
> 7590 surfaces have a centroid inside 09LD1848 — the building cluster sits outside
> the tile. Tile changed to **09LD1874**. See the C1 entry + `docs/07_GOTCHAS.md`.

### 2026-06-03 — M0 C1: CityGML parser + reprojection; tile switched to 09LD1874

**C1 — `src/pointcraft/data/citygml.py`** (committed; `pyproj>=3.6` added to deps):
- `parse_citygml(path)` — stdlib `xml.etree` iterparse over a PLATEAU LOD2 bldg
  file; pulls exterior rings of each semantic surface (`RoofSurface`/`WallSurface`/
  `GroundSurface`) and reprojects them EPSG:6697→6677. Returns `TypedSurfaces`
  (`polygons` list of `(n,3)` 6677 rings + aligned int64 `labels`; `.counts()`).
  `load_citygml(paths)` merges grids. No voxelization here (per plan).
- **Axis order locked by calibration:** posList is `lat lon z`; reproject with
  pyproj `always_xy=True` feeding `(lon, lat)` → `(easting=x, northing=y)`. Verified:
  reprojecting grid 53394622's envelope reproduced its known 6677 extent
  X[-5324,-4146] Y[-35173,-34191]. z passed through unchanged (D3).
- Validated on `53394622_bldg_6697_2_op.gml` (34 MB): 0.6 s, 7590 rings
  {ground 226, roof 2044, facade 5320} — exactly the raw tag counts; reprojected
  extent within the grid bbox; Z∈[1.0, 92.4] m. 28 existing tests still pass.

**Tile gate re-run with real geometry → 09LD1848 DROPPED.** Clipping the reprojected
surfaces to each LAS footprint (centroid-in-bbox over all 9 on-hand grids,
336,382 surfaces):
- `09LD1848`: **0** surfaces (despite `in_lod2_citygml=1`). The flag is only a
  bbox-overlap; 53394622's buildings don't actually intersect the tile.
- Densest tiles: 09LD2815 (15,427), 09LD2817 (14,588), 09LD2805 (13,435) … all with
  6–7 M LiDAR pts and ~100 % tile coverage.
- **Chosen: `09LD1874`** (2061 surfaces; grids 53394610/611/620/621; LiDAR 6.39 M).
  Picked over the denser south tiles for continuity — it's the tile
  `configs/tokyo_station.yaml` already targets and the one the prior OBJ M0 ran on,
  so the CityGML target can be compared apples-to-apples against the OBJ result.
- Logged the bbox-flag trap in `docs/07_GOTCHAS.md`; updated D5 status
  (`docs/06_DECISIONS.md`) and `EXECUTION_PLAN.md` first-tile to 09LD1874.

**Next: Phase C2** — alignment re-verification gate on 09LD1874 (CityGML roof/wall/
ground vs LiDAR cross-sections) before any voxelization.
