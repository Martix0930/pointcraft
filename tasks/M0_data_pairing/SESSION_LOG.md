# M0 — Data Pairing — SESSION LOG

Append a dated entry at the end of every session. Keep the "Current status" and
"Next recommended prompt" blocks at the top up to date.

---

## Current status: **IN PROGRESS** — Phase A + B done; M0-1..M0-4 + C6 (sanity view) + C7 (run_m0 end-to-end) done; produces a valid .npz on the real tile. Next: Phase D regression tests (grid-equality, alignment) + Phase E handoff to M1

Package layout standardized: `src/pointcraft/` is the single importable
`pointcraft` package. Baseline reusable code at `src/pointcraft/baseline/` +
`pointcraft.data.lod2` + `pointcraft.pipeline`. Data config externalized to
`configs/tokyo_station.yaml` (loader: `pointcraft.utils.config`). Tiny fixtures in
`test_data/`. **M0-1 shared voxel grid implemented + tested (7 passing).**

## Next recommended prompt for Claude Code

> Read `CLAUDE.md`, `docs/02_DATA_CONTRACT.md`, and `tasks/M0_data_pairing/`. M0-1
> (the shared `VoxelGrid` in `pointcraft.voxelization`) is done and tested. Start
> **M0-2: LiDAR → partial occupancy** in `src/pointcraft/data/`: load points (CSV
> fixture `test_data/m0_data_pairing/tiny_lidar_points.csv`; real LAS via
> `pointcraft.baseline` loader / laspy), map onto a `VoxelGrid` (dedupe), and
> produce `coords_partial` (int32 [N,3]) + `feats_partial` (float32 [N,C]) with the
> documented `feature_layout` (start minimal: intensity, classification). Add a test
> using the tiny LiDAR CSV asserting partial occupancy sits on the shared grid and
> the roof voxels are present. No neural network. Update this SESSION_LOG.

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
