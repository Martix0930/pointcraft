# M0 — Data Pairing — SESSION LOG

Append a dated entry at the end of every session. Keep the "Current status" and
"Next recommended prompt" blocks at the top up to date.

---

## Current status: **IN PROGRESS** — Phase A + Phase B done (decisions locked, data contract finalized, z-alignment GATE PASSED); M0-1 (voxel grid) done; next is Phase C/M0-2 (LiDAR → partial occupancy)

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
