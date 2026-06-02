# M0 — Data Pairing — EXECUTION PLAN (v2: CityGML target)

Supersedes the OBJ-target assumptions of the earlier plan. Reflects the current
repo reality (per `docs/07_GOTCHAS.md`): the legacy root package is merged into
`src/pointcraft/`, real data lives under `data/raw/` (git-ignored), and CityGML
is downloaded but not yet wired in.

## Locked decisions (carried forward, confirmed)

- **D1** voxel size = 1.0 m.
- **D2** building target = shell (surface voxels), not solid.
- **D3** vertical reference = absolute elevation, shared origin; verified on real
  buildings (LiDAR roof points sit on LOD2 roof surfaces — no datum offset).
- **D4** masks = `observed_mask` / `unobserved_mask` are required M0 outputs.
- **NEW — D5** target source = **CityGML replaces OBJ** as the M0 target.
  Semantics (roof / wall / ground) are read from CityGML surface types, not
  inferred from geometry. This eliminates the geometry-heuristic bug (building
  base mislabelled as roof) seen in the OBJ-based calibration.

## First end-to-end tile

`09LD1848` (LAS sheet → row 14, col 88).

> ⚠ First action in Phase B: confirm in `data/raw/tile_alignment.csv` that
> `09LD1848` has a corresponding LOD2 / CityGML grid (one of 53394600–622). If it
> does not map to an on-hand CityGML grid, **stop and pick a tile that does.**

## What changed vs. v1 (read before planning)

- **No legacy root package.** Reuse modules from the merged `src/pointcraft/`:
  `pointcraft.data.lod2` (OBJ parsing today), `pointcraft.pipeline`,
  `pointcraft.utils.*`. Do not look for or import a repo-root `pointcraft/`.
- **OBJ is no longer the target path.** v1 said "M0 uses OBJ." That is reversed:
  CityGML is the target. OBJ stays only as a fallback/comparison if CityGML
  integration stalls.
- **CityGML is staged, not integrated.** Per GOTCHAS, it needs (a) a GML parser
  and (b) EPSG:6697 (lat/lon) → 6677 reprojection before it can align with the
  LiDAR. This is the first build step, not an afterthought.
- **Real data is git-ignored under `data/raw/`.** Never commit LAS / OBJ / GML /
  `.npz`. Configs use paths relative to the config file.

## Phase A — Sync decisions to memory (docs only, no code)

- `docs/06_DECISIONS.md` — append:
  - **D5:** CityGML replaces OBJ as M0 target; semantics from surface types.
    Reason: removes geometry-heuristic mislabelling; gives true roof/wall/ground.
    Consequence: requires a GML parser + 6697→6677 reprojection; OBJ retained
    only as fallback/comparison.
  - A **correction entry** noting the legacy root `pointcraft/` is merged &
    removed (the earlier "keep legacy as M1 baseline" decision is now executed via
    merge, not a separate package).
- `docs/02_DATA_CONTRACT.md` — finalize:
  - `source_files` for the target = the CityGML grid file(s), not OBJ.
  - `crs` notes: LiDAR native 6677; CityGML 6697 reprojected → 6677; the grid and
    all stored coords are in 6677.
  - Semantic label table mapped to CityGML surface types: RoofSurface→roof(3),
    WallSurface→facade(4), GroundSurface→ground(1); vegetation(5)/road(6) as
    available; ignore(255). With D2=shell, "building solid interior" (2) is unused.
  - `dataset_version = "v0.1"`; pin `feature_layout` (minimal set).
  - Masks promoted to required.
- `tasks/M0_data_pairing/SESSION_LOG.md` — status → IN PROGRESS; record that data
  is staged under `data/raw/`, tile = `09LD1848`, target = CityGML.
- Commit (docs only): `docs(m0): adopt CityGML target (D5), finalize contract`.

**Exit:** decisions + contract committed; no `.py` changed.

## Phase B — Verify tile + inventory reuse (read + minimal probe)

- Confirm `09LD1848` ↔ CityGML mapping in `tile_alignment.csv` (gate above).
- Inventory `src/pointcraft/` reuse: which existing modules give us LAS load,
  voxel/grid helpers, OBJ LOD2 parsing (for the fallback + as a structural
  reference for the GML parser), config path resolution. Note in SESSION_LOG.
- Probe one CityGML file: open the `09LD1848`-matching grid `.gml`, confirm it
  contains LOD2 `bldg:RoofSurface` / `bldg:WallSurface` / `bldg:GroundSurface`,
  and read its CRS declaration (expect 6697 / lat-lon). Log a few surface counts.

**Exit:** tile confirmed to have CityGML; reuse map written; GML structure + CRS
confirmed.

## Phase C — Implement (code)

Order is deliberate: get CityGML into the LiDAR's frame first, re-verify
alignment, then voxelize. Don't build voxelization on an unverified reprojection.

### C1 — CityGML parser + reprojection ★ (the first real step) — `src/pointcraft/data/citygml.py`

- Parse LOD2 buildings; extract surface polygons tagged by type (roof/wall/ground).
- Reproject geometry 6697 → 6677 (use pyproj; add to deps).
- Keep z handling consistent with D3 (absolute elevation); verify the vertical
  datum survives reprojection (6697→6677 is horizontal; confirm z is passed
  through unchanged / correctly).
- Output an in-memory representation: typed surfaces with 6677 coordinates, ready
  for voxelization. No voxelization in this module.

### C2 — Re-verify alignment on 09LD1848 ⚠ gate

- Reproduce the earlier EW/NS slice calibration, now with CityGML-derived
  roof/wall/ground vs. LiDAR points, on a couple of buildings in `09LD1848`.
- Confirm LiDAR roof points still sit on CityGML roof surfaces (datum +
  reprojection both correct), and that ground is now labelled ground (the base
  line mislabelled roof under OBJ should now be GroundSurface).
- If misaligned: fix reprojection/datum before C3. Do not proceed on a bad frame.

### C3 — Shared voxel grid utility — `src/pointcraft/voxelization/`

- Grid from world bounds + voxel_size → origin, grid_shape; world_xyz ↔ (i,j,k)
  per `02_DATA_CONTRACT`. Same grid object for partial and target. Round-trip unit
  test alongside. *(Already exists as `VoxelGrid`; reuse.)*

### C4 — LiDAR → partial occupancy — `src/pointcraft/data/`

- Reuse LAS loader. Map points → voxel indices on the shared grid; drop
  out-of-range (log count); merge duplicates.
- `coords_partial` int32[N,3] + `feats_partial` float32[N,C].
- `feature_layout` v0.1 (minimal): `["height", "point_count"]`. Expand later.
  *(Already exists as `voxelize_partial`; reuse.)*

### C5 — CityGML → target occupancy + semantics — `src/pointcraft/data/`

- Voxelize the typed surfaces (shell, D2) from C1 onto the same grid →
  `coords_target`, `occ_target` (=1 stored), `sem_target` (roof/facade/ground from
  surface type — no geometry heuristic).

### C6 — Masks

- observed = `coords_target ∈ coords_partial`;
  unobserved = `occupied_target ∧ ¬observed`. Same pass, same grid.

### C7 — .npz writer

- All contract fields + full metadata (`tile_id="09LD1848"`, voxel_size, origin,
  bounds, grid_shape, `crs="EPSG:6677"`, `source_files=[CityGML grid, LAS sheet]`,
  `dataset_version="v0.1"`, feature_layout).

### C8 — Sanity visualization

- One eyeball of the produced sample (slice plot reusing the calibration viewer).

### C9 — `scripts/run_m0.py`

- One command: LAS + CityGML(`09LD1848`) → one `.npz`, end to end, config-driven,
  paths relative to config.

## Phase D — Tests (`tests/`)

- World↔index round-trip (from C3).
- Grid-equality: partial & target share origin/voxel_size/grid_shape.
- CRS/reprojection sanity: a known point reprojects 6697→6677 within tolerance.
- Alignment regression: a building's CityGML footprint overlaps its observed roof
  voxels.
- `.npz` loads via `numpy.load`; all contract fields present with right dtypes.

## Phase E — Commit + handoff

Commit granularity (each green, small):

- `feat(data): CityGML LOD2 parser + 6697→6677 reprojection`
- `feat(voxelization): shared grid + world/index transforms (+ round-trip test)`
- `feat(data): LiDAR→partial occupancy (v0.1 feature layout)`
- `feat(data): CityGML shell→target occupancy + surface-type semantics`
- `feat(data): observed/unobserved masks + npz writer (data contract)`
- `feat(scripts): run_m0 end-to-end on 09LD1848 + sanity view`
- `test(m0): round-trip, grid-equality, reprojection, alignment regression`

Then `tasks/M0_data_pairing/SESSION_LOG.md`: status → DONE (or actual); dated
entry (what built, decisions, known limits); "next recommended prompt" → M1 (now
unblocked; baseline scored against this `.npz`).
Final: `docs(m0): session log + handoff to M1`.

## Definition of done (mirrors ACCEPTANCE, + CityGML specifics)

- [ ] CityGML parsed, reprojected 6697→6677, alignment re-verified on `09LD1848`.
- [ ] One `.npz` for `09LD1848`, fully automated via `run_m0.py`.
- [ ] partial/target on identical shared grid (tested).
- [ ] `.npz` independently loadable; all contract fields + complete metadata.
- [ ] `sem_target` from CityGML surface types (ground no longer mislabelled).
- [ ] `observed_mask` / `unobserved_mask` stored.
- [ ] Tests pass (incl. reprojection sanity). Sanity viz produced.
- [ ] SESSION_LOG + 06_DECISIONS (D5) + 02_DATA_CONTRACT updated.

## Scope guards (unchanged)

- ❌ No neural network; no torch / spconv / Minkowski.
- ❌ No semantic learning — deterministic from CityGML surface types only.
- ❌ No Minecraft export beyond optional debug preview.
- ❌ No multi-tile batching system — `09LD1848` proves the contract.
- ❌ Do not commit real data (`data/raw/`) or generated `.npz`.
- ❌ Do not recreate a repo-root `pointcraft/`; one package at `src/pointcraft/`.
