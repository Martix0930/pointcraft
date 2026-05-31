# M0 — Data Pairing — SESSION LOG

Append a dated entry at the end of every session. Keep the "Current status" and
"Next recommended prompt" blocks at the top up to date.

---

## Current status: **NOT STARTED** (repo cleaned up; fixtures added; ready to begin)

Package layout standardized: `src/pointcraft/` is the single importable
`pointcraft` package. The deterministic baseline (LiDAR load, voxelization,
LOD2/LAS alignment) is available for reuse at `src/pointcraft/baseline/` +
`pointcraft.data.lod2` + `pointcraft.pipeline`. Tiny test fixtures now exist under
`test_data/`. No M0 code written yet.

## Next recommended prompt for Claude Code

> Read `CLAUDE.md`, `docs/07_GOTCHAS.md`, `docs/02_DATA_CONTRACT.md`, and
> `tasks/M0_data_pairing/` (TASK_SPEC, ACCEPTANCE, CHECKLIST). Then start **M0-1**:
> implement the shared voxel grid utility in `src/pointcraft/voxelization/`
> (grid from world bounds + `voxel_size`: `origin`, `grid_shape`; `world_xyz`↔voxel
> index for centers/corners). Add a test in `tests/` that loads
> `test_data/m0_voxel_grid/` (`tiny_points.csv`, `tiny_bounds.json`) and asserts the
> computed indices equal `expected_indices.json`, plus a world↔index round-trip.
> Reuse `pointcraft.baseline` / `pointcraft.data.lod2` where helpful. Do NOT
> implement any neural network. Update this SESSION_LOG when done.

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
