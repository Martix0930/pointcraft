# 07 — Gotchas

Practical traps to avoid. Add entries as they bite.

## Naming: package vs. project brand

- **Do not uppercase the Python package directory.** It is `src/pointcraft/`.
- **Do not rename `src/pointcraft/` to `src/PointCraft/`.**
- **Do not change `import pointcraft` to `import PointCraft`.**
- The **display name and the import/package name intentionally differ**:
  - code / path / import / packaging → lowercase `pointcraft`
  - human-facing prose / titles / docs / slides → `PointCraft`
- `pyproject.toml` `[project].name` stays lowercase `pointcraft`.

## Single package — no shadowing

- There must be **exactly one** importable `pointcraft`, at `src/pointcraft/`.
- A previous **repo-root `pointcraft/`** package shadowed it (`import pointcraft`
  resolved to the root, never to `src/`). It has been **merged into
  `src/pointcraft/` and removed**. Do not recreate a top-level `pointcraft/` dir.
- Install editable so imports resolve correctly:
  ```
  pip install -e .
  python -c "import pointcraft; print(pointcraft.__file__)"   # -> src/pointcraft/__init__.py
  ```
- Scripts add `REPO/src` (not `REPO`) to `sys.path` if not relying on the install.

## Imports after the merge

- Pipeline core: `from pointcraft.pipeline import Context, Stage, Pipeline`
- Baseline stages (M1): `from pointcraft.baseline.stages import ...`
- LOD2 parsing: `from pointcraft.data.lod2 import ...`
- Block palette: `from pointcraft.mc_export.palette import ...`
- Viewer: `from pointcraft.utils.viewer import ...`

## Dependencies

- `import pointcraft` stays cheap: heavy deps (laspy, mcschematic, pyvista,
  skimage) are imported **inside** submodules, not at package top level.
- Learning libs (torch / spconv / Minkowski) are **not** installed before M2.

## Test data

- `test_data/` is for **tiny fixtures only** (synthetic / heavily reduced).
- **Do not commit real datasets** (LiDAR/PLATEAU/CityGML/large OBJ) or **generated
  training samples** (`.npz`/`.npy`). Those are globally gitignored and must stay
  outside git or in ignored local folders.
- `.obj` is globally ignored; tiny fixtures are re-included only via the
  `!test_data/**/*.obj` rule in `.gitignore`. Do not add `!test_data/**/*.npz`
  exceptions unless a tiny `.npz` fixture is intentionally created.

## Data location (large local datasets)

- All large source data now lives under **`data/raw/`** (git-ignored). The old
  external paths (`D:/Desktop/实习/三维GIS/...`, `D:/Soft/Chrome/Download/...`)
  are **gone — don't look there or hardcode them.**
- Layout:
  - `data/raw/lidar/` — 60 `09LDxxxx.las` (full tiles, EPSG:6677)
  - `data/raw/lod2/` — 9 mesh OBJ (53394600–622); **current M0 target source**
  - `data/raw/dem/` — terrain (for a future height-above-ground version)
  - `data/raw/citygml/` — 9 LOD2 grids, **EPSG:6697 (lat/lon)**; staged to replace
    the OBJ later (needs a GML parser + 6697→6677 reprojection — not done yet)
  - `data/raw/tile_alignment.csv` + `data/raw/README.md` — LAS↔mesh↔CityGML map
- Configs use **paths relative to the config file** (e.g. `../data/raw/lidar/...`),
  resolved by `pointcraft.utils.config.resolve_path`. Keep new paths relative /
  in-project; regenerate the index with `python scripts/build_tile_index.py`.
- `09LD` sheet decode: `09LD WXYZ` → row(N–S)=`10·W+Y`, col(E–W)=`10·X+Z`.

## CityGML tile coverage: `in_lod2_citygml` is a *bbox* flag, not real coverage

- `tile_alignment.csv`'s `in_lod2_citygml=1` only means the LAS sheet's bbox
  **overlaps the axis-aligned bbox** of an on-hand LOD2 grid. It does **not**
  guarantee any CityGML *buildings* actually fall inside the tile.
- Real example (M0, 2026-06-03): `09LD1848` is flagged `1` (overlaps grid
  `53394622`), but **0** of 53394622's 7590 surfaces have a centroid inside the
  tile — the grid's building cluster sits outside the tile, the bboxes merely
  touch. The planned "first tile" was abandoned for `09LD1874` (Tokyo-Station
  core, ~2061 surfaces, matches `tokyo_station.yaml`).
- **Before committing to a tile, clip reprojected CityGML surfaces to the LAS
  footprint and count them** (`parse_citygml` + centroid-in-bbox), don't trust the
  flag. Mesh grids in `mesh_index.csv` carry *data-extent* bboxes (irregular), so
  bbox overlap ≠ geometry overlap.

## CityGML→tile clipping by centroid mismatches at tile borders (M2 BLOCKER)

- `run_m0` currently keeps a CityGML surface if its **exterior-ring centroid** lands
  in the grid XY extent. A building **straddling a tile boundary** is then taken
  whole or dropped whole by its centroid, while its LiDAR points are clipped to the
  LAS bbox. Result: a border building can have **roof LiDAR points inside the tile
  but no CityGML target** (centroid fell just outside) — or vice versa.
- Verified on 09LD1874: the ~152 m north-edge tower's LiDAR roof points sit inside
  the tile, but 57 % of its CityGML surfaces have centroids north of the edge and
  were clipped → it shows up as "tall LiDAR, no target".
- **M0 (single tile) is unaffected:** those extra LiDAR voxels are just *unanswered
  input context* — they go into `coords_partial` only, never into `coords_target`,
  so they are neither observed nor unobserved and never enter any metric.
- **M2 BLOCKER (see `docs/02_DATA_CONTRACT.md` alignment rules):** before multi-tile
  training, replace centroid clipping with **geometry-overlap clipping** or an
  **ignore margin** at tile borders — otherwise the border band systematically
  teaches the model that visible roofs are empty.

## Windows

- Git may warn `LF will be replaced by CRLF` — harmless.
