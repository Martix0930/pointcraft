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

## Windows

- Git may warn `LF will be replaced by CRLF` — harmless.
