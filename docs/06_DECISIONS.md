# 06 — Decision Log

Append-only log of non-trivial project decisions. Newest at the bottom.

Template:

```
## YYYY-MM-DD - Decision title

Context:
Decision:
Reason:
Consequences:
Status:
```

---

## 2026-06-01 - Use repository documents as shared memory between Claude App and Claude Code

Context:
The project spans research planning (Claude App) and implementation (Claude Code).
Relying on long chat history as memory is lossy and does not transfer between
sessions or tools.

Decision:
The repository is the single source of truth. Stable knowledge lives in `docs/`;
per-milestone work lives in `tasks/M*/` with `TASK_SPEC.md`, `ACCEPTANCE.md`,
`CHECKLIST.md`, and `SESSION_LOG.md`. Decisions are logged here.

Reason:
Future sessions must be able to work from small, self-contained task specs instead
of replaying chat history. Documents are durable, diffable, and tool-agnostic.

Consequences:
- Every session reads the active task files before coding and updates
  `SESSION_LOG.md` after.
- Decisions and data-format changes must be recorded in `docs/`.

Status:
Adopted.

---

## 2026-06-01 - Keep legacy pipeline as M1 baseline under repo-root `pointcraft/`

Context:
An earlier deterministic LiDAR→voxel→Minecraft pipeline already exists in
`pointcraft/` (repo root). The new research code is scaffolded under
`src/pointcraft/`, creating a package-name overlap.

Decision:
Leave the legacy package in place as the **M1 deterministic baseline**; build new
research code under `src/pointcraft/`. `pyproject.toml` packages from `src/`.

Reason:
Avoids destructive refactoring of working baseline code; preserves a comparison
floor for learned models.

Consequences:
- Potential import ambiguity if both are on `sys.path`; keep imports explicit.
- A future decision may move/rename the legacy package once it is no longer run
  directly. Log it here when that happens.

Status:
**Superseded** by "Merge legacy pipeline into `src/pointcraft/baseline`" (below).

---

## 2026-06-01 - Merge legacy pipeline into `src/pointcraft/baseline`; one package only

Context:
The repo-root `pointcraft/` package shadowed `src/pointcraft/` on import
(`import pointcraft` resolved to the legacy root), creating real import ambiguity.

Decision:
Merge the legacy deterministic pipeline into the `src` package and delete the
root package, so `src/pointcraft/` is the only importable `pointcraft`:
- `context.py` → `src/pointcraft/pipeline.py`
- `lod2.py` → `src/pointcraft/data/lod2.py`
- `palette.py` → `src/pointcraft/mc_export/palette.py`
- `viewer.py` → `src/pointcraft/utils/viewer.py`
- `stages.py` → `src/pointcraft/baseline/stages.py` (kept cohesive = M1 baseline)
- root `pointcraft/__init__.py` deleted (it force-imported heavy deps).

Reason:
Eliminate import shadowing; keep one clear package; preserve the M1 baseline as
working code without a risky split of the tightly-coupled stage module.

Consequences:
- Use `pip install -e .` (package discovered from `src/`). Verified
  `import pointcraft` → `src/pointcraft/__init__.py`.
- Scripts now import `pointcraft.pipeline`, `pointcraft.baseline.stages`,
  `pointcraft.utils.viewer`, `pointcraft.data.lod2`, and add `REPO/src` to `sys.path`.
- New M0+ code goes in the topic subpackages, not in `baseline/`.

Status:
Adopted.

---

## 2026-06-01 - Naming: lowercase `pointcraft` for code, `PointCraft` for prose

Context:
Need a single, unambiguous naming convention across Git, Python, shell, and docs.

Decision:
Use `pointcraft` for all code-facing names:
- GitHub repository: `pointcraft`
- local folder: `pointcraft`
- Python package: `src/pointcraft`
- Python import: `import pointcraft`

Use `PointCraft` for human-facing display only:
- README title, paper/project title, documentation prose, presentation materials.

Reason:
Lowercase names reduce friction in Python, Git, shell commands, and Claude Code
sessions. `PointCraft` remains the research/project brand.

Consequences:
- Do not uppercase the package directory or import name.
- `pyproject.toml` `[project].name` stays lowercase `pointcraft`.

Status:
Adopted.
