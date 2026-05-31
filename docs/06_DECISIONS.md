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

---

## 2026-06-01 - M0/D1 - Voxel edge length = 1.0 m

Context:
M0 produces paired voxel samples; the voxel edge length is baked into every
`.npz` and into the shared grid used by both partial and target. It must be fixed
before any voxelization code is written.

Decision:
Use a voxel edge length of **1.0 m**, stored as `voxel_size` in metadata.

Reason:
Matches the Minecraft 1-block scale (M5 embodied demo), and keeps occupancy
counts and memory small enough to iterate fast on a single tile.

Consequences:
- Changing it later requires a new `dataset_version` and re-voxelizing all
  samples. Revisit at M2 if facade resolution proves too coarse.

Status:
Adopted (M0).

---

## 2026-06-01 - M0/D2 - Building target geometry is a SHELL (surface voxels only)

Context:
The LOD2/mesh target can be voxelized either as a thin surface (shell) or as an
interior-filled solid. This choice defines what `occ_target` means and which
voxels land in the M4 `unobserved` denominator.

Decision:
The building target is a **shell**: only LOD2 surface voxels are stored. No
interior flood-fill. A building is represented by its surface voxels labelled
`roof` (3) and `facade` (4); the building-*solid* class `2` is therefore unused.

Reason:
- A mesh/LOD2 is a surface, so shell is the natural, lower-work output of
  mesh→voxel; a solid needs an extra interior flood-fill.
- Aligns with the two headline goals: embodied (walk inside/around) and facade
  completion (the facade *is* an explicit voxel).
- An interior-filled solid would put meaningless interior voxels into
  `unobserved_mask`, polluting the M4 metric's denominator.

Consequences:
- Semantic label table: class `2` (building solid interior) is unused under the
  shell representation (see D2 consequence in the data contract).
- Switching to solid later requires a fill step and a new `dataset_version`;
  recoverable by re-generation from the untouched source LOD2, not a data loss.

Status:
Adopted (M0).

---

## 2026-06-01 - M0/D3 - Vertical (z) reference = absolute elevation, shared origin

Context:
`z` can be stored as absolute elevation or as height-above-ground. The latter is
nicer for learning (translation invariance) but requires a terrain/DTM model.

Decision:
Use **absolute elevation** with a single shared origin for both partial and
target. No ground model / DTM in M0.

Reason:
Simplest correct option; needs no DTM. Height-above-ground is deferred to a later
dataset version once a terrain model is available.

Consequences:
- Adopting height-above-ground later means recomputing `z` and re-voxelizing;
  the offset must be documented in metadata. New `dataset_version`.

Status:
Adopted (M0).

---

## 2026-06-01 - M0/D4 - `observed` / `unobserved` masks are a core M0 deliverable

Context:
The masks distinguishing directly-observed target voxels from never-observed ones
power the M4 unobserved-region metric (the project's headline). They could be
computed in M0 or deferred to M4.

Decision:
**Promote both masks to required M0 deliverables**: derive and store
`observed_mask` and `unobserved_mask` in every `.npz`.
- `observed = coords_target ∈ coords_partial`
- `unobserved = occupied_target ∧ ¬observed`

Reason:
The masks are cheapest and most accurate to compute here, while partial and
target sit on the same grid in one process. Deferring to M4 forces an
after-the-fact intersection where boundary near-misses corrupt the headline
number.

Consequences:
- `02_DATA_CONTRACT.md` moves both masks from Optional to required fields.
- Cheap to add now; expensive to trust if recomputed later.

Status:
Adopted (M0).
