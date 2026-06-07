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

---

## 2026-06-03 - M0/D5 - CityGML replaces OBJ as the M0 target source

Context:
The OBJ-based M0 pipeline (v0.1, DONE) derived building semantics from face
orientation (`|n_z| >= 0.7` → roof, else facade). The alignment audit
(2026-06-01, see `tasks/M0_data_pairing/SESSION_LOG.md`) confirmed two problems
this causes: (a) near-horizontal building *bottom* faces are mislabelled roof;
(b) there is no ground/road/vegetation class because the OBJ is building-only.
The downloaded CityGML carries explicit LOD2 surface types
(`bldg:RoofSurface` / `bldg:WallSurface` / `bldg:GroundSurface`), i.e. the
semantics we were heuristically (and sometimes wrongly) inferring.

Decision:
Use **CityGML** as the M0 target source; read roof / wall / ground semantics
directly from CityGML surface types instead of inferring them from geometry.
OBJ is retained only as a fallback / comparison if CityGML integration stalls.

Reason:
- Removes the geometry-heuristic mislabelling (building base → roof) seen under
  OBJ; gives true roof/facade/ground labels at the source.
- Unlocks a `ground`(1) class for the target (OBJ had none).

Consequences:
- Requires a new GML parser (`src/pointcraft/data/citygml.py`) and a
  **EPSG:6697 (lat/lon) → EPSG:6677** reprojection before alignment with the
  LiDAR (CityGML is delivered in 6697; LiDAR is native 6677). Reprojection is
  horizontal-only — z (absolute elevation, D3) must pass through unchanged and be
  re-verified on a real building.
- `02_DATA_CONTRACT.md`: target `source_files` become the CityGML grid file(s);
  semantic label table maps to CityGML surface types
  (RoofSurface→roof 3, WallSurface→facade 4, GroundSurface→ground 1).
- The shell representation (D2) and label `2` (building solid interior) being
  unused are unchanged.
- `dataset_version` stays `v0.1` for now (field set + feature layout unchanged);
  bump it only if the on-disk schema or feature layout changes.

Status:
Adopted (M0). First tile to re-verify: **`09LD1874`** (Tokyo-Station core, the
tile `configs/tokyo_station.yaml` already targets; grids 53394610/611/620/621,
~2061 CityGML surfaces, LiDAR 6.39 M pts). The originally-planned `09LD1848` was
**dropped at the Phase B gate**: although `tile_alignment.csv` flags it
`in_lod2_citygml=1`, that flag is only a bbox overlap and **0** CityGML buildings
actually fall inside its footprint (see `docs/07_GOTCHAS.md`). 09LD1874 also lets
the CityGML target be compared directly against the prior OBJ run on the same tile.

---

## 2026-06-03 - Correction: legacy root `pointcraft/` merge is executed, not pending

Context:
The 2026-06-01 decision "Keep legacy pipeline as M1 baseline under repo-root
`pointcraft/`" left open the possibility of a separate root package. The
EXECUTION_PLAN v2 asks for an explicit confirmation that this is resolved.

Decision / correction:
There is **no repo-root `pointcraft/`**. The legacy pipeline has been merged into
`src/pointcraft/` and the root package deleted (see the 2026-06-01 "Merge legacy
pipeline into `src/pointcraft/baseline`" decision, which supersedes the "keep
legacy as M1 baseline" entry). `src/pointcraft/` is the **only** importable
`pointcraft`. New M0 CityGML code lives in `src/pointcraft/data/`, not in a
recreated root package.

Status:
Confirmed (no action; documents the executed state to prevent re-introducing a
root package).

---

## 2026-06-03 - M0/D6 - Class-aware observed mask (z-tolerance + genuine mid-wall); dataset_version v0.2

Context:
The v0.1 masks defined `observed = coords_target ∈ coords_partial` by **exact**
`(i,j,k)`. On the real tile this flagged ~55 % of roof voxels "unobserved" even
though aerial LiDAR clearly sees roofs: a roof surface sampled at k=40 and a roof
LiDAR point at k=39 differ by <1 m but miss under exact matching (sub-voxel
z-quantization straddle). Conversely, a quantitative check showed this aerial
LiDAR has **substantial oblique facade returns** — exact matching already marks
38 % of facade observed, and the *genuine mid-wall* signal (excluding ground/roof
points that merely clip the bottom/top wall voxel) is ~30 %. The completion task
must neither (a) reward "completing" roofs that were actually seen, nor (b) be made
artificially impossible by pretending zero facade was observed.

Decision:
`compute_masks` becomes **class-aware** (`sem_target` required for the new rule;
`None` falls back to the v0.1 exact rule):
- **Horizontal surfaces** (roof 3 / ground 1): observed if a partial voxel exists
  at the **same `(i,j)` within `|Δk| ≤ z_tol`** (default `z_tol=1`). Corrects the
  z-quantization straddle.
- **Vertical surfaces** (facade 4): observed only on an **exact** partial hit that
  is a **genuine mid-wall** cell — `≥ wall_margin` voxels above the column's lowest
  target voxel and below its highest (default `wall_margin=2`) — so ground/roof
  points clipping the base/top wall voxel don't count as "facade observed".
- `unobserved = occupied_target ∧ ¬observed` unchanged.

Reason:
Targets the genuine geometry artifact (horizontal z-straddle) while keeping the
facade completion region honest. On 09LD1874 this yields roof 70 % / facade ~35 % /
ground 18 % observed (total unobserved 61 %), matching the intended "facades carry
real but sparse aerial signal (~30 %), most still to complete" research framing.

Consequences:
- **Mask definition changed → `dataset_version` bumped to `v0.2`** (the on-disk
  field set/feature layout are unchanged, but the mask *values* differ from v0.1;
  the version distinguishes them). `02_DATA_CONTRACT.md` updated.
- `z_tol` / `wall_margin` are tunable parameters (facade-observed ≈ 35 % at
  `wall_margin=2`; raise it to push toward 30 %). Defaults pinned for v0.2.
- **Research note (carry to M4):** this LiDAR observes facades more than the
  "roofs-only" premise assumed; the unobserved region is genuinely ~facades+under-
  building ground, not "all walls". Revisit per-class metric weighting at M4.

Status:
Adopted (M0, v0.2).

---

## 2026-06-03 - M0/D7 - `observed`/`unobserved` is a task-oriented line, not physical observation

Context:
D6 fixed the roof z-quantization but left facades exact. A straddle test on
09LD1874 showed this is not a neutral choice: the LiDAR physically grazes **~67 %
of facade within 1 m** (facade observed 38 % → 67 % under an XY ±1 tolerance — the
same magnitude as the roof z-straddle that D6 corrected). So treating roofs with a
tolerance while keeping facades exact is partly an artifact fix (roofs) and partly a
**deliberate task-design choice** (facades). This must be stated explicitly so the
M4 headline isn't mistaken for a physical-observation measurement.

Decision:
- **Input keeps partially-visible facades.** `coords_partial` contains every LiDAR
  voxel, including the real (sparse) facade returns; we do not strip them.
- **`observed` / `unobserved` is a task-oriented definition.** Facade is counted
  observed only on a genuine exact mid-wall hit (~35 %), even though ~67 % is
  physically grazed; no XY tolerance is applied to facades. This intentionally
  carves out an unobserved region so completion is a generative task, not a copy of
  the input.
- The chosen line is **documented as such** in `02_DATA_CONTRACT.md`, and its
  robustness is made a hard M4 acceptance item (multi-definition sensitivity).

Reason:
- Without a preserved unobserved region the task degenerates into reproducing the
  input (the model "completes" what it already sees).
- With facades forced to 0 % observed the task would be needlessly impossible
  (pure roof→facade hallucination); ~35 % keeps it learnable-but-meaningful.
- Honesty: the headline depends on where the observed line sits, so the line is
  declared a design choice and M4 must prove the conclusion survives moving it.

Key fact to record (do not lose):
**facade observed ≈ 35 % (task) vs ≈ 67 % (physical, ±1 voxel).**

Consequences:
- M4 ACCEPTANCE gains a required multi-mask-definition sensitivity check
  (strict ~35 % / mid-wall / tolerant ~67 %); M4 TASK_SPEC scope includes
  implementing multi-definition mask evaluation.
- If a future version wants a physically-honest split, switch to a consistent
  all-class tolerance and bump `dataset_version` (would shrink the unobserved
  region toward genuine occlusion only).

Status:
Adopted (M0). Extends D6.

## 2026-06-03 - M1/D8 - Metrics module is shared and multi-cutoff by design

Context:
M1 needs occupancy + unobserved-region metrics. The same numbers must be directly
comparable across M2/M3/M4, and the M4 headline requires reporting the
unobserved-completion metric under **three mask cutoffs** (strict ~35 % / mid /
tolerant ~67 % facade-observed; see D6/D7) and showing "beats M1 baseline" survives
moving the observation line. If M1's metrics are built around the single stored
`unobserved_mask`, M4 has no comparable baseline and would force a redo.

Decision:
- The metrics module (`src/pointcraft/metrics/`) is **shared**, not M1-private:
  M2/M3/M4 import the same functions so numbers are computed identically.
- Its evaluation entry point takes **a set of mask definitions** (a dict of
  `{cutoff_name: unobserved_mask}`) and reports `unobserved_iou` once per cutoff,
  rather than a single fixed mask. `completion_iou`, precision/recall, and per-class
  breakdown are reported alongside.
- A `build_cutoff_masks` helper generates the strict/mid/tolerant masks from
  `coords_target / coords_partial / sem_target` so any milestone can reproduce the
  three cutoffs from a contract sample. The strict cutoff reproduces the stored
  v0.2 `unobserved_mask`; mid/tolerant relax the facade rule (XY tolerance / drop
  the mid-wall requirement) toward the ~67 % physical-grazing line (D7).
- Unobserved-region IoU is computed over the **unobserved spatial region**
  (voxels neither in the partial input nor flagged observed), so a predictor that
  hallucinates occupancy in unseen free space is penalised (false positives count),
  not just scored on recall.

Reason:
- Building multi-cutoff from day one is the one place M1 is easy to make too narrow
  and force an M4 redo (per the EXECUTION_PLAN).
- A shared module keeps M1's floor and M2+'s numbers on the same definition.

Consequences:
- `compute_masks` gains an additive, default-off `xy_tol` for facades so the
  tolerant cutoff is expressible without changing v0.2 stored-mask behaviour.
- M4 imports `build_cutoff_masks` + the evaluation entry point unchanged.

Status:
Adopted (M1).

## 2026-06-07 - M2/D9 - Backbone = spconv; output stays explicit/semantic-ready/queryable

Context:
M2 is the first learned model: partial occupancy → completed occupancy. Two things
must be decided before any network code — the sparse-conv backbone (and its code
references), and the **output representation**, which is the single non-obvious
requirement that makes M3/M4/M5 possible.

Decision:
- **Backbone = spconv.** The viable SSC code references (SCPNet CVPR'23,
  JS3C-Net AAAI'21) use it and the ecosystem is more mature than MinkowskiEngine
  here. MinkowskiEngine is the **fallback only if spconv won't install** (decided in
  Phase 0, not after building a model).
- **Code-reference re-classification** (see 05_RELATED_WORK / roadmap §2–3):
  - **S3CNet** = architecture-idea reference only (no reliable official code) — not a
    skeleton to clone.
  - **SCPNet / JS3C-Net** = the actual spconv code references (read for structure;
    pinned to old spconv/CUDA, not run as-is).
  - **GeoSVR** = conceptual reference only (image→differentiable-render, per-scene
    optimization; not supervised completion; code not reusable) + active same-lab
    line to watch — never imported.
- **Output-representation constraint (the important one).** The network output is
  **not** a black box that only yields an IoU. It MUST be:
  - emitted in the **data-contract coords format** (same grid as `coords_target`),
  - a per-voxel logit/occupancy you can **threshold** (not a metadata-less dense
    boolean tensor),
  - **world-placeable** — the voxel→world (EPSG:6677) transform stays intact so any
    prediction can be put back in space and later carry a semantic/identity label.
  This keeps the output **explicit, semantic-ready, and queryable** for M3 (semantic
  head added without rework), M4 (observed→completed→GT unobserved-region viz), and
  M5 (agent-queryable occupied/free/unknown + per-voxel identity).

Reason:
- spconv is the lowest-risk path with real code references on a modern stack.
- The architecture is standard; the output contract is what carries the project.
  Collapsing to a dense tensor with no grid metadata would silently block M4/M5.

Consequences:
- M2 first step = **single-tile overfit** on `09LD1874`; must beat the M1 B1 floor
  (strict unobserved IoU 0.061) and is scored by the shared `pointcraft.metrics`
  under strict/mid/tolerant. Occupancy head only (semantics = M3) but the head is
  designed so a semantic head can be added later.
- Multi-tile is a **later** M2 phase, blocked on the centroid tile-crop fix
  (alignment rule 4 / GOTCHAS) — does not affect the single-tile step.
- Phase 0 records the exact torch/spconv/CUDA versions that work on this machine.

Status:
Adopted (M2). Phase 0 result recorded in the M2 SESSION_LOG.
