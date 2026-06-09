# CLAUDE.md — PointCraft project memory

Concise memory for Claude Code sessions. Read this first, then read the active
task's files before writing any code.

## Project goal

**PointCraft** studies **Aerial-to-Embodied Semantic Scene Completion**:
from aerial sparse point clouds (which mostly observe roofs, ground, and
vegetation tops), complete **full urban voxel occupancy + semantic labels**,
then instantiate the result as an embodied interactive environment (Minecraft).

## Milestones

M0 data pairing -> M1 deterministic baseline -> M2 occupancy completion ->
M3 semantic completion -> M4 unobserved completion evaluation -> M5 embodied demo.

## Repository as shared memory

This repo is the **single source of truth** shared between Claude Chat (research
planning) and Claude Code (implementation). Chat history is NOT the source of
truth — the docs and task specs are. Future sessions must be able to work from
small task specs, not long chat logs.

## Where things live

- `docs/` — stable project knowledge (brief, research question, data contract, etc.).
- `tasks/M*/` — per-milestone work units: `TASK_SPEC.md`, `ACCEPTANCE.md`, `CHECKLIST.md`, `SESSION_LOG.md`.
- `src/pointcraft/` — **the only importable `pointcraft` package** (data, voxelization, models, losses, metrics, mc_export, utils). Install with `pip install -e .`.
- `src/pointcraft/baseline/` — the **M1 deterministic pipeline** (merged from the
  former repo-root `pointcraft/` package). Treat it as the baseline reference; do
  not refactor it unless a task explicitly says so.
- `src/pointcraft/pipeline.py` — Context / Stage / Pipeline core (was `context.py`).
- `data/raw/` — **all large local datasets** (git-ignored): `lidar/` (60 LAS),
  `lod2/` (M0 target OBJ), `dem/`, `citygml/` (future high-accuracy target). The
  old `三维GIS/` and `Download/` paths are gone — see `data/raw/README.md` and the
  Data-location gotcha in `docs/07_GOTCHAS.md`. Configs use paths relative to the
  config file.
- `scripts/`, `configs/`, `tests/`, `experiments/`, `outputs/` — see each folder's README.

> Naming: code/package/import/path = lowercase `pointcraft`; human-facing prose = `PointCraft`. Never rename the package dir to `src/PointCraft/`. See `docs/07_GOTCHAS.md`.

## Rules for Claude Code

1. Read the active task's `TASK_SPEC.md`, `ACCEPTANCE.md`, `CHECKLIST.md`, and
   `SESSION_LOG.md` before coding.
2. Update that task's `SESSION_LOG.md` after each implementation session.
3. Log non-trivial decisions in `docs/06_DECISIONS.md`.
4. Stay within the active task scope; do not redesign the repo or touch unrelated
   files unless explicitly asked.
5. Keep dependencies minimal and milestone-gated.

## Test data policy

`test_data/` is for tiny committed fixtures only. Large raw datasets, generated
samples, checkpoints, and real LiDAR/PLATEAU/CityGML/OBJ/NPZ artifacts must stay
ignored. New fixtures should be minimal, synthetic or heavily reduced, and include
a short README describing coordinate system, voxel size / bounds, and tests.

## Current status / active task pointer

This section is a lightweight pointer only. If it conflicts with any
`tasks/M*/SESSION_LOG.md`, trust the task log.

- **M0 data pairing:** DONE. See `tasks/M0_data_pairing/SESSION_LOG.md`.
- **M1 deterministic baseline:** DONE. See
  `tasks/M1_deterministic_baseline/SESSION_LOG.md`.
- **M2 occupancy completion:** first-step DONE (single-tile overfit). See
  `tasks/M2_occupancy_completion/SESSION_LOG.md`.
- **M3 semantic completion:** not started.
- **M4 unobserved facade / volume evaluation:** not started.
- **M5 Minecraft / embodied demo:** not started.

## Active task rule

No implementation task is active by default. For research discussion, read the
relevant docs and task logs before making strategy claims. For coding, the user
or the task spec must identify the active milestone/task; then read that task's
`TASK_SPEC.md`, `ACCEPTANCE.md`, `CHECKLIST.md`, and `SESSION_LOG.md` before
editing.
