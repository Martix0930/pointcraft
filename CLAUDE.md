# CLAUDE.md — PointCraft project memory

Concise memory for Claude Code sessions. Read this first, then read the active
task's files before writing any code.

## Project goal

**PointCraft** studies **Aerial-to-Embodied Semantic Scene Completion**:
from aerial sparse point clouds (which mostly observe roofs, ground, and
vegetation tops), complete **full urban voxel occupancy + semantic labels**,
then instantiate the result as an embodied interactive environment (Minecraft).

## Milestones

- **M0** Data pairing — LiDAR + LOD2/CityGML/mesh → paired voxel training samples. ← current focus
- **M1** Deterministic baseline (legacy pipeline, see below).
- **M2** Learning-based occupancy completion.
- **M3** Semantic completion (occupancy + semantic heads).
- **M4** Unobserved facade / volume completion evaluation.
- **M5** Minecraft / embodied demo.

## Repository as shared memory

This repo is the **single source of truth** shared between Claude App (research
planning) and Claude Code (implementation). Chat history is NOT the source of
truth — the docs and task specs are. Future sessions must be able to work from
small task specs, not long chat logs.

## Where things live

- `docs/` — stable project knowledge (brief, research question, data contract, etc.).
- `tasks/M*/` — per-milestone work units: `TASK_SPEC.md`, `ACCEPTANCE.md`, `CHECKLIST.md`, `SESSION_LOG.md`.
- `src/pointcraft/` — **new research code** (data, voxelization, models, losses, metrics, mc_export, utils).
- `pointcraft/` (repo root, legacy) — the **M1 deterministic pipeline** from the
  earlier phase. Treat it as the baseline reference. Do not refactor it unless a
  task explicitly says so. (Naming overlap with `src/pointcraft` is known and logged in `docs/06_DECISIONS.md`.)
- `scripts/`, `configs/`, `tests/`, `experiments/`, `outputs/` — see each folder's README.

## Rules for Claude Code

1. **Read the active task's `TASK_SPEC.md`, `ACCEPTANCE.md`, `CHECKLIST.md` before coding.**
2. **Update `SESSION_LOG.md`** in the active task folder at the end of every session
   (what you did, decisions, current status, next recommended prompt).
3. **Do not redesign the whole project** or restructure the repo unless explicitly asked.
4. **Do not implement neural networks during M0.** No torch / spconv / Minkowski
   work until M2 and only when a task spec calls for it.
5. **Do not modify unrelated files.** Stay within the scope of the active task;
   respect each task's "Scope exclusions".
6. **Log non-trivial decisions** in `docs/06_DECISIONS.md`.
7. Keep dependencies minimal; heavy libs (torch, spconv, MinkowskiEngine, Open3D,
   PDAL) are optional and introduced only when their milestone needs them.

## Current status

- Research direction locked (Aerial-to-Embodied Semantic Scene Completion).
- Repository scaffolding created.
- **Next: M0 data pairing.** See `tasks/M0_data_pairing/`.
