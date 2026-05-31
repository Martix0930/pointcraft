# PointCraft

**Aerial-to-Embodied Semantic Scene Completion.**

PointCraft studies how to take **aerial sparse point clouds** — which mostly
observe roofs, ground, and vegetation tops — and **complete the full urban 3D
structure** as voxel occupancy plus semantic labels, then instantiate the result
as an **embodied interactive environment** such as Minecraft.

## Research motivation

Airborne LiDAR systematically *under-observes* the world: it captures roofs and
open ground well, but facades, building volumes, and occluded structure are
largely missing. Deterministic conversion (rasterize what you see) is therefore
capped by the data — it cannot recover what was never observed.

We reframe this limitation as the research problem: **given partial top-down
observation, can a model infer and complete the full structure?** LOD2 / CityGML
city models provide complete building geometry as supervision. Minecraft serves
as a discrete voxel world that is both a natural output space and a simulatable
embodied environment.

This sits at the intersection of semantic scene completion, aerial building
reconstruction, and embodied AI.

## Milestones

| ID | Title | Summary |
|----|-------|---------|
| M0 | Data pairing | LiDAR + LOD2/CityGML/mesh → paired voxel training samples. |
| M1 | Deterministic baseline | Rule-based extrusion / volume fill (legacy pipeline). |
| M2 | Occupancy completion | Learning-based partial → complete occupancy. |
| M3 | Semantic completion | Add per-voxel semantic labels (occupancy + semantic heads). |
| M4 | Unobserved facade completion | Evaluate completion of never-observed facades/volume. |
| M5 | Minecraft / embodied demo | Instantiate completed scene as an interactive world. |

## Repository layout

```
pointcraft/
├─ CLAUDE.md              # project memory for Claude Code
├─ README.md              # this file
├─ pyproject.toml         # minimal Python project config
├─ docs/                  # stable project knowledge (numbered)
├─ tasks/M0..M5/          # per-milestone task specs + session logs
├─ src/pointcraft/        # the only importable package (data, voxelization, models, ...)
│  └─ baseline/           # legacy deterministic pipeline = M1 baseline
├─ scripts/               # entry-point scripts
├─ configs/               # experiment configs
├─ tests/                 # tests
├─ experiments/           # experiment runs
└─ outputs/               # generated artifacts (gitignored)
```

> Note: there is a single importable package, `pointcraft`, living in `src/`. The
> original deterministic pipeline has been merged into `src/pointcraft/baseline/`
> and serves as the **M1 deterministic baseline**. (Naming convention and the
> merge are logged in `docs/06_DECISIONS.md`.) Install with `pip install -e .`.

## Working with Claude App and Claude Code

- **Claude App** — research planning and decisions. Reads/writes `docs/` and
  creates/updates task specs in `tasks/`.
- **Claude Code** — implementation, testing, session handoff. Reads the active
  task's spec and updates that task's `SESSION_LOG.md`.
- The **repository is the shared memory.** Decisions live in
  `docs/06_DECISIONS.md`; progress lives in each task's `SESSION_LOG.md`. Long
  chat history is not relied upon.

## How future sessions should proceed

1. Read `CLAUDE.md`.
2. Open the active milestone folder (start with `tasks/M0_data_pairing/`).
3. Read `TASK_SPEC.md`, `ACCEPTANCE.md`, `CHECKLIST.md`.
4. Do the smallest useful unit of work within scope.
5. Update `SESSION_LOG.md` and, if a decision was made, `docs/06_DECISIONS.md`.

## Status

Scaffolding complete. **Next: M0 data pairing.**

## Data sources

Example data is Japanese open data: airborne LiDAR (LAS, EPSG:6677) +
[PLATEAU](https://www.mlit.go.jp/plateau/) LOD2 buildings (CC-BY 4.0).

## License

MIT
