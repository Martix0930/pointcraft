# M1 — Deterministic Baseline — SESSION LOG

## Current status: **NOT STARTED** (baseline code migrated and importable)

The deterministic pipeline code is preserved and now lives under the single
`pointcraft` package:

- stages → `src/pointcraft/baseline/stages.py` (kept cohesive)
- pipeline core (Context/Stage/Pipeline) → `src/pointcraft/pipeline.py`
- LOD2 parsing/rasterizing → `src/pointcraft/data/lod2.py`
- MC block palette → `src/pointcraft/mc_export/palette.py`
- viewer → `src/pointcraft/utils/viewer.py`
- example runner → `scripts/tokyo_station.py`

The former repo-root `pointcraft/` package was removed (it shadowed `src/`).
Imports verified (`pip install -e .`; all submodules import OK). The pipeline is
not yet wrapped to the M0 data contract / metrics — that is the M1 work, blocked
on M0 producing a paired sample.

## Next recommended prompt for Claude Code

> After M0 produces at least one paired `.npz`, read `tasks/M1_deterministic_baseline/`
> and implement a naive roof-extrusion baseline that predicts complete occupancy
> from partial occupancy (reusing `pointcraft.baseline`), then score it (IoU +
> unobserved-region IoU) against the M0 target. No learning. Update this log.

---

## Session entries

### 2026-06-01 — repo cleanup (no M1 implementation)
- Merged the legacy repo-root `pointcraft/` package into `src/pointcraft/` (see
  mapping above) and deleted the root package.
- Rewired internal imports and the three legacy scripts; added `REPO/src` to their
  `sys.path`.
- No deterministic-baseline functionality implemented yet.
