# M1 — Deterministic Baseline — SESSION LOG

## Current status: **IN PROGRESS** (Phase A–B; metrics module being built first)

M0 is **DONE (v0.2)** and produces a contract `.npz` for `09LD1874`
(`outputs/m0/tokyo_citygml.npz`: target 407,269 voxels, **unobserved 61.0 %**;
observed roof 69.9 % / facade 34.9 % / ground 17.5 %), so M1 is unblocked.

**Plan (per `EXECUTION_PLAN.md`):** M1 is three small no-NN pieces — a shared
**multi-cutoff metrics module** (built first), then **B1** naive roof-extrusion
(the observation-only *floor* M2 must beat) and **B2** rule-based footprint volume
fill (a footprint-informed *upper reference* — it peeks at the target footprint, so
it is not a fair predictor). The legacy `baseline/stages.py` (2.5D height-map +
`.schem`) is **not reused**: M1 borrows only the extrusion idea, working in the M0
sparse-voxel contract instead.

**Multi-cutoff is the M1↔M4 contract (D8).** The metrics entry point takes a *set*
of mask definitions and reports `unobserved_iou` under **strict ~35 % / mid /
tolerant ~67 %** facade-observed cutoffs (D6/D7), so M4's "beats M1" claim has a
comparable baseline under all three. Built before the predictors so each is scored
the moment it exists.

The legacy deterministic code still lives at `src/pointcraft/baseline/stages.py`
(+ `pipeline.py`, `data/lod2.py`, `mc_export/palette.py`, `utils/viewer.py`); it is
kept as reference only.

## Next recommended prompt for Claude Code

> (updated once predictors land)

---

## Session entries

### 2026-06-01 — repo cleanup (no M1 implementation)
- Merged the legacy repo-root `pointcraft/` package into `src/pointcraft/` (see
  mapping above) and deleted the root package.
- Rewired internal imports and the three legacy scripts; added `REPO/src` to their
  `sys.path`.
- No deterministic-baseline functionality implemented yet.

### 2026-06-03 — M1 Phase A: sync docs (status → IN PROGRESS)

- Confirmed M0 `.npz` (`outputs/m0/tokyo_citygml.npz`, v0.2) loads with all contract
  fields; baseline mask stats: target 407,269; unobserved **61.0 %**; observed
  roof 69.9 % / facade 34.9 % / ground 17.5 % (strict cutoff).
- Logged **D8** in `docs/06_DECISIONS.md`: the metrics module is shared (importable
  by M2+) and **multi-cutoff by design** (strict/mid/tolerant), with unobserved-IoU
  computed over the unobserved spatial region (false positives penalised).
- Updated this log's status block to the two-baseline + multi-cutoff plan.
- No `.py` changed in Phase A. Next: **Phase B** — build `src/pointcraft/metrics/`
  (completion IoU, unobserved IoU per cutoff, precision/recall, per-class), unit-test,
  report before writing the predictors.

### 2026-06-03 — M1 Phase B: shared multi-cutoff metrics module (built + tested)

- New `src/pointcraft/metrics/`:
  - `occupancy.py` — `occupancy_scores` (overall IoU/precision/recall), `Scores`
    dataclass, `unobserved_scores` (IoU over the **unobserved spatial region** =
    grid minus partial-input ∪ observed-target, so hallucination in unseen free
    space is penalised, D8), `per_class_recall` (ground/roof/facade).
  - `cutoffs.py` — `build_cutoff_masks` → the three masks **strict / mid /
    tolerant** from `compute_masks` (facade `xy_tol` + `wall_margin` knobs;
    roof/ground keep z±1).
  - `evaluate.py` — `load_sample` (contract `.npz` → arrays + reconstructed grid)
    and `evaluate(pred, sample, cutoffs=…)`: the multi-cutoff entry point that takes
    a **set** of mask definitions and reports completion + per-class recall +
    unobserved IoU per cutoff.
- Additive `xy_tol` (default 0) added to `data.compute_masks` for facades, guarded
  against ravel wrap-around; `xy_tol=0` reproduces v0.2 exactly (all 33 M0 tests
  still pass).
- `tests/test_metrics.py` (7 tests): hand-checked IoU/precision/recall, strict
  cutoff == stored mask, cutoff monotonicity strict≥mid≥tolerant (hand-counted on a
  tiny facade column), hallucination → false positive, per-class recall, end-to-end
  `evaluate`. **Full suite 40 passing.**
- Real `09LD1874` cutoff check: strict 61.0 % unobserved / facade-obs 34.9 %
  (== stored), mid 40.7 % / 62.9 %, tolerant 37.4 % / 67.4 %. Cutoffs span the
  task→physical line (D7) as intended.
- Next: **Phase C** — B1 naive roof-extrusion predictor, scored under all cutoffs.
