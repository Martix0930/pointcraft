# M1 — Deterministic Baseline — SESSION LOG

## Current status: **DONE** — B1 + B2 scored on `09LD1874`, multi-cutoff metrics shared

M1 delivers: a **shared multi-cutoff metrics module** (`pointcraft.metrics`,
importable by M2+), two no-NN predictors — **B1** naive roof-extrusion (the
observation-only floor) and **B2** footprint volume fill (footprint-informed upper
reference, peeks at the target) — and a recorded experiment
(`experiments/exp_001_m1_baseline/`). **47 tests passing.** The legacy
`baseline/stages.py` (2.5D height-map + `.schem`) was **not reused** (only the
extrusion idea), per the EXECUTION_PLAN.

**Floor & ceiling (unobserved-region IoU, `09LD1874`):**

| baseline | completion IoU | strict | mid | tolerant |
|----------|---------------|--------|-----|----------|
| B1 floor (obs-only) | 0.0915 | 0.0605 | 0.0397 | 0.0392 |
| B2 ceiling (footprint) | 0.4110 | 0.3590 | 0.3634 | 0.3792 |

The ceiling beats the floor under **all three cutoffs** (conclusion does not flip as
the observation line moves) — the M1↔M4 contract (D8). **M2's number must land
between B1 and B2 under the same cutoffs.** B1 is low by design: naive solid
extrusion has high recall but low precision (over-fills the hollow shell interior)
and cannot recover setback facades — the honest observation-only floor.

## Next recommended prompt for Claude Code (M2)

> Read `CLAUDE.md`, `docs/02_DATA_CONTRACT.md`, the M1 SESSION_LOG, and
> `experiments/exp_001_m1_baseline/`. **M1 is DONE**: it set the floor (B1,
> observation-only) and ceiling (B2, footprint-informed) for occupancy completion,
> scored by the shared `pointcraft.metrics` under strict/mid/tolerant cutoffs. Begin
> **M2** (learning-based occupancy completion) and score it with the **same**
> `pointcraft.metrics.evaluate` + `build_cutoff_masks` so the number is directly
> comparable; M2 must beat B1 (strict unobserved IoU 0.061) and approach B2 (0.359)
> under all three cutoffs. ⚠ Before any **multi-tile** training, first clear the
> **centroid tile-clip M2 BLOCKER** (`docs/02_DATA_CONTRACT.md` alignment rule 4 /
> `docs/07_GOTCHAS.md`) — irrelevant to single-tile M1 but it corrupts multi-tile
> supervision. Torch/spconv/Minkowski enter only here, per the M2 spec.

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

### 2026-06-03 — M1 Phase C–E: B1 + B2 predictors, experiment record, M1 DONE

- **Phase C — B1** `pointcraft.baseline.naive_roof_extrusion` (+ `estimate_ground_k`,
  a vectorised ragged-range `_extrude`): observation-only floor. Per observed
  column, if the top return is `>= ground_k + min_building_height` (3), solid-fill
  ground→top. `ground_k` = 25th-pct of per-column minima (≈17 here, z≈2 m; flat
  tile). Uses **no target**. 4 unit tests.
- **Phase D — B2** `pointcraft.baseline.footprint_volume_fill`: footprint-informed
  upper reference (PEEKS at the target footprint). `shell` mode (default) =
  roof-cap + base on every footprint column + full-height walls on perimeter
  columns (4-neighbour test, grid-border clamped); `solid` mode fills every column
  base→top. 3 unit tests.
- **Phase E** `scripts/run_m1_baseline.py` → `experiments/exp_001_m1_baseline/`
  (`README.md` + `metrics.json`), records both baselines under all three cutoffs,
  code commit, cutoff definitions/fractions.
- **Results (`09LD1874`, unobserved IoU strict/mid/tolerant):**
  B1 floor 0.0605 / 0.0397 / 0.0392 (completion 0.0915; recall 0.93, precision
  0.09 — solid fill over-predicts the hollow shell). B2 ceiling 0.3590 / 0.3634 /
  0.3792 (completion 0.4110; precision 0.85). Ceiling > floor under **all** cutoffs.
- B2-solid was measured too (completion IoU 0.13) and rejected as the ceiling — the
  shell reconstruction (0.41) is the meaningful footprint-informed upper bound.
- Metrics change required for the tolerant cutoff: additive facade `xy_tol` in
  `data.compute_masks` (default 0 = v0.2 unchanged); guarded against ravel
  wrap-around. All 33 M0 tests still pass; **full suite 47 passing.**
- ACCEPTANCE + CHECKLIST ticked. Status → **DONE**. M2 unblocked (handoff prompt
  above); reminder logged that the centroid tile-clip M2 BLOCKER must be cleared
  before multi-tile training.
- Commit sequence: docs(A) → feat(metrics, B) → feat(B1) → feat(B2) →
  feat(experiments) → docs(handoff).
