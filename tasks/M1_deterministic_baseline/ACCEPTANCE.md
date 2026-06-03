# M1 — Deterministic Baseline — ACCEPTANCE

Status: **DONE** (B1 + B2 on `09LD1874`; shared multi-cutoff metrics).

- [x] Baseline runs on at least one M0 tile and outputs a prediction in the data-contract format.
      (B1 + B2 emit `(N,3)` int32 voxel indices on the sample grid;
      `scripts/run_m1_baseline.py`.)
- [x] Occupancy IoU and unobserved-region IoU computed against the M0 target.
      (Shared `pointcraft.metrics`: completion IoU + unobserved IoU under
      **strict / mid / tolerant** cutoffs, + precision/recall + per-class recall.)
- [x] Results recorded (experiment README / metrics file).
      (`experiments/exp_001_m1_baseline/README.md` + `metrics.json`.)
- [x] Session log updated.

## Extra (EXECUTION_PLAN + M4 alignment)
- [x] Two baselines: **B1** observation-only floor, **B2** footprint-informed upper
      reference (labelled as peeking at the target).
- [x] Metrics module is **shared** (importable by M2+) and **multi-cutoff** (D8),
      unit-tested.
- [x] Floor (B1) and ceiling (B2) stated; ceiling beats floor under all three
      cutoffs (conclusion survives moving the observation line).
- [x] No learning / no torch; boundary-crop blocker untouched (single tile).
