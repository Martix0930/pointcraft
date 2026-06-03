# exp_001 — M1 deterministic baseline (tokyo_station)

Floor (B1, observation only) and ceiling (B2, footprint-informed upper
reference — **it peeks at the target footprint, not a fair predictor**) for
occupancy completion on the M0 contract sample. Both are no-NN. Reproduce:

```
python scripts/run_m1_baseline.py
```

- dataset_version: `v0.2`  ·  grid `[400, 300, 222]`  ·  code `fd99f93`
- unobserved fraction per cutoff: strict 61.0%, mid 40.7%, tolerant 37.4%

## Results — IoU

| baseline | completion | unobserved (strict) | unobserved (mid) | unobserved (tolerant) |
|----------|-----------|---------------------|------------------|-----------------------|
| B1 floor | 0.0915 | 0.0605 | 0.0397 | 0.0392 |
| B2 ceiling | 0.4110 | 0.3590 | 0.3634 | 0.3792 |

## Reading the numbers

- **B1 is the honest floor.** Naive solid extrusion recovers facade only
  under the roof footprint; its unobserved IoU is **low** (strict 0.061) — high recall, low precision (it over-fills the hollow shell interior). A low B1 is a *successful* B1.
- **B2 is the ceiling**, not a competitor: knowing the footprint + height,
  a shell reconstruction reaches strict unobserved IoU 0.359.
- **M2 must land between these** under the *same* cutoffs. The ceiling beats
  the floor under all three cutoffs (the conclusion does not flip as the
  observation line moves), which is the M1↔M4 contract (D8).

## Per-class recall (target side)

| baseline | ground (1) | roof (3) | facade (4) |
|----------|-----------|----------|------------|
| B1 floor | 0.918 | 0.767 | 0.968 |
| B2 ceiling | 0.977 | 0.573 | 0.319 |
