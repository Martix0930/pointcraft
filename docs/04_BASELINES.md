# 04 — Baselines

Baselines give a floor to beat and a sanity check on the data contract. They are
**deterministic** (no learning) for M1; the first learned model arrives in M2.

## M1 — Deterministic baseline

The earlier project phase (legacy `pointcraft/` package at repo root) already
implements a deterministic LiDAR → voxel → output pipeline. It is the reference
M1 baseline: whatever a learned model produces should beat this on completion
metrics, especially in unobserved regions.

Key property: it can only reproduce **observed** structure plus simple geometric
fills — by construction it cannot recover genuinely unobserved facades/volumes.
That gap is exactly what M2–M4 aim to close.

## Naive extrusion baseline

- Take the observed roof footprint per building (from partial occupancy).
- Extrude straight down to the ground/terrain to fill the building volume.
- Pros: trivial, surprisingly strong on boxy buildings.
- Cons: wrong for setbacks, overhangs, sloped/complex massing; no semantics beyond
  "building column".

## Rule-based building volume fill

- Use LOD2 footprint + roof height (where available) to fill a solid volume.
- Assign coarse semantics by geometry (top surface → roof, vertical boundary →
  facade, interior → building).
- This is closer to the **target** construction than to a prediction-from-LiDAR
  baseline; useful as an upper-bound style reference for the deterministic path.

## Future — M2 occupancy model baseline

- First learned baseline: a sparse 3D U-Net (spconv/Minkowski) predicting complete
  occupancy from partial occupancy.
- Report against M1 + naive extrusion on overall IoU and unobserved-region IoU.
- Details deferred to `tasks/M2_occupancy_completion/TASK_SPEC.md`.

## Reporting

All baselines report the same metrics as learned models (see
`03_EXPERIMENT_PROTOCOL.md`) so numbers are directly comparable.
