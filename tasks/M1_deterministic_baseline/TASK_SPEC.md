# M1 — Deterministic Baseline — TASK SPEC

Status: **not started** (legacy pipeline exists; needs wrapping to the contract).

## Goal

Provide a non-learning baseline that turns partial occupancy into a "complete"
prediction via deterministic rules (extrusion / volume fill), output in the
`docs/02_DATA_CONTRACT.md` format and scored with the standard metrics. This sets
the floor that M2+ must beat, especially in unobserved regions.

## Scope

- Reuse the deterministic pipeline now at `src/pointcraft/baseline/` (stages in
  `baseline/stages.py`, driven by `pointcraft.pipeline`).
- Add: naive roof-extrusion fill and/or rule-based building volume fill.
- Emit predictions on the M0 grid; compute occupancy IoU (+ unobserved-region IoU).

## Exclusions

- ❌ No learning. ❌ No new model architectures.

Depends on: **M0** (data contract + a paired sample).
