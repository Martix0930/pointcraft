# M1 — Deterministic Baseline — CHECKLIST

Status: **DONE**.

- [x] Load an M0 paired sample. (`pointcraft.metrics.load_sample`)
- [x] Build the shared, multi-cutoff metrics module first. (`src/pointcraft/metrics/`)
- [x] Implement naive roof-extrusion fill (B1). (`baseline.naive_roof_extrusion`)
- [x] Implement rule-based LOD2 volume fill (B2). (`baseline.footprint_volume_fill`)
- [x] Emit predictions in data-contract format.
- [x] Compute IoU + unobserved-region IoU (strict/mid/tolerant) + per-class recall.
- [x] Record results; update session log. (`experiments/exp_001_m1_baseline/`)
