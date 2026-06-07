# exp_002 — M2 occupancy-completion overfit (09LD1874)

**First learned model.** Single-tile *overfit* of a small sparse-conv U-Net
(`OccupancyCompletionUNet`) — the M2 first-step gate: prove the model +
pipeline can learn by clearly beating the M1 B1 floor under the shared
multi-cutoff metrics. This is **not** a generalizing model (one tile, no
val split); multi-tile is a later M2 phase (blocked on the centroid-crop fix).

## Setup
- tile `tokyo_station` (dataset v0.2), code `0b20a24`
- candidate support (input-only: B1 extrusion ∪ observed) = 4,172,459 voxels; the net classifies occupied/free over it
- model params 79,153; base=8; iters=500; lr=0.001; AMP=True; peak CUDA 6536 MB
- trained on the **full dense support** (the target shell ≈ the boundary of
  the solid support, so dense neighbourhoods are needed to learn it);
  occupancy threshold swept at eval, best prob = 0.8
- scored by `pointcraft.metrics` (same as M1), strict/mid/tolerant cutoffs

## Result — unobserved-region IoU (vs M1 band)

| cutoff | B1 floor | **M2** | B2 ceiling |
|--------|---------|--------|------------|
| strict | 0.061 | **0.817** | 0.359 |
| mid | 0.040 | **0.758** | 0.363 |
| tolerant | 0.039 | **0.805** | 0.379 |

- completion IoU **0.8535** (precision 0.924, recall 0.918)
- per-class recall: ground 0.920 / roof 0.756 / facade 0.956


## Deterministic diagnostic (B3) — is the support answering the question?

`scripts/diagnose_support_shell.py` scores the **morphological shell** of the
candidate support (no training) — see `diagnostic_morph_shell.json`. On this
tile it gets only **strict unobserved IoU ≈ 0.15** (vs the learned 0.82): the
support does **not** trivialise the task — per-column extrusion merges adjacent
buildings into solid blocks that bury the true facades inside the volume, so a
naive surface extraction misses them (recall 0.34) while the model recovers
them (recall 0.92). The learned 0.82 is real work, not support leakage.

See `metrics.json` for full scores + training history; `viz_overfit.png` for a
rough observed→completed→GT slice; `pred_coords.npy` is the prediction
(gitignored artifact).
