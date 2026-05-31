# 03 — Experiment Protocol (placeholder)

Conventions for running and recording experiments. Fill in details as M2+ begins.

## Experiment folder naming

- Location: `experiments/`.
- Naming: `exp_NNN_short_slug/` where `NNN` is a zero-padded incrementing id
  (e.g. `exp_000_smoke_test`, `exp_001_occ_unet_baseline`).
- Each experiment folder contains: a `README.md` (what/why), the resolved config
  used, and an `outputs/` subdir (gitignored) for artifacts/logs/checkpoints.

## Config naming

- Location: `configs/`.
- Naming: `<milestone>_<model>_<variant>.yaml` (e.g. `m2_occ_unet_v0.yaml`).
- Configs are declarative and versioned; an experiment records the exact config
  (copy or hash) it ran with.

## Required log files (per experiment)

- `README.md` — purpose, hypothesis, config reference, result summary, conclusion.
- `metrics.json` (or csv) — final + per-epoch metrics.
- training log (stdout capture) and, if used, tensorboard/wandb run id.
- `SESSION_LOG.md` reference in the relevant `tasks/M*/` folder.

## Metrics to record

> Final taxonomy depends on milestone; baseline set:

- **Occupancy completion IoU** (overall).
- **Semantic mIoU** (per-class + mean), once M3.
- **Unobserved-region metrics** — IoU / accuracy restricted to `unobserved_mask`
  (the headline metric for M4).
- Precision / recall for occupancy.
- Optional: Chamfer / surface distance for geometry comparison vs. LOD2.

## Reproducibility notes

- Record: random seed, dataset version (`dataset_version`), config hash, code
  commit hash, library versions, hardware.
- Determinism: set seeds for numpy/torch; note any nondeterministic ops.
- One experiment = one config + one data version; do not mutate a finished
  experiment's config — create a new experiment instead.
