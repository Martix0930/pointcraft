# M4 — Unobserved Facade / Volume Completion — TASK SPEC

Status: **not started**.

## Goal

Make the **headline contribution** measurable: evaluate (and improve) how well the
model completes structure that was **never observed** — primarily facades and
building volume — using `unobserved_mask`.

## Scope (planned)

- Dedicated evaluation restricted to unobserved regions (IoU/precision/recall, per
  class for facade/volume).
- **Multi-definition mask evaluation (required, D6/D7):** compute the
  unobserved-completion metric under ≥3 `observed`/`unobserved` definitions
  (strict ≈35 % facade / mid-wall / tolerant ≈67 %) so the headline "beats M1"
  result can be shown invariant to where the observed line is drawn (see the M4
  ACCEPTANCE item and `docs/02_DATA_CONTRACT.md`).
- Possibly add generative/completion-oriented components (e.g. completion priors,
  diffusion-style refinement) if plain M2/M3 under-performs on unobserved regions.
- Ablations: with/without features, observed-only vs. full supervision.

## Exclusions

- ❌ No Minecraft export (M5).

Depends on: **M2/M3** + M0 masks (`observed_mask` / `unobserved_mask`).
