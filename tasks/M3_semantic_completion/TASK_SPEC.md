# M3 — Semantic Completion — TASK SPEC

Status: **not started**.

## Goal

Extend M2 with a **semantic head**: predict both complete occupancy and per-voxel
semantic class (`sem_target`). Dual-head model (occupancy + semantics).

## Scope (planned)

- Add semantic decoder/head on the M2 backbone.
- Multi-task loss (occupancy + semantic CE, with `ignore_index`).
- Metrics: semantic mIoU (per-class + mean) alongside completion IoU.

## Exclusions

- ❌ No Minecraft export (M5). ❌ No appearance/material modeling.

Depends on: **M2**.
