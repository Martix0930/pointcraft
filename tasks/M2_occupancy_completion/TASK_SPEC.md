# M2 — Occupancy Completion — TASK SPEC

Status: **not started**.

## Goal

First **learning-based** model: predict **complete voxel occupancy** from partial
occupancy. Establishes the learned pipeline (data loading → sparse 3D backbone →
occupancy head → loss → metrics).

## Scope (planned)

- Sparse 3D backbone (spconv or Minkowski) — a U-Net style occupancy completer.
- Input: M0 `coords_partial` + `feats_partial`. Target: `occ_target`.
- Loss: occupancy classification (e.g. BCE / focal); metric: completion IoU +
  unobserved-region IoU.
- First introduction of the learning stack (torch + sparse conv lib) — add to
  optional deps per `pyproject.toml`.

## Exclusions (this milestone)

- ❌ No semantic head yet (that's M3).
- ❌ No Minecraft export.

Depends on: **M0** (data), benchmarks against **M1** (baseline).
