# M0 — Data Pairing — ACCEPTANCE

The milestone is accepted when ALL of the following hold:

- [ ] **End-to-end on one small tile** — a single tile can be processed from raw
      LiDAR + LOD2 to a written `.npz` without manual intervention.
- [ ] **Shared grid** — `coords_partial` and `coords_target` use the same `origin`,
      `voxel_size`, and `grid_shape` (verified by a test).
- [ ] **Independently loadable** — the `.npz` loads with `numpy.load` and exposes
      all fields in `docs/02_DATA_CONTRACT.md` with the specified dtypes/shapes.
- [ ] **Complete metadata** — `tile_id`, `voxel_size`, `origin`, `bounds`,
      `grid_shape`, `crs`, `source_files`, `dataset_version`, `feature_layout` all present.
- [ ] **Basic tests exist and pass** — at minimum: world↔index round-trip and
      partial/target grid-equality; ideally the building-footprint alignment check.
- [ ] **Sanity visualization** — at least one debug view/export of a produced sample.
- [ ] **Session log updated** — `SESSION_LOG.md` reflects final status + next prompt.
- [ ] **Scope respected** — no NN, no spconv/torch, no semantic learning.

## Nice-to-have (not blocking)

- `observed_mask` / `unobserved_mask` stored.
- A `scripts/run_m0.py` reproducing the sample from config.
- Known-limitations section written in `CHECKLIST.md` / `SESSION_LOG.md`.
