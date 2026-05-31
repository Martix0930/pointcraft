# tests/

Automated tests (pytest). Run with `pytest` from the repo root.

Tests should use the tiny committed fixtures in **`test_data/`** (not ad-hoc local
paths). See `test_data/README.md` and `CLAUDE.md` → "Test data policy".

- `test_data/m0_voxel_grid/` — points + hand-computed expected voxel indices.
- `test_data/m0_data_pairing/` — tiny aerial LiDAR + cube LOD2 + expected metadata.

M0 priorities:
- world↔voxel-index round-trip correctness,
- partial/target shared-grid equality,
- LiDAR↔LOD2 alignment regression (building footprint vs. observed roof).
