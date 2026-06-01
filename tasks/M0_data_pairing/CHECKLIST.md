# M0 — Data Pairing — CHECKLIST

Work through in order. Check off as completed; keep notes inline.

- [x] **Inspect existing data** — confirmed LAS `09LD1874.las` (1.75M pts,
      EPSG:6677, ~239×231 m, cls {1,2,3}, no building class) + 2 LOD2 tiles
      (EPSG:6677, larger extent → clip to LAS). Reuse map noted in SESSION_LOG.
      **(Phase B done)**
- [x] **Define coordinate system** — absolute-z (D3), shared origin, `voxel_size`
      1.0 m (D1); contract finalized. z-datum verified aligned (median Δ≈0.3 m,
      roof p90 Δ≈0.06 m). **(Phase B done)**
- [x] **Implement voxel grid** — `pointcraft.voxelization.VoxelGrid`
      (`from_bounds`, `world_to_index`, `index_to_center`/`_corner`, `in_bounds`);
      tests in `tests/test_voxel_grid.py` (7 passing, fixture-driven). **(M0-1 done)**
- [x] **Implement partial occupancy** — `pointcraft.data.voxelize_partial`
      (+ `load_las_xyz`): LiDAR → `coords_partial` (int32 [N,3]) + `feats_partial`
      (float32 [N,2], layout `["height","point_count"]`); drops out-of-range
      (logged), merges duplicates. Tests in `tests/test_partial_occupancy.py`
      (7 passing, fixture-driven). Real-LAS smoke: 1.75M pts → 96,264 voxels.
      **(M0-2 done)**
- [ ] **Implement target occupancy** — LOD2/mesh → `coords_target`, `occ_target`,
      `sem_target` on the same grid.
- [ ] **Save `.npz`** — writer emitting all contract fields + metadata.
- [ ] **Write tests** — round-trip, grid-equality, alignment regression.
- [ ] **Sanity visualization** — eyeball one sample.
- [ ] **Document known limitations** — alignment caveats, dropped points, label gaps.
