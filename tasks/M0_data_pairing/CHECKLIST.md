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
- [x] **Implement target occupancy** — `pointcraft.data.voxelize_target`
      (+ `load_lod2_meshes`): LOD2 surface **shell** (D2) → `coords_target`
      (int32), `occ_target` (uint8 all-1), `sem_target` (int64, roof=3/facade=4
      by face-normal |n_z|≥0.7, majority vote, ties→roof). Tests in
      `tests/test_target_occupancy.py` (7 passing). Real-LOD2 smoke: 240,646 faces
      → 74,850 shell voxels (roof 33k, facade 41k) in ~18 s. **(M0-3 done)**
- [x] **Save `.npz`** — `pointcraft.data.write_sample_npz` emits all 7 contract
      array fields (correct dtypes) + `metadata` as JSON (numpy.load without
      allow_pickle). `compute_masks` derives required observed/unobserved masks
      (D4); `build_metadata` assembles the metadata block. Tests in
      `tests/test_sample_io.py` (4 passing, end-to-end on fixture). **(M0-4 done)**
- [x] **Write tests** — round-trip (`test_voxel_grid.py`), grid-equality +
      building-footprint/observed-roof alignment + end-to-end run_m0 npz load
      (`test_pairing_alignment.py`). Full suite **28 passing**. **(Phase D done)**
- [x] **Sanity visualization** — `scripts/run_m0.py --viz` saves a 1×3 matplotlib
      figure (partial top-down height, target top-down semantic, vertical slice
      showing shell/partial/UNobserved facade). Eyeballed on fixture + real tile;
      looks right. **(C6 done)**
- [ ] **Document known limitations** — alignment caveats, dropped points, label gaps.
