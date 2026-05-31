# M0 — Data Pairing — CHECKLIST

Work through in order. Check off as completed; keep notes inline.

- [ ] **Inspect existing data** — confirm LiDAR + LOD2 files, CRS, bounds, units;
      note what the legacy `pointcraft/` package already provides for reuse.
- [ ] **Define coordinate system** — fix world axes, `origin`, `voxel_size`, and
      the z-reference (absolute vs. height-above-ground); record in data contract if changed.
- [ ] **Implement voxel grid** — grid object + world↔index transforms (shared by
      partial and target).
- [ ] **Implement partial occupancy** — LiDAR → `coords_partial` + `feats_partial`
      with the documented feature layout.
- [ ] **Implement target occupancy** — LOD2/mesh → `coords_target`, `occ_target`,
      `sem_target` on the same grid.
- [ ] **Save `.npz`** — writer emitting all contract fields + metadata.
- [ ] **Write tests** — round-trip, grid-equality, alignment regression.
- [ ] **Sanity visualization** — eyeball one sample.
- [ ] **Document known limitations** — alignment caveats, dropped points, label gaps.
