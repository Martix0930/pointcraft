# M4 — Unobserved Facade / Volume Completion — ACCEPTANCE

Status: **not started**.

- [ ] Unobserved-region metric implemented (restricted to `unobserved_mask`).
- [ ] Reported separately from overall metrics for occupancy and semantics.
- [ ] At least one ablation comparing approaches on unobserved-region performance.
- [ ] Clear improvement over M1 baseline on unobserved regions documented.
- [ ] **Mask-definition sensitivity (REQUIRED, not optional — D6/D7).** The
      unobserved-region completion metric is reported under **at least three mask
      definitions** — strict (facade ≈ 35 %) / mid-wall / tolerant (facade ≈ 67 %) —
      and the **"beats M1 baseline" conclusion is shown to hold under all three**
      (it must not flip as the observed line moves). The `observed` line is a
      task-design choice (see `docs/02_DATA_CONTRACT.md`), so this is the proof that
      the headline result is real, not a masking artifact. If the conclusion flips
      under any definition, surface it — do not hide it.
- [ ] Experiment recorded; session log updated.
