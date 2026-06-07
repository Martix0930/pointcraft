# M2 — Generalization (multi-tile) — TASK SPEC (fork 1)

Status: **planned** (M2 first-step single-tile overfit is DONE; see SESSION_LOG).

The first step proved the model+pipeline can *learn* (overfit `09LD1874`: unobserved
IoU strict 0.817 vs B1 0.061). This phase asks the real question: **does it
generalize** — train on several tiles, beat the deterministic floor **on a held-out
tile**, under the shared multi-cutoff metrics.

## The honesty bar (what "generalizes" means here)

Per the B3 diagnostic (`scripts/diagnose_support_shell.py`), the input-only candidate
support's deterministic morphological shell scores only strict unobserved IoU ≈ 0.15
on `09LD1874`, so the support does **not** trivialise the task. The generalization
bar is therefore, on the **held-out** tile:

- val unobserved IoU (strict/mid/tolerant) **> B1 floor** on that tile, **and**
- val unobserved IoU **> B3 deterministic-shell** on that tile.

The second clause is the real test: it proves the model learned transferable shell
structure, not just "return the support boundary". Re-run the B3 diagnostic on every
tile so this comparison is per-tile, not borrowed from 09LD1874.

## Ordering (decided)

**G0 (ignore-margin) first, then the multi-tile skeleton.** The centroid tile-crop
contamination is a *directional* border-band bias ("visible building roof → taught
empty"), modest in size (~2.2k building columns/tile in a 5-voxel band on 09LD1874,
≈ border ring; most tile-wide "visible-no-target" columns are trees/clutter that are
*correctly* empty for a building-shell target). It is **not** a pipeline blocker —
`run_m0` already cuts per-tile `.npz` — but it is cheap to neutralise and it hits
exactly the metric we report, so fix it before the first multi-tile number. Geometry-overlap
clipping (the "proper" fix) is deferred; the ignore-margin removes the corruption at
a fraction of the cost.

---

## Phases

### G0 — Ignore-margin border band (the cheap blocker fix)
- **Definition.** A voxel is *border-ignored* if
  `min(i, I-1-i, j, J-1-j) < margin` (XY only; z untouched). Default `margin` ≈ 8
  voxels — document and make it a config/CLI knob.
- **Mechanism (no schema break preferred).** Border-ignored voxels are excluded from
  **both** the training loss **and** the metrics (both prediction and target sides),
  so a dropped border building can no longer teach "visible roof = empty", and the
  metric denominator excludes the ambiguous ring. Two implementation options — pick
  in the spec review:
  1. *Compute-at-use* (no `.npz` change): a helper `border_ignore_mask(grid, margin)`
     used by the training label builder (drop those candidates) and by a metrics
     `exclude`/`region` argument. Keeps `dataset_version` unchanged. **Preferred.**
  2. *Stored field*: add an `ignore_mask` to the contract + bump `dataset_version`.
     Only if a stored mask is needed downstream (M4). Heavier.
- **Metrics support.** Extend `pointcraft.metrics` so `evaluate` / `unobserved_scores`
  accept an optional set of **excluded voxel keys** (removed from pred and target
  before scoring), analogous to the existing observed-exclusion in `unobserved_scores`.
  Unit-test that excluding the band changes the denominator as expected.
- **Re-run B3** on 09LD1874 with the band excluded; confirm it stays low (sanity).
- ❗ Do **not** silently change the single-tile overfit number — report it both with
  and without the band so the comparison to the first-step result is legible.

### G1 — Multi-tile dataset
- Pick **K train tiles + ≥1 held-out val tile** from `data/raw/` with **verified
  CityGML coverage** (use the dense, ~100%-coverage tiles found in M0 Phase E:
  `09LD2815` 15.4k surfaces, `09LD2817` 14.6k, `09LD2805` 13.4k, … — re-confirm with
  the centroid-in-footprint count, *not* the `in_lod2_citygml` bbox flag).
- Run `scripts/run_m0.py --config <tile>.yaml` per tile → one `.npz` each (git-ignored).
- Keep the val tile geographically **disjoint** from train tiles (no shared buildings).

### G2 — Multi-tile training loop
- **Memory reality (8 GB):** one tile's candidate support is ~4 M voxels ≈ 6.5 GB at
  base=8 — so **batch = 1 tile / step** (no multi-tile batching). Iterate tiles
  SGD-style (shuffle tiles each epoch); the adapter's `batch_index` already supports
  packing if a smaller-tile regime is adopted later.
- Extend `train/overfit.py` → a `train_multi(tiles, val_tile, …)` (or a new
  `train/generalize.py`) that loads each tile lazily, builds support/features/labels
  with the **G0 border exclusion**, steps once per tile, and periodically evaluates on
  the **held-out** tile via the shared metrics + per-tile B3.
- Watch for the failure mode the first step warned about: if val IoU ≈ the val B3
  shell, the model only learned support-boundary extraction → revisit support / try a
  generative decoder (this is where fork-1's "harder support" question actually lands,
  *if* the data says so).

### G3 — Evaluate + record
- Held-out tile: unobserved IoU (3 cutoffs) vs that tile's **B1 and B3**; state
  whether both honesty-bar clauses hold.
- Experiment record `experiments/exp_NNN_m2_generalize/` (README + metrics.json +
  per-tile B3 json + a val observed→completed→GT slice).
- SESSION_LOG: train/val tiles, the generalization verdict, next step (M3 semantics,
  or harder-support if val≈B3).

---

## Acceptance
- [ ] G0 ignore-margin implemented; metrics support an exclusion region; unit-tested;
      B3 re-checked with the band; overfit number reported with/without band.
- [ ] ≥ K train tiles + ≥1 disjoint held-out val tile produced (CityGML-coverage
      verified, not the bbox flag).
- [ ] Multi-tile training runs within 8 GB (batch=1 tile/step) and evaluates on the
      held-out tile via the shared `pointcraft.metrics` under all three cutoffs.
- [ ] Held-out unobserved IoU **> B1 and > B3** on that tile (both clauses), or the
      gap is analysed and the next step chosen accordingly.
- [ ] Experiment recorded; SESSION_LOG updated.

## Scope guards
- ❌ Occupancy only — semantics is M3 (head already widenable).
- ❌ No geometry-overlap clipping yet (ignore-margin is the chosen G0 fix); revisit
   only if border loss proves material.
- ❌ Don't commit `.npz`, checkpoints, or `pred_coords.npy`.
- ❌ Don't claim generalization from the train tiles — only the held-out tile counts.
- ✅ All runs through `.venv/Scripts/python`; all scoring through `pointcraft.metrics`.
