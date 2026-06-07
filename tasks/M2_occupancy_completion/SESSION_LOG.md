# M2 — Occupancy Completion — SESSION LOG

## Current status: **first-step DONE** — single-tile overfit beats the M1 floor ✅

Phases 0–F complete. A small sparse-conv U-Net (`OccupancyCompletionUNet`, 79k
params) **overfits `09LD1874`** and clearly clears the M2 gate, scored by the shared
`pointcraft.metrics` under all three cutoffs (`experiments/exp_002_m2_overfit/`):

| unobserved-region IoU | strict | mid | tolerant |
|---|---|---|---|
| B1 floor (M1) | 0.061 | 0.040 | 0.039 |
| **M2 overfit** | **0.817** | **0.758** | **0.805** |
| B2 ceiling (M1) | 0.359 | 0.363 | 0.379 |

completion IoU **0.853** (prec 0.92 / rec 0.92); per-class recall ground 0.92 /
roof 0.76 / facade 0.96; peak CUDA 6.5 GB.

**Honest framing (not over-claiming):** this is an **overfit** — trained and
evaluated on the same single tile, no val split, occupancy threshold picked on the
tile. It exceeds even the footprint-informed B2 ceiling *because* it memorises this
tile's shell. The candidate support the net classifies over is **input-only** (the
B1 extrusion volume ∪ observed voxels — never the target), so beating B1 is genuine
completion of the unobserved shell, not transcription; but **no generalization is
claimed**. The key learned insight: the target shell ≈ the **boundary of the solid
candidate support**, which the U-Net recovers from dense neighbourhoods. The gate it
proves: *the model + data + metrics pipeline learns and beats the deterministic
floor.*

**Backbone = spconv** (D9); MinkowskiEngine was the Phase-0 fallback (not needed).
**Output-representation constraint (D9):** predictions are emitted in data-contract
coords format, per-voxel thresholdable logit, world-placeable (6677) — explicit,
semantic-ready (M3), queryable (M5); not a metadata-less dense tensor.

## Next recommended prompt for Claude Code

> The M2 **first step (single-tile overfit) is DONE** — see
> `experiments/exp_002_m2_overfit/` and the table above. Use **`.venv/Scripts/python`**
> for everything (global Python has no torch/spconv); reproduce with
> `.venv/Scripts/python scripts/run_m2_overfit.py --iters 500 --base 8`. Two
> divergent next directions — **flag both, don't auto-start**:
>
> 1. **M2 generalization (multi-tile).** ⛔ BLOCKED first: replace the centroid
>    tile-crop with geometry-overlap clipping / an ignore border (alignment rule 4 /
>    GOTCHAS) before training on >1 tile, else the border band supervises visible
>    roofs as empty. Also needs a held-out val tile + a batching dataloader, and an
>    honest support that doesn't lean on B1 as hard (the current support boundary ≈
>    GT shell is what makes overfit so easy; for generalization, test a learned
>    generative decoder or a thinner support so the model must infer extent).
> 2. **M3 semantics.** Widen the head (`out_channels>1`, already supported) to add a
>    roof/facade/ground head; reuse the same backbone, adapter and metrics (add
>    semantic mIoU). Single-tile overfit first, same pattern.
>
> Keep scoring through the shared `pointcraft.metrics`. No checkpoints/real data in git.

---

## Session entries

### 2026-06-07 — M2 Phase 0 + A: environment validation + docs

**Phase 0 — environment (this machine):**
- OS/Python: **Windows 11, Python 3.13.2**. GPU: **RTX 4060 Laptop (8 GB)**, driver
  **596.49** (supports CUDA 12.6). Neither torch nor spconv was preinstalled.
- Wheel availability probed before installing (the 2020–2023 SSC "version hell"
  risk): **`spconv-cu126 2.3.8` ships a `cp313-cp313-win_amd64` wheel** and
  **`torch 2.9.1+cu126` ships `cp313-win_amd64`** — both **prebuilt, CUDA-12.6
  runtime bundled** (no system CUDA toolkit, no source build). → spconv is viable on
  this exact stack; **MinkowskiEngine fallback not needed.**
- Decision: install into a repo-local **`.venv`** (keeps global Python 3.13 clean;
  `.venv/` already git-ignored). torch installed from `download.pytorch.org/whl/cu126`.
- **Installed + pinned:** `torch 2.9.1+cu126`, `spconv-cu126 2.3.8`,
  `cumm-cu126 0.7.11`, `numpy 2.4.6` (+ laspy/pyproj/Pillow/scikit-image/matplotlib
  for the existing suite). `torch.cuda.is_available()=True`, GPU matmul on the 4060 OK.
- ⚠ Found: legacy `data/lod2.py` imports `PIL`+`skimage` at top level → needs the
  `viz` extra or `test_target_occupancy.py` errors. Installed; **full suite 47/47
  passes in the venv.**
- **GATE — `scripts/check_spconv_env.py` → GATE PASSED ✓** (new, committed; an env
  self-check, not model code). On real `outputs/m0/tokyo_citygml.npz`: built a
  `SparseConvTensor` from 543,803 partial voxels on the (400,300,222) grid, ran
  SubMConv3d(2→16) + strided SparseConv3d(→16) → 230,458 active voxels at half-res;
  backward gives finite input-feature **and** all param grads; peak CUDA mem
  **424 MB** (huge headroom on 8 GB). spconv is viable — **no Minkowski fallback.**
  Reproduction commands + version table recorded in `docs/07_GOTCHAS.md`.
- `pyproject.toml`: added a `learn` optional extra (names only; the +cu126 build
  needs the CUDA wheel index, documented in GOTCHAS).

**Phase A — docs (no model code):**
- `docs/06_DECISIONS.md` **D9**: backbone = spconv; S3CNet/GeoSVR/SCPNet/JS3C-Net
  re-classified; **output-representation constraint** (explicit/semantic-ready/
  queryable) recorded as the carry-the-project requirement.
- `docs/05_RELATED_WORK.md`: GeoSVR re-classified (not a code source; conceptual +
  same-lab line); added SCPNet/JS3C-Net as the real spconv code refs, S3CNet as
  idea-only.
- `docs/research_roadmap.md` §2/§3/§5: S3CNet→idea-only & SCPNet/JS3C-Net→code;
  GeoSVR row corrected; 3DRR row corrected (this edition = low-light/smoke
  degradation, not facade completion, not a venue here); Harada contact map rewritten
  with the lab's exact Future-Directions wording ("limited observation",
  "estimating unobservable areas", "towns and suburbs", "future prediction … by
  accumulating data") and marked **direction-fit, not method-identity** (their text
  says towns/suburbs, not "city"); project positioning updated (understanding-driven
  completion under limited observation → embodiment-ready, identity-semantic env;
  embodiment as future-work vision, e.g. R2-Dreamer alignment).
- M2 status → IN PROGRESS; target band from M1 recorded above.

### 2026-06-08 — M2 Phase B: npz → sparse-tensor adapter

- `src/pointcraft/data/sparse.py` (lazy torch/spconv; **not** re-exported from
  `data/__init__` → `import pointcraft.data` stays torch-free): `to_sparse_tensor`
  (contract coords+feats → `SparseConvTensor`, indices `[batch,i,j,k]`,
  `spatial_shape=grid.shape`), `coords_from_sparse_tensor` (lossless recovery,
  per-batch), `occupancy_logits_to_coords` (threshold → predicted occupied contract
  coords, D9). Added `data.grid_from_metadata` and routed `metrics.load_sample`
  through it (one grid-rebuild path).
- `tests/test_sparse_adapter.py` (importorskip): grid metadata + features preserved,
  `[batch,i,j,k]` order round-trips, per-batch recovery, logit thresholding. Suite
  51 (venv) / 47+skips (global).

### 2026-06-08 — M2 Phase C–F: completion U-Net, overfit, result, M2 first-step DONE

- **Phase C** `models/completion_unet.py` — `OccupancyCompletionUNet`: 2-level
  submanifold U-Net (SubMConv + strided SparseConv down / SparseInverseConv up),
  **additive skips** (concat blew the 8 GB budget at full res), occupancy logit per
  candidate voxel, output aligned to input order (D9). `out_channels` widens for a
  future M3 semantic head with no backbone change (tested).
- **Phase D** `train/overfit.py` + `scripts/run_m2_overfit.py` — candidate **support
  = B1 extrusion ∪ observed (input-only)**; features (observed flag, carried height,
  k-frac, depth-below-top, above-ground); labels = support∩target. Weighted BCE +
  AMP; eval sweeps the occupancy threshold and reports the best by strict IoU.
- **Memory war (logged for the next session):** full grid 400×300×222, support
  **4.17M voxels**. (a) base=16 → 8.4 GB peak → NVIDIA **sysmem fallback** thrash
  (GPU 100% but ~no progress). (b) base=8 + subsampling interior negatives fit
  (4.2 GB) **but the subsampling holes corrupt submanifold neighbourhoods** → train
  loss fell while eval IoU stuck ~0.07 (model couldn't reject unseen interior).
  (c) **Fix:** the target shell ≈ the *boundary of the solid support*, so it needs
  **dense** neighbourhoods → train on the **full** support; **additive skips** bring
  base=8 full-support to **6.5 GB peak**. IoU then climbed 0.42→0.66→0.74→…→0.82.
- **Result (`exp_002_m2_overfit`, best @ prob 0.8):** unobserved IoU
  **strict 0.817 / mid 0.758 / tolerant 0.805**; completion **0.853** (prec 0.92,
  rec 0.92); per-class recall ground 0.92 / roof 0.76 / facade 0.96. **Beats B1
  ~13×, exceeds B2** (overfit — see honest framing above). Rough
  observed→completed→GT slice in `viz_overfit.png`; `pred_coords.npy` (gitignored)
  is the contract-format prediction.
- Output verified in **data-contract coords** (thresholdable, world-placeable), not
  a dense metadata-less tensor (D9 satisfied — checked, not just asserted).
- `tests/test_completion_model.py` (importorskip): U-Net fwd/bwd + order alignment,
  semantic-ready head width, and the input-only support/feature/label builders.
  **Suite 54 (venv) / 47 + 7 skipped (global).**
- DoD (first step) all ticked. Two divergent next paths flagged (multi-tile —
  blocked on centroid-crop; or M3 semantics); neither auto-started.

### 2026-06-08 — M2 diagnostic: deterministic shell of the support (B3)

Zero-cost check requested before choosing the next fork: *does the candidate support
construction "answer the question" for the model?* Added `baseline.morphological_boundary`
(B3, pure-numpy 6-face surface extraction) + `baseline.candidate_support` (moved the
input-only support builder out of `train` so it's torch-free) +
`scripts/diagnose_support_shell.py` (reusable per tile).

**Result on `09LD1874`** (`exp_002/diagnostic_morph_shell.json`), unobserved IoU:
- full solid support (≈B1): strict 0.061
- **morphological shell (deterministic): strict 0.149** / mid 0.114 / tol 0.116
- M2 trained: strict 0.817

The deterministic shell is **low (0.15 ≪ 0.6)** → the support does **not** trivialise
the task; the learned 0.82 is real work. *Why low:* per-column B1 extrusion merges
adjacent buildings into solid blocks, so true facades sit **inside** the merged
volume (not on its 6-face surface) — naive surface extraction misses them (recall
0.34, precision 0.24), while the model recovers them (P/R 0.92). **Decision implied:**
"swap the support / generative decoder" is *not* the primary issue → the clean next
step is **multi-tile generalization** (fork 1), still gated on the centroid-crop fix;
re-run this diagnostic per tile to confirm it stays low. Tests: 51 (global) / 58 (venv).
