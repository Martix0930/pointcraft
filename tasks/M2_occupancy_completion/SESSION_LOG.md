# M2 — Occupancy Completion — SESSION LOG

## Current status: **exp_004 scale-up DONE — scale-up hypothesis FALSIFIED** ⛔→ next: evaluate generative decoder

> The fork-1 generalization claim stands. But **exp_004 (scale-up to stabilize the peak)
> did NOT stabilize it.** Across 10 train tiles + a wd ladder + 3 seeds, the peak-then-
> collapse + ≈0.06 jitter **reproduces** — the "more data/regularization → stable optimum"
> premise is **unsupported**, and the instability is **intrinsic to the discriminative +
> fixed-candidate-support framework**, not parameter-solvable. The only thing scale-up
> improved is the **cross-seed reproducibility of the early-stop ceiling** (peak spread
> 0.011). **Strong-sense stability was not established and the evidence says it cannot be,
> in this framework.** Next step: evaluate a generative decoder (see the 2026-06-13 entry
> and `docs/06_DECISIONS.md`). Earlier fork-1 status retained below for provenance.

## fork-1 generalization (retained) — held-out beats the deterministic shell on STRICT (robust) ✅

> First-step (single-tile overfit, below) DONE; fork-1 multi-tile generalization closed.
> **Conclusive (per-cutoff, 5 runs / 3 seeds — A/A\* labels abandoned):** held-out
> 09LD2814 **robustly clears B3 on strict** (0.234–0.267 vs 0.165, all runs), **marginal
> on mid**, **borderline on tolerant**. fork-1 "yes" rests on strict. Run-to-run noise
> ≈0.015 (unstable peak; same source as the collapse). See the 2026-06-08 G1/G2/G3/G3+
> entries at the bottom. Next gate: M3 semantics vs scale-up-to-stabilize (open).

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

### 2026-06-08 — M2 fork-1 G0: ignore-margin border fix (metrics + trainer)

Spec first absorbed two refinements (committed): **data-driven margin** (default 5,
not 8 — measured cost/poison table on 09LD1874: margin 8 ignores 7.7% of target
supervision vs 4.5% at 5, and the poison proxy is tree-dominated / not
edge-concentrated) and a **required support-recall-ceiling** report in G3 (so a weak
held-out IoU separates model-problem from coverage-problem).

G0 implementation (compute-at-use, **no `dataset_version` bump**):
- `metrics.border_keep_mask(coords, grid, margin)` + `evaluate(..., border_margin=)`
  excludes the XY border band from **both** pred and target (cutoff masks filtered
  row-aligned). `train/overfit.py` excludes the band from the **loss** too;
  `scripts/diagnose_support_shell.py --margin` and `run_m2_overfit.py --border-margin`
  wired. `tests/test_border_margin.py` (numpy): keep-mask drops only the band,
  border FP/FN excluded both sides.
- **Overfit number with/without band (re-eval saved pred, no retrain):**
  no-band strict 0.817 / band(m=5) strict 0.819 — **negligible**, i.e. the single-tile
  result was never inflated by border contamination (expected: small directional bias).
- **B3 with band (m=5):** morphological shell strict 0.164 (vs 0.149 no-band) — still
  ≪ 0.6, conclusion unchanged. (`diagnostic_morph_shell_m5.json`.)
- Tests: **global 54 + 2 skipped; venv 61.**

G0 done as a self-contained unit; the multi-tile skeleton (G1–G3) is the next step.

### 2026-06-08 — M2 fork-1 G1.a: zero-cost tile scan (candidate list)

Ran the G1 execution-book scan. `scripts/scan_tiles.py` (copied from the spec;
fixed one bug — `global` declared *after* the names were read as argparse defaults).
One read-only pass: load the 9 LOD2 GMLs once (336,382 rings: roof 91,502 / facade
236,707 / ground 8,173), clip to each LAS footprint by **centroid-in-bbox** (not the
`in_lod2_citygml` flag), report coverage / density / complexity per tile. LAS read =
header only (bbox + count). Verified the 3 `TODO(repo)` interfaces before running
(`load_citygml` `.polygons`/`.labels`; labels roof 3 / facade 4 / ground 1;
`data/raw/lidar/09LD*.las`).

**Result: 39 / 60 covered** (`coverage_ok` = roof ≥ 200 ∧ ground ≥ 1) →
`outputs/g1/tile_scan.csv` (git-ignored). Sheet decode `09LD WXYZ → row=10W+Y,
col=10X+Z` for the disjointness check.

Tension found (§1): surf/ha and height_std **anti-correlate** in this city (uniform
high-rise blocks → low h_std but high non_flat) — no tile is both dense and
height-mixed-complex. So "complexity" for a dense held-out leans on
`non_flat_roof_ratio`, not `height_std`.

**Recommended split (pending human 拍板 — NOT yet ratified):**
- held-out **09LD2814** (row21,col84): dense ∧ complex on the direct proxies
  (surf/ha 1071, fp 0.479, non_flat 0.265) → stresses *both* known failures (B1
  block-merge coverage shortfall + setback/overhang hallucination).
  Alts: 09LD1875 (height-mix: fp 0.511, h_std 47.7) or 09LD2816 (surf/ha 1093).
- train K=4 gradient, all Chebyshev ≥2 sheets from 2814: 09LD1878 (regular floor,
  fp 0.070) → 09LD1845 (0.083) → 09LD1846 (medium, 0.261) → 09LD1885 (medium-dense
  0.371, the "weak taste" of the merge regime per §1).
- ⚠ 2804/2805/2813/2815 are *adjacent* to 2814 → excluded from train if 2814 held-out.

**Gate:** G1.c (per-tile `run_m0` → 5 `.npz` + per-tile B3 with m=5 band) is **not
started** — waiting on the train/held-out pick. `scripts/scan_tiles.py` is untracked
(not committed yet). Next prompt: ratify the split, then run G1.c.

### 2026-06-08 — M2 fork-1 G1.b/G1.c: split ratified + multi-tile dataset built

**Split ratified (human 拍板):**
- **held-out: 09LD2814** (dense ∧ complex: surf/ha 1071, fp 0.479, non_flat 0.265).
- **train K=4 (regular→medium gradient): 09LD1878 / 09LD1845 / 09LD1846 / 09LD1885.**
- 2804/2805/2813/2815 stay excluded (adjacent to 2814). 2816 / 1875 noted as the
  scale-up second-held-out candidates.
- `scripts/scan_tiles.py` committed (e1c63db) as the reproducible G1.a deliverable.

**Datasets** (`outputs/m0/g1/<tile>.npz`, git-ignored; configs `configs/g1_<tile>.yaml`,
all 9 GML grids listed, centroid-in-extent filter handles coverage):

| tile | role | shell vox (R/F/G) | unobserved % | grid (z) |
|---|---|---|---|---|
| 09LD2814 | held-out | 675,500 (95k/525k/55k) | 60.0% | 400×300×103 |
| 09LD1878 | train | 87,253 (11k/68k/8k) | 50.3% | 400×300×148 |
| 09LD1845 | train | 79,609 (14k/57k/8k) | 55.2% | 400×300×94 |
| 09LD1846 | train | 310,508 (42k/239k/29k) | 55.9% | 400×300×96 |
| 09LD1885 | train | 369,009 (59k/274k/36k) | 63.1% | 400×300×259 |

CityGML coverage verified by the run_m0 centroid-in-extent count (12,855 surfaces in
2814, etc.), not the bbox flag.

**Per-tile B3 (m=5 band; NOT borrowed from 1874)** — `outputs/g1/b3/<tile>_b3_m5.json`.
Honesty bar = max(B1, B3) per tile, unobserved strict / mid / tol:

| tile | B1 (solid) strict | **B3 (shell) strict/mid/tol** | bar to beat (strict) |
|---|---|---|---|
| **09LD2814 (held-out)** | 0.146 | **0.165 / 0.134 / 0.141** | **0.165** |
| 09LD1878 | 0.026 | 0.055 / 0.030 / 0.029 | — |
| 09LD1845 | 0.022 | 0.050 / 0.036 / 0.034 | — |
| 09LD1846 | 0.092 | 0.153 / 0.124 / 0.127 | — |
| 09LD1885 | 0.043 | 0.119 / 0.092 / 0.092 | — |

All shells ≪ 0.6 → the candidate support does **not** trivialize the task on any tile.
Held-out 2814's bar (B3 strict 0.165) is even slightly higher than 1874's m=5 shell
(0.164) — consistent with picking a denser/more complex held-out. **G2 must clear
unobserved strict 0.165 (and > B1 0.146) on 2814.**

**Geographic disjointness (LAS bbox, XY):** all 4 train tiles disjoint from 2814 —
no positive 2D overlap. 1845/1885 touch 2814's x-edge at −6000 exactly but are
600–1800 m clear in Y → no shared buildings. 2814 X[−6400,−6000] Y[−36600,−36300];
1878 X[−4800,−4400] Y[−35400,−35100]; 1845 X[−6000,−5600] Y[−34500,−34200]; 1846
X[−5600,−5200] Y[−34500,−34200]; 1885 X[−6000,−5600] Y[−35700,−35400].

**G1 acceptance: all green.** Scan with 3 column groups ✓; candidate list + ratified
split ✓ (held-out clearly denser/more complex than train); 4 train + 1 disjoint
held-out `.npz` (coverage verified) ✓; per-tile B3 with m=5 ✓; disjointness ✓; no
兜底 triggered (covered set had a usable regular→dense gradient). **Next: G2 —
`train_multi` (batch=1 tile/step, G0 border exclusion), eval on 2814 vs bar 0.165.**

### 2026-06-08 — M2 fork-1 G2: multi-tile training + held-out verdict (A*)

`src/pointcraft/train/generalize.py::train_multi` + `scripts/run_m2_generalize.py`
(`experiments/exp_003_m2_generalize/`). Train 1878/1845/1846/1885 → held-out **2814**.

**Pre-flight (G2.0):**
- **G2.0a recall ceiling on 2814 (m=5):** overall 0.882, unobserved strict **0.758** /
  mid 0.683 / tol 0.787 — well below 1874's 0.93 → the **coverage/merge axis is
  genuinely stressed** (B1 buries interior facades), exactly why 2814 was chosen. This
  is the hard IoU cap (perfect within-support model ≤ ~0.758 strict).
- **G2.0b memory:** peak CUDA **3.4 GB** training on 1885's 3.65M-voxel support — no
  OOM, no chunking/grad-off needed (every tile < the 4.17M/6.5 GB overfit case).
- **§2 risk-2 (feature tile-invariance) — found & fixed.** `build_features` divided the
  height channels by per-tile `K` (94→259), so identical physical structure produced
  different features. Added `z_scale` (fixed metres, default 50) → "metres below column
  top / above ground"; default `None` preserves the overfit behaviour bit-for-bit.
  Verified cross-tile distributions align physically after the fix.
- **§2 risk-3 (variable spatial_shape):** confirmed model forwards across K=94→259 with
  no fixed-grid assumption. **§2 risk-1:** trained on full border-kept support, imbalance
  via per-tile `pos_weight` (no negative subsampling).

**Result — held-out 2814 unobserved IoU vs bar = max(B1,B3), best ckpt (ep40):**

| cutoff | B1 | B3 shell | **held-out** | bar | pass |
|---|---|---|---|---|---|
| strict | 0.146 | 0.165 | **0.251** | 0.165 | ✅ (1.52× B3) |
| mid | 0.091 | 0.134 | **0.138** | 0.134 | ✅ |
| tolerant | 0.087 | 0.141 | **0.132** | 0.141 | ❌ (> B1, < B3) |

**VERDICT A\* — qualified generalization (NOT case B).** The three diagnostics asked for:
1. **Absolute + ratio:** strict **0.251 = 1.52× B3, 1.72× B1, 0.33× the 0.758 ceiling.**
   Clearly *above* the deterministic shell → learned transferable structure, so the
   fork-1 binary question "换 tile 还成不成立" answers **yes**. (The runner's first
   auto-label "VERDICT B → generative authorized" was a heuristic bug — `0.6×B3` band;
   fixed `_verdict` to A/A*/B/C + curve flag.)
2. **train↔held-out gap:** at the operating point (ep40) held-out 0.251 **>** train avg
   0.099 — *not* memorized; the model learned generic shell first. Memorization is a
   *late-training* artifact (ep200: train 0.31 vs held-out 0.025).
3. **Curve shape: peak-then-collapse.** held-out strict 0.245→**0.251@ep40**→0.20→0.18→
   0.05→…→0.025@ep200, monotonically down after ep40 while train climbs 0.10→0.31.
   Textbook overfit-to-4-tiles; **early-stop at the peak is mandatory** (best-ckpt
   selection captured ep40 correctly).

**§4 placement:** A* → proceed to G3 record. The generative-decoder branch does **NOT**
fire (only case B + low ceiling authorizes it; we are 1.52× above the shell, not stuck
on it — scope-guard respected: don't go generative before the ceiling says so, and only
under case B). Current bottleneck = **overfitting on only 4 train tiles** + the eventual
0.758 coverage cap. Levers in order: early-stop (free, done), regularization
(weight-decay/dropout), **more train tiles** (39 covered available; fast collapse with 4
is the classic "need more data" signal). Coverage/generative is a *later* lever — we sit
at 0.33 of the ceiling, ample discriminative headroom remains.

Artifacts: `metrics.json` (ceiling, bar, per-eval history with per-tile train IoU,
verdict), `README.md`, `viz_heldout.png`, `pred_coords_val.npy` (gitignored). New code:
`train/generalize.py`, `scripts/run_m2_generalize.py`, `z_scale` in `train/overfit.py`.
**Next: G3 — formal experiment record + write the A* verdict; then either (a) extend
train tiles + regularize to push past the tolerant cutoff and toward the ceiling, or
(b) M3 semantics — research-lead's call.**

### 2026-06-08 — M2 fork-1 G3: freeze record, close the generalization fork (no new training)

G3 froze the G2 result into a defensible record; **no retraining** (any recompute was
regenerated from saved history). fork-1's binary question — *does it still hold when we
switch tiles* — is **answered: yes (A\*)**.

**Canonical verdict = A\*** (qualified generalization), held-out 09LD2814 unobserved IoU:
strict **0.251** (✅ 1.52× B3, 1.72× B1), mid 0.138 (✅), tolerant 0.132 (❌ > B1, < B3).
NOT A (tolerant misses B3); NOT B (decisively above the shell → generative branch does
**not** fire). The G2 runner's first auto-print "VERDICT B → generative authorized" was a
`_verdict` heuristic bug (`0.6×B3` band); fixed to A/A\*/B/C + curve flag, record regenerated.

**Three caveats locked to the number** (travel with it everywhere): (2a) tolerant < B3 =
precision-favouring/recall-conservative profile, the reason it's A\* not A; (2b) 0.251 is
an **early-stopping peak (ep40)**, not steady state (collapses to 0.025@ep200); (2c) eval
granularity 20 ep → true peak ∈ ep30–50, don't over-quote the 3rd digit ("passes" robust:
ep20=0.245 already clears).

**Methodological findings preserved:** (4a) ceiling 0.758 = roof, bar 0.165 = floor →
diagnostic is **held-out/ceiling = 0.33×**, two-thirds discriminative headroom, coverage
not yet binding; (4b) train↔held-out reversal (ep40 held-out 0.251 > train 0.099) →
structure is tile-agnostic, validates G1's hard-held-out design; (4c) **z_scale=50 m
tile-invariance fix logged as DECISION D10** (`docs/06_DECISIONS.md`), default None
preserves overfit bit-for-bit, residual spread is real domain variation not an artifact.

**Branch + levers:** A\* → generative gated OFF. Bottleneck: overfit-on-4-tiles (now) →
0.758 coverage cap (later). Levers: early-stop (done) → regularization **and** more tiles
tuned *together* (4-tile diversity can't survive 40 ep of fitting pressure) → generative
(later, gated on B+low-ceiling). Hazard recorded: z_scale=50 → high-rise extrapolation
bounded by tallest training structure; check first if a future B-failure is high-rise.

**Next-step gate (venue judgment, left open):** recommended = close fork-1 → **M3 semantic
dual-head** (new high-information question); alternative = scale-up first (8–10 tiles +
regularize toward the ceiling) only if a Harada-facing timeline values a prettier
generalization IoU over the semantic dimension. G3 (this record) closes fork-1 either way.

### 2026-06-08 — M2 fork-1 G3+: peak-confirmation, non-determinism characterized, A/A\* dropped

Triggered by the dense-eval (every-10) re-run disagreeing with the every-20 run at the
ep40 peak (tolerant 0.143 vs 0.132, ~0.011). Two confounds had been introduced together
(eval cadence + the int32/compact-array rewrite), so "GPU non-determinism" was unproven.
Ran the controlled experiment first.

**Infra fixes en route (all real, keep):** (a) host-RAM `MemoryError` — the
precompute-all-resident design exhausted host RAM (only ~5–6 GB free); rewrote
`train_multi` to **compact resident arrays (support int32 / feats f32) + eval-time Sample
reload**, fast and low-resident. (b) CUDA fragmentation OOM — added
`PYTORCH_ALLOC_CONF=expandable_segments:True`. (c) A fully-lazy per-step recompute variant
was ~20 s/step → rejected. (d) User's SD WebUI (PID 33024) held ~2 GB GPU; closed at user
request to free the card.

**Deterministic double-run (the decisive test):** two **identical-config, same-seed(0)**
runs differ by **|Δ| max ≈ 0.014–0.015** on strict/mid/tol across eval points → the
pipeline is **non-deterministic at ~0.015** (spconv GPU atomics). The ~0.011 old↔new gap
is **within** this band → **not a dtype shift**. dtype also excluded mechanistically:
`to_sparse_tensor` casts coords to int32 regardless, so int64→int32 feeds identical GPU
input. ⇒ non-determinism attribution is now earned and recorded.

**Non-determinism is a finding, not a disclaimer:** 0.015 is abnormally large for an IoU
metric; the held-out **peak solution is unstable** (peak epoch swings ep10→ep50, peak
magnitude 0.234→0.267 across seeds) and this is the **same instability** as the
peak-then-collapse — one fragile-region phenomenon.

**Conclusive verdict (per-cutoff, 5 runs / 3 seeds; `peak_confirm.json`; A/A\* abandoned):**

| cutoff | B3 | held-out range | clears | verdict |
|---|---|---|---|---|
| strict | 0.165 | 0.234–0.267 | 5/5 (margin ~0.07–0.10 ≫ noise) | **ROBUST PASS** |
| mid | 0.134 | 0.127–0.150 | 4/5 (margin ≈ noise) | MARGINAL |
| tolerant | 0.141 | 0.119–0.144 | 2/5 | BORDERLINE (mostly below) |

**fork-1 = YES, resting solely on STRICT.** Single-run A vs A\* flipped on noise (det_a tol
0.139 → A\*, det_b tol 0.144 → A, same config) — hence the label is dropped.

**Cross-seed scope (honest limit):** seed-1/2 runs are **60 ep = peak region only**; they
do **not** observe the collapse (only the seed-0 150-ep run does). Not full-trajectory
reproduction.

Records updated: `metrics.json` verdict reframed (old A\* demoted to
`verdict_DEPRECATED_single_run`), `peak_confirm.json` added (5-run summary + same-seed
spread), README canonical-verdict section. Code: `train/generalize.py` (compact-resident
rewrite), `scripts/run_m2_generalize.py` (`expandable_segments`). **Branch unchanged:**
strict robustly above shell → generative does NOT fire; bottleneck is now **peak
instability / 4-tile overfitting** → scale-up-to-stabilize (more tiles + regularization)
is the natural lever if pursued; else M3 semantics. Still the venue/advisor's call.

### 2026-06-10 — M2 fork-1 follow-up: LiDAR class-2 ground QA sweep (diagnostic only)

`scripts/diagnose_ground_coverage.py` (committed; CSV `outputs/g1/ground_class2_sweep.csv`
git-ignored). Read-only sweep of ASPRS classification across all 60 LAS tiles.

**Free authoritative ground, no semantic shortcut.** Class-2 (GROUND) present in **60/60**
tiles, 12.4–51.1% of points (mean 22.4%). Only classes **1/2/3** appear anywhere → the
LiDAR carries authoritative *terrain* ground for free (no DEM download — the gap the data
contract flagged), but **no building/semantic labels** (no class 6/9/5). M3 semantics gets
**zero shortcut**; buildings stay CityGML-sourced.

**Coverage is the catch.** Ground XY coverage of the *sensed footprint* (1 m cells,
ground-return cells / any-return cells): mean **60.2%**, range **37.7–94.2%** → ~40% of the
footprint has no ground return. Structured **under-roof occlusion**, not random.

**Cross-validation (honest, weak-moderate — NOT a tight triangulation).** The holes track
**building footprint fraction**, not raw surface density: Spearman(cov, footprint_ratio)
≈ **−0.21** (60 tiles) / **−0.29** (covered 39); lowest-coverage decile mean footprint_ratio
**0.33 vs 0.18** highest. By contrast surf_per_ha ≈ 0, non_flat ≈ +0.13, height_std −0.29.
So ground-occlusion holes share **one** mechanism (plan-area under roofs) with the
recall-ceiling / coverage axis that makes dense tiles hard — directional corroboration
**through the footprint axis specifically**, not via surface-count or articulation. Caveat:
held-out 2814 is high-footprint with a low recall ceiling (0.758) but only **average** ground
coverage (60.5%), so it is *not* a clean triple-low exemplar — the three signals agree in
direction, loosely, not tile-by-tile.

**Design decision (recorded, NOT implemented): sparse anchors, not interpolation.** Use
class-2 as **sparse ground anchors + explicit unknown mask** — do **NOT** TIN/IDW-interpolate
a continuous DEM. Interpolation would fabricate ground elevations under buildings (exactly
the occluded region the model must reason about), violating the observed/unobserved honesty
(D4/D7). Missing ground stays honestly missing behind the unknown mask.

**Two-step plan + guardrails.** (1) *Now* = QA layer only: validate the existing CityGML
building-base ground / vertical alignment; **not in the model.** (2) *Scale-up* = wire class-2
as an explicit ground feature under a **new `dataset_version`**. Does **NOT** touch D10
(`z_scale`/`above_ground`); any change there opens its own record and is not compared with
exp_003.

### 2026-06-13 — exp_004 M2 scale-up: scale-up hypothesis FALSIFIED; peak instability is intrinsic

`experiments/exp_004_m2_scaleup/`. Objective (from the exec spec) was **stability, not IoU**:
make fork-1's fragile early-stop peak stable by moving exactly two coupled knobs —
**train-tile count (4→10)** and **regularization (weight-decay)** — everything else frozen
(held-out 09LD2814, eval口径 m=5/3-cutoff, D10/`z_scale=50`, architecture).

**Tile selection — four-axis pick, K/support MEASURED not guessed.** Combined candidate table
(`outputs/g1/combined_candidates.csv`, `scripts/build_combined_candidates.py`) merged
density/complexity (`tile_scan.csv`) with ground coverage (`ground_class2_sweep.csv`) over the
39 coverage_ok tiles. Locked **10 train tiles** spanning a difficulty gradient (surf/ha
108→944, held-out 2814 at 1071 stays hardest), ground-coverage ≥ 50%, Chebyshev ≥ 2 from 2814.
Critically, `height_std` proved a **bad proxy** for memory: `scripts/preflight_support.py`
measured K/support in-memory (no train .npz) and showed the dense-AND-tall-AND-articulated
tiles (1864 5.9M, 1874/1876 ~4.2M) **exceed** the proven 1885 HWM (3.77M @ 3385 MB), while the
two most-articulated affordable tiles (2818 nfr=0.317, 1843 nfr=0.308) fit at 0.29×/0.40×.
**Design limitation recorded before training:** the train set has **no dense×articulated**
sample (that corner busts memory), so 2814's density×articulation *interaction* generalization
is **not verifiable** here — a failure on articulated-dense regions would be expected and
uninformative, not evidence against the model.

**Infra (real debt paid).** First ladder chained 3 wd runs in one driver; wd=1e-3 died on a
**host-RAM** numpy MemoryError mid-eval, then wd=1e-2 inherited a **corrupted CUDA context**
(cascade). Fixed: (1) isolated runs + a **GPU-clean gate** between them
(`run_stageA_wd_ladder.sh` / `run_stageB_seeds.sh`); (2) `--train-eval-every-steps 200` —
the per-tile train_iou diagnostic (10 full-Sample reloads/eval, the dominant new host-RAM
churn) gated to a coarser cadence; the held-out `val_unobs` curve stays every 50 tile-steps,
untouched. Also added `--eval-every-steps` (tile-step x-axis, cross-K-comparable) and
`--weight-decay` to `train/generalize.py` + the runner; exp_003 epoch path preserved.

**Stage A (wd ∈ {1e-4,1e-3,1e-2}, seed 0, full precision, tile-steps).** All three still
**peak-then-collapse with large jitter — none is flat** (`stageA_curves.png`). The decisive
finding: **weight-decay flattens the mean DRIFT but not the JITTER — post-peak std ≈ 0.06 is
identical across two orders of wd magnitude.** wd=1e-3 best stops the *mean* collapse
(back-half drift −0.008, plateaus) and was carried to Stage B — not because it stabilizes
(it does not) but as the cleanest reg at which to quantify the residual instability.

**Stage B (3 seeds @ wd=1e-3) — the result.** `stageB_summary.json`, `stageB_curves.png`.

| seed | peak strict | @step | post-peak std | end |
|---|---|---|---|---|
| 0 | 0.294 | 250 | 0.065 | 0.173 |
| 1 | 0.305 | 200 | 0.062 | 0.186 |
| 2 | 0.301 | 400 | 0.062 | 0.038 |

- **cross-seed peak spread = 0.011** (< fork-1's ~0.03; all 3 peaks > B3 0.165).
- **per-seed post-peak std = 0.062–0.065** (reproduces Stage A's ≈0.06, every seed).

**Main conclusion — weighted correctly, NOT a symmetric SPLIT:**
> **The scale-up hypothesis is FALSIFIED.** 10 tiles + a 2-order wd sweep + 3 seeds did **not**
> stabilize the peak: the peak-then-collapse + ≈0.06 jitter **reproduces across seeds and
> across wd**, so the collapse/jitter is **intrinsic to the discriminative + fixed-candidate-
> support framework**, **not** a data-diversity deficit and **not weight-decay-solvable**. The
> **only** improvement scale-up delivered is the **cross-seed reproducibility of the early-stop
> ceiling** — and the honest reading of **spread 0.011 is "the ceiling is reproducible," NOT
> "generalization is robust/stable."** Taking the max over a ~0.06-amplitude oscillating band
> clusters near that band's ceiling regardless; peak-spread agreement is necessary, not
> sufficient, for stability. **Strong-sense stability (hold the peak / no collapse) was NOT
> established, and the cross-seed × cross-wd reproduction of the instability is positive
> evidence that it cannot be established in this framework.**

**Core output of this round (do NOT bury it in the SPLIT narrative): a NEW, independent line
of evidence for evaluating a generative decoder.** The peak instability **reproduces across 3
seeds and across 2 orders of wd** → it is a **framework-intrinsic** property, an authorization
signal **independent of** the 0.758 recall ceiling. The decoder question is now gated on **two
independent signals** — coverage cap (recall ceiling 0.758) **and** framework-intrinsic peak
instability — **not** on field/trend cosmetics. See `docs/06_DECISIONS.md` (2026-06-13).

**Methodological note (the most transferable lesson on this line).** Stage A's **single-seed**
ugly curve induced **both** the executor (pre-committed "evidence → Outcome 2, expect spread
≳0.03") **and** the research lead to **pre-judge Outcome 2**. Stage B's **multi-seed** data
**refuted that specific prediction** (spread came in 0.011, not ≳0.03) and, more importantly,
**decoupled two things the single-seed view had fused**: *(i) is the ceiling reproducible?* →
yes, and *(ii) is the instability solved?* → no. The executor's pre-commit was wrong on (i)
and was **not** spun to fit; recorded as a correction in the exp README. **Lesson: do not draw
structural conclusions from a single-seed noisy curve — run seeds before deciding which
phenomenon you are even looking at.** A jittery single curve cannot tell "unstable optimum"
from "reproducible-ceiling-with-transient-peak"; only seeds separate them.

**Deferrals unchanged / scope honored:** D10 untouched; no class-2 ground integration; no
semantics; generative branch **evaluated next, not armed here**. We did **not** chase the curve
with epochs/lr/dropout (§10) — the falsification is the result, not a tuning failure.
