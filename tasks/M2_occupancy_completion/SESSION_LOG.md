# M2 — Occupancy Completion — SESSION LOG

## Current status: **IN PROGRESS** — Phase 0 ✅ PASSED + Phase A ✅ done (network not started)

First learned model. **First step is an explicit single-tile overfit** on
`09LD1874` (the M0/M1 tile): prove the model + pipeline can learn by clearly beating
the M1 floor under the same multi-cutoff metrics. Generalization / multi-tile is a
*later* M2 phase, **blocked** on the centroid tile-crop fix — not this step.

**Target band to land in (M1, unobserved-region IoU on `09LD1874`):**

| | strict ~35% | mid | tolerant ~67% |
|---|---|---|---|
| B1 floor (obs-only) | 0.061 | 0.040 | 0.039 |
| B2 ceiling (footprint) | 0.359 | 0.363 | 0.379 |

- **Pass bar:** beat B1 (~0.06 strict) — proves completion, not transcription.
- **Strong:** approach/exceed B2 (~0.36) **without seeing the footprint**.
- Score with the **shared `pointcraft.metrics`** (strict/mid/tolerant). No new metric.

**Backbone = spconv** (D9); MinkowskiEngine is the Phase-0 fallback only.
**Output-representation constraint (D9):** predictions emitted in data-contract
coords format, per-voxel thresholdable logit, world-placeable (6677) — explicit,
semantic-ready (M3), queryable (M5). Not a metadata-less dense tensor.

## Next recommended prompt for Claude Code

> Phase 0 (spconv env) and Phase A (docs) are **done**; `scripts/check_spconv_env.py`
> prints GATE PASSED on this machine. Use **`.venv/Scripts/python`** for everything
> (global Python has no torch/spconv). Proceed to **Phase B**: an M0 `.npz` →
> spconv `SparseConvTensor` adapter in `src/pointcraft/data/` (coords int32 as
> `[batch,i,j,k]`, feats float32, target occupancy on the shared `VoxelGrid`),
> with a tiny round-trip test that grid metadata survives. Then **Phase C** the
> sparse-conv occupancy-completion UNet (occupancy head only; output in
> data-contract coords format, thresholdable, world-placeable — D9) and **Phase D**
> the single-tile overfit loop on `09LD1874`. Exit bar: unobserved IoU (strict)
> clearly > B1 floor 0.061, scored via the shared `pointcraft.metrics` under all
> three cutoffs. No multi-tile (centroid-crop blocker), no semantic head (M3).

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
