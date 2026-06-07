# 05 — Related Work

Lightweight placeholders. Do not over-fill or fabricate citations; add precise
references as they are read. Each bullet = "what it is / what we take from it".

## Sparse convolution (Minkowski / spconv)

- Submanifold sparse convolution — efficient 3D conv on sparse voxels. _(read TBD)_
- Minkowski Engine — candidate backbone framework. _(read TBD)_
- spconv / SECOND — common engineering stack. _(read TBD)_

## Semantic Scene Completion (SSC)

- Origin of joint occupancy + semantic completion from partial depth/LiDAR. _(read TBD)_
- LiDAR SSC methods with 2D BEV + 3D fusion — backbone template. _(read TBD)_
- Recent SOTA on SemanticKITTI — comparison points. _(read TBD)_

### Code references for the M2 sparse-conv backbone (corrected 2026-06)

- **SCPNet (CVPR'23, spconv + Cylinder3D)** and **JS3C-Net (AAAI'21, spconv 1.0)**
  are the **actual code references** to read for structure (encoder–decoder, sparse
  tensor plumbing, loss setup). Both are pinned to **old spconv / CUDA** — read for
  design, do not expect them to run as-is on this machine's stack.
- **S3CNet (CoRL'20)** is an **architecture-idea reference only** (BEV 2D branch,
  geometry-aware losses) — there is **no reliable official codebase**. Borrow ideas,
  do **not** treat it as a "skeleton template" to clone. *(corrects the old roadmap
  claim that S3CNet was a direct skeleton template.)*

## 3D Occupancy Prediction

- Query-based occupancy prediction (driving). _(read TBD)_
- Occupancy benchmarks / evaluation protocols. _(read TBD)_

## Aerial LiDAR → LOD2 building reconstruction

- Plane-fitting / polygonal reconstruction (traditional). _(read TBD)_
- Learned LoD2 reconstruction; benchmarks/datasets. _(read TBD)_

## Point2Building

- Closest prior work: airborne LiDAR → LoD2 mesh via autoregressive generation;
  trained on paired LiDAR+LoD2. Key takeaways: data-pairing feasibility; learned
  inference of missing geometry. Differentiate via voxel+semantic+embodied output. _(read in depth TBD)_

## GeoSVR  (re-classified 2026-06 — NOT a code source)

- **What it is:** image → differentiable-render, **per-scene optimization** for
  geometrically accurate surface reconstruction from sparse voxels (Fictionarry,
  arXiv:2509.18090). It is **not** a supervised partial→complete completion network,
  and its code is **not reusable** for M2's pipeline.
- **What we take:** a *conceptual* reference only — evidence that an **explicit
  sparse-voxel representation can yield accurate, complete geometry** (supports M2's
  output-representation constraint). Also an **active same-lab line** (Lin Gu et al.)
  to stay aware of — a direction-fit, not a method we import.
- _(Corrects the earlier "open-source, borrow code into backbone" claim.)_

## Minecraft / embodied AI

- Minecraft as embodied AI platform (agents, benchmarks). _(read TBD)_
- Embodied environment platforms (general). _(read TBD)_

> See `docs/research_roadmap.md` for the ordered reading list and rationale.
