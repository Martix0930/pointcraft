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

## GeoSVR

- Sparse-voxel surface reconstruction (image-input). Reusable: sparse-voxel
  representation / regularization ideas. Direct touchpoint to target lab. _(read code TBD)_

## Minecraft / embodied AI

- Minecraft as embodied AI platform (agents, benchmarks). _(read TBD)_
- Embodied environment platforms (general). _(read TBD)_

> See `docs/research_roadmap.md` for the ordered reading list and rationale.
