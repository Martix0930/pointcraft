# 01 — Research Question

## Main research question

**Can we complete full urban voxel occupancy and semantics from aerial sparse
point clouds that only observe roofs, ground, and vegetation tops — in particular
recovering the unobserved facades and building volumes — using LOD2/CityGML as
supervision?**

## Secondary research questions

1. **Representation** — Which voxel representation and feature set (occupancy only
   vs. height/normal/intensity/fTSDF) best supports completion of unobserved regions?
2. **Completion vs. transcription** — How much of the final structure is genuinely
   *inferred* (unobserved) vs. copied from observation? Can we measure this
   explicitly with an unobserved-region metric?
3. **Semantics under occlusion** — Can per-voxel semantic labels be predicted
   reliably for regions with no direct observation (e.g. facade vs. interior)?
4. **Generalization** — Does a model trained on one city/tile distribution
   generalize across districts and cities?
5. **Embodiment** — Is the completed voxel scene directly usable as an embodied
   environment (geometrically consistent, semantically navigable)?

## What makes this different from ordinary building reconstruction

- Typical aerial → LOD2 reconstruction (e.g. plane-fitting, polygonal/mesh
  generation) outputs **building geometry as meshes/polygons** and often assumes
  flat extruded walls. PointCraft outputs **discrete voxel occupancy + semantics**
  for the **whole scene** (buildings *and* ground/vegetation/road), suited to an
  embodied voxel world.
- We **explicitly target and evaluate unobserved-region completion**, rather than
  fitting primarily to observed roof structure.

## What makes this different from autonomous-driving SSC

- Standard Semantic Scene Completion (SemanticKITTI etc.) uses **street-level,
  forward-facing LiDAR** with a vehicle viewpoint. PointCraft uses **top-down
  airborne LiDAR at city scale**, where the observed/unobserved split is
  vertical (roofs seen, facades hidden) rather than range/occlusion-based.
- Class taxonomy and supervision differ: we supervise with **city-model
  (LOD2/CityGML) geometry**, not driving-scene voxel labels.

## Out of scope (for now)

- Photorealistic appearance / material reconstruction (optional later stretch).
- Real-time / streaming city-scale inference.
- Indoor structure.
