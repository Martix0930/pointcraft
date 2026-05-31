# 99 — Glossary

- **Aerial LiDAR** — Airborne laser scanning that measures 3D points from above.
  Captures roofs/ground/vegetation tops densely; facades and occluded structure
  are largely missing (top-down viewpoint).

- **Partial occupancy** — The voxelized set of voxels that the aerial point cloud
  actually observes (the model **input**). Incomplete by construction.

- **Complete occupancy** — The full set of occupied voxels for the scene, including
  structure not directly observed (the **target**), derived from city models.

- **Semantic Scene Completion (SSC)** — The task of jointly predicting (a) which
  voxels are occupied and (b) their semantic class, from a partial observation.

- **LOD2** — A CityGML Level of Detail in which buildings have differentiated roof
  shapes and wall surfaces (but no fine facade/interior detail). Used as the
  complete-geometry supervision target.

- **CityGML** — An open standard data model/format for representing 3D city
  objects (buildings, terrain, vegetation, …) with defined Levels of Detail.

- **Voxel** — A volumetric pixel: a cube cell in a regular 3D grid. PointCraft uses
  1-block-scale voxels aligned to a shared per-tile grid.

- **Observed region** — Voxels covered by the partial input (intersection of
  partial occupancy with the target). Where the model has direct evidence.

- **Unobserved region** — Target-occupied voxels that were **never** observed by the
  aerial input (mainly facades and building volume). The headline completion target
  and the focus of M4 evaluation.

- **Embodied environment** — A simulatable, interactive 3D world an agent can move
  and act in. Here, the completed voxel scene instantiated as Minecraft.

- **Minecraft export** — Converting the completed voxel occupancy + semantics into
  Minecraft blocks / schematic so the scene can be loaded and explored in-game.
  A downstream demonstration, reusing existing tooling — not a research contribution.
