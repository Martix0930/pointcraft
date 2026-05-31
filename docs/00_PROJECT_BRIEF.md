# 00 — Project Brief

## Aerial-to-Embodied Semantic Scene Completion

PointCraft takes **aerial sparse point clouds** and completes the **full urban 3D
structure** — voxel occupancy plus per-voxel semantic labels — then instantiates
the completed scene as an **embodied interactive environment** (Minecraft).

## Input

- Aerial LiDAR / sparse point cloud (airborne, top-down).
- Voxelized **partial** occupancy derived from that point cloud.
- Optional per-voxel/point features: height-above-ground, surface normal,
  intensity, fTSDF-like signed distance.

## Output

- **Complete** voxel occupancy (including structure never directly observed).
- **Per-voxel semantic labels** (building, roof, facade, ground, vegetation, road, …).
- Optional **Minecraft-compatible** embodied scene for visualization / interaction.

## Why aerial LiDAR's limited observation is the core problem

Airborne LiDAR observes the world from above. It captures roofs, open ground, and
vegetation tops densely, but **facades, building interiors/volumes, and occluded
regions are systematically missing**. A deterministic "rasterize what you see"
pipeline is therefore fundamentally capped: it cannot produce what was never
observed, and improving it requires buying better data rather than better methods.

We treat this under-observation as the **research problem itself**: *given partial
top-down observation, infer and complete the full structure.* This is a completion
/ generation task, not a transcription task — which is where learning adds value.

## LOD2 / CityGML supervision

PLATEAU and similar open city models provide **complete building geometry**
(LOD2 / CityGML) co-registered with the LiDAR. By voxelizing LOD2 we obtain a
**complete-occupancy + semantic target** for each tile, paired with the partial
LiDAR-derived input. This pairing (partial observation → complete target) is the
training signal and is the focus of milestone **M0**.

## Minecraft / embodied environment as downstream demonstration

Minecraft is a **discrete voxel world**, which makes it a natural output space for
voxel occupancy + semantics, and a **simulatable, embodied environment** for
downstream agents. It is a demonstration / instantiation target, **not** the
research contribution — format conversion and rendering reuse existing community
tooling and are deliberately kept lightweight.

## One-line claim

> From partial aerial observation, complete full urban voxel occupancy + semantics
> (supervised by LOD2/CityGML) and instantiate it as an embodied voxel world.
