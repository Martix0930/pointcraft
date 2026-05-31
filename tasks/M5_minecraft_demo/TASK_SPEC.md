# M5 — Minecraft / Embodied Demo — TASK SPEC

Status: **not started**.

## Goal

Instantiate a completed voxel scene (occupancy + semantics) as a **Minecraft /
embodied environment** for visualization and interaction. Demonstration layer —
reuse existing tooling; keep it light.

## Scope (planned)

- Map semantic classes → Minecraft blocks.
- Convert completed voxel grid → schematic/world using existing libs
  (`mcschematic` / Amulet); reuse legacy export where possible.
- Optional: a short walkthrough capture / screenshots.

## Exclusions

- ❌ No new world-format code written from scratch.
- ❌ Not a research contribution — do not over-invest engineering here.

Depends on: **M3** (semantic predictions).
