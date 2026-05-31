# smoke fixture (placeholder)

Reserved for tiny end-to-end **smoke-run** fixtures — the smallest inputs that
exercise a full path (e.g. LiDAR + LOD2 → paired `.npz`, or a baseline run) quickly
in CI / locally.

Empty for now. When added, each fixture must stay tiny and ship its own note:
- what it represents,
- coordinate system,
- voxel size / bounds,
- which test(s) use it.

Do **not** place real or large datasets here (see `CLAUDE.md` → "Test data policy").
