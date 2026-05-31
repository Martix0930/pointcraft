# 06 — Decision Log

Append-only log of non-trivial project decisions. Newest at the bottom.

Template:

```
## YYYY-MM-DD - Decision title

Context:
Decision:
Reason:
Consequences:
Status:
```

---

## 2026-06-01 - Use repository documents as shared memory between Claude App and Claude Code

Context:
The project spans research planning (Claude App) and implementation (Claude Code).
Relying on long chat history as memory is lossy and does not transfer between
sessions or tools.

Decision:
The repository is the single source of truth. Stable knowledge lives in `docs/`;
per-milestone work lives in `tasks/M*/` with `TASK_SPEC.md`, `ACCEPTANCE.md`,
`CHECKLIST.md`, and `SESSION_LOG.md`. Decisions are logged here.

Reason:
Future sessions must be able to work from small, self-contained task specs instead
of replaying chat history. Documents are durable, diffable, and tool-agnostic.

Consequences:
- Every session reads the active task files before coding and updates
  `SESSION_LOG.md` after.
- Decisions and data-format changes must be recorded in `docs/`.

Status:
Adopted.

---

## 2026-06-01 - Keep legacy pipeline as M1 baseline under repo-root `pointcraft/`

Context:
An earlier deterministic LiDAR→voxel→Minecraft pipeline already exists in
`pointcraft/` (repo root). The new research code is scaffolded under
`src/pointcraft/`, creating a package-name overlap.

Decision:
Leave the legacy package in place as the **M1 deterministic baseline**; build new
research code under `src/pointcraft/`. `pyproject.toml` packages from `src/`.

Reason:
Avoids destructive refactoring of working baseline code; preserves a comparison
floor for learned models.

Consequences:
- Potential import ambiguity if both are on `sys.path`; keep imports explicit.
- A future decision may move/rename the legacy package once it is no longer run
  directly. Log it here when that happens.

Status:
Adopted (revisit when legacy pipeline is retired).
