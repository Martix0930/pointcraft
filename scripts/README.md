# scripts/

Entry-point scripts that wire together `src/pointcraft` modules into runnable
commands (e.g. building an M0 sample, running a baseline, training).

Conventions:
- Keep scripts thin — logic belongs in `src/pointcraft/`.
- One script = one clear action; document its usage at the top.
- Read config from `configs/` where applicable.

Existing legacy scripts (`tokyo_station.py`, `render_views.py`,
`inspect_lod2_coverage.py`) belong to the M1 deterministic pipeline.
