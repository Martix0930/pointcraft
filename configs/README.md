# configs/

Declarative experiment/run configurations (YAML).

Experiment-config naming: `<milestone>_<model>_<variant>.yaml`
(e.g. `m2_occ_unet_v0.yaml`). See `docs/03_EXPERIMENT_PROTOCOL.md`. Configs should
be self-contained and versioned; experiments record the exact config they used.

Load configs with `pointcraft.utils.config.load_config(path)` (YAML → dict).

## Data / run configs

- `tokyo_station.yaml` — data paths + params for the M1 deterministic baseline run
  (`scripts/tokyo_station.py`). Holds **absolute, machine-specific** paths to large
  local datasets that live **outside** the repo (not committed). Edit for your
  machine; CLI flags (`--las`/`--lod2`/`--out`/`--name`) override it.
