# `data/` — local large datasets (NOT committed)

Put **downloaded / large / raw datasets here.** This whole directory is
**git-ignored** (see `.gitignore`); only this `README.md` is tracked, so the
convention is recorded without ever committing heavy data. This is the project's
answer to "where do I put the big files?" — here, not in `test_data/`.

> `test_data/` is the opposite: only **tiny committed fixtures** for tests.
> Real LiDAR / LOD2 / DEM / checkpoints / generated `.npz` must live under `data/`
> (or another ignored local path), never in `test_data/`. See `CLAUDE.md`
> "Test data policy".

## Layout

```
data/
├─ raw/                 # untouched source data exactly as downloaded
│  ├─ lidar/            # aerial LiDAR tiles (.las / .laz)
│  ├─ lod2/             # PLATEAU / CityGML / LOD2 mesh tiles (.obj + .mtl + textures)
│  └─ dem/              # terrain / DEM (for a future height-above-ground version)
├─ interim/             # intermediate artifacts (optional)
└─ processed/           # derived datasets, e.g. M0 .npz samples (optional)
```

You don't have to use every folder — `raw/lidar` and `raw/lod2` are the two M0
actually needs.

## Pointing the code at your data

Configs (e.g. `configs/tokyo_station.yaml`) hold the input paths. Two options:

1. **Keep data outside the repo** (current default): leave the absolute paths in
   the config as-is (e.g. `D:/.../三维GIS/09LD1874.las`). Nothing to move.
2. **Move data under `data/`**: drop files into `data/raw/lidar` etc. and edit the
   config to point here. Relative paths in a config resolve against the config's
   own folder, so from `configs/` use `../data/raw/lidar/xxx.las`, or just use an
   absolute path. CLI flags `--las / --lod2` override the config either way.

## Why ignored

Raw LiDAR, PLATEAU/LOD2, CityGML, OBJ, NPZ, checkpoints and generated outputs are
large and/or non-diffable. Committing them would bloat the repo. Keep them local;
the repo stays small and the data contract (`docs/02_DATA_CONTRACT.md`) plus the
configs are enough to reproduce results from your own copy of the data.
