"""Masks + .npz sample writer per docs/02_DATA_CONTRACT.md (M0-4).

Brings the partial input and the LOD2 target (both on the SAME `VoxelGrid`)
together into one on-disk training sample:

  * `compute_masks` — derive the required observed / unobserved masks (D4):
        observed   = coords_target ∈ coords_partial
        unobserved = occupied_target ∧ ¬observed   (= 1 - observed here, since
                     every stored target voxel is occupied)
  * `build_metadata` — assemble the metadata block from the grid + provenance.
  * `write_sample_npz` — write every contract field with the exact dtype/shape,
        plus metadata as a JSON string (no pickle → loadable with numpy.load
        without allow_pickle).

Pure numpy + json; no learning.
"""
from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np

from ..voxelization import VoxelGrid
from .partial import FEATURE_LAYOUT_V01

DATASET_VERSION = "v0.1"


def _ravel(coords: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """Flatten (N,3) in-bounds voxel indices to unique int64 keys for set ops."""
    c = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    sj, sk = int(shape[1]), int(shape[2])
    return c[:, 0] * (sj * sk) + c[:, 1] * sk + c[:, 2]


def compute_masks(
    coords_target: np.ndarray, coords_partial: np.ndarray, grid: VoxelGrid
) -> tuple[np.ndarray, np.ndarray]:
    """Per-target-voxel observed / unobserved masks (uint8), aligned to
    `coords_target` rows. Both arrays must hold in-bounds indices on `grid`."""
    key_t = _ravel(coords_target, grid.shape)
    key_p = _ravel(coords_partial, grid.shape)
    observed = np.isin(key_t, key_p).astype(np.uint8)
    unobserved = (1 - observed).astype(np.uint8)
    return observed, unobserved


def build_metadata(
    grid: VoxelGrid,
    *,
    tile_id: str,
    crs: str,
    source_files: Sequence[str],
    dataset_version: str = DATASET_VERSION,
    feature_layout: Sequence[str] = tuple(FEATURE_LAYOUT_V01),
) -> dict[str, Any]:
    """Assemble the metadata block (all contract keys) from grid + provenance."""
    origin = grid.origin.astype(float)
    shape = grid.shape.astype(int)
    maxc = origin + shape * grid.voxel_size
    return {
        "tile_id": str(tile_id),
        "voxel_size": float(grid.voxel_size),
        "origin": origin.tolist(),
        "bounds": [
            float(origin[0]), float(origin[1]), float(origin[2]),
            float(maxc[0]), float(maxc[1]), float(maxc[2]),
        ],
        "grid_shape": shape.tolist(),
        "crs": str(crs),
        "source_files": [str(s) for s in source_files],
        "dataset_version": str(dataset_version),
        "feature_layout": [str(f) for f in feature_layout],
    }


def write_sample_npz(
    path: str,
    *,
    coords_partial: np.ndarray,
    feats_partial: np.ndarray,
    coords_target: np.ndarray,
    occ_target: np.ndarray,
    sem_target: np.ndarray,
    observed_mask: np.ndarray,
    unobserved_mask: np.ndarray,
    metadata: dict[str, Any],
) -> str:
    """Write one paired voxel sample to `path` (.npz) per the data contract.

    All arrays are cast to their contract dtypes. `metadata` is stored as a 0-d
    string array of JSON (key ``metadata``); load with
    ``json.loads(str(np.load(path)["metadata"]))``.
    """
    np.savez_compressed(
        path,
        coords_partial=np.asarray(coords_partial, dtype=np.int32).reshape(-1, 3),
        feats_partial=np.asarray(feats_partial, dtype=np.float32).reshape(
            -1, len(metadata["feature_layout"])
        ),
        coords_target=np.asarray(coords_target, dtype=np.int32).reshape(-1, 3),
        occ_target=np.asarray(occ_target, dtype=np.uint8).reshape(-1),
        sem_target=np.asarray(sem_target, dtype=np.int64).reshape(-1),
        observed_mask=np.asarray(observed_mask, dtype=np.uint8).reshape(-1),
        unobserved_mask=np.asarray(unobserved_mask, dtype=np.uint8).reshape(-1),
        metadata=np.array(json.dumps(metadata, ensure_ascii=False)),
    )
    return path


def load_sample_metadata(npz) -> dict[str, Any]:
    """Decode the metadata JSON from a loaded .npz (or its ``metadata`` entry)."""
    entry = npz["metadata"] if hasattr(npz, "__getitem__") else npz
    return json.loads(str(entry))
