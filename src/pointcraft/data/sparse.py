"""M0 contract `.npz` ↔ spconv sparse tensor adapter (M2 Phase B).

Bridges the data contract (sparse voxel coords on a shared :class:`VoxelGrid`) to
spconv's :class:`SparseConvTensor`, and back. Keeping the round-trip lossless — and
the grid metadata intact — is what lets M2 predictions stay **explicit, thresholdable
and world-placeable** (decision D9), so M3/M4/M5 can consume them.

`torch` + `spconv` are **optional** deps (the M2 `.venv` only), so they are imported
**lazily inside the functions** and this module is intentionally **not** re-exported
from ``pointcraft.data.__init__`` — ``import pointcraft.data`` must stay torch-free
for M0/M1 and the global interpreter.

Index convention (must match the contract and `docs/07_GOTCHAS.md`):
    contract coords are ``(i, j, k) ↔ (x, y, z)``; the spconv ``indices`` tensor is
    ``[batch, i, j, k]`` (batch column **prepended**) with ``spatial_shape=[I,J,K]``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..voxelization import VoxelGrid

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import spconv.pytorch as spconv


def to_sparse_tensor(
    coords: np.ndarray,
    feats: np.ndarray,
    grid: VoxelGrid,
    *,
    batch_size: int = 1,
    batch_index: np.ndarray | None = None,
    device: str = "cuda",
):
    """Build an spconv ``SparseConvTensor`` from contract coords + features.

    Args:
        coords:      (N, 3) int voxel indices ``(i, j, k)`` on ``grid`` (in-bounds).
        feats:       (N, C) per-voxel features (float32).
        grid:        the shared :class:`VoxelGrid` — supplies ``spatial_shape``.
        batch_size:  number of samples packed in this tensor.
        batch_index: optional (N,) batch id per voxel (default all-zeros → one tile).
        device:      torch device string.

    Returns:
        ``spconv.pytorch.SparseConvTensor`` with ``features`` (N, C) and
        ``indices`` (N, 4) = ``[batch, i, j, k]``.
    """
    import torch
    import spconv.pytorch as spconv

    coords = np.asarray(coords, dtype=np.int32).reshape(-1, 3)
    feats = np.asarray(feats, dtype=np.float32).reshape(coords.shape[0], -1)
    n = coords.shape[0]
    if batch_index is None:
        bcol = np.zeros((n, 1), dtype=np.int32)
    else:
        bcol = np.asarray(batch_index, dtype=np.int32).reshape(n, 1)
    indices = np.concatenate([bcol, coords], axis=1)  # (N, 4): batch,i,j,k

    spatial_shape = [int(s) for s in grid.shape]
    feats_t = torch.as_tensor(feats, device=device)
    indices_t = torch.as_tensor(indices, device=device)
    return spconv.SparseConvTensor(feats_t, indices_t, spatial_shape, batch_size=batch_size)


def sparse_input_from_sample(sample, *, device: str = "cuda"):
    """Convenience: a :class:`pointcraft.metrics.Sample` → partial-input sparse tensor.

    ``sample`` only needs ``.coords_partial``, ``.feats_partial`` and ``.grid``
    (duck-typed), so this works with the metrics ``Sample`` without a hard import.
    """
    return to_sparse_tensor(
        sample.coords_partial, sample.feats_partial, sample.grid, device=device
    )


def coords_from_sparse_tensor(x, *, batch: int | None = None) -> np.ndarray:
    """Recover contract coords ``(i, j, k)`` from a ``SparseConvTensor``'s indices.

    Drops the prepended batch column. If ``batch`` is given, returns only that
    sample's voxels; otherwise returns all rows (single-tile use). The inverse of
    the ``[batch, i, j, k]`` packing done by :func:`to_sparse_tensor`.
    """
    idx = x.indices.detach().cpu().numpy()
    if batch is not None:
        idx = idx[idx[:, 0] == batch]
    return idx[:, 1:4].astype(np.int32)


def occupancy_logits_to_coords(
    coords: np.ndarray,
    logits,
    *,
    threshold: float = 0.0,
) -> np.ndarray:
    """Threshold per-voxel occupancy logits → predicted occupied contract coords.

    Used to emit a learned model's output in the **data-contract coords format**
    (D9): given the candidate voxel ``coords`` (e.g. a decoder's active sites) and
    their occupancy ``logits`` (raw, pre-sigmoid; ``threshold=0`` ↔ prob 0.5),
    return the subset predicted occupied — a plain ``(P, 3)`` int32 array that
    `pointcraft.metrics.evaluate` and any world-placement code can consume.
    """
    import torch

    coords = np.asarray(coords, dtype=np.int32).reshape(-1, 3)
    if isinstance(logits, torch.Tensor):
        keep = (logits.detach().reshape(-1) > threshold).cpu().numpy()
    else:
        keep = np.asarray(logits).reshape(-1) > threshold
    return coords[keep]
