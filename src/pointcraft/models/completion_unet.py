"""Sparse-conv occupancy-completion U-Net (M2 Phase C).

A small submanifold U-Net that predicts **per-voxel occupancy** over a fixed
candidate support (the active set of the input :class:`SparseConvTensor`). The net
does **not** invent new voxels: the caller supplies a candidate support derived
**from the input only** (e.g. the M1 B1 extrusion volume unioned with the observed
voxels), and the network classifies each candidate as occupied (target shell) or
free. This keeps "completion" honest — the support never peeks at the target — while
letting the model recover the unobserved facade/volume structure within it.

Output stays in the **data-contract representation** (decision D9): the returned
:class:`SparseConvTensor` carries one occupancy **logit** per candidate voxel,
aligned with the input indices ``[batch, i, j, k]``, so predictions are
thresholdable (`pointcraft.data.sparse.occupancy_logits_to_coords`) and
world-placeable on the shared grid. Occupancy head only — a semantic head (M3) can
be added by widening the head's out-channels without touching the backbone.

torch + spconv are M2-venv-only deps; this module is under ``pointcraft.models`` and
is imported only by training/eval code, never by ``import pointcraft``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import spconv.pytorch as spconv


def _subm(in_ch: int, out_ch: int, key: str, ksize: int = 3) -> spconv.SparseSequential:
    """Submanifold conv → BN → ReLU (keeps the active set fixed)."""
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_ch, out_ch, ksize, bias=False, indice_key=key),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


def _down(in_ch: int, out_ch: int, key: str) -> spconv.SparseSequential:
    """Strided sparse conv (×2 downsample) → BN → ReLU."""
    return spconv.SparseSequential(
        spconv.SparseConv3d(in_ch, out_ch, 3, stride=2, bias=False, indice_key=key),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


def _up(in_ch: int, out_ch: int, key: str) -> spconv.SparseSequential:
    """Inverse sparse conv (×2 upsample) → BN → ReLU; ``key`` matches its `_down`."""
    return spconv.SparseSequential(
        spconv.SparseInverseConv3d(in_ch, out_ch, 3, bias=False, indice_key=key),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


class OccupancyCompletionUNet(nn.Module):
    """Small 2-level submanifold U-Net; occupancy logit per candidate voxel.

    Args:
        in_channels: input feature dim per candidate voxel.
        base:        stem channel width (encoder widths = base, 2·base, 4·base).
        out_channels: head outputs (1 = occupancy; widen later for semantics, M3).
    """

    def __init__(self, in_channels: int, *, base: int = 16, out_channels: int = 1):
        super().__init__()
        c0, c1, c2 = base, base * 2, base * 4

        self.stem = _subm(in_channels, c0, "subm0")

        self.down1 = _down(c0, c1, "spconv1")
        self.enc1 = _subm(c1, c1, "subm1")
        self.down2 = _down(c1, c2, "spconv2")
        self.enc2 = _subm(c2, c2, "subm2")

        self.up2 = _up(c2, c1, "spconv2")     # back to scale 1
        self.dec1 = _subm(c1, c1, "subm1d")
        self.up1 = _up(c1, c0, "spconv1")     # back to scale 0 (the support)
        self.dec0 = _subm(c0, c0, "subm0d")

        self.head = spconv.SubMConv3d(c0, out_channels, 1, bias=True, indice_key="head")

    @staticmethod
    def _add(a: spconv.SparseConvTensor, b: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        """Add features of two tensors that share the same active set/order.

        Additive skips (vs concat) keep the full-resolution decoder tensors at the
        stem width — important on an 8 GB GPU where the millions of full-res voxels
        dominate memory.
        """
        return a.replace_feature(a.features + b.features)

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        s0 = self.stem(x)              # scale 0 (candidate support)
        d1 = self.enc1(self.down1(s0))  # scale 1
        d2 = self.enc2(self.down2(d1))  # scale 2

        u2 = self.up2(d2)             # -> scale 1, same set as d1
        u2 = self.dec1(self._add(u2, d1))
        u1 = self.up1(u2)             # -> scale 0, same set as s0 (the support)
        u1 = self.dec0(self._add(u1, s0))

        return self.head(u1)          # features: (N_support, out_channels) logits
