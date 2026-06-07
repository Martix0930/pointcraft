"""Phase 0 environment gate for M2: minimal sparse-conv fwd/bwd on real M0 data.

NOT a model. This validates that the learning stack (torch + spconv + CUDA) installs
and runs on *this* machine before any network code is written — the #1 failure mode
for 2020-2023 SSC code is spconv/CUDA version hell (see tasks/M2.../EXECUTION_PLAN.md
Phase 0). It loads a real M0 contract `.npz`, builds an spconv SparseConvTensor from
`coords_partial / feats_partial` on the shared grid, runs a tiny submanifold +
strided sparse conv, and backprops a dummy loss — confirming both feature-input and
parameter gradients flow.

Usage:
    .venv/Scripts/python.exe scripts/check_spconv_env.py \
        --npz outputs/m0/tokyo_citygml.npz

Exit 0 = gate passed. Prints exact torch/spconv/CUDA versions for the SESSION_LOG.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M2 Phase 0 spconv environment gate")
    ap.add_argument("--npz", default=os.path.join(REPO, "outputs", "m0", "tokyo_citygml.npz"),
                    help="M0 contract .npz to feed through a sparse conv")
    args = ap.parse_args(argv)

    import numpy as np
    import torch
    import torch.nn as nn
    import spconv
    import spconv.pytorch as spconv_pt

    print("=== versions ===")
    print("python    ", sys.version.split()[0])
    print("numpy     ", np.__version__)
    print("torch     ", torch.__version__)
    print("spconv    ", spconv.__version__)
    print("cuda build", torch.version.cuda)
    print("cuda avail", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available — spconv GPU path cannot be validated.")
        return 1
    device = torch.device("cuda")
    print("device    ", torch.cuda.get_device_name(0))

    # --- load a real M0 sample ---
    if not os.path.exists(args.npz):
        print(f"FAIL: npz not found: {args.npz}")
        return 1
    d = np.load(args.npz)
    coords = d["coords_partial"].astype(np.int32)        # (N,3) = (i,j,k)
    feats = d["feats_partial"].astype(np.float32)        # (N,C)
    meta = json.loads(str(d["metadata"]))
    I, J, K = (int(s) for s in meta["grid_shape"])
    C = feats.shape[1]
    N = coords.shape[0]
    print(f"\n=== sample {meta['tile_id']} ===")
    print(f"grid_shape (I,J,K) = ({I},{J},{K}); partial voxels N={N:,}; feat dim C={C}")

    # --- build spconv SparseConvTensor (indices = [batch, i, j, k]) ---
    batch_idx = np.zeros((N, 1), dtype=np.int32)
    indices = np.concatenate([batch_idx, coords], axis=1)  # (N,4)
    indices_t = torch.from_numpy(indices).to(device)
    feats_t = torch.from_numpy(feats).to(device).requires_grad_(True)
    x = spconv_pt.SparseConvTensor(feats_t, indices_t, [I, J, K], batch_size=1)
    print(f"SparseConvTensor: features {tuple(x.features.shape)}, spatial {x.spatial_shape}")

    # --- tiny encoder: submanifold conv (keeps sparsity) + strided downsample ---
    net = spconv_pt.SparseSequential(
        spconv_pt.SubMConv3d(C, 16, 3, bias=False, indice_key="subm0"),
        nn.ReLU(),
        spconv_pt.SparseConv3d(16, 16, 3, stride=2, bias=False, indice_key="down0"),
        nn.ReLU(),
    ).to(device)

    # --- forward ---
    out = net(x)
    print(f"\n=== forward ===")
    print(f"out features {tuple(out.features.shape)}, out spatial {out.spatial_shape}, "
          f"out active voxels {out.features.shape[0]:,}")

    # --- backward on a dummy loss ---
    loss = out.features.float().pow(2).mean()
    loss.backward()
    feat_grad_ok = feats_t.grad is not None and torch.isfinite(feats_t.grad).all().item()
    param_grads = [p.grad for p in net.parameters() if p.requires_grad]
    param_grad_ok = len(param_grads) > 0 and all(
        g is not None and torch.isfinite(g).all().item() for g in param_grads
    )
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated() / 1e6
    print(f"\n=== backward ===")
    print(f"loss {loss.item():.6f}")
    print(f"input-feature grad flows: {feat_grad_ok} "
          f"(|grad| sum={feats_t.grad.abs().sum().item():.4g})")
    print(f"all {len(param_grads)} param grads finite: {param_grad_ok}")
    print(f"peak CUDA mem: {mem:.0f} MB")

    ok = feat_grad_ok and param_grad_ok and out.features.shape[0] > 0
    print(f"\n=== GATE {'PASSED' if ok else 'FAILED'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
