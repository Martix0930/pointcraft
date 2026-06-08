"""Multi-tile training loop for the M2 occupancy-completion U-Net (fork-1 G2).

Trains on K tiles and evaluates on a **held-out** tile, to answer the binary
generalization question (GENERALIZATION_TASK_SPEC §G2): does the model beat the
per-tile deterministic floor (B1) **and** the morphological shell (B3) on a tile it
never trained on?

Built on the single-tile `overfit.py` path (same candidate support / features /
labels / metrics), with three multi-tile-specific disciplines from the G2 exec spec:

  * **Memory (8 GB):** batch = 1 tile / step, **no** multi-tile batching. Only one
    tile's sparse tensor lives on the GPU at a time (built per step, freed after).
    Per-tile CPU arrays (support/feats/labels) are precomputed once — they are small.
  * **Tile-invariant features (§2 risk-2):** the height channels are normalised by a
    **fixed physical scale** (`z_scale`, metres) rather than the per-tile grid height
    `K` (which ranges 94→259 here). Otherwise the same physical structure yields
    different features per tile and the model keys onto a tile-specific `K`.
  * **No negative subsampling (§2 risk-1):** submanifold neighbourhoods break under
    downsampling, so we train on the full (border-kept) support and handle imbalance
    with a per-tile `pos_weight`.

Checkpoint selection / early reporting use the **held-out** IoU, never train.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from ..data.sparse import occupancy_logits_to_coords, to_sparse_tensor
from ..metrics import border_keep_mask, build_cutoff_masks, evaluate, load_sample
from ..models.completion_unet import OccupancyCompletionUNet
from .overfit import (
    DEFAULT_PROB_THRESHOLDS,
    build_candidate_support,
    build_features,
    build_labels,
)

#: default fixed height scale (m) for tile-invariant features (see module docstring).
DEFAULT_Z_SCALE = 50.0


@dataclass
class PreppedTile:
    """Per-tile CPU arrays + the loaded sample (built once, reused every epoch)."""

    tile_id: str
    sample: object
    support: np.ndarray
    feats: np.ndarray
    labels: np.ndarray
    keep: np.ndarray          # border-keep mask (True = used for loss)
    cutoffs: dict
    pos_weight: float


@dataclass
class GeneralizeResult:
    val_tile: str
    train_tiles: list
    metrics: dict                       # best held-out evaluate() dict (3 cutoffs)
    best_strict_iou: float
    best_threshold: float
    best_epoch: int
    train_iou_at_best: dict             # {tile_id: strict unobserved IoU} at best ckpt
    val_pred_coords: np.ndarray
    history: list = field(default_factory=list)
    n_params: int = 0
    z_scale: float = DEFAULT_Z_SCALE
    peak_cuda_mb: float = 0.0


def _prep_tile(path: str, *, z_scale: float, border_margin: int) -> PreppedTile:
    s = load_sample(path)
    support = build_candidate_support(s)
    feats = build_features(s, support, z_scale=z_scale)
    labels = build_labels(s, support)
    keep = border_keep_mask(support, s.grid, border_margin)
    cutoffs = build_cutoff_masks(s.coords_target, s.coords_partial, s.sem_target, s.grid)
    ltr = labels[keep]
    pos_weight = float((len(ltr) - ltr.sum()) / max(ltr.sum(), 1.0))
    return PreppedTile(
        tile_id=s.metadata["tile_id"], sample=s, support=support, feats=feats,
        labels=labels, keep=keep, cutoffs=cutoffs, pos_weight=pos_weight,
    )


def _eval_tile(model, pt: PreppedTile, *, border_margin: int, prob_thresholds,
               logit_thr, amp: bool, device: str, fixed_threshold: float | None = None):
    """Forward the full support; return (best_pred, best_res, best_prob).

    If ``fixed_threshold`` is given, use only that logit threshold (for scoring train
    tiles at the val-selected operating point). Otherwise sweep and pick best strict.
    """
    s = pt.sample
    x = to_sparse_tensor(pt.support, pt.feats, s.grid, device=device)
    model.eval()
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=amp):
        logits = model(x).features.reshape(-1).float().cpu().numpy()
    del x
    torch.cuda.empty_cache()

    if fixed_threshold is not None:
        pred = occupancy_logits_to_coords(pt.support, logits, threshold=fixed_threshold)
        res = evaluate(pred, s, cutoffs=pt.cutoffs, border_margin=border_margin)
        return pred, res, None

    best = None
    for p, t in zip(prob_thresholds, logit_thr):
        pred = occupancy_logits_to_coords(pt.support, logits, threshold=t)
        res = evaluate(pred, s, cutoffs=pt.cutoffs, border_margin=border_margin)
        iou = res["unobserved"]["strict"]["iou"]
        if best is None or iou > best[1]["unobserved"]["strict"]["iou"]:
            best = (pred, res, p)
    return best


def train_multi(
    train_npz: list[str],
    val_npz: str,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    base: int = 8,
    eval_every: int = 20,
    border_margin: int = 5,
    z_scale: float = DEFAULT_Z_SCALE,
    prob_thresholds=DEFAULT_PROB_THRESHOLDS,
    device: str = "cuda",
    amp: bool = True,
    seed: int = 0,
    log=print,
) -> GeneralizeResult:
    """Train on ``train_npz`` tiles (1 tile/step, shuffled each epoch), evaluate on
    the held-out ``val_npz`` every ``eval_every`` epochs; keep the best-by-held-out
    checkpoint. Returns the held-out metrics + per-train-tile IoU for gap diagnosis.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    log(f"[prep]    z_scale={z_scale} (tile-invariant height), border_margin={border_margin}")
    train = [_prep_tile(p, z_scale=z_scale, border_margin=border_margin) for p in train_npz]
    for pt in train:
        n_pos = int(pt.labels[pt.keep].sum())
        log(f"  train {pt.tile_id}: support {len(pt.support):,} "
            f"(kept {int(pt.keep.sum()):,}, pos {n_pos:,}, pos_weight {pt.pos_weight:.1f})")
    val = _prep_tile(val_npz, z_scale=z_scale, border_margin=border_margin)
    log(f"  HELD-OUT {val.tile_id}: support {len(val.support):,} "
        f"(pos {int(val.labels.sum()):,})")

    model = OccupancyCompletionUNet(in_channels=train[0].feats.shape[1], base=base).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    logit_thr = [float(np.log(p / (1.0 - p))) for p in prob_thresholds]

    history: list = []
    best_strict = -1.0
    best_state = None
    best_res: dict = {}
    best_prob = 0.5
    best_epoch = -1
    best_pred = np.zeros((0, 3), dtype=np.int32)

    def run_eval(epoch: int):
        nonlocal best_strict, best_state, best_res, best_prob, best_epoch, best_pred
        pred, res, prob = _eval_tile(
            model, val, border_margin=border_margin, prob_thresholds=prob_thresholds,
            logit_thr=logit_thr, amp=amp, device=device,
        )
        strict = res["unobserved"]["strict"]["iou"]
        # per-train-tile strict IoU at the val-selected threshold (gap diagnosis)
        vt = float(np.log(prob / (1.0 - prob)))
        train_iou = {}
        for pt in train:
            _, tr, _ = _eval_tile(
                model, pt, border_margin=border_margin, prob_thresholds=prob_thresholds,
                logit_thr=logit_thr, amp=amp, device=device, fixed_threshold=vt,
            )
            train_iou[pt.tile_id] = round(tr["unobserved"]["strict"]["iou"], 4)
        u = {c: round(res["unobserved"][c]["iou"], 4) for c in ("strict", "mid", "tolerant")}
        history.append({"epoch": epoch, "val_unobs": u, "val_comp_iou": round(res["completion"]["iou"], 4),
                        "best_prob": prob, "train_unobs_strict": train_iou,
                        "val_pred_voxels": int(pred.shape[0])})
        log(f"[ep {epoch:4d}] HELD-OUT {val.tile_id} unobs strict/mid/tol "
            f"{u['strict']}/{u['mid']}/{u['tolerant']} @p={prob}  "
            f"(bar B3=0.165, B1=0.146)  train {train_iou}")
        if strict > best_strict:
            best_strict, best_res, best_prob, best_epoch, best_pred = strict, res, prob, epoch, pred
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_train_iou.clear(); best_train_iou.update(train_iou)

    best_train_iou: dict = {}
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        last_loss = 0.0
        for idx in order:
            pt = train[int(idx)]
            tr = np.where(pt.keep)[0]
            x = to_sparse_tensor(pt.support[tr], pt.feats[tr], pt.sample.grid, device=device)
            y = torch.as_tensor(pt.labels[tr], device=device)
            pw = torch.tensor([pt.pos_weight], device=device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                out = model(x)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    out.features.reshape(-1).float(), y, pos_weight=pw)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            last_loss = float(loss.item())
            del x, y, out, loss
            torch.cuda.empty_cache()

        if epoch % eval_every == 0 or epoch == epochs:
            run_eval(epoch)
            if history:
                history[-1]["last_train_loss"] = round(last_loss, 4)

    peak_mb = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0
    return GeneralizeResult(
        val_tile=val.tile_id, train_tiles=[pt.tile_id for pt in train],
        metrics=best_res, best_strict_iou=float(best_strict), best_threshold=float(best_prob),
        best_epoch=int(best_epoch), train_iou_at_best=dict(best_train_iou),
        val_pred_coords=best_pred, history=history, n_params=int(n_params),
        z_scale=float(z_scale), peak_cuda_mb=round(peak_mb, 1),
    )
