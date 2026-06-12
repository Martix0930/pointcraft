"""Multi-tile training loop for the M2 occupancy-completion U-Net (fork-1 G2).

Trains on K tiles and evaluates on a **held-out** tile, to answer the binary
generalization question (GENERALIZATION_TASK_SPEC §G2): does the model beat the
per-tile deterministic floor (B1) **and** the morphological shell (B3) on a tile it
never trained on?

Built on the single-tile `overfit.py` path (same candidate support / features /
labels / metrics), with three multi-tile-specific disciplines from the G2 exec spec:

  * **Memory.** Two regimes had to be balanced on this box (8 GB GPU, host RAM often
    only ~5 GB free):
      - GPU: batch = 1 tile / step, **no** multi-tile batching; one tile's sparse
        tensor on the GPU at a time, freed after. `PYTORCH_CUDA_ALLOC_CONF=
        expandable_segments:True` (set by the runner) avoids fragmentation OOM with
        the variable tile sizes (support 1.9M→3.7M voxels).
      - Host RAM: keep only the **compact training arrays** resident per tile
        (support int32 / feats f32 / labels f32 / keep bool ≈ ~1 GB for all 5 tiles)
        — NOT the heavy `Sample` (coords_target/masks/…) and NOT the cutoff masks.
        The full `Sample` + cutoffs are reloaded **only at eval** (every `eval_every`
        epochs), transiently. An all-resident precompute exhausted host RAM; a fully
        lazy per-step recompute of `candidate_support` was ~20 s/step. This is the
        middle path: precompute once, hold compactly, reload `Sample` only to score.
  * **Tile-invariant features (§2 risk-2):** height channels normalised by a fixed
    physical scale (`z_scale`, metres), not the per-tile grid height `K` (94→259).
  * **No negative subsampling (§2 risk-1):** train on the full (border-kept) support;
    handle imbalance with a per-tile `pos_weight`.

Checkpoint selection / early reporting use the **held-out** IoU, never train.
"""
from __future__ import annotations

import gc
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
class TrainTile:
    """Compact, resident per-tile arrays for training (no heavy Sample / cutoffs)."""

    tile_id: str
    orig_npz: str          # path to reload the full Sample for eval scoring
    grid: object
    support: np.ndarray    # (N,3) int32
    feats: np.ndarray      # (N,C) float32
    labels: np.ndarray     # (N,) float32  (1 = candidate in target shell)
    keep: np.ndarray       # (N,) bool     (border-keep: True = used for loss)
    pos_weight: float


@dataclass
class GeneralizeResult:
    val_tile: str
    train_tiles: list
    metrics: dict
    best_strict_iou: float
    best_threshold: float
    best_epoch: int
    train_iou_at_best: dict
    val_pred_coords: np.ndarray
    history: list = field(default_factory=list)
    n_params: int = 0
    z_scale: float = DEFAULT_Z_SCALE
    peak_cuda_mb: float = 0.0


def _prep_train_tile(path: str, *, z_scale: float, border_margin: int) -> TrainTile:
    """Build the compact resident arrays for one tile; the heavy Sample is dropped."""
    s = load_sample(path)
    support = np.ascontiguousarray(build_candidate_support(s), dtype=np.int32)
    feats = build_features(s, support, z_scale=z_scale).astype(np.float32, copy=False)
    labels = build_labels(s, support).astype(np.float32, copy=False)
    keep = border_keep_mask(support, s.grid, border_margin)
    ltr = labels[keep]
    pos_weight = float((len(ltr) - ltr.sum()) / max(ltr.sum(), 1.0))
    tile = TrainTile(tile_id=s.metadata["tile_id"], orig_npz=path, grid=s.grid,
                     support=support, feats=feats, labels=labels, keep=keep,
                     pos_weight=pos_weight)
    del s
    gc.collect()
    return tile


def _eval_on(model, tile: TrainTile, *, border_margin: int, prob_thresholds, logit_thr,
             amp: bool, device: str, fixed_threshold: float | None = None):
    """Forward the full support; score with a freshly-reloaded Sample + cutoffs.

    The Sample (coords_target/masks/…) and cutoff masks are reloaded here, transiently,
    so they never stay resident during training. Returns (best_pred, best_res, prob).
    """
    s = load_sample(tile.orig_npz)
    cutoffs = build_cutoff_masks(s.coords_target, s.coords_partial, s.sem_target, s.grid)
    x = to_sparse_tensor(tile.support, tile.feats, tile.grid, device=device)
    model.eval()
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=amp):
        logits = model(x).features.reshape(-1).float().cpu().numpy()
    del x
    torch.cuda.empty_cache()

    if fixed_threshold is not None:
        pred = occupancy_logits_to_coords(tile.support, logits, threshold=fixed_threshold)
        res = evaluate(pred, s, cutoffs=cutoffs, border_margin=border_margin)
        out = (pred, res, None)
    else:
        best = None
        for p, t in zip(prob_thresholds, logit_thr):
            pred = occupancy_logits_to_coords(tile.support, logits, threshold=t)
            res = evaluate(pred, s, cutoffs=cutoffs, border_margin=border_margin)
            iou = res["unobserved"]["strict"]["iou"]
            if best is None or iou > best[1]["unobserved"]["strict"]["iou"]:
                best = (pred, res, p)
        out = best
    del s, cutoffs
    gc.collect()
    return out


def train_multi(
    train_npz: list[str],
    val_npz: str,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    base: int = 8,
    eval_every: int = 20,
    eval_every_steps: int | None = None,
    train_eval_every_steps: int | None = None,
    weight_decay: float = 0.0,
    border_margin: int = 5,
    z_scale: float = DEFAULT_Z_SCALE,
    prob_thresholds=DEFAULT_PROB_THRESHOLDS,
    device: str = "cuda",
    amp: bool = True,
    seed: int = 0,
    log=print,
) -> GeneralizeResult:
    """Train on ``train_npz`` (1 tile/step, shuffled each epoch), keep best-by-held-out.
    Compact resident arrays + eval-time Sample reload (see module docstring).

    Eval cadence: if ``eval_every_steps`` is set, the held-out is scored every N
    **tile-steps** (the cross-tile-count-comparable x-axis, exp_004) regardless of epoch
    boundaries; otherwise it falls back to every ``eval_every`` **epochs** (exp_003 path,
    unchanged). ``weight_decay`` is Adam L2 (0.0 = exp_003 behaviour).

    ``train_eval_every_steps`` (exp_004 host-RAM discipline): the per-tile **train_iou**
    diagnostic reloads a full Sample per train tile, the dominant eval-time host-RAM
    churn at K=10 tiles. When set, train_iou is computed only every N tile-steps; on the
    other (held-out-only) evals ``train_unobs_strict`` is recorded as ``None``. This does
    NOT touch the held-out ``val_unobs`` curve, the checkpoint selection, or the verdict —
    only the cadence of a secondary diagnostic. ``None`` = compute every eval (exp_003)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    logit_thr = [float(np.log(p / (1.0 - p))) for p in prob_thresholds]

    log(f"[prep]    z_scale={z_scale} (tile-invariant height), border_margin={border_margin}, "
        f"compact-resident + eval-time Sample reload")
    train = [_prep_train_tile(p, z_scale=z_scale, border_margin=border_margin) for p in train_npz]
    for t in train:
        log(f"  train {t.tile_id}: support {len(t.support):,} (kept {int(t.keep.sum()):,}, "
            f"pos {int(t.labels[t.keep].sum()):,}, pos_weight {t.pos_weight:.1f})")
    val = _prep_train_tile(val_npz, z_scale=z_scale, border_margin=border_margin)
    log(f"  HELD-OUT {val.tile_id}: support {len(val.support):,} (pos {int(val.labels.sum()):,})")

    model = OccupancyCompletionUNet(in_channels=train[0].feats.shape[1], base=base).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    history: list = []
    best_strict = -1.0
    best_res: dict = {}
    best_prob = 0.5
    best_epoch = -1
    best_pred = np.zeros((0, 3), dtype=np.int32)
    best_train_iou: dict = {}

    def run_eval(epoch: int, tile_step: int, last_loss: float):
        nonlocal best_strict, best_res, best_prob, best_epoch, best_pred
        pred, res, prob = _eval_on(model, val, border_margin=border_margin,
                                   prob_thresholds=prob_thresholds, logit_thr=logit_thr,
                                   amp=amp, device=device)
        strict = res["unobserved"]["strict"]["iou"]
        u = {c: round(res["unobserved"][c]["iou"], 4) for c in ("strict", "mid", "tolerant")}
        vt = float(np.log(prob / (1.0 - prob)))
        # Per-tile train_iou is the dominant eval-time host-RAM churn (one full-Sample
        # reload per train tile). Gate it to a coarser cadence when requested; the
        # held-out curve / checkpoint above are unaffected.
        do_train = (train_eval_every_steps is None
                    or tile_step % train_eval_every_steps == 0)
        train_iou = None
        if do_train:
            train_iou = {}
            for t in train:
                _, tr, _ = _eval_on(model, t, border_margin=border_margin,
                                    prob_thresholds=prob_thresholds, logit_thr=logit_thr,
                                    amp=amp, device=device, fixed_threshold=vt)
                train_iou[t.tile_id] = round(tr["unobserved"]["strict"]["iou"], 4)
        history.append({"epoch": epoch, "tile_step": tile_step, "val_unobs": u,
                        "val_comp_iou": round(res["completion"]["iou"], 4),
                        "best_prob": prob, "train_unobs_strict": train_iou,
                        "val_pred_voxels": int(pred.shape[0]), "last_train_loss": round(last_loss, 4)})
        log(f"[ep {epoch:4d} | step {tile_step:5d}] HELD-OUT {val.tile_id} unobs strict/mid/tol "
            f"{u['strict']}/{u['mid']}/{u['tolerant']} @p={prob}  "
            f"(bar B3=0.165, B1=0.146)  train {train_iou}")
        if strict > best_strict:
            best_strict, best_res, best_prob, best_epoch, best_pred = strict, res, prob, epoch, pred
            best_train_iou.clear()
            if train_iou is not None:
                best_train_iou.update(train_iou)

    tile_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        last_loss = 0.0
        for idx in order:
            t = train[int(idx)]
            tr = np.where(t.keep)[0]
            x = to_sparse_tensor(t.support[tr], t.feats[tr], t.grid, device=device)
            y = torch.as_tensor(t.labels[tr], device=device)
            pw = torch.tensor([t.pos_weight], device=device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                out = model(x)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    out.features.reshape(-1).float(), y, pos_weight=pw)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            last_loss = float(loss.item())
            tile_step += 1
            del x, y, out, loss
            torch.cuda.empty_cache()

            # exp_004: tile-step-based cadence (mid-epoch eval, cross-K-comparable x-axis)
            if eval_every_steps and tile_step % eval_every_steps == 0:
                run_eval(epoch, tile_step, last_loss)
                model.train()

        # exp_003 path: epoch-based cadence (only when step cadence is off)
        if eval_every_steps is None and (epoch % eval_every == 0 or epoch == epochs):
            run_eval(epoch, tile_step, last_loss)

    # ensure a final eval at the last tile-step if step-cadence didn't land on it
    if eval_every_steps and (not history or history[-1]["tile_step"] != tile_step):
        run_eval(epochs, tile_step, last_loss)

    peak_mb = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0
    return GeneralizeResult(
        val_tile=val.tile_id, train_tiles=[t.tile_id for t in train],
        metrics=best_res, best_strict_iou=float(best_strict), best_threshold=float(best_prob),
        best_epoch=int(best_epoch), train_iou_at_best=dict(best_train_iou),
        val_pred_coords=best_pred, history=history, n_params=int(n_params),
        z_scale=float(z_scale), peak_cuda_mb=round(peak_mb, 1),
    )
