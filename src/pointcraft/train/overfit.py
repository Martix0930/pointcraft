"""Single-tile overfit loop for the M2 occupancy-completion U-Net (Phase D).

First-step goal (per the M2 EXECUTION_PLAN): **prove the model + pipeline can learn**
by overfitting `09LD1874` and clearly beating the M1 B1 floor (strict unobserved-IoU
0.061), scored by the shared `pointcraft.metrics` under all three cutoffs.

Honesty of the task:
  * The **candidate support** the net classifies over is built **from the input
    only** — the M1 B1 extrusion volume (`naive_roof_extrusion`) unioned with the
    observed voxels. It never sees `coords_target`. So recovering the unobserved
    shell within it is genuine completion, and the IoU is comparable to B1.
  * Labels (occupied/free per candidate) DO use the target — that is supervision,
    not input. The support's target coverage caps recall at ~0.93 on this tile.

Pure-ish: numpy + torch + spconv; no lightning, no multi-tile, no semantic head.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from ..baseline.predictors import candidate_support
from ..data.sparse import occupancy_logits_to_coords, to_sparse_tensor
from ..metrics import border_keep_mask, build_cutoff_masks, evaluate
from ..metrics.evaluate import Sample
from ..models.completion_unet import OccupancyCompletionUNet

#: occupancy probabilities to sweep at eval (→ logit thresholds); reported best.
DEFAULT_PROB_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

FEATURE_NAMES = ["observed", "height_feat", "k_frac", "depth_below_top", "above_ground"]


def _keys(coords: np.ndarray, grid) -> np.ndarray:
    c = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    sj, sk = int(grid.shape[1]), int(grid.shape[2])
    return c[:, 0] * (sj * sk) + c[:, 1] * sk + c[:, 2]


def build_candidate_support(sample: Sample) -> np.ndarray:
    """Input-only candidate voxels: B1 extrusion ∪ observed (delegates to baseline)."""
    return candidate_support(sample.coords_partial, sample.grid)


def build_features(
    sample: Sample, support: np.ndarray, *, z_scale: float | None = None
) -> np.ndarray:
    """Per-candidate input features `(S, 5)` float32 (see FEATURE_NAMES). Input-only.

    ``z_scale`` controls the normalisation of the **height channels** (``k_frac``,
    ``depth_below_top``, ``above_ground``, ``height_feat``):

      * ``None`` (default) — divide by the per-tile grid height ``K`` / LiDAR height
        range. Keeps the single-tile overfit behaviour bit-for-bit. **Not
        tile-invariant**: ``K`` ranges 94→259 across the M2 tiles, so the same
        physical structure yields different features on different tiles.
      * a float (metres) — divide by this **fixed physical scale** instead, so the
        channels become "metres below column top / above ground / up", which mean
        the same thing on every tile (M2 generalization §2 risk-2: features must be
        tile-invariant or the model keys onto a tile-specific grid height). The
        voxel size is 1 m, so a voxel index *is* a metre offset.
    """
    grid = sample.grid
    S = np.asarray(support, dtype=np.int64)
    K = float(grid.shape[2])
    sk = _keys(S, grid)
    kp = _keys(sample.coords_partial, grid)

    # observed flag + carried partial feature (mean height) at observed voxels
    order = np.argsort(kp)
    pos = np.searchsorted(kp[order], sk)
    pos = np.clip(pos, 0, len(kp) - 1)
    matched = kp[order][pos] == sk
    observed = matched.astype(np.float32)
    height_feat = np.zeros(len(S), dtype=np.float32)
    z = sample.feats_partial[:, 0].astype(np.float32)
    zmin, zptp = float(z.min()), float(np.ptp(z)) or 1.0
    hf_denom = z_scale if z_scale is not None else zptp
    height_feat[matched] = ((z[order][pos][matched] - zmin) / hf_denom)

    k = S[:, 2].astype(np.float32)
    denom = float(z_scale) if z_scale is not None else K
    k_frac = k / denom

    # per-column top of the support and a ground reference (input-derived)
    col = S[:, 0] * int(grid.shape[1]) + S[:, 1]
    ucol, inv = np.unique(col, return_inverse=True)
    coltop = np.full(ucol.shape[0], -1e9, dtype=np.float64)
    np.maximum.at(coltop, inv, k)
    ground = float(np.percentile(S[:, 2], 1))
    depth_below_top = (coltop[inv] - k) / denom
    above_ground = (k - ground) / denom

    return np.stack(
        [observed, height_feat, k_frac, depth_below_top.astype(np.float32),
         above_ground.astype(np.float32)],
        axis=1,
    ).astype(np.float32)


def build_labels(sample: Sample, support: np.ndarray) -> np.ndarray:
    """Occupancy label `(S,)` float32: 1 if candidate ∈ target shell else 0."""
    kt = _keys(sample.coords_target, sample.grid)
    sk = _keys(support, sample.grid)
    return np.isin(sk, kt).astype(np.float32)


@dataclass
class OverfitResult:
    pred_coords: np.ndarray
    metrics: dict
    history: list = field(default_factory=list)
    support_size: int = 0
    n_params: int = 0
    best_strict_iou: float = 0.0
    best_threshold: float = 0.0


def _subsample_train(feats, labels, *, max_neg_ratio: float, seed: int):
    """Keep all positives + observed voxels + a capped sample of interior negatives.

    The candidate support is a *solid* B1 volume; its interior is ~90% negatives that
    don't fit in 8 GB VRAM. We train on a balanced subset (eval still runs on the
    FULL support, so IoU stays honest) — standard class-imbalance subsampling.
    Returns the row indices into the full support to train on.
    """
    pos = labels > 0
    observed = feats[:, 0] > 0  # 'observed' feature channel
    keep = pos | observed
    n_pos = int(pos.sum())
    neg_pool = np.where(~keep)[0]
    n_neg_keep = int(min(len(neg_pool), max_neg_ratio * max(n_pos, 1)))
    rng = np.random.default_rng(seed)
    sel_neg = rng.choice(neg_pool, size=n_neg_keep, replace=False)
    idx = np.concatenate([np.where(keep)[0], sel_neg])
    idx.sort()
    return idx


def train_overfit(
    sample: Sample,
    *,
    iters: int = 400,
    lr: float = 1e-3,
    base: int = 8,
    eval_every: int = 50,
    pos_weight: float | None = None,
    train_full: bool = True,
    border_margin: int = 0,
    max_neg_ratio: float = 2.0,
    prob_thresholds=DEFAULT_PROB_THRESHOLDS,
    device: str = "cuda",
    amp: bool = True,
    seed: int = 0,
    log=print,
) -> OverfitResult:
    """Overfit one tile; returns predicted coords + multi-cutoff metrics.

    The target shell ≈ the **boundary of the solid candidate support**, so the net
    must see dense neighbourhoods to learn it. With `train_full=True` (default) it
    trains and evaluates on the **full** support — additive skips keep this within
    8 GB. `train_full=False` falls back to subsampled-negative training (cheaper,
    but the subsampling holes corrupt the submanifold neighbourhoods and the model
    fails to reject unseen interior negatives — kept only as a low-memory escape).
    """
    torch.manual_seed(seed)
    support = build_candidate_support(sample)
    feats = build_features(sample, support)
    labels = build_labels(sample, support)
    x_full = to_sparse_tensor(support, feats, sample.grid, device=device)

    # G0: exclude the XY border band from the TRAINING loss (it carries the centroid
    # tile-crop contamination). Scoring excludes it too via evaluate(border_margin=).
    keep = border_keep_mask(support, sample.grid, border_margin)
    if border_margin and border_margin > 0:
        log(f"[border]  margin={border_margin}: {int((~keep).sum()):,} candidates "
            f"({100*(~keep).mean():.1f}%) excluded from loss + metrics")

    if train_full:
        tr = np.where(keep)[0]
        if border_margin and border_margin > 0:
            x_tr = to_sparse_tensor(support[tr], feats[tr], sample.grid, device=device)
        else:
            x_tr = x_full
        ltr = labels[tr]
        log(f"[support] full {support.shape[0]:,} (pos {int(labels.sum()):,}, "
            f"{100*labels.mean():.1f}%); training on {len(tr):,} (full support)")
    else:
        tr = _subsample_train(feats, labels, max_neg_ratio=max_neg_ratio, seed=seed)
        tr = tr[keep[tr]]
        x_tr = to_sparse_tensor(support[tr], feats[tr], sample.grid, device=device)
        ltr = labels[tr]
        log(f"[support] full {support.shape[0]:,}; train subset {tr.shape[0]:,}")
    y_tr = torch.as_tensor(ltr, device=device)
    if pos_weight is None:
        pos_weight = float((len(ltr) - ltr.sum()) / max(ltr.sum(), 1.0))
    pw = torch.tensor([pos_weight], device=device)

    model = OccupancyCompletionUNet(in_channels=feats.shape[1], base=base).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    # cutoff masks depend only on the sample (not the prediction) — build once.
    cutoffs = build_cutoff_masks(
        sample.coords_target, sample.coords_partial, sample.sem_target, sample.grid
    )
    logit_thr = [float(np.log(p / (1.0 - p))) for p in prob_thresholds]

    def eval_full():
        """Forward on the full support; sweep thresholds; return the best by strict IoU."""
        model.eval()
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=amp):
            logits = model(x_full).features.reshape(-1).float().cpu().numpy()
        best = None
        for p, t in zip(prob_thresholds, logit_thr):
            pred = occupancy_logits_to_coords(support, logits, threshold=t)
            res = evaluate(pred, sample, cutoffs=cutoffs, border_margin=border_margin)
            if best is None or res["unobserved"]["strict"]["iou"] > best[1]["unobserved"]["strict"]["iou"]:
                best = (pred, res, p)
        return best  # (pred, res, prob)

    history: list = []
    best_iou = 0.0
    best_pred = np.zeros((0, 3), dtype=np.int32)
    best_metrics: dict = {}
    best_thr = 0.5

    for it in range(1, iters + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            out = model(x_tr)
            loss = bce(out.features.reshape(-1).float(), y_tr)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if it % eval_every == 0 or it == iters:
            pred, res, prob = eval_full()
            strict = res["unobserved"]["strict"]["iou"]
            comp = res["completion"]["iou"]
            history.append({"iter": it, "loss": float(loss.item()),
                            "completion_iou": comp, "unobserved_strict_iou": strict,
                            "best_prob_threshold": prob, "pred_voxels": int(pred.shape[0])})
            log(f"[it {it:4d}] loss {loss.item():.4f}  comp_IoU {comp:.4f}  "
                f"unobs_strict {strict:.4f}  pred {pred.shape[0]:,}  @p={prob}")
            if strict > best_iou:
                best_iou, best_pred, best_metrics, best_thr = strict, pred, res, prob

    return OverfitResult(
        pred_coords=best_pred, metrics=best_metrics, history=history,
        support_size=int(support.shape[0]), n_params=int(n_params),
        best_strict_iou=float(best_iou), best_threshold=float(best_thr),
    )
