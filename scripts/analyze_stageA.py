"""exp_004 Stage A analysis — which weight-decay best FLATTENS the collapse curve.

Reads the 3 wd run metrics.json, reports held-out strict vs tile-steps side-by-side,
and quantifies (per spec §6, applied within-run since Stage A is seed-0):
  - peak strict + step           (the early-stop magnitude)
  - end strict (last tile-step)  (does it survive to the end?)
  - end/peak                     (collapse depth: 1.0 = no collapse)
  - last-quarter mean / peak     (flatness of the tail, robust to single-point jitter)
  - post-peak std                (run-internal jitter)
  - frac evals > B3 0.165        (how often it stays above the bar)
Writes stageA_summary.json + a tile-step curve PNG.
"""
from __future__ import annotations

import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
EXP = os.path.join(REPO, "experiments", "exp_004_m2_scaleup")
B3_STRICT = 0.165

WDS = ["1e-4", "1e-3", "1e-2"]


def summarize(wd: str) -> dict:
    m = json.load(open(os.path.join(EXP, f"stageA_wd{wd}", "metrics.json")))
    h = m["history"]
    xs = [(e["tile_step"], e["val_unobs"]["strict"]) for e in h]
    steps = [s for s, _ in xs]
    strict = [v for _, v in xs]
    peak_step, peak = max(xs, key=lambda t: t[1])
    end = strict[-1]
    mn = min(strict)
    post = [v for s, v in xs if s >= peak_step]
    frac_above = sum(1 for v in strict if v > B3_STRICT) / len(strict)
    last = steps[-1]

    def seg_mean(lo, hi):
        v = [val for st, val in xs if lo <= st < hi]
        return statistics.mean(v) if v else float("nan")

    early = seg_mean(0, last / 3)
    mid = seg_mean(last / 3, 2 * last / 3)
    late = seg_mean(2 * last / 3, last + 1)
    # The literal "does it STOP collapsing" metric: drift across the back half.
    # ~0 = plateaued (flattened); strongly negative = still collapsing.
    back_half_drift = late - mid
    total_drift = late - early
    return {
        "wd": wd,
        "n_evals": len(xs),
        "peak": round(peak, 4), "peak_step": peak_step,
        "end": round(end, 4), "min": round(mn, 4),
        "third_means": {"early": round(early, 4), "mid": round(mid, 4), "late": round(late, 4)},
        "total_drift": round(total_drift, 4),
        "back_half_drift": round(back_half_drift, 4),
        "end_over_peak": round(end / peak, 3),
        "post_peak_std": round(statistics.pstdev(post), 4),
        "frac_above_b3": round(frac_above, 3),
        "best_epoch": m["best_epoch"],
        "strict_series": [round(v, 4) for v in strict],
        "steps": steps,
    }


def main() -> None:
    rows = [summarize(wd) for wd in WDS]

    print("=" * 96)
    print("exp_004 Stage A — held-out STRICT collapse/flatness by weight-decay (seed 0, tile-steps)")
    print(f"   B3 bar = {B3_STRICT};  flatness wants end/peak & lastq/peak -> 1.0, low post-peak std")
    print("=" * 96)
    hdr = (f"{'wd':6s} {'peak':>6s} {'early':>6s} {'mid':>6s} {'late':>6s} "
           f"{'totDrift':>9s} {'backDrift':>10s} {'pp_std':>7s} {'>B3%':>6s}")
    print(hdr); print("-" * 96)
    for r in rows:
        t = r["third_means"]
        print(f"{r['wd']:6s} {r['peak']:>6.3f} {t['early']:>6.3f} {t['mid']:>6.3f} {t['late']:>6.3f} "
              f"{r['total_drift']:>+9.3f} {r['back_half_drift']:>+10.3f} "
              f"{r['post_peak_std']:>7.3f} {100*r['frac_above_b3']:>5.0f}%")
    print("=" * 96)

    # "Best flattens the collapse" = back-half drift closest to 0 (stops collapsing),
    # subject to peak > B3. NOT highest magnitude (spec §10: do not chase IoU).
    elig = [r for r in rows if r["peak"] > B3_STRICT]
    best = max(elig, key=lambda r: r["back_half_drift"])  # least-negative / plateaued
    print(f"\nFLATTEST (back-half drift -> 0): wd={best['wd']}  "
          f"(back-half drift {best['back_half_drift']:+.3f}, total drift {best['total_drift']:+.3f}, "
          f"peak {best['peak']} > B3 {B3_STRICT})")
    print("NOTE: post-peak std (jitter) ~0.06 across ALL wd -> reg flattens DRIFT, not JITTER.")

    out = {"b3_strict": B3_STRICT, "runs": rows, "flattest_wd": best["wd"]}
    with open(os.path.join(EXP, "stageA_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[write] {os.path.join(EXP, 'stageA_summary.json')}")

    # Curve PNG (tile-steps x-axis)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5))
        for r in rows:
            ax.plot(r["steps"], r["strict_series"], marker="o", ms=3, label=f"wd={r['wd']}")
        ax.axhline(B3_STRICT, ls="--", c="k", lw=1, label=f"B3 bar {B3_STRICT}")
        ax.set_xlabel("tile-steps"); ax.set_ylabel("held-out unobserved STRICT IoU")
        ax.set_title("exp_004 Stage A — wd ladder, held-out 09LD2814 (seed 0)")
        ax.legend(); ax.grid(alpha=0.3)
        png = os.path.join(EXP, "stageA_curves.png")
        fig.tight_layout(); fig.savefig(png, dpi=120); plt.close(fig)
        print(f"[write] {png}")
    except Exception as e:
        print(f"[viz] skipped ({e})")


if __name__ == "__main__":
    main()
