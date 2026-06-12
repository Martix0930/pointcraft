"""exp_004 Stage B analysis — QUANTIFY the cross-seed instability (not confirm stability).

Per the reframed purpose (README "Main conclusion DIRECTION"): measure two numbers over
seeds {0,1,2} at wd=1e-3 and lay them beside Stage A's seed-0 picture.

  (1) cross-seed peak-strict SPREAD = max-min of per-seed peak strict   vs fork-1 ~0.03
  (2) per-seed POST-PEAK STD (the jitter band)                          vs Stage A ~0.06

Seed 0 = the Stage A wd=1e-3 run (reused). Seeds 1,2 = Stage B runs.
Writes stageB_summary.json + a 3-seed tile-step curve PNG.
Applies the pre-committed decision rule and prints the Outcome it implies.
"""
from __future__ import annotations

import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
EXP = os.path.join(REPO, "experiments", "exp_004_m2_scaleup")
B3_STRICT = 0.165
FORK1_SPREAD = 0.03      # fork-1 cross-seed peak spread (exp_003)
STAGEA_JITTER = 0.06     # Stage A post-peak std, identical across wd

SEED_RUNS = {
    0: "stageA_wd1e-3",      # reused Stage A seed-0 run
    1: "stageB_wd1e-3_s1",
    2: "stageB_wd1e-3_s2",
}


def per_seed(seed: int, run_dir: str) -> dict:
    p = os.path.join(EXP, run_dir, "metrics.json")
    if not os.path.exists(p):
        return {"seed": seed, "run": run_dir, "missing": True}
    m = json.load(open(p))
    xs = [(e["tile_step"], e["val_unobs"]["strict"]) for e in m["history"]]
    peak_step, peak = max(xs, key=lambda t: t[1])
    post = [v for s, v in xs if s >= peak_step]
    return {
        "seed": seed, "run": run_dir, "missing": False,
        "peak": round(peak, 4), "peak_step": peak_step,
        "post_peak_std": round(statistics.pstdev(post), 4),
        "end": round(xs[-1][1], 4),
        "best_epoch": m["best_epoch"],
        "steps": [s for s, _ in xs],
        "strict_series": [round(v, 4) for _, v in xs],
    }


def main() -> None:
    rows = [per_seed(s, d) for s, d in SEED_RUNS.items()]
    present = [r for r in rows if not r["missing"]]
    missing = [r for r in rows if r["missing"]]
    if missing:
        print("WAITING on:", ", ".join(f"seed {r['seed']} ({r['run']})" for r in missing))
        if len(present) < 2:
            return

    print("=" * 84)
    print("exp_004 Stage B — cross-seed instability at wd=1e-3 (held-out 09LD2814 STRICT)")
    print("=" * 84)
    print(f"{'seed':5s} {'peak':>7s} {'@step':>7s} {'post-peak std':>14s} {'end':>7s} {'best_ep':>8s}")
    print("-" * 84)
    for r in present:
        print(f"{r['seed']:>5d} {r['peak']:>7.3f} {r['peak_step']:>7d} "
              f"{r['post_peak_std']:>14.3f} {r['end']:>7.3f} {r['best_epoch']:>8d}")
    print("-" * 84)

    peaks = [r["peak"] for r in present]
    spread = max(peaks) - min(peaks)
    jitters = [r["post_peak_std"] for r in present]
    print(f"\n(1) cross-seed peak-strict SPREAD = {spread:.3f}   (fork-1 ~{FORK1_SPREAD}; "
          f"{'>=' if spread >= FORK1_SPREAD else '<'} fork-1)")
    print(f"    per-seed peaks: {peaks}  -> all > B3 {B3_STRICT}? "
          f"{all(p > B3_STRICT for p in peaks)}")
    print(f"(2) per-seed POST-PEAK STD (jitter) = {jitters}   (Stage A ~{STAGEA_JITTER}; "
          f"mean {statistics.mean(jitters):.3f})")

    # Pre-committed decision rule (README "Main conclusion DIRECTION").
    spread_big = spread >= FORK1_SPREAD
    jitter_repro = statistics.mean(jitters) >= 0.8 * STAGEA_JITTER
    all_above_b3 = all(p > B3_STRICT for p in peaks)
    if spread_big and jitter_repro:
        outcome = ("OUTCOME 2 — instability is intrinsic. Cross-seed spread reproduces / "
                   "exceeds fork-1's and the jitter band matches Stage A; the "
                   "'data-diversity -> stability' premise is NOT supported. Scale-up "
                   "objective NOT achieved; do NOT chase the curve (no epochs/lr/dropout).")
    elif not spread_big and not jitter_repro and all_above_b3:
        outcome = ("OUTCOME 1 — peak stabilized. Spread < fork-1 AND jitter shrank AND every "
                   "seed > B3. (Stage A gave no reason to expect this — verify before claiming.)")
    elif not all_above_b3:
        outcome = ("OUTCOME 3 — a seed's peak fell below B3 0.165: scale-up weakened the "
                   "fork-1 'yes' in the larger-K regime. Record honestly, no spin.")
    else:
        outcome = ("MIXED — spread and jitter disagree; report both numbers and reason "
                   "per-cutoff. Do not force a single label.")
    print(f"\nIMPLIED VERDICT: {outcome}")

    out = {
        "wd": "1e-3", "b3_strict": B3_STRICT, "fork1_spread": FORK1_SPREAD,
        "stageA_jitter": STAGEA_JITTER,
        "seeds": [{k: r[k] for k in ("seed", "run", "peak", "peak_step",
                                     "post_peak_std", "end", "best_epoch")} for r in present],
        "cross_seed_peak_spread": round(spread, 4),
        "per_seed_post_peak_std": jitters,
        "all_peaks_above_b3": all_above_b3,
        "implied_outcome": outcome.split(" — ")[0],
    }
    with open(os.path.join(EXP, "stageB_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[write] {os.path.join(EXP, 'stageB_summary.json')}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5))
        for r in present:
            ax.plot(r["steps"], r["strict_series"], marker="o", ms=3, label=f"seed {r['seed']}")
        ax.axhline(B3_STRICT, ls="--", c="k", lw=1, label=f"B3 bar {B3_STRICT}")
        ax.set_xlabel("tile-steps"); ax.set_ylabel("held-out unobserved STRICT IoU")
        ax.set_title(f"exp_004 Stage B — wd=1e-3, 3 seeds (spread {spread:.3f})")
        ax.legend(); ax.grid(alpha=0.3)
        png = os.path.join(EXP, "stageB_curves.png")
        fig.tight_layout(); fig.savefig(png, dpi=120); plt.close(fig)
        print(f"[write] {png}")
    except Exception as e:
        print(f"[viz] skipped ({e})")


if __name__ == "__main__":
    main()
