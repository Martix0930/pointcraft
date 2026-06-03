"""M1 deterministic baseline: run B1 + B2 on an M0 sample and record results.

Loads one M0 contract `.npz`, runs the two no-NN predictors, scores both under the
three mask cutoffs (strict/mid/tolerant), prints the floor (B1) / ceiling (B2), and
writes an experiment record (`README.md` + `metrics.json`) per
`docs/03_EXPERIMENT_PROTOCOL.md`.

Usage:
    python scripts/run_m1_baseline.py \
        --npz outputs/m0/tokyo_citygml.npz \
        --exp experiments/exp_001_m1_baseline
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from pointcraft.baseline import footprint_volume_fill, naive_roof_extrusion  # noqa: E402
from pointcraft.baseline.predictors import estimate_ground_k  # noqa: E402
from pointcraft.metrics import build_cutoff_masks, evaluate, load_sample  # noqa: E402


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO
        ).decode().strip()
    except Exception:
        return "unknown"


def _fmt_unobs(result: dict) -> str:
    return "  ".join(
        f"{n}={result['unobserved'][n]['iou']:.4f}"
        for n in ("strict", "mid", "tolerant")
        if n in result["unobserved"]
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M1 deterministic baseline B1 + B2")
    ap.add_argument("--npz", default=os.path.join(REPO, "outputs/m0/tokyo_citygml.npz"))
    ap.add_argument("--exp", default=os.path.join(REPO, "experiments/exp_001_m1_baseline"))
    args = ap.parse_args(argv)

    sample = load_sample(args.npz)
    meta = sample.metadata
    cutoffs = build_cutoff_masks(
        sample.coords_target, sample.coords_partial, sample.sem_target, sample.grid
    )
    ground_k = estimate_ground_k(sample.coords_partial, sample.grid)

    # B1 — observation-only floor.
    b1_pred = naive_roof_extrusion(sample.coords_partial, sample.grid, ground_k=ground_k)
    b1 = evaluate(b1_pred, sample, cutoffs=cutoffs)

    # B2 — footprint-informed upper reference (peeks at the target footprint).
    b2_pred = footprint_volume_fill(sample.coords_target, sample.grid, mode="shell")
    b2 = evaluate(b2_pred, sample, cutoffs=cutoffs)

    print(f"[sample]  {meta['tile_id']}  target={len(sample.coords_target):,}  "
          f"dataset_version={meta['dataset_version']}  ground_k={ground_k}")
    print(f"[B1 floor]   voxels={b1_pred.shape[0]:,}  completion IoU={b1['completion']['iou']:.4f}")
    print(f"             unobserved IoU  {_fmt_unobs(b1)}")
    print(f"[B2 ceiling] voxels={b2_pred.shape[0]:,}  completion IoU={b2['completion']['iou']:.4f}")
    print(f"             unobserved IoU  {_fmt_unobs(b2)}")

    metrics = {
        "tile_id": meta["tile_id"],
        "dataset_version": meta["dataset_version"],
        "grid_shape": meta["grid_shape"],
        "code_commit": _git_commit(),
        "cutoff_definitions": {
            "strict": "facade exact mid-wall (v0.2 stored mask; ~35% facade observed)",
            "mid": "facade XY+-1 hit, mid-wall only (~63% facade observed)",
            "tolerant": "facade XY+-1 anywhere (~67%, physical-grazing line, D7)",
        },
        "cutoff_unobserved_fraction": {
            n: float(cutoffs[n].mean()) for n in cutoffs
        },
        "baselines": {
            "B1_naive_roof_extrusion": {
                "role": "observation-only floor (uses no target)",
                "params": {"ground_k": ground_k, "min_building_height": 3},
                "n_pred": int(b1_pred.shape[0]),
                **b1,
            },
            "B2_footprint_volume_fill_shell": {
                "role": "footprint-informed upper reference (PEEKS at target footprint)",
                "params": {"mode": "shell"},
                "n_pred": int(b2_pred.shape[0]),
                **b2,
            },
        },
    }

    os.makedirs(args.exp, exist_ok=True)
    with open(os.path.join(args.exp, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    _write_readme(os.path.join(args.exp, "README.md"), metrics)
    print(f"[write]   {args.exp}/metrics.json + README.md")
    return 0


def _write_readme(path: str, m: dict) -> None:
    b1 = m["baselines"]["B1_naive_roof_extrusion"]
    b2 = m["baselines"]["B2_footprint_volume_fill_shell"]

    def row(name, r):
        u = r["unobserved"]
        c = r["completion"]
        return (f"| {name} | {c['iou']:.4f} | {u['strict']['iou']:.4f} | "
                f"{u['mid']['iou']:.4f} | {u['tolerant']['iou']:.4f} |")

    lines = [
        f"# exp_001 — M1 deterministic baseline ({m['tile_id']})",
        "",
        "Floor (B1, observation only) and ceiling (B2, footprint-informed upper",
        "reference — **it peeks at the target footprint, not a fair predictor**) for",
        "occupancy completion on the M0 contract sample. Both are no-NN. Reproduce:",
        "",
        "```",
        "python scripts/run_m1_baseline.py",
        "```",
        "",
        f"- dataset_version: `{m['dataset_version']}`  ·  grid `{m['grid_shape']}`  ·  "
        f"code `{m['code_commit']}`",
        f"- unobserved fraction per cutoff: "
        + ", ".join(f"{n} {m['cutoff_unobserved_fraction'][n]*100:.1f}%"
                    for n in ('strict', 'mid', 'tolerant')),
        "",
        "## Results — IoU",
        "",
        "| baseline | completion | unobserved (strict) | unobserved (mid) | unobserved (tolerant) |",
        "|----------|-----------|---------------------|------------------|-----------------------|",
        row("B1 floor", b1),
        row("B2 ceiling", b2),
        "",
        "## Reading the numbers",
        "",
        f"- **B1 is the honest floor.** Naive solid extrusion recovers facade only",
        f"  under the roof footprint; its unobserved IoU is **low** (strict "
        f"{b1['unobserved']['strict']['iou']:.3f}) — high recall, low precision "
        f"(it over-fills the hollow shell interior). A low B1 is a *successful* B1.",
        f"- **B2 is the ceiling**, not a competitor: knowing the footprint + height,",
        f"  a shell reconstruction reaches strict unobserved IoU "
        f"{b2['unobserved']['strict']['iou']:.3f}.",
        f"- **M2 must land between these** under the *same* cutoffs. The ceiling beats",
        f"  the floor under all three cutoffs (the conclusion does not flip as the",
        f"  observation line moves), which is the M1↔M4 contract (D8).",
        "",
        "## Per-class recall (target side)",
        "",
        "| baseline | ground (1) | roof (3) | facade (4) |",
        "|----------|-----------|----------|------------|",
        f"| B1 floor | {b1['per_class_recall'].get('1', b1['per_class_recall'].get(1, 0)):.3f} | "
        f"{b1['per_class_recall'].get('3', b1['per_class_recall'].get(3, 0)):.3f} | "
        f"{b1['per_class_recall'].get('4', b1['per_class_recall'].get(4, 0)):.3f} |",
        f"| B2 ceiling | {b2['per_class_recall'].get('1', b2['per_class_recall'].get(1, 0)):.3f} | "
        f"{b2['per_class_recall'].get('3', b2['per_class_recall'].get(3, 0)):.3f} | "
        f"{b2['per_class_recall'].get('4', b2['per_class_recall'].get(4, 0)):.3f} |",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
