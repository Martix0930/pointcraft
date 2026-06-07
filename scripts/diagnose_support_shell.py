"""Zero-cost diagnostic: is the candidate support "answering the question"?

Scores two **deterministic, no-training** predictions on a tile, with the shared
`pointcraft.metrics` under all three cutoffs:

  * the full solid candidate support (≈ B1), and
  * its **morphological shell** (B3, `baseline.morphological_boundary`).

If the deterministic shell already scores high (say strict unobserved IoU > 0.6),
the support construction is doing the model's job and a learned completer's headroom
is inflated — push on a harder support / generative decoder. If it scores low, the
learned model is doing real work and multi-tile generalization is the clean next
step. Re-run this per tile before trusting multi-tile numbers.

Numpy only (no torch); runs anywhere `pointcraft` imports.
    python scripts/diagnose_support_shell.py --npz outputs/m0/tokyo_citygml.npz
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
    ap = argparse.ArgumentParser(description="deterministic support-shell diagnostic")
    ap.add_argument("--npz", default=os.path.join(REPO, "outputs", "m0", "tokyo_citygml.npz"))
    ap.add_argument("--out", default=None, help="optional JSON to write the numbers to")
    args = ap.parse_args(argv)

    from pointcraft.baseline import candidate_support, morphological_boundary
    from pointcraft.metrics import build_cutoff_masks, evaluate, load_sample

    s = load_sample(args.npz)
    support = candidate_support(s.coords_partial, s.grid)  # input-only (B1 ∪ observed)
    shell = morphological_boundary(support, s.grid)
    cut = build_cutoff_masks(s.coords_target, s.coords_partial, s.sem_target, s.grid)

    print(f"tile {s.metadata['tile_id']}: support {len(support):,} -> "
          f"morph shell {len(shell):,} ({100*len(shell)/max(len(support),1):.1f}%); "
          f"target {len(s.coords_target):,}")
    out = {"tile_id": s.metadata["tile_id"], "support_size": int(len(support)),
           "shell_size": int(len(shell)), "predictions": {}}
    for name, pred in [("full_support_solid", support), ("morphological_shell", shell)]:
        r = evaluate(pred, s, cutoffs=cut)
        u = {c: round(r["unobserved"][c]["iou"], 4) for c in ("strict", "mid", "tolerant")}
        out["predictions"][name] = {"completion_iou": round(r["completion"]["iou"], 4),
                                    "precision": round(r["completion"]["precision"], 3),
                                    "recall": round(r["completion"]["recall"], 3),
                                    "unobserved_iou": u}
        print(f"  {name:20s} completion {r['completion']['iou']:.3f} "
              f"(P {r['completion']['precision']:.2f}/R {r['completion']['recall']:.2f})  "
              f"unobs strict/mid/tol {u['strict']}/{u['mid']}/{u['tolerant']}")
    print("  reference: B1 strict 0.061 | B2 strict 0.359 | M2(trained) strict 0.817")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[write] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
