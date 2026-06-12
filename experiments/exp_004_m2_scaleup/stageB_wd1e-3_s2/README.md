# exp_003 — M2 fork-1 multi-tile generalization

Train ['09LD1878', '09LD1845', '09LD1843', '09LD1867', '09LD2818', '09LD1846', '09LD1885', '09LD1886', '09LD1897', '09LD2807'] → held-out **09LD2814** (disjoint). The binary
fork-1 question: does held-out unobserved IoU beat BOTH B1 and B3 on all cutoffs?

## Setup
- model `OccupancyCompletionUNet` base=8, params 79,153; code `05cdf6e`; peak CUDA 4064 MB
- batch=1 tile/step, additive skips, **tile-invariant features** (z_scale=50.0 m, not per-tile K)
- G0 border_margin=5 (loss + metrics); epochs=200, lr=0.001; threshold swept on held-out
- best ckpt by held-out strict IoU @ epoch 40, p=0.3

## G2.0a — candidate-support recall ceiling (held-out)
- overall **0.8815**, unobserved strict **0.7576** / mid 0.6834 / tol 0.7869 (ref 1874 ~0.93)
- support-as-pred IoU = {'strict': 0.146, 'mid': 0.0906, 'tolerant': 0.0869} (the deterministic floor inside support)

## Result — held-out unobserved IoU vs bar = max(B1, B3)

| cutoff | B1 | B3 shell | **held-out** | bar | pass |
|--------|----|----------|--------------|-----|------|
| strict | 0.146 | 0.165 | **0.301** | 0.165 | ✅ |
| mid | 0.091 | 0.134 | **0.183** | 0.134 | ✅ |
| tolerant | 0.087 | 0.141 | **0.183** | 0.141 | ✅ |

- **VERDICT A** — real generalization: held-out > max(B1,B3) on all 3 cutoffs.
- strict ratios: 1.823× B3 shell, 0.397× of the 0.7576 recall ceiling
- per-train-tile strict IoU at best ckpt: {'09LD1878': 0.0659, '09LD1845': 0.0493, '09LD1843': 0.1875, '09LD1867': 0.1, '09LD2818': 0.2226, '09LD1846': 0.1859, '09LD1885': 0.175, '09LD1886': 0.3025, '09LD1897': 0.2565, '09LD2807': 0.255} (train≫held-out → memorized; both low → under-learned/bug)
- ⚠ **curve:** held-out strict peaks 0.301@ep40 then collapses to 0.038@ep200 while train IoU climbs -> overfits the train tiles; early-stop at the peak (more tiles / regularization).

See `metrics.json` for full scores + history; `viz_heldout.png` for the slice; `pred_coords_val.npy` is the held-out prediction (gitignored artifact).
