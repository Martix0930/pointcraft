# exp_004 — M2 fork-1 Scale-up: Stabilize the Generalization Peak

**Objective:** stability, not magnitude. The fork-1 peak (strict 0.234–0.267 over 5 runs)
is fragile: run-to-run jitter ~0.015, peak-then-collapse by ep200, peak epoch swings
ep10→50 across seeds. This experiment pays that debt by (a) expanding train tiles 4→8–10
and (b) adding regularization.

**Held-out:** 09LD2814 (unchanged from exp_003).
**Eval口径:** unobserved IoU, strict/mid/tolerant, border_margin=5, same as exp_003.

---

## Status: COMPLETE — scale-up hypothesis FALSIFIED. Instability intrinsic (not data/reg-solvable); only gain = early-stop-ceiling reproducibility (spread 0.011). Next: evaluate generative decoder. See Stage B RESULT below.

### Locked train set (10 tiles) + per-tile B3(m=5) strict floor

All support sizes MEASURED (§2b), all ≤ HWM (1885 = 3.77M @ 3385 MB). Held-out 2814 unchanged.

| tile | axis role | surf/ha | nfr | K | support | B3 strict floor |
|---|---|---|---|---|---|---|
| 1878 | sparse | 108 | 0.092 | 148 | 1.91M | (exp_003 b3) |
| 1845 | sparse | 129 | 0.186 | 94 | 1.96M | (exp_003 b3) |
| 1843 | sparse·**articulated** | 88 | 0.308 | 165 | 1.50M | **0.1953** |
| 1867 | sparse-mid | 168 | 0.171 | 108 | 1.99M | **0.0702** |
| 2818 | mid·**articulated**·best-gnd | 256 | 0.317 | 154 | 1.10M | **0.1388** |
| 1846 | medium | 465 | 0.157 | 96 | 1.94M | 0.1534 |
| 1885 | medium-tall (**HWM anchor**) | 465 | 0.075 | 259 | 3.77M | (exp_003 b3) |
| 1886 | medium-dense | 577 | 0.172 | 169 | 2.92M | **0.1901** |
| 1897 | dense-flat | 754 | 0.182 | 87 | 1.94M | **0.1968** |
| 2807 | dense (near held-out) | 944 | 0.154 | 63 | 1.71M | **0.1975** |

- **Decision metric remains the held-out 2814 bar (strict 0.165)**, unchanged. Per-tile B3
  floors above are each tile's own honesty floor only.
- 1867's B3 strict floor (0.070) is notably low — sparse-mid tile where the morphological
  shell barely beats the solid; flagged but kept (it spans the sparse-mid gradient gap).
- Excluded for memory (measured > HWM): 1864 (5.91M), 1874 (4.17M), 1876 (4.16M) — see
  design-limitation note below.

Pre-flight artifacts: `scripts/preflight_support.py` (K/support measurement),
`outputs/g1/combined_candidates.csv` (§2a table), `outputs/g1/b3/*_b3_m5.json`.

---

## Success criteria (from spec §6)

- **Primary:** cross-seed peak-strict spread (max−min, 3 seeds) < fork-1's ~0.03.
- **Secondary:** curve flattens — post-peak strict does NOT collapse to B1 by ep200
  (or collapse onset shifts substantially later in tile-steps).
- **Constraint:** every seed's strict peak > B3 0.165.
- **Non-criterion:** peak strict is NOT required to exceed 0.251.

---

## Records

- `metrics.json` — per-seed dense histories in tile-steps + epochs, cross-seed peak spread,
  per-tile B3 floors, recall ceiling. (Written after training.)
- `combined_candidates.csv` → see `outputs/g1/combined_candidates.csv`.

---

## Stage A result — wd reg ladder (seed 0, tile-steps, full precision)

All 3 wd ∈ {1e-4, 1e-3, 1e-2} ran clean (200 ep / 2000 tile-steps each, isolated +
GPU-gated; the earlier cascade crash did not recur). Held-out 09LD2814 STRICT, every 50
tile-steps; per-tile train_iou every 200 (host-RAM discipline). See `stageA_curves.png`,
`stageA_summary.json`.

| wd | peak | early⅓ | mid⅓ | late⅓ | total drift | **back-half drift** | post-peak std | >B3 |
|---|---|---|---|---|---|---|---|---|
| 1e-4 | 0.288 | 0.242 | 0.131 | 0.100 | −0.142 | −0.030 | 0.061 | 42% |
| **1e-3** | 0.294 | 0.239 | 0.141 | 0.133 | −0.106 | **−0.008 (plateaus)** | 0.065 | 52% |
| 1e-2 | 0.316 | 0.276 | 0.194 | 0.147 | −0.130 | −0.048 | 0.064 | 70% |

**Findings (honest, seed-0 only — NOT the verdict; Stage B decides):**
- **All three still peak-then-collapse with large jitter** — none is flat (see PNG). Reg
  shifts the band, it does not stabilize the curve.
- **Reg flattens DRIFT, not JITTER.** post-peak std ≈ 0.06 is *identical* across all wd;
  weight-decay only changes the downward-trend of the mean, not the per-eval swing.
- Two readings of "flattest": **wd=1e-3** uniquely **plateaus** (back-half mean drift
  −0.008 ≈ 0, stops collapsing); **wd=1e-2** sits at the **highest band** (70% of evals
  > B3) but its mean is still drifting down (back-half −0.048) and its late segment is the
  noisiest (±0.067). Peak > B3 0.165 for all three.
- This foreshadows the spec's **Outcome 2** (instability may be intrinsic, not just data
  diversity) — but Stage A is seed-0 / within-run jitter only. The decision metric is the
  **cross-seed peak spread** measured in Stage B; Stage A only picks the reg.

**Stage A pick for Stage B:** wd=1e-3 (best drift-flattening of the three; carried forward
not because it stabilizes — it does not — but because it is the cleanest reg at which to
quantify the residual cross-seed instability).

---

## Main conclusion DIRECTION — pre-committed before Stage B results (do not retro-spin)

> **Locked while the Stage B numbers are not yet in, so the framing cannot be bent toward a
> nicer story once they arrive.**

Stage A shows that **weight-decay swept across two orders of magnitude (1e-4 → 1e-2) does
not suppress the post-peak jitter** — the per-eval std is ≈0.06 *identically* at all three
settings; reg only shifts the mean band and slows the downward drift. The scale-up premise
that **"more train tiles + regularization → a stable optimum" (the data-diversity hypothesis)
is therefore NOT supported** by Stage A: 10 tiles (vs fork-1's 4) plus reg still produces
the same peak-then-collapse-with-violent-jitter curve. The evidence **points to the spec's
Outcome 2 — the instability is intrinsic** to the submanifold-sparse-conv-on-large-candidate-
support setup (or the candidate-support framing itself), not a data-diversity deficit.

**Stage B's purpose is therefore reframed: it is NOT a stability confirmation.** It exists to
**quantify the instability along the cross-seed dimension** and test whether the cross-seed
picture *reproduces* Stage A's within-run picture. Two numbers, side-by-side with Stage A's
seed-0:
1. **cross-seed peak-strict spread** = max−min of the per-seed peak strict over seeds {0,1,2}.
2. **per-seed post-peak std** (the jitter band) for each seed.

**Decision rule (pre-committed, no escape hatch):**
- If cross-seed spread ≳ fork-1's ~0.03 **and** the per-seed jitter bands reproduce Stage A's
  ≈0.06 → **Outcome 2 is accepted as the verdict.** The instability is intrinsic; the
  "scale-up stabilizes the peak" objective of this round is reported as **not achieved**, and
  the finding re-opens optimization/architecture / the generative-decoder question (still
  gated on the 0.758 recall ceiling, not on these field trends).
- Only if the cross-seed spread comes in **well under 0.03 with shrunken jitter** would
  Outcome 1 hold — which Stage A gives us no reason to expect.

**Scope guard reaffirmed (spec §10):** we do **not** respond to the ugly curve by adding
epochs, tuning lr, or switching on dropout to force the curve flat. That would be solving a
problem that is likely **not parameter-solvable**, and it violates this round's
one-variable-at-a-time / let-the-data-speak discipline. If Stage B points to Outcome 2, we
record Outcome 2 — we do not chase the curve.

---

## Stage B RESULT — the pre-commit was PARTIALLY REFUTED (honest correction)

3 seeds {0,1,2} at wd=1e-3, identical口径. seed 0 = the Stage A wd=1e-3 run. See
`stageB_curves.png`, `stageB_summary.json`.

| seed | peak strict | @step | post-peak std | end |
|---|---|---|---|---|
| 0 | 0.294 | 250 | 0.065 | 0.173 |
| 1 | 0.305 | 200 | 0.062 | 0.186 |
| 2 | 0.301 | 400 | 0.062 | 0.038 |

1. **cross-seed peak-strict SPREAD = 0.011** — **well UNDER fork-1's ~0.03**, all 3 peaks
   > B3 0.165 by ~0.13 margin.
2. **per-seed post-peak std = 0.062–0.065** — reproduces Stage A's ≈0.06 exactly, every seed.

**I predicted (pre-commit above) the spread would come in ≳0.03 and confirm Outcome 2. It did
not — it came in at 0.011. That prediction was WRONG and I am not going to spin 0.011 into
"still Outcome 2."** The honest reading is a **decomposition the spec's 3 outcomes did not
anticipate** (they assumed spread and collapse move together; they didn't):

- **Peak MAGNITUDE is now reproducible across seeds** (spread 0.011, down from fork-1's ~0.03;
  peak epoch tightened to ep20–40 from fork-1's ep10–50). → the round's **PRIMARY success
  criterion (spread < 0.03) is MET**, and the **constraint (every seed > B3) is MET**.
- **The peak-then-collapse SHAPE is unchanged** — every seed still rises to ~0.30 then
  collapses/oscillates into a 0.05–0.20 band with ≈0.06 amplitude, identical across all 3
  seeds AND across Stage A's 2 orders of wd. → the round's **SECONDARY criterion (curve
  flattens / collapse eliminated) is NOT met.**

**What this actually means — "stable peak" in the weak sense, not the strong sense.** The
0.011 spread says the **early-stopping CEILING is reproducible**, not that the model reaches a
**stable optimum**. Taking the max over a ~0.06-amplitude oscillating band will cluster near
that band's ceiling regardless — so peak-spread agreement is necessary but not sufficient for
"stable." The fork-1 caveat ("an unstable early-training peak captured by held-out
early-stopping, not a stable trained optimum") **survives Stage B intact**: the model still
cannot HOLD the peak.

**Verdict — NOT a symmetric SPLIT. Weighted: the scale-up hypothesis is FALSIFIED.**

- **PRIMARY (the result): the data-diversity → stability hypothesis is FALSIFIED.** The
  peak-then-collapse + ≈0.06 jitter **reproduces across 3 seeds AND across 2 orders of wd**
  → the instability is **intrinsic to the discriminative + fixed-candidate-support
  framework**, **not** a data-quantity/diversity deficit, and **not weight-decay-solvable**.
  The round's objective — *make the peak stable* in the strong sense (hold it / no collapse)
  — is **NOT achieved, and the cross-seed × cross-wd reproduction is positive evidence it
  cannot be achieved in this framework.**
- **SECONDARY (the only gain, stated narrowly): the early-stop CEILING is reproducible.**
  Cross-seed peak spread tightened to 0.011 and every seed clears B3. Recorded as
  **"ceiling reproducibility," NOT "generalization robustness/stability"** — max-over-a-
  noisy-band agreement is necessary, not sufficient, for a stable optimum. fork-1's
  generalization claim stands and its peak is now reproducible, but its caveat ("unstable
  early-training peak captured by early-stopping, not a stable optimum") is **confirmed
  intrinsic**.

**Core output (do NOT bury under "SPLIT"): a NEW authorization signal for a generative
decoder, independent of the recall ceiling.** The peak instability reproducing across seeds
and wd is a **framework-intrinsic** property — an optimization-stability signal orthogonal to
the 0.758 *coverage* ceiling. The decoder question is now gated on **two independent signals**
(coverage cap **and** framework-intrinsic instability), **not** field trends. Next step:
**evaluate** (not arm) a generative decoder. See `docs/06_DECISIONS.md` (2026-06-13) and the
M2 SESSION_LOG. We do **not** chase the curve with epochs/lr/dropout (§10) — the falsification
*is* the result, not a tuning failure.

---

## Design limitation — recorded BEFORE results (do not retro-fit)

**The train set contains no dense-AND-articulated sample, so 2814's density×articulation
interaction generalization is not verifiable by this experiment.**

Why, concretely (all from measured K/support, §2b):
- The only tiles that combine high footprint density **and** high articulation (high `nfr`)
  also have large K (tall) — 09LD1864 (5.91M), 09LD1874 (4.17M), 09LD1876 (4.16M) — and
  every one of them **exceeds the proven memory HWM** (1885 = 3.77M @ 3385 MB). They are
  excluded for memory, not by choice.
- The two articulation-axis tiles we *can* afford (09LD2818 nfr=0.317, 09LD1843 nfr=0.308)
  are **sparse** (surf/ha 256 / 88). The dense tiles we can afford (09LD1897, 09LD2807,
  09LD1886) are all **low-nfr** (0.18 / 0.15 / 0.17, i.e. flat-roofed).
- So in the train manifold the **density axis and the articulation axis are each covered
  only in isolation** — never jointly at high values. There is a hole at the
  *dense × articulated* corner.

Consequence for the held-out: **09LD2814 is the densest tile in the pool (surf/ha 1071) and
is moderately articulated (nfr 0.265)** — it sits exactly in that uncovered corner (it is
flat, h_std 13.7, which is why it itself fits, but its density×articulation combination has
no train analogue). Therefore:

- A held-out **pass** licenses: "completion transfers to a denser, moderately-articulated
  tile whose density and articulation are *individually* represented in training."
- A held-out pass does **NOT** license: "the model generalizes the *interaction* of high
  density and high articulation," because no training tile exercises that interaction. We
  cannot distinguish 'learned the joint structure' from 'the two axes happen to compose
  linearly here'.
- A held-out **failure concentrated on articulated-dense regions** would be **expected and
  uninformative** about the model — it would reflect this train-set hole, not a modelling
  limit. Do not read such a failure as evidence against generalization.

This limitation is a **direct consequence of the memory budget** (sparse-conv on full
candidate support), not the tile-selection policy. Lifting it needs either a larger GPU
budget or support cropping/tiling of the dense-tall-articulated tiles — both out of scope
for this round (two-variables-only). Recorded so a later round can target it deliberately.

---

## Two variables only

1. Number of train tiles: 4 → 8–10.
2. Regularization: weight-decay ∈ {1e-4, 1e-3, 1e-2}, optional dropout.

Everything else frozen: held-out, eval口径, D10/z_scale, architecture (base=8, additive skips).
