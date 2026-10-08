# v12 — The core claim (locked 2026-10-05)

This document fixes the ONE claim the project argues, before any v12 code is written, so that
neither the implementation nor the write-up can drift toward whatever happens to work. Every
sentence below names the measurement that licenses it. Numbers are from
`docs/eeg2image_v11_unified_architecture.md` §0 (measurement table) and §7 (probe + result), plus
job 645719 (`outputs/eval/rep_ceiling/`).

---

## The claim

> **Cross-subject EEG-to-image retrieval is mis-specified as an inter-subject *alignment* problem.**
> **Measured, inter-subject maps explain 2.4% of the variance and do not compose (no group
> structure); the apparent "alignment gains" invert sign under a concept-permutation control, i.e.
> they are test-set index leakage, not transfer. The ≈25pp that actually matters is *refinement of
> the query's own concept metric against the gallery*, not transfer between subjects. The correct
> object is therefore the SHARED CONCEPT METRIC estimated from several independent noisy views, and
> its accuracy is governed by the number of views K and the per-trial SNR — which our R-curve shows
> is far from saturated.**

Three parts, each with its own falsifier:

### Part A — the framing is empty (diagnostic)

| Measurement | Value | What it kills |
|---|---|---|
| M6 global orthogonal map's explained variance | **+2.4%** | every Procrustes / Riemannian / hyperalignment variant is fitting this 2.4% component |
| M7 held-out map-composition gap | **+5.7%, t = 140, 10/10** | the subject term is **not a group action**; maps fitted on one pair do not compose |
| M8 local vs global Procrustes (power-validated) | local **15% worse**; control detects an injected field | position-dependent maps are worse, so no local geometric fix exists either |
| M1 source-metric template under a concept-permutation control | **+15.37pp → −14.5pp** | apparent "source transfer" is **index leakage**; the control is the method |

**Falsifier.** If some inter-subject map that (i) composes and (ii) survives the permutation control
explains ≫2.4%, Part A is dead. Every such attempt so far has failed, including the four reduced
structural families (P1 0/30 folds, A1/A2, A4 0/30) and the two transfer families (M1, C1).

### Part B — metric denoising is the right replacement (mechanism)

Pooling independent structure estimates raises their agreement with a held-out view **monotonically
in K**, while the correspondence-destroying control sits at zero (job 645697):

| K | pooled | single | shuffled control |
|---|---|---|---|
| 1 | 0.2093 | 0.2093 | −0.0006 |
| 4 | 0.2399 | 0.2093 | −0.0003 |
| 8 | **0.2471** | 0.2093 | +0.0003 |

(t = +42.4, 10/10 folds.) The control is the load-bearing line: an *unordered* pool gains nothing,
so the gain is shared structure, not averaging. Deployed (job 645716, 10 subjects): the fusion is
routed **only** through the structural term (the `structural-off` arm is bit-invariant to B), and
the term's own worth rises 3.85 → 4.70pp with B.

**Falsifier.** If a rank reduction (already falsified three times) or a graph binarisation beat the
richer fused object, Part B is dead. It is not: the term is monotonically richer-is-better.

### Part C — per-trial SNR is the binding constraint, and it is not saturated (the lever)

The deployed T2 operator refit on the first R repetitions (job 645719, subjects 1/5/8):

| R | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 80 |
|---|---|---|---|---|---|---|---|---|
| Top-1 (mean) | 4.0 | 7.8 | 12.3 | 20.0 | 30.8 | 38.7 | 51.2 | **54.5** |
| top-end slope | | | | | | | | **+3.33** |

Neither `1/R` (residual 98pp) nor `log R` (residual 46pp) fits: the curve is still climbing steeply
at R=80 (3/3 subjects positive in the last doubling). So **the query metric is still
estimation-limited at R=80**; making each repetition individually more informative is the one lever
with measured, unexhausted headroom.

**Falsifier.** If the curve had flattened at R=80 and the extrapolated ceiling equalled 54.25, Part
C would be dead and no denoising term could help. It has not flattened.

---

## What the claim does NOT say (the boundary that keeps it from being over-reached)

* It does **not** say alignment is useless. The **recovery rung is worth ≈ +25pp** (M12). The claim
  distinguishes two quantities the literature conflates: **inter-subject transfer** (measured dead)
  and **query↔gallery refinement** (measured alive). Only the former is refuted.
* It does **not** claim the reshape is free: v11 measured that the structure-SNR gain converts to
  only +0.85pp of retrieval (t = 1.72). Part C is exactly the honest response — the structure was
  improved but the representation was not, so the lever must be moved into training.

## The v12 instantiation (what the claim commits us to build)

1. **(a) Metric self-distillation.** Teacher = the reliability-weighted fused metric over K
   independent repetition blocks (stop-grad) — Part B's estimator. Student = the metric from a
   *single* repetition. Loss = `1 − corr(student, teacher)`. This is Part C made trainable: it
   forces one trial to carry what eighty trials carry.
2. **(b) Denoised structural reference.** The same fused consensus is the reference the soft-plan's
   structural term is built against, instead of the single-mean metric.
3. **(c) Metric-level subject consistency.** Constrain the *concept metric* (not the raw embedding)
   to agree across subjects — the 0.565 quantity — with a non-degeneracy guard, since the raw
   analogue collapsed the encoder twice.
4. **(d) `concept.reps_group = "stimulus"`.** The documented fix to a term whose row-grouped form
   was measured as an *anti*-T1 term (sub-08 42.0 → 30.0).

## The decisive experiment (and the pre-registered verdict)

Twin training: identical seed, data order and everything else; the two arms differ only in the v12
weights (`0` vs `w`). Paired across folds. **The claim's constructive half PASSES iff the v12 arm
beats its weight-0 twin on a majority of folds at R=80.** The non-degeneracy guards (rank retention,
metric-gallery correlation not falling) must hold simultaneously, or a "win" is a collapse and is
reported as one.

---

## Implementation status (2026-10-05)

* `src/samclip/losses/metric_distill.py` — `metric_self_distill`, `metric_subject_consistency`.
* `src/samclip/train.py` — `v12` config block + the `assemble` branch (reuses the concept/reps
  encoder path). `run_stage1.py` gained `--v12-distill/--v12-consistency/--v12-r-use/--reps-weight/
  --reps-group` (nested-key overrides, so a flag cannot silently do nothing).
* `scripts/smoke_test.py` §17 pins: loss small iff reps share structure (0.012 vs 1.034), **exact
  zero** when reps are identical, **permutation invariance** (0.0121 → 0.0121), rank reported,
  subject-consistency negative iff shared (−0.980 / corr +0.980 vs −0.110).
* `configs/v12_loso_k20.yaml` = v8 + the v12 block. `slurm/v12_twin.sbatch` runs the three arms.

### A MEASURED LIMIT OF THE TRAINABLE FORM (important, and it bounds Part C honestly)

**The train-side repetition cloud is only R=4 deep** (`train_subNN_all63_trainreps_mvnntrain_std.npy`
is `(1654, 10, 4, 63, 250)`), while the test cloud is R=80. So the teacher can only be as good as 4
train repetitions allow — the R-curve's R=4 point is 12.3 Top-1, not 54.5. The v12 term therefore
does NOT distil the R=80 consensus; it distils the **4-rep consensus**, and the claim is that the
SINGLE-trial student pulled toward it generalises to the 80-rep test cloud because the structure is
the same object. This is the honest ceiling of what is trainable from the data we hold, and it is
recorded here so a small effect is not mistaken for a failed mechanism.

### GPU wiring smoke (job 645753, PASS)

3 steps on sub-08 with the v12 arm. The branch fired and the loss arithmetic was checked by hand:

```
epoch 0 step 0 loss 4.1747   v12_distill 0.8748  agreement 0.1252  rank 4.83  pairs 36
epoch 1 step 0 loss 12.5375  v12_distill 1.0681  agreement -0.0681
  hand-check: 5.99 + 4.98*0.7 + 1.068*1.0 + 0.174*0.5 + 3.97*0.5 - 0.08 = 12.53  ✓
```

Two facts from the smoke worth keeping: the student metric's agreement with the consensus is only
**+0.13 / −0.07** on a real encoder — a numerically independent confirmation of Part C's "a single
trial is nearly uninformative" — and `metric_consistency_pairs` was 8 → 2 across steps because the
first cut required *identical* stimulus sets per subject pair, which almost never holds. Fixed to
the **intersection** (now 36 = C(9,2)). A healthy-looking loss with the term firing on 2 of 9
subjects is exactly the "reported as a mechanism while barely running" failure this project pays for.

### Submitted

`slurm/v12_twin.sbatch`, **job 645754**: arms `base`/`v12`/`v12d` x 10 folds x seed 2025, evaluated
under both the deployed FGW operator (headline, comparable to 54.25) and the hard operator
(comparable to 47.20).
