# v10 execution pipeline — gates, not stages

> Written 2026-10-05. Companion to `docs/eeg2image_v10_plan.md` (the *what*) and
> `docs/eeg2image_v8_architecture.md` (the encoder). This file is the *order of operations*
> and, more importantly, **which result authorises which next step**. Single model
> throughout; no seed ensembling is used or proposed anywhere.

---

## 0. Why this file exists

The v10 plan has a headline (+1.14pp over SCORE, `+T2 reps` with FGW at α=.75) and three
choices behind it (α=0.75, τ=.03, which row to report). Every one of those choices was made
on the same 10 folds that are then reported. That is the standard way a retrieval number
turns out not to replicate, and it is cheap to check, so it is checked **first**, before any
GPU-hour goes into a method that might be measuring its own selection noise.

Two facts that make this non-optional, both verified while building this pipeline:

1. **τ matters at the same magnitude as the claim.** On the banked v8 encoder,
   `+T2 reps` is **53.80 at τ=0.03** but **52.82 at τ=0.01** (`outputs/eval/v8scr/a0.75_t0.03`
   vs `outputs/eval/v8_fgw075`). The v10 plan quotes 53.80; `v8_fgw075` is the τ=0.01
   duplicate of the same α. A 0.98pp swing from a Sinkhorn temperature is the same order as
   the +1.14pp being claimed, so τ is *not* a detail and the sweep gates on both temperatures.
2. **The deployed row's operator had to be reproduced from scratch.** The sweep recomputes
   `+T2 reps` through the same code path, so if the R=80 cells do not reproduce the banked
   Top-1 exactly, the sweep is a different operator and every curve below it is fiction.

---

## 1. Stage 1 — validation sweep (`slurm/v10_sweep.sbatch`, job `645433`)

Runs on the **banked v8 encoder only** (10 subjects × 3 seeds), no training. One embedding of
the R=80 repetition cloud per checkpoint; every grid cell is a slice of that embedding, so the
whole grid is one operator call per cell.

| Dimension | Values | Question it answers |
|---|---|---|
| `R` | 80, 40, 20, 10, 5, 1 | is the structural gain gated by repetition count? |
| `alpha` | 0.25, 0.5, 0.75, 0.875 | the α grid **on the v8 family** (the banked α sweep was on g3) |
| `tau` | 0.01, 0.03, 0.05 | the τ grid on the same family |

Every cell emits a **structural-off twin** (α forced to 0, same τ, same subset). The
quantity of interest is always the *paired* difference, never a difference of two separately
noisy means.

### Gate 1a — fidelity (hard, aborts) — ✅ **PASSED, exact**

`R=80,α=.75,τ=.01` must equal `outputs/eval/v8_fgw075` and `R=80,α=.75,τ=.03` must equal
`outputs/eval/v8scr/a0.75_t0.03`, per checkpoint, tolerance 0.51pp. Measured on all 30
checkpoints: **max|diff| = 0.000 for both references.** The sweep is bit-identical to the
deployed operator, so every curve below is a statement about the shipped code.

### Gate 1b — the gating curve — ⚠️ **GATED** (pre-registered readout)

gain(R) = Top-1(α>0) − Top-1(α=0), paired over the 30 runs, τ=0.03 (the headline's τ):

| R | on | off (α=0 twin) | gain | n_pos | paired t |
|---|---|---|---|---|---|
| **80** | **53.80** | 50.72 | **+3.08** | 25/30 | 6.23 |
| 40 | 46.15 | 43.83 | +2.32 | 22/30 | 3.93 |
| 20 | 35.80 | 34.68 | +1.12 | 24/30 | 2.46 |
| 10 | 27.43 | 26.82 | +0.62 | 17/30 | 1.15 |
| 5 | 18.53 | 18.23 | +0.30 | 14/30 | 0.62 |
| 1 | 5.08 | 5.52 | **−0.43** | 11/30 | −1.69 |

The τ=0.01 curve is the same shape (3.40pp drop). **The structural gain falls monotonically
with R and is negative at R=1.** So the +3.08pp is not a property of the transport term in
isolation: it is gated by the quality of the target-side estimate the term is built on.

**Read this carefully, because the level and the gain are different claims.** The *levels*
collapse at low R because the rep-cloud whitening is fitted on `C·R` samples and is degenerate
at `R=1` (200 samples for a 64-d covariance → 5% Top-1). The *gain* is the on-vs-off
difference **within** a fixed R, so both sides share that whitening and the drop in gain is
about the structural term specifically, not about the level.

**What this does and does not say.**
* It does **not** invalidate the v10 number. `+T2 reps` really is 53.80 on the v8 encoder, and
  it really is +3.08pp over the information-matched (same-subset) baseline.
* It **does** mean the +3.08pp cannot be presented as an information-matched win over a
  repetition-free method like SCORE. Our estimator needs R=80 un-averaged trials to be good;
  SCORE reaches 53.23 with SAMGA **without** a repetition cloud. The comparison is
  protocol-different and must be declared as such.
* The most useful thing in the table is *why* it is gated: the operative quantity is the
  **variance of the target-side estimator** feeding the transport. That is a fixable estimator
  defect, not a data requirement.

### Gate 1c — selection honesty (nested-LOSO) — ✅ **PASSED**

All 10 held-out subjects independently selected `R=80,α=0.75,τ=0.03` on the other nine.
Nested-LOSO mean **53.80** = fixed-config mean **53.80**, optimism **+0.00pp**. The α and τ
grids also have clean interior peaks on the v8 family: α 0.25/0.5/**0.75**/0.875 →
51.92/53.08/**53.80**/50.72; τ 0.01/**0.03**/0.05 → 52.82/**53.80**/53.10.

**So neither the α=0.75 nor the τ=0.03 choice is a selection artefact** — the earlier caveat in
`eeg2image_v10_plan.md` §1 ("τ and the row choice are not yet nested-LOSO") is now discharged
for τ, and α=0.75 is confirmed on the v8 family rather than borrowed from the g3 sweep.

---

## 2. What Stage 1 authorises

| Stage-1 outcome | Result | Authorised next step |
|---|---|---|
| fidelity | ✅ exact | proceed |
| gating curve | ⚠️ **GATED** | **Stage 2: fix the target-side estimator variance** (below), because the gain was shown to be hostage to it |
| nested-LOSO | ✅ +0.00pp | α=0.75, τ=0.03 are the honest config; nothing to restate |

Per the table fixed before the runs, a GATED curve does not stop the project and does not
invalidate the number — it redirects it. The redirect is concrete and cheap:

## 2.5 Stage 2 — remove the variance gate (the authorised next step) — **LAUNCHED, job `645436`**

The curve says the structural term needs a low-variance target-side estimate. It does not say
*how* low-variance, nor whether a better estimator reaches that at small R. That is a
one-parameter experiment:

* `_whiten_from_cloud(..., shrink=0.1)` is the estimator under suspicion. Its shrinkage is a
  free regulariser and 0.1 is a leftover from the probes, never tuned for this purpose.
* Sweep `shrink ∈ {0.1, 0.2, 0.35, 0.5, 0.7}` at each R, on the same 30 checkpoints, and ask
  whether the gain at R ∈ {10, 20} rises to the R=80 level.
* Implemented as `run_eval --rep-shrink S` (one value per run, into its own directory, so a
  run never mixes two estimators) plus `slurm/v10_shrink.sbatch`. `shrink` is recorded in
  every row's diag as `whiten_shrink`.

**Pre-registered success condition** (fixed before the run, so a lucky cell cannot be promoted
after the fact): a single shrink value with

    gain(R=20) >= gain(R=80) - 0.5pp     AND     gain(R=80) >= 3.08 - 0.5pp

means the gate is an estimator artefact and the term works in a *near*-averaged-query regime —
the version comparable to SCORE without a protocol caveat. If no shrink value does that, the
gate is a real data requirement, the paper quotes `R` with the headline, and the positioning
becomes "a rep-cloud method" rather than "a better operator".

`shrink=0.1` is re-run into its own directory as the job's own reproduction of stage 1, rather
than trusting the earlier job's JSONs.

## 3. Stage 3 — mechanism attribution (after Stage 2)

The sweep also settles a live confusion: the rep-cloud row differs from the averaged-query
row in **two** ways — moment matching on C·R = 16000 samples instead of C = 200, and that it
whitens at all. The earlier "variance gating" story conflated them. The `α=0` twin in every
cell is the whitened-but-structureless operator, so the curve separates:

* `α=0` row vs the unwhitened T1 rung → the whitening contribution (already banked:
  `+ CSLS + recovery` 40.78 vs `+ whiten + CSLS + recovery` 34.50 on α=.75/τ=.01);
* `α>0` row minus `α=0` row → the structural term, on the identical subset.

Only after this does the cross-subject concept-geometry-template idea (v10 plan §P2) get an
implementation, because its stated motivation is a mechanism that this stage may show is not
the operative one. **An ablation shipped before its mechanism is measured is a story fit to a
number.**

---

## 4. Stage 4 — training-side gate (deferred, gated on Stage 1/3)

The v9 training arm was falsified: α=0.75 *during training* scored 52.70 vs 60.80 for the v8
encoder (α=0) on identical folds — high-α second-order transport damages feature formation.
So the only training-side lever left is **low α**, and it is gated:

* smoke 1 fold × α ∈ {0.05, 0.1, 0.2} against α=0 on the same fold;
* adopt only if **≥ +0.5pp** on the smoke fold, then run full 10-fold × 3-seed LOSO;
* otherwise the encoder is frozen and v10 ships deployment-only.

---

## 5. Reproducing / resuming

```bash
sbatch slurm/v10_sweep.sbatch          # Stage 1, ~15 s/checkpoint, resumable ([skip] if done)
# summary lands in outputs/v10_summary.json; the fidelity gate aborts on mismatch
python scripts/summarize_v10.py --sweep-dir outputs/eval/v10_sweep \
    --fidelity "R=80,a=0.75,t=0.03@outputs/eval/v8scr/a0.75_t0.03" \
               "R=80,a=0.75,t=0.01@outputs/eval/v8_fgw075" \
    --curve-alpha 0.75 --curve-tau 0.03 --out outputs/v10_summary.json
```

Score matrices for every cell are written under `outputs/scores/v10_sweep/` so a later
question about the same checkpoint does not need a re-embed.
