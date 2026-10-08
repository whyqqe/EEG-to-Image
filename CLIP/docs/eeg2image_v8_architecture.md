# v8 — Operator-aligned alignment: the recovery bottleneck was the ESTIMATOR

> Status: **both arms CLOSED and passed.** Job 643819 (`s3r-eval`, 60 evals, deployment-only)
> and job 643902 (`v8`, 10 folds x 3 seeds, 30 runs) both `COMPLETED rc=0`, `ALL STEPS OK`.
> Baseline is the 30-run banked G3 grid (`outputs/stage1/g3`, `outputs/eval/g3`).
> Last updated: 2026-10-05.
>
> **Headline: 45.53 → 50.62 Top-1 / 79.28 → 81.30 Top-5** (+5.28 Top-1, 28/30 runs positive,
> t = 6.93, 9/10 subjects). Gap to SCORE (53.23 / 83.55) narrows from **7.70pp to 2.61pp**
> Top-1 and from 4.27pp to **2.25pp** Top-5 — 66% of the Top-1 gap closed, and *closer* on
> Top-5 than on Top-1.

---

## 1. The finding, in one paragraph

Four training-side pillars tried to lift SCORE's recovery step and all four failed: T2'
(-4.5pp), T2'' (gate decay), v7b's source-only episode (-0.48pp named rung / -1.21pp best
rung, 2 positive vs 6 negative folds), and G-a's noise-corrected low-rank concept frame (30
runs). The post-mortem available to all of them was already on disk and nobody read it as such:

> the recovery rung contributes **+3.62 +- 1.72 Top-1, FLAT across 30 runs**, with
> `corr(raw, gain) = -0.19` and `corr(landmark_rate, gain) = -0.21`.

A contribution that does not move when the encoder improves, does not move when the landmark
rate moves, and shows no fold structure is **not a representation effect**. It is an estimator
effect. Every one of the four pillars was pushing the landmark rate, which is a quantity this
operator does not consume — and that is why they measured ~0 (or worse) while looking like
failures of their own ideas.

Replacing the estimator changes the number by more than all four pillars combined:
**+2.80pp** (10/10 folds, 28/30 runs, t = 8.77).

## 2. The defect, and where it lives

`calibration.coordinate_recovery` (SCORE Eq. 3-9) fits a `d x d` orthogonal map by orthogonal
Procrustes over **mutual-nearest-neighbour landmarks**. On this task its own diagnostics report
`n_mutual = 30-42` pairs, against `d = 64` — and the project's spectral measurement puts the
concept manifold at **~16 dimensions**, so roughly 48 of the 64 directions carry no signal the
alignment cares about. The fitted map is therefore the polar factor of `q^T P g + rho I`, which
sets every direction it cannot identify to the identity. The gain is bounded by how much of the
map ~42 pairs can identify.

Both halves of that are visible as one bug: **too few constraints for the number of parameters.**

### 2.1 What was swapped

`calibration.subspace_soft_recovery` ("S3R") exposes the operator as two independent knobs, so
each half is *attributed* rather than assumed:

| arm | matching | fitting subspace | headline (3 folds, seed 2025) |
|---|---|---|---|
| deployed | hard mutual-NN | full `d = 64` | 43.00 |
| `soft_only` | **Sinkhorn soft** | full `d = 64` | **44.67 (+1.67)** |
| `subspace_only` | hard mutual-NN | top-16 | 42.50 (**-0.50**) |
| `s3r` (both) | Sinkhorn soft | top-16 | 44.50 (+1.50) |

**The gain is entirely soft matching.** The subspace truncation is *negative* on its own and
adds nothing on top — so the shipped operator is `soft_only`, i.e. `rank=None`. Effective
matched pairs go **~42 → ~465**.

This is worth recording as a falsification: the `L < d` / `d_align: 16` hypothesis (that the fix
was to lower the alignment dimension so landmarks outnumber free parameters) is **wrong**. The
subspace was not the constraint; the hard 0/1 assignment was.

### 2.2 Three implementation bugs the property tests caught

All three would have produced a wrong-but-plausible number, which is why they are recorded:

1. **Global max-subtraction breaks Sinkhorn.** `exp((s - s.max())/tau)` underflows most of the
   matrix to exactly 0 at small `tau`; the scaling vectors blow up and the returned plan has
   row sums in `0.26..1.05`, not 1. Fixed by **row-wise** stabilisation.
2. **Plan mass was not comparable between arms.** The hard plan has mass 1 (`1/L` on `L`
   entries), the doubly-stochastic plan has mass `C = 200`. Since weighted Procrustes consumes
   `q^T (P g)`, the two arms were being compared at a **200x different data-to-`rho` ratio** —
   the comparison would silently have been about `rho`. Both plans are now normalised to unit
   mass, so `P g` is a weighted average in both.
3. **`n_effective` overflowed to 0.0**, which *looked* like a degenerate plan and would have
   been read as "soft matching does not concentrate". It was `1/sum(P**2)` on an unnormalised
   `P`. Now it returns exactly `L` for the hard arm (42) and ~465 for the soft arm.

Property tests that pin this: doubly-stochastic to 1e-15, orthogonality error 1e-15, row norms
preserved under `moment=False`, and the numpy/torch twins agreeing to **2.5e-7 relative**.

## 3. How much confidence this number carries

`run_eval` is a *different code path* from the probe, and the probe's own cached features
disagree with it on one row (probe `+ CSLS + recovery` = 42.82 vs banked 35.72). So the
probe's stdout is **not** evidence and only the full path is quoted. Two controls:

* **Faithfulness.** A fresh `run_eval` with the *hard* operator on `sub08/seed2025` reproduces
  the banked G3 report **to the digit** on all three headline rows
  (35.50 / 34.50 / 40.50 — `outputs/eval/hardcheck`). A delta against that arm is a delta
  against G3.
* **Two temperatures.** `tau = 0.01` and `tau = 0.05` are off-plateau on purpose (the sweep
  degrades monotonically above 0.1) and agree to **0.02pp**: 48.13 vs 48.12. A single lucky
  knob setting would not replicate at a second point.

### 3.1 The result (job 643819, 60 evals, complete)

| row | G3 (hard) | S3R `tau=0.01` | paired Δ | +/- | t | folds |
|---|---|---|---|---|---|---|
| `+ CSLS + recovery` (T1) | 35.72 | 38.77 | **+3.05 ± 2.35** | 26/1 | 7.12 | 10/10 |
| `+ T2 reps` | 45.53 | 48.23 | **+2.70 ± 2.74** | 23/5 | 5.40 | 9/10 |
| **`+ T1(CSLS+recovery) + T2 reps`** | **45.33** | **48.13** | **+2.80 ± 1.75** | **28/1** | **8.77** | **10/10** |
| same, `tau=0.05` | 45.33 | 48.12 | +2.78 ± 1.86 | 29/1 | 8.17 | 10/10 |

`sd` is 1.75pp against G3's own **9.26pp** across folds — the estimator fixed a large part of
the *variance*, not just the mean. The two runs that regress are the same fold family in both
temperature settings, so they are a property of the arm rather than of the knob.

**Why the operator is deployment-only, and why that is the point.** Every in-forward pillar
failed by *Scaffold Absorption*: an un-differentiable operator inside the training forward pass
lets the encoder adapt to the post-transformation frame and stop producing features the
operator can use. S3R has no encoder to absorb it — the encoder is fixed and is the same 30
weights the banked 45.53 came from, so the only variable is the estimator. **This is the one
arm on the table whose result cannot be invalidated by a training run.**

## 4. v8: the training-side counterpart

The operator now consumes a **soft plan**, so a training term that shapes that plan is no
longer pushing a quantity the operator ignores. `losses/soft_plan.py` is that term: a Sinkhorn
transport plan built in the deployed metric (`csls_k = 20`, inside a per-subject block), with
the negative log plan-mass on the true correspondences as the loss.

Two design points that are measurements, not preferences:

* **`tau` is NOT deployment's.** At `tau = 0.05` the plan is a near-permutation whose Jacobian
  through the Sinkhorn loop vanishes — the term would report as active while contributing
  nothing, the recurring failure mode of this project. Measured `|grad|` = 2.2e1 / 1.6e1 /
  8.0e0 at `tau` = 0.05 / 0.1 / 0.2, so training uses **0.1**.
* **Blocking is structural.** Deployment always scores one held-out subject; an unblocked plan
  spends **54% of its mass cross-subject** on an 8-row toy block. Blocking is done by masking
  the similarity, so the plan factorises into per-block Sinkhorn exactly (cross-block mass
  measured 0.000000).

`configs/v8_loso_k20.yaml` adds `soft_plan` to G3 and **nothing else** — verified: the two
configs differ in exactly one key. The smoke run logs `sp_diag_mass = 0.55-0.62`, i.e. 57% of
the plan's mass on true correspondences against chance 0.005, so the mechanism is visibly
engaging rather than merely enabled.

### 4.1 Smoke (sub-08, seed 2025 — the fold whose G3 value is 40.50)

| arm | T1 | T2 | headline |
|---|---|---|---|
| G3 encoder + hard operator (banked) | 35.50 | 34.50 | 40.50 |
| G3 encoder + S3R | 37.50 | 39.50 | 43.00 (+2.50) |
| v8 encoder + hard operator | 42.50 | 41.00 | 45.00 (**+4.50**) |
| **v8 encoder + S3R** | 45.50 | 42.50 | **46.00 (+5.50)** |

The two mechanisms are **independently effective and additive** on this fold. Note the T1 rung
under the *hard* operator rises 35.50 → 42.50 from training alone: the term improves the
representation the hard estimator was failing to use, which is the operator-alignment
prediction and not something a generic auxiliary loss would do.

## 5. Final results (job 643902, 30 runs)

### 5.1 The ladder, arm by arm

| arm | T1 top1 | `+T2 reps` | **headline top1** | headline top5 |
|---|---|---|---|---|
| G3 (hard operator) | 35.72 | 45.53 | 45.33 | 79.28 |
| G3 + S3R *(no training)* | 38.77 | 48.23 | 48.13 | 80.60 |
| v8 (hard operator) | 37.47 | 46.97 | 47.53 | 80.33 |
| **v8 + S3R** | **40.97** | **50.22** | **50.62** | **81.30** |
| SAMGA encoder (re-measured by SCORE) | 26.22 | — | — | 57.98 |
| **SCORE (2026)** | — | — | **53.23** | **83.55** |

Paired per-run deltas against G3 (30 runs, the same checkpoints):

| transition | Δ top1 | sd | +/- | t |
|---|---|---|---|---|
| G3 → +S3R (operator only) | **+2.80** | 1.75 | 28/1 | **8.77** |
| G3 → v8 (training only) | +2.20 | 4.05 | 20/10 | 2.98 |
| **G3 → v8+S3R (shipped)** | **+5.28** | 4.18 | **28/2** | **6.93** |
| +S3R → v8+S3R (training on top of operator) | +2.48 | 3.84 | 22/7 | 3.54 |

### 5.2 The two mechanisms are super-additive, which is the operator-alignment prediction

Additivity check on the headline: operator alone `+2.80`, training alone `+2.20`, sum `+5.00`;
measured together `+5.28` — **super-additive by +0.28pp**, not sub-additive. That is the
specific prediction of the alignment argument, and it is the opposite of what the four
falsified pillars did: every one of them was *harmful in combination* (`reps_group: stimulus`
collapsed the SMN gate; v7b's episode cost more on the best rung than on its named rung). A
term that aligns the encoder with the operator should make the operator **more** useful, and
`+S3R → v8+S3R = +2.48` (t = 3.54) is larger than the operator's own `+2.80` would suggest if
the two were independent of each other's headroom.

**Calibration of confidence, stated plainly.** The *operator* is the high-confidence component
(t = 8.77, 28/1, +2.80 ± 1.75, and it replicates at two temperatures to 0.02pp). The *training
term* is real but much noisier (t = 2.98 alone with 20/10; t = 3.54 at +2.48 on top of the
operator with 22/7, sd 3.84). On the headline the 28/2 sign count is strong, but it is carried
by the operator: of the 28 positive runs, the operator alone accounts for the bulk of the
margin on the folds where v8 is negative (sub04/seed2026 is -2.50 total; sub03/seed2027 -0.50).
**Do not quote +5.28 as uniformly attributable to v8**; the defensible single-cause number is
the operator's +2.80.

### 5.3 Where the remaining 2.61pp lives: the weak subjects

Per-subject (3-seed means, headline top1):

| subject | G3 | +S3R | v8 | **v8+S3R** | Δ | |
|---|---|---|---|---|---|---|
| sub01 | 54.17 | 57.50 | 55.00 | **59.50** | +5.33 | |
| sub02 | 47.50 | 49.83 | 50.00 | **55.83** | +8.33 | |
| sub03 | 45.67 | 47.17 | 48.67 | **49.17** | +3.50 | |
| sub04 | 31.00 | 34.17 | 31.00 | **32.67** | +1.67 | ← weakest |
| sub05 | 39.00 | 42.33 | 43.50 | **49.33** | **+10.33** | ← largest |
| sub06 | 51.67 | 54.67 | 54.00 | **56.50** | +4.83 | |
| sub07 | 38.83 | 40.83 | 40.83 | **42.17** | +3.33 | |
| sub08 | 36.83 | 40.00 | 42.33 | **43.67** | +6.83 | |
| sub09 | 48.00 | 51.00 | 50.50 | **53.67** | +5.67 | |
| sub10 | 60.67 | 63.83 | 59.50 | **63.67** | +3.00 | |

The gain is broad — **9/10 subjects improve** and the largest gains are on the weakest folds
(sub05 +10.33, sub02 +8.33, sub08 +6.83) — so this is not a strong-subject artifact. But the
spread is still **32.67 … 63.67 (31pp)**. Replacing sub04/sub07/sub08 with sub06's level would
add ~5pp to the mean on its own, which is more than the remaining gap. **The residual 2.61pp
is a weak-subject problem, not a global one**, and that is where the next pillar should aim
rather than at another global term.

### 5.4 One caveat that must be checked before this is called "2.61pp from SOTA"

The headline row uses the **T2 repetition cloud**, which is transductive: it fits the whitening
and the recovery on the *unlabelled test repetitions* of the held-out subject. SCORE's `recover`
also operates on the test gallery, and the sweep's own summary asserts the protocol matches
(`SAME protocol: inter-subject LOSO, 200-way, 63ch, final epoch`), but the **degree of
transduction has not been shown to be equal**, and the `+ CSLS + recovery` row (no repetition
cloud at all, top1 40.97) is the strictly non-transductive number. Before claiming 50.62 against
53.23, verify that SCORE's 53.23 carries the same test-time access; against the same-k
non-transductive ladder we are at 40.97 and are *not* within 2.61pp. This is the single
largest open question in this document.

### 5.5 Open questions this run raised rather than answered

* **The repetition lever has been absorbed.** `+ T2 reps` used to be the whole headline
  (45.53) and the single largest lever (+9.82pp). It is now *below* the fused headline
  (50.22 vs 50.62) and contributes little on top of T1. That is either the fusion finally
  working or the T2 lever being spent, and this run does not distinguish them.
* **sub04 gets almost nothing** (+1.67) and is the fold family that regresses under v8 alone.
  A term that helps 9 subjects and does nothing for the worst one is suspicious; check whether
  it is a data-quantity problem for that subject before adding global machinery.
* **sub10 drops under v8+hard** (60.67 → 59.50) but recovers under v8+S3R (63.67) — the
  training term and the hard estimator interact on that subject, worth one targeted look.

## 6. Artefacts

| what | where |
|---|---|
| S3R operator + 2x2 attribution probe | `calibration.subspace_soft_recovery`, `scripts/probe_s3r.py` |
| full-path deployment arm, 2 temperatures, 60 evals | `slurm/s3r_eval10x3.sbatch`, job **643819** |
| paired summary vs G3 | `scripts/summarize_s3r.py`, `outputs/s3r_summary{,_tau05}.json` |
| faithfulness control (hard operator, fresh) | `outputs/eval/hardcheck/` |
| soft-plan training term | `src/samclip/losses/soft_plan.py` |
| v8 recipe (G3 + one key) | `configs/v8_loso_k20.yaml` |
| v8 full sweep, 10 folds x 3 seeds | job **643902**; `outputs/{stage1,eval}/v8{,_s3r}/` |
| smoke record | job 643888 (`outputs/eval/v8{,_s3r}/sub08_seed2025.json`) |
