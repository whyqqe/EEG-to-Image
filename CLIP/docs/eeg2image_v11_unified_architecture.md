# v11 — A unified architecture from the measured evidence

Every previous version (v4–v10) proposed a *transformation* to close the cross-subject gap and
was then falsified. v11 starts from the opposite end: it asks what the ten measured numbers
actually say the gap IS, and designs the inference that follows. The proposal is a reframe from
**alignment** to **denoising**, and it is stated with the measurement that licenses each layer and
the pre-registered test that can kill each layer.

---

## 0. The measurement table (everything this document rests on)

| # | Quantity | Value | Source |
|---|---|---|---|
| M1 | Cross-subject concept metric agreement | corr(D_eeg,s, D_eeg,t) = **+0.565** (chance −0.003) | §1.1 |
| M2 | EEG↔image concept metric agreement | corr(de_target, di) = **0.593** (whitened) | §9.2 |
| M3 | Source metric as a proxy for the gallery metric | corr(de_src, di) = **0.782** | §9.1 |
| M4 | Intrinsic dim, unwhitened / whitened | **25.6 ± 4.2 / 85.0 ± 4.5** | §12.1 |
| M5 | Query↔gallery **topology** agreement | Jaccard **0.269 ± 0.018** | §12.2 |
| M6 | **Global orthogonal map's explained variance** | **+2.4%** (43.9% vs no-map 41.5%) | probe |
| M7 | Map closure gap (held-out composition) | **+5.7%**, t = 140, 10/10 | probe |
| M8 | Local vs global map (power-validated control) | local **15% worse**; control detects injected field | probe |
| M9 | Residual spatial coherence | neighbour 0.089 vs random 0.001 | probe |
| M10 | Cross-subject Top-1 spread | **54.25 ± 10.97** (n=10) | §12 |
| M11 | Structural term's own worth (on − off) | **+3.85pp** | §12 |
| M12 | Recovery rung's contribution | **≈ +25pp** (ATM-baseline → SCORE-operator) | §3 |

---

## 1. What the measurements jointly say

### 1.1 The gap is not a transformation

Four independent measurements falsify the sufficiency of "estimate a map and invert it":

* **M6.** The best global orthogonal map explains **+2.4%** of the cross-subject variance. SCORE,
  Riemannian alignment, ITSA and every Procrustes/hyperalignment variant are all estimating this
  object. Whatever they buy, they buy from a 2.4%-variance component.
* **M7.** The fitted maps do not compose: the held-out composition gap is +5.7% with t = 140 on
  10/10 folds. So the subject term is **not a group action**, and any method that fixes the group
  a priori (CLIP: identity; SCORE: O(d); RA/SATTC: SPD congruence; GWOT: isometry) is fixing the
  wrong object.
* **M8.** Locally-weighted maps are **15% worse** than the global one on held-out concepts. And
  the control *does* detect an injected smooth field, so this is a powered negative: the subject
  term is not a smooth position-dependent field either.
* **M9.** Residual coherence of neighbouring concepts is only 0.089 — marginal, and consistent
  with a small smooth component riding on a large idiosyncratic one.

So the subject term is not a group element, not a field, and any map over it moves 2.4% of the
variance. **The residual is not a structure to be inverted. It is noise to be averaged down.**

### 1.2 Where the accuracy actually comes from

M12 is the number that reorganises the whole field's framing: of the ~39pp between the
no-adaptation baseline and the reported SOTA, **~25pp come from one test-time operation: label-free
correspondence recovery**. Structural geometry contributes +3.85pp (M11). Repetitions and CSLS
contribute single digits. Encoding contributes a few pp.

And **M11's replacement is flat**: the recovery rung's gain is uncorrelated with encoder quality
and landmark rate (corr ≈ −0.19, measured over 30 runs). A mechanism whose benefit does not
respond to the quality of its input is an *estimator-noise-limited* mechanism. That is M12 restated
from the other side.

### 1.3 The regime: a real but low-SNR shared manifold

M1 = 0.565 and M2 = 0.593 say the shared concept geometry is **real but noisy**: two views of the
same structure agree at ~0.58. For a 200-concept kNN graph, a metric correlation of ~0.58 predicts
a neighbourhood overlap of ~0.27 — which is **exactly** the measured M5 (0.269 ± 0.018). The
topology measurement is therefore not a separate fact; it is the *arithmetic consequence* of the
metric correlation under thresholding.

This has a strong implication: **M5 does not say "topology is a different thing". It says the
metric's SNR is low, and thresholding amplifies that.** Any functional of the metric more
non-linear than the metric itself will lose SNR at a predictable rate. That closes the entire
family of "use a cleverer structural functional" approaches (P1, P2, A4) with one explanation
instead of three.

### 1.4 The two untapped resources

Given §1.1–§1.3, the only mechanisms left standing are variance reduction and better inference.
The measurements say which resources are available and unused:

1. **Repetitions are an ensemble, and are only used as a mean.** M12's operator consumes
   `q_bar = mean over R≤80 repetitions`. The 80 repetitions define a *distribution* over
   embeddings, hence a distribution over correspondences, whose spread is a direct measurement of
   query reliability. No method in this project (or in the surveyed literature) propagates that
   spread into the correspondence decision. We even measured that the naive gate on it fails
   (disagreement 0.09) — but the gate *rejects* rather than *reweights*.
2. **The gallery is a second, independent view of the same 200 concepts and is used only as a
   retrieval target.** M2 says it is the single *most informative* view of the shared structure
   (0.593, higher than the EEG–EEG 0.565). It is never used to denoise the query side.

---

## 2. The reframe: from alignment to denoising

**Statement.** Cross-subject EEG→image retrieval is a **multi-view denoising problem**: the target
subject, the source subjects and the image gallery are noisy views of one shared concept manifold,
and the task is to recover that manifold and each view's position on it, not to estimate and invert
a transformation between views.

**Why this is the right reframe and not a slogan.** Every falsified family (M1, C1, P1, P2, A4)
shared one structural feature: it tried to *improve the alignment object* — a better metric, a
better spectrum, a better graph, a better map. Under M6 (2.4%) and M8 (powered negative), none of
those can pay. Every measured success (M11's term, M12's recovery, repetitions, CSLS) shares the
opposite feature: it *pools or regularises an estimate*. The reframe is what distinguishes the
things that worked from the things that failed, using only measured facts.

**The unifying objective.** Let `{D_k}_{k=1..K}` be the noisy pairwise-distance (or
Gram) estimates of the shared structure from the `K` available views (target repetitions, target
mean, gallery, and — as a regulariser only, see §4 — sources). Then:

```
        D*  =  argmin_D  sum_k  w_k * || D - D_k ||^2_{F, R_k}     (structure estimate)
        s_i =  score of concept i, computed from  D*  and the per-view reliability
```

i.e. a **weighted, reliability-aware fusion of order-2 structure estimates**, followed by
correspondence read off the fused structure rather than from a fitted map. The two things that
make this more than averaging:

* the weights `w_k` come from **measured per-view reliability** (repetition agreement,
  leave-one-view-out residual), not a tuned scalar — this is the δ(C1) fix: C1's monotone map had
  no headroom because it acted on *one* view's metric;
* the fusion is done on the **denoised** structure, so the output is a *better estimate of the
  same object the term already consumes at +3.85pp*, which is the only object measured to carry
  signal (M11).

---

## 3. Measured headroom, stated before any implementation

The reframe is only worth building if pooling can actually raise the structure estimate's SNR.
That is measurable **today, on cached data, with no GPU**:

> **PREDICTION 1 (the gate).** Pooling `K` views' metric estimates raises the correlation with the
> gallery metric `di` above the single-view value of **0.593**. The gain must **increase
> monotonically with K** and must exceed a **view-shuffled control** (fuse the same views after
> destroying the concept correspondence) by a margin that itself grows with K.

This is deliberately the twin of the M1 permutation control: a fusion that gains without a control
is M1 again. **If Prediction 1 fails — if pooled structure does not beat 0.593 with a growing,
control-separated margin — the architecture is dead, and it dies before a single GPU hour.**

Two secondary quantities to read at the same time:

> **PREDICTION 2.** The per-view reliability weights should be informative: views ordered by
> agreement with the fused structure should have their leave-one-out gain correlate positively
> with that agreement. (This is the δ(recovery-flatness) test: the mechanism must respond to
> input quality, unlike the recovery rung's corr ≈ −0.19.)
>
> **PREDICTION 3.** The target's own repetitions should be the highest-weighted view of the
> query side. If they are not, T2's measured gain is an artefact of moment counting and the
> reliability model is wrong.

> **MEASURED (2026-10-06, job 650112 — see `docs/eeg2image_v13_croma.md` §3.4).** On the ten banked
> `v8` checkpoints the per-view weights span only **0.0057** (near-flat), because the repetition
> errors are measured to be **uncorrelated across views, $\rho\le0$ on 10/10 folds** — so there are no
> *reliability differences* for the LOO-residual weighting to exploit. The fusion gain is real
> (`fuse − mean = +0.0121`, 8/10 folds, consistent with the +1.03pp Top-1 fusion result), but its
> source is **block structure / metric sharpening**, **not** reliability-weighted pooling as this
> section describes. The weighting remains a valid *operator* (synthetic two-noise-level test at
> §5 still holds: 0.175 vs 0.075); what is corrected is the attribution of the gain on real data. In
> addition, $\rho$ — listed below as an unmeasured HIGH risk — is now measured.

---

## 4. What is deliberately excluded, and why

* **Any source metric entering the plan.** M3 (corr 0.782) is the largest single number in the
  table and it is a trap: the source metric is index-aligned to the same 200 concepts as the
  gallery, so making the FGW plan consistent with it drives the plan toward the identity, which
  under this protocol *is* the answer key (M1, job 645455; control 645593 turned +15.4pp into
  −14.5pp). Sources may enter only as a **permutation-invariant regulariser** on the fusion
  weights — never as a term in the plan's objective. This is a hard architectural constraint, not
  a tuning choice.
* **Spectral truncation in any space.** Falsified in the whitened space (0/30, P1) and in the
  unwhitened space (0/10, A1/A2), with the unwhitened premise *satisfied* (M4 kills the
  "concentration is absent" excuse). The reason is now understood: truncation reduces the
  structural object's **richness**, and the term is monotonically better with a richer one
  (on−off +3.85pp whitened vs +2.50pp unwhitened). Denoising must **not** reduce rank.
* **Any thresholded/graph/phylogenetic functional.** M5 = 0.269 is the arithmetic consequence of
  M2 = 0.593 (§1.3); A4 confirmed it at three densities (0/30 cells).
* **Whitening removal.** M4 shows whitening flattens the spectrum 25.6 → 85.0 and the *flattened*
  metric is the one that works. Whitening is load-bearing and stays.

---

## 5. The architecture, layer by layer

Each layer names the measurement that licenses it and the prediction that can falsify it.

**L1 — View normalization (unchanged).** Per-view moment-match + whitening, exactly as deployed.
*Licensed by:* M4 (the whitened metric is the one with the term's +3.85pp); SATTC independently
reached the same conclusion. *Falsifier:* a whitening-free arm beating it paired.

**L2 — Structure ensemble (new).** Build `K` order-2 estimates of the shared concept structure:
the query mean-metric, the per-repetition-block metrics (split the 80 repetitions into blocks so
the blocks are *independent* samples, not resamples), and the gallery metric. All are in the
whitened frame.

**L3 — Reliability-weighted fusion (new, core).** Estimate per-view weights from **label-free
agreement**: each view's residual to the leave-one-out fused estimate. Fuse in the shared
eigenbasis so the result is a legitimate metric and is **permutation-invariant with respect to
concept order** — the property that keeps the M1 channel shut (this is asserted in
`smoke_test.py` §14/§15 for every prior we built; L3 inherits that test).

**L4 — Correspondence from the denoised structure (new).** Solve the FGW plan against the fused
structure with the term weighted by the *fused confidence*, and — this is the δ(recovery-flatness)
repair — make the structural weight **scale with the measured reliability of the view it came
from**, so the operator stops being flat in input quality (Prediction 2).

**L5 — Repetition-aware ranking (new).** Propagate the repetition spread into a *soft*
reliability weight per query rather than the binary agree/disagree gate that was measured to
reject 0.91 of its input. Ranking = CSLS over the L4 scores, tie-broken by repetition consensus.

**Why this is genuinely new.** The literature's axes are: which transform (CLIP: none; SCORE:
orthogonal; RA/SATTC: congruence; GWOT: isometry; HyFI: hyperbolic embedding). v11 does not add a
transform. It changes *what is being estimated* — from a map between views to the shared structure
itself — and derives the estimator from measured per-view reliability. The reframe is falsifiable
(Prediction 1) and it explains, with one mechanism, why four previously published-shaped
approaches in this project failed (§1.3) while three succeeded (§1.2).

---

## 6. Honest risk register

| Risk | Severity | Read-out |
|---|---|---|
| Pooling cannot beat 0.593 (Prediction 1 fails) | fatal | measured before any GPU time |
| The views are not independent (same encoder → correlated errors) | high | measurable: off-diagonal of the view-residual covariance |
| Reliability weights are uninformative (Prediction 2 fails) | medium | measurable in the same probe |
| L4's confidence scaling does nothing | medium | paired on−off, the standard twin |
| Best case is a small gain over 54.25 | medium | M10's spread is ±10.97; a +2pp mean gain with a tighter sd is the realistic target, and a *tightened spread* is itself the publishable claim |

The last row is the honest expectation, and it is worth stating plainly: **with M10 = 54.25 ± 10.97,
the most credible high-value outcome is not a large mean gain but a large variance reduction.** No
method in the surveyed literature reports the cross-subject spread as a headline; a label-free
test-time method that leaves the mean and halves the spread is a stronger and more useful claim
than one that adds 1pp to the mean.

---

## 7. Immediate next step (before any architecture code)

Implement §3's Prediction-1 probe on cached embeddings: `scripts/probe_fusion.py` (runs on Slurm
GPU nodes, not the login node). It reuses `outputs/src_metric/v8/*.npz` (9 source clouds per
encoder) and
the cached test image features in `data/cache/targets/`. It reports, per encoder:

* corr(pooled structure, di) vs corr(single-view structure, di) = 0.593, as a function of K;
* the same with the concept axis shuffled (the M1-style control);
* per-view reliability weights and their correlation with leave-one-out gain (Prediction 2);
* whether the target's repetitions are the top-weighted view (Prediction 3).

**Go/no-go:** build L2–L5 only if the pooled correlation exceeds the single-view value with a
margin that grows with K and that the shuffled control does not reproduce. Report the failure
openly if it does not.

### 7.1 RESULT — Prediction 1 PASSES (job 645697, `dgx-10`, NVIDIA H800)

Run as a Slurm GPU job, per the standing rule that probes go through Slurm. `scripts/probe_fusion.py`,
`slurm/probe_fusion.sbatch`, 10 folds, full output at `outputs/probe/fusion/fusion_probe.json`.

| K | cross-modal pool | single | **shuffled control** | within-EEG LOO pool | LOO single |
|---|---|---|---|---|---|
| 1 | 0.2093 | 0.2093 | −0.0006 | 0.6978 | 0.6978 |
| 2 | 0.2256 | 0.2093 | +0.0010 | 0.7161 | 0.6576 |
| 4 | 0.2399 | 0.2093 | −0.0003 | 0.7886 | 0.6717 |
| 8 | **0.2471** | 0.2093 | +0.0003 | **0.8057** | 0.6785 |

Paired per fold, pool(K=8) − single = **+0.0379**, **t = +42.4, 10/10 subjects positive**, and
**monotone in K**. The correspondence-destroying control stays within ±0.001 of zero for every K, so
the gain is the *shared structure* and not an artefact of averaging — an unordered pool gains
nothing. The within-EEG arm is stronger still (0.678 → 0.806) and needs no image features and no
cross-space assumption at all.

**One honest caveat, and it does not touch the verdict.** The cross-modal *absolute* baseline here is
0.209, not M2's 0.593, because this probe fuses the gallery as a layer-mean of the cached features
rather than through the model's learned `routed` fusion, and the query side is a source subject
rather than the target. The **internal paired comparison is unaffected** (same `di`, same folds) and
the **within-EEG arm is proxy-free**. Fixing the absolute value means computing the deployed gallery
metric from a checkpoint — a separate, cheap step, not a precondition for the go decision, which is
about whether pooling raises SNR, measured directly.

### 7.2 Deployed implementation (L2/L3/L4) and its pre-registered test

`calibration.rep_cloud_scores(..., rep_blocks=B)` + `--fgw-struct-fuse`. The target's own `R`
repetitions are split into `B` contiguous blocks; each block's whitened metric is an independent
estimate of the same 200 concepts; each block is weighted by its agreement with the leave-one-out
mean of the others (**measured, never tuned**); the weighted fusion replaces `de` inside `_fgw_plan`.
`fuse=0` is the shipped single-mean metric and is the paired twin of every fused cell.

Why this is not another reduction: P1, A1/A2 and A4 all *deleted* directions or binarised, and all
lost; §12 of the v10 note measured the structural term to be worth **more** with a richer object
(+3.85pp vs +2.50pp). Fusion deletes nothing.

`smoke_test.py` §16 pins the operator: `fuse∈{0,1}` are a strict no-op; a noisy block is
down-weighted (0.175 vs 0.075 in a two-noise-level synthetic); relabelling concepts relabels the
scores identically and leaves the weights untouched; a pure-noise cloud self-reports `is_flat` and
disables the fusion.

**Pre-registered test** (`slurm/v11_struct_fuse.sbatch`, job 645716, 10 subjects, one seed, paired):
`fuse=B` passes iff it beats its own `fuse=0` twin, paired per subject, on a majority of the ten
subjects at `R=80`. The `fuse=B (structural-off)` family is read separately and must stay flat — a
gain that appears with the structural term **off** means the fusion leaks through the cross-modal
cost, not the structure, and falsifies the mechanism. First fold (`sub-01`) preview: `fuse=8`
62.50 → 65.00 (**+2.50pp**), structural-off pinned at 60.00, i.e. the structural term's worth doubles
from +2.50 to +5.00; at `R=20` the same fusion is worth +3.50pp, larger exactly where the single
estimate is noisiest — the shape the probe's K-curve predicts.

### 7.3 RESULT — job 645716, 10 subjects, one seed, paired (`fuse ∈ {0,2,4,8,16}`)

`fuse=0` reproduces **54.25** exactly, the shipped FGW baseline of job 645630, so the twin *is* the
deployment and the fused cells differ in exactly one argument.

| R=80 | fuse=0 | fuse=2 | fuse=4 | fuse=8 | fuse=16 |
|---|---|---|---|---|---|
| Top-1 | 54.25 | 54.25 | 54.65 | 54.95 | **55.10** |
| Δ vs twin | — | **+0.00** | +0.40 | +0.70 | **+0.85** |
| t (10 subj) | — | nan | 0.79 | 1.30 | **1.72** |
| subjects positive | — | 0/10 | 6/10 | 6/10 | 6/10 |

**Mechanism: CONFIRMED.** The `structural-off` arm is *bit-invariant to B* (50.40 for every B), so
the fusion reaches the score **only through the structural term** — it does not leak through the
cross-modal cost. And the term's own worth grows with the number of pooled views:

| B | 0 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| structural term (on − off) | +3.85 | +3.85 | +4.25 | +4.55 | **+4.70** |

**Effect size: SUB-THRESHOLD.** The gain is monotone in B and positive on a 6/10 majority, but it
peaks at **+0.85pp with t = 1.72**, short of the pre-registered `t > 2` decisive bar. Per the
criterion written in §7.2 *before* the run, this is **NOT a passing headline architecture**.

Two structural facts fall out of the grid and are worth recording:

* **`B=2` is an exact no-op (+0.00 on all ten subjects, identically).** With two blocks each block's
  leave-one-out reference is just the other block, the reliability weights come out equal, and the
  fusion reduces to the plain mean of the two halves ≈ the full mean. So the mechanism needs `B ≥ 4`
  to have anything to average; `B=2` is a useful *null* that proves `fuse=0` is the honest twin.
* **`R=20` is non-monotone and collapses at `B=16`** (fuse=4 +0.95 on 7/10, fuse=8 +0.55 on 4/10,
  fuse=16 −0.25). This is the opposite of the probe's "more views help most where the single
  estimate is noisiest" prediction, and it is explained by block size: at `R=20` and `B=16` a block
  holds 1–2 repetitions, and a metric estimated from 1–2 samples of a 200-concept space is noise, so
  the fusion starts injecting it. **Number of blocks is bounded by repetitions-per-block, not free** —
  the probe's K-curve pooled nine *subject-mean* clouds (each already ~80 reps), not nine single reps.

**Reading (pre-registered, §6 risk register):** the probe's measured **structure-SNR gain
(+0.038 metric correlation) converts to only +0.85pp of retrieval**, so the binding constraint at
the retrieval stage is **downstream of the structure estimate** — matching the alternative branch
written into §3's go/no-go before the run. This is a WEAK POSITIVE: the mechanism is real and
correctly routed, but it is not a headline. Not falsified, not sufficient.


