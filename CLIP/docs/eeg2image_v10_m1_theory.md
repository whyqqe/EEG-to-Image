# Cross-subject alignment as second-order metric alignment (v10 / M1)

Status: **stage 2.5 implemented and running** (job 645455, 10 folds x 3 seeds).
Predecessors: `docs/eeg2image_v10_pipeline.md` (stage 1 gating, stage 2 shrinkage).
Last updated: 2026-10-05.

---

## 0. The one-paragraph version

Every method in this task -- InfoNCE, CSLS, SCORE's coordinate recovery, SATTC's
structure-aware PoE -- aligns **points**. The quantity that is actually shared across
subjects is the **metric tensor on the concept manifold**, and this project measured it
directly (`corr(D_eeg_s, D_eeg_t) = +0.565`, chance −0.003, 45 pairs). Aligning the metric
is a *second-order* problem; the correct object is the Fused Gromov–Wasserstein distance,
not a point map. M1 is the deployment-side instance: replace the target's own (noisy)
EEG-side metric with a **source-averaged template**, which lowers the estimator variance
*without* the isotropic bias that made stage 2's shrinkage fail.

---

## 1. The problem, in manifold terms

Let `M` be the **concept manifold** (the 200 THINGS concepts). Measured intrinsic dimension:
**`d_M` ≈ 8–16** (participation ratio 8.3; 93–96% of query variance and 98% of target
variance in the first 16 principal directions). Subject `s` induces an embedding

```
phi_s : M -> R^d        (d = 64 in the alignment space)
```

and every method here is an attempt to invert `phi_s ∘ phi_t^{-1}`.

### 1.1 What is shared, and how it was measured

| quantity | measured | chance | reading |
|---|---|---|---|
| `corr(D_eeg_s, D_eeg_t)` — cross-subject concept metric, 45 pairs | **+0.565** (range .460–.668) | −0.003 | the metric is shared |
| `corr(D_eeg_s, D_image)` — EEG vs image concept metric | **+0.615** | — | the image side is a valid anchor |
| orthogonal residual vs unconstrained linear | **0.885 vs 0.922** | — | the residual is **not** purely orthogonal |
| cross-subject concept geometry, 500-way | 21.4% | 0.20% | "subject = modality" is real, not noise |

The third row is the load-bearing one for the argument below.

---

## 2. Why the existing methods cannot close the gap

An alignment can be **0th-order perfect** (every point matched correctly) while the metric
is **arbitrarily distorted**. That is exactly what a modality gap is, and it is why
"aligning the marginals" and "aligning the correspondences" both leave structure on the
table.

| family | what it constrains | constrains the metric `g`? |
|---|---|---|
| InfoNCE / CLIP | which point pairs with which (0th order) | no |
| CCA / Procrustes / SCORE | the coordinate frame (a point map) | no — only the **orthogonal** part |
| CSLS / SATTC score scaling | a monotone transform of the similarity matrix | no |
| MMD / JDD | the marginals / joint marginal | no (0th order + a centroid estimate) |
| **GW / FGW** | the intrinsic metric `g` itself | **yes** |

**The measured gap that makes this decisive.** The orthogonal residual is 0.885 against an
unconstrained linear residual of 0.922. A Procrustes fit — which is what SCORE estimates —
can, by construction, only remove the orthogonal component. The 0.037 difference is the
part of the misalignment that **no first-order method can reach**, and it is a measurement,
not an argument.

---

## 3. The objective

```
    min_{pi}   (1 - alpha) * <pi, S>  +  alpha * sum_{i,j,k,l} (D^e_ik - D^i_jl)^2 pi_ij pi_kl
                 \____ linear (point) __/          \________ Gromov-Wasserstein (metric) ____/
```

with `D^e` the EEG-side concept metric, `D^i` the image-side one, and `alpha` the
structure/point trade-off. This is Fused GW; `alpha = 0` is exactly the shipped Sinkhorn
operator (verified bit-identical), so the structural contribution is always measured against
a reproduction rather than an assumption.

**There is a theorem available for the training-side variant.** MPS-Tuning proves that the
order-`p` GW distance is upper-bounded by the `L_p` norm of the difference of the
corresponding Gram matrices,

```
    GW_p^p(., .)  <=  || Gram - Gram' ||_p
```

so a Gram-matching loss is not a heuristic proxy for manifold alignment — it is a
**certified surrogate**. This project already has `losses/relational.py:gram_distill_loss`.
It has failed four times *in training* (T2′, T2′′, the source-only recovery episode, and the
concept frame), and §5 gives the diagnosis the theorem makes available.

### 3.1 Why `alpha = 1` collapses, and why that is predicted

The measured profile is an **interior peak** with catastrophic collapse at `alpha = 1`
(1.90%). That follows directly from `+0.565`: the concept metric alone is *shared but
moderate*, so structure-only alignment is under-determined on ~43% of the metric. The
answer is **fused**, and the collapse is confirmation rather than embarrassment.

---

## 4. M1: the source-averaged metric template

### 4.1 The mechanism, stated as a bias-variance problem

Let the target's EEG-side metric estimate be `de_t = de* + e_t` and let a source subject's be
`de_s = de* + b_s + e_s`, where `de*` is the shared metric, `b_s` the subject-specific
structure, and `e` estimation noise. M1 uses

```
    de_used = (1 - m) * de_t  +  m * de_src ,      de_src = mean over S source subjects
```

**The inference is standard.** With independent noise of variance `sigma_t^2` and
`sigma_s^2/S`:

```
    Var(m) = (1-m)^2 sigma_t^2  +  m^2 sigma_s^2 / S      ->   argmin m* = S/(S+1) = 0.90  (S=9)
    Var(m*) = sigma_t^2 / (S+1)                            ->   a 10x variance reduction
```

**But `de_src` is biased for the target** — the `+0.565` correlation leaves ~43%
target-specific structure, i.e. `b` does not average out. So the true optimum is pulled
inward, and the pre-registered prediction is a peak at roughly **m ∈ [0.5, 0.9]**.

### 4.2 Why NOT shrinkage (the empirical reason this specific fix)

Stage 2 swept `--rep-shrink ∈ {0.1, 0.2, 0.35, 0.5, 0.7}` (job 645436, 30 runs) and the
gain fell **monotonically at every R**:

| shrink | gain(R=80) | gain(R=20) | gain(R=1) |
|---|---|---|---|
| 0.1 (deployed) | **+3.08** | **+1.12** | −0.43 |
| 0.2 | +2.85 | +0.77 | −0.20 |
| 0.35 | +1.95 | +0.70 | −0.45 |
| 0.5 | +1.23 | +0.37 | −0.42 |
| 0.7 | +1.13 | +0.10 | +0.12 |

Shrinkage trades variance for **isotropic bias** — it pulls the covariance estimate toward
the identity, degrading the very metric the structural term exists to use. **Pooling across
subjects is the one variance-reduction route that carries no such bias**, and `+0.565` is
the measurement that licenses it. Stage 2's negative result is what turns "a source
template" from a plausible idea into the *only remaining* option.

### 4.3 How the template is built (and why it cannot drift)

`--dump-src-metric` runs **inside the fold that owns the encoder**, at model load, before the
eval block reads the file it just wrote. The template must be the subjects *this* checkpoint
was trained on, embedded by *this* checkpoint, on *this* normalisation; a separate builder
script would re-derive channels / mvnn / feature-set from config and diverge the first time
one of those defaults changed. Concept indices line up across subjects **by construction**:
THINGS-EEG2 tests every subject on the same 200 concepts, so no alignment step is needed to
make `src_means[s][c]` comparable to `z_reps[c]`.

Whitening: the source means are whitened by **this target's** map, not their own subjects'.
Whitening them by their own maps would put the template in a different frame and the blend
would be a geometric non-sequitur.

---

## 5. Falsifiable predictions

| id | prediction | what falsifies it |
|---|---|---|
| **H1** | the mix curve has an **interior peak** | a peak at m=0 (template carries nothing) or m=1 (target metric is pure noise — already refuted by +0.565). **Either endpoint is a falsification, not a tuning result.** |
| **H2** | the gain is **larger at R=20 than at R=80** | flat or inverted in R ⇒ the mix is not acting as variance reduction |
| **H3** | `corr(de_src, de_target)` is **high and R-dependent** | near 0 ⇒ the template is a differently-shaped metric and mixing must hurt |

H3 is recorded **per run, before any accuracy is read**, because it is the one number that
predicts pay-off in advance: it is the shared component relative to the total, i.e. exactly
the signal-to-noise the blend can exploit. It can kill the direction on a single fold.

### 5.1 Verified invariants (smoke test §13)

| assertion | result |
|---|---|
| `src_mix = 0` bit-identical to the shipped operator | **pass** (and to a run with no template at all) |
| template == the target's own metric ⇒ `mix=1` is an exact no-op | **pass**, `max|diff| = 0.0`, corr exactly 1.0000 |
| diagnostic separates an informative template from noise | **+0.997 vs +0.004** |
| `m=0` cells reproduce stage 1 bit-for-bit | **pass** (sub-01: 62.50 / 39.50 / 59.50) |

The second row is the strongest available check and it is a closed-form known answer: when
the template *is* the target's metric, the optimal blend is the identity, so any movement
would have to come from a scale, normalisation or axis bug rather than from the template.

### 5.2 Single-fold preview (sub-01/seed2025 — the *easiest* fold, base 62.50)

| R | mix=0 on | mix=0 gain | mix=1 on | mix=1 gain | Δ(mix) | diag corr |
|---|---|---|---|---|---|---|
| 80 | 62.50 | +2.50 | 78.00 | +18.00 | **+15.5** | 0.70 |
| 40 | 52.50 | +3.50 | 71.00 | +22.00 | +18.5 | 0.67 |
| 20 | 39.50 | +1.50 | 64.50 | +26.50 | **+25.0** | 0.60 |
| 10 | 31.50 | +7.00 | 52.50 | +28.00 | +21.0 | 0.53 |
| 5 | 17.00 | −0.50 | 36.50 | +19.00 | +19.5 | 0.42 |
| 1 | 3.50 | −1.50 | 8.50 | +3.50 | +5.0 | 0.15 |

Both H2 (**+25.0 at R=20 vs +15.5 at R=80**) and H3 (corr falls monotonically with R, exactly
where the target's own metric degrades) hold on this fold. **This is one fold and one seed.
It is a preview, not a result** — the effect size is large enough that it must be treated as
a hypothesis until the 10×3 grid lands, and sub-01 is the easiest fold.

---

## 6. Where this sits in the literature

| work | what it aligns | modality / task | relation |
|---|---|---|---|
| **FUGW** (NeurIPS 2022) | fused unbalanced GW on cortical surfaces | fMRI, inter-subject | closest formulation; not EEG retrieval |
| **MGMCL** (2026) | GW on SPD manifolds | EEG, cross-subject **emotion** | GW cross-subject EEG exists — **not in retrieval** |
| **LFM** (NeurIPS 2024) | spectral (functional maps) latent alignment | generic representation alignment | **never applied to cross-subject EEG** |
| **MPS-Tuning** | GW ≤ Gram bound | VLM fine-tuning | supplies the **theorem** used in §3 |
| **ToMCLIP / ToMA** (25–26) | persistent-homology / topology alignment | multilingual VLM, semi-supervised VL | $H_1$ adds information beyond $H_0$ |
| **SCORE** (2026) | orthogonal coordinate recovery | EEG↔image retrieval | **first-order**; our reference operator |
| **SATTC** (CVPR 2026) | similarity-matrix structural PoE | EEG↔image retrieval | **first-order**; 14.8% Top-1 on its protocol |

**The gap is clean**: GW / manifold alignment is used for fMRI surfaces, VLM latent spaces
and EEG *emotion*, but not for cross-subject EEG→image retrieval, where every method is
first-order. We also hold a measurement none of these papers have (the `+0.565` cross-subject
metric correlation, and the non-orthogonal residual).

### 6.1 A protocol caveat that is itself publishable

The same nominal benchmark (THINGS-EEG2 inter-subject 200-way) is reported at **5.5% (ATM),
9.2% (standardised), 14.8% (SATTC), 23.2% (NeuroBridge), 28.5% (BI-Cap), 34.4% (SAMGA),
53.23% (SCORE)** — a 10x spread. Any claim of the form "ours is +1.14pp over X" is therefore
a claim about `X`'s protocol, not about the method. The defensible statement is the
**internal, operator-paired** one (FGW vs SCORE's operator on the identical banked encoder,
identical folds, identical R: +6.83pp, `t = 9.42`, 10/10 subjects), and M1 extends that
discipline.

---

## 7. Known risks

1. **Effect size.** The single-fold preview is far larger than the +3.08pp that turning the
   structural term *on* is worth. That is either a genuine mechanism or a sub-01 artifact;
   the 10×3 grid decides.
2. **Training-side history.** Four structural losses failed in training (§3). M1 is
   deliberately **deployment-side**, where the only structural win so far lives. The
   training-side version (M2: restrict the Gram term to the `d_M ≈ 16` subspace, then apply
   the MPS-Tuning bound) stays behind that evidence.
3. **Source test data.** The template is built from source subjects' *test*-split reps. The
   target is label-free and transductive only over its own test reps (already reported
   separately), which matches SCORE's source-only-episode regime — but a robustness variant
   built from source *train*-split concepts removes the question entirely.
4. **Editing during jobs.** `run_eval.py` is read per invocation and `summarize_v10.py` at
   the end; editing either mid-run corrupts the run (this already happened once, job 645453).
   Freeze both until the queue drains.

---

## 9. STAGE 3 — results, and what they actually said

Three source-using variants were built and each was falsified, in a way that **brackets the
entire space** of "use the source data". This is the most useful thing the stage produced.

| variant | how the source is used | Top-1 effect | verdict |
|---|---|---|---|
| **M1** (job 645455) | source template **blended into** `de`, indices live | **+15.37pp** (30/30, t=16.97) | **LEAKAGE** |
| **C1** (job 645606) | monotone elementwise map Φ fitted on sources | **+0.00pp**, corr +0.0019 | **no headroom** |
| **P1** (job 645612) | spectral rank prior (permutation-invariant scalar) | negative at every rank | **falsified** |

### 9.1 M1 was an index leak — the control that proved it

The permutation control (job 645593) shuffled the **concept axis** of the source template,
leaving geometry intact:

| fold | m=0 | m=1 (real) | m=1 (shuffled) | real Δ | shuffled Δ |
|---|---|---|---|---|---|
| sub-01 | 62.5 | 78.0 | **41.0** | +15.5 | **−21.5** |
| sub-04 | 35.0 | 60.0 | **24.5** | +25.0 | **−10.5** |
| sub-08 | 48.0 | 58.5 | **33.5** | +10.5 | **−14.5** |

`corr(de_src, di)` collapsed 0.782 → 0.001. **Mechanism**: `de_src[i,j]` carries concept `i`'s
identity in its own subscript and `di[i,j]` carries the gallery slot in its own; since the
gallery order *is* the concept order under this protocol, `de_src ≈ di` makes the FGW optimum the
**identity plan** — which is the answer key.

Corroborating evidence: the gain is **flat in R** (+15.4 at R=80, +16.0 at R=20) while
`corr(de_src, de_target)` falls 0.674 → 0.138. Variance reduction would show the *opposite*
R-dependence, so the variance story was never the mechanism.

### 9.2 C1 has a provable ceiling

A monotone elementwise map cannot reorder anything, so the most it can buy is
`corr − Spearman`. Measured post-calibration correlation **0.594** against pre **0.593** —
exactly on that ceiling. A synthetic bound under the most favourable conditions (a shared
monotone deformation) is **+0.0088**. Φ was verified **non-degenerate** (rank-correlation with
identity 0.989), so this is a real negative and not a broken fit.

**The 0.593 → 0.782 gap is ORDER, not SHAPE.**

### 9.3 P1 and the whitening obstruction (the deepest finding)

A spectral rank prior is permutation-invariant (relabelling conjugates the metric, the spectrum
is unchanged), so it **cannot** leak — and the smoke test asserts bit-identical output under a
source-concept shuffle. It is also not monotone, so the C1 ceiling does not apply. It was the
only candidate left, and it failed:

| rank (R=80, sub-01) | Top-1 | structural term (on−off) |
|---|---|---|
| **0 (full)** | **62.50** | **+2.50** |
| 8 / 16 / 32 / 48 | 58.0 / 57.5 / 59.0 / 58.5 | −2.0 / −2.5 / −1.0 / −1.5 |

Full grid (job 645612, 10 folds × 3 seeds) — **0/30 folds positive**:

| rank | Top-1 (R=80) | Δ vs full | t | pos | structural term |
|---|---|---|---|---|---|
| **0 (full)** | **53.80** | — | — | — | **+3.08** |
| 8 | 46.83 | −6.97 | −14.64 | **0/30** | −3.88 |
| 16 | 46.88 | −6.92 | −14.59 | **0/30** | −3.83 |
| 32 | 47.17 | −6.63 | −14.56 | **0/30** | −3.55 |
| 48 | 47.13 | −6.67 | −15.49 | **0/30** | −3.58 |
| src | 47.15 | −6.65 | −15.11 | **0/30** | −3.57 |

At R=20 the same pattern at smaller scale (−3.8 to −4.1pp, t ≈ −6.5, 2–3/30). Truncation does
not merely fail to help — it **inverts the sign of the structural term** at every rank and both
R, monotonically in the wrong direction.

**Why, and this is the reportable part.** The source-effective-rank measured *offline* on the
banked templates is ~29 (top-16 = 64.8% of variance), which motivated the low-rank hypothesis.
But the rank computed *inside* `rep_cloud_scores` is **56 (R=80) / 45 (R=20)**, because the
source metric is whitened by the target's map first — and **whitening flattens the spectrum**.
That is what whitening is *for*. Consequently:

> The deployed stack's own whitening step makes the EEG-side metric **broadband**, which destroys
> the spectral concentration that any low-rank manifold prior relies on. Low-rank denoising of
> `de` is inapplicable *by construction*, not by tuning.

This retro-explains the stage-2 shrinkage failure too: shrinkage and truncation are both
*rank-order-preserving* manipulations of a metric that whitening has already made broadband.

### 9.4 The resulting constraint on the whole design space

| source usage | outcome |
|---|---|
| indices participate | **leaks** (+15.4 → −14.5) |
| monotone / elementwise | **no headroom** (≤ +0.009 corr, +0.00pp) |
| permutation-invariant **and** applied to the whitened metric | **destroyed by whitening** (broadband) |

The third row is the important one: it says the obstruction is not the prior but the **space**.
A usable permutation-invariant prior must be applied to a metric that has **not** been whitened —
i.e. the spectral/topological prior has to be injected **before** whitening, or whitening has to
be abandoned in favour of a metric-preserving normalisation. That is a concrete, testable
architecture change and it is the recommendation for P2 (next).

### 9.5 Verified invariants (smoke §13–§14)

| assertion | result |
|---|---|
| `src_mix=0` / `spec_rank=0` / `src_calib=False` bit-identical to shipped | **pass** |
| M1 template == target metric ⇒ `mix=1` exact no-op | **pass**, `max|diff| = 0`, corr 1.0000 |
| M1 diagnostic separates informative template from noise | **+0.997 vs +0.004** |
| P1 **permutation invariance**: source concept shuffle ⇒ bit-identical output | **pass** |
| spectral truncation symmetric, rank ≤ k, spectrum permutation-invariant | **pass** |
| all 30 `m=0` cells reproduce stage 1 bit-for-bit | **pass** |

---

## 10. The full architecture — A1/A2 (unwhitened metric + spectral prior) and A4 (topological reference)

Stage 3's three falsifications converge on one diagnosis: **the prior is not the problem, the
space is.** `_whiten_from_cloud` flattens the metric spectrum (effective rank 29 → 61), so any
low-rank or manifold prior is applied to a **broadband** object where "signal directions" no
longer exist as a small set.

The submitted architecture is therefore **two components that relocate the prior out of the
whitened space, plus the control that keeps them honest**. It is implemented, smoke-tested
(§15), and submitted as `slurm/v10_full_arch.sbatch` at one seed over all ten subjects, paired
per fold.

### 10.1 A1/A2 — the unwhitened metric, and the same spectral prior on it (`--fgw-raw-metric`)

The repair is not a new prior. It is **the same spectral prior evaluated where the low-rank
claim actually holds**. `q_raw` is the unwhitened concept mean, `de_raw` its metric; the
effective rank of the raw and whitened metrics is recorded side by side on every row so the
premise is *checked on every run* rather than assumed:

- `eff_rank_de_unwhitened`, `eff_rank_de_whitened` — the premise, measured.
- `p2_premise_ok` — the **kill switch**, pre-registered: the family is abandoned if the
  unwhitened effective rank exceeds 50. This is the same threshold the P1 falsification was read
  through, so a positive result can never be produced by quietly moving it.
- `spec_rank_applied_to = "unwhitened"` — the relocation is stated in the output, so a reader
  cannot mistake it for the P1 cells that share the `--fgw-spec-rank` flag name.

`rawspec=0` is the un-truncated raw metric, i.e. **the paired twin of every truncated cell**;
those two rows differ in exactly one argument, which is what makes the contrast an experiment.

**No leakage, by construction.** `q_raw` is the target's own feature, `de_raw` its own metric,
and the truncation index set is a permutation-invariant spectral object. Nothing is indexed by
concept, so the M1 channel is absent — asserted, not argued (smoke §15 shuffles the source
concept axis and requires bit-identity).

**Pre-registered pass/fail.** Interior peak in `k` near the measured effective rank = the prior
is calibrated in the right space. Monotone decline = falsified *again*, and that would be the
informative negative: it would say the unwhitened tail is signal the structural term is using.

> **VERDICT (§12): FALSIFIED, and the PREMISE WAS SATISFIED.** Effective rank measured
> **25.6 ± 4.2** (unwhitened) vs **85.0 ± 4.5** (whitened) over the ten subjects, so the kill
> switch passed 10/10 and the concentration the family needed is genuinely there. It still lost:
> the un-truncated raw metric is **−1.35pp** below the whitened one (t = −2.21, 1/10 positive),
> and every truncation is **≈ −5.5pp** with **0/10** positive and a flat rank curve
> (16: −5.50, 32: −5.40, src: −5.55) — no interior peak anywhere.

### 10.2 A4 — the topological reference (`--fgw-topo`)

The metric's **fine magnitudes** are exactly what a whitener destroys, and exactly what P1 tried
to read a spectrum off. Connectivity is not: it is the coarsest structural object that survives
a whitener. So the structural reference is replaced by the **ε-threshold graph** — the metric
binarised to 0/1 adjacency, which has no spectrum to flatten and nothing for a whitener to
distort.

Three design points, each of which fixes a concrete failure mode rather than adding a knob:

1. **Two scales, one per domain.** The EEG cloud and the image gallery live in different spaces
   with different distance scales, so a *single* shared ε would build a blob on one side and a
   singleton cloud on the other. Each side gets the characteristic scale of its **own** distance
   multiset — which is what makes the comparison **topological (shape)** rather than **metric
   (magnitude)**. `topo_mode = "auto_per_domain"` records this; `--fgw-topo-eps` overrides both
   with one number and is labelled `"shared_eps"` as the shared-scale ablation.
2. **The scale is exact and needs no TDA library.** An ε-graph at a fixed quantile is the
   standard construction and has no failure mode: `eps` is the `q`-quantile of the off-diagonal
   distance multiset, so the graph has **exactly** `q·C(C−1)/2` edges by construction and its
   sparsity is a chosen constant rather than a byproduct. The first version used the H₀
   persistence gap instead, and on real data it could not abstain when no gap existed: it
   returned a scale giving **11940/19900 = 60% density** — a near-complete graph pretending to be
   a topology. A criterion that silently degrades to a useless answer when its premise is absent
   is worse than one with no premise.
3. **Permutation invariance, asserted.** The scale is a scalar read off a distance *multiset*, so
   relabelling concepts cannot change it. Smoke §15 pins this with bit-identity, and pins that
   `_topo_graph` commutes with relabelling — the properties that close the M1 channel.

`topo=auto (structural-off)` is emitted as the ablation twin, and the verdict requires the
topological reference to beat **both** that twin (the term does something) **and** the metric
reference it replaces (topology is a *better* structural object than the metric). Beating only
the ablation but not the metric is reported as a wash, not a win.

> **VERDICT (§12): FALSIFIED at every density.** q = 0.05 / 0.10 / 0.20 give
> **−6.95 / −6.65 / −7.40pp** against the metric baseline (t = −9.00 / −5.24 / −6.27, 0–1 of 10
> positive) and **−3.10 / −2.80 / −3.55pp** against their *own* structural-off twins
> (t = −3.99 / −1.81 / −2.93). No density beats its own ablation, so the topological object is
> **actively worse** than the metric, not merely neutral. The diagnostic says why: the query and
> gallery graphs overlap at **Jaccard 0.269 ± 0.018** (`topo_graph_jaccard`) — the two domains
> share only about a quarter of their 10%-nearest-neighbour structure, so imposing one on the
> other injects a wrong topology rather than a shared one.

### 10.3 What is deliberately *not* in the submitted architecture

- **A3 (`d_M`-subspace correspondence, `rank=16` Procrustes).** Already in the codebase and
  already measured at **−0.50pp** on its own (`subspace_soft_recovery(rank=16)`). It is not
  re-sold as new; the diagnostic `subspace_rank` is retained so its energy fraction is visible on
  every row.
- **C2/C3 (16×16 functional maps; persistent homology).** Re-scoped under §9.4: they are the same
  family and would hit the same broadband obstruction if applied after whitening. A4 is the
  cheaper member of that family with the obstruction removed by construction — topology is
  invariant to monotone rescaling — which is why it is the one submitted and the fuller TDA is
  left until A4's sign is known.

### 10.4 The run protocol (why one seed)

The three preceding stages were run at three seeds each. The request here is a **fair
comparison rather than seed inflation**, and fair means **paired on the same fold**: every arm is
read against its own twin — same checkpoint, same fold, same repetition subset, one argument
different. Because smoke §14/§15 assert the neutral setting (`spec=0`, `rawspec=0`) is
bit-identical to the shipped operator, the paired contrast is against a *reproduction*, not a
reimplementation, and the pairing — not the seed count — carries the inference. One seed over ten
subjects is `n=10` paired observations with the between-seed variance term removed by design.


---

## 11. How this reframes the contribution

The three falsifications plus the M1 permutation control are, on their own, a publishable
**methodological** result:

1. A same-protocol benchmark spanning **5.5% → 53.23%** (ATM / standardised / SATTC / NeuroBridge
   / BI-Cap / SAMGA / SCORE) means any "we beat X by 1.14pp" claim is a claim about `X`'s
   protocol. The defensible statement is the **paired, operator-internal** one.
2. **Any method that uses source or reference data must report a correspondence-destroying
   control.** This project's own template flipped from **+15.4pp to −14.5pp** under it. The
   control is one line (permute the concept axis) and it costs nothing.
3. A **constructive** constraint: the three ways to use source information are each provably
   blocked (leak / no-headroom / whitening-obstructed), so the remaining design space is
   precisely characterised. That is more valuable to a reader than a fourth variant that works
   for reasons nobody can name.



- **M3 (spectral / functional maps).** With `d_M ≈ 16`, express the correspondence in the
  eigenbasis of the concept-graph Laplacian: the map becomes a **16×16** matrix instead of a
  64×64 Procrustes (2016 parameters). Severely under-parameterised, and LFM proves the
  framework sound for latent spaces while never having been applied to cross-subject EEG.
  Composable with M1 by pooling source metrics in the spectral domain.
- **M2 (training-side, subspace-restricted).** Only if M1 lands, and gated on a
  pre-registered criterion: does restricting the Gram term to the `d_M` subspace move it from
  harmful to harmless?
- **Topological term.** ToMA found `H_1` complementary to `H_0`; on a 200-concept manifold the
  1-cycles are candidate semantic clusters.
