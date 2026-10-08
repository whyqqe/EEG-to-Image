# v13 / CROMA — Concept-Relational Order-2 Metric Alignment

> A **unified** training architecture for cross-subject EEG→image retrieval, derived from a single
> axiom rather than assembled from modules. Includes the zero-GPU gate that was run **before** any
> training code, and its decisive verdict.
> Status: **gate PASSED on the bias channel, FAILED-and-removed on the variance channel. The
> training term itself was then built, smoked, run and EARLY-STOPPED — see §6: it fires, but it does
> not transfer to the deployed estimator and costs 9–13pp of Top-1.**
> Last updated: 2026-10-06.

---

## 0. TL;DR

The system estimates **one** object: the concept metric $D$. Its estimator error decomposes into
exactly three terms, so training can only have three gradients — and a measurement (job `650112`,
10 folds, no labels, no retrieval) shows **two of the three are dead and one is large**:

| term | measurement | verdict |
|---|---|---|
| **bias** | $\mathrm{corr}(D_{R1},D_g)=0.134 \to \mathrm{corr}(D_{R80},D_g)=0.572$, **10/10 folds +**, shuffle control $\le 0.006$; a 9-subject pooled EEG metric reaches **0.796** (t=+22.2, 10/10), so the residual is **subject-specific bias**, not a ceiling | **GO** — the only live lever |
| **variance** | $\rho \le 0$ on 10/10 folds (exchangeable model rejects on 0/10), $K_\text{eff}=80=R$ | **DEAD** — remove the hinge term |
| **rank** | eff-rank 47.1 (single) / 24.5 (pooled) | **CONSTRAINT** — never an objective |

The architecture therefore has **exactly one** new training term, anchored to an **exogenous,
full-rank** target the encoder cannot shrink — which is precisely why it cannot collapse the way
v12 did.

---

## 1. Axiom, and the (complete) error decomposition

> **Cross-subject EEG→image retrieval is a concept-metric estimation problem.** Deployment reads the
> agreement between the query-side metric $D_q$ and the gallery-side metric $D_g$, where $D_q$ is
> estimated by fusing $R$ noisy views (repetitions).

Every error term of a pooled estimator lives in

$$
\mathrm{IMSE}(D^*) \;=\; \underbrace{\mathrm{Bias}(D^*)^2}_{\text{systematic}} \;+\; \underbrace{\frac{\sigma^2}{K_\text{eff}}}_{\text{variance}},\qquad K_\text{eff}=\frac{K}{1+(K-1)\rho},
$$

with $\rho$ the intra-concept correlation of view errors. There is **no fourth term**; anything a
training objective optimises that is not the gradient of one of these three is orthogonal to the
deployed quantity. That is the test that separates this architecture from the six previously
falsified training-side attempts ($T2'/T2''$/SCORE-episode/G-a, four order-1 alignment pillars,
v12 metric distillation, C1/P1/P2/A4).

### 1.1 Why the three terms, and only these, exist

- **Bias** — a systematic per-view error against the shared structure. Pooling $K$ views divides the
  *independent* part but **cannot touch the shared part**, so any residual bias survives at $R=80$.
- **Variance** — removed by pooling. The only thing training could do is change $\rho$; whether that
  is a lever is an empirical question, measured in §3.
- **Rank** — a *degeneracy* condition, not an objective: a lower bias bought by rank collapse is a
  loss (v12: metric rank 2–3, raw cosine −13.85pp, headline −20.15pp).

---

## 2. Architecture

```
┌─ DEPLOYMENT (unchanged, shipped) ───────────────────────────────────────────┐
│  z = f_θ(x) → {D_r}_{r=1..R} → D* = Σ w_r D_r → FGW(D*, D_g) → scores        │
└──────────────────────────────────────────────────────────────────────────────┘
                      │ training backprops ONLY into the quality of D_r
┌─ TRAINING (the only new part is one term) ──────────────────────────────────┐
│ ① L_bias  = 1 − corr( D(z_e^{(r)}),  stopgrad D(z_i) )   averaged over r    │  ← ONLY new objective
│             target D_g is exogenous + full-rank ⇒ collapse unreachable      │
│ ② L_o1    = InfoNCE(z_e_mean, z_i)                       skeleton           │
│ ③ L_plan  = soft_plan(D*, D_g; τ=0.1)                    operator alignment │
│ ④ L_rank  = 1[effrank(D_eeg) < effrank_base]            guard (constraint)  │
│    (variance hinge REMOVED — see §3; it has no headroom)                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

$$
\mathcal{L} \;=\; \mathcal{L}_{\text{o1}} \;+\; \lambda_{sp}\mathcal{L}_{\text{plan}}
\;+\; \lambda_b\underbrace{\big[1-\mathrm{corr}\big(D(z_e^{(r)}),\ \mathrm{sg}\,D(z_i)\big)\big]}_{\text{only new term}}
\;+\; \underbrace{\mathbb 1[\text{effrank}<\text{base}]\,\mathcal{L}_{\text{rank}}}_{\text{guard}}
$$

### 2.1 Why the bias anchor must be the gallery, and not a self-consistency target

Deployment succeeds $\iff \mathrm{corr}(D_q, D_g)$ is high. The gallery metric is

- the **richest view** (measured $M2=0.593 > 0.565$ the EEG–EEG agreement),
- **frozen and clean** (one fused image embedding, no neural noise),
- **subject-independent and test-visible**,
- **full-rank and not a function of the encoder**.

Hence training each EEG view's metric toward $D_g$ closes the collapse channel v12 opened: v12's
target was *self-consistent and shrinkable* (a student can be correlated with anything at any rank),
whereas $D_g$ is **externally fixed and full-rank** — the encoder cannot make it low-rank.

### 2.2 Why order-1 and order-2 are one principle, not two modules

Both consume the same pair $(z_e, z_i)$; they differ only in what they read out:

```text
L_o1 = -log  exp(z_e·z_i/τ) / Σ_j exp(z_e·z_j/τ)     ← point agreement (the diagonal neighbourhood)
L_o2 = 1 - corr( D(z_e), D(z_i) )                     ← relation agreement (the whole matrix)
```

Order-1 constrains a neighbourhood of $D$'s diagonal; order-2 constrains all of $D$. Deployment reads
all of $D$. So order-1 is the degenerate $R{=}1$ readout of the same principle, not a second module.

---

## 3. The gate: measurement before code (job `650112`)

A zero-GPU-hour precondition, run on the ten banked `v8` seed-2025 checkpoints, with **no labels and
no retrieval**. Script: `scripts/probe_view_independence.py`; Slurm: `slurm/probe_view_independence.sbatch`.

### 3.1 Q1 — bias headroom (and the M1 control)

$\mathrm{corr}(D_{R'}, D_g)$ for $R' = 1,2,4,\dots,80$, plus a concept-axis shuffle control that
destroys the correspondence while keeping the geometry.

| | sub01 | sub02 | sub03 | sub04 | sub05 | sub06 | sub07 | sub08 | sub09 | sub10 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| $R'=1$ | .150 | .134 | .092 | .152 | .099 | .155 | .157 | .158 | .107 | .135 | **.134** |
| $R'=80$ | .593 | .596 | .588 | .516 | .614 | .580 | .546 | .537 | .566 | .588 | **.572** |
| shuffled | −.003 | .002 | −.006 | .001 | −.006 | −.000 | −.003 | −.001 | .004 | .002 | **≤.006** |

**Headroom $= +0.438$, 10/10 folds positive; the control collapses to $\le 0.006$ on every fold.**
The gain is not M1 index leakage (which would survive the permutation).

### 3.2 Q2 — $\rho$ and $K_\text{eff}$ (the variance channel)

Split-half reliability of two **disjoint** repetition halves as a function of how many reps each
averages. With a shared error component (variance $\rho\sigma^2$) plus an independent part
($(1-\rho)\sigma^2$), the average of $m$ views has

$$
\mathrm{corr}(m)=\frac{V_s}{V_s+\rho\sigma^2+(1-\rho)\sigma^2/m}
\;\Longrightarrow\;
\frac{1}{\mathrm{corr}(m)}-1=\underbrace{\frac{\rho\sigma^2}{V_s}}_{\alpha}+\underbrace{\frac{(1-\rho)\sigma^2}{V_s}}_{\beta}\cdot\frac{1}{m},
$$

linear in $1/m$, so $\rho=\alpha/(\alpha+\beta)$ needs **no ground truth**. The model is only
trusted when $\alpha>0$, $\beta>0$, $R^2>0.9$.

**Result:** `rho_raw_mean = −0.064`; the exchangeable model is **rejected on 0/10 folds**. Observed
reliability exceeds the Spearman-Brown independence prediction by $1.29$–$1.69\times$ over $m=2..40$
(sub01: obs $[.046,.119,.264,.469,.671,.812,.847]$ vs SB $[.046,.088,.162,.278,.435,.607,.659]$).

**Reading:** view errors are **not positively correlated; if anything slightly anti-correlated**, so
pooling is *super-additive* and already optimal. $K_\text{eff}=80=R$. **There is nothing for a
training-side variance term to gain**, and the hinge term is removed from the architecture rather
than kept as decoration. (v11 had listed $\rho$ as an unmeasured HIGH risk; it is now measured.)

### 3.3 The gate's sharpest consequence — and the correction it needed

Because $\rho\le0$, pooling already removes **all** the repetition-level variance, so the residual

$$
1-\mathrm{corr}(D_{R80},D_g)=1-0.572=\mathbf{0.428}
$$

is the part pooling cannot remove. An first reading of this said "so 0.428 *is* the bias, and it is
the only quantity training can move". **That is too coarse, and the bottleneck test (§3.5) shows
why**: the 0.428 is a *mixture* of (i) subject-specific bias, which averaging over **independent
encodings** removes, and (ii) an irreducible cross-modal residual, which nothing removes. Only (i)
is reachable by training.

### 3.5 THE DECISIVE BOTTLENECK TEST (what makes the EEG side worth training at all)

Before writing any training code, the obvious kill shot had to be ruled out: if $0.572$ were already
the cross-modal **ceiling** (M2 $=0.593$ is the same object — one subject's full cloud vs the gallery
metric), then no EEG-side denoising could ever move the deployed number and `L_bias` would be futile
no matter how healthy its loss curve looked.

Test: compare the target's own 80-repetition estimate against an **essentially noise-free** EEG
metric — the pool of the 9 source subjects' concept metrics ($9\times80=720$ repetitions and 9
independent encodings) — both moment-matched and scored against the same $D_g$.

| | mean (10 folds) |
|---|---|
| `corr(target80, D_g)` | **0.616** |
| `corr(source9pool, D_g)` | **0.796** |
| difference | **+0.180**, paired $t=+22.2$, **10/10 folds positive** |
| source-pool concept-shuffle control | **−0.0005** |
| EEG side saturated? | **NO** |

**The ceiling hypothesis is dead and the EEG residual is dominated by subject-specific bias.**
Pooling removes repetition jitter ($\rho\le0$) but cannot remove a *subject's own* deviation from the
shared structure; averaging over independent **encodings** can, and does (+0.18). So the error
decomposition is, concretely:

$$
\underbrace{0.616}_{\text{target estimate}} \;\xrightarrow{\ \text{remove subject-specific bias}\ }\; \underbrace{0.796}_{\text{best EEG estimate}} \;\xrightarrow{\ \text{irreducible}\ }\; 1.0 .
$$

**This is the quantitative authorisation for $L_\text{bias}$**: the movable quantity is a
subject-specific bias of size $\approx+0.18$ in metric agreement, and the exogenous anchor $D_g$ is
exactly the target that removes it.

**What this arm is NOT.** It is an *estimability reference*, not a deployable operator: the source
subjects' test concept means are index-aligned to the same 200 concepts, and using them directly is
the M1 leakage channel that turned +15.37pp into −14.5pp under permutation (hence the mandatory
shuffle control above, which collapses). The **training** term is unaffected — it anchors on $D_g$,
which comes from images and carries no EEG index.

### 3.6 The pre-registered prediction, restated for this geometry

Since the gap is subject-specific bias rather than view noise, the discriminator in §4.1 sharpens to:

$$
\textbf{PASS} \iff \Delta\mathrm{corr}(D_{R80},D_g)>0 \ \text{and}\ \Delta\mathrm{corr}(D_{R1},D_g)\ge\Delta\mathrm{corr}(D_{R80})>0,
$$

i.e. training must lift the **pooled** agreement (the deployed estimator), not merely the single-view
one. A run where only $R{=}1$ moves has taught the encoder to be less noisy, not less
*subject-specific* — which the 0.80 reference says is the actual gap.

### 3.4 Secondary findings

- **eff-rank**: 47.1 (single view) $\to$ 24.5 (pooled). Pooling concentrates the metric; the guard in
  §2 must keep post-training eff-rank $\ge$ baseline, since v12 collapsed exactly here.
- **The v11 fusion gain is not "reliability weighting."** The fused weights span only $0.0057$
  (near-flat), because $\rho=0$ leaves no reliability differences to exploit. The observed
  `fuse − mean` $=+0.0121$ (8/10 folds, consistent with the $+1.03$pp Top-1 fusion gain) therefore
  comes from **block structure / sharpening**, not from LOO reliability weights. The v11 document's
  description should be corrected.

---

## 4. Decision experiment (pre-registered)

- **Twin**: $\lambda_b = 0$ vs $\lambda_b = w$; identical seed, data order and folds; 10 folds × 1 seed, paired.
- **Headline cell**: deployment unit `R=80 + fuse=16` (the same cell that shipped 55.10 / 54.83).
- **Guards — all must hold, or a "win" is booked as a collapse:**
  1. `effrank(D_eeg)` $\ge$ baseline (single **and** pooled);
  2. structure-off arm **bit-identical** (no scaffold absorption);
  3. concept-axis permutation control stays dead;
  4. $\mathrm{corr}(D_{R1}, D_g)$ rises — the mechanism is really lifting single-view information.
- **Credible ceiling**: $+1\sim+2$pp (56–57); a larger jump needs a control replicate or is unbooked.

### 4.1 The discriminating prediction (the crux — added after §3)

The gate measures the **single-view** gap, but deployment reads $R=80$. Whether training can reach
deployment depends on **which** error component $L_\text{bias}$ removes, and the two cases are
distinguishable:

- if $L_\text{bias}$ removes the **per-repetition independent** noise, pooling already removes it, so
  $\mathrm{corr}(D_{R80}, D_g)$ stays flat and there is **no deployment gain** (a null result that is
  *not* a failure of the estimator, just of the lever's reach);
- if $L_\text{bias}$, averaged over repetitions of the **same concept**, removes the
  **concept-level (shared)** error, then it attacks exactly the $0.428$ that pooling cannot, and
  $\mathrm{corr}(D_{R80}, D_g)$ rises.

The gradient of $L_\text{bias}$ is computed against a **per-concept** target $D_g$ and averaged over
$r$, so it preferences the concept-level component by construction — but that is a prediction to be
**tested**, not assumed. Hence the pre-registered readout is a **pair**:

$$
\Delta_{R1} = \Delta\,\mathrm{corr}(D_{R1},D_g)\quad\text{and}\quad \Delta_{R80} = \Delta\,\mathrm{corr}(D_{R80},D_g),
\qquad \textbf{PASS} \iff \Delta_{R80} > 0 \ \text{with}\ \Delta_{R1} \ge \Delta_{R80} > 0 .
$$

**A run where only $\Delta_{R1}$ moves is booked as "lever did not reach the deployed estimator", not
as a success** — that distinction is the whole reason §3 was run first.

---

## 5. What is NOT in the architecture (and why)

| removed / excluded | reason (measured) |
|---|---|
| variance hinge | $\rho\le0$: no headroom (§3.2) |
| order-1 alignment pillars | cross-subject bias on the global-map axis is empty (M6: 2.4% explained) |
| v12 metric self-distillation | unanchored target $\Rightarrow$ rank collapse (2–3; −20.15pp) |
| exact whitening (GQF) | destroys amplitude of $PD(C)$; −26.15pp |
| landmark-rate / T2′ objectives | operator does not consume them ($\mathrm{corr}=-0.21$) |

---

## 6. The twin, run and early-stopped (job `650254`, 2026-10-06) — NEGATIVE

**What was run.** The §4 twin (`configs/v13_bias_loso_k20.yaml` vs `configs/v13_bias_off_loso_k20.yaml`,
single-variable: `bias_anchor.enabled` only — asserted by the sbatch gate *and* re-asserted by
`summarize_v13.py`), 10 folds × 1 seed. It was **cancelled after 2 of 10 folds** on the user's call,
because the two folds that did finish were uniformly and strongly negative. This is therefore an
**early-stop**, not a completed 10-fold verdict: every number below is n=2.

**1. The term fires — the failure is transfer, not implementation.**

| epoch | `bias_anchor` | `ba_agreement` | `ba_student_rank` |
|---|---|---|---|
| 0 | 0.992 | 0.008 | 5.88 |
| 9 | 0.629 | 0.371 | 3.89 |
| 19 | 0.364 | 0.636 | 3.99 |
| 49 | 0.445 | 0.555 | 4.90 |

The loss falls and source-subject agreement rises 0.008 → 0.65, so the objective is genuinely
optimised. The effective rank dips 5.9 → 3.9 over the first ten epochs and then **stabilises** at
4–4.9 — bounded, and *not* v12's collapse to 2.09. The guard mattered, and this time it held.

**2. The deployed headline cell got worse on 2/2 folds.**

| fold | cell `R=80 + fuse=16`, ON | same, OFF | $\Delta$ |
|---|---|---|---|
| sub-01 | 54.5 / 83.0 | 63.5 / — | **−9.0** |
| sub-02 | 49.0 / — | 62.0 / — | **−13.0** |

All six readouts (raw cosine → CSLS → the T2 rungs) are negative on both folds, and the **T2 rungs
are hit hardest** (−9 … −13, against −3 … −8 for raw cosine). The term constrains the *pooled*
second-order metric; the deployed readout is a structural fusion over the *repetition cloud*, so the
largest damage landing exactly on the cloud-dependent rungs is consistent with the term flattening
the very cross-repetition structure T2 consumes. This is a mechanism hypothesis from the gradient
pattern, not a measurement.

**3. The pre-registered criterion (§3.6 / §4.1) is met on 1/2 folds — and the whitened frame is the
honest one.** The criterion is stated on the deployed, whitened frame (the 0.616 / 0.796 reference
values in §3.5 are whitened):

| fold | $\Delta_{R1}$ | $\Delta_{R80}$ | verdict |
|---|---|---|---|
| sub-01 | −0.0012 | +0.0011 | **FAIL** ($\Delta_{R1}<\Delta_{R80}$ and $<0$) |
| sub-02 | +0.0140 | +0.0107 | PASS (marginal) |

**The raw (un-whitened) frame tells the opposite story** — $\Delta_{R80}=+0.025 \ldots +0.041$,
$\Delta_{R1}=+0.097\ldots+0.103$, in the predicted $\Delta_{R1}>\Delta_{R80}>0$ shape. The contrast
is the reading: the lever moves the **scale-sensitive raw geometry** and essentially nothing once
the deployed whitening removes global scale. A gain that survives only in the raw frame is a
scale/anisotropy change being reported as a structural one, which is exactly the failure mode the
whitened criterion exists to catch. Booking the raw-frame numbers as a success would have repeated
the M1 mistake in a new costume (§3.1).

**4. The eff-rank guard is mildly violated, 2/2.** ON is below OFF on all four measurements
(sub-01: 44.65/21.79 vs 45.74/22.58; sub-02: 47.58/22.38 vs 48.09/23.52), same sign as the
training-side rank dip. Small (−0.5 … −1.1) but consistent, and the §4 guard requires
$\ge$ baseline.

**Verdict.** The unbiased reading of n=2 is *falsified-in-progress*: the term optimises its own
objective on source subjects, does **not** lift the deployed whitened agreement beyond noise, and
costs 9–13pp of Top-1 on the cell that matters. Combined with the raw-vs-whitened split, the most
defensible conclusion is that $L_\text{bias}$ as specified moves a quantity the deployed operator
does not read. Two caveats are recorded rather than argued away: n=2, and $\lambda_b=2.0$ was never
swept. A single smaller-$\lambda_b$ fold would separate "wrong object" from "right object, too
strong" before any re-run is worth 1.7 h of GPU.

**Shipped configuration is unchanged.** Nothing outside `configs/v13_*` and
`src/samclip/losses/bias_anchor.py` was touched, and the OFF arm of this twin *is* the shipped
baseline, so the 55.10 / 54.83 result is untouched by this experiment.

---

## 7. v14 — the frame fix, run for real (jobs `650708` / `655174`, 2026-10-07) — NEGATIVE, by collapse

§6 ended on a diagnosis: $L_\text{bias}$ moved the raw-frame geometry and not the deployed whitened
geometry, so it was scored in a frame the operator does not read. §7 is the attempt to *take the
diagnosis seriously* — recompute the loss **inside the deployed (whitened) frame** — and its verdict
is negative. It is recorded in full because §6's "wrong object" reading is now settled: even with the
frame corrected, the object is wrong. **But the way it is wrong is new, and it is the useful part.**

### 7.1 What was run

`configs/v14_anchor_{off,raw,whit,whit05}.yaml`, 10 folds × 1 seed, in one chained sbatch
(`slurm/v14_anchor10x1.sbatch`). The four arms are a **single-variable ladder**, with the pairs
asserted by the sbatch preflight *and* re-asserted by `scripts/summarize_v14_mve.py`:

| pair | the one thing that differs | what it isolates |
|---|---|---|
| `raw` vs `whit` | `bias_anchor.frame` (`raw` → `whitened`) | the frame, at fixed weight 2.0 |
| `whit05` vs `whit` | `bias_anchor.weight` (0.5 vs 2.0) | the magnitude, at fixed frame |

(Verified against the configs: `raw` and `whit` differ in exactly one key, `frame`; `whit` and
`whit05` differ in exactly one key, `weight`. `whiten_shrink: 0.0` is equal in both so it never
appears in a diff.)

The same job also ran the **deployment-side** estimator O2-MVE (order-2 multi-view pooling over
`cont`/`stride`/`rand` repetition partitions), whose gate is reported in §7.4.

### 7.2 The first attempt died numerically, and the failure was *load-bearing* (job `650708`)

All **20/20** whitened-arm trainings (both `whit` and `whit05`, all 10 folds) died at
**epoch 0, step 102**:

```
FloatingPointError: non-finite gradient at epoch 0 step 102:
  loss=4.907 (finite), grad_norm=nan
```

The forward value was finite and only the backward exploded — the signature of `torch.linalg.eigh`'s
gradient, which carries $1/(\lambda_i-\lambda_j)$ terms. With `whiten_shrink=0.0` the smallest
eigenvalues sat on the absolute `eps` floor, so those terms were unbounded **by construction**.
Because every whitened training died, every `whit`/`whit05` number in job `650708` — including the
probe readings and the MVE rows — was computed on a **random-init checkpoint** and is void. Reading
those (~0.5% Top-1) as "the frame fix does not help" would have been wrong.

**The fix** (`_whiten_cloud` in `src/samclip/losses/bias_anchor.py`): (i) the whitening map is
computed under `torch.no_grad()` — the frame is a *constant*, and this is also the faithful choice,
since deployment fits its whitener on frozen features and no gradient passes through it; (ii) the
eigenvalue floor is **relative** (`1e-6 * \lambda_max`) instead of an absolute `eps`, which caps the
amplification at $10^3$ instead of $10^4$ and keeps the floor scale-covariant. Verified: the
per-subject-linear-map invariance is **unchanged** at $|\Delta| = 2.02\times10^{-6}$ (the value moved
by nothing; only the backward path changed), near-singular inputs now give finite gradients, and
`scripts/smoke_test.py` is green including §22.

> This is the second time in this project that a *correct* idea was reported as a failure because the
> run it was tested in had crashed. n=20 NaN deaths is a strong prior for "the harness is broken",
> not "the idea is broken"; the tell is `loss` finite while `grad_norm` is NaN.

### 7.3 The real verdict: the whitened-frame anchor **collapses the encoder to rank 1**

Job `655174` re-ran the grid post-fix. **No NaN**: all whitened trainings reached epoch 49 and
converged. They also produced **no working model at any point** — 6/6 completed folds, and the
`whit05` folds agree, i.e. 12/12:

| arm | `offset_ratio` | `spec_top_frac` | **`best_seen`** (best Top-1 over all 50 epochs) |
|---|---|---|---|
| `off` | 0.55 – 0.70 | 0.966 – 0.978 | **24.5 – 35.5** |
| `raw` | 0.67 – 0.80 | 0.971 – 0.984 | **17 – 23** |
| `whit` | 0.77 – **1.0000** | **0.9998 – 1.0000** | **1.0 – 2.5** |
| `whit05` | 0.62 – **0.9999** | **0.9998 – 1.0000** | **1.5 – 2.0** |

`best_seen` is the decisive column: it is the **best** checkpoint of the whole run, so this is not
late-stage drift — no usable model exists at any epoch. Top-1 of 1.0 is chance for 100-way.
The end-of-training line is equally blunt: `test top1 1.00 top5 2.50 meanrank 107.3 | offset 1.000
spec_top 1.000 | smn_gate 0.000`, against `off`'s `offset 0.546 spec_top 0.971`.

**Attribution is clean, and it is the frame — not the weight.** `raw` and `whit` differ *only* in
`frame`, at the same weight 2.0, and `raw` does not collapse (17–23). `whit` and `whit05` differ
*only* in `weight` (2.0 vs 0.5), and **both** collapse. So the collapse is caused by putting the loss
in the whitened frame, and is insensitive to how strongly it is applied.

**Mechanism.** The first half is a theorem, not a hypothesis: whitening subtracts each cloud's mean
and normalises its scale, so a loss computed on whitened coordinates has **exactly zero gradient**
along the per-subject mean and scale. The encoder found that blind spot and moved all of its energy
into it — `offset_ratio → 1.0000` means the per-subject offset *is* the representation, the residual
collapses to rank 1 (`spec_top_frac → 1.0000`), and the gate that would otherwise have contained it
shut (`smn_gate 0.000`). The offset is precisely the direction the deployed operator leans on most,
so the two other terms (`img`, `cross`) were left pinned at chance and retrieval went to 1%. Making
the loss invariant to per-subject linear maps therefore did not make it *structural*; it made it
**blind**, and blindness is exploitable. The finer gradient pathway that actively *rewards* the
collapse (rather than merely failing to punish it) is a hypothesis and is not claimed here.

**Guard gap, named.** `smoke_test.py` §22 pins the invariance and the `whiten_shrink=0` distinction
on a *healthy* cloud. It never tested the **rank-1 direction** the whitened loss can walk into, so
nothing in the suite could have caught this before 20 trainings were spent. That is the missing test.

### 7.4 The deployment-side ladder (O2-MVE): gates pass, result is null

| check | result |
|---|---|
| Kill switch: `cont-only` must reproduce shipped `fuse=16` | ✅ `55.05` vs `55.05`, `max|Δ| = 0.00e+00` |
| H2: `structural-off` twins must be bit-identical | ✅ `max|Δ| = 0.00e+00` |
| H1: multi-family beats its own `cont-only` control | ❌ `54.95` vs `55.05`, `Δ = −0.10 (t = −0.19, 4/10 folds)` |

The gates passing is what makes the null *readable*: the estimator is correctly wired and the
structural term cannot leak. Its mechanism readout explains the null rather than papering over it —
the 48 views received nearly flat leave-one-out weights (range `0.00173`), and the smoke test
measures `cont`-vs-`stride` view correlation at **0.9093**. `stride`/`rand` are therefore near-copies
of `cont`, not new views: the premise that interleaving would decorrelate slow drift is refuted by
measurement, $K_\text{eff}$ never rose, and pooling bought nothing. **Not bookable.**

### 7.5 Verdict and shipped state

- **`raw`** reproduces §6 at 10-fold scale. Clean 10-fold numbers from job `650708` (where `raw`
  trained fine, all 10 folds): base headline `fuse=16` and the MVE `cont-only` control both land at
  **43.95**, against `off`'s **55.05** — a stable **≈ −11 pp** from scoring the order-2 loss in a
  frame the operator does not read, matching §6's −9 … −13 pp on the same cells. (The partially
  re-run `raw` folds in `655174` land within ~1.6 pp of this; the sign and magnitude do not move.)
- **`whit` / `whit05`** are **falsified by collapse** (12/12 folds), not by "no lift". §6 could not
  separate "wrong object" from "right object, too strong"; §7 does: the whitened frame is not the
  right object either, because it is blind to the direction that matters.
- **O2-MVE** is falsified as a *test-time* lever (its own families are duplicates).
- **Shipped results stand unchanged: `54.83 / 83.45`.** Nothing outside `configs/v14_*`,
  `src/samclip/losses/bias_anchor.py`, `src/samclip/calibration.py`, `scripts/run_eval.py` and
  `slurm/v14_anchor10x1.sbatch` was touched.

**Caveats, recorded not argued away.** The `whit`/`whit05` verdict is n=6 folds (job `655174` was
cancelled once 12/12 folds had collapsed — the remaining folds could not change the sign of a
hard rank-1 degeneracy). The `off`/`raw` arms in `655174` were re-run from scratch rather than
skipped, because `should_skip()` in the sbatch treats `src/**` mtime as a dependency; that is why
the job was 3 h in at 25/40 units. `\lambda_b` was swept only as {0.5, 2.0}, but §7.3 shows the
collapse is frame-driven and weight-insensitive, so a finer sweep addresses the wrong variable.

**What would have to change for a third attempt to be worth GPU time.** Not the frame and not the
weight — an anchor whose invariance is *bounded* rather than total: the loss must be blind to
per-subject linear maps but **not** to the mean, i.e. constrain the whitened *shape* while the other
terms retain a restoring gradient on the offset. As written, the whitened frame removes the offset
from the loss's view entirely, and the encoder will always prefer a free direction to a constrained
one.
