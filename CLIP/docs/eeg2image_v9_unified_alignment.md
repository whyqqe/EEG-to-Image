# v9 — A unified theory of alignment for cross-subject concept CLIP

> Status: **theory + four measurements on frozen banked checkpoints.** No training was run for
> this document. Every number in §2-§4 is reproducible from `outputs/probe/rigidity_cv.json`
> and `outputs/probe/fgw_r*.json`; §5 states exactly what is falsified and what is not.
> Last updated: 2026-10-05.

---

## 1. The problem, stated in manifold terms

Let `M` be the **concept manifold**: the intrinsic space of the 200 THINGS concepts. It has
intrinsic dimension `d_M` — measured at **8.3 (participation ratio)** and 93-96% of variance in
**16** principal directions, so `d_M` is somewhere in **8-16**.

Each subject observes `M` through its own embedding

    z_e = φ_s(c) + ε,      φ_s : M -> R^64,    c ∈ M,

and the image side through a fixed `ψ : M -> R^64`. `φ_s` is *subject-specific* for a physical
reason: electrode geometry, skull conductivity and cortical folding differ per subject, so the
map from a concept to its EEG signature is not the same map for two people.

**The alignment problem is to recover the correspondence without knowing `φ_s`.** Everything the
project has built is an attempt at this, and the failures are all of one kind: each mechanism
constrains the correspondence using a *different, and insufficient, amount of structure*.

## 2. What the current mechanisms actually assume, and what survives measurement

| mechanism | assumption about `φ_s` | measured |
|---|---|---|
| InfoNCE / CSLS | `φ_s ≈ ψ` in a shared frame, up to small perturbation | it is imperfect: alignment residual leaves 58-78% of `z_i` unexplained in L2 |
| `coordinate_recovery` (Procrustes) | `φ_t = R ∘ φ_s` with `R ∈ O(d)` — a **rigid** motion | **wrong as a model class, but the right regulariser** (§3.1) |
| `L_mmd` | only the *marginals* agree | 0th-order: no correspondence at all |
| T2 repetition cloud | the cloud's covariance captures the subject's frame | 0th-order + a centroid estimate |
| **GW structural term** | only that the intrinsic metric `g` is preserved | **holds: `+0.565` cross-subject** (§3.3) — and unused today |

### 3.1 The rigid model class is NOT the bottleneck (my first hypothesis, falsified)

The obvious theory was "the map is non-rigid, so an orthogonal fit has irreducible bias". A first
probe appeared to confirm it spectacularly — orthogonal residual 0.78 vs a "smooth" fit at 0.009,
a 90x rigidity gap. **That number was overfitting and must not be quoted**: the smooth arm used
`(64 + 1024) × 64 = 69632` free parameters against `200 × 64 = 12800` constraints. A near-zero
residual from an underdetermined fit is not evidence.

Redone with capacity capped below the data and **5-fold cross-validation over concepts**
(`analyze_rigidity.py` §A):

| model class | params | 5-fold CV residual |
|---|---|---|
| orthogonal `O(d)` | 2016 | **0.885** |
| unconstrained linear `GL(d)` | 4096 | **0.922** ← *worse* |
| smooth (32 random Fourier features) | 6144 | 0.777 |

Two conclusions, both against the hypothesis:

* **Relaxing orthogonality makes generalisation worse** (-4.2%). At `C = 200` the rigid class is
  a *better* regulariser than the wider class — a textbook bias-variance result, and a reason to
  **keep** the orthogonal parameterisation rather than replace it.
* The most a richer map class recovers is **~12%** (0.885 → 0.777). The model class is not where
  the loss is.

### 3.2 But alignment quality predicts the score, strongly

| correlation over 10 subjects | r |
|---|---|
| subject accuracy vs `cv_orth` residual | **-0.742** |
| subject accuracy vs `cv_linear` residual | **-0.769** |
| subject accuracy vs the orth/linear gap | -0.203 |

`sub04` — the weakest subject at **32.7%** — has the *worst* orthogonal residual (0.919);
`sub01` at 59.5% has the best (0.840). So the 31pp subject spread tracks **how well that
subject's cloud can be aligned**, not how expressive the aligner is. **Alignment is the lever.**

### 3.3 The concept metric is shared — the signal the current coupling throws away

If the intrinsic geometry were subject-specific, no correspondence-free method could work and the
proposal below would be dead on arrival. Measured on the 200 concept centroids:

| quantity | value | chance |
|---|---|---|
| `corr(D_eeg_s, D_image)` — EEG geometry vs concept geometry | **+0.615** (range .574-.637) | — |
| `corr(D_eeg_s, D_eeg_t)` — cross-subject concept metric, 45 pairs | **+0.565** (range .460-.668) | **-0.003** |

The concept metric is substantially invariant across subjects. **This is a direct consequence of
the project's core innovation**: CLIP-style cross-subject concept supervision is what forces all
ten subjects onto a shared `M`. The geometry is the *footprint* of that innovation — and **no
current mechanism uses it.** Every existing operator (InfoNCE, CSLS, Procrustes, MMD, the T2
cloud) is built from *cross-domain similarities* or *marginals*. Nothing uses the intra-domain
metric.

## 4. The unified formulation: alignment is a fused optimal transport problem

A correspondence is a coupling `π ∈ R^{C×C}`. Three sources of information constrain it, and they
are exactly the three orders of transport:

    FGW(π) = (1-α) ⟨π, C⟩  +  α Σ_{ijkl} π_ij π_kl (D^e_ik − D^i_jl)²
             └── 1st order ──┘   └──────────── 2nd order ────────────┘

* `⟨π, C⟩` — the **anchored** term: `C_ij = -CSLS(z_e_i, z_i_j)`. Uses cross-modal supervision.
  This is InfoNCE/CSLS's object; `α = 0` is the operator shipped in v8.
* the **structural** term — GW: the coupling must preserve pairwise distances within each domain.
* marginal matching (`L_mmd`, plain OT) is the **0th-order** relaxation, `C_ij = 0`.

**Why the optimum must be interior — a rank argument.** The anchored cost is `C ∝ Z_e Z_i^T`,
which has **rank ≤ 64** (in practice `≤ d_M ≈ 8-16`). A rank-`r` cost matrix cannot distinguish
between two correspondences that differ only in the orthogonal complement of that `r`-dimensional
subspace of matrix space. The structural term lives in the `C(C−1)/2 = 19900`-dimensional space of
pairwise comparisons, which no rank-64 object can span. **The anchored term is structurally
incapable of expressing the information the structural term carries** — so they are
complementary, and neither extreme is optimal. Measured, on the frozen v8-era features:

| α | 0.0 (current) | 0.10 | **0.25** | 0.50 | 0.75 | 1.0 (pure GW) |
|---|---|---|---|---|---|---|
| mean top-1 (5 subjects) | 37.00 | 37.30 | **38.80 (+1.80)** | 37.90 | 36.30 | **1.90 (-35.10)** |

The profile is exactly the predicted one: **interior peak, catastrophic collapse at α = 1**. The
collapse is informative rather than embarrassing — it says the concept metric alone is *not*
sufficient (`+0.565`, moderate, not high), which is why the answer is *fused*, not *structural*.

Per subject, the gain at the best α:

| subject | acc (v8+S3R) | best α | Δ top-1 |
|---|---|---|---|
| sub10 | 63.7 | 0.25 | **+3.50** |
| sub08 | 43.7 | 0.50 | **+2.50** |
| sub07 | 42.2 | 0.25 | **+2.00** |
| sub01 | 59.5 | 0.25 | **+2.00** |
| sub04 | 32.7 | 0.10 | +0.50 |

### 4.1 The root cause, unified: the correspondence is CONSTRAINT-STARVED

This is the single sentence that ties the whole project together:

> **Every intervention that has ever worked on this task added constraints to an
> under-determined correspondence problem; every intervention that failed tried to add them
> through the encoder instead of the estimator.**

* `coordinate_recovery` fits `d(d−1)/2 = 2016` rotation parameters from **~42** mutual-NN
  landmarks. Under-determined.
* **S3R** (v8, shipped, **+2.80pp**, t = 8.77) replaced the hard 0/1 assignment with a
  doubly-stochastic plan: effective landmarks **42 → 465**. Same model class, same encoder, more
  constraints.
* **FGW** adds `C(C−1)/2 ≈ 19900` pairwise geometric constraints on top of S3R's `C` row-wise
  ones — two orders of magnitude more (§4).
* The four falsified pillars (T2', T2'', SCORE's source-only episode, G-a's low-rank frame) all
  tried to raise the **hard landmark rate via training**. The operator's contribution measured
  **flat** (+3.62 ± 1.72, uncorrelated with encoder quality *and* with landmark rate over 30 runs)
  because the deficit is in the *estimator*, not in the features being supplied to it.

### 4.2 A manifold-aware refinement that did NOT work (recorded because it was predicted to)

Since `d_M ≈ 8-16` inside `R^64`, the Euclidean distance used in the structural term is
dominated by ~48-56 noise directions, so truncating the distance estimate to the top-k principal
directions should de-noise it. Tested at `k = 8 / 16 / 64`:

| geo-rank | best Δ | at α |
|---|---|---|
| 64 (full) | **+1.80** | 0.25 |
| 16 | +1.80 | 0.25 |
| 8 | +1.20 | 0.10-0.25 |

**No improvement.** Full-space distances are already adequate; `k = 8` is actively worse (it
truncates past the signal). The manifold-dimension de-noising argument is intuitive and wrong
here, and this is worth stating plainly rather than quietly dropping. Truncation is kept as an
option (`--geo-rank`) and defaults to 0.

## 5. What this implies for the path to SOTA

The residual gap is **2.61pp Top-1 / 2.25pp Top-5** on the fused ladder, plus the open question in
`eeg2image_v8_architecture.md` §5.4 about whether the transduction is matched. Ranked by
expected value, grounded in the measurements above:

1. **Deploy the FGW coupling (`α ≈ 0.25`).** Measured **+1.80pp** on top of the shipped S3R
   operator, on frozen checkpoints, with no training — the same playbook that produced v8's
   +2.80pp. Expected ~**52.4**, i.e. ~0.8pp from SCORE. **This is the single highest-confidence
   next step and it costs one job.**
2. **Carry the fusion into training (v9).** Add the structural term to the `soft_plan` loss, so
   the coupling the encoder shapes is the same `α`-fused object deployment solves. §4.1 predicts
   this compounds with the operator, as S3R and `soft_plan` already measured super-additively
   (+5.28 measured vs +5.00 predicted-additive).
3. **Per-subject / learned `α`.** The best α ranged 0.10-0.50 across five subjects, and the
   ordering is *not* simply "weak subjects want more structure" (sub04 at 32.7% wanted the least,
   sub08 at 43.7% the most). Estimating α label-free (e.g. from the agreement between the two
   costs) is a small, self-contained win.
4. **Higher-order structure.** The structural term is 2nd order. A **3rd-order (triplet/triangle)
   consistency** term or a **diffusion/geodesic** rather than Euclidean metric are the natural
   extensions — but §4.2's negative result says to test them before believing them.

### 5.1 Honest risks

* The `+1.80pp` is measured on **5 subjects** (one seed). The v8 result taught that fold-level
  variance on this task is ±1.72-4.18pp, so this must be run on all 30 runs before being
  believed. The **α-profile shape** (interior peak + α=1 collapse) is the more robust evidence
  than the peak height.
* `α` is a real hyperparameter with a peak, not a plateau in the sense that S3R's `tau` was
  (which replicated to 0.02pp across two settings). Guard against picking α on the test folds.
* The rank argument in §4 is a statement about expressiveness, not about generalisation: a
  rank-64 cost *could* still generalise better if the structural term were misestimated. The
  `α = 1` collapse says the structural estimate is noisy, so the fused optimum is where the two
  error sources balance — not where either is best.

## 6. Artefacts

| what | where |
|---|---|
| rigidity probe (first pass, superseded) | `scripts/probe_rigidity.py` → `outputs/probe/rigidity.json` |
| capacity-controlled CV analysis + geometry tests | `scripts/analyze_rigidity.py` → `outputs/probe/rigidity_cv.json` |
| FGW coupling probe, swept in α and geo-rank | `scripts/probe_fgw.py` → `outputs/probe/fgw_r{0,8,16}.json` |
| frozen features reused by all of the above | `outputs/probe/rigid_feats_sub*_seed2025.npz` |
| the shipped operator this builds on | `calibration.subspace_soft_recovery` (v8) |
