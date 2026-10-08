# v10 — Single-model plan to clear SCORE, built on the cross-subject concept geometry

> Written 2026-10-05. Follows v9 (job 645154/645268, whose high-α training arm was
> falsified) and the α/τ screening of job 645408. Single model only — **no seed
> ensembling** is used or proposed anywhere in this document.

---

## 1. Where we actually stand (single model, like-for-like)

All numbers are 10-fold × 3-seed LOSO on THINGS-EEG2, 200-way, 63 ch, reported as the mean
over the 30 runs. "Single model" means each of the 30 runs is scored on its own; nothing is
pooled across seeds.

| # | Encoder | Deployed operator | Row | Top-1 | vs SCORE |
|---|---|---|---|---|---|
| 1 | G3 | mutual-NN Procrustes | `+T1+T2` | 45.33 | −7.90 |
| 2 | G3 | Sinkhorn (S3R) | `+T1+T2` | 48.13 | −5.10 |
| 3 | v8 | Sinkhorn (S3R) | `+T1+T2` | 50.62 | −2.61 |
| 4 | v8 | **FGW(α=.75,τ=.03)** | `+T1+T2` (sum fusion) | 52.72 | −0.51 |
| 5 | v8 | **FGW(α=.75,τ=.03)** | **`+T2 reps` alone** | **53.80** | **+0.57** |
| 6 | v8 | FGW(α=.75,τ=.03) + CSLS | `+T2 reps` | **54.37** | **+1.14** |
| — | **SCORE (2026)** | coordinate recovery | — | **53.23 ± 1.62** | — |

Two facts drive everything below.

**(a) The currently-deployed row is the wrong one, and that is a real bug, not a tuning
choice.** Row 4 fuses T1 (40.43) with T2 (53.80) using a normalised sum, and the fusion
*destroys* 1.18pp: `T2 alone vs sum = +1.18pp (sd 2.73, t=2.37, 18/10)`. T1 is 13pp weaker,
so averaging it in drags the ranking down. Row 5 simply reports the stronger row.

**(b) The gain is entirely in the rep-cloud row.** On the v8 encoder, FGW moves
`+T2 reps` 50.22 → 53.80 = **+3.58pp (27/1, t=7.23, all 10 subjects positive)**, while the
T1 rung is unchanged (−0.18). That localisation is the mechanism, not an artefact: the
structural term is defined on *cloud-to-cloud* geometry, and only the rep-cloud operator
gives it a cloud to act on.

**Honesty caveat.** Rows 5–6 involve three choices made on these same 10 folds (α=0.75,
τ=0.03, which row to report). α=0.75 is the only one that survives nested-LOSO
(10/10 subjects picked it independently); τ and the row choice are not yet nested-LOSO
validated. The paired evidence for the *row* choice (+1.18, t=2.37) is stronger than for
τ (+0.57, t=2.19, only 16/10). Treat row 6 as "at or slightly above SCORE", not as a
decisive win.

---

## 2. What the literature says, and how it maps onto us

**SCORE (arXiv 2608.19134) is solving our problem with our mechanism.** Its stated finding:
*"different subjects preserve similar relationships among concepts but express them along
different coordinate directions."* That is exactly the quantity this project measured
independently — cross-subject concept-geometry correlation **+0.565** over 45 pairs
(chance −0.003), with the residual being an orthogonal re-embedding (the 5-fold-CV
`O(d)` residual 0.885 vs unconstrained linear 0.922). SCORE recovers it with
hubness-corrected landmarks + an orthogonal map; **we recover it with a Fused
Gromov-Wasserstein coupling, which at α=0 *is* the landmark matching and at α>0 adds the
2nd-order structure SCORE does not use.** So FGW is a strict generalisation of SCORE's
operator, and the interior optimum we measured (α=0 → 37.0, α=0.25 → 38.8, α=1 → 1.9)
is the empirical evidence that the added term is worth fusing rather than replacing.

**GWOT is established for exactly this in neuroscience.** Unsupervised Gromov-Wasserstein
aligns point clouds from internal distances alone, with no stimulus labels (iScience 2025;
the GWOT toolbox, bioRxiv 2023), and GWOT plans are used to *guide* Procrustes alignment.
Fused (Unbalanced) GW aligns individual brains by combining functional similarity with a
geometric prior (NeurIPS 2022). Our EEG↔image setting is the cross-modal case of the same
object; the "unbalanced" variant is the natural handle if the two clouds have unequal mass.

**SATTC (CVPR 2026) is the closest deployed competitor and it tells us what we are missing.**
It is a *label-free test-time head on the frozen similarity matrix* that fuses a **geometric
expert** (subject-adaptive whitening + adaptive CSLS) with a **structural expert**
(mutual-NN + bidirectional top-k rank agreement + class popularity) via a
**Product-of-Experts**. Two things follow: (i) our S3R/FGW operator is the same family as
its geometric expert, but *stronger* because FGW uses the metric structure rather than
ranks; (ii) our fusion rule is the naive sum, which we have now measured to be the weakest
of the rules tested (52.62 vs 54.37).

**CORTIVA (arXiv 2608.01355)** argues for fusing at the *candidate-score* level rather than
consolidating embeddings, because early consolidation "imposes one similarity geometry on
every candidate order and removes encoder-specific disagreements from the final ranking".
Its reported 73.5% Top-1 is on a **different protocol and is not comparable to SCORE's
53.23** — it must not be quoted against us without checking the setup. The transferable
lesson is the *fusion level*, which is what §3-P0 does.

---

## 3. v10 — four components, each independently measurable

The design principle is that **every component is a separable claim with its own baseline**,
because this project's most expensive recurring failure has been a mechanism that was
accepted by the config and then silently did nothing.

### P0 — Report the rep-cloud row, and fix the fusion rule (`no training`)
*Change*: stop deploying the T1+T2 sum. Deploy the `+T2 reps` operator, with CSLS applied on
top; keep the fused and T1 rows in the report as diagnostics.
*Measured*: **54.37** (vs 52.62 deployed today). *Evidence*: row choice +1.18pp (t=2.37);
CSLS +0.57pp (t=2.19).
*Research*: the fusion-level argument of CORTIVA; CSLS is SATTC's geometric expert.
*Cost*: ~15 min of evals. **This alone already places us above SCORE single-model.**

### P1 — Estimate the target's concept geometry from the repetition cloud
*Hypothesis*: T2 beats T1 (53.80 vs 40.43) because the rep cloud has **R×80 the sample
count** for moment estimation, so its second-order statistics are far less noisy. The FGW
structural term currently consumes the geometry of the *averaged* `(C, d)` cloud, i.e. it
throws that sample count away exactly where it matters.
*Change*: build `D^e_ik = mean_r ||z_ir − z_kr||²` from the `(C, R, d)` cloud and pass *that*
into the FGW structural term; keep the averaged cloud for the anchored term.
*Expected*: +0.3…0.8pp. *Cost*: implementation + ~1h of evals on banked checkpoints.

### P2 — Cross-subject concept-geometry template (**the innovation centrepiece**)
*Hypothesis*: the concept geometry is shared across subjects (+0.565 measured). The source
subjects therefore give a **label-free, target-independent estimate of the structural
reference** — the target's own geometry is noisier and only one view.
*Change*: the FGW structural reference `D^i` (the "second domain" of the transport) becomes
a **source-averaged concept geometry** rather than the image gallery's alone; the target
subject's own geometry (`D^e`, from P1) is aligned *to it*. At deployment this uses no target
labels and no target-side fitting beyond the existing label-free refinement.
*Why this is our differentiator*: SCORE recovers a coordinate map from landmarks (1st order).
This term is a 2nd-order structure **aggregated across subjects**, which is precisely the
"cross-subject concept CLIP" thesis and cannot be obtained from SCORE's formulation.
*Expected*: +0.5…1.2pp. *Ablation is mandatory*: FGW with source-templated `D^i` vs
gallery-only `D^i`, same folds, same seeds — that single delta is the innovation's measured
advantage and is the number to quote.

### P3 — Low-α structural term in training (`soft_plan.alpha ≈ 0.1`)
*What was falsified*: v9 trained at α=0.75 (the deployment optimum) and the encoder came out
**8.1pp worse** than v8 on the shared folds. So the deployment α must not be copied into
training.
*What is untested*: a **small** α. The banked optimum is α=0 (v8); the question is whether a
gentle structural signal helps or is simply inert — not whether a strong one helps, which is
answered.
*Change*: `soft_plan.alpha ∈ {0, 0.05, 0.1, 0.2}`, everything else identical to v8.
*Gate*: adopt only if it beats α=0 by ≥0.5pp paired on **all 30 runs**, otherwise v8 stays
the encoder and v10 is a pure deployment contribution.
*Expected*: ±0.5pp. *Cost*: 4 configs × 1 fold smoke, then 30 runs for the winner.

### Not included, and why
- **Seed ensembling** — excluded by the user's requirement and by fairness; SCORE is a
  single model. (It is worth +4.28pp and is orthogonal.)
- **High-α training (v9's arm)** — falsified; see P3.
- **Subspace truncation of the recovery** (`rank>0`) — measured −0.50pp, adds nothing.

---

## 4. Decision gate and cost

```
P0  30 evals on banked v8 checkpoints                        ~30 min   -> expect 54.4
P1  P0 + rep-cloud geometry                                 ~1.5 h    -> gate: >= 54.4
P2  P1 + source-geometry template  (innovation ablation)    ~2 h      -> gate: >= 55.0
P3  4 x 1-fold smoke, then 30 runs for the winner           ~4 h      -> adopt iff >= +0.5
------------------------------------------------------------------------------
    total                                                   ~8 h
```

**Target**: single-model **≥ 55.5** Top-1, i.e. **+2.3pp over SCORE** — outside SCORE's
reported ±1.62. Below +1.5pp the claim should be "comparable/better within noise", not
"clearly beyond".

**Honest probability**: P0 is already measured and lands at +1.14. P1+P2 are plausible but
unmeasured; the mechanism (variance reduction in a quantity we have measured to be shared
across subjects) is sound, and the failure mode to watch for is that the source template is
*too* averaged to resolve per-concept structure on the target. P3 is a coin flip and is
gated, not assumed.

---

## 5. What must be true before any SOTA claim is written

1. α chosen by **nested-LOSO**, not on the reported folds; τ and row choice validated the
   same way. (α=0.75 already passes; τ=0.03 and the row choice do **not** yet.)
2. Single model only; any ensembling reported separately and never as the headline.
3. The P2 ablation (source template vs gallery-only) run on identical folds and seeds.
4. Numbers reported against the **same protocol** as SCORE (10-fold LOSO, 200-way, 63 ch,
   final epoch, label-free target adaptation) — no cross-protocol comparisons, including
   not quoting CORTIVA's 73.5.
5. Every mechanism claim carries its paired statistic and its per-subject sign count, so a
   silent no-op is visible.
