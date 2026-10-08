# COMMET — Concept-Metric Multi-view Ensemble Transduction

The complete architecture, its innovations, and the deficiency analysis it answers.
Companion to `docs/eeg2image_v12_core_claim.md` (the claim) and `docs/eeg2image_commet_plan.md`
(positioning + the E1/E2 results). Written 2026-10-05.

---

## 0. What is wrong with the current architecture (every item measured)

The shipped stack is: `SMN(embed(EEG))` → mean over R repetitions → cosine/CSLS → orthogonal
recovery → (optionally) the FGW structural operator. Six deficiencies, each with the measurement
that identifies it:

**D1 — The objective and the operator disagree about the object.**
Training aligns the **mean** (`InfoNCE(mean_eeg, image)`), but the deployed operator is refit on the
**per-trial cloud** (`rep_cloud_scores`). The encoder is never asked to give a single trial any
structure, and it does not: on a real encoder the single-repetition metric agrees with the 4-rep
consensus at only **+0.13 / −0.07** (v12 smoke, job 645753). The R-curve says this matters — Top-1
runs 4.0 → 54.5 from R=1 to R=80 (job 645719) — so the encoder's per-trial output is the thing
limiting us, and training does not touch it.

**D2 — The invariance objective targets the wrong order.**
`HSIC`/`MMD` subject-invariance act on **first-order features**, but first-order cross-subject
structure is measured to be nearly empty: the best global map explains **+2.4%** (M6) and maps do not
compose (**+5.7%** held-out gap, t=140, M7). The structure that *does* transfer is **order-2**:
corr(D_eeg,s, D_eeg,t) = **+0.565** (M1). We are regularising the object we proved is not there.

**D3 — The query is collapsed to a point before it is scored.**
Every SOTA method and our own point pipeline average the R repetitions into one embedding, then
score. E2 measured what that costs: point+recovery 39.95 vs the cloud 54.25 — **+14.30pp** on the
same encoder and folds (job 646572). The mean throws away the spread that measures query
reliability.

**D4 — Reliability is a heuristic, not an estimated quantity.**
The v11 fusion weights come from a fixed leave-one-out residual formula. Reliability estimation
elsewhere in the neural literature is either at the alignment layer only (Bratrix) or has been
measured **not to beat a tuned fixed weight** (a 2026 P300 study across 22 priors, 47 subjects). So
D4 must be treated as an open question, not an assumed win.

**D5 — The gallery is a retrieval target, never a view.**
The gallery is the *most informative* view of the shared structure (M2 = 0.593, higher than the
EEG–EEG 0.565) and is never used to denoise the query side. NEAR uses gallery *anchors* but for the
few-repetition, participant-specific regime, not cross-subject.

**D6 — Train/deploy repetition mismatch.**
The train cloud is **R=4** (`train_subNN_all63_trainreps_*.npy` is `(1654,10,4,63,250)`); the test
cloud is **R=80**. The encoder has never been trained under the input distribution the operator
consumes.

---

## 1. The main line (one sentence)

> **Cross-subject EEG→image retrieval is an order-2 ESTIMATION problem, not an order-1 alignment
> problem: the query is a distribution observed through R noisy draws, and the right object is the
> concept-level metric estimated by fusing the draws' independent metric views, then coupled to the
> gallery by a structured transport — with the encoder trained to support that estimator rather than
> to align a mean.**

### Why this line and not another

* It is the only reframe that explains **both** the failures and the successes with measured facts.
  Every falsified family (M1, C1, P1, P2, A4, v12) tried to improve or invert an **order-1 map**; every
  measured success (T2 +25pp, CSLS +5.5pp, recovery +25pp, v11 fusion +1.03pp) **pools or regularises
  an order-2 estimate**.
* It survives the field test: no 2026 EEG-to-image paper uses Gromov–Wasserstein, and the strongest
  cross-subject methods (SCORE, SATTC) are explicitly order-1 (coordinate recovery / similarity
  calibration on a mean embedding). The order-2 object is **unoccupied**, and the measurement says
  that is where the signal is.

---

## 2. The architecture

Three stages. **Stage A is deliberately unchanged** — six training-side attempts to improve the
encoder are falsified (T2′, T2″, SCORE-episode, G-a, and v12's two metric terms), so the burden of
proof for changing it is not met. The novelty is in Stages B and C, which are isolated from the
deployment path and therefore cannot collapse it.

### Stage A — Encoder (frozen, proven)

`z = SMN(embed_θ(EEG))`, trained with the banked recipe (InfoNCE + soft-plan, `csls_k=20`). Produces
both the mean embedding and the R=80 per-trial cloud. **Unchanged.** Its 3-seed 10-fold Top-1 is
53.80 and its cloud+fusion cell is 54.83.

### Stage B — Metric Estimator (the core novelty; trained head + measured fusion)

Deployment object: the query's concept metric `D_q ∈ R^{C×C}` over the C=200 test concepts.

1. **Per-trial views.** The R=80 cloud gives R embeddings of each concept. Split into B blocks; each
   block's mean gives an independent metric estimate `D_b = standardised_sq_chordal(block_b)`.
2. **Measured reliability fusion.** `D_q = Σ_b w_b D_b`, with `w_b = inv_resid_b / Σ inv_resid`,
   where `inv_resid_b` is the leave-one-out residual of block `b` against the other blocks. This is
   the v11 estimator, measured to carry shared structure monotonically in the number of views with a
   correspondence-destroying control pinned at ~0.000 (job 645697).
3. **Metric adapter (the trained, isolated component).** A projection `φ : R^d → R^{d'}` trained on
   **source subjects only** to maximise the agreement of the source per-trial metric with the source
   consensus, with a **hard rank floor** guard. It is applied to the target cloud; the *trunk is
   frozen* (or stop-grad), so this term cannot alter the deployed embedding — the v12 collapse
   (SMN gate → 1.0, raw cosine −13pp, 0/4 folds) is structurally impossible here.
   The objective, on source subject s with stimulus ids and repetitions:
   `L_φ = 1 − corr( metric(φ(z_single)), stopgrad( Σ_b w_b metric(φ(z_block_b)) ) )`.
4. **Gallery view.** The gallery's own concept metric `D_g` (200×200) is a second, independent view
   (M2 = 0.593). Fused into the structural target so both sides of the coupling are denoised (D5).

### Stage C — Coupling (deployed operator)

**Fused Gromov–Wasserstein** between the query metric `D_q` and the gallery metric `D_g` with the
cross-modal cost `C` from φ-projected embeddings:

`T* = argmin_T ⟨C, T⟩ + α·⟨L(D_q, D_g), T⊗T⟩`,  scores read from `T*`.

FGW differs from Sinkhorn-on-CSLS in that it uses the **structure of both sides**, which is exactly
the order-2 object the claim is about. It is the deployed operator already (α=0.75, τ=0.03) and is
where the +25pp recovery rung lives.

### Optional Stage D — Score calibration

SATTC-style structural calibration on the fused scores (implemented, `calibration.structural_scores`,
`lam=0.2`). Measured on our encoder to add only +0.45pp, so it is a plug-in, not a load-bearing part.

---

## 3. Innovations, each with its evidence and its novelty

| # | Innovation | Evidence we own | Novelty vs 2026 SOTA |
|---|---|---|---|
| **I1** | **Order-2 reframe**: retrieval as metric estimation from a view ensemble | M1/M6/M7 (alignment empty) + D1/D3 measured | SCORE and SATTC are order-1 (coordinate/similarity on a mean embedding); **no 2026 paper states or tests the order-2 framing** |
| **I2** | **Reliability-weighted per-trial metric fusion** | v11: +1.03pp Top-1 (t=3.49, 20/30), permutation-controlled, structural-off bit-flat | Reliability weighting appears in EEG decoding at the *trial* level and at the *alignment* level (Bratrix); **never as an order-2 metric fusion for retrieval** |
| **I3** | **Fused Gromov–Wasserstein coupling for retrieval** | deployed, α=0.75; no prior use in EEG retrieval (search-verified) | GW/FGW is used in vision (ProtoOT) and shape matching, **not EEG-to-image** |
| **I4** | **Bilateral structure**: the gallery's metric as a denoising view | M2 = 0.593 (the richest view), currently unused | NEAR uses gallery *anchors* but in the few-rep, participant-specific regime (not cross-subject); no method fuses the gallery **metric** |
| **I5** | **Metric adapter on a frozen trunk** (safe, isolated training) | v12's collapse defines the constraint; the adapter cannot touch deployment | First order-2 training objective for this task; SCORE's recovery-aware training is order-1 |
| **I6** | **Diagnostic: "inter-subject alignment is empty"** | M6 +2.4%, M7 no-group (t=140), M1 +15.4→−14.5 under permutation | A measurement contribution that closes a whole family of methods |

---

## 4. Positioning (cross-subject, 200-way, THINGS-EEG2)

| Method | Top-1 | Top-5 | Object | Cross-subject? |
|---|---|---|---|---|
| leaderboard inter-subject avg | 28.5 | 62.0 | point | yes |
| SAMGA (our re-measure) | 26.22 | 57.98 | point | yes |
| SVTL (2609.36971) | 48.1 | 77.1 | point + transductive | yes |
| **SCORE (2608.19134)** | **53.23** | **83.55** | order-1 map | yes |
| SATTC (CVPR'26) | — (calibration) | — | similarity matrix | yes |
| NEAR (2608.19128) | few-rep gains | — | point anchors | **no** (participant-specific) |
| CORTIVA (2608.01355) | 73.5 | 95.3 | candidate-score fusion | protocol to verify |
| **ours (3-seed, `fuse=16`)** | **54.83** | **83.45** | **order-2 metric + FGW** | **yes** |

> **Caveat that must not be dropped.** CORTIVA's 73.5 Top-1 is far above SCORE's 53.23; its protocol
> (candidate-set size, adaptation) needs to be verified before we quote any comparison against it.
> Our headline comparison is against SCORE: **+1.60pp Top-1**, cross-paper (not paired), with Top-5
> tied. E2's +14.70pp is an **internal** cloud-vs-point ablation, not a comparison to SCORE.

---

## 5. Accuracy plan and its honest ceiling

Current: 54.83 (3-seed). The two unmeasured components:

* **I4 (bilateral gallery view)** — never tested. If M2's 0.593 is exploitable, this is the largest
  single opportunity.
* **I5 (metric adapter)** — never trained. Its ceiling is bounded: it can sharpen the metric but
  cannot create per-trial information the trunk does not produce.

Measured bound: the R-curve is still climbing +3.33pp/octave at R=80, so the estimator is not
saturated — but the train cloud is only R=4, so an adapter trained on R=4 cannot demonstrate the
R=80 regime. Target: **+1 to +2pp over 54.83** (i.e. 56–57), which would be a clear SOTA; anything
above must be treated as unexplained until a control reproduces it.

---

## 6. Gates — nothing is submitted to a full grid before the cheap gate passes

Because six training-side attempts have already failed, and one of them (v12) looked healthy for
half its epochs before collapsing:

| Gate | Question | Cost | Kill criterion |
|---|---|---|---|
| **G0** | Does the metric adapter improve a *frozen* encoder's per-trial metric on **source** subjects? | probe, no training | no improvement over the identity projection |
| **G1** | Does the bilateral (gallery-view) fusion beat `fuse=16` without the gallery view, paired? | 10 folds, no training | ≤0pp or Top-5 degrades |
| **G2** | Does the adapter + fusion beat the 3-seed 54.83 cell, paired, on a majority of folds? | 10 folds x 1 seed | not significant |
| **G3** | Only if G0–G2 pass: full 10-fold x 3-seed. | full grid | — |

### The guardrails every cell must carry

* **Permutation control** (the M1 channel): a fusion gain that vanishes under a concept-axis shuffle
  is leakage, and M1's +15.4pp → −14.5pp is the proof it can happen.
* **Structural-off twin**: must stay flat.
* **Hard rank floor** on any trained metric head (v12's rank 4.83 looked healthy while the deployed
  space had already collapsed — a rank diagnostic in the *wrong space* is not a guard).
* **No metric term on the deployment pathway.** v12 is the evidence: it drove `smn_gate` to 1.0 and
  raw cosine down 13pp, and lost 0/4 folds.

---

## 7. Gate verdicts (measured 2026-10-05, job 647161 + job 647339)

Both cheap gates were run to completion over the full **10 folds** (paired, one seed = 10 paired
observations, between-seed variance removed by design). **Both failed**, and the failures are the
clean kind — the controls separate from the mechanism, so the null is not an artefact.

### G0 — the metric adapter: FAILED (`scripts/probe_g0_adapter.py`, job 647161)

The adapter is the BEST-CASE linear map fitted on SOURCE subjects' un-averaged repetitions: the
whitening by the *within-stimulus* covariance `Sw`, which suppresses exactly the repetition noise
the per-trial metric is limited by. Fitted in closed form, so no learned `phi` can do better.

| arm | Top-1 | Δ vs identity | folds + | t |
|---|---|---|---|---|
| identity (the deployed row) | 55.10 | — | — | — |
| **within** (the adapter) | 53.90 | **−1.20pp** | 2/10 | −1.98 |
| total (unsupervised control) | 54.20 | −0.90pp | 2/10 | −1.69 |
| shuffled (label-destroying control) | 54.20 | −0.90pp | 2/10 | −1.57 |
| lda_k (rank-16 projection) | 38.55 | −16.55pp | 0/10 | −15.68 |

Reading. The adapter does not beat identity, and it does not separate from either control — so the
grouping information it captures (it is the only arm whose `metric_agreement` beats the shuffled
control, −0.039 vs −0.118) **does not convert to retrieval**. The label-free operator already
inside `rep_cloud_scores` (SAW whitening + moment matching) has taken the second-order denoising
gain; there is no room left for a source-fitted adapter. **I5 is closed for the cost of one job.**

### G1 — the bilateral gallery view: FAILED (job 647161)

`--fgw-gallery-fuse` pools the gallery's own K layer-views into a reliability-weighted metric and
injects it as the FGW gallery-side reference (`_fgw_plan(di_ref=...)`), the gallery twin of v11's
query-block fusion.

| cell | Top-1 | Δ vs its twin | folds + | t |
|---|---|---|---|---|
| `fuse=16` (baseline) | 55.10 | — | — | — |
| `fuse=16,gv` (gallery view) | 55.00 | **−0.10pp** | 4/10 | −0.24 |
| structural-off twin | 50.40 | **0.00pp (bit-flat)** | — | — |

Reading. The gallery metric is already clean (it is the metric of a **frozen** image encoder over
one fused embedding per concept), so pooling its layer views adds nothing — unlike the QUERY side,
which is limited by neural noise and where the same pooling buys +1.03pp (E1). The bit-flat
structural-off twin confirms the plumbing: the injected reference can only route through the
structural term, so this is a real negative and not a dropped argument. **I4's gallery half is
closed.**

### G3 — the adopted arm: MULTI-SUBJECT CONCEPT-FRAME TRANSDUCTION

The two COMMET components the gates closed (adapter, gallery view) are dropped. In their place the
**M1 mechanism is re-adopted as the method** (`src_mix=1` in `rep_cloud_scores`): pool the target's
own repetition-cloud metric with the 9 SOURCE subjects' metrics — all in the shared 200-concept
frame — and hand the pooled metric to the FGW structural term.

**Full grid, 10 folds × 3 seeds = 30 paired cells** (`slurm/v10_m1.sbatch` on the v8 checkpoints;
`slurm/commet_g3_frame.sbatch` reproduces it on the canonical g3 checkpoints):

| cell | Top-1 | Top-5 |
|---|---|---|
| `m=0` (deployed baseline) | 53.80 ± 9.78 | 82.55 |
| **`m=1` (concept-frame transduction)** | **69.17 ± 6.19** | **91.88** |
| paired Δ | **+15.37pp, t=16.97, 30/30** (min +6.0, max +25.5) | +9.33pp, t=10.86, 30/30 |
| `m=1` structural-off twin | 50.72 | 82.28 |
| `m=0` structural-off twin | 50.72 | 82.28 |

**Independent reproduction on the canonical g3 checkpoints** (`slurm/commet_g3_frame.sbatch`, job
647339, same recipe, different training runs):

| cell | Top-1 | Top-5 |
|---|---|---|
| `m=0` | 51.18 ± 9.31 | 81.43 |
| **`m=1`** | **65.22 ± 6.10** | **90.62** |
| paired Δ | **+14.03pp, t=17.97, 30/30** (min +7.0, max +24.0) | +9.18pp, t=11.91, 30/30 |
| structural-off twins | 48.20 (bit-flat) | 80.53 |

What is **established**: the effect is 30/30 folds in EACH of two independent checkpoint grids
(v8: +15.37pp, min +6.0; g3: +14.03pp, min +7.0 — **60/60 cells positive**), it is entirely routed
through the structural term (the structural-off twins are **bit-identical** in both grids), and it
reproduces across two independent checkpoint sets of the same recipe. The ~1.3pp grid-to-grid gap is
run-to-run variation of the same recipe, not a configuration difference (`--rep-shrink 0.1` is the
default and was identical in both).

What is **NOT resolved, and is carried as a caveat on every artefact**: the concept-axis permutation
control (`docs/eeg2image_v10_m1_theory.md` §9.1) flips this from +15.4pp to −14.5pp. That control
destroys TWO correspondences at once — source↔gallery (the answer key) and source↔target (the
legitimate shared metric, measured `corr = +0.565`) — so it does not by itself separate them.
Separating them needs a source encoder trained under a **different gallery ordering**, which has not
been run.

What is **cross-protocol**: this uses the SOURCE subjects' unlabelled test EEG, which the published
cross-subject cells (SCORE 53.23/83.55, SATTC) do not — they take only the held-out subject's test
EEG. The +15.94pp over SCORE is therefore **not** a like-for-like claim.

---

## 8. Adjudication of the concept-frame gain (job 647505, 2026-10-05)

§7 carried the concept-frame transduction (+13.20pp on the g3 grid) with an unresolved caveat: a
concept-axis permutation flips it to −14.5pp, and that control destroys TWO correspondences at once,
so it cannot separate "shared neural geometry" from "the source EEG acting as an answer key". This
section runs the separating experiment.

### 8.1 The four arms (one argument apart, `--fgw-src-ref-mode`)

Every arm replaces the same slot (`de`, the query-side metric, at `src_mix=1`); only the reference
changes. `gallery` uses the metric of the **public image gallery**, which carries **zero EEG** and is
available at test. 10 folds x 1 seed, all from the banked g3 checkpoints and templates.

| arm | reference | Top-1 | Δ vs baseline | folds + | Top-5 | `plan_acc` |
|---|---|---|---|---|---|---|
| base (`m=0`) | the target's own EEG metric `de_t` | 51.50 | — | — | 80.80 | 0.450 |
| `eeg` | the 9 SOURCE subjects' EEG metric | 64.70 | +13.20 | 10/10 | 90.55 | 0.796 |
| **`gallery`** | the **PUBLIC** image-gallery metric `di` | **70.55** | **+19.05** | 10/10 | 93.25 | **0.998** |
| `self` | `de_t` (a no-op) | 51.50 | +0.00 | — | 80.80 | 0.450 |
| `rand` | fixed-seed random metric | 41.90 | −9.60 | 0/10 | 75.25 | 0.184 |

`self` is bit-identical to the baseline (plumbing verified), and every arm's gain is **entirely**
through the structural term (all structural-off twins are bit-flat, Δ = 0.00).

### 8.2 The verdict: no neural leak, but an evaluation artefact

**The +13.20pp does not need any source EEG.** The public image-gallery metric reproduces it and is
*better* by 5.85pp. So the arm is not a source-EEG leak — but the reason it works is now exposed:

`de = di` makes the FGW structural cost `sum (de_ik - di_jl)^2` vanish at the **identity** coupling.
The evaluation is index-aligned (per-concept query row `i` and gallery row `i` are the same concept;
retrieval scores the diagonal), so the identity coupling **is the ground-truth correspondence**.
`plan_acc` — the plan's mass on the true diagonal — measures exactly how much of the answer the
coupling has recovered, and it tracks Top-1 monotonically across arms: 0.450 -> 0.796 -> **0.998**.

So the "concept-frame" family is a **protocol exploit**: supplying the structural term with a
gallery-proxy metric drives the coupling onto the answer, and the best proxy is the gallery metric
itself. The source EEG is a noisy (0.796) route to the same place. `rand` confirms the axis: there,
the same term drives the plan *away* from the diagonal (0.184) and costs −9.60pp.

### 8.3 Consequences

* **The `m=1` concept-frame cell is retracted from any SOTA claim.** Not because it is a neural
  leak, but because it is an index-aligned-evaluation artefact: it recovers the known identity
  correspondence, which a genuine cross-subject decoder is supposed to have to *learn*.
* **Our honest, defensible number is unchanged: 54.83 / 83.45** (cloud + `fuse=16`, §6). The E2
  "cloud vs point" gain is a different mechanism (the target's OWN repetition cloud, no reference
  metric; `plan_acc` stays at the base level) and is unaffected by this finding.
* **This is itself a reportable methodological result**: the THINGS-EEG2 cross-subject protocol
  admits a public-gallery-metric exploit worth +19.05pp, which invalidates the *class* of transductive
  claims that substitute a reference metric into the structural term. `plan_acc` is the one-number
  diagnostic that exposes it, and it needs no labels.
