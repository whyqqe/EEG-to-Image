# COMMET — the cross-subject SOTA plan (positioning + design), 2026-10-05

Companion to `docs/eeg2image_v12_core_claim.md` (the claim). This file records the field
positioning measured today and the architecture it implies. Everything below cites a number that
exists on disk or a paper that was read, not a preference.

---

## 1. Where we stand (measured)

10-fold LOSO, 200-way, THINGS-EEG2, deployed FGW operator (`tau=0.03, alpha=0.75`,
`rep-shrink 0.1`). **The 3-seed columns are the E1 result (job 646465, 10 folds x 3 seeds = 30
paired samples)** — this is the number the SOTA claim rests on, because the single-seed +0.85pp was
t=1.72 and did not clear the bar.

| Cell | Top-1 (1 seed) | Top-1 (3 seeds) | Top-5 (3 seeds) |
|---|---|---|---|
| ours, shipped single-mean metric (`fuse=0`) | 54.25 | 53.80 | 82.55 |
| ours, `fuse=16` structural fusion | **55.10** | **54.83** | **83.45** |
| ours, `fuse=16 (structural-off)` | 50.40 | 50.72 | 82.28 |

**The fusion's own contribution, paired (n=30):**

| metric | delta | t | folds positive |
|---|---|---|---|
| Top-1 (`fuse=16` − `fuse=0`) | **+1.03pp** | **3.49** | 20/30 |
| Top-5 (`fuse=16` − `fuse=0`) | **+0.90pp** | **3.75** | 19/30 |
| structural-off family | **0.00pp** | — | 0/30 (bit-flat) |

Published / re-measured cross-subject cells on the same benchmark (from search, 2026-10-05):

| Method | Top-1 | Top-5 | Mechanism |
|---|---|---|---|
| leaderboard inter-subject average | 28.5 | 62.0 | — |
| SAMGA (our re-measure) | 26.22 | 57.98 | contrastive alignment |
| SVTL (arXiv 2609.36971) | 48.1 | 77.1 | transductive target refinement |
| **SCORE (arXiv 2608.19134)** | **53.23** | **83.55** | label-free coordinate recovery |
| SATTC (CVPR'26, 2603.20738) | — | — | similarity-matrix calibration (PoE) |

**Our position (3-seed, the defensible cell).** Top-1 **+1.60pp over SCORE** (54.83 vs 53.23), and
the improvement is a *measured, significant* mechanism (t=3.49) rather than a single-seed lucky
draw. Top-5 is **tied** (−0.10pp). The structural-off family is bit-flat for every B, so the fusion
routes strictly through the structural term.

**E1 VERDICT (pre-registered): PASS.** `fuse=16` beats its own `fuse=0` twin by +1.03pp Top-1,
t=3.49, on 20/30 folds — t>2 and a majority positive, exactly as the sbatch's pre-registered
criterion required. Top-5 passes too (+0.90pp, t=3.75).

### The decisive field observation

**Every SOTA gain in this benchmark is a test-time calibration method, and every one of them
operates on a SINGLE MEAN embedding per concept.** SCORE recovers an order-1 coordinate; SATTC
calibrates the similarity matrix with rank/neighbourhood priors; both average the 80 test
repetitions into one point before doing anything. That mean is the object §1.4 of the v11 doc
already flagged as an untapped resource: the R repetitions are an *ensemble*, and averaging throws
away the spread that measures query reliability.

This is not a small gap. The R-curve (job 645719) says the retrieval operator is
estimation-limited and still climbing at R=80 (+3.33pp in the last doubling, 3/3 subjects), so the
information is there and is being discarded.

---

## 2. The claim COMMET makes

> Cross-subject EEG→image retrieval is neither coordinate recovery (SCORE) nor similarity-matrix
> calibration (SATTC). It is **metric completion from multiple independent noisy views**: the query
> is not a point but a cloud of R per-trial embeddings, and the right object is the concept-level
> metric (order-2), fused across views by measured reliability and matched against the gallery by a
> structured coupling. The per-trial scatter that every existing method averages away is exactly the
> reliability signal this needs.

Why this is a different object and not a better tuning of SCORE/SATTC: those act on a **point**
(order-1). COMMET acts on a **distance/Gram matrix** (order-2). §1.3 of the v11 doc shows the
shared metric is real but low-SNR (corr 0.565 / 0.593), and that everything more non-linear than
the metric loses SNR at a predictable rate — which is why thresholding (topology, graphs) failed
three times (P1, A4) while the metric itself carries signal (M1 +15.4pp real, though
permutation-lethal). Order-2 pooling is the one operation that *raises* the SNR of the object that
carries signal.

---

## 3. COMMET: the four components

| # | Component | Evidence it works | New relative to SCORE/SATTC |
|---|---|---|---|
| 1 | **Query = per-trial cloud** (R=80), not a mean | R-curve 4.0→54.5; T2 = +25pp over the mean | they use a mean point; discards the scatter |
| 2 | **Reliability-weighted FGW metric fusion** `D* = Σ w_k D_k` | probe t=+42.4, control ≈0; fuse monotone in B | metric-level fusion ≠ matrix calibration |
| 3 | **Gallery as a second view** to denoise the query | M2 = 0.593, the richest view, never used to denoise | entirely unused in the literature surveyed |
| 4 | **First-order mean as a regulariser** (protects Top-5) | fusion lifts Top-5 9/10 folds | complementary, not a replacement |

### Why it is test-time and label-free (and why that matters for accuracy)

The v12 twin (job 645754, cancelled) measured that putting the metric objective into the **encoder**
collapses it: the arm drove the SMN gate to 1.0, took raw cosine down 13pp and lost 0/4 folds.
COMMET's four components are all **test-time**: no labels, no encoder update. This is the same
regime SCORE and SATTC are evaluated in (and the regime that makes the benchmark honest), and it is
why COMMET can be high-innovation and high-accuracy at once — the innovation is in the estimator,
not in a fragile training term.

### Pre-registered verdicts

* **V1 (E1, DONE — PASS).** The fusion's contribution is real: `fuse=16` beats its own `fuse=0`
  twin, paired over 10 folds x 3 seeds, by **+1.03pp Top-1 (t=3.49, 20/30 positive)** and **+0.90pp
  Top-5 (t=3.75)**. The structural-off family is bit-flat, so the mechanism routes through the
  structural term.
* **V2 (E2, DONE — see §1b).** Cloud vs point on the same encoder and folds.
* **V3 (E3, not started).** Gallery as a second view: improves Top-1 without degrading Top-5.

---

## 1b. E2: cloud vs point, and the honest decomposition (job 646572)

10 folds x 1 seed, seed 2025, no training — every cell is from the banked v8 checkpoint, so the
comparison is paired by construction. `--sim-calib satc` adds SATTC's structural expert (the
strongest label-free POINT-based calibration, CVPR'26) on our encoder; this is the first time that
baseline exists on our folds (its `_ranks` helper was broken — see §4).

| Cell | Top-1 | Top-5 |
|---|---|---|
| point + CSLS + recovery (no structural calib) | 39.95 | 73.00 |
| point + **SATTC** structural expert | 40.40 | 73.35 |
| point + SATTC, `lam=0` twin | 39.95 | 73.00 |
| **cloud** (shipped T2, `fuse=0`) | 54.25 | 81.75 |
| **cloud + fusion (COMMET, `fuse=16`)** | **55.10** | **83.15** |
| cloud + fusion, structural-off | 50.40 | 82.05 |

**Paired against the best point-based method (`SATTC(CSLS+recovery)`, Top-1 40.40):**
COMMET **+14.70pp, 10/10 folds, t=7.39** (Top-5 +9.80pp, t=4.42).

**THE DECOMPOSITION, WHICH IS THE PART THAT MUST NOT BE SPUN (Top-1, 10 folds):**

| Step | Delta | Meaning |
|---|---|---|
| cloud vs point (point→shipped T2 `fuse=0`) | **+14.30pp** | using the R=80 per-trial cloud rather than the mean embedding |
| SATTC structural calibration on the point | **+0.45pp** | the field's strongest point-side calibration, on our encoder |
| **COMMET's fusion** on top of the cloud | **+0.85pp** (1 seed) / **+1.03pp** (3 seeds) | the order-2 metric estimator |

**Reading it honestly.** The +14.70pp headline is NOT "our fusion beats SATTC by 14.7pp": it is
overwhelmingly the **cloud-vs-point** step (+14.30pp), and that step is the shipped T2 operator, not
the v11 fusion. What E2 establishes is (a) the cloud is where the large gain lives and (b) the
field's point-side calibration cannot reach it — SATTC's structural expert moves our point pipeline
by only +0.45pp (its `lam=0` twin is bit-identical to the base, i.e. it essentially does not fire
here despite a mutual-top-k enrichment of 16.6). COMMET's own contribution is the smaller, but
significant and independently verified, **+0.85/+1.03pp**.

**What E2 does NOT establish.** The point baseline here (40.40) is our encoder under a point
pipeline; it is NOT a reproduction of SCORE's published 53.23, which uses a different (stronger)
coordinate recovery. So E2 is an **internal ablation** — it isolates the cloud's worth on our
encoder — and must not be reported as "we beat SCORE by 14.7pp". The external comparison remains the
one in §1: our 3-seed 54.83 vs SCORE's published 53.23 = **+1.60pp**, which is a cross-paper number
and is flagged as such.

### Guardrails carried over from the falsified families

* **Permutation control on every fusion cell.** The control is what turned M1's +15.4pp into
  −14.5pp; a fusion gain that survives it is shared structure, one that does not is leakage.
* **Structural-off twin on every cell.** Must stay flat; a gain with the structural term OFF means
  the pooling leaked through the cross-modal cost.
* **No encoder-side term without a rank/non-degeneracy guard.** v12 is the evidence.

---

## 4. A silent bug found while building the E2 baseline (and why it mattered)

`calibration._ranks` — the helper behind the SATTC structural expert — was **broken on any
real-sized matrix**. Its backward rank was computed with a fancy-index assignment
(`rb[np.argsort(-s, axis=0), np.arange(ng)[:, None]] = np.arange(nq)[:, None]`) whose two advanced
index arrays broadcast to the wrong (rank, column) pairs: on 200x200 it returned **non-integer,
out-of-range** values (measured −4.7 … 199, dtype-dependent), whereas the same code is CORRECT on a
4x4 toy — which is why it survived. The failure was silent in the worst way: `mutual` became
**negative**, `log` returned NaN, and the NaN propagated into the fusion through `0 * NaN` (so even
`lam=0` was poisoned). It is fixed to a double-`argsort` (provably the rank) and pinned in
`smoke_test.py` §18: both ranks valid on 200x200 for float32/float64, no NaN/Inf from
`structural_scores`, and `lam=0` recovering the base per-row ranking exactly.

Consequence: the SATTC baseline number for our encoder did not previously exist and had to be
produced by E2 rather than quoted — and the baseline's own docstring ("an unweighted fusion loses 17
points") was describing this bug, not a property of the method.
