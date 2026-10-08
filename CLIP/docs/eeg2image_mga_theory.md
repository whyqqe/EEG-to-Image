# MGA — Manifold Gauge Alignment: a theory of cross-modal alignment by gauge fixing, and its falsification on our encoder (2026-10-06)

This document states a new theory of cross-modal / cross-subject alignment grounded in data-manifold
geometry, gives its mathematical core, and records the pre-registered probe that **killed its
temporal branch on our encoder** (job 648638) before any training was spent. It is the theory half
of the direction surveyed in §I; §IV is the honest verdict.

---

## I. Where the field is, and why the theory targets the gap nobody filled

The 2026 frontier, read from the literature:

| family | representative | what it aligns | why it does NOT fix our problem |
|---|---|---|---|
| functional maps / spectral | [arXiv 2604.08579](https://arxiv.org/pdf/2604.08579) | eigenbasis linear operator | **underperforms** Procrustes / relative reps 5–13×; eigenvalue spectra agree (0.043) but eigenvector bases do not — the "spectral complexity–orientation gap" |
| sheaf / obstruction | [arXiv 2604.07632](https://arxiv.org/pdf/2604.07632) | local-to-global consistency | gives a *criterion* (obstruction energy bounds excess global error), not a fix |
| Gromov–Wasserstein | [GW-MDS 2604.23912](https://arxiv.org/abs/2604.23912), [2605.04175](https://arxiv.org/html/2605.04175), [MGMCL 2608.08440](https://arxiv.org/abs/2608.08440) | the metric (order-2) | $O(d)$-**invariant**: it cannot fix orientation |
| hyperbolic / product | [ProCLIP](https://openreview.net/pdf?id=Sp3BrGpo5w), [PHyCLIP 2510.08919](https://arxiv.org/pdf/2510.08919), [2510.27391](https://arxiv.org/abs/2510.27391) | hierarchy / composition | aligns *curvature classes*, not the frame |

**The gap**: orientation / gauge. Every family above aligns a quantity that is either $O(d)$-invariant
(so blind to the frame) or supervised. MGA asks whether the frame can be fixed by **intrinsic
dynamics**.

---

## II. The theory

### II.1 Setup and the gauge

Let the frozen encoder be $E$, and let $M \subset \mathbb{R}^d$ be a subject's concept manifold. The
deployment whitening $W$ (fitted on the repetition cloud) makes the query edge **isotropic in second
order**. Hence for any $R \in O(d)$ the whitened marginal is unchanged:

$$\text{any } O(d)\text{-invariant functional is blind to the frame.}$$

Spectral distances, GW distances, heat-kernel scalars, topology — all $O(d)$-invariant by
construction. **This is the theorem behind our own falsification list**: C1 (≤ +0.009 corr), P1/P2
(0/30), A4 (≤ 0), G0 (−1.20pp) all targeted $O(d)$-invariant objects.

### II.2 Dynamics supplies the frame

Let a single trial be a curve $z_r(\tau)$ on $M$ (an ERP is a controlled trajectory). Define the
**dynamic frame**

$$V \;=\; \mathrm{eig}_k\!\left(\mathbb{E}_{r,\tau}\!\left[\dot z_r(\tau)\,\dot z_r(\tau)^{\top}\right]\right),\qquad \dot z = \tfrac{dz}{d\tau}.$$

$V$ is a $k$-frame; it is stabilised only by the rotations that fix it:

$$\mathrm{Stab}(V) = O(d-k)\quad\Rightarrow\quad \text{residual gauge } O(d-k).$$

**A $k$-frame cuts $O(d)$ down to $O(d-k)$.** Crucially $V$ is intrinsic (built from the subject's own
trials), legitimate (no source test data, no labels beyond the task's own concept index), and — unlike
every quantity in §I — **not $O(d)$-invariant**.

### II.3 Gauge fixing with a canonical source frame, and an obstruction gate

The source domain has labels and the encoder was trained on it, so its dynamic frame $V_s$ is a
known canonical frame. Fix the target frame by

$$R^\star=\arg\min_{R\in O(d)}\ \sum_c \big\|\Pi_c(R\,V_t)-V_s\big\|_F^2
 \;+\;\lambda\,\mathcal{R}_{\text{spec}}(R)\;+\;\mu\,\|R^\top R-I\|_F^2,$$

where $\Pi_c$ localises to concept $c$, $\mathcal{R}_{\text{spec}}$ is a multi-scale heat-kernel
regulariser ([FSAlign](https://aiconfpaper.com/paper/icml-2026-pGkM5BjfD1)), and the last term pins the
solution to $O(d)$. Finally, gate global vs local with the sheaf obstruction $E_0$ ([2604.07632](https://arxiv.org/pdf/2604.07632)):

$$E_0<\varepsilon \Rightarrow \text{one global } R^\star;\qquad E_0\ge\varepsilon \Rightarrow
\text{local charts }\{R_c\}_c.$$

This uses **only source labelled train dynamics + the target's own trials** — never a source *test*
metric substituted into the structural term — so it is immune to the evaluation artefact §6/§8 of
`eeg2image_commet_architecture.md` exposed.

---

## III. The pre-registered probe (MGA-0), and its guards

`scripts/probe_mga0.py`, frozen encoder, no training, 10 folds × 9 source→target pairs, 200 shared
concepts. Descriptors, all on the repetition-averaged trial:

| tag | definition | role |
|---|---|---|
| `static` | $E(x)$ | the deployment object |
| `rev` | $E(x)-E(x\ \text{reversed in time})$ | **full-support order contrast** — the theory's object |
| `scr` | $E(x)-E(x\ \text{time-permuted})$ | order-destroying control (same support) |
| `dyn` | $E(x_{:T/2})-E(x_{T/2:})$ | half-masked contrast — a **positive control for the trap** |

Criterion, paired on held-out concepts: `rev` must beat `static` **and** `scr` on held-out concept
recovery (NN in the target's space after an orthogonal-Procrustes fit on the other half).

---

## IV. Verdict: the temporal branch is falsified (job 648638)

| descriptor | held-out NN concept recovery | transfer residual |
|---|---|---|
| `static` | **0.3464** | 0.804 |
| `rev` (order) | 0.2787 | 0.594 |
| `scr` (order destroyed) | 0.2784 | 0.548 |
| `dyn` (masked) | 0.2976 | **0.421** |

* **`rev` is significantly WORSE than `static`**: −6.76pp, t = −6.24, 10/10 folds.
* **`rev` ≈ `scr`**: order gain +0.0004, t = 0.02. The temporal **order** carries essentially zero
  usable information beyond a random reordering of the same spectral content.
* The masked `dyn` shows the trap precisely: **lowest** transfer residual (0.421) but **worse** than
  static on recovery — its dominant direction is the encoder's response to the *masking intervention*
  (concept-independent), i.e. trivially alignable and non-discriminative. A residual-only reading
  would have called this a win.

Combined with the earlier **static** gauge probe (`outputs/probe/gauge/gauge_probe_v2.json`:
"GROUP SUFFICES: global maps already compose"), the two probes close the premise from both sides:

> **Our encoder is jointly trained, so its subjects already share a frame. There is no residual
> gauge for dynamics to fix, and the temporal-order contrast adds no alignable structure.**

So the "orientation gap" of [2604.08579](https://arxiv.org/pdf/2604.08579) — a phenomenon of
*independently* pretrained encoders — does not transfer to a jointly trained metric-learning encoder.

---

## V. What this means for the direction

MGA is a sound theory for the regime it was written for (independently pretrained, frame-misaligned
encoders). It is **falsified for our setting** for a measurable reason, not a philosophical one, and at
the cost of one GPU job rather than a training campaign. The consequence is the same as the operator
analysis reached from the other end:

> The obstruction is **not** the frame and **not** the prior. It is the **residual** — the
> subject-shared signal itself (cross-subject concept-metric corr $+0.565$ ⇒ ≈ 32% shared variance),
> plus the measurement ceiling that the repetition cloud already spends.

The reportable contributions remain: (1) the **cloud-vs-point** mechanism (+14.30pp, legal, our own
reps), (2) the **plan_acc exploit** diagnostic (§8 of the architecture doc), and (3) this **theory +
two-sided falsification** (static gauge, dynamic order) — three pre-registered kills that each name a
concrete boundary of the alignment design space.
