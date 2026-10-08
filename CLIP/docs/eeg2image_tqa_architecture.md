# TQA -- Theory of Quotient Alignment

> The unified account of an EEG-to-image encoder's cross-subject invariance, and the
> architecture it produces. This document is the single theoretical frame for four things
> that had been treated as separate tricks: the temporal anchor, the training augmentation,
> the test-time Reynolds operator, and the new group-quotient front-end.
> Last updated: 2026-10-06.

---

## 1. The one object

Everything below is the same operator seen from four sides. Let `G = GL(d_ch)` be the group
of invertible linear recombinations of the electrode channels. The physical claim is that
volume conduction acts on a trial `X ∈ R^{d_ch × T}` as

```
X  ->  M X ,      M ∈ G ,
```

because an electrode montage measures invertible linear mixtures of the underlying sources,
and the mixture differs per subject. Cross-subject generalization is therefore the problem
of learning a function that is constant on the `G`-orbits.

The exact invariant of that problem is the **Reynolds operator**

```
P_G f(X) = E_{M ~ G} f(M X) ,
```

the group average. A representation is `G`-invariant iff `P_G z = z`. Every method in this
project is an attempt to compute, approximate, or serve `P_G`, and the four look different
only because `G` is non-compact (`GL`), which makes `P_G` degenerate as a projection and
forces every explicit form of it to fail.

---

## 2. The two-horn dilemma, restated

Two horns, both measured on this project:

* **Learn invariance explicitly** → collapse. A penalty `||z(MX) - z(X)||` has to trade
  invariance against discrimination *inside one objective*, and because `P_G` is a
  degenerate projection on a non-compact group, the optimum of the trade-off is a
  representation that is nearly constant. Every explicit-invariance arm this project ran
  died this way.
* **Average it at inference only** → nothing. `P_G` on an encoder that never saw the orbit
  is a large move, and its residual carries concept signal, so averaging *introduces bias*.
  Measured (job 648881): the Reynolds residual of the shipped v8 encoder is 0.172 at K=8,
  `TTGA`'s ΔTop-1 is noise (`+0.35pp`, t=0.55) and its ΔTop-5 is *negative* (`-0.80pp`).

A third horn is available and is the whole point of TQA: **put the quotient where it costs
nothing.** If the invariant factor can be computed in closed form on the *input*, no
trainable capacity is spent on invariance and no objective trades it against discrimination.

---

## 3. The decomposition: where the concept lives

`GL(d_ch)` factors by the polar decomposition as

```
G = O(d_ch) · PD(d_ch) ,
```

a **re-referencing rotation** and a **positive-definite stretch** (per-channel gain plus
channel-correlation reshaping). The measurement that decides the architecture is *which
factor the concept is in*, and it was made in `scripts/probe_sba_anchor.py`:

| object | affinity | chance | reading |
|---|---|---|---|
| temporal (row-space) subspace, ERP window | **0.6247** | 0.02 | concept signal |
| spatial topographies (left singular vectors) | 0.0794 | 0.0794 | pure group orbit |

The concept lives in the factor the group action does **not** move -- the row space, i.e.
the temporal structure. The spatial topography is at exact chance, i.e. it is the orbit.

---

## 4. The unified architecture: four levels, one operator

| level | object | how `P_G` is computed | status |
|---|---|---|---|
| **L0 front-end (new)** | quotient the stretch `PD` | closed form, per trial | this job |
| **L1 anchor** | the invariant factor | exists by construction | measured 0.6247 |
| **L2 training** | Monte-Carlo estimate of `P_G` | group augmentation `X -> MX` | **+2.10pp**, 8/10, t=1.98 |
| **L3 deployment** | `P_G` on the repetition cloud | reps + structure ensemble | shipped, 55.10 |

**L0 -- Group-Quotient Front-End (GQF).** One closed-form, zero-parameter operation:

```
W = (X X^T + eps I)^{-1/2} X ,      shape unchanged (d_ch, T) .
```

It quotients the `PD` factor exactly: for any `M`, `W(MX) = O W(X)` with `O` orthogonal
(verified: row norms = 1 to 7e-6; the `O`-invariant of `W`, its column Gram `W^T W`, agrees
to 4e-5 across a dense `M` at `‖M − I‖_F = 3.15`). **The residual `O(d_ch)` is honestly
labelled as not quotiented**, and it cannot be: the only canonical representative of the
row space is the `(T, T)` projector `X^T (X X^T)^{-1} X`, which does not fit a `(d_ch, T)`
trunk. The right-singular-subspace mode (`gqf: rowspace`) is the shape-compatible
approximation and is measurably *not* invariant on EEG (0.41 max-diff), because EEG
singular values are near-degenerate; it is kept only as that ablation.

**L2 -- group augmentation** is the Monte-Carlo estimate of `P_G`, and it is the level that
needed retraining. It is worth a paired **+2.10pp Top-1 / +1.70pp Top-5** (job 648728),
and it is the only level whose contribution could be attributed to a retrain, because L1
and L3 are deployment operators identical across the two arms.

**L3 -- the repetition cloud** is `P_G` served at inference: the `R` un-averaged
repetitions give an `R`-times better estimate of the same mean/covariance the deployment
query needs, and the FGW structure ensemble denoises the metric rather than truncating it.

---

## 5. Why TQA is falsifiable, and what would kill it

The frame makes three predictions, two of which are already in:

1. **Reynolds residual**: an encoder trained *with* the augmentation should be closer to
   invariance. **Holds** (job 648881): residual at K=8 is `sqa 0.1553 < v8 0.1719 <
   sqa_noaug 0.1753`, ordered as predicted on every K.
2. **Crossover at inference**: `TTGA` should help the augmented encoder and hurt the
   un-augmented one. **Fails**: `sqa` is `+0.00pp`, `sqa_noaug` is `-1.30pp` (as
   predicted) but `v8` is `+0.35pp` (noise), and Top-5 is uniformly negative. The honest
   reading is that a residual of 0.155 leaves almost no variance for the orbit average to
   remove while still carrying concept signal -- i.e. **the invariance axis is saturated**,
   which is exactly the argument for moving the lever to the *representation* (L0).
3. **Subgroup ablation**: if the win is the group action and not generic perturbation risk,
   the diagonal subgroup (rescalings only, the `gain` direction) should reproduce much less
   of it than the off-diagonal recombination. Designed and pre-registered in
   `scripts/summarize_sqa_attribution.py`; the `dense`/`diag`/`orth` panel is pending.

If (3) shows the diagonal subgroup carries the gain, the group claim is not what earned the
+2.10pp and TQA's level-2 story is wrong -- which is a falsification, not a disappointment.

---

## 6. What ships in this submission

`slurm/tqa_loso10x1.sbatch`, 10 folds × 1 seed, two arms differing in one config key:

* **Arm A `tqa_v8`** = `v8` (55.10 base) + group augmentation. The SOTA arm: the two
  measured effects composed, expected ≈ 57.
* **Arm B `tqa_v8_gqf`** = Arm A + GQF. The architecture arm.

Both are evaluated on the deployed cell `+ T2 R=80,a=0.75,t=0.03,fuse=16`, and Arm B is
paired against Arm A so the GQF contribution reads off as one number. Evaluation is
additive: Arm A and the banked `v8` row remain on disk whatever Arm B does.
