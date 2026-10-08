# SBA feasibility gate — is the temporal factor a subject-invariant concept anchor?

2026-10-06. `scripts/probe_sba_anchor.py`, `outputs/probe/sba_anchor.json`. **Zero training** —
this is a measurement on banked raw EEG, not a model. It answers *only* the first of SBA's two
questions, and the answer is sharp.

## 0. Why this needs no training

SBA asks two different things and only one is a data question:

| question | what it needs | status |
|---|---|---|
| **① does a subject-invariant temporal factor exist?** | a measurement on the raw trials | **answered here, no training** |
| ② does a learned SBA encoder convert it into retrieval gains? | a trained model | not asked here |

① is a claim about the DATA (`X_c^(s) = A_s S_c + noise`, `A_s` a subject-specific spatial mix),
so it is measured, not trained. The full SBA architecture needs training only if ① passes.

## 1. Design

Per subject, per concept: grand-average ERP `X_c^(s) ∈ R^{63×250}` (mean over the 80 test
repetitions), per-channel z-scored. Five views, all from `X`:

| view | object | expected |
|---|---|---|
| `temporal_svd` | top-5 RIGHT singular vectors (time basis) | invariant to `A_s` |
| `gfp` | global field power `‖X[:,t]‖₂` | partly invariant |
| `spatial_svd` | top-5 LEFT singular vectors (topography) | NOT invariant |
| `spatial_amp` | channel amplitude profile | NOT invariant |
| `full` | `vec(X)` | reference |

Measured over all `C(10,2)=45` subject pairs and 200 concepts: `A` = same-concept cross-subject
affinity, `B` = different-concept affinity, `margin = A−B`, `ID` = cross-subject concept-identity
top-1 (chance 0.5%). `mix` = apply a random invertible 63×63 spatial mixing to one subject.

Subspace affinity = mean squared principal cosine between the two `k=5` subspaces. **Chance for
two random `k`-subspaces in `R^T` is `k/T`.**

## 2. Result (10 subjects, 45 pairs, 200 concepts)

| view | A (same) | B (diff) | margin | ID top-1 |
|---|---|---|---|---|
| **temporal_svd** | **0.6247** | 0.6047 | 0.0201 | **2.01% (4× chance)** |
| gfp | 0.4944 | 0.4502 | **0.0442** | 1.38% |
| spatial_svd | 0.0794 | 0.0794 | 0.0000 | 0.39% (**below chance**) |
| spatial_amp | 0.0009 | 0.0003 | 0.0006 | 0.51% (= chance) |
| full | 0.1050 | 0.0871 | 0.0180 | 1.27% |

`mix` stress test: temporal same-concept affinity 0.6257 → 0.5859 (drop **+0.04**); spatial
0.0794 → 0.0794 (cannot drop — already at chance).

## 3. What is ESTABLISHED

**The invariance claim is confirmed, and sharply.**

- Temporal subspace affinity **0.6247 vs chance `k/T = 5/250 = 0.02`** → **31× chance**.
- Spatial subspace affinity **0.0794 = `5/63` = exactly chance** → the topographies are
  **completely unaligned** across subjects, as volume conduction predicts.

So the SBA premise — *time is the symmetry-broken (invariant) axis, space is the corrupted one* —
is **measurably true**, and the gap is not marginal (31× chance vs exactly chance). Under a random
invertible channel mix the temporal factor moves by 0.04.

**Temporal is also the most cross-subject concept-discriminative raw view:** ID 2.01% vs spatial
0.39–0.51%. The small cross-subject concept signal in the raw ERP lives in the TIME axis, not the
spatial one (z≈20 for temporal vs chance, so real, if small).

## 4. What is NOT established — and what it costs SBA

**The invariance is largely TRIVIAL.** `B (different-concept) = 0.6047` is almost as high as
`A (same-concept)`. The temporal subspace is invariant because it is **concept-agnostic** — a
generic ERP template (N170/P300 shape) shared by essentially all concepts and all subjects. The
concept-SPECIFIC part is the margin, and it is small (0.020).

**There is a trade-off, not a free lunch:**

- `temporal_svd` — most invariant (0.62) but least concept-specific (margin 0.02).
- `gfp` — most concept-specific (margin 0.044) but least invariant (0.49).

No raw view is BOTH strongly invariant AND strongly concept-specific. And the absolute ceiling of
the raw temporal factor is ID ≈ 2%, against the **trained encoder's ~35–55%** cross-subject
retrieval — so a purely raw/linear temporal factor is **not task-sufficient**.

## 5. Verdict

**① SBA's premise (temporal invariance) is TRUE and sharp. ② A raw temporal factor is NOT
sufficient to carry the task; the concept-specific signal must be LEARNED.**

This is the honest boundary, and it answers the question precisely:

- The check that needed **no training** is done, and it passed on the invariance half — the novel
  geometric claim is real.
- The check that **does need training** is whether a learned encoder can extract the
  concept-specific temporal structure (the margin), because the raw margin (0.02) is too small to
  drive alignment and the encoder is demonstrably able to reach 35–55%.

So SBA is **worth building, but not as a "free" alignment operator** — the invariance is the
anchor and the encoder must learn what to hang on it. The next gate is a trained temporal-factor
encoder whose cross-subject concept-ID must clear the raw baseline (≈2%) by a large margin, with
the invariant temporal subspace as a fixed anchor.

## 6. Caveats

- The `mix` stress test is uninformative on the spatial side because spatial affinity is already
  at chance (it cannot drop further); it only confirms the temporal side.
- `B` is estimated from 2000 random off-diagonal concept pairs per subject pair; the margin is
  small relative to `A`, so it is the load-bearing number and should be reported with its spread.
- One window (full 0–1000 ms) was used; a post-stimulus-restricted window may raise the temporal
  margin and is the cheapest follow-up.
EOF
