"""Collapse and gradient-budget diagnostics for Stage 2.

Why these exist
---------------
The previous Stage 2 run trained without any of this and was only caught by a
post-hoc audit, at which point the conclusion was that the model had learned almost
nothing: `best.pt` (epoch 14) scored a total loss of 31.90 against a **constant
encoder's** 32.57, i.e. 2.3% better than emitting one fixed vector.  Three separate
statistics, all cheap, would each have shown it within the first few epochs:

* `top1_sv_ratio` on `z_inv` was 0.867 -- one principal direction held 87% of the
  variance, so the 512-d embedding was effectively one-dimensional.
* The raw cross-sample cosine was +0.9075 -- any two trials produced nearly the
  same vector, which is the same fact seen from the direction side.
* The total loss never separated from the constant-encoder baseline.

`representation_diagnostics` is the first two; `constant_encoder_loss` is the third.
Both are used by `loso.train.align` at evaluation time and by
`scripts/smoke_align.py` as a gate.

The two statistics are not redundant
------------------------------------
They are computed on different views of the same batch and detect different
failures, so a run must clear both:

* `mean_offdiag_cosine` is taken on the **raw** outputs.  It catches "one vector for
  everything", but is blind to a spread that is genuinely low-rank.
* `top1_sv_ratio` / `eff_rank` are taken on the **centred** outputs.  They catch a
  used-width collapse (all variance in one direction) but are blind to a constant,
  because mean-centring removes it first -- an encoder returning `c + noise` looks
  perfectly healthy by spectrum alone.

The measurements make the division of labour concrete, and they are why
`collapse_verdict` judges *drift from the initialisation* rather than absolute
values.  Measured on the real encoder at random initialisation
(`scripts/measure_z_scale.py`, 512 real trials):

    initialisation      top1_sv_ratio 0.107   mean_cosine +0.8037
    failed run          top1_sv_ratio 0.867   mean_cosine +0.9075

`head_inv` ends in a LayerNorm, so `z_inv` is a point on a sphere of radius
`sqrt(d_inv)`, and on that sphere per-dimension variance plus squared per-dimension
mean sums to exactly 1 per dimension: the initialisation puts 80.4% of that energy in
the shared mean direction.  So the raw cosine is already at +0.80 before training and
only moved 0.10 further, while `top1_sv_ratio` moved by 8x.  An absolute threshold on
the cosine would therefore flag every untrained model and the gate would be ignored;
the drift form catches what actually happened.

Neither half can be "simplified away": `scripts/smoke_align.py` constructs one case
for each failure and asserts that the statistic that should fire does, and that the
other one does not.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

#: Reference numbers from the failed run, kept as literals so the gate in
#: `scripts/smoke_align.py` asserts against the real failure rather than a guess.
FAILED_RUN_TOP1_SV_RATIO = 0.867
FAILED_RUN_MEAN_COSINE = 0.9075


def representation_diagnostics(z: torch.Tensor) -> dict[str, float]:
    """Collapse indicators for one representation, (N, D).

    Returns `top1_sv_ratio`, `eff_rank`, `mean_offdiag_cosine` and `per_dim_std`.

    `eff_rank` is `exp(entropy)` of the normalised singular-value spectrum -- the
    standard "how many directions are actually in use" measure.  It is an order of
    magnitude rather than a count, so a healthy 16-d embedding reports ~14 and a
    healthy 512-d one reports ~50+; a threshold on it therefore has to be relative
    to `min(N, D)`, which is why the gate uses `< 5` (unambiguous collapse) rather
    than anything near the healthy value.
    """
    z = z.float()
    n = z.shape[0]
    if n < 2:
        return {"top1_sv_ratio": float("nan"), "eff_rank": float("nan"),
                "mean_offdiag_cosine": float("nan"), "per_dim_std": float("nan")}

    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    unit = F.normalize(z, dim=-1)
    cosine = unit @ unit.t()
    mean_raw_cosine = float(cosine[~eye].mean())

    centred = z - z.mean(dim=0, keepdim=True)
    gram = centred @ centred.t()
    eigenvalues = torch.linalg.eigvalsh(gram.double()).clamp_min(0)
    spectrum = eigenvalues.flip(0)
    total = spectrum.sum()
    if total <= 0:
        # Every row identical under centring: no variance in any direction.
        return {"top1_sv_ratio": 1.0, "eff_rank": 1.0,
                "mean_offdiag_cosine": mean_raw_cosine,
                "per_dim_std": float(centred.std(dim=0).mean())}

    share = spectrum / total
    entropy = -(share * (share + 1e-12).log()).sum()
    return {
        "top1_sv_ratio": float(spectrum[0] / total),
        "eff_rank": float(entropy.exp()),
        "mean_offdiag_cosine": mean_raw_cosine,
        "per_dim_std": float(centred.std(dim=0).mean()),
    }


def collapse_verdict(stats: dict[str, float],
                     initial: dict[str, float] | None = None) -> tuple[bool, str]:
    """Turn the statistics into a pass/fail with a readable reason.

    Judged as *drift from the initialisation* when `initial` is given, and against
    absolute thresholds otherwise.  The drift form is the one that matters, and the
    measurement is why: at random initialisation this encoder already reports
    `mean_offdiag_cosine = +0.80` (80.4% of the sphere's energy sits in the shared
    mean direction -- see `scripts/measure_z_scale.py`).  The failed run ended at
    +0.9075.  So an absolute threshold on the raw cosine cannot distinguish "not
    trained yet" from "collapsed", and a gate built on one would either reject every
    run at epoch 0 or accept the failure.

    `top1_sv_ratio` is the discriminator, and its behaviour is the opposite: it
    started at 0.107 for exactly the same initialisation and the failed run drove it
    to 0.867 -- an 8x increase, one direction absorbing the entire spectrum.  That is
    a real change in the representation, and it is invisible in the raw cosine.

    Neither statistic alone is sufficient, which is the whole point of computing both:
    the raw cosine catches a constant added to everything (invisible after centring),
    the spectrum catches a used-width collapse (invisible against a large constant).
    The failed run violated both; the initialisation violates only the first.
    """
    top1 = stats["top1_sv_ratio"]
    cosine = stats["mean_offdiag_cosine"]
    rank = stats["eff_rank"]
    if math.isnan(top1):
        return True, "not enough rows to judge; skipping"

    if initial is not None and not math.isnan(initial["top1_sv_ratio"]):
        grown = top1 / max(initial["top1_sv_ratio"], 1e-6)
        cosine_delta = cosine - initial["mean_offdiag_cosine"]
        base = (f"top1_sv_ratio {initial['top1_sv_ratio']:.3f}->{top1:.3f} "
                f"({grown:.2f}x), mean_cosine "
                f"{initial['mean_offdiag_cosine']:+.3f}->{cosine:+.3f} "
                f"({cosine_delta:+.3f}), eff_rank={rank:.1f}")
        # 3x is the threshold rather than 1.5x because the quantity is noisy for small
        # batches and because the failure it must catch was 8x.  A gate tuned to
        # detect a 1.5x drift would fire on runs that are merely still moving.
        if grown > 3.0 or cosine_delta > 0.15:
            return False, f"collapsing: {base}"
        if grown > 1.5 or cosine_delta > 0.05:
            return True, f"warning, drifting toward collapse: {base}"
        return True, f"holding: {base}"

    # Absolute form, for callers with no baseline to compare against.  These
    # thresholds are on the *representation*, so they are read together with the
    # failed run's terminal values (0.867 / +0.9075).
    fired = []
    if top1 > 0.5:
        fired.append(f"top1_sv_ratio={top1:.3f} > 0.5")
    if cosine > 0.9:
        fired.append(f"mean_cosine={cosine:+.3f} > 0.9")
    if rank < 5:
        fired.append(f"eff_rank={rank:.2f} < 5")
    if fired:
        return False, "collapsed: " + ", ".join(fired)
    if top1 > 0.3:
        return True, (f"warning: top1_sv_ratio={top1:.3f} > 0.3; "
                      f"mean_cosine={cosine:+.3f}, eff_rank={rank:.2f}")
    return True, (f"healthy: top1_sv_ratio={top1:.3f}, mean_cosine={cosine:+.3f}, "
                  f"eff_rank={rank:.2f}")


@torch.no_grad()
def constant_encoder_loss(query_batch: torch.Tensor, key_batch: torch.Tensor,
                          logit_scale: torch.Tensor, center: bool = False) -> float:
    """What the alignment term evaluates to when the encoder collapses to a constant.

    This is the baseline the training loss has to separate from, and the failed run
    very nearly did not: its total was 31.90 against 32.57 for a constant encoder, a
    2.3% margin.  Logged at every evaluation, because it is the one number that says
    whether the term is measuring anything -- a term sitting at its constant-encoder
    value is being satisfied by the *projector* alone, with no information flowing from
    the encoder.

    A constant encoder maps every input to the same `z`, so every row of the query
    batch is the same vector and the rows of the similarity matrix are identical.  The
    loss is then a single number, `-mean_b log softmax(q . k_c)[b]`, and it is worth
    being precise that this is *not* `log(B)`: `log(B)` is what you get when the
    logits are constant across both axes, which requires the *keys* to coincide too.
    A constant query still discriminates between keys -- that is exactly why a
    collapsed encoder can still produce a finite, apparently reasonable loss, and why
    the comparison has to be made rather than eyeballed.

    The constant used is the batch-mean query direction, not an arbitrary vector.
    That makes the baseline the strongest available degenerate solution, so beating it
    is a conservative claim; an arbitrary constant would let a run appear to improve on
    the baseline through the choice of that vector rather than through the encoder.

    The real `soft_label_contrastive` is called rather than the closed form being
    recomputed: if the loss later acquires a different reduction, or a temperature
    applied in a different place, a reimplemented formula would silently stop
    describing it.  `scripts/smoke_align.py` pins the equivalence from both sides.
    """
    b = query_batch.shape[0]
    if b < 2:
        return float("nan")
    from loso.losses import align as L  # local: `loso.losses` imports nothing from here
    keys = F.normalize(key_batch.float(), dim=-1)
    # The single fixed output of the collapsed encoder: the mean direction of the real
    # queries, broadcast to every row.
    mean_direction = F.normalize(query_batch.float().mean(dim=0, keepdim=True), dim=-1)
    constant = mean_direction.expand(b, -1).contiguous()
    # `center` must match what the live objective does, or the baseline is computed on
    # a different function than the loss it is meant to be compared against.  Centring
    # a set of identical rows leaves zeros, and `center_normalize` re-normalises those
    # from a zero vector, so the collapsed case is degenerate here in a way the
    # uncentred one is not -- which is itself the point: once the shared direction is
    # removed, a constant encoder has nothing left to score with.
    return float(L.soft_label_contrastive(constant, keys, logit_scale.float(),
                                          center=center))


class GradientBudget:
    """Running record of each loss term's share of the total gradient.

    The failed run's `L_time` carried weight 3.0 and produced roughly 58% of the
    gradient, which is how an objective with a false correspondence (EEG token i
    matched to VAE raster patch i) came to dominate everything else.  Weights alone
    do not tell you this: a term's share depends on its weight *and* on the gradient
    magnitude it produces, which is a property of the data and the current
    parameters.  Measuring it is the only way to see the budget as it actually is.

    The measurement is the L2 norm of the gradient each term produces on the shared
    encoder parameters, as a fraction of the sum over terms.  It costs one extra
    backward per logged step, so it is off by default and used in diagnostics runs.
    """

    def __init__(self, param_names: list[str] | None = None):
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.param_names = param_names

    def record(self, term: str, norm: float) -> None:
        self.totals[term] = self.totals.get(term, 0.0) + norm
        self.counts[term] = self.counts.get(term, 0) + 1

    def shares(self) -> dict[str, float]:
        means = {k: v / max(1, self.counts[k]) for k, v in self.totals.items()}
        total = sum(means.values())
        if total <= 0:
            return {k: 0.0 for k in means}
        return {k: v / total for k, v in means.items()}

    def report(self) -> str:
        shares = self.shares()
        ordered = sorted(shares.items(), key=lambda kv: -kv[1])
        return "gradient budget: " + " ".join(
            f"{k}={v * 100:.0f}%" for k, v in ordered)
