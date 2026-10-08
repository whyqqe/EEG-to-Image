"""v13 / CROMA: the order-2 BIAS-ANCHOR term -- the only new training objective.

THE ARGUMENT, IN ONE PARAGRAPH. Deployment reads the agreement between the query-side concept
metric $D_q$ (fused from $R$ repetitions) and the gallery-side metric $D_g$. The estimator error
decomposes into bias, variance and rank, and there is no fourth term -- so a training objective is
only justified if it is the gradient of one of those three. A measurement on ten banked checkpoints
(job 650135, `scripts/probe_view_independence.py`) settled which:

  * **variance is exhausted** -- the repetition-error correlation is $\rho \le 0$ on 10/10 folds, so
    $K_\text{eff} = R = 80$: pooling already removes every independent component, and a training-side
    variance term has nothing to gain (it is deliberately ABSENT from this module);
  * **bias is the live lever, and it is subject-specific** -- a 9-subject pooled EEG metric, i.e.
    ~720 repetitions and 9 independent *encodings*, agrees with the gallery at 0.796 while the
    target's own 80 repetitions reach only 0.616 (+0.180, paired $t=+22.2$, 10/10 folds, its own
    concept-shuffle control at -0.0005). Pooling cannot remove a subject's own deviation; averaging
    over encodings can. That gap is bias, and it is what this term attacks.

WHY THE ANCHOR IS $D_g$ AND NOT A SELF-CONSISTENCY TARGET. v12 distilled the repetition cloud's own
consensus metric into a single repetition and collapsed (metric rank 2-3, raw cosine -13.85pp,
headline -20.15pp): a self-consistent target is SHRINKABLE, so a low-rank student can be arbitrarily
correlated with it. $D_g$ is the metric of one fused image embedding -- exogenous, not a function of
the encoder, and full-rank -- so the encoder cannot make it low-rank, and the collapse channel is
closed by construction rather than by a guard.

WHY IT IS NOT "JUST ANOTHER CONTRASTIVE TERM". It consumes the same pair $(z_e, z_i)$ the existing
InfoNCE consumes and differs only in the readout: InfoNCE constrains the diagonal neighbourhood
(point agreement), this constrains the whole concept-metric matrix (relation agreement). Deployment
reads the whole matrix. Order-1 is the degenerate readout of the same principle.

THE COLLAPSE GUARD IS STILL EXPLICIT. "Cannot collapse" is an argument, not a measurement, so every
call returns the student metric's effective rank and the achieved agreement. `smoke_test.py` asserts
the rank does not fall below the baseline, because a bias number bought by rank collapse is a loss.
"""
from __future__ import annotations

import torch

from .metric_distill import _corr, _metric, _subject_blocks


def _whiten_cloud(x: torch.Tensor, shrink: float = 0.0, eps: float = 1e-8) -> torch.Tensor:
    """Differentiable whitening of a point cloud `(n, d)` -- the frame the OPERATOR reads.

    WHY THIS IS THE FIX AND NOT A REFINEMENT. v13 measured that its bias anchor improved the metric
    agreement in the RAW encoder frame (`dR80 = +0.025..+0.041`) and essentially nothing in the
    DEPLOYED whitened frame (`+0.001..+0.011`), while losing 9-13pp of Top-1. A genuine structural
    gain cannot vanish under a change of basis; a gain that does is ANISOTROPY, i.e. the encoder
    moved along the direction the deployment whitener deletes. The loss was scored in one frame and
    the operator reads another. This function puts the loss in the operator's frame.

    What it buys, provably: `Cov(whiten(x A)) = Cov(whiten(x))` for any invertible `A` applied to
    the whole cloud, because the whitener is fitted from that same cloud. So `D_student` -- and
    therefore the loss -- is INVARIANT to per-subject invertible linear maps, which is exactly the
    anisotropy class the encoder was exploiting. The raw-frame loss was not.

    THE WHITENING MAP IS DETACHED, AND THAT IS LOAD-BEARING FOR NUMERICAL SURVIVAL, NOT A
    SHORTCUT. The first version of this function differentiated through `torch.linalg.eigh`, and the
    first real run (job 650708) died at epoch 0 step 102 with `non-finite gradient` in BOTH whitened
    arms -- 20 failed trainings. The forward value was finite, so only the backward exploded, which
    is the signature of eigh's gradient: it carries `1 / (lambda_i - lambda_j)` terms, and the
    encoder's early-training spectrum is nearly degenerate (`effrank` ~47 of 64, and the smallest
    eigenvalues sit near the `eps` floor), so those terms are unbounded by construction. Detaching
    the map removes that backward entirely while leaving the FORWARD value -- and therefore the
    invariance above -- bit-for-bit unchanged. It is also the more faithful choice: deployment fits
    its whitener on frozen features and no gradient ever passes through it, so differentiating
    through it here would be optimising a quantity the operator does not have.

    A RELATIVE eigenvalue floor replaces the absolute one for the same reason. `evals.clamp_min(eps)`
    with a fixed `1e-8` lets a nearly-singular direction be scaled by `1/sqrt(1e-8) = 1e4`; flooring
    at `1e-6 * lambda_max` caps that amplification at `1e3` AND stays scale-covariant, so it does not
    break the invariance the way absolute shrinkage did (smoke §22 pins `shrink=0` vs `0.1`).

    Eigendecomposition rather than Cholesky: the spectrum is clamped, so a rank-deficient cloud
    (fewer points than dimensions, which the 3-subject block layout can hit) degrades to partial
    whitening instead of raising inside a training step.
    """
    if x.dim() != 2:
        raise ValueError(f"_whiten_cloud expects (n, d), got {tuple(x.shape)}")
    n, d = x.shape
    if n < 2:
        return x
    mu = x.mean(dim=0, keepdim=True)
    with torch.no_grad():                       # the frame is a constant; see the docstring
        xc = x - mu
        cov = (xc.transpose(-1, -2) @ xc) / float(n - 1)
        tr = cov.diagonal().mean().clamp_min(eps)
        cov = cov + (float(shrink) * tr + eps) * torch.eye(d, dtype=x.dtype, device=x.device)
        evals, evecs = torch.linalg.eigh(cov)
        floor = evals.max().clamp_min(eps) * 1e-6
        w = evecs @ torch.diag(1.0 / evals.clamp_min(floor).sqrt())
    return (x - mu) @ w                         # gradient flows through `x`, not through the map


def _block_concept_emb(block: torch.Tensor, inv: torch.Tensor, n: int, r_stu: int,
                       frame: str, shrink: float) -> torch.Tensor:
    """One subject's rows `(rows, R, d)` -> its `(n, d)` per-concept embedding, in `frame`.

    In the `whitened` frame the whitener is fitted on THIS subject's `(rows * r_stu, d)` cloud and
    applied BEFORE the per-concept averaging, which is the order deployment uses (`mu` and `w_map`
    come from the whole `(C*R, d)` cloud, then the cloud is averaged). Fitting on the `n` concept
    means instead would whiten a rank-`n` object with an `n`-sample covariance and manufacture a
    metric -- the P1 mistake (a prior evaluated in a different space than it was measured in).
    """
    rows, rr, dd = block.shape
    sel = block[:, :r_stu]
    if frame == "whitened":
        sel = _whiten_cloud(sel.reshape(rows * rr, dd), shrink=shrink).reshape(rows, rr, dd)
    elif frame != "raw":
        raise ValueError(f"unknown frame {frame!r}; expected 'raw' or 'whitened'")
    emb = block.new_zeros((n, dd))
    cnt = block.new_zeros((n,))
    cnt.index_add_(0, inv, torch.ones_like(inv, dtype=block.dtype))
    emb.index_add_(0, inv, sel.mean(dim=1))
    return emb / cnt.clamp_min(1.0)[:, None]


def _eff_rank(m: torch.Tensor) -> float:
    ev = torch.linalg.svdvals(m)
    return float((ev.sum() ** 2) / (ev ** 2).sum().clamp_min(1e-12))


def _empty_diag(prefix: str = "bias_anchor") -> dict:
    """The degenerate-block diagnostic, keyed by TERM.

    The prefix is not cosmetic. Both order-2 terms route their "no usable block" path through this
    helper, and the two consume DIFFERENT keys (`bias_anchor_*` vs `cs_*`). A single fixed key set
    made the consensus term return `bias_anchor_pairs` and then raise `KeyError: 'cs_pairs'` in the
    caller -- a crash in the degenerate branch that a well-formed batch never exercises, so it would
    have surfaced only on the one fold whose sampler produced a short block, hours into a job. The
    smoke test caught it precisely because it feeds a deliberately degenerate block.
    """
    return {f"{prefix}_pairs": 0, f"{prefix}_agreement": float("nan"),
            f"{prefix}_student_rank": float("nan")}


def gallery_metric(z_img: torch.Tensor, grp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The exogenous anchor: one metric matrix per unique concept, plus the id order.

    Rows sharing a concept are averaged first -- with `images_per_pair = 1` each concept has exactly
    one image, but averaging rather than assuming keeps this correct if that changes, and the concept
    index is the one axis that must never be mis-paired (it is the axis whose permutation turned M1's
    +15.4pp into -14.5pp).

    Detached on purpose: the target is a fixed feature of the task, not something the encoder is
    allowed to bend. `freeze_img_pre` is about the projector; this is about the loss.
    """
    uniq, inv = torch.unique(grp, sorted=True, return_inverse=True)
    d = z_img.shape[-1]
    pooled = z_img.new_zeros((int(uniq.numel()), d))
    pooled.index_add_(0, inv, z_img)
    counts = z_img.new_zeros((int(uniq.numel()),))
    counts.index_add_(0, inv, torch.ones_like(inv, dtype=z_img.dtype))
    pooled = pooled / counts.clamp_min(1.0)[:, None]
    return _metric(pooled.detach()), uniq


def metric_anchor_loss(z_rep: torch.Tensor, z_img: torch.Tensor, grp: torch.Tensor,
                       student_reps: int | None = None, min_concepts: int = 4,
                       return_diag: bool = True):
    """``1 - corr(D_student, D_gallery)`` over the WHOLE batch (one pooled metric).

    `z_rep` is `(N, R, d)` (per-repetition embeddings of the selected rows); `z_img` is `(N, d)`
    (each row's image embedding); `grp` is the stimulus id per row.

    This is the diagnostic/aggregate form. The DEPLOYED-FAITHFUL objective is
    `metric_anchor_loss_by_subject`, because deployment scores one subject's queries against the
    gallery alone; a mixed-subject metric asks the encoder to describe a distribution the operator
    never sees. Prefer the per-subject form for training and keep this one for probes.

    `student_reps=None` uses ALL `R` repetitions, which is the DEPLOYED readout -- the pooled query
    metric. That is deliberate: the measured gap is a subject-level bias that no amount of pooling
    removes, so the term must be scored on the pooled quantity the operator actually consumes.
    Passing an int restricts the student to the first `student_reps` repetitions, which is the
    single-view arm used to read the discriminating prediction (does the lever reach the deployed
    estimator, or only the noisy one?) rather than an alternative recipe.
    """
    if z_rep.dim() != 3:
        raise ValueError(f"z_rep must be (N, R, d), got {tuple(z_rep.shape)}")
    n_rows, R, _d = z_rep.shape
    if z_img.shape[0] != n_rows:
        raise ValueError(f"z_img has {z_img.shape[0]} rows but z_rep has {n_rows}")
    if grp.shape[0] != n_rows:
        raise ValueError(f"grp has {grp.shape[0]} entries but z_rep has {n_rows} rows")

    r_stu = int(R if student_reps is None else min(int(student_reps), R))
    if r_stu < 1:
        raise ValueError(f"student_reps must leave at least one repetition, got {r_stu}")
    dg, uniq = gallery_metric(z_img, grp)
    n_c = int(uniq.numel())
    if n_c < int(min_concepts):
        zero = z_rep.new_zeros(())
        return zero, ({"bias_anchor_pairs": 0, "bias_anchor_agreement": float("nan"),
                       "bias_anchor_student_rank": float("nan")} if return_diag else {})
    # Pool rows sharing a concept (and the selected repetitions) so that the student metric is
    # `(n_concepts, n_concepts)` and therefore comparable with `dg`. Pooling ACROSS subjects here is
    # what makes this form an aggregate diagnostic rather than the deployed objective -- deployment
    # scores one subject against the gallery, which is `metric_anchor_loss_by_subject`.
    _uniq_g, inv = torch.unique(grp, sorted=True, return_inverse=True)
    emb = z_rep.new_zeros((n_c, z_rep.shape[-1]))
    emb.index_add_(0, inv, z_rep[:, :r_stu].mean(dim=1))
    cnt = z_rep.new_zeros((n_c,))
    cnt.index_add_(0, inv, torch.ones_like(inv, dtype=z_rep.dtype))
    emb = emb / cnt.clamp_min(1.0)[:, None]
    ds = _metric(emb)
    c = _corr(ds, dg)
    loss = 1.0 - c
    if not return_diag:
        return loss, {}
    with torch.no_grad():
        ev = torch.linalg.svdvals(ds)
        rank = float((ev.sum() ** 2) / (ev ** 2).sum().clamp_min(1e-12))
    return loss, {"bias_anchor_agreement": float(c.detach()),
                  "bias_anchor_student_rank": rank,
                  "bias_anchor_pairs": 1}


def metric_anchor_loss_by_subject(z_rep: torch.Tensor, z_img: torch.Tensor, grp: torch.Tensor,
                                  subject: torch.Tensor, student_reps: int | None = None,
                                  min_concepts: int = 4, return_diag: bool = True,
                                  frame: str = "raw", whiten_shrink: float = 0.1):
    """The deployed-faithful form: one metric per subject block, gallery gathered per block.

    Split out (rather than a flag inside the term) so the signature makes the requirement explicit:
    without `subject` the term cannot be scored the way deployment scores it, and a silent fallback
    to a mixed-subject metric would be a different -- and easier -- objective reported under this
    one's name.

    `frame` is the v14 fix for why the v13 twin failed: `raw` scores the metric in the encoder's own
    (anisotropy-exposed) frame, `whitened` scores it in the frame the deployed operator reads. The
    default stays `raw` so every v13 number on disk keeps its meaning.
    """
    if z_rep.dim() != 3:
        raise ValueError(f"z_rep must be (N, R, d), got {tuple(z_rep.shape)}")
    n_rows, R, _d = z_rep.shape
    if not (z_img.shape[0] == grp.shape[0] == subject.shape[0] == n_rows):
        raise ValueError("z_img / grp / subject must all have one entry per row of z_rep")
    if frame not in ("raw", "whitened"):
        raise ValueError(f"unknown frame {frame!r}; expected 'raw' or 'whitened'")
    r_stu = int(R if student_reps is None else min(int(student_reps), R))
    dg, uniq = gallery_metric(z_img, grp)

    losses, agree, ranks, pairs = [], [], [], 0
    for _s, block, n, inv in _subject_blocks(z_rep, grp, subject):
        if n < int(min_concepts):
            continue
        emb = _block_concept_emb(block, inv, n, r_stu, frame, whiten_shrink)
        # the gallery rows for THIS block's concepts: `inv` indexes into the block's own uniq of
        # ids, so map through the ids themselves to the batch-global gallery order
        ids = torch.unique(grp[subject == _s], sorted=True)
        if int(ids.numel()) != n:
            continue
        gpos = torch.searchsorted(uniq, ids)
        dg_b = dg[gpos][:, gpos]
        ds = _metric(emb)
        c = _corr(ds, dg_b)
        losses.append(1.0 - c)
        pairs += 1
        with torch.no_grad():
            agree.append(float(c))
            ranks.append(_eff_rank(ds))
    if not losses:
        return z_rep.new_zeros(()), (_empty_diag() if return_diag else {})
    loss = torch.stack(losses).mean()
    if not return_diag:
        return loss, {}
    return loss, {"bias_anchor_agreement": sum(agree) / len(agree),
                  "bias_anchor_student_rank": sum(ranks) / len(ranks),
                  "bias_anchor_pairs": pairs}


def subject_consensus_anchor_loss(z_rep: torch.Tensor, z_img: torch.Tensor, grp: torch.Tensor,
                                  subject: torch.Tensor, min_concepts: int = 4,
                                  return_diag: bool = True, frame: str = "whitened",
                                  whiten_shrink: float = 0.1,
                                  include_gallery: bool = True):
    """v14 C-ANCHOR: agree with the OTHER SUBJECTS' metric, not only with the gallery's.

    WHY THIS IS A DIFFERENT OBJECT FROM `metric_anchor_loss_by_subject`. That term's anchor is one
    gallery view. The decisive bottleneck probe (job 650135) measured that the movable quantity is
    SUBJECT-specific: a 9-subject pooled EEG metric reaches `corr(D, D_g) = 0.796` while one
    subject's own 80 repetitions reach `0.616`, `+0.180`, `t = +22.2`, 10/10 folds. Nine ENCODINGS
    beat one subject's repetitions. A single gallery anchor cannot express "subjects must agree";
    this term can, because the anchor is a CONSENSUS over the other subjects present in the batch.

    RICHER, NOT REDUCED. The one monotone lesson of this project is that the structural term is
    better with a RICHER object (`+3.85pp` whitened vs `+2.50pp` unwhitened) and that every
    REDUCTION (P1 truncation, A4 binarised graph) lost. A mean over the gallery plus `subj_sel - 1`
    other subjects is strictly richer than the gallery alone.

    WHY IT DOES NOT REOPEN v12's COLLAPSE CHANNEL. The gallery term is detached, exogenous and
    full-rank, and it is INCLUDED in the consensus with non-zero weight, so a solution that makes
    every subject's metric agree by collapsing them to rank 1 would lose the gallery agreement it
    still has to satisfy. That is an argument, not a measurement, so `*_student_rank` is returned
    for every subject and the caller guards on it -- `include_gallery: false` is exposed precisely
    so that arm can be MEASURED rather than assumed to be safe.

    Concepts are aligned by ID (`searchsorted` on the shared stimulus ids), never by row stride: the
    concept axis is the one whose permutation turned M1's `+15.4pp` into `-14.5pp`.
    """
    if z_rep.dim() != 3:
        raise ValueError(f"z_rep must be (N, R, d), got {tuple(z_rep.shape)}")
    n_rows, R, _d = z_rep.shape
    if not (z_img.shape[0] == grp.shape[0] == subject.shape[0] == n_rows):
        raise ValueError("z_img / grp / subject must all have one entry per row of z_rep")
    if frame not in ("raw", "whitened"):
        raise ValueError(f"unknown frame {frame!r}; expected 'raw' or 'whitened'")

    per: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for s, block, n, inv in _subject_blocks(z_rep, grp, subject):
        ids = torch.unique(grp[subject == s], sorted=True)
        if int(ids.numel()) != n:
            continue
        per[s] = (ids, _block_concept_emb(block, inv, n, R, frame, whiten_shrink))
    if len(per) < 2:
        return z_rep.new_zeros(()), (_empty_diag("cs") if return_diag else {})

    # concepts common to EVERY subject in the batch; the batch layout gives each subject the same
    # stimuli, but this is computed rather than assumed so a sampler change degrades to skipping.
    common = per[min(per)][0]
    for s in per:
        common = common[torch.isin(common, per[s][0])]
    if int(common.numel()) < int(min_concepts):
        return z_rep.new_zeros(()), (_empty_diag("cs") if return_diag else {})

    dg, uniq = gallery_metric(z_img, grp)
    gpos = torch.searchsorted(uniq, common)
    dg_c = dg[gpos][:, gpos]
    losses, agree, ranks, pairs = [], [], [], 0
    for s in sorted(per):
        ids_s, emb_s = per[s]
        emb_s = emb_s[torch.searchsorted(ids_s, common)]
        parts = [dg_c] if include_gallery else []
        for t in sorted(per):
            if t == s:
                continue
            ids_t, emb_t = per[t]
            parts.append(_metric(emb_t[torch.searchsorted(ids_t, common)]))
        if len(parts) < 2:
            continue
        anchor = torch.stack(parts).mean(dim=0).detach()
        ds = _metric(emb_s)
        c = _corr(ds, anchor)
        losses.append(1.0 - c)
        pairs += 1
        with torch.no_grad():
            agree.append(float(c))
            ranks.append(_eff_rank(ds))
    if not losses:
        return z_rep.new_zeros(()), (_empty_diag("cs") if return_diag else {})
    loss = torch.stack(losses).mean()
    if not return_diag:
        return loss, {}
    return loss, {"cs_agreement": sum(agree) / len(agree),
                  "cs_student_rank": sum(ranks) / len(ranks),
                  "cs_pairs": pairs}
