"""v12 training terms: make each repetition carry what the whole repetition cloud carries.

THE CLAIM THIS IMPLEMENTS (docs/eeg2image_v12_core_claim.md). Cross-subject EEG-to-image is
mis-specified as inter-subject *alignment*: inter-subject maps explain 2.4% and do not compose
(M6/M7), and apparent alignment gains invert sign under a concept-permutation control (M1). What
actually matters is refinement of the query's own concept metric against the gallery. The right
object is the SHARED CONCEPT METRIC estimated from several independent noisy views. This module
turns that estimator into a TRAINING OBJECTIVE, so the improvement happens in the representation
rather than in a post-hoc operator -- which v11 measured to be insufficient (structure-SNR +0.038
converted to only +0.85pp of retrieval, t = 1.72).

WHY DISTILLATION AND NOT ANOTHER CONTRASTIVE TERM. The deployed T2 operator is refit on the first R
repetitions; its Top-1 rises 4.0 -> 54.5 from R=1 to R=80 and is STILL climbing by +3.33pp in the
last doubling (job 645719, 3/3 subjects). So the query metric is estimation-limited at the deployed
R and no existing term targets that quantity. The four earlier training attempts (T2', T2'', the
SCORE episode, G-a) all targeted the HARD mutual-NN landmark rate, which the operator does not
consume -- which is why their contribution measured FLAT (corr with raw = -0.19, 30 runs). This term
targets the metric itself, i.e. exactly the object whose poverty the R-curve measures.

THE TEACHER IS THE v11 ESTIMATOR, NOT AN ARBITRARY TARGET. `metric_self_distill` builds the teacher
by splitting the repetitions into `teacher_blocks` independent blocks, scoring a metric per block,
and fusing them with leave-one-out reliability weights -- the exact estimator `scripts/probe_fusion.py`
measured to carry shared structure monotonically in the number of views with a
correspondence-destroying control pinned at zero (job 645697). The student is the metric from a
SINGLE repetition. The teacher EXCLUDES the student's repetitions, so the target cannot be satisfied
by copying the teacher's input.

THE FAILURE MODE IS COLLAPSE, AND IT IS MEASURED ON THIS PROJECT. Two earlier consistency terms
degenerated (L_spec to rank-1; the subject-conditioning arm to cosine 0.9999). Everything here is
therefore a CORRELATION against a stop-gradient target, never a pull-to-a-point, and each term
returns diagnostics (`*_rank`, `*_agreement`) so a degenerate "win" is visible rather than assumed.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _metric(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Standardised squared chordal distance. EXACT twin of `calibration._sq_cos_dist`.

    Standardised (zero mean, unit scale) so that a correlation between two metrics is their
    elementwise inner product, and so that a metric's OVERALL scale cannot be a shortcut for the
    distillation: the student cannot lower the loss by matching a scale, only a PATTERN.

    `unbiased=False` IS NOT COSMETIC. `torch.std` defaults to the SAMPLE std (ddof=1) while
    `numpy.std` -- and therefore `calibration._sq_cos_dist` -- is the POPULATION std (ddof=0). The
    two differ by `sqrt(n/(n-1))`, and because `_corr` re-standardises, the LOSS is unchanged
    either way; but "twin" then held only up to a scale, and a smoke assertion pinning the
    training metric to the deployed one could not be written. Fixing the ddof makes the identity
    exact, which is the property that actually matters: the training anchor must be scored on the
    SAME object the deployed operator consumes.
    """
    n = F.normalize(x, dim=-1, eps=1e-9)
    d = (n @ n.t() - 1.0) ** 2
    return (d - d.mean()) / d.std(unbiased=False).clamp_min(eps)


def _corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Correlation over the upper triangle (the metric is symmetric with a zero diagonal)."""
    iu = torch.triu_indices(a.shape[0], a.shape[1], offset=1, device=a.device)
    x = a[iu[0], iu[1]]
    y = b[iu[0], iu[1]]
    x = x - x.mean()
    y = y - y.mean()
    return (x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-8)


def _subject_blocks(z: torch.Tensor, grp: torch.Tensor, subject: torch.Tensor):
    """Iterate (subject_id, (n_stim, R, d) block) with a CONSISTENT stimulus order per subject.

    The selected rows are stimulus-major (n_stim stimuli x subj_sel subjects each), but the block is
    rebuilt from `grp`/`subject` rather than by assuming that stride: an assumed stride is exactly
    the kind of layout coupling that silently mis-pairs concepts when the sampler changes, and the
    concept index is the one thing that must never be wrong here (it is the axis whose permutation
    turned M1's +15.4pp into -14.5pp).
    """
    for s in subject.unique():
        mask = (subject == s)
        ids = grp[mask]
        uniq, inv = torch.unique(ids, sorted=True, return_inverse=True)
        n = int(uniq.numel())
        if n < 3:
            continue
        yield int(s), z[mask], n, inv


def metric_self_distill(z_rep: torch.Tensor, grp: torch.Tensor, subject: torch.Tensor,
                        student_reps: int = 1, teacher_blocks: int = 4,
                        return_diag: bool = False):
    """Distil the multi-view consensus metric into a FEW-repetition metric, per subject.

    `z_rep` is `(N, R, d)` (the encoder's embeddings of the `R` unaveraged repetitions of the `N`
    selected rows); `grp` is the stimulus id per row; `subject` the subject id per row.

    Returns `(loss, diag)`. `loss = mean_s [ 1 - corr(student_s, teacher_s) ]` with the teacher
    detach'd, and `diag` carries the per-subject agreement and rank so a collapse is visible.
    """
    if z_rep.dim() != 3:
        raise ValueError(f"z_rep must be (N, R, d), got {tuple(z_rep.shape)}")
    R = int(z_rep.shape[1])
    r_stu = int(min(student_reps, R - 2))
    if r_stu < 1:
        raise ValueError(
            f"metric_self_distill needs R >= 3 repetitions (student>=1, teacher>=2), got R={R}. "
            f"Otherwise the teacher has no independent view and the term would be a no-op "
            f"reported as a mechanism.")
    losses, agree, ranks = [], [], []
    for _s, block, n, _inv in _subject_blocks(z_rep, grp, subject):
        stu = _metric(block[:, :r_stu].mean(dim=1))          # (n, n)
        tail = block[:, r_stu:]                              # (n, R-r_stu, d) -- disjoint from stu
        Rt = int(tail.shape[1])
        K = int(min(max(teacher_blocks, 1), Rt))
        edges = torch.linspace(0, Rt, K + 1).to(tail.device).long()
        mats = torch.stack([_metric(tail[:, edges[b]:edges[b + 1]].mean(dim=1))
                            for b in range(K)])               # (K, n, n)
        if K > 1:
            # Reliability weights are a MEASUREMENT (leave-one-out residual), detached: they
            # reweight the target, they are not a second thing to be optimised.
            with torch.no_grad():
                rel = torch.stack([
                    (mats[b] - torch.stack([mats[j] for j in range(K)
                                            if j != b]).mean(dim=0)).norm()
                    / torch.stack([mats[j] for j in range(K)
                                   if j != b]).mean(dim=0).norm().clamp_min(1e-8)
                    for b in range(K)])
                inv = 1.0 / rel.clamp_min(1e-6)
                w = inv / inv.sum()
            tea = (w[:, None, None] * mats).sum(dim=0)
        else:
            tea = mats[0]
        tea = tea.detach()
        c = _corr(stu, tea)
        losses.append(1.0 - c)
        with torch.no_grad():
            agree.append(float(c))
            # rank of the student metric: a distillation that wins by collapsing the metric to
            # rank 1 would show up here as a vanishing rank, not as a low loss.
            ev = torch.linalg.svdvals(stu)
            ranks.append(float((ev.sum() ** 2) / (ev ** 2).sum().clamp_min(1e-12)))
    if not losses:
        zero = z_rep.new_zeros(())
        return zero, ({"metric_student_agreement": float("nan"),
                       "metric_student_rank": float("nan")} if return_diag else {})
    loss = torch.stack(losses).mean()
    if not return_diag:
        return loss, {}
    return loss, {
        "metric_student_agreement": sum(agree) / len(agree),
        "metric_student_rank": sum(ranks) / len(ranks),
        "metric_distill_subjects": len(losses),
    }


def metric_subject_consistency(z_rep: torch.Tensor, grp: torch.Tensor, subject: torch.Tensor,
                               student_reps: int = 1, return_diag: bool = False):
    """Raise the agreement of the CONCEPT METRIC across subjects (the measured 0.565 quantity).

    This is Part A's surviving signal, made trainable: inter-subject *embedding* maps are noise
    (M6/M7), but the shared concept METRIC is real (corr(D_eeg,s, D_eeg,t) = +0.565, chance -0.003).
    The term is a CORRELATION between two subjects' metrics on the SAME stimuli, both built from
    `student_reps` repetitions, so it cannot be lowered by a scale change and cannot be satisfied by
    a constant (the correlation is undefined there, so the gradient vanishes rather than rewarding
    the collapse). Non-degeneracy of the result is checked separately by `smoke_test.py` §17.

    COMPARISON IS ON THE INTERSECTION OF THE TWO SUBJECTS' STIMULI, NOT ON "IDENTICAL SETS". The
    sampler assigns a random subject triple to each stimulus, so two subjects typically see
    overlapping but different stimulus sets; requiring equal sets made the term fire on 2 of 9
    subjects on a real batch (measured in the v12 smoke, job 645751) while the loss still looked
    healthy. Intersecting makes the term use the FULL overlap, and keeps it a statement about
    shared concepts rather than about who happened to be sampled together.
    """
    blocks, ids = {}, {}
    for s, block, n, inv in _subject_blocks(z_rep, grp, subject):
        blocks[s] = _metric(block[:, :max(1, int(student_reps))].mean(dim=1))
        ids[s] = inv
    keys = sorted(blocks)
    if len(keys) < 2:
        zero = z_rep.new_zeros(())
        return zero, ({"metric_consistency_pairs": 0} if return_diag else {})
    cs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = ids[keys[i]], ids[keys[j]]
            am = torch.isin(a, b)
            bm = torch.isin(b, a)
            if int(am.sum()) < 4 or int(bm.sum()) < 4:
                continue
            # `am`/`bm` are already sorted by stimulus order within each subject (torch.unique
            # sorted=True and the batch is stimulus-major), so the two masks pick the same stimuli
            # in the same order and the correlation is over genuinely corresponding rows.
            A, B = blocks[keys[i]][am][:, am], blocks[keys[j]][bm][:, bm]
            cs.append(_corr(A, B))
    if not cs:
        zero = z_rep.new_zeros(())
        return zero, ({"metric_consistency_pairs": 0} if return_diag else {})
    stacked = torch.stack(cs)
    loss = -stacked.mean()                            # maximise agreement
    if not return_diag:
        return loss, {}
    return loss, {"metric_consistency_pairs": len(cs),
                  "metric_consistency_mean": float(stacked.mean().detach())}
