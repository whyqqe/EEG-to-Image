"""Contrastive objectives.

Two distinct contrasts are used, and keeping them as two functions (rather than one
function called twice with different flags) is deliberate: they answer different
questions and their parameterisation differs.

  * ``clip_alignment_loss``  -- EEG vs frozen image/text target. "Is this EEG in the
    right *semantic* place?"  Symmetric InfoNCE on the diagonal.
  * ``cross_subject_loss``   -- EEG vs EEG, same stimulus, different subjects.
    "Do two subjects who saw the same picture land in the same place?"
    Multi-positive InfoNCE over stimulus groups (SCORE Eq. 1).

The multi-positive mask is load-bearing: with 9 subjects tiled per stimulus, the
diagonal-only objective treats co-stimulus rows as *negatives* and spends gradient
pushing apart subjects who looked at the same image. ``groups=None`` degenerates to
the pairwise loss exactly, which is what makes the multi-positive setting a
single-variable ablation rather than a scale change.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


#: The learnable temperature is clamped to this range, matching CLIP's own
#: `logit_scale.clamp(0, log(100))` (i.e. inverse temperature in ``[1, 100]``).
#:
#: The LOWER bound is what matters and it is not cosmetic. `d(loss)/d(scale)` is the
#: mean over off-diagonal logits, i.e. an O(1) quantity that carries no learning
#: signal about the temperature itself. Once it happens to be negative, Adam --
#: which normalises by the gradient magnitude -- drives the scale to 0 within a few
#: dozen steps, no matter how small that gradient is. At scale 0 every logit is 0,
#: so the contrastive loss equals ``ln(N)`` exactly (the best value reachable
#: *without* representing anything) and, because the encoder's gradient is
#: proportional to the scale, both alignment terms stop training the encoder while
#: the non-contrastive terms (variance, HSIC, MMD) keep moving. The run then looks
#: like a plateau rather than a failure: measured here as `img` pinned at
#: `ln(72) = 4.2767` and `cross` at `4.15` for 12k consecutive steps while test
#: top-1 never left its initialisation value.
SCALE_MIN = 1.0
SCALE_MAX = 100.0


def effective_logit_scale(logit_scale: torch.Tensor, softplus: bool = True) -> torch.Tensor:
    """Bounded inverse temperature from a raw parameter.

    Extracted from `InfoNCE` so that every contrast in the project sharpens through the
    SAME clamp. That matters because the floor is load-bearing in a way that is only
    visible in a long run (see `SCALE_MIN`), and a second contrast that re-implemented
    "softplus, then clamp" would drift from this one the first time either was edited.
    It is also what `PrototypeEMA` uses: its logits are cosines in ``[-1, 1]``, so
    without a scale its softmax is uniform and the term cannot train at all.
    """
    s = F.softplus(logit_scale) if softplus else logit_scale.exp()
    return s.clamp(min=SCALE_MIN, max=SCALE_MAX)


def csls_correct(sim: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Differentiable CSLS hubness correction on a similarity matrix.

    ``s'_ij = 2*s_ij - r_q(i) - r_g(j)``, where ``r_q(i)`` is the mean of row ``i``'s
    ``k`` largest similarities and ``r_g(j)`` the mean of column ``j``'s ``k`` largest.
    This is the exact form of :func:`samclip.calibration.csls_scores` (``k=10``), rewritten
    in torch so it can sit INSIDE the training objective.

    WHY THIS EXISTS. Post-hoc calibration is exhausted on this project's representation:
    the whole coordinate family -- whitening, SAW, adaptive CSLS, learned mappings, and the
    label-ORACLE diagonal reweighting -- was measured to land at or BELOW plain centring
    (21.0/20.0/19.5 against 21.5/19.5/20.5 across three seeds). So a gain from CSLS cannot
    be collected after training any more; the only way left to have it is to optimise the
    encoder IN the CSLS metric, which is what SCORE's "recovery-aware training" does and
    where its 26.22 -> 53.23 comes from. That requires the operator to be differentiable.

    Transposition is consistent, which is what keeps the symmetric InfoNCE valid:
    ``csls_correct(S).T == csls_correct(S.T)``, because the per-row term transposes to a
    per-column term and the formula is symmetric in the two. So correcting once and reading
    the transpose is the same as correcting the transpose -- no double correction is
    needed in :meth:`InfoNCE.forward`, and applying one would be a silent second
    subtraction of the neighbourhood density.

    The ``topk`` selection is a subgradient (gradient flows to the selected entries, as
    with ``max``), which is the standard treatment and is what makes hubness itself a
    differentiable quantity: a target that is everyone's neighbour has a high ``r_g``, so
    lowering its corrected logit is how the encoder is pushed off relying on it.

    Returns ``sim`` unchanged when the matrix is too small to have a neighbourhood
    (``< 2`` rows or columns), which keeps the degenerate batch inert rather than NaN.
    """
    if sim.dim() != 2:
        raise ValueError(f"expected a (Q, G) similarity matrix, got {tuple(sim.shape)}")
    if sim.shape[0] < 2 or sim.shape[1] < 2:
        return sim
    kq = max(1, min(int(k), sim.shape[1]))
    kg = max(1, min(int(k), sim.shape[0]))
    r_g = sim.topk(kq, dim=1).values.mean(dim=1, keepdim=True)
    r_q = sim.topk(kg, dim=0).values.mean(dim=0, keepdim=True)
    return 2.0 * sim - r_g - r_q


class InfoNCE(nn.Module):
    """Symmetric contrastive loss with a learnable, bounded temperature.

    ``softplus`` keeps the objective ~5x softer at the same initialisation than the
    bare ``exp`` parameterisation, which is a strong regulariser for a high-capacity
    encoder on a small EEG set (SAMGA's ``--softplus`` flag).
    """

    def __init__(self, init_temp: float = 0.07, softplus: bool = True,
                 learnable: bool = True) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / init_temp)), requires_grad=learnable)
        self.softplus = bool(softplus)

    def effective_scale(self) -> torch.Tensor:
        """Bounded inverse temperature. Logging this is how a collapse is spotted."""
        return effective_logit_scale(self.logit_scale, self.softplus)

    def forward(self, a: torch.Tensor, b: torch.Tensor,
                groups: torch.Tensor | None = None,
                csls_k: int | None = None) -> torch.Tensor:
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        cos = a @ b.t()
        # RECOVERY-AWARE LOGITS. When set, the correlation is CSLS-corrected BEFORE the
        # temperature, i.e. the contrast is computed in the deployed metric rather than in
        # raw cosine. This is the whole of the "train in the metric you are scored in"
        # change; note it must not be scaled before the correction, because CSLS is defined
        # on the cosine (the temperature would rescale `r_q`/`r_g` by the same factor and
        # leave the fraction they remove different).
        if csls_k is not None:
            cos = csls_correct(cos, k=int(csls_k))
        logits = self.effective_scale() * cos
        if groups is None:
            labels = torch.arange(a.shape[0], device=a.device)
            return 0.5 * (F.cross_entropy(logits, labels)
                          + F.cross_entropy(logits.t(), labels))
        return _multi_positive(logits, groups)


def _multi_positive(logits: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE where rows sharing a group are all positives."""
    mask = groups[:, None] == groups[None, :]
    size = mask.sum(dim=1)
    if bool((size == 0).any()):
        raise ValueError("multi-positive mask has an empty group; every row is its "
                         "own positive, so this means `groups` was built from the "
                         "wrong tensor")
    logp = F.log_softmax(logits, dim=1)
    e2i = -(logp * mask).sum(dim=1).div(size).mean()
    logp_t = F.log_softmax(logits.t(), dim=1)
    mask_t = mask.t()
    i2e = -(logp_t * mask_t).sum(dim=1).div(mask_t.sum(dim=1)).mean()
    return 0.5 * (e2i + i2e)


def clip_alignment_loss(z_eeg: torch.Tensor, z_target: torch.Tensor,
                        criterion: InfoNCE, csls_k: int | None = None) -> torch.Tensor:
    """Align EEG against a frozen image/text target, one target per row.

    The cross-subject batch hands one stimulus to `g` subjects, so `g` rows carry the
    *same* target vector. It is worth recording why the diagonal form is nonetheless
    exactly right here, because "several rows share a target, so this needs a
    multi-positive mask" is the natural instinct and it is wrong -- and because the
    reason it is wrong is a proof, not an empirical accident.

    Write the matrix of a single group's block as `A[i, j] = <z_e[i], t_j>`. Co-stimulus
    columns are equal vectors, so `A[i, j] = f(i)` does not depend on `j` inside the
    group. Then

      * `eeg -> image`: the `g` equal positive columns each hold `1/g` of the group's
        softmax mass, and `sum_{j} (1/g) . dlogit_j = df(i)` cancels the `-df(i)` term
        exactly, so the shared target contributes no gradient at all.
      * `image -> eeg`: `sum_{j in G} mean_{i in G} A[i, j] = sum_{j in G} f(j) =
        sum_{j in G} A[j, j]`, so averaging over the group members returns the same
        number as taking each row's own diagonal entry.

    Both halves of the grouped and diagonal losses are therefore *identical*, not merely
    close: the mask is a no-op for this layout. It becomes a real relaxation only when
    the group's targets differ, i.e. when the batch does not tile a stimulus across
    subjects in a way that duplicates columns -- `images_per_pair > 1`, or a
    subject-specific target (`target_fusion: routed_sr`). `smoke_test.py` 9f pins this
    equality; if it ever fires, the layout changed and the mask has to be reconsidered
    rather than assumed.

    One consequence is worth stating because it is the thing that cannot be masked away:
    with `g` equivalent positives, both halves bottom out at `ln g` (`log(g e^l) - l`),
    so the image contrast is saturated at `ln 9` on this layout no matter what mask is
    used. Sharpening cannot help either, which is exactly why the temperature must be
    floored rather than allowed to run to zero (see `SCALE_MIN`).
    """
    return criterion(z_eeg, z_target, csls_k=csls_k)


def cross_subject_loss(z_eeg: torch.Tensor, stimulus: torch.Tensor,
                       criterion: InfoNCE, csls_k: int | None = None) -> torch.Tensor:
    """Align EEG representations of the SAME stimulus across different subjects.

    `stimulus` is the global stimulus id per row (concept * n_images + slot). Rows
    with equal ids are positives, including the same subject's own repetition -- the
    latter is a useful within-subject invariance signal, not a bug.
    """
    groups = torch.unique(stimulus, return_inverse=True)[1].to(z_eeg.device)
    return criterion(z_eeg, z_eeg, groups=groups, csls_k=csls_k)


def repetition_collapse_loss(z_reps: torch.Tensor, criterion: InfoNCE,
                             csls_k: int | None = None,
                             groups: torch.Tensor | None = None) -> torch.Tensor:
    """Align the ``R`` repetitions of each row, grouped by ``groups`` (v7 T2' / T2'').

    ``z_reps`` is ``(B, R, d)``: the encoder's embeddings of the ``R`` UNAVERAGED
    repetitions of each row in the batch. ``groups`` is a length-``B`` tensor giving each
    row's group, and it is the whole design decision:

      * ``groups=None`` -- one group PER ROW. The ``R`` repeats of a row are positives of
        each other and every other row's repeats are negatives. **This was the first
        implementation and it is WRONG on a cross-subject batch**, because the batch tiles
        9 subjects per stimulus (``CrossSubjectBatchSampler``), so the negatives include
        other subjects' embeddings of the SAME image. The term then actively pushes apart
        exactly what :func:`cross_subject_loss` pulls together: it is an anti-T1 term.
        Measured consequence on sub-08/seed2025 -- the encoder's cross-subject geometry
        degraded, the SMN's scale gate decayed 0.498 -> 0.203 -> 0.000 as the weight went
        0 -> 0.1 -> 0.5, and Top-1 fell 42.0 -> 37.5 -> 30.0. A clean monotone
        dose-response in the WRONG direction, which is what identified the conflict.
      * ``groups=stimulus`` -- one group PER STIMULUS, so the positives of an anchor are
        its own repeats AND the other subjects' views of the same image. This is a SUPERSET
        of T1's positive set, so it cannot fight T1, and it is the only form that is
        consistent with our own cross-subject objective. It also makes the term exactly
        the object we want: the encoder must put one image's every (subject, repetition)
        pair on one point.

    WHY A CONTRAST AND NOT A PULL-TO-MEAN. The obvious form -- minimise
    ``1 - cos(z_r, mean_r z)`` -- has a trivial global optimum: map every input to one
    constant. That collapse is measured on this project, not hypothetical (`L_spec` to
    rank-1, v6 §9.1; the subject-conditioning arm to cosine 0.9999). Making the repeats
    positives AGAINST a batch of negatives keeps the term a proper retrieval objective, so
    the only way to lower it is to make a group agree *relative to everything else*.

    ``csls_k`` is forwarded for the same reason ``cross_subject_loss`` forwards it -- when
    recovery-aware training is on, the term is scored in the metric deployment uses.
    """
    if z_reps.dim() != 3:
        raise ValueError(f"z_reps must be (B, R, d), got {tuple(z_reps.shape)}")
    b, r, d = z_reps.shape
    if r < 2:
        raise ValueError(
            f"repetition_collapse_loss needs R >= 2 repetitions, got {r}. With one "
            f"repetition the group has no positive and the term is a no-op that would "
            f"report a mechanism it is not using.")
    flat = z_reps.reshape(b * r, d)
    if groups is None:
        g = torch.arange(b, device=z_reps.device).repeat_interleave(r)
    else:
        g = torch.as_tensor(groups, device=z_reps.device).reshape(-1)
        if g.shape[0] != b:
            raise ValueError(
                f"groups must have one entry per row of z_reps ({b}), got {g.shape[0]}; "
                f"a length mismatch here would group the repeats of DIFFERENT rows "
                f"together and train a mechanism that is not the one configured")
        g = g.repeat_interleave(r)
    return criterion(flat, flat, groups=g, csls_k=csls_k)


def score_fusion_loss(
    score_matrices: list[torch.Tensor],
    criterion: "InfoNCE",
    groups: torch.Tensor | None = None,
    normalize: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Sum-rule late fusion of per-route score matrices, scored by the shared contrast.

    ``score_matrices`` is a list of ``(N, N)`` similarity matrices, one per route, each
    already in whatever metric that route is trained in (raw cosine, or CSLS-corrected
    when recovery-aware is on). CORTIVA's fusion is a weighted sum of scores -- its sum
    rule -- and the weights are uniform by default (see `config.DEFAULT_FUSION_WEIGHTS`).

    ``normalize`` divides each route's matrix by its own standard deviation before
    summing. This is a GLOBAL scalar per route, not a per-row z-score: a per-row affine
    map changes each row's effective temperature and would distort the multi-positive
    structure, while a global one only equalises the routes' scales. Without it a route
    whose scores happen to have a larger spread (CSLS-corrected scores are unbounded,
    raw cosines are not) dominates the sum, which is the same failure
    `calibration.structural_scores` records as a 35.50 -> 18.00 collapse when two experts
    on incomparable scales were added unweighted.

    The fused matrix is then scored by the SAME symmetric-contrast criterion as the
    per-route terms, so the fusion is trained against the metric it is evaluated in and no
    second temperature is introduced.
    """
    if not score_matrices:
        raise ValueError("score_fusion_loss needs at least one score matrix")
    total = None
    for s in score_matrices:
        if s.dim() != 2 or s.shape[0] != s.shape[1]:
            raise ValueError(f"each fused score matrix must be square (N, N), got "
                             f"{tuple(s.shape)}")
        term = s
        if normalize:
            term = term / term.std().clamp_min(eps)
        total = term if total is None else total + term
    logits = criterion.effective_scale() * total
    if groups is None:
        labels = torch.arange(total.shape[0], device=total.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
    return _multi_positive(logits, groups)


def recovery_aware_alignment(z_eeg: torch.Tensor, z_target: torch.Tensor,
                             subject: torch.Tensor | None, criterion: InfoNCE,
                             csls_k: int | None = 10) -> torch.Tensor:
    """Image alignment **per subject block**, scored in the deployed metric.

    WHY PER SUBJECT, AND NOT ON THE WHOLE BATCH. The quantity CSLS removes is a property
    of the RETRIEVAL SET, so "which rows are in the matrix" is part of the operator, not an
    implementation detail. Deployment always scores ONE subject: its 200 queries against
    the shared gallery, with ``r_q``/``r_g`` computed over exactly those 200 subjects-worth
    of rows (:func:`samclip.calibration.csls_scores`). A batch here holds ~3 subjects, and
    their rows differ systematically (that is the entire premise of the project), so
    correcting a mixed 3-subject matrix computes a neighbourhood density no deployment
    ever sees -- and an encoder optimised under it is optimised for the wrong density.

    That is precisely the failure mode this term exists to avoid, so the block structure is
    load-bearing rather than cosmetic. It also makes the training task closer to the real
    one: a block is a ``n_s``-way retrieval instead of a ``3 * n_s``-way one, i.e. the same
    difficulty class as deployment rather than an easier one.

    Blocks smaller than 2 rows cannot form a neighbourhood and are skipped (they contribute
    no gradient rather than a degenerate one). If no block qualifies the whole-batch form is
    used, which keeps a tiny/debug batch working instead of returning 0.
    """
    if subject is None:
        return clip_alignment_loss(z_eeg, z_target, criterion, csls_k=csls_k)
    total = z_eeg.new_zeros(())
    n = 0
    for s in torch.unique(subject):
        rows = subject == s
        count = int(rows.sum())
        if count < 2:
            continue
        total = total + count * clip_alignment_loss(
            z_eeg[rows], z_target[rows], criterion, csls_k=csls_k)
        n += count
    if n == 0:
        return clip_alignment_loss(z_eeg, z_target, criterion, csls_k=csls_k)
    return total / n


def subject_supervised_contrast(z_eeg: torch.Tensor, stimulus: torch.Tensor,
                                subject: torch.Tensor, criterion: InfoNCE,
                                same_subject_negative: bool = True) -> torch.Tensor:
    """Cross-subject contrast that keeps a separate 'same subject' view.

    Optional arm: build negatives only from *other* stimuli, and report the
    cross-subject positive term separately. Useful when diagnosing whether the
    alignment is being carried by within-subject invariance instead of the
    cross-subject signal we actually want.
    """
    groups = torch.unique(stimulus, return_inverse=True)[1].to(z_eeg.device)
    sim = F.normalize(z_eeg, dim=-1) @ F.normalize(z_eeg, dim=-1).t()
    pos = groups[:, None] == groups[None, :]
    if same_subject_negative:
        return _multi_positive(sim, groups)
    # mask out all same-subject pairs that are not the same stimulus
    same_subject = subject[:, None] == subject[None, :]
    neg_mask = (~pos) & same_subject
    logits = sim.masked_fill(neg_mask, float("-inf"))
    return _multi_positive(logits, groups)
