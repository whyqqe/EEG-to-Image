"""POLARIS losses: the multi-positive fix, L_spec and L_aug.

`ClipLoss_Modified_DDP` is vendored from `third_party/CognitionCapturerPro/src/cogcappro/
utils.py:222-320`. Its soft-label construction is

    mask_sim  = the top-`top_k` rows by *target-target* cosine similarity (plus diagonal)
    mask_class = rows whose `img_index` matches
    sim_mask  = mask_sim * mask_class          <- INTERSECTION
    labels    = sim_mask / row_sums

and this file replaces the intersection with a union. Why, precisely:

`img_index` in CogCapPro is a *global* image identity built across all subjects
(`data/eeg.py:157-164` populates `img_path_to_idx` from every subject's list), so with S
source subjects each stimulus appears S times in the dataset. Those S-1 cross-subject
copies are the positives that make the loss an inter-subject objective at all.

The intersection only admits a positive if it *also* lands in the top-`top_k` by target
similarity. Whether that happens depends on how many rows tie at the maximum target
similarity, which is not a controlled quantity:

  * If the target features are per-image and distinct, a stimulus's S-1 siblings are the
    S-1 most similar rows and, for S-1 <= top_k = 10, none are lost.
  * If several stimuli share a target (identical captions, or a target that does not
    separate instances), the tied block is larger than `top_k` and the mask keeps an
    arbitrary `top_k` slice of it -- so positives *are* dropped, and the dropped ones are
    exactly the rows with the largest logits, i.e. the largest gradient contribution.

Rather than assert which regime the data is in, `clip_loss_multi_positive` computes both
masks and reports the upstream intersection count as a diagnostic. The repair is to stop
gating positives by similarity entirely (`repair_topk=True` -> `weights = pos`).

That is deliberately **not** "union with the hard negatives", which was this file's first
formulation and is wrong. The log-softmax denominator already contains every row, so
`top_k` is not selecting negatives -- negatives need no selecting. What the intersection
does is *restrict the positive set*, and a `mask_sim & ~pos` row is a row whose target looks
like the query but which belongs to a **different stimulus**; promoting it into `weights`
would teach the model to confuse two different images. So `hard` is computed only to report
how much of the `top_k` budget was spent on non-siblings.

Design doc §2.4 states the truncation mechanism with numbers ("36 rows per image, 25 pushed
away") derived from 4 repetitions x 9 subjects. That arithmetic is wrong: with CogCapPro's
`train_avg: True` (`configs/cogcappro.yaml:35`) the 4 repetitions are averaged away in
`load_data` (`data/eeg.py:280-288`) before the loss sees them, so the per-stimulus row count
is S, not 4S. The conclusion (positives can be dropped, and the diagnostic is worth
reporting) survives; the number does not. `test_cogcap.py` measures both regimes so the
claim cannot quietly come back at its original strength.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def stimulus_groups(img_index: torch.Tensor) -> torch.Tensor:
    """Compress arbitrary stimulus ids to 0..n-1 preserving identity."""
    uniq, inv = torch.unique(img_index, return_inverse=True)
    return inv


def clip_loss_multi_positive(eeg_z, target_z, logit_scale, img_index, top_k=10,
                            cos_batch=512, repair_topk=True, want_diag=False):
    """Symmetric contrastive loss with cross-subject multi-positive soft labels.

    `repair_topk=True` is the POLARIS fix (union); `False` reproduces upstream's
    intersection and is kept so the two can be compared in one run.
    """
    device = eeg_z.device
    eeg_z = F.normalize(eeg_z, dim=1) if eeg_z.norm(dim=1).median() > 1.5 else eeg_z

    logits_per_eeg = logit_scale * eeg_z @ target_z.t()
    logits_per_tgt = logit_scale * target_z @ eeg_z.t()

    n = eeg_z.shape[0]
    diag = torch.eye(n, dtype=torch.bool, device=device)
    pos = img_index[:, None] == img_index[None, :]            # includes diagonal

    with torch.no_grad():
        tn = F.normalize(target_z, p=2, dim=1)
        sim = tn @ tn.t()
        sim = sim.masked_fill(diag, 0.0)
        k = min(top_k, n - 1)
        topk_idx = torch.topk(sim, k=k, dim=1, sorted=False).indices
        mask_sim = torch.zeros_like(sim, dtype=torch.bool)
        mask_sim.scatter_(1, topk_idx, True)
        mask_sim = mask_sim | diag

        hard = mask_sim & ~pos                                     # non-positive, near target
        assert not bool((hard & pos).any())
        if repair_topk:
            # Repair: every same-stimulus row is a positive, none is filtered by similarity.
            weights = pos
        else:
            weights = mask_sim & pos                               # upstream intersection
        weights = weights.to(eeg_z.dtype)
        row = weights.sum(dim=1, keepdim=True)
        labels = weights / torch.where(row == 0, torch.ones_like(row), row)

        diag_info = {
            "n_pos_mean": float(pos.sum(dim=1).float().mean()) if want_diag else 0.0,
            "n_pos_kept_mean": float((weights * pos).sum(dim=1).float().mean()) if want_diag else 0.0,
            "n_hard_total_mean": (float(hard.sum(dim=1).float().mean()) if want_diag else 0.0),
            "n_pos_dropped": (
                float((pos & ~(mask_sim & pos)).sum(dim=1).float().mean()) if want_diag else 0.0
            ),
        }

    e2t = -(F.log_softmax(logits_per_eeg, dim=1) * labels).sum(dim=1).mean()
    labels_t = labels.t()
    t2e = -(F.log_softmax(logits_per_tgt, dim=1) * labels_t).sum(dim=1).mean()
    loss = 0.5 * (e2t + t2e)
    return loss, logits_per_eeg, diag_info


# ------------------------------------------------------------------ L_spec
def _aligned_subject_matrices(z, stim, subj):
    """Per-subject [n, d] matrices over the stimuli that every subject shares.

    Requires the averaged-row regime (each (subject, stimulus) pair appears once), which
    is what `train_avg: True` gives; a repeated pair would silently be folded by the
    index_add below into a mean, which is still well defined but no longer a raw row.
    """
    subs = [int(s) for s in torch.unique(subj).tolist() if s >= 0]
    per = {}
    for s in subs:
        m = subj == s
        zs, ss = z[m], stim[m]
        per[s] = (zs, ss)
    common = None
    for s in subs:
        u = torch.unique(per[s][1])
        common = u if common is None else common[torch.isin(common, u)]
    if common is None or common.numel() < 2:
        return None
    mats = []
    for s in subs:
        zs, ss = per[s]
        order = torch.argsort(ss)
        zs, ss = zs[order], ss[order]
        loc = torch.searchsorted(ss, common)
        loc = loc.clamp(max=ss.numel() - 1)
        if not bool((ss[loc] == common).all()):
            return None
        mats.append(zs[loc])
    return torch.stack(mats, dim=0)                            # [S, n, d]


def spectral_flatness_loss(z, stim, subj, rank=64):
    """L_spec (design doc condition C3).

    Penalises anisotropy of the cross-subject consensus mapping

        M = mean_s  X_s^T  mean_{s' != s} X_{s'}

    by pushing the normalised top-`rank` eigenvalues of M^T M towards 1/rank. This is the
    moment condition whose violation is what makes a deployment-time *orthogonal* recovery
    biased (an anisotropic subject operator cannot be absorbed by any rotation). The
    gradient-friendly form ||G/||G||_F - I/d||_F^2 is used rather than an SVD, because SVD
    gradients are unstable at repeated singular values and EEG spectra are nearly flat by
    construction -- i.e. repeated values is the generic case here, not the corner case.
    """
    mats = _aligned_subject_matrices(z, stim, subj)
    if mats is None:
        return None
    S = mats.shape[0]
    if S < 2:
        return None
    sub_means = mats.mean(dim=0)                               # [n, d]
    total = 0.0
    for s in range(S):
        others = (mats.sum(dim=0) - mats[s]) / (S - 1)
        xs = mats[s] - mats[s].mean(dim=0, keepdim=True)
        ys = others - others.mean(dim=0, keepdim=True)
        m = xs.t() @ ys                                        # [d, d]
        g = m.t() @ m
        g = g / (torch.linalg.matrix_norm(g) + 1e-12)
        ev = torch.linalg.eigvalsh(g)
        top = ev[-rank:]
        top = top / (top.sum() + 1e-12)
        target = torch.full_like(top, 1.0 / rank)
        total = total + ((top - target) ** 2).sum() * rank
    del sub_means
    return total / S


# ------------------------------------------------------------------ L_aug
def random_rotation(n_channels: int, batch: int, device, generator=None):
    """Haar-ish random rotation via QR of a Gaussian, sign-corrected to be uniform on O(n)."""
    a = torch.randn(batch, n_channels, n_channels, device=device, generator=generator)
    q, r = torch.linalg.qr(a)
    d = torch.diagonal(r, dim1=-2, dim2=-1)
    q = q * torch.sign(d).unsqueeze(-2)
    return q


def rotation_aug_loss(model_forward, eeg, subject_ids, stim, subj, n_channels,
                      generator=None, weight_by_subject=None):
    """L_aug: concept structure must survive an arbitrary sensor-space rotation.

    Compared on the *rank structure* (RSM), not on the representations. Element-wise
    invariance would be satisfied by an encoder that discards concept information
    altogether; the RSM only asks that "which stimuli are similar" is preserved, which is
    what the downstream retrieval and generation actually consume.

    Only one subject's rows are used (the most frequent), because the RSM is only
    meaningful among rows that are independent samples of the concept space; pooling
    subjects would inject the subject offset into the similarity structure being compared.
    """
    B = eeg.shape[0]
    R = random_rotation(n_channels, B, eeg.device, generator=generator)
    eeg_aug = torch.einsum("bcd,bdt->bct", R, eeg)

    z_clean = model_forward(eeg, subject_ids)
    z_aug = model_forward(eeg_aug, subject_ids)

    counts = torch.bincount(subj.clamp(min=0), minlength=int(subj.max().item()) + 1)
    s0 = int(torch.argmax(counts).item())
    m = subj == s0
    if int(m.sum()) < 4:
        return None

    a = F.normalize(z_clean[m], dim=1)
    b = F.normalize(z_aug[m], dim=1)
    ra = a @ a.t()
    rb = b @ b.t()
    n = ra.shape[0]
    off = ~torch.eye(n, dtype=torch.bool, device=ra.device)
    return ((ra - rb)[off] ** 2).mean()


# ------------------------------------------------------------------ augmentation
def sample_modality_mask(n_modalities: int, mask_count: int, device, generator=None):
    """`training/module.py:164-194`: zero out 0..mask_count non-EEG modalities, >=1 kept."""
    actual = max(0, min(mask_count, n_modalities - 1))
    if actual == 0:
        return set()
    perm = torch.randperm(n_modalities, device=device, generator=generator)[:actual]
    return set(int(i) for i in perm.tolist())


class FixedTemperature(nn.Module):
    """Kept for parity with the upstream softplus(log(1/0.07)) scale."""
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * float(torch.log(torch.tensor(1 / 0.07))))
        self.softplus = nn.Softplus()

    def forward(self):
        return self.softplus(self.logit_scale)
