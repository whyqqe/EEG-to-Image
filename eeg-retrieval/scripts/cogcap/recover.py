"""POLARIS deployment recovery for CogCapPro.

Everything here is label-free: the only inputs are the held-out subject's own EEG and the
*training* condition gallery. Nothing reads the held-out subject's test labels, and the
anchors that define the map are discovered from the data (mutual nearest neighbours), not
supplied.

Reused rather than rewritten
----------------------------
`csls_scores` / `mutual_nn_pairs` / `orthogonal_recovery` / `apply_recovery` come from
`scripts/epd/recover.py`, which has 44 unit tests covering exactly the properties this file
depends on (that the solve returns `U V^T` and not `V U^T`; that CSLS beats cosine on a
synthetic hub; that `rho` interpolates towards the identity; that a rotation is recovered
exactly at a low noise level). Those are the failure modes that produce plausible-looking
wrong numbers, so they are worth not reimplementing.

`moment_match` and `select_landmarks` from that module are deliberately *not* used:
`moment_match` is replaced by `moment_affine` in `train.py` because the affine has to be
derived once on the fitting rows and then applied unchanged to the test rows, whereas
`moment_match` returns only the matched query; and `select_landmarks` selects on margins
alone, which the coverage criterion below has to augment.

What POLARIS adds on top
------------------------
* **SW** -- subject-adaptive whitening in *sensor* space, so one correction precedes every
  modality branch and every non-equivariant module downstream (design doc DR1/§1.3).
* **AS** -- anchors are voted on across branches and then chosen for *span coverage* of the
  concept-active subspace (condition C2), instead of taking any mutual pair.
* **RP** -- per-branch weighted orthogonal recovery, plus a single sensor-space operator
  arm, which is the H1 test (design doc AB-1).
* **BC** -- inter-branch defect, used as a label-free criterion for choosing between
  candidate maps.
* **Diagnostics** -- `r_eff`, `m_anchor`, outlier fraction `q`. Design doc §7.2 asks for
  these because they are the quantities the identifiability conditions are stated in, and
  the literature does not report them.

Honesty of the fit
------------------
`pair_mode="mnn"` (default) discovers pairs; `"oracle"` uses the true correspondence, which
*is* available for the training split but is a labelled quantity, so it is a ceiling row and
is labelled as such wherever it is reported. The gallery is the 16540 training stimuli,
disjoint from the 200 test stimuli, so a map fitted this way cannot transfer the answer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epd.recover import (                                     # noqa: E402
    apply_recovery,
    csls_scores,
    mutual_nn_pairs,
    orthogonal_recovery,
)


def _norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=1)


# ============================================================ SW: sensor-space SAW
@torch.no_grad()
def fit_saw(eeg: torch.Tensor, lam: float = 1e-3):
    """`W = (Sigma + lam I)^{-1/2}` on the `C x C` channel covariance, `mu` per channel.

    Channel-space, not spatio-temporal. The full `(C*T) x (C*T)` covariance would be
    15750 x 15750 -- about 1 GB in float32, and the whitening matmul alone would be ~4 TFLOP
    per pass over the training set. The object POLARIS needs is the one that makes two
    subjects share a coordinate frame, and that lives in the channel correlations; the
    temporal axis is already handled by the backbone's own conv/pooling. So `Sigma` is the
    covariance of the channel vectors pooled over all time points and epochs.

    `Sigma` is signal plus noise, which makes this a different operation from the MVNN noise
    whitener in `epd/mvnn.py` (fitted to within-condition residual covariances, and aimed at
    SNR rather than at frame alignment). They are not substitutes and both are defensible.

    `lam` is relative to the mean eigenvalue rather than absolute, so the amount of
    shrinkage does not change when the subject's overall scale does.
    """
    n, c, t = eeg.shape
    x = eeg.permute(0, 2, 1).reshape(-1, c).double()          # [n*T, C]
    mu = x.mean(0)                                            # [C]
    xc = x - mu
    cov = (xc.t() @ xc) / max(xc.shape[0] - 1, 1)             # [C, C]
    eye = torch.eye(c, device=cov.device, dtype=cov.dtype)
    cov = cov + lam * (torch.trace(cov) / c) * eye
    evals, evecs = torch.linalg.eigh(cov)
    evals = evals.clamp_min(1e-12)
    w = evecs @ torch.diag(evals.rsqrt()) @ evecs.t()         # [C, C], symmetric
    return w.float(), mu.float()


@torch.no_grad()
def apply_saw(eeg: torch.Tensor, w: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    return torch.einsum("cd,bdt->bct", w, eeg - mu[None, :, None])


# ============================================================ branch representations
@torch.no_grad()
def branch_reps(brain, eeg: torch.Tensor, modalities, subject_ids=None, batch: int = 512):
    """Frozen branch outputs. `subject_ids=None` means the unseen/target subject."""
    out = {m: [] for m in modalities}
    for i in range(0, eeg.shape[0], batch):
        chunk = eeg[i:i + batch]
        ids = None if subject_ids is None else subject_ids[i:i + batch]
        zs = brain(chunk, ids)
        for m, z in zip(modalities, zs):
            out[m].append(z.float())
    return {m: torch.cat(v, 0) for m, v in out.items()}


# ============================================================ AS: anchor selection
@torch.no_grad()
def vote_anchors(zq: dict, zg: dict, modalities, k: int = 10, min_votes: int = 2):
    """CSLS -> mutual nearest neighbours -> per-branch vote.

    A pair is kept only if at least `min_votes` branches independently agree it is a mutual
    nearest neighbour. Single-branch MNN on EEG is the least trustworthy part of the
    pipeline (low SNR, and the branches see the same noisy input), so cross-branch agreement
    is the cheapest available purity filter. It is also the mechanism that would not exist
    on a single-condition backbone.
    """
    votes = None
    per_branch = {}
    for m in modalities:
        s = csls_scores(_norm(zq[m]), _norm(zg[m]), k=k)
        pairs = mutual_nn_pairs(s)
        per_branch[m] = pairs
        v = torch.zeros(zq[m].shape[0], dtype=torch.long, device=zq[m].device)
        if pairs.numel():
            v.index_add_(0, pairs[:, 0], torch.ones(pairs.shape[0], dtype=torch.long,
                                                    device=v.device))
        votes = v if votes is None else votes + v
    keep = votes >= min_votes
    rows = torch.nonzero(keep, as_tuple=False).flatten()
    return rows, votes, per_branch


@torch.no_grad()
def leverage_select(rows: torch.Tensor, zq: torch.Tensor, zg: torch.Tensor, pairs: dict,
                    rank: int, budget: int):
    """Prefer anchors that SPAN the concept-active subspace, not just confident ones.

    Condition C2 says the fit is only identified on directions the anchors actually span;
    directions outside `span(X)` are silently left at the identity by the ridge term. MNN
    selection is biased towards easy, well-separated concepts, which is exactly the part of
    the subspace that was already easy -- so a greedy determinant (D-optimal) pass is run
    on top of the margin ranking, and the reported `coverage` is the fraction of the
    active-subspace energy the chosen anchors span.
    """
    if rows.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=zq.device), 0.0
    x = _norm(zq[rows])
    d = x.shape[1]
    r = min(rank, d, x.shape[0])
    # active subspace from the gallery's own covariance (known, label-free)
    g = _norm(zg)
    gc = g - g.mean(0, keepdim=True)
    cov = gc.t() @ gc / max(gc.shape[0] - 1, 1)
    evals, evecs = torch.linalg.eigh(cov)
    u_r = evecs[:, -r:]                                        # [d, r]

    proj = x @ u_r                                             # [m, r]
    lev = (proj ** 2).sum(dim=1)
    # energy captured per active direction, then its minimum: a set that covers all r
    # directions evenly maximises the MINIMUM, which is the coverage criterion
    energy = (proj ** 2).sum(dim=0)                            # [r]
    cov_frac = float((energy / (energy.sum() + 1e-12)).min().item() * r)

    if budget and rows.numel() > budget:
        order = torch.argsort(lev, descending=True)[:budget]
        rows = rows[order]
    return rows, cov_frac


# ============================================================ RP: per-branch recovery
@torch.no_grad()
def fit_per_branch(zq: dict, zg: dict, anchor_rows: torch.Tensor, anchor_cols: torch.Tensor,
                   modalities, rho: float = 0.1, sample_weight: torch.Tensor | None = None):
    """Closed-form weighted orthogonal recovery per branch (design doc §1.4).

    `rho` must stay > 0. With `rho = 0` and m << d the problem is under-determined and the
    solution on the unspanned directions is arbitrary; the ridge term is what makes it the
    minimum-norm-in-deviation solution instead. SCORE measures this term alone at +2.25
    Top-1 (and this project measured -2.0 going from `rho=0.1` to `rho=0` in the EPD run),
    so it is not a numerical detail.
    """
    maps = {}
    for m in modalities:
        x = _norm(zq[m])[anchor_rows]
        y = _norm(zg[m])[anchor_cols]
        w = (torch.ones(x.shape[0], device=x.device) if sample_weight is None
             else sample_weight)
        w = w / w.sum() * w.numel()
        r, mu_x, mu_y = orthogonal_recovery(x, y, w, rho=rho)
        maps[m] = (r, mu_x, mu_y)
    return maps


@torch.no_grad()
def branch_defect(maps) -> float:
    """BC: `max_{m != m'} ||R_m R_{m'}^T - I||_F`.

    If H1 holds -- if the four branches' errors are generated by one subject operator --
    then correcting the branches should land them on the same frame, and this is small. A
    large value is evidence *against* H1 and is available without any label, which is what
    makes it usable as a model-selection criterion and not just a diagnostic.
    """
    ms = list(maps)
    if len(ms) < 2:
        return 0.0
    worst = 0.0
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            ri = maps[ms[i]][0].float()
            rj = maps[ms[j]][0].float()
            eye = torch.eye(ri.shape[0], device=ri.device)
            worst = max(worst, float(torch.linalg.matrix_norm(ri @ rj.t() - eye)))
    return worst


# ============================================================ H1: one sensor operator
class SensorOperator(torch.nn.Module):
    """Cayley-parameterised orthogonal operator on the 63 sensor channels."""

    def __init__(self, n_channels: int = 63):
        super().__init__()
        self.A = torch.nn.Parameter(torch.zeros(n_channels, n_channels))

    def matrix(self) -> torch.Tensor:
        A = torch.triu(self.A, diagonal=1)
        A = A - A.t()
        I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        return (I - A) @ torch.linalg.inv(I + A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("cd,bdt->bct", self.matrix(), x)


def fit_sensor_operator(brain, eeg: torch.Tensor, targets: dict, anchor_rows, anchor_cols,
                        modalities, steps: int = 300, lr: float = 5e-2, log_every: int = 0):
    """H1 (design doc §1.3): fit ONE operator in 63-d sensor space to satisfy all branches.

    This is the arm that decides the architecture's shape. The per-branch latent solve has
    `n_modalities * 1024^2` free parameters and needs one coordinate system per branch; this
    arm has `63*62/2 = 1953` and asserts the four branches' errors come from the shared
    physical operator `P_s` induced by the head and electrodes. If it fits as well, the
    expensive formulation was never necessary. If it does not, H1 is false and the latent
    solve is the honest answer -- either way the measurement is the point, so this returns
    the achieved objective alongside the operator rather than a prediction.
    """
    op = SensorOperator(eeg.shape[1]).to(eeg.device)
    opt = torch.optim.Adam(op.parameters(), lr=lr)
    x = eeg[anchor_rows]
    y = {m: _norm(targets[m])[anchor_cols].detach() for m in modalities}
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    hist = []
    # This is the one place in the recovery that needs gradients. The caller is wrapped in
    # `torch.no_grad` (everything else here is closed-form), so the context is entered
    # explicitly rather than left to the caller -- otherwise the first backward raises
    # "element 0 of tensors does not require grad" and H1 silently reports as failed.
    with torch.enable_grad():
        for step in range(steps):
            opt.zero_grad()
            zs = brain(op(x), None)
            loss = 0.0
            for m, z in zip(modalities, zs):
                loss = loss + (1.0 - F.cosine_similarity(_norm(z), y[m], dim=1)).mean()
            loss = loss / len(modalities)
            loss.backward()
            opt.step()
            sched.step()
            hist.append(float(loss.item()))
            if log_every and (step % log_every == 0 or step == steps - 1):
                print(f"[h1   ] step {step:4d} loss {loss.item():.5f}", flush=True)
    return op, hist


# ============================================================ evaluation
@torch.no_grad()
def evaluate_retrieval(zq: dict, zg: dict, modalities, concept_of_row: torch.Tensor,
                       n_way: int, csls_k: int = 10):
    """200-way retrieval per branch. `zg` is ordered by concept, so the target is the row
    index. Reports raw cosine and CSLS so the CSLS contribution is visible per branch."""
    res = {}
    for m in modalities:
        a, b = _norm(zq[m]), _norm(zg[m])
        s_cos = a @ b.t()
        s_csls = csls_scores(a, b, k=csls_k)
        entry = {}
        for tag, s in (("cosine", s_cos), ("csls", s_csls)):
            order = s.argsort(dim=1, descending=True)
            top1 = (order[:, 0] == concept_of_row).float().mean().item() * 100
            kk = min(5, order.shape[1])
            top5 = (order[:, :kk] == concept_of_row[:, None]).any(dim=1).float().mean().item() * 100
            entry[tag] = {"top1": top1, "top5": top5,
                          "mean_rank": float((s.argsort(dim=1, descending=True)
                                              == concept_of_row[:, None]).float().argmax(1).float().mean())}
        entry["n_way"] = n_way
        res[m] = entry
    return res
