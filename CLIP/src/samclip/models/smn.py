"""Subject-Modality Normalisation (SMN) -- the per-subject offset, removed by construction.

WHAT THIS IS, IN ONE SENTENCE
-----------------------------
Each subject is a different *observation modality* of the same concept space, so the
architecture is given the modality's own unlabelled statistics and removes the part of
the embedding they explain -- instead of being asked to discover that invariance through
a penalty it cannot enforce.

WHY A DATA STATISTIC AND NOT A LEARNED SUBJECT VECTOR
----------------------------------------------------
This is the part that decides whether the mechanism works, and it is the reason v1's
conditioning failed. v1 learned a free per-subject embedding ``z_s`` and the objective
rewarded subject INVARIANCE, so a constant ``z_s`` was the optimum of the sub-problem as
written: measured pairwise cosine 0.9999 between different subjects, separation
-0.00000. Nothing was broken; the objective had no reason to make ``z_s`` informative.

A statistic pooled over ~128-200 trials that span many different concepts is
concept-blind **by construction** -- the mean over 128 concepts cannot say which concept
any single trial was. That is a different kind of guarantee from "the loss punishes
using it", and it is the only kind that survived: it needs no incentive, and there is no
free parameter to collapse.

WHY IT MATTERS MORE THAN ANY OTHER COMPONENT HERE
-------------------------------------------------
Measured on three trained checkpoints (`scripts/probe_signal_weights.py`), separating the
two things `calibration.saw_whiten` does:

    metric                          seed2025   seed2026   seed2027
    raw cosine                         13.50      17.50      18.00
    center the QUERY cloud only        20.00      22.00      20.50      <-- almost all of it
    + SAW whitening (best shrink)      19.50      21.50      24.50      <-- ~+0..3 on top

and the image cloud carries an offset of only ``||mean(g)||/mean||g|| = 0.21-0.24``
against the query cloud's ``0.27-0.42``, with ``cos(mean_q, mean_g) = +0.10..+0.14`` --
i.e. the query cloud is displaced by a per-subject vector that is nearly ORTHOGONAL to
the gallery's. On a cosine metric a constant unmatched offset becomes a per-gallery-item
bias, which is exactly hubness.

So the measured largest single lever on this task is a TRANSLATION, and it is worth more
than the whole coordinate-change family: the SNR-optimal diagonal reweighting, given the
labels, reaches only 21.0 / 20.0 / 19.5 -- at or below plain centering.

THE TWO MODES, AND WHY THEY ARE THE SAME OPERATION
--------------------------------------------------
* **training** -- the batch tiles several concepts across a few subjects, so each
  subject's rows are grouped by ``subject_ids`` and the statistics are per subject.
* **deployment** -- the query set is 200 trials of ONE subject, so ``subject_ids=None``
  and the whole batch is one group.

They are not two mechanisms: in both cases the statistic is "the mean over a random
sample of many concepts for one subject". That is why the training-time estimate is a
matched proxy for the deployment-time one rather than a leak -- and it is why the SMN
has to run INSIDE the forward path during training. A centering applied only at test
time would leave the encoder optimised in a metric the deployment does not use.

PROTOCOL STATUS
---------------
Deployment-side SMN uses only the EEG of the subject being scored and never which
stimulus a trial was, so it is the same class of operation as `saw_whiten`: legitimate
under a strict LOSO protocol, and it must be stated explicitly in any write-up. It is
NOT interchangeable with a learned map -- see `probe_subspace_alignment.py`, where a
supervised linear alignment, given 199 of 200 labels, peaks at 17.0 leave-one-out while
fitting those very 200 pairs to 100%. An estimated statistic generalises; a fitted map
does not.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubjectModalityNorm(nn.Module):
    """Subtract each subject's mean embedding; optionally divide by its per-direction
    scale under a learnable gate.

    ``gate_scale`` is a gate rather than a weight because the evidence on the second
    moment is weak and its SIGN is not certain: on three seeds the covariance half of
    `saw_whiten` contributed between -0.5 and +4.0 points (best-shrink 19.5/21.5/24.5
    against 20.0/22.0/20.5 for centering alone), so forcing it on would be fitting one
    seed. The gate starts at ``init_gate = 0.0``, where ``scale**0 == 1`` makes the module
    EXACTLY a centering, and opens only if the objective pays for it. That is a strict
    generalisation of the measured operation, not a bet on it.

    THE GATE IS CLAMPED, NOT SIGMOIDED, AND THAT IS THE POINT. A sigmoid would put the
    initial gate at ``sigmoid(0) = 0.5``, i.e. the module would ship with the second
    moment already half-applied -- exactly the un-evidenced half the measurement says may
    be harmful. ``clamp(0, 1)`` makes 0 reachable exactly and makes the two ends hard
    bounds (gradient zero outside), so the gate can open from 0 and cannot run away. The
    asymmetry is intentional: at the lower end the clamp is a floor on a quantity the
    evidence says should start at zero, so "the objective did not ask for it" and "the
    objective asked against it" both leave the module at pure centring.

    ``min_rows`` is a safety floor, not a regulariser. A group with one row subtracts
    itself and yields a zero vector, which would silently destroy that row's gradient and
    make the retrieval matrix depend on batch composition. Groups below the floor are
    left untouched (``mu = 0``, ``scale = 1``), which degrades to "no SMN for that row"
    rather than to "no row".
    """

    def __init__(self, d_embed: int, enabled: bool = True, gate_scale: bool = True,
                 init_gate: float = 0.0, min_rows: int = 4, eps: float = 1e-5) -> None:
        super().__init__()
        self.d_embed = int(d_embed)
        self.enabled = bool(enabled)
        self.min_rows = int(min_rows)
        self.eps = float(eps)
        if not 0.0 <= float(init_gate) <= 1.0:
            raise ValueError(f"init_gate must be in [0, 1] (it is an exponent, and 0 "
                             f"means 'centring only'), got {init_gate}")
        # `gate_raw` is kept even when the gate is off so a checkpoint's state_dict has a
        # stable shape across the `gate_scale` ablation; it is simply unused.
        self.gate_scale = bool(gate_scale)
        self.gate_raw = nn.Parameter(torch.tensor(float(init_gate)))

    # ------------------------------------------------------------------ helpers
    def _group_stats(self, z: torch.Tensor, subject_ids: torch.Tensor | None
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(mu_per_row, var_per_row, ok_per_row)``.

        ``subject_ids=None`` means "one group" -- the deployment case, where the whole
        batch is a single subject's query set.
        """
        n = z.shape[0]
        if subject_ids is None:
            ok = torch.full((n,), n >= self.min_rows, dtype=torch.bool, device=z.device)
            mu = z.mean(dim=0, keepdim=True).expand(n, -1)
            zc = z - mu
            var = (zc ** 2).mean(dim=0, keepdim=True).expand(n, -1)
            return mu, var, ok
        ids = subject_ids.to(z.device).long().reshape(-1)
        if ids.shape[0] != n:
            raise ValueError(
                f"subject_ids has {ids.shape[0]} entries for {n} rows; a broadcast "
                f"here would group the wrong rows and silently centre across subjects")
        uniq, inv = torch.unique(ids, return_inverse=True)
        g = uniq.numel()
        onehot = F.one_hot(inv, g).to(z.dtype)                 # (N, G)
        cnt = onehot.sum(dim=0)                                # (G,)
        mu_g = (onehot.t() @ z) / cnt.clamp_min(1.0).unsqueeze(-1)   # (G, D)
        mu = mu_g[inv]
        zc = z - mu
        var_g = (onehot.t() @ (zc ** 2)) / cnt.clamp_min(1.0).unsqueeze(-1)
        var = var_g[inv]
        ok = (cnt[inv] >= self.min_rows)
        return mu, var, ok

    def forward(self, z: torch.Tensor, subject_ids: torch.Tensor | None = None,
                detach_gate: bool = False) -> torch.Tensor:
        if not self.enabled:
            return z
        if z.dim() != 2:
            raise ValueError(f"SMN expects (B, D) embeddings, got {tuple(z.shape)}")
        mu, var, ok = self._group_stats(z, subject_ids)
        # `ok` selects rows, so a below-floor group degrades to the identity rather than
        # to a zero vector. `torch.where` keeps the graph intact on both branches.
        keep = ok.unsqueeze(-1)
        zc = torch.where(keep, z - mu, z)
        if self.gate_scale:
            sd = torch.sqrt(var + self.eps)
            # Normalise the per-direction scale by its own mean so the gate moves the
            # SHAPE, not the overall magnitude. Without this the module would be partly a
            # gain term, and a gain is scale-invariant under the L2 normalisation that
            # follows -- i.e. it would spend a parameter on nothing while looking active.
            shape = (sd / sd.mean(dim=-1, keepdim=True).clamp_min(self.eps))
            gate = self.gate()
            if detach_gate:
                # `detach_gate` is how an AUXILIARY objective is stopped from moving this
                # scalar while still receiving gradient into the encoder through it. It
                # exists because of a measured failure: the cross-trial term (T2') and this
                # scale correction are SUBSTITUTES -- both remove the within-concept
                # scatter -- so the auxiliary term found it cheaper to let the gate decay
                # than to shape the encoding. On sub-08/seed2025 the gate went 0.498
                # (baseline) -> 0.203 -> 0.000 as the weight went 0 -> 0.1 -> 0.5, and it
                # took the deployment rung down 40.5 -> 30.0 with it.
                #
                # The gate's optimum is defined by the DEPLOYED regime -- 200 averaged
                # trials of one subject -- which the main terms model. The auxiliary term
                # runs on UNAVERAGED repetitions, a different noise regime with R times the
                # within-row variance, so letting it set the gate lets it optimise a
                # statistic for an input distribution the model is not deployed on. Cutting
                # only the gate's own gradient keeps the term's effect on the ENCODER, which
                # is what it is for.
                gate = gate.detach()
            zc = torch.where(keep, zc / shape.pow(gate), zc)
        return zc

    def gate(self) -> torch.Tensor:
        """The scale exponent: exactly 0 (centring only) unless the objective opens it."""
        if not self.gate_scale:
            return self.gate_raw.new_zeros(())
        return self.gate_raw.clamp(0.0, 1.0)

    # -------------------------------------------------------------- diagnostics
    @torch.no_grad()
    def offset_ratio(self, z: torch.Tensor, subject_ids: torch.Tensor | None = None
                     ) -> float:
        """``||per-subject mean|| / mean||row||`` -- the quantity the SMN exists to remove.

        Logged every epoch because it is the ONE number that says whether C1 (the shared
        head) and C2 together did what they were built for. On the v3 checkpoints it is
        0.27-0.42; the reference architecture needs no test-time centering at all, and
        0.2 is where the image cloud already sits.
        """
        return subject_offset_ratio(z, subject_ids, self.min_rows)


@torch.no_grad()
def subject_offset_ratio(z: torch.Tensor, subject_ids: torch.Tensor | None = None,
                         min_rows: int = 2) -> float:
    """``||per-subject mean|| / mean||row||``, and arch-agnostic so the v3 path can be
    measured with the same ruler as the v4 path.

    ``min_rows`` guards the same degenerate case as the module: a group of one row has
    that row as its own mean, so the ratio sums to 1 by construction instead of
    measuring anything. Such rows are excluded from the average.
    """
    if z.dim() != 2 or z.shape[0] == 0:
        return 0.0
    n = z.shape[0]
    if subject_ids is None:
        if n < min_rows:
            return 0.0
        mu = z.mean(dim=0, keepdim=True).expand(n, -1)
    else:
        ids = subject_ids.to(z.device).long().reshape(-1)
        uniq, inv = torch.unique(ids, return_inverse=True)
        onehot = F.one_hot(inv, uniq.numel()).to(z.dtype)
        cnt = onehot.sum(dim=0)
        mu_g = (onehot.t() @ z) / cnt.clamp_min(1.0).unsqueeze(-1)
        mu = mu_g[inv]
        keep = (cnt[inv] >= min_rows)
        if not bool(keep.any()):
            return 0.0
        mu = mu[keep]
    row = z.norm(dim=-1).mean().clamp_min(1e-8)
    return float(mu.norm(dim=-1).mean() / row)
