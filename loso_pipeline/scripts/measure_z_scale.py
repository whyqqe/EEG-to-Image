"""Measure the scale of z_inv at initialisation, to set the VICReg gamma.

`vicreg_variance` is a hinge at `gamma` on each dimension's batch standard
deviation.  If `gamma` is above what the architecture can produce, the term sits
saturated and its gradient is zero -- it would appear in the loss table at a constant
value forever while doing nothing, which is worse than not having it (a term that
looks active is a term nobody re-examines).

So the achievable scale is measured rather than assumed, on the real data with the
real initialisation, and `HeadConfig.vicreg_gamma` is set from the result.  Run:

  python scripts/measure_z_scale.py
"""
from __future__ import annotations

import numpy as np
import torch

from loso import paths
from loso.data import eeg as eeg_mod
from loso.data import things
from loso.models.eeg_encoder import EEGEncoder, EncoderConfig
from loso.models.heads import AlignmentHeads, HeadConfig


def main() -> None:
    torch.manual_seed(0)
    train_subjects, test_subject = things.loso_split("sub-08")
    n_subjects = len(train_subjects)

    cfg = EncoderConfig(pretrained_subjects=n_subjects)
    model = EEGEncoder(cfg)
    heads = AlignmentHeads(HeadConfig(n_subjects=n_subjects,
                                      d_model=cfg.d_model, d_inv=cfg.d_inv,
                                      n_time_patches=cfg.n_tokens))
    model.eval()
    heads.eval()

    # Real EEG, real normalisation -- an affine map applied to whitened data, so the
    # input scale is what training will actually see.
    stats = eeg_mod.ChannelStats(torch.zeros(paths.N_CHANNELS), torch.ones(paths.N_CHANNELS))
    ds = eeg_mod.TrainEEGDataset([train_subjects[0]], {train_subjects[0]: stats},
                                 {train_subjects[0]: train_subjects[0]},
                                 avg_trials=True)
    idx = np.linspace(0, len(ds) - 1, 512).astype(int)
    x = torch.stack([ds[int(i)].x for i in idx])
    subject_id = torch.zeros(512, dtype=torch.long)

    with torch.no_grad():
        out = model(x, subject_id, return_tokens=True)
        z = out["z_inv"].float()

    print(f"z_inv shape        : {tuple(z.shape)}")
    print(f"per-sample L2 norm : mean={z.norm(dim=-1).mean():.4f}  "
          f"std={z.norm(dim=-1).std():.4f}")
    std = z.std(dim=0)
    print(f"per-dim batch std  : mean={std.mean():.4f}  median={std.median():.4f}  "
          f"min={std.min():.4f}  max={std.max():.4f}")
    print(f"1/sqrt(d_inv)      : {1 / cfg.d_inv ** 0.5:.4f}")

    # LayerNorm makes `head_inv`'s output a point on a sphere of radius sqrt(d_inv)
    # (per-sample norm 22.6271 above, with zero variance across samples -- that is
    # what LayerNorm does: it fixes each sample's *within-sample* variance to 1).
    # On that sphere, per-dim batch variance + squared per-dim batch mean sum to
    # exactly 1 per dimension, so the split between "spread" and "shared direction"
    # is the quantity that says whether the representation is already collapsed.
    var = z.var(dim=0, unbiased=False)
    mean_sq = z.mean(dim=0) ** 2
    total = float((var + mean_sq).sum())
    print(f"sphere identity    : sum_d var_d={float(var.sum()):.2f} + "
          f"sum_d mean_d^2={float(mean_sq.sum()):.2f} = {total:.2f} "
          f"(expected d_inv={cfg.d_inv})")
    print(f"energy in the shared direction : {float(mean_sq.sum()) / total * 100:.1f}%  "
          "<- 0% is isotropic, ~87% is the failed run's collapse")

    from loso.diagnostics import collapse_verdict, representation_diagnostics
    stats = representation_diagnostics(z)
    ok, verdict = collapse_verdict(stats)
    print(f"diagnostics at init: top1_sv_ratio={stats['top1_sv_ratio']:.4f} "
          f"eff_rank={stats['eff_rank']:.2f} "
          f"mean_cosine={stats['mean_offdiag_cosine']:+.4f} -> {verdict}")

    from loso.losses import align as L
    print()
    for gamma in (0.1, 0.25, 0.5, 0.75, 1.0):
        value = float(L.vicreg_variance(z, gamma=gamma))
        collapsed = z.mean(0, keepdim=True).expand_as(z)
        value_collapsed = float(L.vicreg_variance(collapsed, gamma=gamma))
        # The gradient this term applies at the current point, which is what decides
        # whether it is doing anything.  A saturated hinge has a zero gradient and a
        # constant loss, and would sit in the loss table looking active forever.
        probe = z.detach().clone().requires_grad_(True)
        L.vicreg_variance(probe, gamma=gamma).backward()
        grad = float(probe.grad.norm())
        print(f"gamma={gamma:<5}: var_term={value:.5f}  constant={value_collapsed:.5f}  "
              f"|grad|={grad:.5f}")

    cov = float(L.vicreg_covariance(z))
    cov_collapsed = float(L.vicreg_covariance(z.mean(0, keepdim=True).expand_as(z)))
    print(f"cov term           : real={cov:.6f}  constant={cov_collapsed:.6f}")


if __name__ == "__main__":
    main()
