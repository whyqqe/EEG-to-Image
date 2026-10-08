"""Why did the GQF whitening front-end halve the training-time Top-1?

The whitening `W = (XX^T + eps*I)^{-1/2} X` is exact (W W^T = I), so the row space is
preserved in principle. Measure the two mechanisms that break it in practice:

  1. CONDITIONING. `C = X X^T` in EEG is near-singular; `C^{-1/2}` amplifies the
     smallest spatial directions by `1/sqrt(lambda_min)`. With `eps = 1e-4 * trace/C`
     the regulariser is far below the noise floor, so the noise subspace gets blown up.
  2. ENERGY. Whitening forces `tr(W^T W) = C` (constant per trial), which is a strong
     per-trial normalisation the trunk was never trained under.

Run: PYTHONPATH=src .venv/bin/python scratch/probe_gqf_failure.py
"""
from __future__ import annotations

import numpy as np
import torch

from samclip import config
from samclip.data import things_eeg
from samclip.models.samclip import group_quotient


def main() -> None:
    subj = 5
    reps = things_eeg.load_test_reps(subj, channels=None, mvnn="off")
    print("test reps shape", reps.shape, reps.dtype)          # (200, R, Ch, T)
    x = torch.from_numpy(np.ascontiguousarray(np.asarray(reps).mean(axis=1))).float()
    print("averaged trials", tuple(x.shape))

    # ---- 1. conditioning of C before regularisation -------------------------------
    c = x @ x.transpose(-1, -2)                                # (N, C, C)
    scale = c.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)
    w = torch.linalg.eigvalsh(c.double())                      # (N, C) ascending
    w = w.clamp_min(0)
    wmax = w[:, -1:]
    cond = (wmax / w[:, :1].clamp_min(1e-30)).squeeze(-1)
    frac_small = (w < 1e-4 * wmax).float().sum(-1) / w.shape[-1]
    frac_trace_small = (w * (w < 1e-4 * wmax)).sum(-1) / w.sum(-1)
    print(f"\n[C] condition number: median {cond.median():.3e}  "
          f"p10 {cond.quantile(0.1):.3e}  p90 {cond.quantile(0.9):.3e}")
    print(f"[C] frac of dims with lambda < 1e-4*lambda_max: "
          f"median {frac_small.median():.3f}")
    print(f"[C] frac of TRACE carried by those dims:         "
          f"median {frac_trace_small.median():.4f}")
    print(f"[C] eps*scale / lambda_min: median "
          f"{(1e-4 * scale.squeeze(-1) / w[:, 0]).median():.3e}")

    # ---- 2. what whitening does to the magnitude ---------------------------------
    y = group_quotient(x, "whiten")                            # the actual front-end
    def rms(t):  # per-trial RMS of the tensor
        return t.flatten(1).pow(2).mean(1).sqrt()
    print(f"\n[scale] RMS(x) median {rms(x).median():.4e}   "
          f"RMS(W) median {rms(y).median():.4e}   "
          f"ratio {float((rms(y)/rms(x)).median()):.3e}")
    # whitened row Gram must be identity by construction
    g = y @ y.transpose(-1, -2)
    eye = torch.eye(g.shape[-1]).expand_as(g)
    print(f"[scale] max|W W^T - I| = {float((g - eye).abs().max()):.3e}")

    # ---- 3. energy in the amplified (small-eigenvalue) directions ------------------
    # Project the ORIGINAL trial onto the eigenbasis of C; the SNR-relevant energy sits
    # in the leading dims. After whitening every dim has equal energy, so the noise dims
    # are scaled UP by 1/sqrt(lambda_i) * sqrt(total/C).
    q = torch.linalg.eigh(c.double()).eigenvectors       # (N, C, C)
    proj = torch.einsum("nij,njt->nit", q.transpose(-1, -2), x.double())
    e_orig = proj.pow(2).sum(-1)                          # (N, C) energy per eigendim
    e_orig = e_orig / e_orig.sum(-1, keepdim=True)
    tail = e_orig[:, :8].sum(-1)                          # bottom-8 dims share
    head = e_orig[:, -8:].sum(-1)                         # top-8 dims share
    print(f"\n[spectrum] original energy share: top-8 dims {head.median():.4f}  "
          f"bottom-8 dims {tail.median():.5f}")
    amp = (1.0 / w.clamp_min(1e-30).sqrt())
    print(f"[spectrum] whitening gain 1/sqrt(lambda): top-8 dims {amp[:, -8:].median():.3e}  "
          f"bottom-8 dims {amp[:, :8].median():.3e}")
    print(f"[spectrum] gain ratio bottom/top = "
          f"{float((amp[:, :8].median()/amp[:, -8:].median())):.3e}")


if __name__ == "__main__":
    main()
