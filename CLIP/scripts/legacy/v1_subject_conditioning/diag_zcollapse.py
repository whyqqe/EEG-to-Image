#!/usr/bin/env python
"""Did the subject conditioning collapse to a constant? Measure z_s directly.

Stage 2's own diagnostic alarm (job 637415) says: `cos similarity between 3 DIFFERENT
support sets = 1.0000` and `gap between the episode's own support set and another
subject's = +0.0000`. Both say `z_s` carries no subject information at all. That is
consistent with the eval table, where the "conditioned" rows do beat the "shared" rows --
a CONSTANT FiLM modulation is still a learnable re-parameterisation, so it helps without
being subject-specific.

If confirmed, the cause is structural rather than a bug: nothing in the objective rewards
`z_s` for being SUBJECT-SPECIFIC, while `dec` (HSIC) and `mmd` actively reward making the
representation subject-INVARIANT. A constant `z_s` makes `dec` trivially satisfied, so the
collapse is the optimum of the sub-problem. This script measures how far it has gone.

Runs on CPU in seconds (the support encoder is tiny), so no GPU needed.

  python scripts/diag_zcollapse.py --ckpt outputs/validate/stage2/sub-08/last.pt --subjects 1 2 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import config  # noqa: E402
from samclip.data import things_eeg  # noqa: E402
from samclip.models.subject_conditioning import SupportSetEncoder  # noqa: E402
from samclip.utils import load_config  # noqa: E402


def pairwise_cos(z: torch.Tensor) -> np.ndarray:
    n = torch.nn.functional.normalize(z, dim=-1)
    return (n @ n.T).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=str(config.CONFIGS / "loso_sub08.yaml"))
    ap.add_argument("--subjects", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6, 7, 9, 10])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    src = cfg.get("source_subjects") or [s for s in config.all_subjects()
                                         if s != int(cfg["target_subject"])]
    channels = (config.CHANNELS_OCCIPITO_PARIETAL
                if cfg.get("channel_set", "all63") == "occipital17" else None)
    data = things_eeg.load_loso(src, int(cfg["target_subject"]), channels,
                               mvnn=cfg.get("mvnn", "off"))
    n_ch, n_t = np.asarray(data.tr_eeg[0]).shape[-2:]
    print(f"[data] {data.n_subjects} source subjects, support trial shape ({n_ch}, {n_t})")

    # Rebuild just the support encoder from the checkpoint and load its weights. The
    # rest of the network is irrelevant to the question.
    enc = SupportSetEncoder(n_ch, n_t, d_trial=int(cfg.get("conditioning", {}).get(
        "support_d_trial", 128)), d_z=int(cfg.get("conditioning", {}).get("d_z", 64)),
        anchor=cfg.get("conditioning", {}).get("support_anchor", "none"))
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    mine = {k.split("support_encoder.", 1)[1]: v for k, v in sd.items()
            if "support_encoder." in k}
    missing, unexpected = enc.load_state_dict(mine, strict=False)
    print(f"[ckpt] loaded {len(mine)} support-encoder tensors from {Path(args.ckpt).name}")
    if missing:
        print(f"[ckpt] !! MISSING (stayed at random init): {missing}")
    if unexpected:
        print(f"[ckpt] unexpected: {unexpected}")
    enc.eval()

    rng = np.random.default_rng(args.seed)
    n_img = data.n_images
    idx = {s: np.asarray(data.tr_eeg[s]).reshape(-1, *np.asarray(data.tr_eeg[s]).shape[-2:])
           for s in range(data.n_subjects)}

    # (a) different support sets from DIFFERENT subjects -- must not be identical
    draws_sub, draws_z = [], []
    for s in range(data.n_subjects):
        for _ in range(args.draws):
            picks = rng.choice(len(idx[s]), size=args.k, replace=False)
            draws_sub.append(s)
            draws_z.append(idx[s][picks])
    with torch.no_grad():
        zs = enc(torch.from_numpy(np.stack(draws_z)).float())
    # (a) same-subject vs cross-subject similarity, unnormalised z
    print(f"\n[a] raw z: |z| mean {zs.norm(dim=-1).mean():.4f} "
          f"std {zs.norm(dim=-1).std():.4f}  (post-normalisation it is pinned to 0.16)")
    c = pairwise_cos(zs)
    same, diff = [], []
    for i in range(len(draws_sub)):
        for j in range(i + 1, len(draws_sub)):
            (same if draws_sub[i] == draws_sub[j] else diff).append(c[i, j])
    print(f"[a] cos(z_i, z_j) SAME subject, different support sets: "
          f"mean {np.mean(same):.4f}  min {np.min(same):.4f}")
    print(f"[a] cos(z_i, z_j) DIFFERENT subjects:                    "
          f"mean {np.mean(diff):.4f}  max {np.max(diff):.4f}")
    print(f"[a] separation (same - different) = {np.mean(same) - np.mean(diff):+.5f}"
          f"   <-- 0.0 means z_s is subject-agnostic")

    # (b) does a constant z beat the real one? Compare the spread ACROSS subjects.
    per_subj_mean = torch.stack([zs[[i for i, s in enumerate(draws_sub) if s == k]].mean(0)
                                 for k in range(data.n_subjects)])
    print(f"\n[b] |mean z per subject| mean {per_subj_mean.norm(dim=-1).mean():.4f}, "
          f"pairwise cos between subject means {pairwise_cos(per_subj_mean)[np.triu_indices(data.n_subjects, 1)].mean():.4f}")

    print(f"\n[b] the SAME support set twice (determinism check): "
          f"cos {pairwise_cos(zs[:2])[0,1]:.6f}")


if __name__ == "__main__":
    main()
