#!/usr/bin/env python3
"""Does EEG carry a SECOND, IDENTITY-BEARING channel?

WHY THIS IS THE DECISIVE QUESTION
  Any architecture that SAMPLES candidates and SELECTS one needs a scoring channel
  that is (a) informative about stimulus identity and (b) not merely a re-encoding
  of the channel already used to generate. If no such channel exists, sampling and
  selection is decoration: argmax over candidates collapses back to the top-1 that
  a single point estimate already gives.

  The project has two candidates for that second channel:
    * the semantic pathway  (z_eeg_proj -> CLIP, 2-way 0.92-0.96)
    * the layout pathway    (z_eeg_proj -> low-frequency VAE latent, trained
                             separately, its own head, its own target)
  This script measures how much IDENTITY information the layout pathway carries,
  and whether that information is complementary to the semantic pathway.

METHOD
  Ranking, not generation, so no images are needed and this costs seconds:
    s_sem(i,j) = cosine(cond_i, CLIP_j)                over the 200 test images
    s_lay(i,j) = cosine(LF(pred_latent_i), LF(latent_j))
  Reported for each alone and fused: top-1, recall@N and 2-way identification.
  A fusion gain over s_sem alone is the evidence that a verification channel
  exists. No gain means it does not.

CAVEAT, stated plainly
  Candidate latents here come from the TEST images. In a deployed system the
  candidates would be TRAIN images, so this is an UPPER BOUND on the layout
  channel's usefulness, not an achievable number. It is exactly the right
  quantity to test existence: if the upper bound shows nothing, the design is dead.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def l2n(x: np.ndarray) -> np.ndarray:
    x = x.reshape(len(x), -1).astype(np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def radial_grid(h: int, w: int) -> np.ndarray:
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    return np.sqrt(fy**2 + fx**2)


def low_band(x: np.ndarray, r: np.ndarray, cut: float) -> np.ndarray:
    """Identical to g2_build_targets.low_band: keep only r < cut, Parseval exact."""
    F = np.fft.fft2(x.astype(np.float32), axes=(-2, -1))
    m = (r < cut).astype(np.float32)
    return np.real(np.fft.ifft2(F * m, axes=(-2, -1))).astype(np.float32)


def lf_stack(x: np.ndarray, cut: float, drop_dc: bool) -> np.ndarray:
    """(n,C,H,W) -> (n, C*H*W) low-frequency band, optionally with DC removed."""
    r = radial_grid(x.shape[-2], x.shape[-1])
    out = np.stack([low_band(x[i], r, cut) for i in range(len(x))])
    if drop_dc:
        out = out - out.mean(axis=(-2, -1), keepdims=True)
    return out.reshape(len(x), -1)


def rank_metrics(s: np.ndarray) -> dict[str, float]:
    n = s.shape[0]
    order = np.argsort(-s, axis=1)
    idx = np.arange(n)
    rng = np.random.default_rng(0).permutation(n)
    ok = idx != rng
    return {
        "top1": float(np.mean(order[:, 0] == idx)),
        "top5": float(np.mean([i in order[i, :5] for i in idx])),
        "twoway": float(np.mean(s[idx, idx][ok] > s[idx, rng][ok])),
        "mean_rank": float(np.mean([np.where(order[i] == i)[0][0] for i in idx])),
    }


def zs(s: np.ndarray) -> np.ndarray:
    return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", type=str, default="outputs/g3f/sub-08/conds/ip_fused_test.npy")
    ap.add_argument("--clip-test", type=str,
                    default="/project/peilab/why/eeg-brainit/outputs/atm_bridge/clip_img_test_1024.npy")
    ap.add_argument("--pred-latent", type=str,
                    default="outputs/sdedit_ll_full10/sub-08/vae_head/pred_vae_test.npy")
    ap.add_argument("--gt-latent", type=str,
                    default="outputs/sdedit_ll_full10/shared/vae_cache/test_vae_latents_f16.npy")
    ap.add_argument("--cut", type=float, default=0.0625)
    ap.add_argument("--out-json", type=str, default="outputs/g3f/channel_basis.json")
    args = ap.parse_args()

    C = l2n(np.load(args.cond).astype(np.float32))
    CLIP = l2n(np.load(args.clip_test).astype(np.float32))
    P = np.load(args.pred_latent).astype(np.float32).reshape(200, 4, 64, 64)
    G = np.load(args.gt_latent).astype(np.float32).reshape(200, 4, 64, 64)

    res: dict = {"n": 200, "cut": args.cut}

    # ---- pathway 1: semantic (EEG condition vs CLIP of the image)
    s_sem = C @ CLIP.T
    res["semantic"] = rank_metrics(s_sem)

    # ---- pathway 2: layout, two variants (with / without the DC component)
    for tag, drop in (("layout_lf_withdc", False), ("layout_lf_nodc", True)):
        Lp = l2n(lf_stack(P, args.cut, drop))
        Lg = l2n(lf_stack(G, args.cut, drop))
        s_lay = Lp @ Lg.T
        res[tag] = {**rank_metrics(s_lay), "hf_note": "low-frequency band only"}

    # ---- is the layout prediction itself informative about the true latent?
    Lp = lf_stack(P, args.cut, True)
    Lg = lf_stack(G, args.cut, True)
    a, b = l2n(Lp), l2n(Lg)
    res["layout_pred_vs_gt_cos"] = float((a * b).sum(1).mean())
    res["layout_pred_vs_gt_corr_perdim"] = float(
        np.mean([np.corrcoef(Lp[i], Lg[i])[0, 1] for i in range(len(Lp))]))

    # ---- fusion: does layout add anything ON TOP of semantic?
    s_lay = l2n(lf_stack(P, args.cut, True)) @ l2n(lf_stack(G, args.cut, True)).T
    best = None
    for lam in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0):
        m = rank_metrics(zs(s_sem) + lam * zs(s_lay))
        if best is None or m["twoway"] > best[1]["twoway"]:
            best = (lam, m)
    res["fusion_sem_plus_layout"] = {"best_lambda": best[0], **best[1]}
    res["fusion_gain_twoway"] = best[1]["twoway"] - res["semantic"]["twoway"]
    res["fusion_gain_top1"] = best[1]["top1"] - res["semantic"]["top1"]

    Path(args.out_json).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
