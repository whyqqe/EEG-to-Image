#!/usr/bin/env python3
"""DECISIVE PRE-CHECK for verification-based decoding.

THE QUESTION
  Selection can only help if the score used to SELECT is a different measurement
  from the score used to RANK the candidates. If both are the same posterior
  score, argmax over candidates is just top-1 and Best-of-N gains nothing.

  The only genuinely different measurement available is: render the candidate,
  re-encode the RENDERED image with CLIP, and score it against the EEG. That is
  a new observation because the renderer's output is not the candidate embedding.

  So the whole design reduces to ONE measurable quantity:

      V_acc = fraction of pairs (i,j) for which
              sim( cond(z_i), phi(generated_image_i) )  >
              sim( cond(z_i), phi(generated_image_j) )

  i.e. can the EEG tell its OWN generated image from somebody else's? This is the
  same 2-way identification used everywhere else in this project, except that the
  images are GENERATED rather than ground truth.

WHY IT IS THE RIGHT PRE-CHECK
  A verifier trained on ground-truth CLIP embeddings may not transfer: generated
  images have their own systematic offset. Measuring V_acc on real generations
  settles that empirically instead of assuming it.

  Given recall@N (the true image in the candidate top-N) and V_acc = p, the
  expected selection accuracy is approximately

      top1_expected(N) ~= recall@N * p^(N-1)

  which is what this script reports next to the measured V_acc.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", type=str, required=True, help="dir with 000.png .. 199.png")
    ap.add_argument("--cond-npy", type=str, required=True, help="(200,1024) EEG-derived condition")
    ap.add_argument("--gt-test-npy", type=str,
                    default="/project/peilab/why/eeg-brainit/outputs/atm_bridge/clip_img_test_1024.npy")
    ap.add_argument("--out-json", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    import torch
    from PIL import Image
    import open_clip
    from tqdm import tqdm

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    gdir = Path(args.gen_dir)
    paths = [gdir / f"{i:03d}.png" for i in range(200)]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"[FATAL] {len(missing)} missing generations, e.g. {missing[0]}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", device=dev)
    model = model.to(dev).eval()
    feats = []
    with torch.no_grad():
        for p in tqdm(paths, desc="clip(gen)"):
            x = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
            feats.append(model.encode_image(x).float().cpu())
    PHI = l2n(torch.cat(feats, 0).numpy())

    C = l2n(np.load(args.cond_npy).astype(np.float32))
    GT = l2n(np.load(args.gt_test_npy).astype(np.float32))
    n = len(PHI)

    # --- V_acc: can the condition identify its own generation?
    S = C @ PHI.T                                  # (cond_i, gen_j)
    rng = np.random.default_rng(0)
    r = rng.permutation(n)
    ok = np.arange(n) != r
    v_acc = float(np.mean(S[np.arange(n), np.arange(n)][ok] > S[np.arange(n), r][ok]))
    v_top1 = float(np.mean(S.argmax(1) == np.arange(n)))
    v_top5 = float(np.mean([i in np.argsort(-S[i])[:5] for i in range(n)]))

    # --- recall@N of the TRUE image under the condition (coverage for Best-of-N)
    SG = C @ GT.T
    recall = {k: float(np.mean([i in np.argsort(-SG[i])[:k] for i in range(n)]))
              for k in (1, 2, 4, 8, 16, 32)}

    # --- do the GENERATED images sit on the CLIP manifold, or are they off it?
    def c_self(a: np.ndarray) -> float:
        m = l2n(a.mean(0, keepdims=True))[0]
        return float((l2n(a) @ m).mean())
    manifold = {"c_self_condition": c_self(C), "c_self_generated": c_self(PHI),
                "c_self_gt": c_self(GT), "c_self_gen_vs_gt_centroid":
                    float((PHI @ l2n(GT.mean(0, keepdims=True))[0]).mean())}

    # --- how close are generations to their own GT, and does the condition
    #     rank that closeness correctly? (monotonicity == the proxy is valid)
    d_gen_own = (PHI * GT).sum(1)                   # sim(gen_i, gt_i)
    d_cond_own = (C * GT).sum(1)                    # sim(cond_i, gt_i)
    order_g = np.argsort(np.argsort(d_gen_own))
    order_c = np.argsort(np.argsort(d_cond_own))
    from scipy.stats import spearmanr
    rank_corr = float(spearmanr(d_cond_own, d_gen_own).statistic)

    out = {
        "gen_dir": args.gen_dir, "cond_npy": args.cond_npy, "n": n,
        "verifier_acc_2way": v_acc, "verifier_top1_own_generation": v_top1,
        "verifier_top5": v_top5,
        "recall_at_N_true_image": recall,
        "manifold": manifold,
        "generation_closeness": {
            "mean_sim_gen_own_gt": float(d_gen_own.mean()),
            "mean_sim_cond_own_gt": float(d_cond_own.mean()),
            "spearman_cond_vs_gen_closeness": rank_corr,
        },
        "predicted_best_of_n": {
            str(N): round(recall[N] * v_acc ** (N - 1), 4) for N in (2, 4, 8, 16, 32)
        },
        "note": ("verifier_acc_2way is measured on GENERATED images, so it includes the "
                 "domain gap that a GT-embedding verifier would hide. "
                 "predicted_best_of_n assumes independence and is an UPPER bound."),
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
