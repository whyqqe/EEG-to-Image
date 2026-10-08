#!/usr/bin/env python
"""How much of the shipped embedding is CONCEPT and how much is still SUBJECT?

`probe_per_subject_accuracy.py` shows the same encoder scores 45.7 on the nine subjects
it trained on and 17.5 on the held-out one, over the identical 200 concepts. That gap
(28.2 pp) is a statement about the representation, so this probe decomposes the
representation itself:

  1. A one-way ANOVA-style variance split over the (10 subjects) x (200 concepts) grid.
     With E[s, c, :] the embedding of subject s on concept c, and SMN applied per subject
     exactly as deployment does:
        between-concept SS   concept c's mean over subjects vs the grand mean   -> SIGNAL
        between-subject SS   subject s's mean over concepts vs the grand mean   -> NUISANCE
        residual             the rest                                           -> NOISE
     Reported for the pre-SMN and post-SMN views, so the split also says exactly what the
     SMN buys and what it leaves behind.

  2. Cross-validated subject identification from the SHIPPED embedding. Chance is 10%.
     Anything well above that is subject identity the encoder still exposes, i.e. an
     unlabelled shortcut available to the training objective that no term forbids.

Neither number uses target labels, and the classifier is fitted on the embeddings the
deployment metric actually consumes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import config                                    # noqa: E402
from samclip.data.things_eeg import load_subject_std          # noqa: E402
from samclip.models import build_model                        # noqa: E402
from samclip.utils import load_config                          # noqa: E402


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--held-out", type=int, default=8)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg") or load_config("configs/loso_sub08_v4.yaml")
    from samclip import train as train_mod
    _, tte = train_mod.build_targets(cfg)
    model = build_model(cfg, tte.shape[2], tte.shape[-1]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[variance] ckpt={args.ckpt} epoch={ck.get('epoch')} arch={cfg.get('arch')}")

    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    pre, post = [], []
    for s in subjects:
        mvnn = "test" if s == args.held_out else "train"
        _, te = load_subject_std(s, None, None, mvnn=mvnn)
        x = np.asarray(te[:, 0], dtype=np.float32)
        chunks = []
        for i in range(0, len(x), args.batch):
            t = torch.as_tensor(np.ascontiguousarray(x[i:i + args.batch])).to(device)
            chunks.append(model.embed_eeg(t).cpu())
        raw = torch.cat(chunks)
        pre.append(raw.numpy())
        post.append(torch.nn.functional.normalize(
            model.apply_smn(raw.to(device), None), dim=-1).cpu().numpy())
    P = np.stack(pre)      # (S, C, d) pre-SMN
    Z = np.stack(post)     # (S, C, d) shipped

    def decompose(X: np.ndarray, name: str) -> None:
        grand = X.mean(axis=(0, 1), keepdims=True)
        cmean = X.mean(axis=0, keepdims=True)        # (1, C, d) per concept
        smean = X.mean(axis=1, keepdims=True)        # (S, 1, d) per subject
        # total variance summed over dims, in the same units for each term
        tot = float(((X - grand) ** 2).sum())
        concept = float(((cmean - grand) ** 2).sum()) * X.shape[0]
        subj = float(((smean - grand) ** 2).sum()) * X.shape[1]
        inter = float(((X - cmean - smean + grand) ** 2).sum())
        print(f"\n  {name}:  total SS {tot:.3e}")
        print(f"    between-CONCEPT (signal)   {concept / tot * 100:6.2f} %")
        print(f"    between-SUBJECT (nuisance) {subj / tot * 100:6.2f} %")
        print(f"    residual        (noise)    {inter / tot * 100:6.2f} %")
        # the quantity that actually sets retrieval: concept spread vs the rest
        print(f"    concept / (subject + residual) = "
              f"{concept / max(subj + inter, 1e-30):.3f}")

    decompose(P, "PRE-SMN  (what the shared head produces)")
    decompose(Z, "POST-SMN (what ships / what the metric sees)")

    # --- is subject identity still recoverable from the SHIPPED embedding? ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    Xs = Z.reshape(-1, Z.shape[-1])
    ys = np.repeat(np.arange(len(subjects)), Z.shape[1])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    acc = cross_val_score(clf, Xs, ys, cv=5, scoring="accuracy")
    print(f"\n  subject identification from the SHIPPED embedding: "
          f"{acc.mean() * 100:.2f} % +/- {acc.std() * 100:.2f}   (chance 10.00 %)")
    Xp = P.reshape(-1, P.shape[-1])
    accp = cross_val_score(clf, Xp, ys, cv=5, scoring="accuracy")
    print(f"  subject identification from the PRE-SMN embedding:  "
          f"{accp.mean() * 100:.2f} % +/- {accp.std() * 100:.2f}")


if __name__ == "__main__":
    main()
