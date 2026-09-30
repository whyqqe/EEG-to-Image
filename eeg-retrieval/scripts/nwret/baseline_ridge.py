#!/usr/bin/env python
"""Linear ridge baseline: EEG -> image features.

Why this exists
---------------
A deep model's score is uninterpretable on its own. This is the calibration line:
if a closed-form linear map reaches the same Top-1, the deep encoder is not
earning its complexity, and if it reaches much more, the deep encoder is
undertrained or misconfigured. Both are actionable, and neither is visible from
the deep model's numbers alone.

It also isolates a property of the standard THINGS-EEG2 protocol that is easy to
mistake for a bug:

    train repetitions are averaged 4x, test repetitions 80x

so test trials have ~sqrt(80/4) ~= 4.5x better SNR than training trials, and the
validation holdout -- being drawn from the training side -- is *noisier* than the
test set. Evaluating the same closed-form model on both therefore separates
"our model is broken" from "the protocol's noise levels differ".

Usage
-----
    python scripts/nwret/baseline_ridge.py --subject 8 --channels occipito_parietal
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nwret import config
from nwret.data import concept_split, load_subject
from nwret.metrics import mean_rank, retrieval_report


def ridge_dual_eig(Xtr, Ytr):
    """Precompute the dual eigendecomposition once for a whole lambda sweep.

    K = X X^T = U diag(s) U^T, so (K + lam I)^-1 = U diag(1/(s+lam)) U^T.
    The eigendecomposition is the expensive part (n^3); each lambda afterwards
    costs only a pair of matmuls, which makes an 8-point sweep nearly free.
    """
    K = (Xtr @ Xtr.T).double()
    K = 0.5 * (K + K.T)                      # enforce symmetry before eigh
    s, U = torch.linalg.eigh(K)
    UtY = U.T @ Ytr.double()
    return U, s, UtY


def ridge_predict_eig(Xte, Xtr, U, s, UtY, lam):
    """Y_pred = X_te X_tr^T U diag(1/(s+lam)) U^T Y."""
    Kte = (Xte @ Xtr.T).double()
    return Kte @ (U @ (UtY / (s + lam).unsqueeze(-1)))


def ridge_fit_predict(Xtr, Ytr, Xte, lam):
    """Single-lambda convenience wrapper (used only when a sweep is not needed)."""
    U, s, UtY = ridge_dual_eig(Xtr, Ytr)
    return ridge_predict_eig(Xte, Xtr, U, s, UtY, lam)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--channels", default="occipito_parietal", choices=["all", "occipito_parietal"])
    ap.add_argument("--val-concepts", type=int, default=150)
    ap.add_argument("--split-seed", type=int, default=2025)
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ridge] device={dev}")

    channels = None if a.channels == "all" else config.CHANNELS_OCCIPITO_PARIETAL
    tr_eeg, te_eeg = load_subject(a.subject, channels)      # (C,10,Ch,T), (200,1,Ch,T)
    if channels is None:
        ch = json.loads((config.EEG_DIR / "info.json").read_text())["ch_names"]
    else:
        ch = channels
    print(f"[ridge] channels={len(ch)}  train={tr_eeg.shape}  test={te_eeg.shape}")

    img_tr = np.load(config.IMAGE_FEATURE_DIR / "image_train.npy")   # (1654,10,1024)
    img_te = np.load(config.IMAGE_FEATURE_DIR / "image_test.npy")    # (200,1,1024)

    def l2n(x):
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)

    split = concept_split(a.val_concepts, a.split_seed)

    def flat(eeg, concepts):
        """(n_c, n_i, Ch, T) -> (n_c*n_i, Ch*T)"""
        x = eeg[concepts]                       # (n_c, n_i, Ch, T)
        return x.reshape(-1, x.shape[-2] * x.shape[-1])

    # Val sweeps EVERY image slot and averages, which is what train.py's
    # evaluate_selection now does and what probe_layers.py does. This block used to
    # evaluate slot 0 only. That was correct when train.py's TestDataset hard-coded
    # [:, 0], but once train.py moved to a 10-slot sweep the definitions drifted
    # apart, and a lambda selected on 1/10th of the selection signal is not the
    # lambda a 10-slot sweep would pick -- so the floor and the deep arms would no
    # longer be comparable at the same operating point.
    #
    # Kept as (n_concepts, n_slots, ...) and indexed [:, si], NOT flattened to
    # (n_concepts*n_slots, ...) and sliced [si*n:(si+1)*n]. A flat reshape is
    # concept-major, so a contiguous slice of n rows takes n/n_slots whole concepts
    # with all their slots -- a 150-way retrieval over 15 distinct concepts, not
    # over 150. It still runs and still returns a plausible number, which is why
    # the layout has to be stated explicitly rather than inferred from the code.
    n_slots = tr_eeg.shape[1]
    v_c = split.val_concepts
    X_val_all = torch.from_numpy(
        tr_eeg[v_c].reshape(len(v_c), n_slots, -1)
    ).float()
    Y_val_all = torch.from_numpy(l2n(img_tr[v_c])).float()

    X_fit = torch.from_numpy(flat(tr_eeg, split.fit_concepts)).float()
    Y_fit = torch.from_numpy(l2n(img_tr[split.fit_concepts].reshape(-1, img_tr.shape[-1]))).float()
    X_te = torch.from_numpy(te_eeg[:, 0].reshape(te_eeg.shape[0], -1)).float()
    Y_te = torch.from_numpy(l2n(img_te[:, 0])).float()

    # z-score per feature, using fit statistics only
    mu = X_fit.mean(0, keepdim=True)
    sd = X_fit.std(0, keepdim=True).clamp_min(1e-6)
    X_fit = (X_fit - mu) / sd
    X_val_all = (X_val_all - mu) / sd
    X_te = (X_te - mu) / sd

    X_fit, Y_fit = X_fit.to(dev), Y_fit.to(dev)
    X_val_all, Y_val_all = X_val_all.to(dev), Y_val_all.to(dev)
    X_te, Y_te = X_te.to(dev), Y_te.to(dev)

    print(f"[ridge] fit={tuple(X_fit.shape)} -> {tuple(Y_fit.shape)}")
    print(f"[ridge] val={tuple(X_val_all.shape)} ({len(v_c)}-way x {n_slots} slots, averaged)   "
          f"test={tuple(X_te.shape)} (200-way, 1 slot)")
    print(flush=True)

    # Lambda is chosen on val ONLY (never on test).
    results = []
    t0 = time.time()
    print("[ridge] eigendecomposing the dual Gram matrix (the expensive step)...", flush=True)
    U, s, UtY = ridge_dual_eig(X_fit, Y_fit)
    print(f"[ridge] eigendecomposition done in {time.time()-t0:.0f}s; "
          f"cond={float(s[-1]/s[s>1e-9][0]):.2e}", flush=True)
    for lam in a.lams:
        # Validation sweeps all slots and averages (selection signal, never test).
        # Index [:, si] on the (concepts, slots, feat) tensor -- see the layout note
        # above for why a flat reshape plus a contiguous slice is not the same thing.
        t1, t5, mr = [], [], []
        for si in range(n_slots):
            P = ridge_predict_eig(X_val_all[:, si], X_fit, U, s, UtY, lam).float().cpu().numpy()
            y = Y_val_all[:, si].cpu().numpy()
            t1.append(retrieval_report(P, y)["top1"])
            t5.append(retrieval_report(P, y)["top5"])
            mr.append(mean_rank(P, y))
        results.append({"lam": lam, "split": "val",
                        "top1": float(np.mean(t1)), "top5": float(np.mean(t5)),
                        "mean_rank": float(np.mean(mr)),
                        "top1_std": float(np.std(t1))})
        # Test is scored once per lambda for reporting, and is never selected on.
        P = ridge_predict_eig(X_te, X_fit, U, s, UtY, lam).float().cpu().numpy()
        y = Y_te.cpu().numpy()
        _rep = retrieval_report(P, y)
        _rep["mean_rank"] = mean_rank(P, y)
        results.append({"lam": lam, "split": "test", **_rep})
        v = [r for r in results if r["lam"] == lam and r["split"] == "val"][0]
        print(f"[ridge] lam={lam:<9.1e} val_top1={v['top1']:6.2f} "
              f"+-{v['top1_std']:4.2f} (elapsed {time.time()-t0:.0f}s)", flush=True)

    best = max((r for r in results if r["split"] == "val"), key=lambda r: r["top1"])
    test_at_best = [r for r in results if r["split"] == "test" and r["lam"] == best["lam"]][0]

    print()
    print("=" * 74)
    print("RIDGE BASELINE (closed-form linear map, no deep encoder)")
    print("=" * 74)
    print(f"  lambda* (chosen on val)      : {best['lam']:.1e}")
    print(f"  VAL   (150-way) top1/top5    : {best['top1']:.2f} / {best['top5']:.2f}   mean rank {best['mean_rank']:.1f}")
    print(f"  TEST  (200-way) top1/top5    : {test_at_best['top1']:.2f} / {test_at_best['top5']:.2f}   mean rank {test_at_best['mean_rank']:.1f}")
    print()
    print("  Reading: the val split draws from TRAIN-side recordings (4-rep averaged),")
    print("  while test is 80-rep averaged, i.e. ~4.5x better SNR. A val score BELOW")
    print("  the test score at the same lambda is therefore expected, not a bug.")
    print("  Chance level: top1 0.5%, mean rank 100.5 (200-way).")
    print()
    print("  Deep-model comparison (ViT-B/16 layer scan, 30 epochs):")
    print("    best test top1 21.50 at layer 4, mean rank ~11.6")

    out = Path(a.out) if a.out else (config.OUTPUTS / f"sub{a.subject:02d}" / "baseline_ridge.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "subject": a.subject, "channels": a.channels, "n_channels": len(ch),
        "lambda_best_on_val": best["lam"],
        "val": {"top1": best["top1"], "top5": best["top5"], "mean_rank": best["mean_rank"],
                "top1_std": best.get("top1_std"),
                "note": "mean over all image slots; matches train.py's evaluate_selection"},
        "test": {"top1": test_at_best["top1"], "top5": test_at_best["top5"],
                 "mean_rank": test_at_best["mean_rank"]},
        "sweep": results,
        "note": "lambda chosen on val only; test reported at that lambda and not tuned",
    }, indent=2))
    print(f"\n[ridge] wrote {out}")


if __name__ == "__main__":
    main()
