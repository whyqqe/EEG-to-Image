#!/usr/bin/env python3
"""SHB step 1: export frozen-backbone `raw` features + TRIAL-LEVEL posterior.

Motivation (mechanism 1 of the Structural Hypothesis Branch):
The existing pipeline averages all EEG repetitions away (train 4 reps, test 80
reps) and then feeds ONE vector into projection heads that all share the same
backbone output. Those branches therefore fail together and cannot compensate
each other -- which is why even an ORACLE per-sample gate could not move the
Pareto frontier. The trial-to-trial variability is a *free, unused,
independent* source of evidence about structural uncertainty.

What this script does
  1. Replicates EEGPreImageDataset preprocessing EXACTLY:
     mean over reps -> channel select -> time window [0, 250] (no other norm).
  2. Exports frozen EEGProject `raw` (1024-d, l2-normalised) for:
       - full-trial mean  : raw_train (16540,1024), raw_test (200,1024)
       - trial subsets    : raw_train_sub (16540, K_TR, 1024) [K_TR=4 LOO]
                            raw_test_sub  (200,  K_TE, 1024) [K_TE=8 x 10]
  3. GATE-#0 diagnostics: is trial dispersion real signal or noise?
       (a) within-sample dispersion vs between-sample dispersion
       (b) does per-sample dispersion correlate with structural quality
           (per-sample pearson of baseline EEG->depth vs GT depth)?
     (a)+(b) holding => mechanism 1 is a genuine independent evidence source.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT))

from module.eeg_encoder.model import EEGProject  # noqa: E402

DEFAULT_CHANNELS = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]


def l2(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-8)).astype(np.float32)


def pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.reshape(len(a), -1).astype(np.float64)
    b = b.reshape(len(b), -1).astype(np.float64)
    ac, bc = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    den = np.sqrt((ac * ac).sum(1) * (bc * bc).sum(1)).clip(1e-12)
    return (ac * bc).sum(1) / den


def sel_ch_time(a: np.ndarray, ch_idx: list[int], t: int = 250) -> np.ndarray:
    """channel select on axis -2 then time window. Mirrors EEGPreImageDataset."""
    return np.take(a, ch_idx, axis=-2)[..., :t]


@torch.no_grad()
def encode(model: torch.nn.Module, x: np.ndarray, device: torch.device, bs: int = 2048) -> np.ndarray:
    model.eval()
    outs = []
    for s in tqdm(range(0, len(x), bs), desc="raw-encode", leave=False):
        xb = torch.from_numpy(np.ascontiguousarray(x[s : s + bs])).float().to(device)
        outs.append(model(xb).float().cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def dispersion(vecs: np.ndarray) -> np.ndarray:
    """vecs (N,K,D) l2-normalised -> per-sample mean pairwise (1-cos) (N,)."""
    N, K, _ = vecs.shape
    sims = np.einsum("nkd,njd->nkj", vecs, vecs)
    iu = np.triu_indices(K, k=1)
    return (1.0 - sims[:, iu[0], iu[1]]).mean(axis=1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg-dir", type=str, default=str(NB_ROOT / "data/things_eeg/preprocessed_eeg"))
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--n-test-groups", type=int, default=8)
    ap.add_argument("--gt-depth-test", type=str, default="")
    ap.add_argument("--pred-depth-test", type=str, default="")
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    eeg_dir = Path(args.eeg_dir)

    info = json.loads((eeg_dir / "info.json").read_text(encoding="utf-8"))
    all_ch = info["ch_names"]
    ch_idx = [all_ch.index(c) for c in DEFAULT_CHANNELS]
    C = len(ch_idx)
    print(f"[INFO] channels kept: {C}/{len(all_ch)}")

    tr = np.load(eeg_dir / f"sub-{args.subject:02}" / "train.npy")   # (1654,10,4,63,250)
    te = np.load(eeg_dir / f"sub-{args.subject:02}" / "test.npy")    # (200,1,80,63,250)
    print(f"[INFO] train.npy {tr.shape} test.npy {te.shape}")
    R_tr, R_te = tr.shape[2], te.shape[2]

    # channel-select FIRST (keeps peak memory ~5GB instead of ~20GB), then time window
    tr = np.take(tr, ch_idx, axis=-2)[..., :250]        # (1654,10,4,17,250)
    te = np.take(te, ch_idx, axis=-2)[..., :250]
    print(f"[INFO] after channel select: train {tr.shape} test {te.shape}")

    # ---------- full-trial means (exactly what the current pipeline feeds) ----------
    tr_full = tr.mean(axis=2).reshape(-1, C, 250)
    te_full = te.mean(axis=2).reshape(-1, C, 250)
    print(f"[INFO] full-trial EEG: train {tr_full.shape} test {te_full.shape}")

    # ---------- trial subsets (independent evidence) ----------
    tr_groups = [np.array([k for k in range(R_tr) if k != j]) for j in range(R_tr)]   # LOO over 4 reps
    te_groups = np.array_split(np.arange(R_te), args.n_test_groups)                    # 8 x 10 of 80 reps
    tr_sub = np.stack([tr[..., g, :, :].mean(axis=-3) for g in tr_groups], axis=0).reshape(len(tr_groups), -1, C, 250)
    te_sub = np.stack([te[..., g, :, :].mean(axis=-3) for g in te_groups], axis=0).reshape(len(te_groups), -1, C, 250)
    print(f"[INFO] subsets: train {tr_sub.shape} test {te_sub.shape}")
    del tr, te

    # ---------- frozen backbone ----------
    model = EEGProject(feature_dim=1024, eeg_sample_points=250, channels_num=C).to(device)
    ckpt = torch.load(Path(args.checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[INFO] loaded EEGProject from {args.checkpoint}")

    raw_tr = encode(model, tr_full, device)
    raw_te = encode(model, te_full, device)
    raw_tr_sub = np.stack([encode(model, tr_sub[k], device) for k in range(len(tr_groups))], axis=1)
    raw_te_sub = np.stack([encode(model, te_sub[k], device) for k in range(len(te_groups))], axis=1)

    raw_tr, raw_te = l2(raw_tr), l2(raw_te)
    raw_tr_sub, raw_te_sub = l2(raw_tr_sub), l2(raw_te_sub)
    np.save(out / "raw_train.npy", raw_tr)
    np.save(out / "raw_test.npy", raw_te)
    np.save(out / "raw_train_sub.npy", raw_tr_sub)
    np.save(out / "raw_test_sub.npy", raw_te_sub)
    print(f"[OK] raw_train {raw_tr.shape} raw_test {raw_te.shape} "
          f"raw_train_sub {raw_tr_sub.shape} raw_test_sub {raw_te_sub.shape}")

    # ---------------- GATE #0 ----------------
    within = dispersion(raw_te_sub)
    sim_bt = raw_te @ raw_te.T
    iu = np.triu_indices(len(raw_te), k=1)
    between = float((1.0 - sim_bt[iu[0], iu[1]]).mean())

    diag = {
        "pipeline": "shb_export_raw",
        "subject": args.subject,
        "n_train": int(len(raw_tr)), "n_test": int(len(raw_te)),
        "k_train_subsets": int(raw_tr_sub.shape[1]),
        "k_test_subsets": int(raw_te_sub.shape[1]),
        "within_disp_mean": float(within.mean()),
        "within_disp_std": float(within.std()),
        "between_disp": between,
        "ratio_within_over_between": float(within.mean() / max(between, 1e-8)),
        "gate0_note": "ratio << 1 => subsets agree far more with each other than samples differ "
                      "=> trial evidence is reproducible, not pure noise",
    }

    if args.gt_depth_test and args.pred_depth_test:
        gt = np.load(args.gt_depth_test).astype(np.float32)
        pd = np.load(args.pred_depth_test).astype(np.float32)
        if len(gt) == len(within) and len(pd) == len(within):
            q = pearson_rows(pd, gt)
            from scipy.stats import pearsonr, spearmanr
            r_p, p_p = pearsonr(within, q)
            r_s, p_s = spearmanr(within, q)
            diag.update({
                "struct_quality_mean": float(q.mean()),
                "corr_disp_vs_structquality_pearson": float(r_p),
                "corr_disp_vs_structquality_pearson_p": float(p_p),
                "corr_disp_vs_structquality_spearman": float(r_s),
                "corr_disp_vs_structquality_spearman_p": float(p_s),
                "gate0_verdict": ("PASS: dispersion carries structural signal (negative corr)"
                                  if (r_p < 0 and p_p < 0.05)
                                  else "WEAK/FAIL: dispersion not informative about structure"),
            })
        else:
            diag["gate0_verdict"] = f"skipped shape mismatch gt={gt.shape} pd={pd.shape}"
    else:
        diag["gate0_verdict"] = "skipped (no gt/pred depth passed)"

    (out / "trial_posterior_report.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
    print(json.dumps(diag, indent=2))


if __name__ == "__main__":
    main()
