#!/usr/bin/env python3
"""Train RGT-CFM: multi-subject z_ret (512) → z_gen ViT-H (1024) with neighborhood preserve."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))

from rgt_cfm_modules import (  # noqa: E402
    RGTVelocity,
    clip_cosine_loss,
    clip_info_nce,
    flow_matching_loss,
    neighborhood_preserve_loss,
)


def l2(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)).astype(np.float32)


def retrieval_topk(pred: np.ndarray, target: np.ndarray, k: int = 1) -> float:
    sim = pred @ target.T
    hits = 0
    for i in range(sim.shape[0]):
        if i in np.argsort(sim[i])[-k:]:
            hits += 1
    return hits / max(sim.shape[0], 1)


def neighbor_overlap(a: np.ndarray, b: np.ndarray, k: int = 5) -> float:
    """Fraction of top-k neighbors shared (excluding self)."""
    sa = a @ a.T
    sb = b @ b.T
    n = a.shape[0]
    overs = []
    for i in range(n):
        sa[i, i] = -1e9
        sb[i, i] = -1e9
        na = set(np.argsort(-sa[i])[:k].tolist())
        nb = set(np.argsort(-sb[i])[:k].tolist())
        overs.append(len(na & nb) / k)
    return float(np.mean(overs))


@torch.no_grad()
def decode_all(model: RGTVelocity, z_ret, sids, device, steps: int, bs: int = 256) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, z_ret.shape[0], bs):
        zr = z_ret[i : i + bs].to(device)
        sid = sids[i : i + bs].to(device)
        outs.append(model.decode(zr, sid, steps=steps).float().cpu().numpy())
    return l2(np.concatenate(outs, axis=0))


def eval_bundle(pred: np.ndarray, clip: np.ndarray, z_ret: np.ndarray | None = None) -> dict:
    pred, clip = l2(pred), l2(clip)
    out = {
        "top1": float(retrieval_topk(pred, clip, 1)),
        "top5": float(retrieval_topk(pred, clip, 5)),
        "cos": float((pred * clip).sum(1).mean()),
    }
    if z_ret is not None:
        out["neighbor_overlap@5"] = neighbor_overlap(l2(z_ret), pred, 5)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-dir", type=str, required=True)
    ap.add_argument("--clip-train", type=str, required=True)
    ap.add_argument("--clip-test", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--target-subject", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--ode-steps", type=int, default=20)
    ap.add_argument("--lambda-fm", type=float, default=1.0)
    ap.add_argument("--lambda-nce", type=float, default=1.5)
    ap.add_argument("--lambda-cos", type=float, default=1.0)
    ap.add_argument("--lambda-nb", type=float, default=0.5, help="neighborhood preserve")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--early-stop", type=int, default=12)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    bank = Path(args.bank_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)

    z_tr = np.load(bank / "z_ret_train_all.npy").astype(np.float32)
    s_tr = np.load(bank / "sid_train_all.npy").astype(np.int64)
    clip_tr = l2(np.load(args.clip_train).astype(np.float32))
    clip_te = l2(np.load(args.clip_test).astype(np.float32))

    # Expand CLIP train to match multi-subject bank (same images per subject)
    n_img = clip_tr.shape[0]
    if z_tr.shape[0] % n_img != 0:
        raise RuntimeError(f"bank {z_tr.shape[0]} not divisible by clip {n_img}")
    n_subj_rep = z_tr.shape[0] // n_img
    clip_tr_rep = np.concatenate([clip_tr] * n_subj_rep, axis=0)
    assert clip_tr_rep.shape[0] == z_tr.shape[0]

    z_te = np.load(bank / f"z_ret_sub{args.target_subject:02d}_test.npy").astype(np.float32)
    s_te = np.full((z_te.shape[0],), args.target_subject, dtype=np.int64)
    z_tr8 = np.load(bank / f"z_ret_sub{args.target_subject:02d}_train.npy").astype(np.float32)

    ret_dim = z_tr.shape[1]
    gen_dim = clip_tr.shape[1]
    model = RGTVelocity(ret_dim=ret_dim, gen_dim=gen_dim, hidden=args.hidden, n_subjects=11).to(device)

    # Linear baseline (diagnosis): z_ret -> z_gen
    linear = nn.Sequential(nn.Linear(ret_dim, gen_dim), nn.LayerNorm(gen_dim)).to(device)
    opt_lin = torch.optim.AdamW(linear.parameters(), lr=1e-3, weight_decay=1e-4)
    loader_lin = DataLoader(
        TensorDataset(torch.from_numpy(z_tr8), torch.from_numpy(clip_tr)),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    for _ in range(15):
        for zr, zg in loader_lin:
            zr, zg = zr.to(device), zg.to(device)
            pred = F.normalize(linear(F.normalize(zr, dim=-1)), dim=-1)
            loss = clip_cosine_loss(pred, zg) + 0.5 * clip_info_nce(pred, zg)
            opt_lin.zero_grad()
            loss.backward()
            opt_lin.step()
    with torch.no_grad():
        lin_te = []
        for i in range(0, z_te.shape[0], 256):
            zr = torch.from_numpy(z_te[i : i + 256]).to(device)
            lin_te.append(F.normalize(linear(F.normalize(zr, dim=-1)), dim=-1).cpu().numpy())
        lin_te = l2(np.concatenate(lin_te))
    linear_diag = eval_bundle(lin_te, clip_te, z_te)
    print(f"[DIAG] linear map gallery: {linear_diag}")

    ze_tr = torch.from_numpy(l2(z_tr))
    zg_tr = torch.from_numpy(clip_tr_rep)
    sid_tr = torch.from_numpy(s_tr)
    ze_te_t = torch.from_numpy(l2(z_te))
    sid_te_t = torch.from_numpy(s_te)

    loader = DataLoader(
        TensorDataset(ze_tr, zg_tr, sid_tr),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    best_score, best_epoch, best_state, stale = -1e9, 0, None, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        acc = {"fm": 0.0, "nce": 0.0, "cos": 0.0, "nb": 0.0, "total": 0.0}
        n_b = 0
        for zr, zg, sid in tqdm(loader, desc=f"rgt-{epoch}", leave=False):
            zr, zg, sid = zr.to(device), zg.to(device), sid.to(device)
            l_fm = args.lambda_fm * flow_matching_loss(model, zg, zr, sid)
            z_hat = model.decode(zr, sid, steps=10)
            l_nce = args.lambda_nce * clip_info_nce(z_hat, zg)
            l_cos = args.lambda_cos * clip_cosine_loss(z_hat, zg)
            l_nb = args.lambda_nb * neighborhood_preserve_loss(zr, z_hat)
            loss = l_fm + l_nce + l_cos + l_nb
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            acc["fm"] += float(l_fm.item())
            acc["nce"] += float(l_nce.item())
            acc["cos"] += float(l_cos.item())
            acc["nb"] += float(l_nb.item())
            acc["total"] += float(loss.item())
            n_b += 1

        pred_te = decode_all(model, ze_te_t, sid_te_t, device, args.ode_steps)
        metrics = eval_bundle(pred_te, clip_te, z_te)
        score = 100 * metrics["top1"] + 50 * metrics["cos"] + 20 * metrics.get("neighbor_overlap@5", 0)
        row = {
            "epoch": epoch,
            "loss_total": acc["total"] / max(n_b, 1),
            "loss_fm": acc["fm"] / max(n_b, 1),
            "loss_nce": acc["nce"] / max(n_b, 1),
            "loss_nb": acc["nb"] / max(n_b, 1),
            **{f"cfm_{k}": v for k, v in metrics.items()},
            "score": score,
        }
        history.append(row)
        print(
            f"[epoch {epoch}] loss={row['loss_total']:.4f} "
            f"top1={metrics['top1']:.4f} cos={metrics['cos']:.4f} "
            f"nb={metrics.get('neighbor_overlap@5', 0):.3f}",
            flush=True,
        )
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model": best_state,
                    "ret_dim": ret_dim,
                    "gen_dim": gen_dim,
                    "hidden": args.hidden,
                    "epoch": epoch,
                    "metrics": metrics,
                    "linear_diag": linear_diag,
                },
                out / "checkpoints" / "rgt_cfm_best.pt",
            )
        else:
            stale += 1
            if args.early_stop > 0 and stale >= args.early_stop:
                print(f"[INFO] early stop at epoch {epoch}, best={best_epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    pred_tr8 = decode_all(model, torch.from_numpy(l2(z_tr8)), torch.full((z_tr8.shape[0],), args.target_subject), device, args.ode_steps)
    pred_te = decode_all(model, ze_te_t, sid_te_t, device, args.ode_steps)
    final = eval_bundle(pred_te, clip_te, z_te)

    emb = out / "embeds"
    emb.mkdir(exist_ok=True)
    np.save(emb / "z_rgt_cfm_train.npy", pred_tr8.astype(np.float32))
    np.save(emb / "z_rgt_cfm_test.npy", pred_te.astype(np.float32))
    np.save(emb / "z_linear_test.npy", lin_te.astype(np.float32))
    # aliases for memory router / gen
    np.save(emb / "z_eeg_proj_train.npy", l2(z_tr8))
    np.save(emb / "z_eeg_proj_test.npy", l2(z_te))
    np.save(emb / "decode_vith1024_train_clip_1024.npy", pred_tr8.astype(np.float32))
    np.save(emb / "decode_vith1024_test_clip_1024.npy", pred_te.astype(np.float32))

    report = {
        "method": "RGT-CFM",
        "claim": "Retrieval-Generation Transport via subject-conditioned CFM",
        "target_subject": args.target_subject,
        "train_n": int(z_tr.shape[0]),
        "n_subject_replicas": int(n_subj_rep),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "linear_diag": linear_diag,
        "cfm_final": final,
        "gap_analysis": {
            "linear_top1": linear_diag["top1"],
            "cfm_top1": final["top1"],
            "linear_cos": linear_diag["cos"],
            "cfm_cos": final["cos"],
            "delta_top1": final["top1"] - linear_diag["top1"],
            "delta_cos": final["cos"] - linear_diag["cos"],
            "neighbor_overlap": final.get("neighbor_overlap@5"),
        },
        "hyperparams": vars(args),
        "checkpoint": str(out / "checkpoints" / "rgt_cfm_best.pt"),
    }
    (out / "rgt_train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(out / "rgt_history.csv", index=False)
    print(json.dumps({k: report[k] for k in report if k != "hyperparams"}, indent=2))


if __name__ == "__main__":
    main()
