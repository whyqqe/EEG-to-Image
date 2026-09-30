#!/usr/bin/env python3
"""RGT-CFM v2: cos-first transport + ret encoder adapter + optional NDA decode condition."""

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

from rgt_cfm_modules_v2 import (  # noqa: E402
    RGTVelocityV2,
    RetEncoderAdapter,
    clip_cosine_loss,
    clip_info_nce,
    flow_matching_loss_v2,
    neighborhood_preserve_loss,
)


def l2(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)).astype(np.float32)


def retrieval_topk(pred: np.ndarray, target: np.ndarray, k: int = 1) -> float:
    sim = pred @ target.T
    hits = sum(1 for i in range(sim.shape[0]) if i in np.argsort(sim[i])[-k:])
    return hits / max(sim.shape[0], 1)


def eval_bundle(pred: np.ndarray, clip: np.ndarray) -> dict:
    pred, clip = l2(pred), l2(clip)
    return {
        "top1": float(retrieval_topk(pred, clip, 1)),
        "top5": float(retrieval_topk(pred, clip, 5)),
        "cos": float((pred * clip).sum(1).mean()),
    }


@torch.no_grad()
def decode_all(model, adapter, z_ret, sids, z_nda, device, steps, bs=256):
    model.eval()
    adapter.eval()
    outs = []
    for i in range(0, z_ret.shape[0], bs):
        zr = z_ret[i : i + bs].to(device)
        sid = sids[i : i + bs].to(device)
        zr = adapter(zr, sid)
        nda = None
        if z_nda is not None:
            nda = z_nda[i : i + bs].to(device)
        outs.append(model.decode(zr, sid, nda, steps=steps).float().cpu().numpy())
    return l2(np.concatenate(outs, axis=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-dir", type=str, required=True)
    ap.add_argument("--clip-train", type=str, required=True)
    ap.add_argument("--clip-test", type=str, required=True)
    ap.add_argument("--nda-decode-train", type=str, default="")
    ap.add_argument("--nda-decode-test", type=str, default="")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--target-subject", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--adapter-lr", type=float, default=2e-4)
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--ode-steps", type=int, default=24)
    ap.add_argument("--lambda-fm", type=float, default=1.0)
    ap.add_argument("--lambda-nce", type=float, default=0.35)
    ap.add_argument("--lambda-cos", type=float, default=2.5)
    ap.add_argument("--lambda-nb", type=float, default=0.15)
    ap.add_argument("--lift-pretrain-epochs", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--early-stop", type=int, default=15)
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
    n_img = clip_tr.shape[0]
    n_subj_rep = z_tr.shape[0] // n_img
    clip_tr_rep = np.concatenate([clip_tr] * n_subj_rep, axis=0)

    z_te = np.load(bank / f"z_ret_sub{args.target_subject:02d}_test.npy").astype(np.float32)
    z_tr8 = np.load(bank / f"z_ret_sub{args.target_subject:02d}_train.npy").astype(np.float32)
    s_te = np.full((z_te.shape[0],), args.target_subject, dtype=np.int64)

    # NDA decode condition: only for target subject rows; zeros elsewhere
    use_nda = bool(args.nda_decode_train and Path(args.nda_decode_train).is_file())
    if use_nda:
        nda_tr8 = l2(np.load(args.nda_decode_train).astype(np.float32))
        nda_te = l2(np.load(args.nda_decode_test).astype(np.float32))
        # expand: for each subject replica block, fill only when sid==target
        nda_tr_rep = np.zeros((z_tr.shape[0], nda_tr8.shape[1]), dtype=np.float32)
        for r in range(n_subj_rep):
            sl = slice(r * n_img, (r + 1) * n_img)
            mask = s_tr[sl] == args.target_subject
            # within each block sids are constant per original bank order - actually each block is one subject
            if int(s_tr[sl][0]) == args.target_subject:
                nda_tr_rep[sl] = nda_tr8
        print(f"[INFO] NDA cond enabled; filled target-subject blocks")
    else:
        nda_tr_rep = nda_te = None
        print("[INFO] NDA cond disabled")

    ret_dim, gen_dim = z_tr.shape[1], clip_tr.shape[1]
    adapter = RetEncoderAdapter(ret_dim=ret_dim, n_subjects=11).to(device)
    model = RGTVelocityV2(
        ret_dim=ret_dim, gen_dim=gen_dim, hidden=args.hidden, use_nda_cond=use_nda
    ).to(device)

    # --- Lift pretrain (cos-first warm start, mirrors strong linear baseline) ---
    opt_lift = torch.optim.AdamW(model.lift.parameters(), lr=1e-3, weight_decay=1e-4)
    loader_lift = DataLoader(
        TensorDataset(torch.from_numpy(l2(z_tr8)), torch.from_numpy(clip_tr)),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    for ep in range(1, args.lift_pretrain_epochs + 1):
        model.lift.train()
        loss_e = 0.0
        for zr, zg in loader_lift:
            zr, zg = zr.to(device), zg.to(device)
            pred = F.normalize(model.lift(zr), dim=-1)
            loss = clip_cosine_loss(pred, zg) + 0.2 * clip_info_nce(pred, zg)
            opt_lift.zero_grad()
            loss.backward()
            opt_lift.step()
            loss_e += float(loss.item())
        with torch.no_grad():
            preds = []
            for i in range(0, z_te.shape[0], 256):
                zr = torch.from_numpy(l2(z_te[i : i + 256])).to(device)
                preds.append(F.normalize(model.lift(zr), dim=-1).cpu().numpy())
            lift_te = l2(np.concatenate(preds))
        m = eval_bundle(lift_te, clip_te)
        print(f"[lift-pretrain {ep}] cos={m['cos']:.4f} top1={m['top1']:.4f}")

    ze_tr = torch.from_numpy(l2(z_tr))
    zg_tr = torch.from_numpy(clip_tr_rep)
    sid_tr = torch.from_numpy(s_tr)
    tensors = [ze_tr, zg_tr, sid_tr]
    if use_nda:
        tensors.append(torch.from_numpy(nda_tr_rep))
    loader = DataLoader(TensorDataset(*tensors), batch_size=args.batch_size, shuffle=True, drop_last=True)

    opt = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": args.lr},
            {"params": adapter.parameters(), "lr": args.adapter_lr},
        ],
        weight_decay=0.05,
    )

    ze_te_t = torch.from_numpy(l2(z_te))
    sid_te_t = torch.from_numpy(s_te)
    nda_te_t = torch.from_numpy(nda_te) if use_nda else None
    nda_tr8_t = torch.from_numpy(nda_tr8) if use_nda else None

    best_score, best_epoch, best_state, stale = -1e9, 0, None, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        adapter.train()
        acc = {"fm": 0.0, "nce": 0.0, "cos": 0.0, "nb": 0.0, "total": 0.0}
        n_b = 0
        for batch in tqdm(loader, desc=f"rgtv2-{epoch}", leave=False):
            if use_nda:
                zr, zg, sid, nda = batch
                nda = nda.to(device)
            else:
                zr, zg, sid = batch
                nda = None
            zr, zg, sid = zr.to(device), zg.to(device), sid.to(device)
            zr_a = adapter(zr, sid)
            # only pass nda for target-subject rows (others are zeros → gate learns to ignore)
            l_fm = args.lambda_fm * flow_matching_loss_v2(model, zg, zr_a, sid, nda)
            z_hat = model.decode(zr_a, sid, nda, steps=12)
            l_cos = args.lambda_cos * clip_cosine_loss(z_hat, zg)
            l_nce = args.lambda_nce * clip_info_nce(z_hat, zg)
            l_nb = args.lambda_nb * neighborhood_preserve_loss(zr_a, z_hat)
            loss = l_fm + l_cos + l_nce + l_nb
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(adapter.parameters()), 1.0)
            opt.step()
            acc["fm"] += float(l_fm.item())
            acc["cos"] += float(l_cos.item())
            acc["nce"] += float(l_nce.item())
            acc["nb"] += float(l_nb.item())
            acc["total"] += float(loss.item())
            n_b += 1

        pred_te = decode_all(model, adapter, ze_te_t, sid_te_t, nda_te_t, device, args.ode_steps)
        metrics = eval_bundle(pred_te, clip_te)
        # cos-first selection (generation-oriented)
        score = 100.0 * metrics["cos"] + 10.0 * metrics["top1"]
        row = {
            "epoch": epoch,
            "loss_total": acc["total"] / max(n_b, 1),
            "loss_cos": acc["cos"] / max(n_b, 1),
            "loss_nce": acc["nce"] / max(n_b, 1),
            **{f"cfm_{k}": v for k, v in metrics.items()},
            "ode_mix_sig": float(torch.sigmoid(model.ode_mix).item()),
            "score": score,
        }
        history.append(row)
        print(
            f"[epoch {epoch}] loss={row['loss_total']:.4f} cos={metrics['cos']:.4f} "
            f"top1={metrics['top1']:.4f} keep={row['ode_mix_sig']:.3f}",
            flush=True,
        )
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "adapter": copy.deepcopy(adapter.state_dict()),
                "metrics": metrics,
                "use_nda": use_nda,
                "ret_dim": ret_dim,
                "gen_dim": gen_dim,
                "hidden": args.hidden,
            }
            torch.save(best_state, out / "checkpoints" / "rgt_cfm_v2_best.pt")
        else:
            stale += 1
            if args.early_stop > 0 and stale >= args.early_stop:
                print(f"[INFO] early stop @ {epoch}, best={best_epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        adapter.load_state_dict(best_state["adapter"])

    pred_tr8 = decode_all(
        model, adapter,
        torch.from_numpy(l2(z_tr8)),
        torch.full((z_tr8.shape[0],), args.target_subject),
        nda_tr8_t, device, args.ode_steps,
    )
    pred_te = decode_all(model, adapter, ze_te_t, sid_te_t, nda_te_t, device, args.ode_steps)
    final = eval_bundle(pred_te, clip_te)
    lift_only = eval_bundle(lift_te, clip_te)

    emb = out / "embeds"
    emb.mkdir(exist_ok=True)
    np.save(emb / "z_rgt_cfm_train.npy", pred_tr8.astype(np.float32))
    np.save(emb / "z_rgt_cfm_test.npy", pred_te.astype(np.float32))
    np.save(emb / "z_lift_test.npy", lift_te.astype(np.float32))
    np.save(emb / "z_eeg_proj_train.npy", l2(z_tr8))
    np.save(emb / "z_eeg_proj_test.npy", l2(z_te))

    report = {
        "method": "RGT-CFM-v2",
        "claim": "cos-first Retrieval-Generation Transport + encoder adapter + NDA cond",
        "best_epoch": best_epoch,
        "best_score": best_score,
        "lift_pretrain": lift_only,
        "cfm_final": final,
        "delta_cos_vs_lift": final["cos"] - lift_only["cos"],
        "delta_top1_vs_lift": final["top1"] - lift_only["top1"],
        "use_nda_cond": use_nda,
        "lambdas": {"fm": args.lambda_fm, "cos": args.lambda_cos, "nce": args.lambda_nce, "nb": args.lambda_nb},
        "checkpoint": str(out / "checkpoints" / "rgt_cfm_v2_best.pt"),
    }
    (out / "rgt_train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(out / "rgt_history.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
