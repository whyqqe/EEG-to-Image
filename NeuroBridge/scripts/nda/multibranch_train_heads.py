#!/usr/bin/env python3
"""Train EEG → {depth, edge} CLIP heads for multi-branch IP-Adapter conditioning.

WHY
---
CogCapPro (arXiv:2603.12722) reaches SSIM 0.398 / Inception 0.779 by injecting
THREE parallel IP-Adapter branches (image, depth, edge) instead of chaining
structure through ControlNet + SDEdit. Their ablation shows the depth and edge
branches are what lift SSIM (image-only 0.317 -> all 0.398).

Our pipeline has depth as an RGB map (ControlNet) but no CLIP-space depth/edge
condition. This trains it, so the depth/edge conditions live in the SAME space
the IP-Adapter projection expects (CLIP ViT-H-14, the encoder used to build
outputs/gem/cond_cache/clip_{depth,edge}1024_*.npy).

LEAK-FREE
---------
Gradients on `fit` concepts, checkpoint selected on `val_b` concepts, both from
outputs/leakfree/split.json. The 200 test concepts only ever appear at the final
export. Calibration reference is the TRAIN bank of the same modality.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocf_train import l2n, l2t, calibrate_quantile  # noqa: E402
import leakfree as LF  # noqa: E402

MODALITIES = {
    "depth": "clip_depth1024",
    "edge": "clip_edge1024",
    "image": "clip_img1024",
}


class Head(nn.Module):
    def __init__(self, dim: int = 1024, hidden: int = 1024, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        d = dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers += [nn.Linear(d, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def cos_loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    return (1.0 - (l2t(pred) * l2t(tgt)).sum(-1)).mean()


def train_one(mod: str, ztr: np.ndarray, ttr: np.ndarray,
              fit_i: np.ndarray, val_i: np.ndarray,
              args, dev: torch.device, out: Path) -> tuple[Head, dict]:
    torch.manual_seed(args.seed)
    head = Head(depth=args.depth).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    z_t = torch.from_numpy(ztr).to(dev)
    t_t = torch.from_numpy(ttr).to(dev)
    fit_t = torch.from_numpy(fit_i.astype(np.int64)).to(dev)
    val_t = torch.from_numpy(val_i.astype(np.int64)).to(dev)

    best = {"val_cos": -2.0, "epoch": -1}
    hist = []
    for ep in range(args.epochs):
        head.train()
        perm = fit_t[torch.randperm(len(fit_t), device=dev)]
        tot = 0.0
        n = 0
        for s in range(0, len(perm), args.batch_size):
            idx = perm[s:s + args.batch_size]
            opt.zero_grad(set_to_none=True)
            loss = cos_loss(head(z_t[idx]), t_t[idx])
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            n += 1
        head.eval()
        with torch.no_grad():
            pv = head(z_t[val_t])
            vcos = float((l2t(pv) * l2t(t_t[val_t])).sum(-1).mean().cpu())
            # retrieval: is the paired target the nearest train target? (val rows only)
        row = {"epoch": ep, "train_cos_loss": tot / max(n, 1), "val_cos": vcos}
        hist.append(row)
        if vcos > best["val_cos"]:
            best = {"val_cos": vcos, "epoch": ep}
            torch.save({"state_dict": head.state_dict(), "epoch": ep,
                        "val_cos": vcos, "modality": mod}, args.out / f"head_{mod}.pth")
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"[{mod} ep{ep:02d}] train_cos_loss={row['train_cos_loss']:.4f} "
                  f"val_cos={vcos:.4f}")
    head.load_state_dict(torch.load(args.out / f"head_{mod}.pth",
                                    map_location=dev, weights_only=False)["state_dict"])
    head.eval()
    return head, {"best": best, "history_tail": hist[-4:]}


def retrieval_top1(pred: np.ndarray, bank: np.ndarray, rows: np.ndarray,
                   labels: np.ndarray) -> float:
    """Fraction of `rows` whose paired target is the nearest row of `bank`."""
    p = l2n(pred[rows]) @ l2n(bank).T
    top = p.argmax(1)
    return float((labels[rows] == labels[top]).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--modalities", type=str, default="depth,edge")
    ap.add_argument("--test-subject", type=int, default=8)
    ap.add_argument("--z-root", type=str, default=str(NB_ROOT / "outputs/ocf/intra_z"))
    ap.add_argument("--cond-cache", type=str, default=str(NB_ROOT / "outputs/gem/cond_cache"))
    ap.add_argument("--split-json", type=str, default=str(NB_ROOT / "outputs/leakfree/split.json"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--depth", type=int, default=2, help="MLP depth (hidden blocks)")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "conds").mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sid = f"{args.test_subject:02d}"
    mods = [m.strip() for m in args.modalities.split(",") if m.strip()]

    ztr = l2n(np.load(Path(args.z_root) / f"sub-{sid}" / "shared_r_train.npy").astype(np.float32))
    zte = l2n(np.load(Path(args.z_root) / f"sub-{sid}" / "shared_r_test.npy").astype(np.float32))
    split = LF.load(args.split_json)
    fit_i = LF.rows_for(split, "fit", len(ztr))
    val_i = LF.rows_for(split, "val_b", len(ztr))
    if set(fit_i.tolist()) & set(val_i.tolist()):
        raise SystemExit("[FATAL] fit/val overlap")

    # concept id per row (train only); used only for the retrieval diagnostic
    from ocf_train import build_concept_bank
    _, cid_tr, phrases = build_concept_bank(
        Path(str(NB_ROOT / f"outputs/nda_ss/sub-{sid}/clip_text")),
        Path(str(NB_ROOT / "outputs/g2/captions/captions_train.jsonl")))
    if len(cid_tr) != len(ztr):
        raise SystemExit(f"[FATAL] cid {len(cid_tr)} vs z {len(ztr)}")

    report: dict = {"subject": f"sub-{sid}", "modalities": {}, "params": vars(args)}
    for mod in mods:
        base = MODALITIES[mod]
        ttr = l2n(np.load(Path(args.cond_cache) / f"{base}_train.npy").astype(np.float32))
        tte = l2n(np.load(Path(args.cond_cache) / f"{base}_test.npy").astype(np.float32))
        if len(ttr) != len(ztr) or len(tte) != len(zte):
            raise SystemExit(f"[FATAL] {mod} rows train {len(ttr)} test {len(tte)}")

        head, tinfo = train_one(mod, ztr, ttr, fit_i, val_i, args, dev)
        with torch.no_grad():
            pte = head(torch.from_numpy(zte).to(dev)).cpu().numpy().astype(np.float32)
            ptr = head(torch.from_numpy(ztr).to(dev)).cpu().numpy().astype(np.float32)

        # sanity: paired-target cosine on the held-out VAL concepts (leak-free)
        val_cos = float((l2n(ptr[val_i]) * l2n(ttr[val_i])).sum(1).mean())
        # shuffled control
        rng = np.random.default_rng(args.seed)
        shuf = rng.permutation(val_i)
        val_cos_shuf = float((l2n(ptr[val_i]) * l2n(ttr[shuf])).sum(1).mean())
        val_ret_top1 = retrieval_top1(ptr, ttr, val_i, cid_tr)

        pte_cal, cal_rep = calibrate_quantile(pte, ttr, two_sided=True)
        np.save(out / "conds" / f"{mod}_pred_test.npy", l2n(pte).astype(np.float32))
        np.save(out / "conds" / f"{mod}_pred_test_cal.npy", l2n(pte_cal).astype(np.float32))
        np.save(out / "conds" / f"{mod}_gt_test.npy", tte)

        report["modalities"][mod] = {
            **tinfo,
            "val_cos_held": val_cos,
            "val_cos_shuffled_control": val_cos_shuf,
            "val_retrieval_top1_within_concept": val_ret_top1,
            "val_cos_gain_over_shuffle": val_cos - val_cos_shuf,
            "calibration": {k: cal_rep[k] for k in cal_rep
                            if not isinstance(cal_rep[k], (np.ndarray, list))},
            "chance_retrieval": 1.0 / max(len(set(cid_tr.tolist())), 1),
            "NOTE": "val_b concepts only; test targets never used for selection",
        }
        print(f"[{mod}] val_cos={val_cos:.4f} shuffled={val_cos_shuf:.4f} "
              f"ret_top1={val_ret_top1:.4f} best_ep={tinfo['best']['epoch']}")

    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["modalities"], indent=2)[:2000])
    print(f"[heads] wrote {out}")


if __name__ == "__main__":
    main()
