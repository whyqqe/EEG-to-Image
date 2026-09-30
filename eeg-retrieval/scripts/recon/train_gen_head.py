#!/usr/bin/env python3
"""Train SAMGA-R's generation head: SAMGA's 1024-d encoder output -> CLIP-H/14 image_embeds.

THE CONTRACT
------------
Train on source subjects only. The held-out subject contributes nothing here -- not its
EEG, not its features, not the choice of hyperparameters. That is what makes the resulting
number a LOSO number, and it is the one property of this pipeline that cannot be relaxed
without invalidating the comparison it exists to make.

WHAT THE LOSSES DO, AND WHY ALL THREE
-------------------------------------
    cosine  aligns each prediction with its own target. Necessary, and on its own it is
            what most reconstruction work reports as "CLIP score".
    mse     matches magnitudes, which cosine ignores. Without it the predictions can be
            direction-perfect but badly scaled for a decoder whose image projection was
            trained on real CLIP embeddings with real norms.
    infonce gives the batch *rank* structure. This is the term that matters most for this
            particular model: SAMGA's advantage is inter-subject retrieval, and the
            downstream pipeline (neighbour lookup, brain-consistency re-selection, 2-way
            identification) all depend on the prediction being discriminative against the
            other 16539 gallery images, not merely close to its own target. Cosine alone
            leaves that uncontrolled. It is also what ENIGMA optimises (MSE + InfoNCE).

VALIDATION IS SOURCE-DOMAIN ONLY
--------------------------------
The held-out split below is 5% of *source* concepts. It exists to detect overfitting and
to pick a checkpoint. It is not the target subject and must never be reported as the
model's result. The `identity` diagnostic printed at the start is the important one to
read first: it is the paired cosine you get with no head at all, i.e. how far SAMGA's
retrieval-aligned features already sit from CLIP space. If that number is already high,
the head is a small correction and the interesting content is upstream in the encoder; if
it is near zero, the head is doing the whole job and its capacity is the binding
constraint.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "recon"))

from samga_recon import GenHead  # noqa: E402

DEFAULT_CLIP_TRAIN = REPO / "data" / "image_feature" / "clip_h14_ip_adapter" / "clip_h14_train.npy"


def load_sources(patterns: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Concatenate source-subject feature files into (x, object_idx, image_idx)."""
    paths: list[str] = []
    for pat in patterns:
        hits = sorted(glob.glob(pat)) or ([pat] if Path(pat).is_file() else [])
        if not hits:
            raise SystemExit(f"[FATAL] no feature files match '{pat}'")
        paths.extend(h for h in hits if h not in paths)
    xs, ois, iis, used = [], [], [], []
    for p in paths:
        z = np.load(p)
        xs.append(z["hidden"].astype(np.float32))
        ois.append(z["object_idx"].astype(np.int64))
        iis.append(z["image_idx"].astype(np.int64))
        used.append(p)
        print(f"[INFO] source {Path(p).name}: hidden={z['hidden'].shape}")
    return (np.concatenate(xs, 0), np.concatenate(ois, 0), np.concatenate(iis, 0), used)


def build_pairs(x: np.ndarray, oi: np.ndarray, ii: np.ndarray,
                clip: np.ndarray) -> np.ndarray:
    """Index the CLIP target array by stimulus identity, not by row position."""
    if clip.ndim == 3:                      # (Nconcept, Nimg, D)
        if oi.max() >= clip.shape[0] or ii.max() >= clip.shape[1]:
            raise SystemExit(
                f"[FATAL] feature file references stimulus ({oi.max()}, {ii.max()}) but the "
                f"CLIP target array is {clip.shape[:2]}. The two were built from different "
                f"splits."
            )
        return clip[oi, ii].astype(np.float32)
    if clip.ndim == 2:                      # (N, D) flat, concept-major
        n_img = ii.max() + 1
        return clip[oi * n_img + ii].astype(np.float32)
    raise SystemExit(f"[FATAL] unexpected CLIP target ndim {clip.ndim}")


def info_nce(pred: torch.Tensor, tgt: torch.Tensor, temp: float) -> torch.Tensor:
    logits = pred @ tgt.t() / temp
    labels = torch.arange(pred.shape[0], device=pred.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


@torch.no_grad()
def diagnose(head: nn.Module | None, x: torch.Tensor, y: torch.Tensor, oi: np.ndarray,
             device: torch.device, batch: int = 4096) -> dict:
    """Paired cosine, plus concept-level Top-1/Top-5 retrieval on this split.

    The retrieval part is the diagnostic that matters: paired cosine rewards being close
    to your own target, while Top-1 rewards being closer to your target than to every
    other concept's, which is what the decoder's neighbour lookup actually uses.

    The `head is None` case is the identity baseline, and it is deliberately given the
    *easier* problem: the encoder output is centred on the split mean before normalising,
    which removes the shared direction that otherwise dominates every cosine. This makes
    the baseline stronger and the comparison against a trained head conservative -- if a
    centred identity already looked good, a marginal head would show no gain, and that is
    exactly what we want the diagnostic to reveal rather than hide.
    """
    center = x.mean(0, keepdim=True)
    preds = []
    for s in range(0, x.shape[0], batch):
        xb = x[s:s + batch].to(device)
        if head is None:
            preds.append(F.normalize(xb.float() - center.to(device), dim=-1).cpu())
        else:
            preds.append(head(xb).float().cpu())
    pred = torch.cat(preds, 0)
    paired = (pred * y).sum(1)

    # Concept-level gallery: average the 10 image predictions of each concept, then ask
    # which concept's *target* is nearest. Chance is 1/len(concepts).
    concepts = np.unique(oi)
    cidx = {int(c): i for i, c in enumerate(concepts)}
    rows = np.array([cidx[int(c)] for c in oi])
    psum = np.zeros((len(concepts), pred.shape[1]), dtype=np.float64)
    np.add.at(psum, rows, pred.numpy().astype(np.float64))
    pmean = psum / np.maximum(np.bincount(rows, minlength=len(concepts))[:, None], 1)
    pmean /= np.maximum(np.linalg.norm(pmean, axis=1, keepdims=True), 1e-8)

    ymean = np.zeros_like(psum)
    np.add.at(ymean, rows, y.numpy().astype(np.float64))
    ymean /= np.maximum(np.bincount(rows, minlength=len(concepts))[:, None], 1)
    ymean /= np.maximum(np.linalg.norm(ymean, axis=1, keepdims=True), 1e-8)

    sim = pmean @ ymean.T
    top5 = np.argsort(-sim, axis=1)[:, :5]
    hit1 = (top5[:, 0] == np.arange(len(concepts))).mean()
    hit5 = (top5 == np.arange(len(concepts))[:, None]).any(1).mean()
    return {"paired_cos": float(paired.mean()),
            "paired_cos_p10": float(paired.quantile(0.10)),
            "concept_top1": float(hit1), "concept_top5": float(hit5),
            "n_concepts": int(len(concepts)), "chance_top1": 1.0 / max(len(concepts), 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-feats", nargs="+", required=True,
                    help="npz files from export_eeg_feats.py (globs allowed)")
    ap.add_argument("--clip-train", type=Path, default=DEFAULT_CLIP_TRAIN)
    ap.add_argument("--out", type=Path, required=True, help="output .pt for the head")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--w-cos", type=float, default=1.0)
    ap.add_argument("--w-mse", type=float, default=0.5)
    ap.add_argument("--w-nce", type=float, default=1.0)
    ap.add_argument("--no-standardize", action="store_true")
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.05,
                    help="fraction of SOURCE concepts held out for checkpoint selection")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-rows", type=int, default=0, help="smoke: cap training rows")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        # Same reasoning as export_conditions.py: an implicit CUDA fallback would fail at
        # the first tensor op with a driver message that does not name the real problem.
        raise SystemExit(
            "[FATAL] no CUDA device visible. Run on a GPU node, or pass --allow-cpu to "
            "train on CPU deliberately.")
    print(f"[INFO] device={device}")

    clip_path = args.clip_train if args.clip_train.is_absolute() else REPO / args.clip_train
    clip = np.load(clip_path)
    print(f"[INFO] CLIP target {clip_path.name} shape={clip.shape}")

    x_np, oi, ii, used = load_sources(args.src_feats)
    y_np = build_pairs(x_np, oi, ii, clip)
    print(f"[INFO] pooled source rows={x_np.shape[0]} x_dim={x_np.shape[1]} "
          f"y_dim={y_np.shape[1]}")

    # Concept-level split. Splitting by row would leak: the 10 images of a concept are
    # highly correlated, so a row split puts near-duplicates on both sides and reports an
    # optimistic validation number that means nothing.
    concepts = np.unique(oi)
    rng = np.random.RandomState(args.seed)
    n_val = max(int(round(len(concepts) * args.val_frac)), 1)
    val_concepts = set(rng.choice(concepts, size=n_val, replace=False).tolist())
    is_val = np.array([int(c) in val_concepts for c in oi])
    tr = np.where(~is_val)[0]
    va = np.where(is_val)[0] if is_val.any() else tr[:1]
    if args.limit_rows:
        tr = tr[:args.limit_rows]
    print(f"[INFO] train rows={len(tr)} val rows={len(va)} "
          f"({len(val_concepts)}/{len(concepts)} val concepts)")

    x_tr = torch.from_numpy(x_np[tr])
    head = GenHead(dim_in=x_np.shape[1], dim_out=y_np.shape[1], hidden=args.hidden,
                   dropout=args.dropout, residual=not args.no_residual,
                   standardize=not args.no_standardize)
    if head.standardize:
        # Statistics from SOURCE TRAINING ROWS ONLY. Using the validation or target rows
        # would leak scale information across the LOSO boundary.
        head.set_norm_stats(x_tr.mean(0), x_tr.std(0))
    head.to(device)

    y_all = torch.from_numpy(y_np)
    base = diagnose(None, x_tr[:min(len(tr), 20000)], y_all[tr[:min(len(tr), 20000)]],
                    oi[tr[:min(len(tr), 20000)]], device)
    print(f"[DIAG] identity (no head): paired_cos={base['paired_cos']:.4f} "
          f"concept_top1={base['concept_top1']:.4f} (chance={base['chance_top1']:.4f})")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(args.epochs * max(len(tr) // args.batch, 1), 1),
        pct_start=0.15)

    best, best_state, best_epoch, history = -1.0, None, -1, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        head.train()
        perm = torch.randperm(len(tr))
        run, nb = 0.0, 0
        for s in range(0, len(tr), args.batch):
            sel = tr[perm[s:s + args.batch].numpy()]
            if len(sel) < 8:
                continue
            xb = torch.from_numpy(x_np[sel]).to(device)
            yb = y_all[sel].to(device)
            pred = head(xb)
            cos = (1.0 - (pred * yb).sum(1)).mean()
            mse = F.mse_loss(pred, yb)
            nce = info_nce(pred, yb, args.temp)
            loss = args.w_cos * cos + args.w_mse * mse + args.w_nce * nce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            run += float(loss.detach()) * len(sel)
            nb += len(sel)

        head.eval()
        d = diagnose(head, torch.from_numpy(x_np[va]), y_all[va], oi[va], device)
        score = d["paired_cos"] + d["concept_top1"]
        history.append({"epoch": epoch, "loss": run / max(nb, 1), **d})
        if score > best:
            best, best_epoch = score, epoch
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        print(f"[{epoch:3d}] loss={run / max(nb, 1):.4f} val paired_cos={d['paired_cos']:.4f} "
              f"top1={d['concept_top1']:.4f} top5={d['concept_top5']:.4f} "
              f"({time.time() - t0:.0f}s){' *' if score >= best else ''}")

    if best_state is not None:
        head.load_state_dict(best_state)

    dest = args.out if args.out.is_absolute() else REPO / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"gen_head": head.state_dict(),
                "config": {"dim_in": x_np.shape[1], "dim_out": y_np.shape[1],
                           "hidden": args.hidden, "dropout": args.dropout,
                           "residual": not args.no_residual,
                           "standardize": not args.no_standardize},
                "train_args": vars(args) | {"clip_train": str(clip_path)},
                "source_feats": used,
                "val_concepts": sorted(int(c) for c in val_concepts),
                "history": history, "best_epoch": best_epoch}, dest)
    final = history[best_epoch - 1] if 0 < best_epoch <= len(history) else history[-1]
    (dest.with_suffix(".json")).write_text(json.dumps(
        {"best_epoch": best_epoch, "identity_baseline": base, "best": final,
         "n_train_rows": int(len(tr)), "n_val_rows": int(len(va)),
         "source_feats": used}, indent=2), encoding="utf-8")
    print(f"[OK] wrote {dest}  best epoch {best_epoch}: paired_cos={final['paired_cos']:.4f} "
          f"top1={final['concept_top1']:.4f} (identity was {base['paired_cos']:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
