#!/usr/bin/env python3
"""Train Fusion-1024 -> ViT-H-1024 bridge for IP-Adapter decode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
NB_ADAPTER = SCRIPT_DIR.parent / "nb_adapter"
sys.path.insert(0, str(NB_ADAPTER))

from train_nb_adapter import (  # noqa: E402
    LinearAdapter,
    MLPAdapter,
    concept_split,
    l2_np,
    predict,
    retrieval_metrics,
    train_one,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion-train", type=str, required=True)
    ap.add_argument("--clip-train", type=str, required=True)
    ap.add_argument("--clip-test", type=str, required=True)
    ap.add_argument("--gallery", type=str, required=True, help="train ViT-H gallery for retrieval eval")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--fusion-test", type=str, nargs="*", default=[], help="tag:path pairs via --fusion-test-src")
    ap.add_argument(
        "--fusion-test-src",
        type=str,
        nargs="*",
        default=[],
        help="entries like cft:/path/to.npy",
    )
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr-linear", type=float, default=1e-3)
    ap.add_argument("--lr-mlp", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--skip-mlp", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    x_train = np.load(args.fusion_train).astype(np.float32)
    y_train = np.load(args.clip_train).astype(np.float32)
    y_test = np.load(args.clip_test).astype(np.float32)
    gallery = np.load(args.gallery).astype(np.float32)

    if x_train.shape[0] != y_train.shape[0]:
        raise RuntimeError(f"train n mismatch fusion={x_train.shape[0]} clip={y_train.shape[0]}")

    tr_idx, va_idx, val_concepts = concept_split(1654, 10, args.val_frac, args.seed)
    x_tr = torch.from_numpy(x_train[tr_idx])
    y_tr = torch.from_numpy(y_train[tr_idx])
    x_va = torch.from_numpy(x_train[va_idx])
    y_va = torch.from_numpy(y_train[va_idx])

    test_sources: dict[str, str] = {}
    for entry in args.fusion_test_src:
        if ":" not in entry:
            raise ValueError(f"bad fusion-test-src entry: {entry}")
        tag, path = entry.split(":", 1)
        test_sources[tag.strip()] = path.strip()

    report: dict = {"val_concepts": val_concepts, "models": {}}

    specs: list[tuple[str, torch.nn.Module, float]] = [
        ("linear", LinearAdapter(1024, 1024), args.lr_linear),
    ]
    if not args.skip_mlp:
        specs.append(("mlp", MLPAdapter(1024, 1024), args.lr_mlp))

    for name, model, lr in specs:
        pack = train_one(
            name,
            model,
            x_tr,
            y_tr,
            x_va,
            y_va,
            device,
            args.epochs,
            lr,
            args.weight_decay,
            args.batch_size,
            args.patience,
        )
        model = pack.pop("model")
        torch.save({"state_dict": model.state_dict(), "name": name}, out_dir / f"{name}_adapter.pt")

        model_report = {k: v for k, v in pack.items() if k != "history"}
        model_report["history_tail"] = pack.get("history", [])[-3:]

        for tag, path in test_sources.items():
            x_te = np.load(path).astype(np.float32)
            pred = predict(model, x_te, device)
            out_npy = out_dir / f"{name}_{tag}_test_clip_1024.npy"
            np.save(out_npy, pred)
            gt_cos = float(np.mean(np.sum(l2_np(pred) * l2_np(y_test), axis=1)))
            retr = retrieval_metrics(pred, gallery)
            model_report[f"test_{tag}"] = {
                "path": str(out_npy),
                "gt_cos": gt_cos,
                "retrieval": retr,
            }
            print(f"[{name}/{tag}] gt_cos={gt_cos:.4f} top1={retr['top1']:.3f}")

        report["models"][name] = model_report

    (out_dir / "bridge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] {out_dir / 'bridge_report.json'}")


if __name__ == "__main__":
    main()
