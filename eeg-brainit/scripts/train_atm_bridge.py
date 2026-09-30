#!/usr/bin/env python3
"""Train ATM->Brain-IT bridge with frozen ATM semantics.

Stage 1: train bridge + bridge_clip readout (ATM emb frozen for retrieval).
Stage 2: train bridge + BIT + bit_clip readout for generation path.
Early-stop on held-out test Top-1 of the loss head (not train cos_gap alone).
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from eeg_brainit.data.atm_dataset import AtmEmbeddingDataset
from eeg_brainit.models.atm_bridge import AtmBrainITPipeline
from eeg_brainit.training.losses import EEGBrainITLoss
from eeg_brainit.utils.config import ensure_dirs, load_config
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def retrieval_top1(pred: np.ndarray, target: np.ndarray) -> float:
    sim = pred @ target.T
    hit = 0
    for i in range(sim.shape[0]):
        if int(np.argmax(sim[i])) == i:
            hit += 1
    return hit / max(sim.shape[0], 1)


@torch.no_grad()
def eval_test_retrieval(
    model: AtmBrainITPipeline,
    eeg: np.ndarray,
    img: np.ndarray,
    device: torch.device,
    keys: list[str],
    batch_size: int = 256,
) -> dict[str, float]:
    model.eval()
    outs: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    atm_chunks: list[np.ndarray] = []
    x = torch.from_numpy(eeg.astype(np.float32))
    for i in range(0, len(x), batch_size):
        batch = x[i : i + batch_size].to(device)
        out = model(batch)
        atm_chunks.append(F.normalize(out["atm_emb"].float(), dim=-1).cpu().numpy())
        for k in keys:
            if k not in out:
                continue
            outs[k].append(F.normalize(out[k].float(), dim=-1).cpu().numpy())
    atm = np.concatenate(atm_chunks, axis=0).astype(np.float32)
    metrics = {}
    for k, chunks in outs.items():
        if not chunks:
            continue
        pred = np.concatenate(chunks, axis=0).astype(np.float32)
        metrics[f"test_top1_{k}"] = float(retrieval_top1(pred, img))
        metrics[f"test_distill_cos_{k}"] = float((pred * atm).sum(1).mean())
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--override", type=str, nargs="*", default=[])
    parser.add_argument("--init-checkpoint", type=str, default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for ov in args.override:
        cfg = deep_update(cfg, load_config(ov))

    project_root = Path(cfg.get("project_root", ROOT))
    out_dir = project_root / cfg.get("output_dir", "outputs/atm_bridge_train")
    ensure_dirs(out_dir, out_dir / "checkpoints")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    if device.type == "cuda":
        print(f"[INFO] GPU={torch.cuda.get_device_name(0)}")

    atm_cfg = cfg.get("atm", {})
    subject = atm_cfg.get("subject", "sub-08")
    bridge_dir = Path(atm_cfg.get("bridge_dir", "outputs/atm_bridge"))
    if not bridge_dir.is_absolute():
        bridge_dir = project_root / bridge_dir

    train_ds = AtmEmbeddingDataset(bridge_dir, subject, split="train")
    val_ds = AtmEmbeddingDataset(bridge_dir, subject, split="val")
    print(f"[INFO] subject={subject} train={len(train_ds)} val={len(val_ds)}")

    bs = int(cfg.get("train", {}).get("batch_size", 256))
    nw = int(cfg.get("train", {}).get("num_workers", 4))
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True
    )

    model = AtmBrainITPipeline.from_config(cfg, project_root=str(project_root)).to(device)
    init_ck = args.init_checkpoint or cfg.get("train", {}).get("init_checkpoint", "")
    if init_ck:
        p = Path(init_ck)
        if not p.is_absolute():
            p = project_root / p
        ck = torch.load(p, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        print(f"[INFO] loaded init {p} missing={len(missing)} unexpected={len(unexpected)}")

    stage = int(cfg.get("train", {}).get("stage", 1))
    model.apply_stage(stage)
    loss_on = str(cfg.get("train", {}).get("loss_on", "bridge_clip"))
    print(f"[INFO] loss_on={loss_on} freeze_atm_semantics={model.freeze_atm_semantics}")

    loss_cfg = cfg.get("loss", {})
    criterion = EEGBrainITLoss(
        lambda_clip=float(loss_cfg.get("lambda_clip", 0.25)),
        lambda_nce=float(loss_cfg.get("lambda_nce", 1.0)),
        lambda_siglip=float(loss_cfg.get("lambda_siglip", 0.0)),
        lambda_reg=float(loss_cfg.get("lambda_reg", 1e-4)),
        lambda_distill=float(loss_cfg.get("lambda_distill", 0.0)),
        temperature=float(loss_cfg.get("temperature", 0.07)),
        use_proxy_fallback=False,
        learnable_logit_scale=False,
    ).to(device)
    print(
        f"[INFO] loss lambdas clip={loss_cfg.get('lambda_clip')} "
        f"nce={loss_cfg.get('lambda_nce')} distill={loss_cfg.get('lambda_distill', 0)}",
        flush=True,
    )

    params = [p for p in list(model.parameters()) + list(criterion.parameters()) if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters — check stage / freeze settings")
    optim = torch.optim.AdamW(
        params,
        lr=float(cfg.get("train", {}).get("learning_rate", 1e-3)),
        weight_decay=float(cfg.get("train", {}).get("weight_decay", 0.01)),
    )
    scaler = GradScaler(enabled=device.type == "cuda")
    bank_size = int(cfg.get("teacher_bank", {}).get("size", 0))
    bank = None
    if bank_size > 0:
        img = torch.from_numpy(np.load(bridge_dir / "clip_img_train_1024.npy")).float()

        class _Bank:
            def __init__(self, emb):
                self.embeddings = F.normalize(emb, dim=-1)
                self.n = emb.shape[0]

            def sample(self, k, exclude=None):
                idx = torch.randperm(self.n)[:k]
                return self.embeddings[idx].to(device, non_blocking=True)

        bank = _Bank(img)
        print(f"[INFO] teacher bank enabled size={bank_size} from train CLIP imgs")

    # Test retrieval monitors (200-way).
    test_eeg = np.load(bridge_dir / f"{subject}_test_eeg_1024.npy").astype(np.float32)
    test_eeg = test_eeg / np.linalg.norm(test_eeg, axis=1, keepdims=True).clip(min=1e-8)
    img_feat_path = project_root / "outputs/eval/atm_baseline/test_ViT-H-14_laion2b_s32b_b79k_features.npy"
    if not img_feat_path.is_file():
        raise FileNotFoundError(img_feat_path)
    test_img = np.load(img_feat_path).astype(np.float32)
    test_img = test_img / np.linalg.norm(test_img, axis=1, keepdims=True).clip(min=1e-8)
    eval_keys = ["atm_emb", "bridge_clip"]
    if model.use_bit:
        eval_keys.append("bit_clip")
    metric_key = str(cfg.get("train", {}).get("early_stop_metric", f"test_top1_{loss_on}"))
    min_distill = float(cfg.get("train", {}).get("min_distill_cos", 0.0))
    # patience<=0 disables early stopping (run all epochs).
    patience = int(cfg.get("train", {}).get("early_stopping_patience", 6))
    disable_early_stop = patience <= 0

    epochs = int(cfg.get("train", {}).get("epochs", 20))
    max_steps = int(cfg.get("train", {}).get("max_steps", 0))
    grad_clip = float(cfg.get("train", {}).get("grad_clip", 1.0))
    eval_every = int(cfg.get("train", {}).get("eval_every", 1))
    best_score = -1e9
    bad = 0
    global_step = 0
    history = []
    print(
        f"[INFO] early_stop_metric={metric_key} min_distill_cos={min_distill} "
        f"patience={patience} disable_early_stop={disable_early_stop}",
        flush=True,
    )

    def run_epoch(loader, train: bool) -> dict[str, float]:
        nonlocal global_step
        model.train(train)
        meters = {}
        n = 0
        if train:
            optim.zero_grad(set_to_none=True)
        pbar = tqdm(loader, desc=("train" if train else "val"))
        for batch in pbar:
            atm = batch["atm_emb"].to(device, non_blocking=True)
            clip = batch["clip_emb"].to(device, non_blocking=True)
            batch_dev = {"clip_emb": clip, "image": batch["image"].to(device)}
            if bank is not None and train:
                batch_dev["teacher_bank"] = bank.sample(bank_size)
            with autocast(enabled=device.type == "cuda"):
                out = model(atm)
                if loss_on not in out:
                    raise KeyError(f"loss_on={loss_on} missing in model outputs {list(out.keys())}")
                loss_out = dict(out)
                loss_out["clip_emb"] = out[loss_on]
                losses = criterion(loss_out, batch_dev)
            if train:
                scaler.scale(losses["total"]).backward()
                if grad_clip > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                global_step += 1
            n += 1
            for k, v in losses.items():
                meters[k] = meters.get(k, 0.0) + float(v.detach().item())
            pbar.set_postfix(
                loss=float(losses["total"]),
                gap=float(losses.get("cos_gap", 0)),
                dist=float(losses.get("distill_cos", 0)),
            )
            if train and max_steps > 0 and global_step >= max_steps:
                break
        return {k: v / max(n, 1) for k, v in meters.items()}

    # Baseline before train.
    base = eval_test_retrieval(model, test_eeg, test_img, device, eval_keys)
    print(f"[INFO] pretest retrieval={base}", flush=True)

    for epoch in range(1, epochs + 1):
        tr = run_epoch(train_loader, True)
        va = run_epoch(val_loader, False)
        row = {"epoch": epoch, "train": tr, "val": va}
        if epoch % eval_every == 0 or epoch == epochs:
            test_m = eval_test_retrieval(model, test_eeg, test_img, device, eval_keys)
            row["test"] = test_m
            print(
                f"[epoch {epoch}] train_gap={tr.get('cos_gap', 0):.4f} "
                f"val_gap={va.get('cos_gap', 0):.4f} test={test_m}",
                flush=True,
            )
        else:
            print(f"[epoch {epoch}] train={tr} val={va}", flush=True)
        history.append(row)

        ckpt = {
            "epoch": epoch,
            "stage": stage,
            "model": model.state_dict(),
            "cfg": cfg,
            "subject": subject,
            "loss_on": loss_on,
            "test": row.get("test", {}),
        }
        torch.save(ckpt, out_dir / "checkpoints" / f"atm_stage{stage}_last.pt")

        # Score for checkpoint selection (NOT the same as residual distill which starts at 1.0).
        if metric_key == "val_distill_cos":
            # Prefer closeness to ATM only as a soft signal; residual identity already maxes this.
            score = float(va.get("distill_cos", -1e9))
        elif metric_key == "val_total_neg":
            score = -float(va.get("total", 1e9))
        elif metric_key.startswith("test_"):
            score = float(row.get("test", {}).get(metric_key, -1e9))
        else:
            score = float(va.get(metric_key, va.get("cos_gap", -1e9)))

        atm_top1 = float(row.get("test", {}).get("test_top1_atm_emb", 0.345))
        distill_key = f"test_distill_cos_{loss_on}"
        distill_now = float(row.get("test", {}).get(distill_key, va.get("distill_cos", 1.0)))
        min_atm_top1 = float(cfg.get("train", {}).get("min_atm_top1_for_ckpt", 0.28))
        ok_atm = atm_top1 >= min_atm_top1
        ok_distill = distill_now >= min_distill if min_distill > 0 else True

        if not ok_atm:
            print(f"[WARN] skip ckpt: atm_emb top1={atm_top1:.3f}", flush=True)
            bad += 1
        elif not ok_distill:
            print(
                f"[WARN] skip ckpt: {distill_key}={distill_now:.4f} < min={min_distill}",
                flush=True,
            )
            bad += 1
        elif score > best_score:
            best_score = score
            bad = 0
            torch.save(ckpt, out_dir / "checkpoints" / f"atm_stage{stage}_best.pt")
            print(
                f"[INFO] New best {metric_key}={best_score:.4f} "
                f"distill={distill_now:.4f} top1_{loss_on}="
                f"{float(row.get('test', {}).get(f'test_top1_{loss_on}', -1)):.3f} "
                f"(bridge_res={float(model.bridge_res_scale):.4f} "
                f"bit_res={float(model.bit_res_scale):.4f})",
                flush=True,
            )
        else:
            bad += 1

        if (not disable_early_stop) and bad >= patience:
            print(f"[INFO] Early stopping at epoch {epoch} (bad={bad}/{patience})", flush=True)
            break
        if max_steps > 0 and global_step >= max_steps:
            print(f"[INFO] Reached max_steps={max_steps}", flush=True)
            break

    summary = {
        "subject": subject,
        "stage": stage,
        "loss_on": loss_on,
        "best_metric": metric_key,
        "best_score": best_score,
        "pretest": base,
        "history": history,
    }
    with (out_dir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    best_path = out_dir / "checkpoints" / f"atm_stage{stage}_best.pt"
    last_path = out_dir / "checkpoints" / f"atm_stage{stage}_last.pt"
    if not best_path.is_file() and last_path.is_file():
        shutil.copy2(last_path, best_path)
        print(
            f"[WARN] no best ckpt saved during training; copied last -> {best_path.name}",
            flush=True,
        )
        summary["best_score"] = float(
            row.get("test", {}).get(metric_key, summary.get("best_score", -1e9))
        )

    print(f"[INFO] Finished stage={stage} best {metric_key}={best_score:.6f}", flush=True)


if __name__ == "__main__":
    main()
