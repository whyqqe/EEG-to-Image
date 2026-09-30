#!/usr/bin/env python3
"""Train NeuroBOLT fMRI ROI tokens -> Brain-IT with residual ATM distill.

Stage 0: direct MLP ablation (no BIT)
Stage 1: RoiToBitBridge + bridge_clip
Stage 2/3: + BIT + bit_clip
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from eeg_brainit.data.neurobolt_dataset import NeuroBoltBridgeDataset
from eeg_brainit.models.neurobolt_bridge import NeuroBoltBrainITPipeline
from eeg_brainit.training.losses import EEGBrainITLoss
from eeg_brainit.utils.config import ensure_dirs, load_config


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
    hit = sum(1 for i in range(sim.shape[0]) if int(np.argmax(sim[i])) == i)
    return hit / max(sim.shape[0], 1)


@torch.no_grad()
def eval_test_retrieval(
    model: NeuroBoltBrainITPipeline,
    tokens: np.ndarray,
    atm: np.ndarray,
    img: np.ndarray,
    device: torch.device,
    keys: list[str],
    batch_size: int = 128,
) -> dict[str, float]:
    model.eval()
    outs: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    atm_chunks: list[np.ndarray] = []
    tok_t = torch.from_numpy(tokens.astype(np.float32))
    atm_t = torch.from_numpy(atm.astype(np.float32))
    for i in range(0, len(tok_t), batch_size):
        out = model(tok_t[i : i + batch_size].to(device), atm_t[i : i + batch_size].to(device))
        atm_chunks.append(F.normalize(out["atm_emb"].float(), dim=-1).cpu().numpy())
        for k in keys:
            if k not in out:
                continue
            outs[k].append(F.normalize(out[k].float(), dim=-1).cpu().numpy())
    atm_np = np.concatenate(atm_chunks, axis=0).astype(np.float32)
    metrics = {}
    for k, chunks in outs.items():
        if not chunks:
            continue
        pred = np.concatenate(chunks, axis=0).astype(np.float32)
        metrics[f"test_top1_{k}"] = float(retrieval_top1(pred, img))
        metrics[f"test_distill_cos_{k}"] = float((pred * atm_np).sum(1).mean())
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
    out_dir = project_root / cfg.get("output_dir", "outputs/nb_bit_train")
    ensure_dirs(out_dir, out_dir / "checkpoints")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    if device.type == "cuda":
        print(f"[INFO] GPU={torch.cuda.get_device_name(0)}")

    nb_cfg = cfg.get("neurobolt", {})
    subject = nb_cfg.get("subject", "sub-08")
    bridge_dir = Path(nb_cfg.get("bridge_dir", "outputs/atm_bridge"))
    if not bridge_dir.is_absolute():
        bridge_dir = project_root / bridge_dir
    neurobolt_dir = Path(
        nb_cfg.get("neurobolt_dir", "/project/peilab/why/cache/things_eeg2_b2/neurobolt")
    )

    train_ds = NeuroBoltBridgeDataset(neurobolt_dir, bridge_dir, subject, split="train")
    val_ds = NeuroBoltBridgeDataset(neurobolt_dir, bridge_dir, subject, split="val")
    print(
        f"[INFO] subject={subject} train={len(train_ds)} val={len(val_ds)} "
        f"tok_shape={tuple(train_ds[0]['fmri_tokens'].shape)}"
    )

    bs = int(cfg.get("train", {}).get("batch_size", 256))
    nw = int(cfg.get("train", {}).get("num_workers", 4))
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True
    )

    model = NeuroBoltBrainITPipeline.from_config(cfg, project_root=str(project_root)).to(device)
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
    print(f"[INFO] loss_on={loss_on}")

    loss_cfg = cfg.get("loss", {})
    criterion = EEGBrainITLoss(
        lambda_clip=float(loss_cfg.get("lambda_clip", 0.05)),
        lambda_nce=float(loss_cfg.get("lambda_nce", 0.1)),
        lambda_siglip=float(loss_cfg.get("lambda_siglip", 0.0)),
        lambda_reg=float(loss_cfg.get("lambda_reg", 1e-4)),
        lambda_distill=float(loss_cfg.get("lambda_distill", 3.0)),
        temperature=float(loss_cfg.get("temperature", 0.07)),
        use_proxy_fallback=False,
        learnable_logit_scale=False,
    ).to(device)

    params = [p for p in list(model.parameters()) + list(criterion.parameters()) if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters")
    optim = torch.optim.AdamW(
        params,
        lr=float(cfg.get("train", {}).get("learning_rate", 3e-4)),
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
        print(f"[INFO] teacher bank size={bank_size}")

    # Test monitors
    test_ds = NeuroBoltBridgeDataset(
        neurobolt_dir,
        bridge_dir,
        subject,
        split="test",
        teacher_img=project_root
        / "outputs/eval/atm_baseline/test_ViT-H-14_laion2b_s32b_b79k_features.npy",
    )
    test_tok = np.stack([test_ds[i]["fmri_tokens"].numpy() for i in range(len(test_ds))])
    test_atm = np.stack([test_ds[i]["atm_emb"].numpy() for i in range(len(test_ds))])
    test_img = np.stack([test_ds[i]["clip_emb"].numpy() for i in range(len(test_ds))])
    eval_keys = ["atm_emb", "bridge_clip"]
    if model.use_bit:
        eval_keys.append("bit_clip")
    if model.use_direct and model.direct_head is not None:
        eval_keys.append("direct_clip")
    metric_key = str(cfg.get("train", {}).get("early_stop_metric", f"test_top1_{loss_on}"))
    min_distill = float(cfg.get("train", {}).get("min_distill_cos", 0.98))
    patience = int(cfg.get("train", {}).get("early_stopping_patience", 0))
    disable_early_stop = patience <= 0

    epochs = int(cfg.get("train", {}).get("epochs", 30))
    grad_clip = float(cfg.get("train", {}).get("grad_clip", 1.0))
    eval_every = int(cfg.get("train", {}).get("eval_every", 1))
    best_score = -1e9
    bad = 0
    history = []
    print(
        f"[INFO] early_stop_metric={metric_key} min_distill={min_distill} "
        f"patience={patience} disable_early_stop={disable_early_stop}",
        flush=True,
    )

    def run_epoch(loader, train: bool) -> dict[str, float]:
        model.train(train)
        meters: dict[str, float] = {}
        n = 0
        if train:
            optim.zero_grad(set_to_none=True)
        pbar = tqdm(loader, desc=("train" if train else "val"))
        for batch in pbar:
            tok = batch["fmri_tokens"].to(device, non_blocking=True)
            atm = batch["atm_emb"].to(device, non_blocking=True)
            clip = batch["clip_emb"].to(device, non_blocking=True)
            batch_dev = {"clip_emb": clip, "image": batch["image"].to(device)}
            if bank is not None and train:
                batch_dev["teacher_bank"] = bank.sample(bank_size)
            with autocast(enabled=device.type == "cuda"):
                out = model(tok, atm)
                if loss_on not in out:
                    raise KeyError(f"loss_on={loss_on} missing; have {list(out.keys())}")
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
            n += 1
            for k, v in losses.items():
                meters[k] = meters.get(k, 0.0) + float(v.detach().item())
            pbar.set_postfix(
                loss=float(losses["total"]),
                gap=float(losses.get("cos_gap", 0)),
                dist=float(losses.get("distill_cos", 0)),
            )
        return {k: v / max(n, 1) for k, v in meters.items()}

    base = eval_test_retrieval(model, test_tok, test_atm, test_img, device, eval_keys)
    print(f"[INFO] pretest retrieval={base}", flush=True)

    for epoch in range(1, epochs + 1):
        tr = run_epoch(train_loader, True)
        va = run_epoch(val_loader, False)
        row: dict = {"epoch": epoch, "train": tr, "val": va}
        if epoch % eval_every == 0 or epoch == epochs:
            test_m = eval_test_retrieval(model, test_tok, test_atm, test_img, device, eval_keys)
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
        torch.save(ckpt, out_dir / "checkpoints" / f"nb_stage{stage}_last.pt")

        score = float(row.get("test", {}).get(metric_key, -1e9))
        atm_top1 = float(row.get("test", {}).get("test_top1_atm_emb", 0.345))
        distill_key = f"test_distill_cos_{loss_on}"
        distill_now = float(row.get("test", {}).get(distill_key, va.get("distill_cos", 1.0)))
        # atm_skip runs keep ATM floor; standalone (residual none) only gates on distill if set.
        require_atm_floor = str(cfg.get("neurobolt", {}).get("residual_mode", "atm_skip")) == "atm_skip"
        ok_atm = (atm_top1 >= 0.30) if require_atm_floor else True
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
            torch.save(ckpt, out_dir / "checkpoints" / f"nb_stage{stage}_best.pt")
            print(
                f"[INFO] New best {metric_key}={best_score:.4f} distill={distill_now:.4f} "
                f"top1_{loss_on}={float(row.get('test', {}).get(f'test_top1_{loss_on}', -1)):.3f} "
                f"(bridge_res={float(model.bridge_res_scale):.4f} "
                f"bit_res={float(model.bit_res_scale):.4f})",
                flush=True,
            )
        else:
            bad += 1

        if (not disable_early_stop) and bad >= patience:
            print(f"[INFO] Early stopping at epoch {epoch}", flush=True)
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
    print(f"[INFO] Finished stage={stage} best {metric_key}={best_score:.6f}", flush=True)


if __name__ == "__main__":
    main()
