#!/usr/bin/env python3
"""Train PerceptFlow: EEG → VAE latent + depth with Cond-CFM + perc losses.

Fixes weak perception encoding (L1+LPIPS+SSIM+CFM) and exports blurry RGB / depth
maps for low-strength img2img injection into the frozen semantic a40 path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from percept_flow_modules import PerceptFlowModel, spatial_cfm_loss  # noqa: E402


def resolve_vae(hub: Path, device: torch.device):
    from diffusers import AutoencoderKL

    sdxl_root = hub / "models--stabilityai--stable-diffusion-xl-base-1.0" / "snapshots"
    if sdxl_root.is_dir():
        for snap in sorted(sdxl_root.iterdir(), reverse=True):
            vae_dir = snap / "vae"
            if (vae_dir / "config.json").is_file():
                vae = AutoencoderKL.from_pretrained(str(vae_dir), torch_dtype=torch.float32)
                return vae.to(device).eval()
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae", torch_dtype=torch.float32
    )
    return vae.to(device).eval()


def ssim_loss_map(x: torch.Tensor, y: torch.Tensor, win: int = 7) -> torch.Tensor:
    """Differentiable SSIM loss on (B,C,H,W) in [0,1]. Returns 1 - mean SSIM."""
    c1, c2 = 0.01**2, 0.03**2
    pad = win // 2
    mu_x = F.avg_pool2d(x, win, 1, pad)
    mu_y = F.avg_pool2d(y, win, 1, pad)
    sigma_x = F.avg_pool2d(x * x, win, 1, pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, win, 1, pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, win, 1, pad) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )
    return 1.0 - ssim.mean()


def contrastive_aux(pred: torch.Tensor, tgt: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    """Batch InfoNCE on global-pooled spatial features."""
    a = F.normalize(pred.float().flatten(1), dim=-1)
    b = F.normalize(tgt.float().flatten(1), dim=-1)
    logits = a @ b.T / temp
    labels = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def decode_latents(vae, latents: torch.Tensor, scaling: float) -> torch.Tensor:
    """Return (B,3,H,W) in [0,1]. Gradients flow to latents (VAE frozen)."""
    x = (latents.float() / scaling).to(dtype=vae.dtype)
    imgs = vae.decode(x).sample
    return (imgs / 2 + 0.5).clamp(0, 1)


def depth_to_rgb_u8(depth: np.ndarray) -> np.ndarray:
    d = depth.astype(np.float32)
    d = d - d.min()
    mx = float(d.max()) if float(d.max()) > 1e-8 else 1.0
    d = (d / mx * 255.0).clip(0, 255).astype(np.uint8)
    return np.stack([d, d, d], axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg-train-npy", type=str, required=True)
    ap.add_argument("--eeg-test-npy", type=str, required=True)
    ap.add_argument("--vae-train-npy", type=str, required=True)
    ap.add_argument("--vae-test-npy", type=str, required=True)
    ap.add_argument("--depth-train-npy", type=str, required=True)
    ap.add_argument("--depth-test-npy", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--num-epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scaling-factor", type=float, default=0.13025)
    ap.add_argument("--w-l1", type=float, default=1.0)
    ap.add_argument("--w-depth", type=float, default=0.5)
    ap.add_argument("--w-cfm-vae", type=float, default=0.35)
    ap.add_argument("--w-cfm-depth", type=float, default=0.25)
    ap.add_argument("--w-lpips", type=float, default=0.15)
    ap.add_argument("--w-ssim", type=float, default=0.10)
    ap.add_argument("--w-ctr", type=float, default=0.05)
    ap.add_argument("--perc-every", type=int, default=2, help="compute LPIPS/SSIM every N batches")
    ap.add_argument("--perc-n", type=int, default=8, help="sub-batch size for perc losses")
    ap.add_argument("--ode-steps", type=int, default=12)
    ap.add_argument("--decode-rgb", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    z_tr = np.load(args.eeg_train_npy).astype(np.float32)
    z_te = np.load(args.eeg_test_npy).astype(np.float32)
    v_tr = np.load(args.vae_train_npy).astype(np.float32)
    v_te = np.load(args.vae_test_npy).astype(np.float32)
    d_tr = np.load(args.depth_train_npy).astype(np.float32)
    d_te = np.load(args.depth_test_npy).astype(np.float32)
    if d_tr.ndim == 3:
        d_tr = d_tr[:, None]
        d_te = d_te[:, None]
    assert len(z_tr) == len(v_tr) == len(d_tr)
    assert len(z_te) == len(v_te) == len(d_te)
    if not np.isfinite(v_tr).all() or not np.isfinite(v_te).all():
        raise RuntimeError("VAE latents contain NaN/Inf — rebuild cache")

    z_tr = z_tr / np.linalg.norm(z_tr, axis=1, keepdims=True).clip(1e-8)
    z_te = z_te / np.linalg.norm(z_te, axis=1, keepdims=True).clip(1e-8)

    t_mean = v_tr.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    t_std = v_tr.std(axis=(0, 2, 3), keepdims=True).astype(np.float32).clip(1e-3)
    np.save(out / "vae_target_mean.npy", t_mean.squeeze())
    np.save(out / "vae_target_std.npy", t_std.squeeze())
    v_tr_n = (v_tr - t_mean) / t_std
    v_te_n = (v_te - t_mean) / t_std

    d_mean = float(d_tr.mean())
    d_std = float(max(d_tr.std(), 1e-3))
    np.save(out / "depth_stats.npy", np.array([d_mean, d_std], dtype=np.float32))
    d_tr_n = (d_tr - d_mean) / d_std
    d_te_n = (d_te - d_mean) / d_std

    ds = TensorDataset(
        torch.from_numpy(z_tr),
        torch.from_numpy(v_tr_n),
        torch.from_numpy(d_tr_n),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

    model = PerceptFlowModel(in_dim=z_tr.shape[1], vae_ch=v_tr.shape[1], spatial=v_tr.shape[-1]).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)

    hub = Path(os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
    vae = resolve_vae(hub, device)
    for p in vae.parameters():
        p.requires_grad_(False)
    import lpips

    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    t_mean_t = torch.from_numpy(t_mean).to(device)
    t_std_t = torch.from_numpy(t_std).to(device)
    history = []
    best_score, best_ep = -1e9, 0
    global_step = 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        meters = {"loss": 0.0, "l1": 0.0, "depth": 0.0, "cfm_v": 0.0, "cfm_d": 0.0, "lpips": 0.0, "ssim": 0.0}
        n_ok = 0
        for zb, vb, db in tqdm(loader, desc=f"percept-flow-{epoch}"):
            zb, vb, db = zb.to(device), vb.to(device), db.to(device)
            opt.zero_grad(set_to_none=True)
            pred_v = model.vae_head(zb)
            pred_d = model.depth_head(zb)
            loss_l1 = F.l1_loss(pred_v, vb)
            loss_d = F.l1_loss(pred_d, db)
            loss_cfm_v = spatial_cfm_loss(model.cfm_vae, vb, zb)
            loss_cfm_d = spatial_cfm_loss(model.cfm_depth, db, zb)
            loss_ctr = contrastive_aux(pred_v, vb)
            loss = (
                args.w_l1 * loss_l1
                + args.w_depth * loss_d
                + args.w_cfm_vae * loss_cfm_v
                + args.w_cfm_depth * loss_cfm_d
                + args.w_ctr * loss_ctr
            )
            loss_lp = torch.zeros((), device=device)
            loss_ss = torch.zeros((), device=device)
            if global_step % max(args.perc_every, 1) == 0:
                n = min(args.perc_n, zb.shape[0])
                # denorm → decode → LPIPS/SSIM (blur GT slightly for structure focus)
                pred_raw = pred_v[:n] * t_std_t + t_mean_t
                tgt_raw = vb[:n] * t_std_t + t_mean_t
                with torch.cuda.amp.autocast(enabled=False):
                    pred_rgb = decode_latents(vae, pred_raw, args.scaling_factor)
                    tgt_rgb = decode_latents(vae, tgt_raw, args.scaling_factor)
                # downsample for speed/stability
                pred_s = F.interpolate(pred_rgb, size=128, mode="bilinear", align_corners=False)
                tgt_s = F.interpolate(tgt_rgb, size=128, mode="bilinear", align_corners=False)
                # soft blur on GT to emphasize structure
                tgt_blur = F.avg_pool2d(tgt_s, 5, 1, 2)
                pred_b = pred_s * 2 - 1
                tgt_b = tgt_blur * 2 - 1
                loss_lp = lpips_fn(pred_b, tgt_b).mean()
                loss_ss = ssim_loss_map(pred_s, tgt_blur)
                loss = loss + args.w_lpips * loss_lp + args.w_ssim * loss_ss

            if not torch.isfinite(loss):
                print("[WARN] non-finite loss, skip batch")
                global_step += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            meters["loss"] += float(loss.item())
            meters["l1"] += float(loss_l1.item())
            meters["depth"] += float(loss_d.item())
            meters["cfm_v"] += float(loss_cfm_v.item())
            meters["cfm_d"] += float(loss_cfm_d.item())
            meters["lpips"] += float(loss_lp.item())
            meters["ssim"] += float(loss_ss.item())
            n_ok += 1
            global_step += 1

        sched.step()
        if n_ok == 0:
            raise RuntimeError("all batches non-finite")
        for k in meters:
            meters[k] /= n_ok

        # eval
        model.eval()
        with torch.no_grad():
            zt = torch.from_numpy(z_te).to(device)
            pred_head = model.vae_head(zt)
            pred_cfm = model.cfm_vae.decode(zt, steps=args.ode_steps)
            # blend head + CFM (learned-ish fixed α)
            alpha = 0.55
            pred_n = alpha * pred_cfm + (1.0 - alpha) * pred_head
            pred_te = (pred_n * t_std_t + t_mean_t).cpu().numpy()
            depth_head = model.depth_head(zt)
            depth_cfm = model.cfm_depth.decode(zt, steps=args.ode_steps)
            depth_n = 0.5 * depth_head + 0.5 * depth_cfm
            depth_te = (depth_n * d_std + d_mean).cpu().numpy()

        if not np.isfinite(pred_te).all():
            print(f"[WARN] ep {epoch} VAE pred NaN — skip ckpt")
            history.append({"epoch": epoch, **meters, "bad": True})
            continue

        mae = float(np.mean(np.abs(pred_te - v_te)))
        rs = []
        for i in range(len(pred_te)):
            a, b = pred_te[i].ravel(), v_te[i].ravel()
            rs.append(0.0 if a.std() < 1e-8 or b.std() < 1e-8 else float(np.corrcoef(a, b)[0, 1]))
        pearson = float(np.mean(rs))
        d_mae = float(np.mean(np.abs(depth_te - d_te)))
        dr = []
        for i in range(len(depth_te)):
            a, b = depth_te[i].ravel(), d_te[i].ravel()
            dr.append(0.0 if a.std() < 1e-8 or b.std() < 1e-8 else float(np.corrcoef(a, b)[0, 1]))
        d_pearson = float(np.mean(dr))
        # higher better
        score = 0.55 * pearson + 0.35 * d_pearson - 0.10 * min(mae, 2.0)
        row = {
            "epoch": epoch,
            **meters,
            "vae_mae": mae,
            "vae_pearson": pearson,
            "depth_mae": d_mae,
            "depth_pearson": d_pearson,
            "score": score,
        }
        history.append(row)
        print(
            f"[ep {epoch}] loss={meters['loss']:.4f} vae_p={pearson:.3f} "
            f"depth_p={d_pearson:.3f} mae={mae:.4f} score={score:.4f}"
        )
        if score > best_score:
            best_score, best_ep = score, epoch
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "in_dim": z_tr.shape[1],
                    "metrics": row,
                    "scaling_factor": args.scaling_factor,
                    "vae_target_mean": t_mean.squeeze(),
                    "vae_target_std": t_std.squeeze(),
                    "depth_mean": d_mean,
                    "depth_std": d_std,
                    "blend_alpha_vae": alpha,
                    "ode_steps": args.ode_steps,
                },
                out / "checkpoint_percept_flow_best.pth",
            )
            np.save(out / "pred_vae_test.npy", pred_te.astype(np.float16))
            np.save(out / "pred_depth_test.npy", depth_te.astype(np.float32))

    if best_ep == 0 or not (out / "checkpoint_percept_flow_best.pth").is_file():
        raise RuntimeError("no valid checkpoint saved")

    ckpt = torch.load(out / "checkpoint_percept_flow_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mean = torch.as_tensor(ckpt["vae_target_mean"], device=device, dtype=torch.float32).view(1, -1, 1, 1)
    std = torch.as_tensor(ckpt["vae_target_std"], device=device, dtype=torch.float32).view(1, -1, 1, 1)
    d_mean = float(ckpt["depth_mean"])
    d_std = float(ckpt["depth_std"])
    alpha = float(ckpt.get("blend_alpha_vae", 0.55))
    with torch.no_grad():
        zt = torch.from_numpy(z_te).to(device)
        pred_n = alpha * model.cfm_vae.decode(zt, steps=args.ode_steps) + (1.0 - alpha) * model.vae_head(zt)
        pred_te = pred_n * std + mean
        depth_n = 0.5 * model.depth_head(zt) + 0.5 * model.cfm_depth.decode(zt, steps=args.ode_steps)
        depth_te = (depth_n * d_std + d_mean).cpu().numpy()
    np.save(out / "pred_vae_test.npy", pred_te.cpu().numpy().astype(np.float16))
    np.save(out / "pred_depth_test.npy", depth_te.astype(np.float32))

    rgb_dir = out / "pred_blur_rgb_512"
    depth_rgb_dir = out / "pred_depth_rgb_512"
    if args.decode_rgb:
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_rgb_dir.mkdir(parents=True, exist_ok=True)
        bs = 8
        for start in tqdm(range(0, len(pred_te), bs), desc="decode-blur-rgb"):
            chunk = pred_te[start : start + bs]
            imgs = decode_latents(vae, chunk, args.scaling_factor)
            for j in range(imgs.shape[0]):
                arr = (imgs[j].float().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                Image.fromarray(arr).save(rgb_dir / f"{start + j:03d}.png")
        for i in range(len(depth_te)):
            dmap = depth_te[i, 0] if depth_te.ndim == 4 else depth_te[i]
            rgb = depth_to_rgb_u8(dmap)
            Image.fromarray(rgb).resize((512, 512), Image.Resampling.BICUBIC).save(
                depth_rgb_dir / f"{i:03d}.png"
            )

    import pandas as pd

    pd.DataFrame(history).to_csv(out / "percept_flow_history.csv", index=False)
    report = {
        "pipeline": "PerceptFlow",
        "best_epoch": best_ep,
        "best_score": best_score,
        "final": history[-1] if history else {},
        "pred_vae": str(out / "pred_vae_test.npy"),
        "pred_depth": str(out / "pred_depth_test.npy"),
        "pred_rgb_dir": str(rgb_dir) if args.decode_rgb else None,
        "pred_depth_rgb_dir": str(depth_rgb_dir) if args.decode_rgb else None,
        "n_train": int(len(z_tr)),
        "n_test": int(len(z_te)),
        "weights": {
            "l1": args.w_l1,
            "depth": args.w_depth,
            "cfm_vae": args.w_cfm_vae,
            "cfm_depth": args.w_cfm_depth,
            "lpips": args.w_lpips,
            "ssim": args.w_ssim,
            "ctr": args.w_ctr,
        },
    }
    (out / "percept_flow_train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
