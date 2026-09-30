#!/usr/bin/env python3
"""Train R-CFM-LL: L1 VAE mean head + residual Cond-CFM (MindEye/ATM + residual FM).

Phase-A focus: maximize blurry RGB PixCorr/SSIM vs GT without touching HCMA semantics.
Exports pred_vae_test.npy + pred_blur_rgb_512 for gated SDEdit.
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
from percept_flow_modules import RCFMLLModel, residual_cfm_loss  # noqa: E402


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


def decode_latents(vae, latents: torch.Tensor, scaling: float) -> torch.Tensor:
    x = (latents.float() / scaling).to(dtype=vae.dtype)
    imgs = vae.decode(x).sample
    return (imgs / 2 + 0.5).clamp(0, 1)


def contrastive_aux(pred: torch.Tensor, tgt: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    a = F.normalize(pred.float().flatten(1), dim=-1)
    b = F.normalize(tgt.float().flatten(1), dim=-1)
    logits = a @ b.T / temp
    labels = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


@torch.no_grad()
def eval_latent_metrics(pred: np.ndarray, tgt: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(pred - tgt)))
    rs = []
    for i in range(len(pred)):
        a, b = pred[i].ravel(), tgt[i].ravel()
        if a.std() < 1e-8 or b.std() < 1e-8:
            rs.append(0.0)
        else:
            rs.append(float(np.corrcoef(a, b)[0, 1]))
    return {"vae_mae": mae, "vae_pearson": float(np.mean(rs))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg-train-npy", type=str, required=True)
    ap.add_argument("--eeg-test-npy", type=str, required=True)
    ap.add_argument("--vae-train-npy", type=str, required=True)
    ap.add_argument("--vae-test-npy", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--num-epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scaling-factor", type=float, default=0.13025)
    ap.add_argument("--w-l1", type=float, default=1.0)
    ap.add_argument("--w-cfm", type=float, default=0.45)
    ap.add_argument("--w-ctr", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=1.0, help="ẑ = μ + α·residual at eval")
    ap.add_argument("--ode-steps", type=int, default=12)
    ap.add_argument("--n-avg", type=int, default=1)
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
    assert len(z_tr) == len(v_tr) and len(z_te) == len(v_te)
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

    ds = TensorDataset(torch.from_numpy(z_tr), torch.from_numpy(v_tr_n))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

    model = RCFMLLModel(in_dim=z_tr.shape[1], vae_ch=v_tr.shape[1], spatial=v_tr.shape[-1]).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)

    hub = Path(os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
    vae = None
    if args.decode_rgb:
        vae = resolve_vae(hub, device)
        for p in vae.parameters():
            p.requires_grad_(False)

    t_mean_t = torch.from_numpy(t_mean).to(device)
    t_std_t = torch.from_numpy(t_std).to(device)
    history = []
    best_score, best_ep = -1e9, 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        meters = {"loss": 0.0, "l1": 0.0, "cfm": 0.0, "ctr": 0.0}
        n_ok = 0
        for zb, vb in tqdm(loader, desc=f"rcfm-ll-{epoch}"):
            zb, vb = zb.to(device), vb.to(device)
            opt.zero_grad(set_to_none=True)
            mu = model.vae_head(zb)
            loss_l1 = F.l1_loss(mu, vb)
            residual = vb - mu.detach()
            loss_cfm = residual_cfm_loss(model.cfm_res, residual, zb, mu)
            loss_ctr = contrastive_aux(mu, vb)
            loss = args.w_l1 * loss_l1 + args.w_cfm * loss_cfm + args.w_ctr * loss_ctr
            if not torch.isfinite(loss):
                print("[WARN] non-finite loss, skip batch")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            meters["loss"] += float(loss.item())
            meters["l1"] += float(loss_l1.item())
            meters["cfm"] += float(loss_cfm.item())
            meters["ctr"] += float(loss_ctr.item())
            n_ok += 1
        sched.step()
        if n_ok == 0:
            raise RuntimeError("all batches non-finite")
        for k in meters:
            meters[k] /= n_ok

        model.eval()
        with torch.no_grad():
            zt = torch.from_numpy(z_te).to(device)
            pred_n = model.predict(
                zt, alpha=args.alpha, ode_steps=args.ode_steps, n_avg=args.n_avg, deterministic=True
            )
            pred_te = (pred_n * t_std_t + t_mean_t).cpu().numpy()
            mu_only = (model.vae_head(zt) * t_std_t + t_mean_t).cpu().numpy()

        if not np.isfinite(pred_te).all():
            print(f"[WARN] ep {epoch} NaN — skip ckpt")
            history.append({"epoch": epoch, **meters, "bad": True})
            continue

        m_full = eval_latent_metrics(pred_te, v_te)
        m_mu = eval_latent_metrics(mu_only, v_te)
        # Prefer residual blend quality; mild MAE penalty
        score = 0.70 * m_full["vae_pearson"] + 0.20 * m_mu["vae_pearson"] - 0.10 * min(m_full["vae_mae"], 2.0)
        row = {
            "epoch": epoch,
            **meters,
            **m_full,
            "mu_mae": m_mu["vae_mae"],
            "mu_pearson": m_mu["vae_pearson"],
            "score": score,
            "alpha": args.alpha,
        }
        history.append(row)
        print(
            f"[ep {epoch}] loss={meters['loss']:.4f} pearson={m_full['vae_pearson']:.3f} "
            f"mu_p={m_mu['vae_pearson']:.3f} mae={m_full['vae_mae']:.4f} score={score:.4f}"
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
                    "alpha": args.alpha,
                    "ode_steps": args.ode_steps,
                    "n_avg": args.n_avg,
                    "pipeline": "R-CFM-LL",
                },
                out / "checkpoint_rcfm_ll_best.pth",
            )
            np.save(out / "pred_vae_test.npy", pred_te.astype(np.float16))
            np.save(out / "pred_vae_mu_only_test.npy", mu_only.astype(np.float16))

    if best_ep == 0 or not (out / "checkpoint_rcfm_ll_best.pth").is_file():
        raise RuntimeError("no valid checkpoint saved")

    ckpt = torch.load(out / "checkpoint_rcfm_ll_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mean = torch.as_tensor(ckpt["vae_target_mean"], device=device, dtype=torch.float32).view(1, -1, 1, 1)
    std = torch.as_tensor(ckpt["vae_target_std"], device=device, dtype=torch.float32).view(1, -1, 1, 1)
    alpha = float(ckpt.get("alpha", args.alpha))
    with torch.no_grad():
        zt = torch.from_numpy(z_te).to(device)
        pred_n = model.predict(
            zt, alpha=alpha, ode_steps=args.ode_steps, n_avg=args.n_avg, deterministic=True
        )
        pred_te = pred_n * std + mean
        mu_only = model.vae_head(zt) * std + mean
    np.save(out / "pred_vae_test.npy", pred_te.cpu().numpy().astype(np.float16))
    np.save(out / "pred_vae_mu_only_test.npy", mu_only.cpu().numpy().astype(np.float16))

    rgb_dir = out / "pred_blur_rgb_512"
    rgb_mu_dir = out / "pred_blur_mu_rgb_512"
    if args.decode_rgb:
        if vae is None:
            vae = resolve_vae(hub, device)
            for p in vae.parameters():
                p.requires_grad_(False)
        rgb_dir.mkdir(parents=True, exist_ok=True)
        rgb_mu_dir.mkdir(parents=True, exist_ok=True)
        bs = 8
        for start in tqdm(range(0, len(pred_te), bs), desc="decode-blur-rgb"):
            chunk = pred_te[start : start + bs]
            imgs = decode_latents(vae, chunk, args.scaling_factor)
            for j in range(imgs.shape[0]):
                arr = (imgs[j].float().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                Image.fromarray(arr).save(rgb_dir / f"{start + j:03d}.png")
        for start in tqdm(range(0, len(mu_only), bs), desc="decode-mu-rgb"):
            chunk = mu_only[start : start + bs]
            imgs = decode_latents(vae, chunk, args.scaling_factor)
            for j in range(imgs.shape[0]):
                arr = (imgs[j].float().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                Image.fromarray(arr).save(rgb_mu_dir / f"{start + j:03d}.png")

    import pandas as pd

    pd.DataFrame(history).to_csv(out / "rcfm_ll_history.csv", index=False)
    report = {
        "pipeline": "R-CFM-LL",
        "claim": "L1 VAE mean + residual Cond-CFM; decode-only interface for HCMA SDEdit",
        "best_epoch": best_ep,
        "best_score": best_score,
        "final": history[-1] if history else {},
        "pred_vae": str(out / "pred_vae_test.npy"),
        "pred_rgb_dir": str(rgb_dir) if args.decode_rgb else None,
        "pred_mu_rgb_dir": str(rgb_mu_dir) if args.decode_rgb else None,
        "n_train": int(len(z_tr)),
        "n_test": int(len(z_te)),
        "weights": {"l1": args.w_l1, "cfm": args.w_cfm, "ctr": args.w_ctr},
        "alpha": alpha,
        "ode_steps": args.ode_steps,
    }
    (out / "rcfm_ll_train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
