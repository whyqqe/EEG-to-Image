#!/usr/bin/env python3
"""Train TCDA perception (Pc/Pf multi-granularity) + relational saliency R.

Semantic tower S is frozen externally (MG-Flow a40). This script only trains P/R.
Targets are built on-the-fly (blur / saliency) to avoid large VAE caches.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcda_modules import TCDAModel, spatial_cfm_loss  # noqa: E402


def list_split_images(images_root: Path, split: str) -> list[Path]:
    root = images_root / ("training_images" if split == "train" else "test_images")
    paths: list[Path] = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.JPEG")) + list(d.glob("*.png")))
        paths.extend(imgs)
    return paths


def load_rgb64(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((64, 64), Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def make_blur(rgb: np.ndarray, radius: float = 3.5) -> np.ndarray:
    pil = Image.fromarray((rgb * 255).astype(np.uint8))
    blur = pil.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(blur, dtype=np.float32) / 255.0


def make_saliency(rgb: np.ndarray) -> np.ndarray:
    """Object-centric saliency in [0,1] (NO center blob prior).

    Combines spectral residual + DoG + edge energy so R learns real
    figure/ground instead of a Gaussian spotlight (v1 failure mode).
    """
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
    # spectral residual (Hou & Zhang style)
    g = gray + 1e-6
    fft = np.fft.fft2(g)
    amp = np.abs(fft)
    log_amp = np.log(amp + 1e-8)
    # smooth log amplitude then residual
    from numpy.lib.stride_tricks import sliding_window_view

    pad = np.pad(log_amp, 1, mode="edge")
    windows = sliding_window_view(pad, (3, 3))
    avg = windows.mean(axis=(-1, -2))
    residual = log_amp - avg
    sal_sr = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(fft)))) ** 2
    pil = Image.fromarray((gray * 255).astype(np.uint8))
    blur = np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=4)), dtype=np.float32) / 255.0
    dog = np.abs(gray - blur)
    # Sobel-ish via PIL emboss substitute: local contrast
    blur2 = np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=1.2)), dtype=np.float32) / 255.0
    edge = np.abs(gray - blur2)
    sal = 0.50 * sal_sr + 0.30 * dog + 0.20 * edge
    # mild smooth then normalize
    sal_pil = Image.fromarray((sal / (sal.max() + 1e-8) * 255).astype(np.uint8))
    sal = np.asarray(sal_pil.filter(ImageFilter.GaussianBlur(radius=1.5)), dtype=np.float32) / 255.0
    sal = sal - sal.min()
    mx = float(sal.max()) if float(sal.max()) > 1e-8 else 1.0
    return (sal / mx).astype(np.float32)


class TCDADataset(Dataset):
    def __init__(
        self,
        eeg: np.ndarray,
        depth: np.ndarray,
        image_paths: list[Path],
        blur_radius: float = 3.5,
    ):
        assert len(eeg) == len(depth) == len(image_paths)
        self.eeg = eeg.astype(np.float32)
        self.depth = depth.astype(np.float32)
        if self.depth.ndim == 3:
            self.depth = self.depth[:, None]
        self.paths = image_paths
        self.blur_radius = blur_radius

    def __len__(self) -> int:
        return len(self.eeg)

    def __getitem__(self, idx: int):
        z = self.eeg[idx]
        z = z / (np.linalg.norm(z) + 1e-8)
        rgb = load_rgb64(self.paths[idx])
        blur = make_blur(rgb, self.blur_radius)
        sal = make_saliency(rgb)
        depth = self.depth[idx]
        # normalize depth per-sample for stable regression
        d = depth.copy()
        d = d - d.mean()
        std = float(d.std()) if float(d.std()) > 1e-6 else 1.0
        d = d / std
        return (
            torch.from_numpy(z),
            torch.from_numpy(blur.transpose(2, 0, 1)),  # 3,64,64
            torch.from_numpy(d.astype(np.float32)),  # 1,64,64
            torch.from_numpy(sal[None]),  # 1,64,64
        )


def ssim_loss(x: torch.Tensor, y: torch.Tensor, win: int = 7) -> torch.Tensor:
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


def pearson_np(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


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
    ap.add_argument("--depth-train-npy", type=str, required=True)
    ap.add_argument("--depth-test-npy", type=str, required=True)
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--num-epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--blur-radius", type=float, default=3.5)
    ap.add_argument("--ode-steps", type=int, default=10)
    ap.add_argument("--early-stop-patience", type=int, default=8)
    ap.add_argument("--w-pc", type=float, default=1.0)
    ap.add_argument("--w-ssim", type=float, default=0.35)
    ap.add_argument("--w-pf", type=float, default=0.45)
    ap.add_argument("--w-r", type=float, default=0.55)
    ap.add_argument("--w-cfm-pc", type=float, default=0.25)
    ap.add_argument("--w-cfm-pf", type=float, default=0.20)
    ap.add_argument("--w-cfm-r", type=float, default=0.15)
    ap.add_argument("--w-c2f", type=float, default=0.15)
    ap.add_argument("--w-hier-consist", type=float, default=0.20)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    z_tr = np.load(args.eeg_train_npy).astype(np.float32)
    z_te = np.load(args.eeg_test_npy).astype(np.float32)
    d_tr = np.load(args.depth_train_npy).astype(np.float32)
    d_te = np.load(args.depth_test_npy).astype(np.float32)
    img_root = Path(args.images_root)
    paths_tr = list_split_images(img_root, "train")
    paths_te = list_split_images(img_root, "test")
    assert len(paths_tr) == len(z_tr), f"train images {len(paths_tr)} != eeg {len(z_tr)}"
    assert len(paths_te) == len(z_te)

    ds = TCDADataset(z_tr, d_tr, paths_tr, blur_radius=args.blur_radius)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

    model = TCDAModel(in_dim=z_tr.shape[1]).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)

    history = []
    best_score, best_ep, stale = -1e9, 0, 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        meters = {k: 0.0 for k in ["loss", "pc", "ssim", "pf", "r", "cfm", "hier"]}
        n_ok = 0
        for zb, blur, depth, sal in tqdm(loader, desc=f"tcda-{epoch}"):
            zb = zb.to(device)
            blur = blur.to(device)
            depth = depth.to(device)
            sal = sal.to(device)
            opt.zero_grad(set_to_none=True)
            pc, pf, r = model.forward_heads(zb)
            loss_pc = F.l1_loss(pc, blur)
            loss_ssim = ssim_loss(pc, blur)
            loss_pf = F.l1_loss(pf, depth)
            loss_r = F.l1_loss(r, sal)
            loss_cfm = (
                args.w_cfm_pc * spatial_cfm_loss(model.cfm_pc, blur, zb)
                + args.w_cfm_pf * spatial_cfm_loss(model.cfm_pf, depth, zb)
                + args.w_cfm_r * spatial_cfm_loss(model.cfm_r, sal, zb)
            )
            # hierarchical: low-pass of fine depth energy should match coarse luminance
            pc_energy = model.pc_to_energy(pc)
            pf_blur = F.avg_pool2d(torch.sigmoid(pf), 5, 1, 2)
            # map depth to [0,1] roughly for consistency
            pf_n = (pf_blur - pf_blur.amin(dim=(2, 3), keepdim=True)) / (
                pf_blur.amax(dim=(2, 3), keepdim=True) - pf_blur.amin(dim=(2, 3), keepdim=True) + 1e-6
            )
            pc_n = (pc_energy - pc_energy.amin(dim=(2, 3), keepdim=True)) / (
                pc_energy.amax(dim=(2, 3), keepdim=True) - pc_energy.amin(dim=(2, 3), keepdim=True) + 1e-6
            )
            loss_hier = F.l1_loss(pf_n, pc_n.detach())
            # c2f CFM: transport coarse energy → fine depth (normalized)
            loss_c2f = spatial_cfm_loss(model.cfm_c2f, depth, zb)
            loss = (
                args.w_pc * loss_pc
                + args.w_ssim * loss_ssim
                + args.w_pf * loss_pf
                + args.w_r * loss_r
                + loss_cfm
                + args.w_c2f * loss_c2f
                + args.w_hier_consist * loss_hier
            )
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            meters["loss"] += float(loss.item())
            meters["pc"] += float(loss_pc.item())
            meters["ssim"] += float(loss_ssim.item())
            meters["pf"] += float(loss_pf.item())
            meters["r"] += float(loss_r.item())
            meters["cfm"] += float(loss_cfm.item())
            meters["hier"] += float(loss_hier.item())
            n_ok += 1
        sched.step()
        if n_ok == 0:
            raise RuntimeError("no finite batches")
        for k in meters:
            meters[k] /= n_ok

        # eval on test (batched heads + CFM blend)
        model.eval()
        with torch.no_grad():
            zt = torch.from_numpy(z_te / np.linalg.norm(z_te, axis=1, keepdims=True).clip(1e-8)).to(device)
            pcs, pfs, rs = [], [], []
            strengths = []
            bs = 32
            for i in range(0, len(zt), bs):
                zb = zt[i : i + bs]
                pc_h, pf_h, r_h = model.forward_heads(zb)
                pc_c = model.cfm_pc.decode(zb, steps=args.ode_steps).clamp(0, 1)
                pf_c = model.cfm_pf.decode(zb, steps=args.ode_steps)
                r_c = model.cfm_r.decode(zb, steps=args.ode_steps).clamp(0, 1)
                pc = 0.55 * pc_c + 0.45 * pc_h
                pf = 0.55 * pf_c + 0.45 * pf_h
                r = 0.55 * r_c + 0.45 * r_h
                pcs.append(pc.cpu())
                pfs.append(pf.cpu())
                rs.append(r.cpu())
                # gate features
                pc_e = pc.mean(dim=(1, 2, 3))
                pf_e = pf.mean(dim=(1, 2, 3))
                r_e = r.mean(dim=(1, 2, 3))
                sem = zb.norm(dim=-1)  # ~1 after l2
                feat = torch.stack([pc_e, pf_e, r_e, sem], dim=-1)
                strengths.append(model.gate(feat).cpu())
            pc_te = torch.cat(pcs).numpy()
            pf_te = torch.cat(pfs).numpy()
            r_te = torch.cat(rs).numpy()
            strength_te = torch.cat(strengths).numpy()

        # build GT blur/sal for metrics (test only 200 — cheap)
        blur_gt, sal_gt = [], []
        for p in paths_te:
            rgb = load_rgb64(p)
            blur_gt.append(make_blur(rgb, args.blur_radius).transpose(2, 0, 1))
            sal_gt.append(make_saliency(rgb)[None])
        blur_gt = np.stack(blur_gt)
        sal_gt = np.stack(sal_gt)
        d_te_n = d_te[:, None] if d_te.ndim == 3 else d_te
        # per-sample normalize depth gt like train
        d_gt = []
        for i in range(len(d_te_n)):
            d = d_te_n[i].astype(np.float32)
            d = d - d.mean()
            std = float(d.std()) if float(d.std()) > 1e-6 else 1.0
            d_gt.append(d / std)
        d_gt = np.stack(d_gt)

        pc_p = float(np.mean([pearson_np(pc_te[i], blur_gt[i]) for i in range(len(pc_te))]))
        pf_p = float(np.mean([pearson_np(pf_te[i], d_gt[i]) for i in range(len(pf_te))]))
        r_p = float(np.mean([pearson_np(r_te[i], sal_gt[i]) for i in range(len(r_te))]))
        pc_mae = float(np.mean(np.abs(pc_te - blur_gt)))
        score = 0.45 * pc_p + 0.30 * pf_p + 0.25 * r_p - 0.05 * min(pc_mae, 1.0)
        row = {
            "epoch": epoch,
            **meters,
            "pc_pearson": pc_p,
            "pf_pearson": pf_p,
            "r_pearson": r_p,
            "pc_mae": pc_mae,
            "strength_mean": float(strength_te.mean()),
            "score": score,
        }
        history.append(row)
        print(
            f"[ep {epoch}] loss={meters['loss']:.4f} pc_p={pc_p:.3f} pf_p={pf_p:.3f} "
            f"r_p={r_p:.3f} str={row['strength_mean']:.3f} score={score:.4f}"
        )

        if score > best_score:
            best_score, best_ep, stale = score, epoch, 0
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "in_dim": z_tr.shape[1],
                    "metrics": row,
                    "blur_radius": args.blur_radius,
                    "ode_steps": args.ode_steps,
                },
                out / "checkpoint_tcda_best.pth",
            )
            np.save(out / "pred_pc_test.npy", pc_te.astype(np.float16))
            np.save(out / "pred_pf_test.npy", pf_te.astype(np.float32))
            np.save(out / "pred_r_test.npy", r_te.astype(np.float32))
            np.save(out / "pred_strength_test.npy", strength_te.astype(np.float32))
        else:
            stale += 1
            if stale >= args.early_stop_patience:
                print(f"[EARLY STOP] no improve for {stale} epochs (best={best_ep})")
                break

    if best_ep == 0:
        raise RuntimeError("no checkpoint")

    # export RGB maps from best preds
    ckpt = torch.load(out / "checkpoint_tcda_best.pth", map_location="cpu", weights_only=False)
    pc_te = np.load(out / "pred_pc_test.npy").astype(np.float32)
    pf_te = np.load(out / "pred_pf_test.npy").astype(np.float32)
    r_te = np.load(out / "pred_r_test.npy").astype(np.float32)
    blur_dir = out / "pred_pc_rgb_512"
    depth_dir = out / "pred_pf_depth_rgb_512"
    sal_dir = out / "pred_r_sal_rgb_512"
    blur_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    sal_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(pc_te)):
        rgb = (pc_te[i].transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)
        Image.fromarray(rgb).resize((512, 512), Image.Resampling.BICUBIC).save(blur_dir / f"{i:03d}.png")
        Image.fromarray(depth_to_rgb_u8(pf_te[i, 0])).resize((512, 512), Image.Resampling.BICUBIC).save(
            depth_dir / f"{i:03d}.png"
        )
        Image.fromarray(depth_to_rgb_u8(r_te[i, 0])).resize((512, 512), Image.Resampling.BICUBIC).save(
            sal_dir / f"{i:03d}.png"
        )

    import pandas as pd

    pd.DataFrame(history).to_csv(out / "tcda_history.csv", index=False)
    report = {
        "pipeline": "TCDA",
        "claim": "Tri-channel decode alignment: frozen S(a40) + multi-granular P(Pc/Pf) + relational R",
        "best_epoch": best_ep,
        "best_score": best_score,
        "best_metrics": ckpt.get("metrics", {}),
        "pred_pc_rgb": str(blur_dir),
        "pred_pf_depth_rgb": str(depth_dir),
        "pred_r_sal_rgb": str(sal_dir),
        "strength_npy": str(out / "pred_strength_test.npy"),
        "n_train": int(len(z_tr)),
        "n_test": int(len(z_te)),
        "early_stopped": stale >= args.early_stop_patience,
    }
    (out / "tcda_train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
