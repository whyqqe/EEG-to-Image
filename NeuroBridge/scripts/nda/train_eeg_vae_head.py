#!/usr/bin/env python3
"""Train EEG→SDXL-VAE latent head (MindEye-style low-level submodule).

Input: frozen EEG embeds (N,D)
Target: GT VAE latents (N,4,64,64) float16
Output: checkpoint + predicted test latents + decoded blurry RGB (512)
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
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class VAEHead(nn.Module):
    """z → (4, 64, 64): project to 8×8 then CNN upsample (more stable than flat MLP)."""

    def __init__(self, in_dim: int, hidden: int = 1024, spatial: int = 64, ch: int = 4):
        super().__init__()
        self.spatial = spatial
        self.ch = ch
        self.base = 8
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 128 * self.base * self.base),
        )
        self.up = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 16
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 32
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 64
            nn.Conv2d(32, ch, 3, padding=1),
        )
        # small init on last conv to avoid explosion
        nn.init.zeros_(self.up[-1].weight)
        nn.init.zeros_(self.up[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 128, self.base, self.base)
        return self.up(x)


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


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor, scaling: float) -> list[Image.Image]:
    x = (latents.float() / scaling).to(dtype=vae.dtype)
    imgs = vae.decode(x).sample
    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    out = []
    for i in range(imgs.shape[0]):
        arr = (imgs[i].float().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        out.append(Image.fromarray(arr))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg-train-npy", type=str, required=True)
    ap.add_argument("--eeg-test-npy", type=str, required=True)
    ap.add_argument("--vae-train-npy", type=str, required=True)
    ap.add_argument("--vae-test-npy", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--num-epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decode-rgb", action="store_true", help="export blurry RGB for test")
    ap.add_argument(
        "--val-split-json",
        default="",
        help="leakfree.py split; when given, checkpoint selection uses held-in train concepts "
        "instead of the TEST set (audit finding P1/A4)",
    )
    ap.add_argument("--scaling-factor", type=float, default=0.13025)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rgb_dir = out / "pred_lowlevel_rgb_512"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    z_tr = np.load(args.eeg_train_npy).astype(np.float32)
    z_te = np.load(args.eeg_test_npy).astype(np.float32)
    v_tr = np.load(args.vae_train_npy).astype(np.float32)
    v_te = np.load(args.vae_test_npy).astype(np.float32)
    assert len(z_tr) == len(v_tr) and len(z_te) == len(v_te)
    if not np.isfinite(v_tr).all() or not np.isfinite(v_te).all():
        raise RuntimeError("VAE target latents contain NaN/Inf — rebuild cache with float32 encode")
    z_tr = z_tr / np.linalg.norm(z_tr, axis=1, keepdims=True).clip(1e-8)
    z_te = z_te / np.linalg.norm(z_te, axis=1, keepdims=True).clip(1e-8)

    # ---- LEAK-FREE: select the checkpoint on held-in train concepts, not on test ----
    # Audit finding P1/A4: this script used to pick best_mae on the TEST VAE latents.
    z_va, v_va = None, None
    if args.val_split_json:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import leakfree as LF

        sp = LF.load(args.val_split_json)
        fi = LF.rows_for(sp, "fit", len(z_tr))
        vi = LF.rows_for(sp, "val_b", len(z_tr))
        assert len(set(fi.tolist()) & set(vi.tolist())) == 0
        z_va, v_va = z_tr[vi], v_tr[vi]
        z_tr, v_tr = z_tr[fi], v_tr[fi]
        print(f"[leakfree] fit {z_tr.shape[0]} rows | val {z_va.shape[0]} rows (held-in train concepts)")

    # normalize targets for stable regression; denorm at export
    # (statistics from the FIT split only, so val/test never touch the scaler)
    t_mean = v_tr.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    t_std = v_tr.std(axis=(0, 2, 3), keepdims=True).astype(np.float32).clip(1e-3)
    np.save(out / "target_mean.npy", t_mean.squeeze())
    np.save(out / "target_std.npy", t_std.squeeze())
    v_tr_n = (v_tr - t_mean) / t_std
    v_te_n = (v_te - t_mean) / t_std

    ds = TensorDataset(torch.from_numpy(z_tr), torch.from_numpy(v_tr_n))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

    head = VAEHead(in_dim=z_tr.shape[1], spatial=v_tr.shape[-1], ch=v_tr.shape[1]).to(device)
    opt = optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    best_mae, best_ep = 1e9, 0
    t_mean_t = torch.from_numpy(t_mean).to(device)
    t_std_t = torch.from_numpy(t_std).to(device)

    for epoch in range(1, args.num_epochs + 1):
        head.train()
        ep = 0.0
        n_ok = 0
        for zb, vb in tqdm(loader, desc=f"vae-head-{epoch}"):
            zb, vb = zb.to(device), vb.to(device)
            opt.zero_grad()
            pred = head(zb)
            loss = F.l1_loss(pred, vb)
            if not torch.isfinite(loss):
                print("[WARN] non-finite loss, skip batch")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            ep += float(loss.item())
            n_ok += 1
        if n_ok == 0:
            raise RuntimeError("all batches non-finite — abort")
        head.eval()
        with torch.no_grad():
            # select on held-in val when available; test is exported but never used to choose
            if z_va is not None:
                pv = head(torch.from_numpy(z_va).to(device))
                sel_pred = ((pv * t_std_t + t_mean_t).cpu().numpy())
                sel_ref = v_va
            else:
                sel_pred, sel_ref = None, None
            pred_n = head(torch.from_numpy(z_te).to(device))
            pred_te = (pred_n * t_std_t + t_mean_t).cpu().numpy()
        if not np.isfinite(pred_te).all():
            print(f"[WARN] ep {epoch} pred has NaN — skip checkpoint")
            history.append({"epoch": epoch, "loss": ep / n_ok, "mae": None, "pearson": None, "bad": True})
            continue
        test_mae = float(np.mean(np.abs(pred_te - v_te)))
        if sel_pred is None:
            mae = test_mae
        else:
            mae = float(np.mean(np.abs(sel_pred - sel_ref)))
        rs = []
        for i in range(len(pred_te)):
            a, b = pred_te[i].ravel(), v_te[i].ravel()
            rs.append(0.0 if a.std() < 1e-8 or b.std() < 1e-8 else float(np.corrcoef(a, b)[0, 1]))
        row = {"epoch": epoch, "loss": ep / n_ok, "mae": mae, "test_mae": test_mae,
               "selected_on": "val" if sel_pred is not None else "test(contaminated)",
               "pearson": float(np.mean(rs))}
        history.append(row)
        print(f"[ep {epoch}] loss={row['loss']:.4f} val_mae={mae:.4f} test_mae={test_mae:.4f} pearson={row['pearson']:.3f}")
        if mae < best_mae:
            best_mae, best_ep = mae, epoch
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": head.state_dict(),
                    "in_dim": z_tr.shape[1],
                    "spatial": v_tr.shape[-1],
                    "ch": v_tr.shape[1],
                    "metrics": row,
                    "scaling_factor": args.scaling_factor,
                    "target_mean": t_mean.squeeze(),
                    "target_std": t_std.squeeze(),
                },
                out / "checkpoint_vae_head_best.pth",
            )
            np.save(out / "pred_vae_test.npy", pred_te.astype(np.float16))

    if best_ep == 0 or not (out / "checkpoint_vae_head_best.pth").is_file():
        raise RuntimeError("no valid checkpoint saved (training never produced finite MAE)")

    # reload best + optional RGB decode
    ckpt = torch.load(out / "checkpoint_vae_head_best.pth", map_location=device, weights_only=False)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    mean = torch.as_tensor(ckpt["target_mean"], device=device, dtype=torch.float32).view(1, -1, 1, 1)
    std = torch.as_tensor(ckpt["target_std"], device=device, dtype=torch.float32).view(1, -1, 1, 1)
    with torch.no_grad():
        pred_n = head(torch.from_numpy(z_te).to(device))
        pred_te = (pred_n * std + mean)
    np.save(out / "pred_vae_test.npy", pred_te.cpu().numpy().astype(np.float16))

    if args.decode_rgb:
        import os

        hub = Path(os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
        vae = resolve_vae(hub, device)
        rgb_dir.mkdir(parents=True, exist_ok=True)
        bs = 8
        for start in tqdm(range(0, len(pred_te), bs), desc="decode-rgb"):
            chunk = pred_te[start : start + bs]
            imgs = decode_latents(vae, chunk, args.scaling_factor)
            for j, im in enumerate(imgs):
                im.save(rgb_dir / f"{start + j:03d}.png")

    import pandas as pd

    pd.DataFrame(history).to_csv(out / "vae_head_history.csv", index=False)
    report = {
        "pipeline": "eeg_vae_lowlevel_head",
        "best_epoch": best_ep,
        "best_mae": best_mae,
        "final": history[-1] if history else {},
        "pred_vae": str(out / "pred_vae_test.npy"),
        "pred_rgb_dir": str(rgb_dir) if args.decode_rgb else None,
        "n_train": int(len(z_tr)),
        "n_test": int(len(z_te)),
        "eeg_dim": int(z_tr.shape[1]),
    }
    (out / "vae_head_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
