#!/usr/bin/env python3
"""SHB step 3: heteroscedastic geometry field + per-pixel uncertainty gating.

Mechanisms 2 (+1) of the Structural Hypothesis Branch
----------------------------------------------------
2. SPATIAL, uncertainty-bearing geometry instead of a point estimate.
   `GeoDecoder(s_vec) -> (depth_mean, log sigma^2)` at 128^2 (vs 64^2 before),
   trained with a Gaussian NLL so the variance is calibrated.
   Uncertainty is then used to build a SPATIALLY composited ControlNet control
   image (uncertain pixels -> neutral gray). Because `control_image` is just an
   image this spatial gate is FREE, and unlike the scalar CN gate it changes
   *where* structure is imposed rather than *how much* -- a new degree of
   freedom that can move the structure<->semantics frontier instead of sliding
   along it (the scalar gate was proven unable to do so, even with oracle u).

1. TRIAL-LEVEL posterior supplies the hypotheses: the decoder is conditioned on
   a per-trial-subset struct vector, so the spread across subsets yields an
   epistemic uncertainty term that is independent of any other branch's error.

Control dirs produced (512^2 PNG, index-aligned 000..199):
  geo_pt        point estimate (full-trial vector), UNMASKED
  geo_pt_mask   point estimate, spatially masked
  geo_mu_mask   posterior mean over subsets, masked
  geo_h1_mask   subset-0 prediction, masked
  geo_h2_mask   subset-1 prediction, masked
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

GRAY = 128


class GeoDecoder(nn.Module):
    """struct vec (dim) -> depth mean + log-variance at res x res."""

    def __init__(self, dim: int = 512, res: int = 128, base: int = 8, c: int = 256):
        super().__init__()
        self.res, self.base = res, base
        self.fc = nn.Sequential(nn.Linear(dim, 1024), nn.GELU(), nn.Linear(1024, base * base * c))
        self.up = nn.Sequential(
            nn.Conv2d(c, 128, 3, padding=1), nn.GELU(), nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, 3, padding=1), nn.GELU(), nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(), nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 16, 3, padding=1), nn.GELU(), nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(16, 2, 3, padding=1),
        )

    def forward(self, s: torch.Tensor):
        x = self.fc(s).view(-1, 256, self.base, self.base)
        x = self.up(x)
        mean = torch.sigmoid(x[:, 0:1])
        lv = x[:, 1:2].clamp(-8.0, 4.0)
        return mean, lv


def grad_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    px = pred[:, :, 1:] - pred[:, :, :-1]
    py = pred[:, 1:, :] - pred[:, :-1, :]
    gx = gt[:, :, 1:] - gt[:, :, :-1]
    gy = gt[:, 1:, :] - gt[:, :-1, :]
    return F.l1_loss(px, gx) + F.l1_loss(py, gy)


def pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.reshape(len(a), -1).astype(np.float64)
    b = b.reshape(len(b), -1).astype(np.float64)
    ac, bc = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    den = np.sqrt((ac * ac).sum(1) * (bc * bc).sum(1)).clip(1e-12)
    return (ac * bc).sum(1) / den


def depth_to_rgb_u8(d: np.ndarray) -> np.ndarray:
    """Min-max normalise a single map to uint8 (same convention as the baseline)."""
    d = d.astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    return (d * 255.0).clip(0, 255).astype(np.uint8)


def save_maps(maps: np.ndarray, dst: Path, size: int, mask: np.ndarray | None = None) -> None:
    """maps (N,res,res) float -> (N,size,size) PNG, optionally masked to gray."""
    dst.mkdir(parents=True, exist_ok=True)
    for i in range(len(maps)):
        u8 = depth_to_rgb_u8(maps[i])
        img = Image.fromarray(np.stack([u8] * 3, axis=-1)).resize((size, size), Image.Resampling.BICUBIC)
        if mask is not None:
            m = mask[i]
            if m.ndim == 3:
                m = m[0]
            mu8 = (Image.fromarray((m * 255).astype(np.uint8))
                   .resize((size, size), Image.Resampling.BILINEAR))
            arr = np.asarray(img).copy()
            arr[np.asarray(mu8) > 127] = GRAY
            img = Image.fromarray(arr)
        img.save(dst / f"{i:03d}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct-train", type=str, required=True)
    ap.add_argument("--struct-test", type=str, required=True)
    ap.add_argument("--struct-train-sub", type=str, default="")
    ap.add_argument("--struct-test-sub", type=str, default="")
    ap.add_argument("--depth-train", type=str, required=True)
    ap.add_argument("--depth-test", type=str, required=True)
    ap.add_argument("--baseline-depth-test", type=str, default="",
                    help="64^2 depth prediction of the EXISTING structure head, for an apples-to-apples check")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--num-epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-grad", type=float, default=0.5)
    ap.add_argument("--mask-quantile", type=float, default=0.30, help="fraction of MOST uncertain pixels to gray out")
    ap.add_argument("--rgb-size", type=int, default=512)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    s_tr = np.load(args.struct_train).astype(np.float32)
    s_te = np.load(args.struct_test).astype(np.float32)
    d_tr = np.load(args.depth_train).astype(np.float32)[:, None]
    d_te = np.load(args.depth_test).astype(np.float32)[:, None]
    n, R = len(s_tr), args.res
    assert len(d_tr) == n, f"train mismatch {n} vs {len(d_tr)}"

    s_tr_sub = np.load(args.struct_train_sub).astype(np.float32) if args.struct_train_sub else None
    s_te_sub = np.load(args.struct_test_sub).astype(np.float32) if args.struct_test_sub else None
    print(f"[INFO] struct train {s_tr.shape} test {s_te.shape} "
          f"sub_train={None if s_tr_sub is None else s_tr_sub.shape} "
          f"sub_test={None if s_te_sub is None else s_te_sub.shape}")

    # targets at the field resolution
    t_tr = F.interpolate(torch.from_numpy(d_tr), size=(R, R), mode="bicubic", align_corners=False)
    t_te = F.interpolate(torch.from_numpy(d_te), size=(R, R), mode="bicubic", align_corners=False)

    model = GeoDecoder(s_tr.shape[1], R).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[INFO] GeoDecoder params {nparam/1e6:.2f}M  res={R}  n_train={n}")
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # hypothesis pool: full-trial vector + each trial subset
    pool = [s_tr] + ([s_tr_sub[:, k] for k in range(s_tr_sub.shape[1])] if s_tr_sub is not None else [])
    pool_t = [torch.from_numpy(p).float() for p in pool]
    print(f"[INFO] hypothesis pool size {len(pool_t)}")

    loader = DataLoader(TensorDataset(torch.arange(n)), batch_size=args.batch_size, shuffle=True, drop_last=True)
    hist = []
    for ep in range(1, args.num_epochs + 1):
        model.train()
        acc = 0.0
        for (idx,) in loader:
            idx = idx.to(device)
            k = int(torch.randint(0, len(pool_t), (1,)).item())
            sb = pool_t[k][idx].to(device)
            tb = t_tr[idx].to(device)
            opt.zero_grad()
            mean, lv = model(sb)
            nll = (0.5 * torch.exp(-lv) * (mean - tb) ** 2 + 0.5 * lv).mean()
            loss = nll + args.lambda_grad * grad_loss(mean.squeeze(1), tb.squeeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            acc += float(loss.item())
        if ep % 5 == 0 or ep == 1:
            hist.append({"epoch": ep, "loss": acc / max(len(loader), 1)})
            print(f"[ep {ep}] loss={acc/max(len(loader),1):.4f}")

    torch.save({"state_dict": model.state_dict(), "res": R, "dim": s_tr.shape[1]}, out / "geofield_best.pth")

    # ---------------- inference ----------------
    @torch.no_grad()
    def predict(sv: np.ndarray):
        model.eval()
        means, lvs = [], []
        for st in range(0, len(sv), 2048):
            sb = torch.from_numpy(sv[st : st + 2048]).to(device)
            m, lv = model(sb)
            means.append(m.squeeze(1).cpu().numpy())
            lvs.append(lv.squeeze(1).cpu().numpy())
        return np.concatenate(means, 0).astype(np.float32), np.concatenate(lvs, 0).astype(np.float32)

    m_pt, lv_pt = predict(s_te)                      # point estimate from full-trial vector
    hyp_means, hyp_lvs = [], []
    if s_te_sub is not None:
        for k in range(s_te_sub.shape[1]):
            mk, lvk = predict(s_te_sub[:, k])
            hyp_means.append(mk)
            hyp_lvs.append(lvk)
    if hyp_means:
        hm = np.stack(hyp_means, 0)                  # (K,N,R,R)
        mu_map = hm.mean(0)
        epi = hm.std(0)
        ale = np.sqrt(np.exp(np.stack(hyp_lvs, 0)).mean(0))
    else:
        mu_map, epi, ale = m_pt, np.zeros_like(m_pt), np.sqrt(np.exp(lv_pt))

    def zn(x: np.ndarray) -> np.ndarray:
        return ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32)

    unc = zn(epi) + zn(ale)
    thr = np.quantile(unc, 1.0 - args.mask_quantile, axis=(1, 2), keepdims=True)
    mask = (unc > thr).astype(np.uint8)
    print(f"[INFO] mask ratio mean={mask.mean():.3f} (target {args.mask_quantile})")

    # ---------------- exports ----------------
    np.save(out / "pred_depth_point_128.npy", m_pt)
    np.save(out / "pred_depth_mu_128.npy", mu_map)
    np.save(out / "unc_epistemic_128.npy", epi)
    np.save(out / "unc_aleatoric_128.npy", ale)
    np.save(out / "unc_total_mask.npy", mask)
    if hyp_means:
        np.save(out / "pred_depth_hyp_128.npy", np.stack(hyp_means, 0))

    gd = out / "generation_controls"
    save_maps(m_pt, gd / "geo_pt", args.rgb_size, None)
    save_maps(m_pt, gd / "geo_pt_mask", args.rgb_size, mask)
    save_maps(mu_map, gd / "geo_mu_mask", args.rgb_size, mask)
    h_tags = []
    if hyp_means:
        for k in range(min(2, len(hyp_means))):
            tag = f"geo_h{k+1}_mask"
            save_maps(hyp_means[k], gd / tag, args.rgb_size, mask)
            h_tags.append(tag)

    # ---------------- diagnostics vs the 0.689 baseline ----------------
    def down(x: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(x)[:, None]
        return F.interpolate(t, size=(64, 64), mode="area").squeeze(1).numpy()

    r_pt = float(pearson_rows(down(m_pt), d_te.squeeze(1)).mean())
    r_mu = float(pearson_rows(down(mu_map), d_te.squeeze(1)).mean())
    r_h1 = (float(pearson_rows(down(hyp_means[0]), d_te.squeeze(1)).mean()) if hyp_means else None)
    # same computation for the BASELINE 64^2 prediction, for an apples-to-apples check
    r_base = None
    if args.baseline_depth_test and Path(args.baseline_depth_test).is_file():
        bp = np.load(args.baseline_depth_test).astype(np.float32)
        if len(bp) == len(d_te):
            r_base = float(pearson_rows(bp, d_te.squeeze(1)).mean())

    report = {
        "pipeline": "shb_geofield",
        "decoder": f"GeoDecoder(struct vec) -> mean+logvar @ {R}x{R}, Gaussian NLL + grad",
        "hypotheses": {"pool": len(pool_t), "test_subsets": None if s_te_sub is None else int(s_te_sub.shape[1])},
        "mask_quantile": args.mask_quantile,
        "uncertainty": {
            "epistemic_mean": float(epi.mean()), "aleatoric_mean": float(ale.mean()),
            "corr_epi_ale": float(np.corrcoef(epi.ravel(), ale.ravel())[0, 1]),
        },
        "test_depth_pearson_onetwork": r_pt,
        "test_depth_pearson_posterior_mean": r_mu,
        "test_depth_pearson_subset1": r_h1,
        "test_depth_pearson_baseline_64": r_base,
        "baseline_reference": 0.6889,
        "control_dirs": {
            "geo_pt": "point estimate, unmasked",
            "geo_pt_mask": "point estimate, masked",
            "geo_mu_mask": "posterior mean, masked",
            **{t: "subset prediction, masked" for t in h_tags},
        },
        "history": hist,
    }
    (out / "geofield_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
