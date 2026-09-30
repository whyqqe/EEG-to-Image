#!/usr/bin/env python3
"""NMB-DADEM: Memory-conditioned Decode-Aligned Diffusion Latent Embed Mapper.

Inspired by D²-FOSA DDLG but conditions on [NeuroBridge z_proj + episodic mem_vith]
and refines ViT-H CLIP embeddings for IP-Adapter decoding.

Training: L = L_align + λ_e2i * L_E2I + λ_i2e * L_I2E (optional bidirectional)
Inference: DDIM reverse diffusion in CLIP-1024 space → refined embed for generation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "nb_adapter"))
from train_nb_adapter import concept_split, eval_cos, l2_np, retrieval_metrics  # noqa: E402


def sinusoidal_emb(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)


class ConditionEncoder(nn.Module):
    """Fuse NeuroBridge proj + episodic memory (+ optional bridge prior) into cond vector."""

    def __init__(self, din: int, dcond: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, dcond * 2),
            nn.GELU(),
            nn.LayerNorm(dcond * 2),
            nn.Linear(dcond * 2, dcond),
            nn.GELU(),
            nn.LayerNorm(dcond),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FiLMBlock(nn.Module):
    def __init__(self, feat_dim: int, cond_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feat_dim)
        self.beta = nn.Linear(cond_dim, feat_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + self.gamma(cond)) + self.beta(cond)


class LatentDenoiser(nn.Module):
    """MLP denoiser with time + condition FiLM (D²-FOSA DDLG style, lightweight)."""

    def __init__(self, latent_dim: int = 1024, cond_dim: int = 512, hidden: int = 2048, n_blocks: int = 5):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
        )
        self.in_proj = nn.Linear(latent_dim, hidden)
        blocks = []
        for _ in range(n_blocks):
            blocks.append(nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden)))
        self.blocks = nn.ModuleList(blocks)
        self.films = nn.ModuleList([FiLMBlock(hidden, cond_dim) for _ in range(n_blocks)])
        self.out_proj = nn.Linear(hidden, latent_dim)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_emb(t, cond.shape[-1])
        c = cond + self.time_mlp(t_emb)
        h = self.in_proj(z_t)
        for block, film in zip(self.blocks, self.films):
            h = film(block(h), c)
        return self.out_proj(h)


class DDLEM(nn.Module):
    """Full DADEM module: condition encoder + E2I denoiser (+ optional I2E)."""

    def __init__(
        self,
        cond_in: int,
        latent_dim: int = 1024,
        cond_dim: int = 512,
        hidden: int = 2048,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.cond_enc = ConditionEncoder(cond_in, cond_dim)
        self.e2i = LatentDenoiser(latent_dim, cond_dim, hidden)
        self.bidirectional = bidirectional
        self.i2e = LatentDenoiser(latent_dim, cond_dim, hidden) if bidirectional else None
        self.i2e_cond_proj = nn.Linear(latent_dim, cond_dim) if bidirectional else None
        self.align_head = nn.Linear(cond_dim, latent_dim)

    def encode_cond(self, x_cond: torch.Tensor) -> torch.Tensor:
        return self.cond_enc(x_cond)

    def align_predict(self, x_cond: torch.Tensor) -> torch.Tensor:
        return self.align_head(self.encode_cond(x_cond))


def make_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


def make_diffusion(betas: torch.Tensor, device: torch.device) -> dict:
    betas = betas.to(device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "sqrt_alpha_bar": torch.sqrt(alpha_bar),
        "sqrt_one_minus_alpha_bar": torch.sqrt(1.0 - alpha_bar),
    }


def q_sample(z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, diff: dict) -> torch.Tensor:
    sa = diff["sqrt_alpha_bar"][t].unsqueeze(1)
    so = diff["sqrt_one_minus_alpha_bar"][t].unsqueeze(1)
    return sa * z0 + so * noise


def ddpm_loss(denoiser: nn.Module, z0: torch.Tensor, cond: torch.Tensor, diff: dict, t_max: int) -> torch.Tensor:
    b = z0.shape[0]
    t = torch.randint(0, t_max, (b,), device=z0.device)
    noise = torch.randn_like(z0)
    z_t = q_sample(z0, t, noise, diff)
    pred = denoiser(z_t, t, cond)
    return F.mse_loss(pred, noise)


@torch.no_grad()
def ddim_sample(
    denoiser: nn.Module,
    cond: torch.Tensor,
    diff: dict,
    steps: int = 30,
    warm_start: torch.Tensor | None = None,
    warm_t_frac: float = 0.35,
    eta: float = 0.0,
) -> torch.Tensor:
    """Reverse DDIM; warm_start required for CLIP-sphere targets (no pure-noise init)."""
    device = cond.device
    b = cond.shape[0]
    t_max = diff["betas"].shape[0]

    if warm_start is None:
        raise ValueError("warm_start required: CLIP embeddings must not be sampled from N(0,I)")

    t0 = max(2, int(t_max * warm_t_frac))
    t_start = torch.full((b,), t0 - 1, device=device, dtype=torch.long)
    noise = torch.randn_like(warm_start)
    z = q_sample(F.normalize(warm_start, dim=-1), t_start, noise, diff)
    step_ids = np.linspace(t0 - 1, 0, steps, dtype=int).tolist()

    for i, t in enumerate(step_ids):
        t_b = torch.full((b,), t, device=device, dtype=torch.long)
        eps = denoiser(z, t_b, cond)
        alpha_bar_t = diff["alpha_bar"][t]
        z0_pred = (z - diff["sqrt_one_minus_alpha_bar"][t] * eps) / diff["sqrt_alpha_bar"][t].clamp(min=1e-8)
        z0_pred = F.normalize(z0_pred, dim=-1)
        if i == len(step_ids) - 1:
            z = z0_pred
            break
        t_prev = step_ids[i + 1]
        alpha_bar_prev = diff["alpha_bar"][t_prev]
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t).clamp(min=1e-8))
            * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev.clamp(min=1e-8))
        )
        dir_xt = torch.sqrt((1 - alpha_bar_prev - sigma**2).clamp(min=0)) * eps
        z = torch.sqrt(alpha_bar_prev) * z0_pred + dir_xt
        z = F.normalize(z, dim=-1)
        if eta > 0:
            z = z + sigma * torch.randn_like(z)
    return F.normalize(z, dim=-1)


@torch.no_grad()
def predict_embeds(
    model: DDLEM,
    x_cond: np.ndarray,
    device: torch.device,
    diff: dict,
    steps: int = 30,
    warm_start: np.ndarray | None = None,
    warm_t_frac: float = 0.35,
    bs: int = 256,
) -> np.ndarray:
    model.eval()
    outs = []
    xt = torch.from_numpy(x_cond.astype(np.float32))
    ws = torch.from_numpy(warm_start.astype(np.float32)) if warm_start is not None else None
    for i in range(0, len(xt), bs):
        xb = xt[i : i + bs].to(device)
        cond = model.encode_cond(xb)
        align = F.normalize(model.align_head(cond), dim=-1)
        wsb = ws[i : i + bs].to(device) if ws is not None else align
        z = ddim_sample(model.e2i, cond, diff, steps=steps, warm_start=wsb, warm_t_frac=warm_t_frac)
        outs.append(z.float().cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def cosine_align_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(F.normalize(pred, dim=-1), F.normalize(target, dim=-1))


def train_ddlem(
    model: DDLEM,
    x_tr: torch.Tensor,
    mem_tr: torch.Tensor,
    clip_tr: torch.Tensor,
    x_va: torch.Tensor,
    mem_va: torch.Tensor,
    clip_va: torch.Tensor,
    device: torch.device,
    diff: dict,
    epochs: int,
    lr_main: float,
    lr_diff: float,
    weight_decay: float,
    batch_size: int,
    patience: int,
    lambda_align: float,
    lambda_e2i: float,
    lambda_i2e: float,
    t_max: int,
) -> dict:
    model = model.to(device)
    main_params = list(model.cond_enc.parameters()) + list(model.align_head.parameters())
    diff_params = list(model.e2i.parameters())
    if model.i2e is not None:
        diff_params += list(model.i2e.parameters())
        diff_params += list(model.i2e_cond_proj.parameters())  # type: ignore[union-attr]
    opt = torch.optim.AdamW(
        [
            {"params": main_params, "lr": lr_main},
            {"params": diff_params, "lr": lr_diff},
        ],
        weight_decay=weight_decay,
    )
    loader = DataLoader(
        TensorDataset(x_tr, mem_tr, clip_tr),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    best_val = -1.0
    best_state = None
    bad = 0
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        for xb, mb, yb in loader:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            cond_in = torch.cat([xb, mb], dim=-1)
            cond = model.encode_cond(cond_in)
            align_pred = model.align_head(cond)
            loss_align = cosine_align_loss(align_pred, yb)
            loss_e2i = ddpm_loss(model.e2i, yb, cond, diff, t_max)
            loss = lambda_align * loss_align + lambda_e2i * loss_e2i
            if model.i2e is not None:
                i2e_target = F.normalize(model.align_head(cond).detach(), dim=-1)
                clip_cond = model.i2e_cond_proj(F.normalize(yb, dim=-1))  # type: ignore[operator]
                loss_i2e = ddpm_loss(model.i2e, i2e_target, clip_cond, diff, t_max)
                loss = loss + lambda_i2e * loss_i2e
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * xb.shape[0]
            n += xb.shape[0]
        model.eval()
        with torch.no_grad():
            cond_va = model.encode_cond(torch.cat([x_va, mem_va], dim=-1).to(device))
            pred_va = F.normalize(model.align_head(cond_va), dim=-1)
            tgt_va = F.normalize(clip_va.to(device), dim=-1)
            val_cos = float((pred_va * tgt_va).sum(dim=-1).mean().item())
        print(f"[ddlem] ep={ep:03d} loss={loss_sum/max(n,1):.4f} val_align_cos={val_cos:.4f}")
        if val_cos > best_val + 1e-5:
            best_val = val_cos
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_cos": best_val, "model": model}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train NMB-DADEM DDLG module")
    ap.add_argument("--embed-dir", type=str, required=True)
    ap.add_argument("--mem-train", type=str, required=True)
    ap.add_argument("--mem-test", type=str, required=True)
    ap.add_argument("--clip-train", type=str, required=True)
    ap.add_argument("--clip-test", type=str, required=True)
    ap.add_argument("--gallery", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--input-key", type=str, default="proj", choices=["proj", "raw"])
    ap.add_argument("--bidirectional", action="store_true", help="Enable I2E diffusion branch")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr-main", type=float, default=1e-4)
    ap.add_argument("--lr-diff", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--lambda-align", type=float, default=1.0)
    ap.add_argument("--lambda-e2i", type=float, default=0.5)
    ap.add_argument("--lambda-i2e", type=float, default=0.5)
    ap.add_argument("--timesteps", type=int, default=200)
    ap.add_argument("--ddim-steps", type=int, default=50)
    ap.add_argument("--warm-t-frac", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--infer-only", action="store_true")
    ap.add_argument("--checkpoint", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    key = "proj" if args.input_key == "proj" else "raw"
    embed_dir = Path(args.embed_dir)
    x_train = np.load(embed_dir / f"z_eeg_{key}_train.npy").astype(np.float32)
    x_test = np.load(embed_dir / f"z_eeg_{key}_test.npy").astype(np.float32)
    mem_train = l2_np(np.load(args.mem_train).astype(np.float32))
    mem_test = l2_np(np.load(args.mem_test).astype(np.float32))
    clip_train = l2_np(np.load(args.clip_train).astype(np.float32))
    clip_test = l2_np(np.load(args.clip_test).astype(np.float32))
    gallery = np.load(args.gallery).astype(np.float32)

    cond_in = x_train.shape[1] + mem_train.shape[1]
    latent_dim = clip_train.shape[1]

    betas = make_beta_schedule(args.timesteps)
    diff = make_diffusion(betas, device)

    model = DDLEM(
        cond_in=cond_in,
        latent_dim=latent_dim,
        cond_dim=512,
        hidden=2048,
        bidirectional=args.bidirectional,
    )

    ckpt_path = Path(args.checkpoint) if args.checkpoint else out_dir / "ddlem.pt"

    if not args.infer_only:
        tr_idx, va_idx, val_concepts = concept_split(val_frac=args.val_frac, seed=args.seed)
        x_tr = torch.from_numpy(x_train[tr_idx])
        mem_tr = torch.from_numpy(mem_train[tr_idx])
        clip_tr = torch.from_numpy(clip_train[tr_idx])
        x_va = torch.from_numpy(x_train[va_idx])
        mem_va = torch.from_numpy(mem_train[va_idx])
        clip_va = torch.from_numpy(clip_train[va_idx])

        t0 = time.time()
        report = train_ddlem(
            model,
            x_tr,
            mem_tr,
            clip_tr,
            x_va,
            mem_va,
            clip_va,
            device,
            diff,
            epochs=args.epochs,
            lr_main=args.lr_main,
            lr_diff=args.lr_diff,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            patience=args.patience,
            lambda_align=args.lambda_align,
            lambda_e2i=args.lambda_e2i,
            lambda_i2e=args.lambda_i2e,
            t_max=args.timesteps,
        )
        torch.save(model.state_dict(), ckpt_path)
        train_secs = time.time() - t0
    else:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        report = {"best_val_cos": None, "infer_only": True, "model": model}
        train_secs = 0.0

    model = report["model"].to(device)
    model.eval()

    # Inference variants
    x_cond_train = np.concatenate([x_train, mem_train], axis=1)
    x_cond_test = np.concatenate([x_test, mem_test], axis=1)

    with torch.no_grad():
        cond_te = model.encode_cond(torch.from_numpy(x_cond_test).to(device))
        align_te = F.normalize(model.align_head(cond_te), dim=-1).cpu().numpy()
        align_tr = []
        for i in range(0, len(x_cond_train), 512):
            xb = torch.from_numpy(x_cond_train[i : i + 512]).to(device)
            c = model.encode_cond(xb)
            align_tr.append(F.normalize(model.align_head(c), dim=-1).cpu().numpy())
        align_tr = np.concatenate(align_tr, axis=0)

    # DDIM refine from align_head (semantic init, light noise)
    refine_align_tr = predict_embeds(
        model, x_cond_train, device, diff, steps=args.ddim_steps,
        warm_start=align_tr, warm_t_frac=0.30,
    )
    refine_align_te = predict_embeds(
        model, x_cond_test, device, diff, steps=args.ddim_steps,
        warm_start=align_te, warm_t_frac=0.30,
    )

    # DDIM refine from mem (perceptual anchor, heavier noise)
    refine_mem_tr = predict_embeds(
        model, x_cond_train, device, diff, steps=args.ddim_steps,
        warm_start=mem_train, warm_t_frac=args.warm_t_frac,
    )
    refine_mem_te = predict_embeds(
        model, x_cond_test, device, diff, steps=args.ddim_steps,
        warm_start=mem_test, warm_t_frac=args.warm_t_frac,
    )

    def blend(a, b, alpha):
        return l2_np(alpha * a + (1 - alpha) * b)

    blend_align_mem_te = blend(refine_align_te, mem_test, 0.5)
    blend_refine_mem_te = blend(refine_mem_te, mem_test, 0.5)

    np.save(out_dir / "ddlem_align_test.npy", align_te)
    np.save(out_dir / "ddlem_refine_align_test.npy", refine_align_te)
    np.save(out_dir / "ddlem_refine_mem_test.npy", refine_mem_te)
    np.save(out_dir / "ddlem_blend_align_mem_test.npy", blend_align_mem_te)
    np.save(out_dir / "ddlem_blend_refine_mem_test.npy", blend_refine_mem_te)
    # legacy names for shell compatibility
    np.save(out_dir / "ddlem_e2i_warm_test.npy", refine_mem_te)
    np.save(out_dir / "ddlem_blend_a50_test.npy", blend_refine_mem_te)

    metrics = {
        "align_head": retrieval_metrics(align_te, gallery),
        "refine_align": retrieval_metrics(refine_align_te, gallery),
        "refine_mem": retrieval_metrics(refine_mem_te, gallery),
        "blend_align_mem": retrieval_metrics(blend_align_mem_te, gallery),
        "blend_refine_mem": retrieval_metrics(blend_refine_mem_te, gallery),
        "mem_baseline": retrieval_metrics(mem_test, gallery),
    }
    for k, v in metrics.items():
        print(f"[retrieval] {k}: top1={v['top1']:.4f} paired_cos={v['paired_cos']:.4f}")

    full_report = {
        "method": "NMB-DADEM",
        "bidirectional": args.bidirectional,
        "best_val_align_cos": report.get("best_val_cos"),
        "train_seconds": train_secs,
        "timesteps": args.timesteps,
        "ddim_steps": args.ddim_steps,
        "warm_t_frac": args.warm_t_frac,
        "lambdas": {
            "align": args.lambda_align,
            "e2i": args.lambda_e2i,
            "i2e": args.lambda_i2e,
        },
        "retrieval_test": metrics,
        "outputs": {
            "align_test": str(out_dir / "ddlem_align_test.npy"),
            "refine_align_test": str(out_dir / "ddlem_refine_align_test.npy"),
            "refine_mem_test": str(out_dir / "ddlem_refine_mem_test.npy"),
            "blend_align_mem_test": str(out_dir / "ddlem_blend_align_mem_test.npy"),
            "blend_refine_mem_test": str(out_dir / "ddlem_blend_refine_mem_test.npy"),
        },
    }
    (out_dir / "ddlem_report.json").write_text(json.dumps(full_report, indent=2))
    print(json.dumps({"best_val": report.get("best_val_cos"), "retrieval": metrics}, indent=2))


if __name__ == "__main__":
    main()
