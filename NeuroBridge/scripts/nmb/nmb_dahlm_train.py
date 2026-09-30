#!/usr/bin/env python3
"""DA-HLM-S: Decode-Aligned Hierarchical Latent Model (Simplified).

Unified encode-to-decode alignment in ViT-H CLIP space:
  1) DAH — learnable semantic-perceptual binding (replaces fixed mem_linEns blend)
  2) ATM DiffusionPrior — manifold refinement on ViT-H (proven UNet, not lightweight DDLG)
  3) SOTA distillation — soft target from mem + Fusion-bridge teacher

Single output space, single training script, generation via IP-Adapter ViT-H.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
NB_ADAPTER = SCRIPT_DIR.parent / "nb_adapter"
sys.path.insert(0, str(NB_ADAPTER))
sys.path.insert(0, str(Path("/project/peilab/why/eeg-brainit/src")))

from train_nb_adapter import (  # noqa: E402
    LinearAdapter,
    concept_split,
    l2_np,
    retrieval_metrics,
)
from train_nb_adapter_ext import train_diffprior  # noqa: E402
from eeg_brainit.models.atm_diffusion_prior import AtmDiffusionPriorPipe  # noqa: E402


def blend_np(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return l2_np(alpha * a + (1.0 - alpha) * b)


class DecodeAlignedHead(nn.Module):
    """DAH: per-sample learnable binding between episodic memory and semantic projection."""

    def __init__(self, din: int = 512, mem_dim: int = 1024, hidden: int = 2048, res_scale: float = 0.1):
        super().__init__()
        dm = din + mem_dim
        self.res_scale = res_scale
        self.sem = nn.Sequential(
            nn.Linear(dm, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, mem_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(dm, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )
        self.res = nn.Sequential(
            nn.Linear(dm + mem_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, mem_dim),
        )

    def forward(self, z: torch.Tensor, mem: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([z, mem], dim=-1)
        z_sem = F.normalize(self.sem(x), dim=-1)
        alpha = torch.sigmoid(self.gate(x))
        z_bind = F.normalize(alpha * mem + (1.0 - alpha) * z_sem, dim=-1)
        delta = self.res(torch.cat([x, z_bind], dim=-1))
        z_out = F.normalize(z_bind + self.res_scale * delta, dim=-1)
        return z_out, alpha.squeeze(-1)


def cosine_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(F.normalize(pred, dim=-1), F.normalize(target, dim=-1))


@torch.no_grad()
def predict_dah(model: DecodeAlignedHead, z: np.ndarray, mem: np.ndarray, device: torch.device, bs: int = 512) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    outs, alphas = [], []
    zt, mt = torch.from_numpy(z.astype(np.float32)), torch.from_numpy(mem.astype(np.float32))
    for i in range(0, len(zt), bs):
        zb, mb = zt[i : i + bs].to(device), mt[i : i + bs].to(device)
        zo, a = model(zb, mb)
        outs.append(zo.cpu().numpy())
        alphas.append(a.cpu().numpy())
    return l2_np(np.concatenate(outs, axis=0)), np.concatenate(alphas, axis=0)


@torch.no_grad()
def predict_diffprior(pipe: AtmDiffusionPriorPipe, cond: np.ndarray, device: torch.device, steps: int, bs: int = 64) -> np.ndarray:
    pipe.prior.eval()
    outs = []
    ct = torch.from_numpy(cond.astype(np.float32))
    for i in range(0, len(ct), bs):
        outs.append(
            pipe.generate(ct[i : i + bs].to(device), num_inference_steps=steps, guidance_scale=5.0)
            .float()
            .cpu()
            .numpy()
        )
    return l2_np(np.concatenate(outs, axis=0))


def train_dah(
    model: DecodeAlignedHead,
    z_tr: torch.Tensor,
    mem_tr: torch.Tensor,
    gt_tr: torch.Tensor,
    sota_tr: torch.Tensor,
    z_va: torch.Tensor,
    mem_va: torch.Tensor,
    gt_va: torch.Tensor,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    lambda_sota: float,
    alpha_target: float,
    lambda_alpha: float,
) -> dict:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(z_tr, mem_tr, gt_tr, sota_tr), batch_size=batch_size, shuffle=True)
    best_val, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = n = 0.0
        for zb, mb, yb, sb in loader:
            zb, mb, yb, sb = zb.to(device), mb.to(device), yb.to(device), sb.to(device)
            pred, alpha = model(zb, mb)
            loss = cosine_mse(pred, yb) + lambda_sota * cosine_mse(pred, sb)
            if lambda_alpha > 0:
                loss = loss + lambda_alpha * ((alpha - alpha_target) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * zb.shape[0]
            n += zb.shape[0]
        model.eval()
        with torch.no_grad():
            pv, _ = model(z_va.to(device), mem_va.to(device))
            val_cos = float((F.normalize(pv, dim=-1) * F.normalize(gt_va.to(device), dim=-1)).sum(-1).mean().item())
        print(f"[dah] ep={ep:03d} loss={loss_sum/max(n,1):.4f} val_cos={val_cos:.4f}")
        if val_cos > best_val + 1e-5:
            best_val, best_state, bad = val_cos, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_cos": best_val, "model": model}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train DA-HLM-S (DAH + DiffusionPrior)")
    ap.add_argument("--embed-dir", type=str, required=True)
    ap.add_argument("--mem-train", type=str, required=True)
    ap.add_argument("--mem-test", type=str, required=True)
    ap.add_argument("--fusion-mem-train", type=str, required=True)
    ap.add_argument("--clip-train", type=str, required=True)
    ap.add_argument("--clip-test", type=str, required=True)
    ap.add_argument("--bridge-pt", type=str, required=True, help="linear_adapter.pt for SOTA teacher")
    ap.add_argument("--gallery", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--sota-alpha", type=float, default=0.5, help="teacher blend: alpha*mem + (1-alpha)*bridge(fusion_mem)")
    ap.add_argument("--dah-epochs", type=int, default=80)
    ap.add_argument("--diff-epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--diff-batch-size", type=int, default=512)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--diff-patience", type=int, default=10)
    ap.add_argument("--diff-steps", type=int, default=25)
    ap.add_argument("--lambda-sota", type=float, default=0.3)
    ap.add_argument("--lambda-alpha", type=float, default=0.01)
    ap.add_argument("--alpha-target", type=float, default=0.5)
    ap.add_argument("--blend-dp", type=float, default=0.5, help="blend DAH with DiffPrior at inference")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--skip-diffprior", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    z_tr = np.load(Path(args.embed_dir) / "z_eeg_proj_train.npy").astype(np.float32)
    z_te = np.load(Path(args.embed_dir) / "z_eeg_proj_test.npy").astype(np.float32)
    mem_tr = l2_np(np.load(args.mem_train).astype(np.float32))
    mem_te = l2_np(np.load(args.mem_test).astype(np.float32))
    fus_tr = np.load(args.fusion_mem_train).astype(np.float32)
    clip_tr = l2_np(np.load(args.clip_train).astype(np.float32))
    clip_te = l2_np(np.load(args.clip_test).astype(np.float32))
    gallery = np.load(args.gallery).astype(np.float32)

    # SOTA teacher: mem + linear_bridge(fusion_mem)
    bridge_ckpt = torch.load(args.bridge_pt, map_location="cpu")
    bridge = LinearAdapter(1024, 1024)
    bridge.load_state_dict(bridge_ckpt["state_dict"] if "state_dict" in bridge_ckpt else bridge_ckpt)
    bridge.eval()
    with torch.no_grad():
        sem_tr = F.normalize(bridge(torch.from_numpy(fus_tr)), dim=-1).numpy()
    sota_tr = blend_np(mem_tr, sem_tr, args.sota_alpha)

    # test SOTA teacher uses bridge on fusion_mem_test if available
    fusion_mem_test_path = Path(args.fusion_mem_train).parent / "fusion_mem_test.npy"
    if fusion_mem_test_path.is_file():
        fus_te = np.load(fusion_mem_test_path).astype(np.float32)
        with torch.no_grad():
            sem_te = F.normalize(bridge(torch.from_numpy(fus_te)), dim=-1).numpy()
        sota_te = blend_np(mem_te, sem_te, args.sota_alpha)
    else:
        sota_te = blend_np(mem_te, sem_tr[: len(mem_te)], args.sota_alpha)

    train_idx, val_idx, _ = concept_split(val_frac=0.1, seed=args.seed)

    dah = DecodeAlignedHead(din=z_tr.shape[1], mem_dim=mem_tr.shape[1])
    t0 = time.time()
    dah_pack = train_dah(
        dah,
        torch.from_numpy(z_tr[train_idx]),
        torch.from_numpy(mem_tr[train_idx]),
        torch.from_numpy(clip_tr[train_idx]),
        torch.from_numpy(sota_tr[train_idx]),
        torch.from_numpy(z_tr[val_idx]),
        torch.from_numpy(mem_tr[val_idx]),
        torch.from_numpy(clip_tr[val_idx]),
        device,
        epochs=args.dah_epochs,
        lr=3e-4,
        batch_size=args.batch_size,
        patience=args.patience,
        lambda_sota=args.lambda_sota,
        alpha_target=args.alpha_target,
        lambda_alpha=args.lambda_alpha,
    )
    dah = dah_pack["model"]
    torch.save({"state_dict": dah.state_dict(), "din": z_tr.shape[1], "mem_dim": mem_tr.shape[1]}, out / "dah.pt")

    dah_tr, alpha_tr = predict_dah(dah, z_tr, mem_tr, device)
    dah_te, alpha_te = predict_dah(dah, z_te, mem_te, device)
    np.save(out / "dah_train.npy", dah_tr)
    np.save(out / "dah_test.npy", dah_te)
    np.save(out / "dah_alpha_test.npy", alpha_te)

    # DiffusionPrior on concat(z, mem) — decode manifold refinement
    dp_te = dp_tr = None
    dp_pack = None
    if not args.skip_diffprior:
        cond_tr = np.concatenate([z_tr, mem_tr], axis=1).astype(np.float32)
        cond_te = np.concatenate([z_te, mem_te], axis=1).astype(np.float32)
        dp_pack = train_diffprior(
            torch.from_numpy(cond_tr[train_idx]),
            torch.from_numpy(clip_tr[train_idx]),
            torch.from_numpy(cond_tr[val_idx]),
            torch.from_numpy(clip_tr[val_idx]),
            device,
            epochs=args.diff_epochs,
            lr=1e-3,
            batch_size=args.diff_batch_size,
            patience=args.diff_patience,
            sample_steps=args.diff_steps,
        )
        pipe = dp_pack["pipe"]
        prior = dp_pack["prior"]
        torch.save(
            {"state_dict": prior.state_dict(), "cond_dim": cond_tr.shape[1], "steps": args.diff_steps},
            out / "diffprior.pt",
        )
        dp_tr = predict_diffprior(pipe, cond_tr, device, args.diff_steps)
        dp_te = predict_diffprior(pipe, cond_te, device, args.diff_steps)
        np.save(out / "diffprior_train.npy", dp_tr)
        np.save(out / "diffprior_test.npy", dp_te)

    ba = args.blend_dp
    hybrid_te = blend_np(dah_te, dp_te, ba) if dp_te is not None else dah_te
    hybrid_tr = blend_np(dah_tr, dp_tr, ba) if dp_tr is not None else dah_tr
    np.save(out / "hybrid_test.npy", hybrid_te)
    np.save(out / "hybrid_train.npy", hybrid_tr)

    # Also: DAH + SOTA teacher blend (ensemble of learned + teacher)
    ens_te = blend_np(dah_te, sota_te, 0.5)
    np.save(out / "dah_sota_ens_test.npy", ens_te)
    np.save(out / "sota_teacher_test.npy", sota_te)

    metrics = {
        "sota_teacher": retrieval_metrics(sota_te, gallery),
        "dah": retrieval_metrics(dah_te, gallery),
        "hybrid": retrieval_metrics(hybrid_te, gallery),
        "dah_sota_ens": retrieval_metrics(ens_te, gallery),
        "mem": retrieval_metrics(mem_te, gallery),
    }
    if dp_te is not None:
        metrics["diffprior"] = retrieval_metrics(dp_te, gallery)

    report = {
        "method": "DA-HLM-S",
        "theory": "Decode-Aligned Head + ATM DiffusionPrior in unified ViT-H space",
        "dah_best_val_cos": dah_pack["best_val_cos"],
        "dah_seconds": time.time() - t0,
        "diffprior": {k: v for k, v in (dp_pack or {}).items() if k not in ("pipe", "prior", "model")},
        "alpha_test_mean": float(alpha_te.mean()),
        "alpha_test_std": float(alpha_te.std()),
        "sota_alpha": args.sota_alpha,
        "blend_dp": ba,
        "retrieval_test": metrics,
        "outputs": {
            "dah_test": str(out / "dah_test.npy"),
            "diffprior_test": str(out / "diffprior_test.npy") if dp_te is not None else None,
            "hybrid_test": str(out / "hybrid_test.npy"),
            "dah_sota_ens_test": str(out / "dah_sota_ens_test.npy"),
            "sota_teacher_test": "computed inline",
        },
    }
    (out / "dahlm_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"dah_val": dah_pack["best_val_cos"], "retrieval": metrics}, indent=2))


if __name__ == "__main__":
    main()
