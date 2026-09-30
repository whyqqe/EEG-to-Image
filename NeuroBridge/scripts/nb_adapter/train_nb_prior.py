#!/usr/bin/env python3
"""Phase 1: NB-conditioned diffusion prior (scratch or ATM-pretrained init)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from train_nb_adapter import concept_split, l2_np, retrieval_metrics  # noqa: E402

BRAINIT_SRC = Path("/project/peilab/why/eeg-brainit/src")
sys.path.insert(0, str(BRAINIT_SRC))
from eeg_brainit.models.atm_diffusion_prior import (  # noqa: E402
    AtmDiffusionPriorPipe,
    DiffusionPriorUNet,
)


def train_prior(
    cond_tr: torch.Tensor,
    y_tr: torch.Tensor,
    cond_va: torch.Tensor,
    y_va: torch.Tensor,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    sample_steps: int,
    init_ckpt: str | None,
    cond_dim: int,
) -> dict:
    prior = DiffusionPriorUNet(embed_dim=1024, cond_dim=cond_dim, dropout=0.1).to(device)
    if init_ckpt:
        state = torch.load(init_ckpt, map_location="cpu", weights_only=True)
        prior.load_state_dict(state, strict=True)
        print(f"[INFO] loaded prior init from {init_ckpt}")
    pipe = AtmDiffusionPriorPipe(prior, device)
    opt = torch.optim.AdamW(prior.parameters(), lr=lr, weight_decay=1e-2)
    loader = DataLoader(TensorDataset(cond_tr, y_tr), batch_size=batch_size, shuffle=True)
    best_val, best_epoch, best_state, bad = -1.0, -1, None, 0
    history = []
    t0 = time.time()

    def val_cos() -> float:
        idx = torch.randperm(len(cond_va))[: min(512, len(cond_va))]
        with torch.no_grad():
            pred = pipe.generate(
                cond_va[idx].to(device),
                num_inference_steps=sample_steps,
                guidance_scale=5.0,
            )
            pred = F.normalize(pred, dim=-1)
            tgt = F.normalize(y_va[idx].to(device), dim=-1)
            return float((pred * tgt).sum(dim=-1).mean().item())

    for ep in range(1, epochs + 1):
        prior.train()
        loss_sum, n = 0.0, 0
        for cond, y1 in loader:
            cond, y1 = cond.to(device), y1.to(device)
            y1 = F.normalize(y1, dim=-1)
            noise = torch.randn_like(y1)
            timesteps = torch.randint(
                0, pipe.scheduler.config.num_train_timesteps, (y1.shape[0],), device=device
            ).long()
            noisy = pipe.scheduler.add_noise(y1, noise, timesteps)
            pred = prior(noisy, timesteps.float(), cond)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * y1.shape[0]
            n += y1.shape[0]
        vc = val_cos() if ep == 1 or ep % 2 == 0 or ep == epochs else best_val
        history.append({"epoch": ep, "loss": loss_sum / max(n, 1), "val_cos": vc})
        print(f"[prior] ep={ep:03d} loss={history[-1]['loss']:.4f} val_cos={vc:.4f}")
        if vc > best_val + 1e-5:
            best_val, best_epoch = vc, ep
            best_state = {k: v.detach().cpu().clone() for k, v in prior.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        prior.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_val_cos": best_val,
        "seconds": time.time() - t0,
        "history": history,
        "pipe": AtmDiffusionPriorPipe(prior, device),
        "prior": prior,
    }


@torch.no_grad()
def sample_embeds(
    pipe: AtmDiffusionPriorPipe,
    cond: np.ndarray,
    device: torch.device,
    steps: int,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    cond_t = torch.from_numpy(cond.astype(np.float32))
    outs = []
    for s in range(n_samples):
        gen = torch.Generator(device=device).manual_seed(seed + s)
        chunks = []
        for i in range(0, len(cond_t), 64):
            chunks.append(
                pipe.generate(
                    cond_t[i : i + 64].to(device),
                    num_inference_steps=steps,
                    guidance_scale=5.0,
                    generator=gen,
                ).float().cpu().numpy()
            )
        outs.append(l2_np(np.concatenate(chunks, axis=0)))
    # average multi-sample embeddings
    return l2_np(np.mean(outs, axis=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dir", type=str, required=True)
    ap.add_argument("--clip-train", type=str, required=True)
    ap.add_argument("--clip-test", type=str, required=True)
    ap.add_argument("--gallery", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--tag", type=str, default="prior_scratch")
    ap.add_argument("--input-key", type=str, default="raw", choices=["proj", "raw"])
    ap.add_argument("--init-ckpt", type=str, default="", help="ATM diffusion_prior.pt for warm start")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--n-samples", type=int, default=3, help="multi-sample average at export")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    embed_dir = Path(args.embed_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train = np.load(embed_dir / f"z_eeg_{args.input_key}_train.npy")
    x_test = np.load(embed_dir / f"z_eeg_{args.input_key}_test.npy")
    y_train = np.load(args.clip_train).astype(np.float32)
    y_test = np.load(args.clip_test).astype(np.float32)
    gallery = np.load(args.gallery).astype(np.float32)

    train_idx, val_idx, _ = concept_split(1654, 10, 0.1, args.seed)
    cond_dim = int(x_train.shape[1])

    pack = train_prior(
        torch.from_numpy(x_train[train_idx]),
        torch.from_numpy(y_train[train_idx]),
        torch.from_numpy(x_train[val_idx]),
        torch.from_numpy(y_train[val_idx]),
        device,
        args.epochs,
        1e-3,
        min(args.batch_size, 512),
        args.patience,
        args.sample_steps,
        args.init_ckpt if args.init_ckpt else None,
        cond_dim,
    )
    pipe = pack.pop("pipe")
    prior = pack.pop("prior")
    torch.save(prior.state_dict(), out_dir / f"{args.tag}_prior.pt")

    pred_test = sample_embeds(pipe, x_test, device, args.sample_steps, args.n_samples, args.seed + 100)
    np.save(out_dir / f"{args.tag}_test_clip_1024.npy", pred_test.astype(np.float32))

    report = {
        "tag": args.tag,
        "input_key": args.input_key,
        "init_ckpt": args.init_ckpt or None,
        "best_epoch": pack["best_epoch"],
        "best_val_cos": pack["best_val_cos"],
        "test_gt_cos": float(np.mean(np.sum(pred_test * l2_np(y_test), axis=1))),
        "test_retrieval": retrieval_metrics(pred_test, gallery),
        "n_samples": args.n_samples,
    }
    (out_dir / f"{args.tag}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[OK] {out_dir / args.tag}_test_clip_1024.npy")


if __name__ == "__main__":
    main()
