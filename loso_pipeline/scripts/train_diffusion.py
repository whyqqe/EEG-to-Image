#!/usr/bin/env python
"""Stage 3 CLI: train EEG->IP-Adapter projector (+ optional UNet LoRA) on frozen SDXL."""
from __future__ import annotations

import argparse

from loso import paths
from loso.models.condition import ConditionConfig
from loso.train.diffusion import DiffusionConfig, run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-subject", default="sub-08")
    ap.add_argument("--align-ckpt", required=True,
                    help="Stage-2 best.pt (encoder is frozen from it)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16", "none"])
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps-per-epoch", type=int, default=0)
    ap.add_argument("--prompt", default="", help="text prompt; empty = IP-only")
    ap.add_argument("--snr-gamma", type=float, default=5.0)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--d-inv", type=int, default=512,
                    help="overridden by the align checkpoint when present")
    args = ap.parse_args()

    paths.ensure_dirs()
    cfg = DiffusionConfig(
        test_subject=args.test_subject,
        align_ckpt=args.align_ckpt,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, warmup_steps=args.warmup_steps,
        amp_dtype=args.amp_dtype, num_workers=args.num_workers,
        log_every=args.log_every, save_every=args.save_every, seed=args.seed,
        max_steps_per_epoch=args.max_steps_per_epoch, prompt=args.prompt,
        snr_gamma=args.snr_gamma,
        condition=ConditionConfig(
            d_inv=args.d_inv, lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha, use_lora=not args.no_lora,
        ),
    )
    path = run(cfg)
    print(f"[train_diffusion] best -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
