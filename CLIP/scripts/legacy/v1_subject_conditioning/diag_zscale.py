#!/usr/bin/env python
"""Why does the support-conditioned Stage 1 fail to learn? Two candidate mechanisms.

Probe 637386: with `z_source: support`, both `anchor` arms went dead -- `img` pinned at
4.2767 (= ln 72, the no-information value for a 72-row batch), test top-1 0.50 = chance --
while the `ids` control trained. Both dead arms show `dec` (HSIC subject dependence) RISING
from 0.486 to 0.516, the same signature the `z_norm: unit` arm had, so it is probably one
mechanism rather than two. Two candidates remain:

  H1 (scale)  the conditioning input is too large. `z_from_support` returns the raw encoder
              output, `|z| ~ 4.5`; the id table sits at `|z| ~ 0.16`. The FiLM head's last
              layer is zero-init, so its input is `GELU(Linear(z))`, whose magnitude -- and
              with it the gradient into the modulation -- scales with `|z|`. `z_norm: unit`
              pushed `|z|` to `sqrt(d_z) = 8` and killed that run; the support encoder is at
              4.5, the same order. Prediction: shrinking `z` revives the run.
  H2 (noise)  a fresh support set is drawn every step, so `z` is a different sample each
              step: the subject modulation becomes noise the trunk cannot learn around. The
              id table is a DETERMINISTIC per-subject constant. Prediction: freezing the
              support set revives the run, at any `|z|`.

The two predict that a DIFFERENT intervention works, so the arms below separate them.

  python scripts/diag_zscale.py --steps 400
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import config  # noqa: E402
from samclip.data import things_eeg  # noqa: E402
from samclip.models import build_model  # noqa: E402
from samclip.train import (Trainer, build_loaders, build_subject_support,  # noqa: E402
                           build_targets, build_test_loader, evaluate_fold, to_device)
from samclip.utils import load_config, set_seed  # noqa: E402


def run_arm(tag: str, base: dict, mode: str, steps: int, z_scale: float) -> dict:
    cfg = dict(base)
    cond = dict(cfg.get("conditioning", {}) or {})
    cond["support_anchor"] = "none"
    cond["z_norm"] = "none"
    cond["dropout"] = 0.0          # isolate the mechanism from subject dropout
    cfg["conditioning"] = cond

    set_seed(cfg.get("seed", 2025))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    target_subject = int(cfg["target_subject"])
    sources = cfg.get("source_subjects") or \
        [s for s in config.all_subjects() if s != target_subject]
    channels = (config.CHANNELS_OCCIPITO_PARIETAL
                if cfg.get("channel_set", "all63") == "occipital17" else None)
    data = things_eeg.load_loso(sources, target_subject, channels,
                                mvnn=cfg.get("mvnn", "off"))
    targets_tr, targets_te = build_targets(cfg)
    cfg_eff = dict(cfg)
    cfg_eff["n_subjects"] = data.n_subjects
    cfg_eff["n_channels"] = data.tr_eeg[0].shape[-2]
    cfg_eff["n_timepoints"] = data.tr_eeg[0].shape[-1]
    model = build_model(cfg_eff, targets_tr.shape[2], targets_tr.shape[-1]).to(device)
    model.train()
    loader, _ = build_loaders(cfg, data, targets_tr)
    test_loader = build_test_loader(cfg, data, targets_te)
    trainer = Trainer(model=model, cfg=cfg, device=device, n_subjects=data.n_subjects)
    opt = torch.optim.AdamW(list(model.parameters()) + trainer.criterion_parameters(),
                            lr=cfg.get("lr", 1e-3),
                            weight_decay=cfg.get("weight_decay", 1e-4))
    rng = np.random.default_rng(cfg.get("seed", 2025) + 977)

    fixed_sup, z_norms, imgs = None, [], []
    it = iter(loader)
    for step in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        if mode == "ids":
            sid = batch["subject"].to(device)
            batch = to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            loss, parts = trainer.assemble(batch)
            z = model.conditioner.z_from_ids(sid)
        else:
            if mode == "fixed_support":
                if fixed_sup is None:
                    fixed_sup = build_subject_support(data, batch, cfg, rng)
                sup = fixed_sup
            else:
                sup = build_subject_support(data, batch, cfg, rng)
            batch = to_device(batch, device)
            z_all = model.conditioner.z_from_support(sup.to(device)) * z_scale
            z = z_all[batch["subject"]]
            opt.zero_grad(set_to_none=True)
            loss, parts = trainer.assemble(batch, z=z)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            zn = float(z.norm(dim=-1).mean())
            z_norms.append(zn)
            imgs.append(float(parts["img"]))
            print(f"  [{tag}] step {step:>4} | img {parts['img']:.4f} | |z| {zn:.4f} "
                  f"| dec {parts['dec']:.4f} | var {parts['var']:.4f}", flush=True)

    model.eval()
    m = evaluate_fold(model, test_loader, device)
    z_std = float(np.std(z_norms)) if len(z_norms) > 1 else 0.0
    print(f"  [{tag}] FINAL top1 {m['top1']:.2f} top5 {m['top5']:.2f} "
          f"meanrank {m['mean_rank']:.1f} | mean |z| {np.mean(z_norms):.3f} "
          f"(across-step std {z_std:.3f})", flush=True)
    return {"tag": tag, "top1": m["top1"], "mean_rank": m["mean_rank"],
            "z_norm": float(np.mean(z_norms)), "z_across_step_std": z_std,
            "img_first": imgs[0] if imgs else float("nan"),
            "img_last": imgs[-1] if imgs else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(config.CONFIGS / "loso_sub08.yaml"))
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()
    base = load_config(args.config)

    arms = [
        ("A__ids (control, alive)",            "ids",           1.00),
        ("B1__support fresh  z*1    (dead)",   "fresh_support", 1.00),
        ("B2__support FIXED  z*1    (tests H2)", "fixed_support", 1.00),
        ("B3__support fresh  z*0.03 (tests H1)", "fresh_support", 0.03),
    ]
    out = {}
    for tag, mode, scale in arms:
        print(f"\n{'=' * 72}\n{tag}\n{'=' * 72}", flush=True)
        out[tag] = run_arm(tag, base, mode, args.steps, scale)

    print("\n" + "=" * 72)
    print("VERDICT")
    print("  H1 (scale): B3 revives while B2 stays dead.")
    print("  H2 (noise): B2 revives while B3 stays dead.")
    print("  Both/neither: the mechanism is something else -- keep digging.")
    print("=" * 72)
    for tag, r in out.items():
        print(f"  {tag:<40} top1 {r['top1']:>5.2f} meanrank {r['mean_rank']:>6.1f} "
              f"| |z| {r['z_norm']:>6.3f} step-std {r['z_across_step_std']:.3f} "
              f"| img {r['img_first']:.3f} -> {r['img_last']:.3f}")


if __name__ == "__main__":
    main()
