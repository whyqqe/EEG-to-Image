"""Few-shot subject-adapter calibration + end-to-end evaluation on the held-out subject.

Calibration (design §14): for k ∈ {5, 10, 25, 50} held-out *training-split* images
of the test subject, freeze the trunk and fit only that subject's SubjectAdapter
(+ optional LayerNorm) so the population-mean prior becomes subject-specific.
k images are drawn from the subject's *train* EEG (1,654 concepts × 10 images),
never from the 200 test concepts -- otherwise the calibration would leak the
evaluation set.

Evaluation then reports:
  * retrieval on the 200 test concepts (top-1/5, 2-way, 40-way)
  * reconstruction metrics on images generated from the calibrated encoder
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from loso import paths
from loso.data import eeg as eeg_mod
from loso.data import things
from loso.data.targets import TargetStore
from loso.eval.metrics import (
    LPIPSMetric, RetrievalScores, cosine_np, pixcorr, psnr,
    retrieval_from_similarity, ssim,
)
from loso.models.condition import (
    ConditionConfig, EEGImageProj, build_sdxl_components, encode_prompts,
    inject_ip_tokens, resolve_ip_adapter_file, resolve_ip_adapter_root,
    sdxl_time_ids,
)
from loso.models.eeg_encoder import EEGEncoder, EncoderConfig
from loso.models.heads import AlignmentHeads, HeadConfig
from loso.train.diffusion import load_align_checkpoint


def calibrate_adapter(encoder: EEGEncoder, store: TargetStore, subject: str,
                      stats: eeg_mod.ChannelStats, k: int, steps: int = 200,
                      lr: float = 1e-3, seed: int = 0, device: str = "cuda") -> dict:
    """Fit a fresh single-subject adapter on k train-split images of `subject`."""
    rng = np.random.default_rng(seed)
    # Sample k distinct image slots from the 16,540 train images.
    slots = rng.choice(paths.N_TRAIN_CONCEPTS * paths.N_IMAGES_PER_CONCEPT,
                       size=k, replace=False)

    # Build a tiny dataset: for each slot, average the subject's 4 train reps.
    arr = eeg_mod.load_eeg(subject, "train", mmap=True)
    xs, ys = [], []
    for slot in slots:
        c, i = divmod(int(slot), paths.N_IMAGES_PER_CONCEPT)
        x = np.asarray(arr[c, i], dtype=np.float32).mean(axis=0)  # (C, T)
        xs.append(stats(torch.from_numpy(x)))
        ys.append(int(slot))
    x = torch.stack(xs).to(device)
    slots_t = torch.tensor(ys, device=device)

    # Replace adapters with a 1-subject bank initialised from the population mean.
    n_adapt = len(encoder.adapters)
    mean_state = []
    for a in encoder.adapters:
        mean_state.append({
            "down": a.down.mean(0, keepdim=True).detach().clone(),
            "up": a.up.mean(0, keepdim=True).detach().clone(),
        })
    from loso.models.eeg_encoder import SubjectAdapter
    new_adapters = torch.nn.ModuleList([
        SubjectAdapter(encoder.cfg.d_model, 1, encoder.cfg.adapter_bottleneck,
                       encoder.cfg.dropout)
        for _ in range(n_adapt)
    ]).to(device)
    for a, init in zip(new_adapters, mean_state):
        with torch.no_grad():
            a.down.copy_(init["down"])
            a.up.copy_(init["up"])
    encoder.adapters = new_adapters
    encoder.cfg.pretrained_subjects = 1

    for p in encoder.parameters():
        p.requires_grad_(False)
    for a in encoder.adapters:
        for p in a.parameters():
            p.requires_grad_(True)

    opt = torch.optim.AdamW(
        [p for p in encoder.adapters.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )
    teacher = store.as_normalized(slots_t, device=device)["clip_image"]
    sid = torch.zeros(k, dtype=torch.long, device=device)

    encoder.train()
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        z = encoder(x, sid, adapter_mode="subject")["z_inv"]
        z = F.normalize(z.float(), dim=-1)
        # Soft cosine pull toward the CLIP image teacher of the k images.
        loss = (1.0 - (z * teacher).sum(-1)).mean()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"  [calib k={k}] step {step}/{steps} loss={float(loss):.4f}",
                  flush=True)
    encoder.eval()
    return {"k": k, "steps": steps, "slots": ys}


@torch.inference_mode()
def evaluate_retrieval(encoder, heads, test_subject, stats, test_store,
                       device, avg_reps: int = 1) -> RetrievalScores:
    ds = eeg_mod.TestEEGDataset(test_subject, stats, avg_trials=avg_reps > 1,
                                avg_reps=avg_reps)
    loader = DataLoader(ds, batch_size=64, shuffle=False,
                        collate_fn=eeg_mod.collate, num_workers=4)
    # Aggregate predictions per concept (average over remaining trials).
    preds = torch.zeros(paths.N_TEST_CONCEPTS, heads.cfg.d_teacher, device=device)
    counts = torch.zeros(paths.N_TEST_CONCEPTS, device=device)
    for batch in loader:
        x = batch["x"].to(device)
        slots = batch["target_slot"].to(device)
        z = encoder(x, None, adapter_mode="mean")["z_inv"]
        # If calibrated to 1 subject, use subject mode with id 0.
        if encoder.cfg.pretrained_subjects == 1:
            z = encoder(x, torch.zeros(x.shape[0], dtype=torch.long, device=device),
                        adapter_mode="subject")["z_inv"]
        p = F.normalize(heads.img(z).float(), dim=-1)
        preds.index_add_(0, slots, p)
        counts.index_add_(0, slots, torch.ones_like(slots, dtype=torch.float))
    preds = F.normalize(preds / counts.clamp_min(1).unsqueeze(1), dim=-1)
    gallery = test_store.as_normalized(
        torch.arange(paths.N_TEST_CONCEPTS, device=device), device=device,
    )["clip_image"]
    sim = preds @ gallery.t()
    return retrieval_from_similarity(sim)


@torch.inference_mode()
def generate_from_eeg(encoder, cond, components, x, device, dtype,
                      steps: int = 20, guidance: float = 0.0,
                      seed: int = 0) -> np.ndarray:
    """One-shot SDXL generation conditioned on EEG via IP tokens. Returns uint8 HxWx3."""
    from diffusers import DDIMScheduler

    unet = components["unet"]
    vae = components["vae"]
    scheduler = DDIMScheduler.from_config(components["scheduler"].config)
    scheduler.set_timesteps(steps, device=device)
    b = x.shape[0]
    generator = torch.Generator(device=device).manual_seed(seed)

    if encoder.cfg.pretrained_subjects == 1:
        z = encoder(x, torch.zeros(b, dtype=torch.long, device=device),
                    adapter_mode="subject")["z_inv"]
    else:
        z = encoder(x, None, adapter_mode="mean")["z_inv"]
    _, ip_tokens = cond(z)
    prompt, pooled = encode_prompts(components, [""] * b, device, dtype)
    encoder_hs = inject_ip_tokens(prompt, ip_tokens)
    add_time = sdxl_time_ids(b, 512, device, dtype)

    latents = torch.randn(
        (b, 4, 64, 64), generator=generator, device=device, dtype=dtype,
    )
    latents = latents * scheduler.init_noise_sigma
    for t in scheduler.timesteps:
        latents_in = scheduler.scale_model_input(latents, t)
        noise_pred = unet(
            latents_in, t, encoder_hs,
            added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time},
        ).sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Decode
    latents = latents / components["vae_scale"]
    imgs = vae.decode(latents.to(vae.dtype)).sample
    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    arr = (imgs.permute(0, 2, 3, 1).float().cpu().numpy() * 255).round().astype(np.uint8)
    return arr


def load_diffusion(ckpt: Path, device: str, dtype: torch.dtype):
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    ccfg = ConditionConfig(**{
        k: v for k, v in payload["condition_cfg"].items()
        if k in ConditionConfig.__dataclass_fields__
    })
    cond = EEGImageProj(ccfg).to(device)
    cond.load_state_dict(payload["condition"], strict=True)
    cond.eval()
    components = build_sdxl_components(device, dtype)
    if ccfg.use_lora and "lora" in payload:
        from loso.models.condition import attach_lora
        components["unet"] = attach_lora(
            components["unet"], ccfg.lora_rank, ccfg.lora_alpha,
        )
        missing = components["unet"].load_state_dict(payload["lora"], strict=False)
        print(f"[lora] restored ({len(payload['lora'])} tensors)", flush=True)
    components["unet"].eval()
    return cond, components, payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-subject", default="sub-08")
    ap.add_argument("--align-ckpt", required=True)
    ap.add_argument("--diffusion-ckpt", default="",
                    help="optional; without it only retrieval is reported")
    ap.add_argument("--k-shots", type=int, nargs="+", default=[0, 5, 10, 25, 50])
    ap.add_argument("--calib-steps", type=int, default=200)
    ap.add_argument("--gen-steps", type=int, default=20)
    ap.add_argument("--max-gen", type=int, default=50,
                    help="cap generated test images (0 = all 200)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    paths.ensure_dirs()
    device = args.device
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    encoder, heads, _ = load_align_checkpoint(Path(args.align_ckpt), device)
    train_subjects, test_subject = things.loso_split(args.test_subject)
    per_subject, global_stats = eeg_mod.build_normalizers(
        train_subjects, "train_subjects",
        cache_path=paths.DATA_ROOT / f"norm_train_subjects_{len(train_subjects)}.json",
    )
    stats = global_stats
    train_store = TargetStore("train", names=("clip_image",))
    test_store = TargetStore("test", names=("clip_image", "dino"))
    pixels = np.load(paths.TARGET_DIR / "pixels_test.npy", mmap_mode="r")

    cond = components = None
    if args.diffusion_ckpt:
        cond, components, _ = load_diffusion(Path(args.diffusion_ckpt), device, dtype)

    out_dir = paths.EVAL_DIR / args.test_subject
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"test_subject": args.test_subject, "align_ckpt": args.align_ckpt,
               "diffusion_ckpt": args.diffusion_ckpt, "runs": []}

    lpips_m = None
    for k in args.k_shots:
        print(f"\n=== k={k} ===", flush=True)
        # Reload a fresh encoder each k so calibrations don't accumulate.
        encoder, heads, _ = load_align_checkpoint(Path(args.align_ckpt), device)
        calib_info = None
        if k > 0:
            calib_info = calibrate_adapter(
                encoder, train_store, test_subject, stats, k,
                steps=args.calib_steps, seed=args.seed, device=device,
            )

        ret = evaluate_retrieval(encoder, heads, test_subject, stats, test_store, device)
        print(f"  retrieval top1={ret.top1:.3f} top5={ret.top5:.3f} "
              f"2way={ret.two_way:.3f} 40way={ret.forty_way:.3f}", flush=True)
        run = {"k": k, "calib": calib_info, "retrieval": ret.__dict__}

        if cond is not None and components is not None:
            if lpips_m is None:
                lpips_m = LPIPSMetric(device)
            n_gen = paths.N_TEST_CONCEPTS if not args.max_gen else min(
                args.max_gen, paths.N_TEST_CONCEPTS)
            # Average all 80 test reps per concept for generation SNR.
            ds = eeg_mod.TestEEGDataset(test_subject, stats, avg_trials=True,
                                        avg_reps=80)
            # One sample per concept.
            gens, refs = [], []
            gen_dir = out_dir / f"gen_k{k}"
            gen_dir.mkdir(exist_ok=True)
            for i in range(n_gen):
                sample = ds[i]  # concept i when avg_reps=80 covers the whole set
                x = sample.x.unsqueeze(0).to(device)
                img = generate_from_eeg(
                    encoder, cond, components, x, device, dtype,
                    steps=args.gen_steps, seed=args.seed + i,
                )[0]
                ref = np.asarray(pixels[i])
                Image.fromarray(img).save(gen_dir / f"{i:03d}.png")
                gens.append(img)
                refs.append(ref)
            scores = {
                "pixcorr": float(np.mean([pixcorr(g, r) for g, r in zip(gens, refs)])),
                "ssim": float(np.mean([ssim(g, r) for g, r in zip(gens, refs)])),
                "psnr": float(np.mean([psnr(g, r) for g, r in zip(gens, refs)])),
                "lpips": float(np.mean([lpips_m(g, r) for g, r in zip(gens, refs)])),
                "n": n_gen,
            }
            print(f"  recon pixcorr={scores['pixcorr']:.3f} ssim={scores['ssim']:.3f} "
                  f"psnr={scores['psnr']:.2f} lpips={scores['lpips']:.3f}", flush=True)
            run["reconstruction"] = scores

        results["runs"].append(run)
        (out_dir / "results.json").write_text(json.dumps(results, indent=1))

    print(f"\n[OK] evaluation -> {out_dir / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
