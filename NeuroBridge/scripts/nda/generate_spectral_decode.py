#!/usr/bin/env python3
"""Spectral Assembly decode.

Keeps HCMA's frozen semantic path (IP-Adapter + prompts) but replaces the
monolithic structural conditioning (`img2img strength` + weak depth ControlNet)
with a PER-BAND latent anchor applied inside the denoising loop:

    for each step, for every latent frequency k:
        z_k  <-  (1 - gamma_k) * z_k  +  gamma_k * (sqrt(alpha_t) * a_k)

    gamma_k = gamma  for k < cut   (layout band: hard-anchored to the EEG anchor)
    gamma_k = 0      for k >= cut  (texture band: left to the diffusion prior)

WHY (measured, see build_spectral_anchor.py)
--------------------------------------------
A scalar `strength` puts the pipeline on a Pareto frontier: raising it improves
semantics and FID but destroys structure (SSIM/PixCorr), consistently over three
independent ControlNet scales. The reason is that `strength` trades away the
whole latent at once, while the EEG anchor is only trustworthy in the low band
(corr 0.5955 at r<0.0625 vs 0.0355 above r>0.5).

Anchoring per band lets us run at strength=1.0 -- so the prior supplies fully
realistic high-frequency texture -- while the low-frequency layout stays pinned
to the EEG evidence. That is the intended break of the trade-off.

`--cut 0` disables anchoring entirely and reproduces plain text2img/img2img, so
the control condition is the same script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

BRAINIT = Path("/project/peilab/why/eeg-brainit")
sys.path.insert(0, str(BRAINIT / "scripts"))
sys.path.insert(0, str(BRAINIT / "src"))

from eval_atm_pipeline import resolve_ip_adapter_dir, resolve_sdxl_model_path  # type: ignore


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)


def resolve_controlnet_path(hub: Path) -> str:
    root = hub / "models--diffusers--controlnet-depth-sdxl-1.0" / "snapshots"
    if root.is_dir():
        for snap in sorted(root.iterdir(), reverse=True):
            if (snap / "config.json").is_file():
                return str(snap)
    return "diffusers/controlnet-depth-sdxl-1.0"


def radial_mask(H: int, W: int, cut: float, device: torch.device, dtype: torch.dtype):
    """Boolean low-pass mask in rfft layout (H, W//2+1)."""
    fy = torch.fft.fftfreq(H, device=device)[:, None]
    fx = torch.fft.rfftfreq(W, device=device)[None, :]
    r = torch.sqrt(fy * fy + fx * fx) / 0.5
    return (r < cut).to(dtype)


def make_callback(anchor, cut, gamma, start_step, stats=None):
    """anchor: (1,C,H,W) clean VAE latent in pipeline space (already * scaling_factor).

    `stats` is a run-level dict so we can verify from the log that anchoring
    ACTUALLY fired. A silently no-op callback is the most dangerous failure mode
    of this design, so we track it and hard-fail on it in main().
    """
    state = {"mask": None, "n_anchor": 0}
    if stats is None:
        stats = {}

    def cb(pipe, step_index, timestep, kwargs):
        lat = kwargs["latents"]
        if cut <= 0.0 or step_index < start_step or gamma <= 0.0:
            return kwargs
        if state["mask"] is None:
            state["mask"] = radial_mask(lat.shape[-2], lat.shape[-1], cut,
                                        lat.device, torch.float32)
        # z_t after this step corresponds to the NEXT scheduled timestep
        ts = pipe.scheduler.timesteps
        j = min(step_index + 1, len(ts) - 1)
        t_next = ts[j].reshape(1).to(lat.device)
        a = anchor.to(device=lat.device, dtype=torch.float32)
        # Scheduler-agnostic clean-content level: add_noise(a, 0, t) == c_t * a,
        # with c_t = sqrt(alpha_bar) for VP schedulers (DDIM/DDPM) and c_t = 1 for
        # EDM-style ones (Euler, which is SDXL's default). Composing with ZERO noise
        # in the anchored band is correct under BOTH conventions, and t -> 0 makes
        # the band exactly equal to the anchor.
        a_t = pipe.scheduler.add_noise(a, torch.zeros_like(a), t_next)
        if stats.get("c_scale") is None:
            stats["c_scale"] = float(a_t.abs().mean() / max(float(a.abs().mean()), 1e-8))
            print(f"[ANCHOR] scheduler={type(pipe.scheduler).__name__} "
                  f"clean-scale c_t at first anchored step = {stats['c_scale']:.4f}", flush=True)
        L = torch.fft.rfft2(lat.float())
        F = torch.fft.rfft2(a_t)
        m = state["mask"].unsqueeze(0).unsqueeze(0)
        # gamma=1 -> hard replace of the anchored band; 0<gamma<1 -> soft blend
        L = L * (1.0 - gamma * m) + (gamma * m) * F
        kwargs["latents"] = torch.fft.irfft2(L, s=(lat.shape[-2], lat.shape[-1])).to(lat.dtype)
        state["n_anchor"] += 1
        stats["steps"] = stats.get("steps", 0) + 1
        return kwargs

    cb.n_anchored_steps = lambda: state["n_anchor"]
    return cb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-npy", required=True, help="IP-Adapter embeddings (semantic path, frozen)")
    ap.add_argument("--prompts-json", required=True)
    ap.add_argument("--anchor-latent-npy", default="",
                    help="spectral structural anchor (N,4,64,64), unscaled VAE latent")
    ap.add_argument("--anchor-std-target", type=float, default=0.0,
                    help="if >0, rescale the anchor so its r<cut band has this std, in "
                         "unscaled-latent units. Set it on BOTH sides whenever two anchors "
                         "from different regressors are compared, otherwise the comparison "
                         "measures gain rather than spatial accuracy. 0 disables.")
    ap.add_argument("--depth-rgb-dir", default="", help="optional depth ControlNet condition")
    ap.add_argument("--lowlevel-rgb-dir", default="", help="optional img2img init (needed if strength<1)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tag", default="sa")
    ap.add_argument("--cut", type=float, default=0.0625, help="low-band cutoff; 0 disables anchoring")
    ap.add_argument("--gamma", type=float, default=1.0, help="anchoring weight inside the low band")
    ap.add_argument("--start-step", type=int, default=0, help="first denoising step that anchors")
    ap.add_argument("--cn-scale", type=float, default=0.0)
    ap.add_argument("--ip-scale", type=float, default=1.0)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--gen-steps", type=int, default=28)
    ap.add_argument("--gen-guidance", type=float, default=5.0)
    ap.add_argument("--gen-size", type=int, default=512)
    ap.add_argument("--scaling-factor", type=float, default=0.13025)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--debug-anchor", action="store_true",
                    help="print the scheduler's clean-component scale so the anchoring math can be verified")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--negative-prompt", default="blurry, low quality, distorted, watermark")
    args = ap.parse_args()

    cache = Path(os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
    os.environ.setdefault("HF_HOME", str(cache.parent))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache.parent / "open_clip"))
    os.environ.setdefault("TORCH_HOME", str(cache.parent / "torch"))
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    embeds = l2(np.load(args.embed_npy).astype(np.float32))
    prompts = json.loads(Path(args.prompts_json).read_text(encoding="utf-8"))
    if len(prompts) < len(embeds):
        raise ValueError(f"prompts {len(prompts)} < embeds {len(embeds)}")
    n = len(embeds) if args.max_images <= 0 else min(len(embeds), args.max_images)

    anchor = None
    if args.anchor_latent_npy and args.cut > 0.0:
        a = np.load(args.anchor_latent_npy).astype(np.float32)
        if len(a) < n:
            raise ValueError(f"anchor rows {len(a)} < {n}")
        anchor = torch.from_numpy(a[:n])
        # Optional amplitude equalisation of the anchored band.
        #
        # Only the r < cut band of the anchor is ever used (see make_callback), and
        # that band's amplitude is a property of the regressor, not of its spatial
        # accuracy: regression to the conditional mean systematically under-scales it
        # (measured: our structural head produced 0.1468 against a 0.5127 target,
        # i.e. 29% amplitude, which makes the SDEdit init washed out and depresses
        # SSIM/PixCorr for a reason unrelated to what is being tested). When two
        # anchors come from DIFFERENT regressors, comparing them at their native
        # amplitudes confounds spatial accuracy with trivial gain, so the equalisation
        # is applied to both sides at the same target value.
        #
        # Uses the same radial_mask() as the callback, so the band that is measured is
        # exactly the band that is anchored. Applied before the *scaling_factor
        # multiplication, so `--anchor-std-target` is in unscaled-latent units.
        if args.anchor_std_target > 0:
            a32 = anchor.to(torch.float32)
            hh, ww = a32.shape[-2], a32.shape[-1]
            m = radial_mask(hh, ww, args.cut, a32.device, a32.dtype)
            # radial_mask is in rfft layout (H, W//2+1), so rfft2/irfft2 must be used;
            # pairing it with fft2/ifft2 raises a shape mismatch.
            lf = torch.fft.irfft2(torch.fft.rfft2(a32) * m, s=(hh, ww))
            s = float(lf.std())
            if not np.isfinite(s) or s <= 1e-8:
                raise ValueError(f"anchor low band has no usable amplitude (std={s})")
            g = args.anchor_std_target / s
            anchor = anchor * g
            print(f"[anchor] amplitude equalised x{g:.4f} "
                  f"(band std {s:.4f} -> {args.anchor_std_target:.4f})", flush=True)
        anchor = anchor * args.scaling_factor
    # run-level anchoring telemetry; a silently no-op callback must not pass
    stats: dict = {"steps": 0, "c_scale": None, "requested": bool(anchor is not None)}

    out = Path(args.output_dir)
    gen_dir = out / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = Path(args.depth_rgb_dir) if args.depth_rgb_dir else None
    ll_dir = Path(args.lowlevel_rgb_dir) if args.lowlevel_rgb_dir else None

    did_generate = False
    if all((gen_dir / f"{i:03d}.png").is_file() for i in range(n)):
        print(f"[SKIP] {n} images already present")
    else:
        did_generate = True
        from diffusers import (StableDiffusionXLControlNetImg2ImgPipeline,
                               StableDiffusionXLImg2ImgPipeline)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        sdxl = resolve_sdxl_model_path(cache)
        use_cn = depth_dir is not None and args.cn_scale > 0.0
        print(f"[INFO] sdxl={sdxl}")
        print(f"[INFO] cut={args.cut} gamma={args.gamma} start_step={args.start_step} "
              f"strength={args.strength} cn={args.cn_scale if use_cn else 0.0} "
              f"anchoring={'ON' if anchor is not None else 'OFF'}")

        if use_cn:
            from diffusers import ControlNetModel
            cn_path = resolve_controlnet_path(cache)
            print(f"[INFO] controlnet={cn_path}")
            controlnet = ControlNetModel.from_pretrained(
                cn_path, torch_dtype=dtype, local_files_only=Path(str(cn_path)).is_dir())
            pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                sdxl, controlnet=controlnet, torch_dtype=dtype,
                variant="fp16" if device.type == "cuda" else None,
                use_safetensors=True, local_files_only=Path(str(sdxl)).is_dir()).to(device)
        else:
            controlnet = None
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                sdxl, torch_dtype=dtype,
                variant="fp16" if device.type == "cuda" else None,
                use_safetensors=True, local_files_only=Path(str(sdxl)).is_dir()).to(device)

        ip_root = resolve_ip_adapter_dir(cache)
        kw = {"subfolder": "sdxl_models", "image_encoder_folder": None, "local_files_only": True}
        try:
            pipe.load_ip_adapter(str(ip_root), weight_name="ip-adapter_sdxl_vit-h.bin", **kw)
        except Exception:
            pipe.load_ip_adapter(str(ip_root), weight_name="ip-adapter_sdxl.bin", **kw)
        pipe.set_ip_adapter_scale(float(args.ip_scale))

        if args.debug_anchor:
            ts = pipe.scheduler.timesteps
            probe = torch.ones((1, 4, 8, 8), device=device, dtype=torch.float32)
            cs = [float(pipe.scheduler.add_noise(probe, torch.zeros_like(probe),
                                                 ts[min(k + 1, len(ts) - 1)].reshape(1).to(device))[0, 0, 0, 0])
                  for k in range(min(5, len(ts)))]
            print(f"[DEBUG] scheduler={type(pipe.scheduler).__name__} "
                  f"clean-scale c_t at first steps={['%.4f' % c for c in cs]} "
                  f"(expect ~0 at max noise for VP schedulers)")


        # one shared init RGB is required by the img2img signature; with strength=1.0
        # it is unused (latents start from pure noise), which is exactly the point.
        init_default = None
        if ll_dir is not None:
            init_default = ll_dir / "000.png"
        if init_default is None or not init_default.is_file():
            blank = gen_dir / "_blank_init.png"
            Image.new("RGB", (args.gen_size, args.gen_size), (128, 128, 128)).save(blank)
            init_default = blank

        g = torch.Generator(device=device).manual_seed(args.seed)
        for i in tqdm(range(n), desc=f"sa[{args.tag}]"):
            path = gen_dir / f"{i:03d}.png"
            if path.is_file():
                continue
            emb = torch.from_numpy(embeds[i:i + 1]).to(device=device, dtype=pipe.dtype)
            ip_emb = (torch.cat([torch.zeros_like(emb), emb], dim=0).unsqueeze(1)
                      if args.gen_guidance > 1.0 else emb.unsqueeze(1))
            prompt = prompts[i] if prompts[i] else ""
            neg = args.negative_prompt if prompt else ""
            ip = Image.open(ll_dir / f"{i:03d}.png" if ll_dir else init_default).convert("RGB")
            ip = ip.resize((args.gen_size, args.gen_size), Image.Resampling.BICUBIC)
            call = dict(prompt=prompt, negative_prompt=neg, image=ip,
                        strength=float(args.strength), ip_adapter_image_embeds=[ip_emb],
                        num_inference_steps=int(args.gen_steps),
                        guidance_scale=float(args.gen_guidance), generator=g)
            if anchor is not None and args.cut > 0.0:
                # per-sample anchor; rebuilt each row so no batch-alignment risk
                call["callback_on_step_end"] = make_callback(
                    anchor[i:i + 1], args.cut, args.gamma, args.start_step, stats)
                call["callback_on_step_end_tensor_inputs"] = ["latents"]
            if use_cn:
                call["control_image"] = Image.open(depth_dir / f"{i:03d}.png").convert("RGB").resize(
                    (args.gen_size, args.gen_size), Image.Resampling.BICUBIC)
                call["controlnet_conditioning_scale"] = float(args.cn_scale)
            img = pipe(**call).images[0]
            img.save(path)

        del pipe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    n_expected = n * max(0, args.gen_steps - args.start_step)
    anchored = int(stats.get("steps", 0))
    print(f"[ANCHOR] steps anchored={anchored} (expected ~{n_expected} for n={n}) "
          f"c_scale={stats.get('c_scale')}", flush=True)
    if stats["requested"] and did_generate and anchored == 0:
        # the callback was requested but never fired -> the whole method silently
        # degraded to plain generation. Fail loudly rather than ship a fake result.
        raise RuntimeError(
            "anchoring was requested but the step callback never fired; "
            "the run would silently be plain generation")

    report = {
        "tag": args.tag,
        "method": "Spectral Assembly (per-band latent anchoring)",
        "cut": args.cut, "gamma": args.gamma, "start_step": args.start_step,
        "strength": args.strength, "cn_scale": args.cn_scale, "ip_scale": args.ip_scale,
        "anchoring": bool(anchor is not None and args.cut > 0.0),
        "anchor_steps": anchored, "anchor_c_scale": stats.get("c_scale"),
        "anchor_steps_expected": n_expected if did_generate else 0,
        "anchor_latent_npy": args.anchor_latent_npy,
        "depth_rgb_dir": args.depth_rgb_dir, "lowlevel_rgb_dir": args.lowlevel_rgb_dir,
        "embed_npy": args.embed_npy, "prompts_json": args.prompts_json,
        "gen_steps": args.gen_steps, "gen_guidance": args.gen_guidance, "n_gen": n,
        "note": ("LF band hard-anchored to EEG; HF band generated by the prior. "
                 "cut=0 is the plain-generation control."),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
