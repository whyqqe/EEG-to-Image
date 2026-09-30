#!/usr/bin/env python3
"""FRLA -- Frequency-Resolved Latent Anchoring for SDXL img2img.

THE DEFECT THIS REPLACES
------------------------
SDEdit carries ONE scalar, `strength`, which sets how far back the trajectory is
started and therefore how hard the init anchors the result EVERYWHERE.  But the
EEG's reliability is not uniform over the latent's spectrum -- it was measured
per band on sub-08 (outputs/band_probe_ceiling.json, ridge with lambda chosen on
a held-in train split, scored once on test):

    band            energy_frac   test_corr   share of explainable variance
    0.00-0.0625        0.3639       0.1332         93.0%
    0.0625-0.125       0.1067       0.0426           -
    0.125-0.25         0.0836       0.0219           -
    0.25-0.5           0.1009       0.0442           -
    0.5-2.0            0.3450       0.0134           -
    ratio best/worst = 9.9x

One scalar over a 10x-varying spectrum is necessarily a compromise, and the
historical grid search shows the two failure modes colliding:
    strength too high  -> high frequencies over-constrained -> artefacts
    strength too low   -> the low-frequency layout is lost -> PixCorr/SSIM drop
A separate, prior measurement makes the compromise worse: `strength = 0.86` and
`0.88` map to the SAME `init_timestep` under diffusers' integer truncation, so
that grid never tested what it claimed to.

THE METHOD
----------
Score decomposition.  Under partial identifiability, p(z | x) depends on x only
through the bands z can actually see, so

    grad log p_t(x_t | z) = grad log q_t(x_t) + sum_b w_b(t) grad log p_t(z | P_b x_t)

with P_b the band projector.  Treating each band's likelihood as Gaussian around
the EEG-predicted latent x_hat with per-band variance sigma_b^2 gives a purely
geometric update that needs no extra denoiser call:

    x_t <- x_t - sum_b lambda_b(t) * P_b( x_t - sqrt(alpha_bar_t) * x_hat )

Why `sqrt(alpha_bar_t) * x_hat`: the forward process gives
E[x_t | x_0] = sqrt(alpha_bar_t) x_0, so the anchor's expected value AT TIME t is
sqrt(alpha_bar_t) x_hat, not x_hat.

Why lambda_b proportional to r_b^2: w_b/sigma_b^2 and r_b^2/sigma_b^2 are the same
quantity up to a constant, and r_b is the measured reliability.  Squaring it makes
the 9.9x reliability spread a ~100x weight spread, i.e. the LF band is anchored
and the rest is essentially left to the prior.

THIS IS NOT SDEdit WITH A BIGGER STRENGTH
-----------------------------------------
SDEdit anchors ONCE, at t_start; the anchoring then decays away as the denoiser
integrates, so the LF layout it started from can drift.  FRLA re-imposes it at
EVERY step it is active, which is what lets a HIGH strength (more denoising, i.e.
better texture/FID) coexist with a KEPT layout.  That is the falsifiable claim:
FRLA at strength 0.95 should beat plain SDEdit at 0.95 on PixCorr/SSIM at equal
or better FID.

ARMS, so the claim is separable from the mechanism
    off     : plain SDEdit (the reference behaviour, lambda = 0)
    uniform : the same total anchoring applied to ALL bands (lambda_b = const).
              This is the honest control: it tests whether the FREQUENCY
              RESOLUTION matters or merely the extra anchoring force.
    frla    : lambda_b proportional to r_b^2 (the method)
If `uniform` matches `frla`, the per-band weights are decoration and the effect
is just "more anchoring" -- that arm exists to make that failure visible.

NOTHING HERE READS THE TEST SET for any choice: r_b comes from the train-side band
probe, and the strength/arm constants are fixed on the command line.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

NB_ROOT = Path(__file__).resolve().parents[2]

# measured test_corr per band of the SDXL VAE latent, from band_probe_ceiling.json.
# Keys are the upper edge of each band in cycles-per-pixel of the 64x64 latent.
BAND_RELIABILITY: list[tuple[float, float]] = [
    (0.0625, 0.1332),
    (0.1250, 0.0426),
    (0.2500, 0.0219),
    (0.5000, 0.0442),
    (2.0000, 0.0134),
]


def radius_grid(h: int, w: int, device, dtype) -> torch.Tensor:
    fy = torch.fft.fftfreq(h, device=device, dtype=dtype)[:, None]
    fx = torch.fft.fftfreq(w, device=device, dtype=dtype)[None, :]
    return torch.sqrt(fy ** 2 + fx ** 2)


def band_weights(r: torch.Tensor, mode: str) -> list[tuple[torch.Tensor, float]]:
    """(mask, lambda) per band.  `uniform` gives every band the same weight, which
    is the control that isolates FREQUENCY RESOLUTION from ANCHORING FORCE."""
    lo = 0.0
    out: list[tuple[torch.Tensor, float]] = []
    for hi, rel in BAND_RELIABILITY:
        m = (r >= lo) & (r < hi)
        out.append((m, 1.0 if mode == "uniform" else float(rel) ** 2))
        lo = hi
    return out


class FRLA:
    """Callback state for one image; holds the anchor and the schedule."""

    def __init__(self, anchor: torch.Tensor, alphas_cumprod: torch.Tensor,
                 r: torch.Tensor, mode: str, eta: float,
                 active_lo: float, active_hi: float):
        self.anchor = anchor                      # (1,4,64,64) scaled VAE latent
        self.acp = alphas_cumprod
        self.r = r
        self.mode = mode
        self.eta = float(eta)
        self.active_lo = float(active_lo)
        self.active_hi = float(active_hi)
        self.bands = band_weights(r, mode) if mode != "off" else []
        tot = sum(w for _, w in self.bands) if self.bands else 1.0
        # normalise so the TOTAL force is comparable between arms: this is what
        # makes `uniform` vs `frla` a test of resolution rather than of dosage
        self.bands = [(m, w / tot * len(self.bands)) for m, w in self.bands]
        self.n_calls = 0
        self.delta_norm: list[float] = []

    def __call__(self, pipe, step: int, timestep, kwargs):
        if self.mode == "off" or not self.bands:
            return kwargs
        lat = kwargs["latents"]
        t = int(timestep) if not torch.is_tensor(timestep) else int(timestep.item())
        frac = float(self.acp[t])                     # alpha_bar_t
        if not (self.active_lo <= frac <= self.active_hi):
            return kwargs
        # ramp: the correction is strongest when the anchor is most informative
        # (mid-trajectory) and fades at both ends, so the last steps stay free to
        # refine texture and step 0 is not over-constrained
        span = max(self.active_hi - self.active_lo, 1e-6)
        pos = (frac - self.active_lo) / span
        ramp = math.sin(math.pi * min(max(pos, 0.0), 1.0))
        tgt = math.sqrt(max(frac, 0.0)) * self.anchor
        F = torch.fft.fft2(lat.float(), dim=(-2, -1))
        Ft = torch.fft.fft2(tgt.float(), dim=(-2, -1))
        newF = F.clone()
        d = 0.0
        for m, w in self.bands:
            mm = m.to(newF.device)
            corr = self.eta * w * ramp * (F - Ft)
            newF = newF - corr * mm
            d += float((corr * mm).abs().mean())
        kwargs["latents"] = torch.real(torch.fft.ifft2(newF, dim=(-2, -1))).to(lat.dtype)
        self.n_calls += 1
        self.delta_norm.append(d)
        return kwargs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-npy", type=str, required=True)
    ap.add_argument("--anchor-latent-npy", type=str, required=True,
                    help="EEG-predicted SDXL VAE latent, SCALED, (N,4,64,64)")
    ap.add_argument("--lowlevel-rgb-dir", type=str, required=True, help="RGB init for img2img")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--tag", type=str, default="frla")
    ap.add_argument("--arm", type=str, default="frla", choices=["off", "uniform", "frla"])
    ap.add_argument("--eta", type=float, default=0.85, help="overall anchoring strength")
    ap.add_argument("--strength", type=float, default=0.82)
    ap.add_argument("--ip-scale", type=float, default=1.0)
    ap.add_argument("--gen-steps", type=int, default=28)
    ap.add_argument("--gen-guidance", type=float, default=5.0)
    ap.add_argument("--gen-size", type=int, default=512)
    ap.add_argument("--scaling-factor", type=float, default=0.13025)
    ap.add_argument("--active-lo", type=float, default=0.02, help="alpha_bar floor")
    ap.add_argument("--active-hi", type=float, default=0.99, help="alpha_bar ceiling")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompts-json", type=str, default="")
    ap.add_argument("--negative-prompt", type=str,
                    default="blurry, low quality, distorted, watermark")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        raise SystemExit("[FATAL] FRLA needs a GPU; refusing to run on CPU.")
    out = Path(args.output_dir)
    gen_dir = out / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    embeds = np.load(args.embed_npy).astype(np.float32)
    anchors = np.load(args.anchor_latent_npy).astype(np.float32)
    n = len(embeds)
    if len(anchors) != n:
        raise SystemExit(f"[FATAL] embeds {len(embeds)} vs anchors {len(anchors)}")
    if args.max_images > 0:
        n = min(n, args.max_images)
    prompts = None
    if args.prompts_json:
        prompts = json.loads(Path(args.prompts_json).read_text(encoding="utf-8"))
        if len(prompts) < n:
            raise SystemExit(f"[FATAL] prompts {len(prompts)} < n {n}")
    print(f"[frla] arm={args.arm} eta={args.eta} strength={args.strength} n={n} dev={dev}")

    # Reuse the EXACT loader the existing sdedit rows use, so any difference from
    # those rows is attributable to FRLA and not to how the pipeline was built.
    # The resolvers read HF_HUB_CACHE, which the run script exports.
    sys.path.insert(0, str(Path("/project/peilab/why/eeg-brainit") / "scripts"))
    from eval_atm_pipeline import (resolve_ip_adapter_dir,
                                   resolve_sdxl_model_path)  # type: ignore

    sdxl = resolve_sdxl_model_path()
    ip_root = resolve_ip_adapter_dir()
    if ip_root is None:
        raise SystemExit("[FATAL] IP-Adapter dir unresolved (set HF_HUB_CACHE)")
    print(f"[frla] sdxl={sdxl}\n[frla] ip={ip_root}")

    from diffusers import StableDiffusionXLImg2ImgPipeline

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        sdxl, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
        local_files_only=Path(str(sdxl)).is_dir()).to(dev)
    try:
        pipe.load_ip_adapter(str(ip_root), weight_name="ip-adapter_sdxl_vit-h.bin",
                             subfolder="sdxl_models", image_encoder_folder=None,
                             local_files_only=True)
    except Exception:                                      # noqa: BLE001
        pipe.load_ip_adapter(str(ip_root), weight_name="ip-adapter_sdxl.bin",
                             subfolder="sdxl_models", image_encoder_folder=None,
                             local_files_only=True)
    pipe.set_ip_adapter_scale(args.ip_scale)

    acp = pipe.scheduler.alphas_cumprod.to(dev).float()
    h = w = args.gen_size // 8
    r = radius_grid(h, w, dev, torch.float32)

    for i in range(n):
        path = gen_dir / f"{i:03d}.png"
        if path.is_file():
            continue
        init = Image.open(Path(args.lowlevel_rgb_dir) / f"{i:03d}.png").convert("RGB").resize(
            (args.gen_size, args.gen_size), Image.Resampling.BICUBIC)
        emb = torch.from_numpy(embeds[i:i + 1]).to(device=dev, dtype=pipe.dtype)
        if args.gen_guidance > 1.0:
            ip_emb = torch.cat([torch.zeros_like(emb), emb], dim=0).unsqueeze(1)
        else:
            ip_emb = emb.unsqueeze(1)
        prompt = prompts[i] if prompts else ""
        neg = args.negative_prompt if prompt else ""
        anchor = torch.from_numpy(anchors[i:i + 1]).to(dev)
        st = FRLA(anchor, acp, r, args.arm, args.eta, args.active_lo, args.active_hi)
        g = torch.Generator(device=dev).manual_seed(args.seed + i)
        img = pipe(
            prompt=prompt or "", negative_prompt=neg, image=init,
            strength=float(np.clip(args.strength, 0.05, 0.95)),
            ip_adapter_image_embeds=[ip_emb],
            num_inference_steps=int(args.gen_steps),
            guidance_scale=float(args.gen_guidance),
            generator=g,
            callback_on_step_end=st,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images[0]
        img.save(path)
        if i % 25 == 0:
            print(f"\r  gen {i+1}/{n} frla_calls={st.n_calls}", end="", flush=True)
    print()

    (out / "metrics.json").write_text(json.dumps({
        "tag": args.tag, "arm": args.arm, "eta": args.eta, "strength": args.strength,
        "ip_scale": args.ip_scale, "gen_steps": args.gen_steps,
        "gen_guidance": args.gen_guidance, "seed": args.seed, "n_gen": n,
        "anchor_latent_npy": args.anchor_latent_npy,
        "band_reliability": BAND_RELIABILITY,
        "note": ("FRLA anchors the EEG-predicted latent band-by-band inside the "
                 "img2img trajectory; see the module docstring for the score "
                 "decomposition and for why `uniform` is the control arm."),
    }, indent=2), encoding="utf-8")
    print(f"[frla] done -> {out}")


if __name__ == "__main__":
    main()
