"""Decisive test: can the LAYOUT channel verify a GENERATED image, non-circularly?

Why this is non-circular
------------------------
The generated images in `g3f_ll_gen` were produced from the EEG *semantic*
condition (ip_fused) + prompt + low-level RGB init.  They were never shown to the
EEG *layout* head.  The layout head predicts the low-frequency VAE latent from
z_decode_vith, a different readout with a different target (and it was trained on
real images, not on generations).

So the quantity
    s(i, g) = cos( LF(EEG_i predicted latent), LF(VAE(generated image g)) )
is an honest, zero-leakage verification score: "does this generated image look
like what this EEG trial says the low-frequency content should be?"

What we measure
---------------
1. 2-way: does s(i, gen_i) beat s(i, gen_j) for a random j?
2. top-1 over the 200 generations.
3. Score fusion with the semantic channel: does adding the layout score improve
   identification of one's own generation?  (This is the generation-side analogue
   of the ranking-side fusion gain that was measured at +8.5pp.)
4. Same with the DC (mean colour) component removed, to check the verifier is not
   just a colour detector.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def resolve_vae(hub: Path):
    from diffusers import AutoencoderKL

    sdxl_root = hub / "models--stabilityai--stable-diffusion-xl-base-1.0" / "snapshots"
    if sdxl_root.is_dir():
        for snap in sorted(sdxl_root.iterdir(), reverse=True):
            vae_dir = snap / "vae"
            if (vae_dir / "config.json").is_file():
                return AutoencoderKL.from_pretrained(str(vae_dir), torch_dtype=torch.float32)
    return AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae", torch_dtype=torch.float32
    )


def l2n(x: np.ndarray) -> np.ndarray:
    x = x.reshape(len(x), -1).astype(np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def radius_grid(h: int, w: int) -> np.ndarray:
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    return np.sqrt(fy ** 2 + fx ** 2)


def lowpass(x: np.ndarray, r: np.ndarray, cut: float) -> np.ndarray:
    f = np.fft.fft2(x.astype(np.float32), axes=(-2, -1))
    return np.real(np.fft.ifft2(f * (r < cut).astype(np.float32), axes=(-2, -1))).astype(np.float32)


def lf_feat(x: np.ndarray, cut: float) -> np.ndarray:
    r = radius_grid(x.shape[-2], x.shape[-1])
    return np.stack([lowpass(x[i], r, cut) for i in range(len(x))]).reshape(len(x), -1)


def zs(s: np.ndarray) -> np.ndarray:
    return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-8)


def metrics(s: np.ndarray, n: int, idx: np.ndarray) -> dict:
    r = np.random.default_rng(0).permutation(n)
    ok = idx != r
    return {
        "top1": float(np.mean(s.argmax(1) == idx)),
        "twoway": float(np.mean(s[idx, idx][ok] > s[idx, r][ok])),
        "mean_rank": float(np.mean([(s[i] > s[i, i]).sum() for i in range(n)])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", type=str, required=True, help=".../gen/sub-08/g3f_ll_gen/generated")
    ap.add_argument("--pred-latent", type=str, required=True)
    ap.add_argument("--cond-npy", type=str, required=True, help="semantic condition used for generation")
    ap.add_argument("--clip-feats", type=str, default="", help="cache of CLIP feats of the same generated images")
    ap.add_argument("--cache", type=str, default="", help="where to cache encoded latents")
    ap.add_argument("--cut", type=float, default=0.0625)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--out-json", type=str, required=True)
    args = ap.parse_args()

    paths = sorted(Path(args.gen_dir).glob("*.png"))
    if not paths:
        raise SystemExit(f"no png under {args.gen_dir}")
    n = len(paths)
    idx = np.arange(n)
    print(f"[layout-verifier] {n} generated images from {args.gen_dir}")

    cache = Path(args.cache) if args.cache else Path(args.out_json).with_suffix(".lat_f16.npy")
    if cache.is_file() and cache.stat().st_size > 1000:
        lat = np.load(cache, mmap_mode="r")
        if lat.shape[0] == n:
            print(f"[SKIP] cached latents {lat.shape}")
        else:
            lat = None
    else:
        lat = None

    if lat is None:
        hub = Path(os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        vae = resolve_vae(hub).to(device).eval()
        tfm = transforms.Compose(
            [
                transforms.Resize((args.image_size, args.image_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        scaling = float(getattr(vae.config, "scaling_factor", 0.13025))
        h = w = args.image_size // 8
        buf = np.zeros((n, 4, h, w), dtype=np.float16)
        for start in range(0, n, args.batch_size):
            bp = paths[start : start + args.batch_size]
            imgs = torch.stack([tfm(Image.open(p).convert("RGB")) for p in bp]).to(
                device=device, dtype=torch.float32
            )
            with torch.no_grad():
                out = vae.encode(imgs).latent_dist.mean * scaling
            buf[start : start + len(bp)] = out.detach().cpu().numpy().astype(np.float16)
            print(f"\r  encode {start + len(bp)}/{n}", end="", flush=True)
        print()
        np.save(cache, buf)
        lat = buf
        print(f"[save] {cache}")

    P = np.load(args.pred_latent).astype(np.float32)
    G = np.asarray(lat, dtype=np.float32)
    assert P.shape == G.shape, (P.shape, G.shape)

    C = l2n(np.load(args.cond_npy))
    clip_cache = Path(args.clip_feats) if args.clip_feats else Path(args.out_json).with_suffix(".clip.npy")
    if clip_cache.is_file() and clip_cache.stat().st_size > 1000:
        CLIP = l2n(np.load(clip_cache))
        print(f"[SKIP] cached CLIP feats {CLIP.shape}")
    else:
        import open_clip

        dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained="laion2b_s32b_b79k", device=dev
        )
        model = model.to(dev).eval()
        feats = []
        with torch.no_grad():
            for start in range(0, n, 16):
                bp = paths[start : start + 16]
                x = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in bp]).to(dev)
                feats.append(model.encode_image(x).float().cpu())
                print(f"\r  clip(gen) {start + len(bp)}/{n}", end="", flush=True)
        print()
        CLIP = l2n(torch.cat(feats, 0).numpy())
        np.save(clip_cache, CLIP)
        print(f"[save] {clip_cache}")
    S = C @ CLIP.T

    res: dict = {"n": n, "cut": args.cut, "gen_dir": args.gen_dir}

    for tag, drop_dc in (("lf_withdc", False), ("lf_nodc", True)):
        p, g = lf_feat(P, args.cut), lf_feat(G, args.cut)
        if drop_dc:
            p = p - p.mean(1, keepdims=True)
            g = g - g.mean(1, keepdims=True)
        L = l2n(p) @ l2n(g).T
        res[tag] = metrics(L, n, idx)
        res[tag]["cos_pred_gt"] = float(np.mean(np.sum(l2n(p) * l2n(g), 1)))

    # --- fusion: does the layout score add on top of the semantic score?
    p, g = lf_feat(P, args.cut), lf_feat(G, args.cut)
    L = l2n(p) @ l2n(g).T
    res["semantic_own_gen"] = metrics(S, n, idx)
    res["fusion"] = {}
    for lam in (0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0):
        m = metrics(zs(S) + lam * zs(L), n, idx)
        res["fusion"][f"{lam:.1f}"] = m
        print(
            f"  lam={lam:<4} top1={m['top1']:.3f}  2way={m['twoway']:.3f}  mean_rank={m['mean_rank']:.1f}"
        )

    base = res["fusion"]["0.0"]["top1"]
    best = max(res["fusion"].items(), key=lambda kv: kv[1]["top1"])
    res["fusion_best"] = {"lambda": float(best[0]), **best[1]}
    res["fusion_gain_top1"] = float(best[1]["top1"] - base)
    res["fusion_gain_twoway"] = float(best[1]["twoway"] - res["fusion"]["0.0"]["twoway"])

    # --- cross-validated lambda (avoid selecting lambda on the eval set)
    rng = np.random.default_rng(42)
    folds = np.array_split(rng.permutation(n), 5)
    gains, lams = [], []
    for f in folds:
        tr = np.setdiff1d(idx, f)
        best_lam, bv = 0.0, -1.0
        for lam in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8):
            v = float(np.mean((zs(S[np.ix_(tr, tr)]) + lam * zs(L[np.ix_(tr, tr)])).argmax(1) == np.arange(len(tr))))
            if v > bv:
                best_lam, bv = lam, v
        Sf, Lf = S[np.ix_(f, f)], L[np.ix_(f, f)]
        v0 = float(np.mean(Sf.argmax(1) == np.arange(len(f))))
        v1 = float(np.mean((zs(Sf) + best_lam * zs(Lf)).argmax(1) == np.arange(len(f))))
        gains.append(v1 - v0)
        lams.append(best_lam)
    res["cv"] = {
        "folds": 5,
        "lambda_star": [float(x) for x in lams],
        "gain_per_fold": [float(x) for x in gains],
        "gain_mean": float(np.mean(gains)),
        "gain_positive_folds": int(sum(1 for x in gains if x > 0)),
    }

    Path(args.out_json).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "fusion"}, indent=2))
    print(f"[save] {args.out_json}")


if __name__ == "__main__":
    main()
