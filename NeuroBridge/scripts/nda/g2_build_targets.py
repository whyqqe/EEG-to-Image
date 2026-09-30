#!/usr/bin/env python3
"""G2 stage 1: build granularity-factorised targets for the parallel dual-tower model.

Target layout
-------------
SEMANTIC tower (5 targets, all in CLIP ViT-H-14 space so they share one geometry):
    sem_image        image-level encoding            (CLIP image branch)
    sem_overall      description of the whole scene  (VLM text -> CLIP text)
    sem_subject      description of the main object
    sem_background   description of the background
    sem_detail       description of fine detail / texture

PERCEPTUAL tower (two granularities):
    perc_struct      VAE latent restricted to r < cut     (structure)
    perc_texture     spectral statistics of the VAE latent (texture)
    perc_depth       cached depth, when available          (extra structure)

Why the perceptual targets take these exact forms
-------------------------------------------------
Measured per-band ceiling (ridge; lambda chosen on a held-in split, test scored
once -- see `band_ceiling_probe.py`, artifact `outputs/band_probe_ceiling.json`):

    band            E_frac   test_corr   var_expl    share
    r<0.0625        0.364     0.133      0.00645     92.9%
    0.0625-0.125    0.107     0.043      0.00019      2.8%
    0.125-0.25      0.084     0.022      0.00004      0.6%
    0.25-0.5        0.101     0.044      0.00020      2.8%
    r>0.5           0.345     0.013      0.00006      0.9%

  1. STRUCTURE is anchored in the COARSE band because 92.9% of the recoverable
     latent variance lives there. The shipped VAE head regresses the FULL latent
     with L1, so most of its capacity is spent on bands it cannot predict and is
     pushed to the conditional mean there. That -- not a head bug -- is the real
     source of the observed collapse.

  2. TEXTURE is supervised on LOW-DIMENSIONAL SPECTRAL STATISTICS rather than on
     the fine coefficients themselves. Those coefficients are 26424-dimensional
     and ~99% noise for EEG (corr 0.043 in 0.0625-0.125, 0.013 above Nyquist);
     regressing them is asking for the conditional mean. The fine band's energy
     SHAPE is the well-posed part.

The texture target retains an ABSOLUTE log-power term. Normalising by each
sample's own total power makes the target scale-invariant and lets the head
collapse amplitude to zero (measured on the CPU harness during the T3
iteration), so the energy term is kept.

Redundancy is measured, not assumed
-----------------------------------
The report carries the mean pairwise cosine between the four text targets. If
the VLM emits four near-identical sentences the granularity axis is vacuous, and
that has to be visible in the artifact instead of being claimed away. The report
also compares caption embeddings against the old concept-name template
embedding, so the gain over the oracle-style target is quantified.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENCLIP_CACHE_DIR", "/project/peilab/why/cache/eeg-brainit/open_clip")

FIELDS = ("overall", "subject", "background", "detail")


# ------------------------------------------------------------------ helpers

def l2(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-8)


def list_split_images(images_root: Path, split: str) -> list[Path]:
    """Same ordering as build_gt_depth_cache and the VAE caches."""
    root = images_root / ("training_images" if split == "train" else "test_images")
    paths: list[Path] = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.JPEG")) + list(d.glob("*.png")))
        paths.extend(imgs)
    return paths


def radial_grid(h: int, w: int) -> np.ndarray:
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    return np.sqrt(fy ** 2 + fx ** 2) / 0.5


def low_band(x: np.ndarray, r: np.ndarray, cut: float) -> np.ndarray:
    """Keep only r < cut. Parseval-exact and dtype preserving.

    numpy's FFT promotes to complex128, so the result is cast back explicitly;
    otherwise every downstream torch tensor silently becomes double.
    """
    F = np.fft.fft2(x.astype(np.float32), axes=(-2, -1))
    m = (r < cut).astype(np.float32)
    return np.real(np.fft.ifft2(F * m, axes=(-2, -1))).astype(np.float32)


def spectral_stats(x: np.ndarray, r: np.ndarray, n_rad: int, n_ang: int) -> np.ndarray:
    """Spectral summary, (N, C, 1 + n_rad + n_ang), all in log space."""
    n, C, H, W = x.shape
    F = np.fft.fft2(x.astype(np.float32), axes=(-2, -1))
    P = (np.abs(F) ** 2).astype(np.float32)
    amp = np.log(np.maximum(P.sum(axis=(2, 3)), 1e-12))          # absolute energy

    rr = r.ravel()
    fy = np.fft.fftfreq(H)[:, None] * np.ones((1, W))
    fx = np.ones((H, 1)) * np.fft.fftfreq(W)[None, :]
    ang = np.arctan2(fy, fx).ravel()
    Pf = P.reshape(n, C, -1)

    re_ = np.linspace(0.0, 1.0, n_rad + 1)
    rad = np.log(np.maximum(np.stack(
        [Pf[:, :, (rr >= re_[i]) & (rr < re_[i + 1])].mean(axis=-1)
         for i in range(n_rad)], axis=-1), 1e-12))

    ae = np.linspace(-np.pi, np.pi, n_ang + 1)
    keep = rr > 1e-6                                             # DC angle undefined
    angp = np.log(np.maximum(np.stack(
        [Pf[:, :, keep & (ang >= ae[i]) & (ang < ae[i + 1])].mean(axis=-1)
         for i in range(n_ang)], axis=-1), 1e-12))

    return np.concatenate([amp[:, :, None], rad, angp], axis=-1).astype(np.float32)


def load_caption_fields(jsonl: Path) -> tuple[dict[str, dict], int]:
    """path -> fields. Later lines win, so a rerun overwrites a failed row."""
    rows: dict[str, dict] = {}
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            rows[r["path"]] = r
    n_bad = sum(1 for r in rows.values() if "error" in r)
    return rows, n_bad


def fallback_texts(concept: str) -> dict[str, str]:
    """Used only for rows the VLM failed on, so no row is silently dropped."""
    art = "an" if concept[:1].lower() in "aeiou" else "a"
    return {
        "overall": f"a photo of {art} {concept}",
        "subject": f"the main object is {art} {concept}",
        "background": f"the background behind the {concept}",
        "detail": f"close-up fine detail of the {concept}",
    }


def encode_texts(texts: list[str], model, tokenizer, device, batch_size: int) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="clip-text"):
            tok = tokenizer(texts[i : i + batch_size]).to(device)
            feats.append(F.normalize(model.encode_text(tok).float(), dim=-1).cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


def pairwise_cos(A: np.ndarray, B: np.ndarray) -> float:
    return float((l2(A) * l2(B)).sum(-1).mean())


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", default="/project/peilab/why/data/images_set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--captions-dir", required=True,
                    help="dir containing captions_train.jsonl / captions_test.jsonl")
    ap.add_argument("--clip-layer-dir", required=True,
                    help="dir with {split}_clip/layer_*.npy or {split}/layer_*.npy (pooled ViT-H-14)")
    ap.add_argument("--clip-layer-subdirs", default="train,test")
    ap.add_argument("--vae-train", required=True)
    ap.add_argument("--vae-test", required=True)
    ap.add_argument("--depth-train", default="")
    ap.add_argument("--depth-test", default="")
    ap.add_argument("--cut", type=float, default=0.0625)
    ap.add_argument("--n-radial", type=int, default=8)
    ap.add_argument("--n-angular", type=int, default=8)
    ap.add_argument("--vae-chunk", type=int, default=512)
    ap.add_argument("--clip-model", default="ViT-H-14")
    ap.add_argument("--clip-pretrained", default="laion2b_s32b_b79k")
    ap.add_argument("--text-batch", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--skip-text", action="store_true",
                    help="only build perceptual targets (debug / partial reruns)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    caps_dir = Path(args.captions_dir)

    vae_paths = {"train": args.vae_train, "test": args.vae_test}
    dep_paths = {"train": args.depth_train, "test": args.depth_test}
    layer_subdirs = [s.strip() for s in args.clip_layer_subdirs.split(",")]

    text_model = text_tok = None
    device = None
    if not args.skip_text:
        import torch
        import open_clip
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        text_model, _, _ = open_clip.create_model_and_transforms(
            args.clip_model, pretrained=args.clip_pretrained, device=device)
        text_model.eval()
        text_tok = open_clip.get_tokenizer(args.clip_model)
        print(f"[g2] CLIP text encoder {args.clip_model}/{args.clip_pretrained} on {device}")

    report: dict = {"cut": args.cut, "n_radial": args.n_radial, "n_angular": args.n_angular,
                    "clip_model": args.clip_model, "splits": {}}

    for split in args.splits:
        paths = list_split_images(Path(args.images_root), split)
        n = len(paths)
        print(f"[g2] split={split} images={n}")

        # ---------------- perceptual: VAE structure + texture ----------------
        V = np.load(vae_paths[split], mmap_mode="r")
        if V.shape[0] != n:
            raise SystemExit(f"{split}: vae rows {V.shape[0]} != images {n}")
        C, H, W = V.shape[1], V.shape[2], V.shape[3]
        r = radial_grid(H, W)
        n_tx = 1 + args.n_radial + args.n_angular
        lf = np.empty((n, C, H, W), dtype=np.float16)
        tx = np.empty((n, C, n_tx), dtype=np.float32)
        for s in range(0, n, args.vae_chunk):
            e = min(s + args.vae_chunk, n)
            blk = np.asarray(V[s:e], dtype=np.float32)
            lf[s:e] = low_band(blk, r, args.cut).astype(np.float16)
            tx[s:e] = spectral_stats(blk, r, args.n_radial, args.n_angular)
        del V
        np.save(out / f"perc_struct_{split}.npy", lf)
        np.save(out / f"perc_texture_{split}.npy", tx)
        lf_energy = float(np.mean(lf.astype(np.float32) ** 2))
        full_energy = None
        print(f"[g2]   perc_struct {lf.shape} perc_texture {tx.shape}")

        if dep_paths[split] and Path(dep_paths[split]).is_file():
            d = np.load(dep_paths[split]).astype(np.float32)
            if d.ndim == 4:
                d = d[:, 0]
            if d.shape[0] != n:
                raise SystemExit(f"{split}: depth rows {d.shape[0]} != images {n}")
            np.save(out / f"perc_depth_{split}.npy", d.astype(np.float16))
            print(f"[g2]   perc_depth {d.shape}")

        # ---------------- semantic: image encoding -------------------------
        sub = layer_subdirs[0] if split == "train" else (
            layer_subdirs[1] if len(layer_subdirs) > 1 else split)
        cand = [Path(args.clip_layer_dir) / sub, Path(args.clip_layer_dir) / split]
        ldir = next((c for c in cand if c.is_dir() and list(c.glob("layer_*.npy"))), None)
        if ldir is None:
            raise SystemExit(f"{split}: no layer_*.npy under {cand}")
        layers = sorted(ldir.glob("layer_*.npy"))
        acc = None
        for lp in layers:
            a = l2(np.load(lp).astype(np.float32))
            if a.shape[0] != n:
                raise SystemExit(f"{lp}: rows {a.shape[0]} != images {n}")
            acc = a if acc is None else acc + a
        sem_image = l2(acc / len(layers))
        np.save(out / f"sem_image_{split}.npy", sem_image.astype(np.float16))
        print(f"[g2]   sem_image {sem_image.shape} from {len(layers)} layers in {ldir}")

        # ---------------- semantic: four description granularities ---------
        sem_stats = {}
        if not args.skip_text:
            cap_jsonl = caps_dir / f"captions_{split}.jsonl"
            if not cap_jsonl.is_file():
                raise SystemExit(f"missing captions: {cap_jsonl}")
            rows, n_bad = load_caption_fields(cap_jsonl)
            miss = 0
            texts: dict[str, list[str]] = {f: [] for f in FIELDS}
            fallback_used = 0
            for p in paths:
                rec = rows.get(str(p))
                if rec is None or "error" in rec:
                    miss += 1
                    t = fallback_texts(p.parent.name)
                    fallback_used += 1
                else:
                    t = {f: str(rec[f]) for f in FIELDS}
                for f in FIELDS:
                    texts[f].append(t[f])
            if fallback_used:
                print(f"[g2]   WARNING {fallback_used}/{n} rows used the concept-name "
                      f"fallback (captions missing or unparseable); n_bad_in_file={n_bad}")

            embs = {}
            for f in FIELDS:
                e = l2(encode_texts(texts[f], text_model, text_tok, device, args.text_batch))
                np.save(out / f"sem_{f}_{split}.npy", e.astype(np.float16))
                embs[f] = e
                print(f"[g2]   sem_{f} {e.shape}")

            # redundancy between granularities -- reported, not assumed
            pairs = {}
            for i, a in enumerate(FIELDS):
                for b in FIELDS[i + 1:]:
                    pairs[f"{a}|{b}"] = round(pairwise_cos(embs[a], embs[b]), 4)
            # gain over the old oracle-style concept-name target
            concept_txt = []
            for p in paths:
                c = p.parent.name.replace("_", " ")
                concept_txt.append(f"a photo of a {c}")
            ce = l2(encode_texts(concept_txt, text_model, text_tok, device, args.text_batch))
            vs_concept = {f: round(pairwise_cos(embs[f], ce), 4) for f in FIELDS}
            sem_stats = {
                "fallback_rows": fallback_used,
                "n_bad_in_file": n_bad,
                "pairwise_cos_between_granularities": pairs,
                "max_pairwise_cos": max(pairs.values()) if pairs else None,
                "cos_vs_concept_name_target": vs_concept,
                "unique_text_per_field": {f: len(set(texts[f])) for f in FIELDS},
            }
            print(f"[g2]   granularity redundancy (max pairwise cos) = "
                  f"{sem_stats['max_pairwise_cos']}")
            print(f"[g2]   unique captions per field = {sem_stats['unique_text_per_field']}")
            np.save(out / f"sem_concept_tmpl_{split}.npy", ce.astype(np.float16))
            del embs

        # VAE full-band energy, for the structure/texture allocation claim
        Vb = np.load(vae_paths[split], mmap_mode="r")
        sub_n = min(1024, n)
        blk = np.asarray(Vb[:sub_n], dtype=np.float32)
        full_energy = float(np.mean(blk ** 2))
        del Vb, blk

        report["splits"][split] = {
            "n": int(n),
            "dim": {
                "sem_image": int(sem_image.shape[1]),
                "sem_overall": int(sem_image.shape[1]) if args.skip_text else None,
                "perc_struct": [int(C), int(H), int(W)],
                "perc_texture": int(n_tx),
            },
            "lf_energy": lf_energy,
            "vae_full_energy": full_energy,
            "lf_energy_share_of_full": lf_energy / max(full_energy, 1e-12),
            "semantic": sem_stats,
        }
        print(f"[g2]   lf energy share of full latent = "
              f"{report['splits'][split]['lf_energy_share_of_full']:.3f}")
        del lf, tx, sem_image

    (out / "g2_targets_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"[g2] wrote {out}/g2_targets_report.json")


if __name__ == "__main__":
    main()
