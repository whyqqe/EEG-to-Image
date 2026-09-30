#!/usr/bin/env python3
"""Standard-protocol 7 metrics + FID for brain-reconstruction on THINGS-EEG.

Replicates the OFFICIAL evaluation pipeline used by the EEG_Image_decode
(ATM, NeurIPS'24), CognitionCapturer (AAAI'25) and META-MEG (Benchetrit'24)
comparison tables -- i.e. the Ozcelik & VanRullen / MindEye metric family.

For each generated dir (200 png, index-aligned to the shared 200 test images):

  PixCorr  : Pearson over flattened RGB pixels, resize 425 BILINEAR
             (official: np.corrcoef per sample)
  SSIM     : resize 425 -> rgb2gray -> skimage structural_similarity
             (gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
              data_range=1.0)
  AlexNet(2) / AlexNet(5): 2-way identification, Pearson corr on
             alexnet.features.4 / alexnet.features.11 activations
  Inception: 2-way identification, Pearson corr on inception_v3 avgpool
  CLIP     : 2-way identification, Pearson corr on OpenCLIP ViT-H/14 final
             (same backbone family as MindEye / META-MEG / CogCap)
  SwAV (low): mean scipy correlation distance (1-pearson) on SwAV-RN50 avgpool
  EffNet-B (low): same distance on efficientnet_b1 avgpool
  FID      : torchmetrics FrechetInceptionDistance(normalize=True) @299
             (identical to erdc_fid_metrics used for all historical FIDs)

NOTE: 2-way uses PEARSON correlation (not cosine) per official ATM notebook.
GT features are computed once and cached; per-model features are cached too.

Usage:
  python eval_standard7.py --manifest manifest.json --images-root PATH \
      --out-dir DIR --device cuda:0 --batch-size 16 [--no-fid] [--overwrite]

manifest.json:
  {"rows": [{"tag": "...", "gen_dir": "..."}, ...],
   "pooled_fid": {"tag": "...", "value": 129.47} | null,   # optional reference
   "n": 200}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------- helpers

def l2_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n < 1e-8] = 1.0
    return x / n


def pearson_sim(real: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix S[i,j] = corr(real_i, pred_j).

    Mathematically identical to the official
      np.corrcoef(vstack([real, pred]))[:n, n:]
    but computed as centered-cosine (fast / low-memory).
    """
    rc = real - real.mean(axis=1, keepdims=True)
    pc = pred - pred.mean(axis=1, keepdims=True)
    rn = np.linalg.norm(rc, axis=1, keepdims=True)
    pn = np.linalg.norm(pc, axis=1, keepdims=True)
    rn[rn < 1e-8] = 1.0
    pn[pn < 1e-8] = 1.0
    rc = rc / rn
    pc = pc / pn
    return rc @ pc.T


def two_way(real: np.ndarray, pred: np.ndarray) -> float:
    """2-way identification percent-correct (chance ~ 50%)."""
    n = real.shape[0]
    sim = pearson_sim(real, pred)  # (n, n)
    hit = 0
    total = 0
    for i in range(n):
        s_ii = sim[i, i]
        for j in range(n):
            if i == j:
                continue
            hit += float(s_ii > sim[i, j])
            total += 1
    return float(hit / max(total, 1))


def mean_corr_dist(real: np.ndarray, pred: np.ndarray) -> float:
    """mean scipy.spatial.distance.correlation = mean(1 - pearson)."""
    vals = []
    for i in range(real.shape[0]):
        a, b = real[i].ravel(), pred[i].ravel()
        if a.std() < 1e-8 or b.std() < 1e-8:
            vals.append(1.0)
        else:
            vals.append(float(1.0 - np.corrcoef(a, b)[0, 1]))
    return float(np.mean(vals))


def list_gen(gen_dir: Path, n: int) -> list[Path]:
    out = []
    for i in range(n):
        p = gen_dir / f"{i:03d}.png"
        if not p.is_file():
            p = gen_dir / f"{i:03d}.jpg"
        if not p.is_file():
            raise FileNotFoundError(f"{gen_dir} missing {i:03d}.png")
        out.append(p)
    return out


def to_tensor_pil(p: Path) -> torch.Tensor:
    img = Image.open(p).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


# ------------------------------------------------------- low-level metrics

def lowlevel(gen_paths: list[Path], gt_paths: list[Path]) -> dict:
    """PixCorr + SSIM with the exact official Ozcelik/ATM protocol.

    The per-image values are returned alongside the means, not only the means. PixCorr
    is the metric that decides the low-level experiments in this project, and it is the
    one metric in the suite with NO per-concept two-way decomposition -- so without
    this list the decisive comparison would be the only one reported as a bare
    difference of two means, with no interval. Paired by position, which is valid
    because `gen_paths[i]` and `gt_paths[i]` are the same concept and every arm is
    generated in the same concept order.
    """
    from skimage.color import rgb2gray
    from skimage.metrics import structural_similarity as ssim

    n = len(gen_paths)
    pix, ssims = [], []
    for i in tqdm(range(n), desc="lowlevel"):
        g = np.asarray(Image.open(gen_paths[i]).convert("RGB").resize((425, 425), Image.Resampling.BILINEAR))
        t = np.asarray(Image.open(gt_paths[i]).convert("RGB").resize((425, 425), Image.Resampling.BILINEAR))
        g = g.astype(np.float64) / 255.0
        t = t.astype(np.float64) / 255.0
        pix.append(float(np.corrcoef(t.reshape(1, -1), g.reshape(1, -1))[0, 1]))
        gg = rgb2gray(g)
        tt = rgb2gray(t)
        ssims.append(
            float(ssim(tt, gg, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0))
        )
    # `pix` can contain a NaN when a generated image is constant; the mean is then NaN
    # too and the arm is visibly broken, which is the intended behaviour rather than
    # silently dropping the concept.
    return {"pixcorr": float(np.mean(pix)), "ssim": float(np.mean(ssims)),
            "pixcorr_per_image": pix, "ssim_per_image": ssims}


# ------------------------------------------------------ feature encoders

class Encoders:
    """One place for all feature extractors (official configs)."""

    def __init__(self, device: torch.device, cache_root: Path):
        self.device = device
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._cache = {}

    def _cached(self, key: str, build):
        if key in self._cache:
            return self._cache[key]
        obj = build()
        self._cache[key] = obj
        return obj

    # -- AlexNet (2-way, official nodes features.4 / features.11)
    @torch.no_grad()
    def alex_activations(self, paths: list[Path], node: str) -> np.ndarray:
        from torchvision.models import AlexNet_Weights, alexnet
        from torchvision.models.feature_extraction import create_feature_extractor

        model = self._cached(
            f"alex_{node}",
            lambda: create_feature_extractor(
                alexnet(weights=AlexNet_Weights.IMAGENET1K_V1), return_nodes=[node]
            ).to(self.device).eval(),
        )
        feats = []
        for p in tqdm(paths, desc=f"alex[{node}]"):
            x = to_tensor_pil(p).to(self.device)
            x = torch.nn.functional.interpolate(x.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False)
            x = (x - torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)) / torch.tensor(
                [0.229, 0.224, 0.225], device=self.device
            ).view(1, 3, 1, 1)
            out = model(x)[node]
            feats.append(out.flatten(1).float().cpu())
        return torch.cat(feats, dim=0).numpy()

    # -- Inception-v3 avgpool
    @torch.no_grad()
    def inception_activations(self, paths: list[Path]) -> np.ndarray:
        from torchvision.models import Inception_V3_Weights, inception_v3
        from torchvision.models.feature_extraction import create_feature_extractor

        model = self._cached(
            "inception",
            lambda: create_feature_extractor(
                inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1), return_nodes=["avgpool"]
            ).to(self.device).eval(),
        )
        feats = []
        for p in tqdm(paths, desc="inception"):
            x = to_tensor_pil(p).to(self.device)
            x = torch.nn.functional.interpolate(x.unsqueeze(0), size=(342, 342), mode="bilinear", align_corners=False)
            x = (x - torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)) / torch.tensor(
                [0.229, 0.224, 0.225], device=self.device
            ).view(1, 3, 1, 1)
            out = model(x)["avgpool"]
            feats.append(out.flatten(1).float().cpu())
        return torch.cat(feats, dim=0).numpy()

    # -- OpenCLIP ViT-H/14 (same family as MindEye / META-MEG / CogCap)
    @torch.no_grad()
    def clip_features(self, paths: list[Path]) -> np.ndarray:
        import open_clip

        model, _, preprocess = self._cached(
            "clip",
            lambda: open_clip.create_model_and_transforms(
                "ViT-H-14", pretrained="laion2b_s32b_b79k", device=self.device
            ),
        )
        if isinstance(model, tuple):
            model, _, preprocess = model
        model = model.to(self.device).eval()
        feats = []
        for p in tqdm(paths, desc="clip"):
            x = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(self.device)
            feats.append(model.encode_image(x).float().cpu())
        return torch.cat(feats, dim=0).numpy()

    # -- SwAV-RN50 avgpool
    @torch.no_grad()
    def swav_features(self, paths: list[Path]) -> np.ndarray:
        from torchvision import transforms

        model = self._cached("swav", lambda: self._load_swav())
        tfm = transforms.Compose(
            [
                transforms.Resize(224, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        feats = []
        for p in tqdm(paths, desc="swav"):
            x = tfm(Image.open(p).convert("RGB")).unsqueeze(0).to(self.device)
            feats.append(model(x).float().cpu())
        return torch.cat(feats, dim=0).numpy()

    def _load_swav(self):
        from torchvision.models import resnet50

        model = resnet50(weights=None)
        model.fc = torch.nn.Identity()
        state = torch.load(
            "/project/peilab/why/cache/eeg-brainit/torch/hub/checkpoints/swav_800ep_pretrain.pth.tar",
            map_location="cpu",
            weights_only=False,
        )
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        cleaned = {}
        for k, v in state.items():
            nk = k.replace("module.", "")
            if nk.startswith("projection_head") or nk.startswith("prototypes"):
                continue
            cleaned[nk] = v
        model.load_state_dict(cleaned, strict=False)
        return model.to(self.device).eval()

    # -- EfficientNet-B1 avgpool
    @torch.no_grad()
    def effnet_features(self, paths: list[Path]) -> np.ndarray:
        from torchvision.models import EfficientNet_B1_Weights, efficientnet_b1
        from torchvision.models.feature_extraction import create_feature_extractor

        model = self._cached(
            "effnet",
            lambda: create_feature_extractor(
                efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1), return_nodes=["avgpool"]
            ).to(self.device).eval(),
        )
        feats = []
        for p in tqdm(paths, desc="effnet"):
            x = to_tensor_pil(p).to(self.device)
            x = torch.nn.functional.interpolate(x.unsqueeze(0), size=(255, 255), mode="bilinear", align_corners=False)
            x = (x - torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)) / torch.tensor(
                [0.229, 0.224, 0.225], device=self.device
            ).view(1, 3, 1, 1)
            out = model(x)["avgpool"]
            feats.append(out.flatten(1).float().cpu())
        return torch.cat(feats, dim=0).numpy()


# ------------------------------------------------------------------- FID

def fid_dir(gen_dir: Path, gt_paths: list[Path], device: torch.device, batch_size: int) -> float:
    """torchmetrics FID (Inception-v3), identical impl to erdc_fid_metrics."""
    from erdc_fid_metrics import compute_fid  # type: ignore

    n = len(gt_paths)
    gd = Path(gen_dir)
    if not (gd / f"{n - 1:03d}.png").is_file():
        # fall back to global scan for flat dirs that are not 0..n-1 labelled
        from eval_atm_pipeline import list_test_images  # noqa: F401
        raise FileNotFoundError(f"{gd} not labelled 000..{n-1:03d}.png")
    return float(compute_fid(gd, gt_paths, device, batch_size=batch_size)["fid"])


# ------------------------------------------------------------------- main

def encode_gt(gt_paths, enc, cache_root, force: bool) -> dict:
    npz = Path(cache_root) / "gt_feats.npz"
    if npz.is_file() and not force:
        print(f"[INFO] load cached GT features: {npz}")
        return dict(np.load(npz))
    data = {
        "alex2": enc.alex_activations(gt_paths, "features.4"),
        "alex5": enc.alex_activations(gt_paths, "features.11"),
        "inception": enc.inception_activations(gt_paths),
        "clip": enc.clip_features(gt_paths),
        "swav": enc.swav_features(gt_paths),
        "effnet": enc.effnet_features(gt_paths),
    }
    np.savez_compressed(npz, **data)
    return {k: v for k, v in data.items()}


def encode_gen(gen_paths, enc, cache_root, tag: str, force: bool) -> dict:
    npz = Path(cache_root) / f"gen_{tag}.npz"
    if npz.is_file() and not force:
        print(f"[INFO] load cached gen features: {npz}")
        return dict(np.load(npz))
    data = {
        "alex2": enc.alex_activations(gen_paths, "features.4"),
        "alex5": enc.alex_activations(gen_paths, "features.11"),
        "inception": enc.inception_activations(gen_paths),
        "clip": enc.clip_features(gen_paths),
        "swav": enc.swav_features(gen_paths),
        "effnet": enc.effnet_features(gen_paths),
    }
    np.savez_compressed(npz, **data)
    return {k: v for k, v in data.items()}


def evaluate_row(
    tag: str,
    gen_dir: Path,
    gt_paths: list[Path],
    enc: Encoders,
    cache_root: Path,
    device: torch.device,
    batch_size: int,
    do_fid: bool,
    force: bool,
) -> dict:
    n = len(gt_paths)
    gen_paths = list_gen(Path(gen_dir), n)
    row = {"tag": tag, "gen_dir": str(gen_dir), "n": n}

    lo = lowlevel(gen_paths, gt_paths)
    row.update(lo)

    gf = encode_gen(gen_paths, enc, cache_root, tag, force)
    gt = encode_gt(gt_paths, enc, cache_root, force)

    row["alex2"] = two_way(gt["alex2"], gf["alex2"])
    row["alex5"] = two_way(gt["alex5"], gf["alex5"])
    row["inception"] = two_way(gt["inception"], gf["inception"])
    row["clip"] = two_way(gt["clip"], gf["clip"])
    row["swav"] = mean_corr_dist(gt["swav"], gf["swav"])
    row["effnet"] = mean_corr_dist(gt["effnet"], gf["effnet"])

    if do_fid:
        row["fid"] = fid_dir(gen_dir, gt_paths, device, batch_size)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-fid", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="recompute cached features")
    args = ap.parse_args()

    cache = Path("/project/peilab/why/cache/eeg-brainit")
    os.environ.setdefault("HF_HOME", str(cache / "hf"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "hf" / "hub"))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache / "open_clip"))
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    sys.path.insert(0, "/project/peilab/why/eeg-brainit/scripts")

    from eval_atm_pipeline import list_test_images  # type: ignore

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = out_dir / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows_cfg = manifest["rows"]
    gt = list_test_images(Path(args.images_root))
    n = min(len(gt), int(manifest.get("n", len(gt))))
    gt = gt[:n]
    print(f"[INFO] {len(gt)} GT test images")

    enc = Encoders(device, cache_root)
    rows = []
    t0 = time.time()
    for cfg in rows_cfg:
        tag = cfg["tag"]
        gd = Path(cfg["gen_dir"])
        if not (gd / f"{n - 1:03d}.png").is_file():
            # allow manifest to point at the parent that contains generated/
            alt = gd / "generated"
            if (alt / f"{n - 1:03d}.png").is_file():
                gd = alt
            else:
                raise FileNotFoundError(f"bad gen_dir {gd}")
        print(f"\n===== {tag} @ {gd}")
        row = evaluate_row(
            tag, gd, gt, enc, cache_root, device, args.batch_size, not args.no_fid, args.overwrite
        )
        rows.append(row)
        print(
            f"[RESULT {tag}] Pix={row['pixcorr']:.4f} SSIM={row['ssim']:.4f} "
            f"A2={row['alex2']:.4f} A5={row['alex5']:.4f} Inc={row['inception']:.4f} "
            f"CLIP={row['clip']:.4f} SwAV={row['swav']:.4f} Eff={row['effnet']:.4f} "
            + (f"FID={row['fid']:.2f}" if "fid" in row else "")
        )

    # optional pooled-FID reference (e.g. 10-subject pooled HCMA)
    pooled = manifest.get("pooled_fid")
    report = {
        "protocol": {
            "family": "Ozcelik&VanRullen / ATM NeurIPS24 / MindEye 2-way; CogCap AAAI25 follows Benchetrit'24",
            "pixcorr": "Pearson RGB, resize 425 BILINEAR",
            "ssim": "skimage gray@425 gaussian sigma=1.5 use_sample_covariance=False data_range=1.0",
            "alex2_node": "features.4",
            "alex5_node": "features.11",
            "inception_node": "avgpool",
            "clip_backbone": "OpenCLIP ViT-H/14 laion2b_s32b_b79k (MindEye/META-MEG/CogCap family; ATM paper uses OpenAI ViT-L/14)",
            "twoway": "Pearson correlation (np.corrcoef) percent-correct",
            "swav": "mean 1-pearson, SwAV-RN50 avgpool, resize 224",
            "effnet": "mean 1-pearson, EfficientNet-B1 avgpool, resize 255",
            "fid": "torchmetrics FrechetInceptionDistance(normalize=True) @299, 200 fake vs 200 real",
            "n_images": n,
        },
        "rows": rows,
    }
    if pooled is not None:
        report["pooled_fid"] = pooled

    (out_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[OK] wrote {out_dir / 'results.json'}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
