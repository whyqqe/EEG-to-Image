#!/usr/bin/env python3
"""Standard MindEye/ATM/CogCap reconstruction seven-metric table.

Seven metrics (paper protocol):
  Low-level:  PixCorr↑, SSIM↑ (skimage)
  High-level: AlexNet(2)↑, AlexNet(5)↑, Inception↑, CLIP↑  — all as two-way ID
  Dist:       SwAV↓  — mean correlation distance on SwAV-ResNet50 features

Reuses existing paper_metrics + erdc_2wc when present; only computes SwAV (+ optional refresh).
Writes per-subject JSON and aggregated STANDARD_SEVEN_TABLE.md / .json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

BRAINIT = Path("/project/peilab/why/eeg-brainit")
sys.path.insert(0, str(BRAINIT / "scripts"))
from eval_atm_pipeline import list_test_images  # type: ignore


def list_gen(gen_dir: Path, n: int) -> list[Path]:
    out = []
    for i in range(n):
        p = gen_dir / f"{i:03d}.png"
        if not p.is_file():
            p = gen_dir / f"{i:03d}.jpg"
        if not p.is_file():
            raise FileNotFoundError(p)
        out.append(p)
    return out


def _pearson_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a,b (N,D) -> per-row pearson (N,)."""
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    num = (a * b).sum(dim=1)
    den = (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-8)
    return num / den


def load_swav_resnet50(device: torch.device) -> torch.nn.Module:
    """SwAV-ResNet50 backbone (no FC), MindEye/Brain-Diffuser style."""
    from torchvision.models import resnet50

    model = resnet50(weights=None)
    model.fc = torch.nn.Identity()
    # Prefer local torch hub cache / explicit checkpoint
    candidates = [
        Path(os.environ.get("TORCH_HOME", "")) / "hub" / "checkpoints" / "swav_800ep_pretrain.pth.tar",
        Path("/project/peilab/why/cache/eeg-brainit/torch/hub/checkpoints/swav_800ep_pretrain.pth.tar"),
        Path("/project/peilab/why/cache/eeg-brainit/torch/swav_800ep_pretrain.pth.tar"),
    ]
    ckpt_path = next((p for p in candidates if p.is_file()), None)
    if ckpt_path is None:
        url = "https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar"
        print(f"[INFO] downloading SwAV weights from {url}")
        state = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=True)
        # cache copy
        cache = Path("/project/peilab/why/cache/eeg-brainit/torch/hub/checkpoints")
        cache.mkdir(parents=True, exist_ok=True)
        torch.save(state, cache / "swav_800ep_pretrain.pth.tar")
    else:
        print(f"[INFO] loading SwAV weights {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # strip module. / projection head
    cleaned = {}
    for k, v in state.items():
        nk = k.replace("module.", "")
        if nk.startswith("projection_head") or nk.startswith("prototypes"):
            continue
        cleaned[nk] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[INFO] SwAV load missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(device).eval()


@torch.no_grad()
def compute_swav_distance(
    gen_paths: list[Path],
    gt_paths: list[Path],
    device: torch.device,
    batch_size: int = 16,
) -> float:
    """Mean (1 - pearson) on SwAV-ResNet50 pooled features — lower is better."""
    from torchvision import transforms

    model = load_swav_resnet50(device)
    tfm = transforms.Compose(
        [
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def encode(paths: list[Path]) -> torch.Tensor:
        feats = []
        for start in tqdm(range(0, len(paths), batch_size), desc="swav"):
            batch = torch.cat(
                [tfm(Image.open(p).convert("RGB")).unsqueeze(0) for p in paths[start : start + batch_size]],
                dim=0,
            ).to(device)
            feats.append(model(batch).float().cpu())
        return torch.cat(feats, dim=0)

    g = encode(gen_paths)
    t = encode(gt_paths)
    corr = _pearson_rows(g, t)
    dist = float((1.0 - corr).mean().item())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return dist


def load_subject_seven(sub_dir: Path, tag: str) -> dict:
    paper = json.loads((sub_dir / "metrics" / "paper_metrics.json").read_text())
    row = paper["results"][0] if "results" in paper else paper
    tw = json.loads((sub_dir / "metrics" / "erdc_2wc.json").read_text())["twoway"]
    swav_path = sub_dir / "metrics" / "swav.json"
    swav = json.loads(swav_path.read_text())["swav"] if swav_path.is_file() else None
    return {
        "subject": sub_dir.name,
        "pixcorr": float(row["pixcorr"]),
        "ssim": float(row.get("ssim_skimage", row["ssim"])),
        "alex2": float(tw["alex2"]),
        "alex5": float(tw["alex5"]),
        "inception": float(tw["inception"]),
        "clip": float(tw["clip"]),
        "swav": swav,
        "fid": float(row.get("fid")) if row.get("fid") is not None else None,
        "tag": tag,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--tag", type=str, default="hcma_full_a40")
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--force-swav", action="store_true")
    args = ap.parse_args()

    cache = Path("/project/peilab/why/cache/eeg-brainit")
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache / "open_clip"))

    root = Path(args.root)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    gt = list_test_images(Path(args.images_root))
    n = len(gt)

    rows = []
    for sub in sorted(root.glob("sub-*")):
        gdir = sub / "generation" / args.tag / "generated"
        if not (gdir / f"{n-1:03d}.png").is_file():
            raise FileNotFoundError(gdir)
        swav_json = sub / "metrics" / "swav.json"
        if args.force_swav or not swav_json.is_file():
            gens = list_gen(gdir, n)
            dist = compute_swav_distance(gens, gt, device, args.batch_size)
            swav_json.parent.mkdir(parents=True, exist_ok=True)
            swav_json.write_text(
                json.dumps(
                    {
                        "n": n,
                        "swav": dist,
                        "protocol": "mean(1-pearson) SwAV-ResNet50 features; MindEye/ATM style (lower better)",
                        "gen_dir": str(gdir),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[OK] {sub.name} SwAV={dist:.4f}")
        row = load_subject_seven(sub, args.tag)
        assert row["swav"] is not None
        rows.append(row)
        print(
            f"[OK] {row['subject']}: Pix={row['pixcorr']:.3f} SSIM={row['ssim']:.3f} "
            f"A2={row['alex2']:.3f} A5={row['alex5']:.3f} Inc={row['inception']:.3f} "
            f"CLIP={row['clip']:.3f} SwAV={row['swav']:.3f}"
        )

    def mean_std(key: str) -> tuple[float, float]:
        vals = [float(r[key]) for r in rows]
        return float(np.mean(vals)), float(np.std(vals))

    ours = {k: {"mean": mean_std(k)[0], "std": mean_std(k)[1]} for k in [
        "pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "swav"
    ]}
    fid_m, fid_s = mean_std("fid") if rows[0]["fid"] is not None else (None, None)

    # Literature (standard seven; ATM/CogCap/CogCapPro). FID optional extras.
    sota = [
        {
            "method": "ATM (Li et al., NeurIPS'24)",
            "scope": "THINGS-EEG sub-08",
            "pixcorr": 0.160, "ssim": 0.345, "alex2": 0.776, "alex5": 0.866,
            "inception": 0.734, "clip": 0.786, "swav": 0.582,
            "source": "arXiv:2403.07721 Table 1",
        },
        {
            "method": "CogCap all (Zhang et al., AAAI'25)",
            "scope": "THINGS-EEG 10subj mean",
            "pixcorr": 0.150, "ssim": 0.347, "alex2": 0.754, "alex5": 0.623,
            "inception": 0.669, "clip": 0.715, "swav": 0.590,
            "source": "arXiv:2412.10489 Table 3",
        },
        {
            "method": "CogCap all",
            "scope": "THINGS-EEG sub-08",
            "pixcorr": 0.175, "ssim": 0.366, "alex2": None, "alex5": None,
            "inception": None, "clip": 0.744, "swav": None,
            "source": "CogCap supp. Table 6 (partial)",
        },
        {
            "method": "CogCapPro (Zhang et al.)",
            "scope": "THINGS-EEG",
            "pixcorr": 0.163, "ssim": 0.398, "alex2": None, "alex5": None,
            "inception": 0.779, "clip": 0.830, "swav": 0.553,
            "source": "arXiv:2603.12722 Table III",
        },
        {
            "method": "MindEye (fMRI ref)",
            "scope": "NSD-fMRI (not EEG)",
            "pixcorr": 0.309, "ssim": 0.323, "alex2": 0.947, "alex5": 0.978,
            "inception": 0.938, "clip": 0.941, "swav": 0.367,
            "source": "CogCap Table 3 citation",
        },
    ]

    # Load pooled FID if present
    pooled = None
    pf = root / "metrics_pooled_fid.json"
    if pf.is_file():
        pooled = json.loads(pf.read_text()).get("pooled_fid_unique_gt")

    report = {
        "protocol": {
            "name": "MindEye / ATM / CogCap standard seven",
            "metrics": [
                "PixCorr↑ (pixel pearson @256)",
                "SSIM↑ (skimage RGB 256, data_range=255)",
                "AlexNet(2)↑ two-way ID",
                "AlexNet(5)↑ two-way ID",
                "Inception↑ two-way ID",
                "CLIP↑ two-way ID (OpenCLIP ViT-H/14 features in erdc_twoway)",
                "SwAV↓ mean correlation distance (SwAV-ResNet50)",
            ],
            "n_subjects": len(rows),
            "n_images_per_subject": n,
            "tag": args.tag,
        },
        "ours_per_subject": rows,
        "ours_mean": {k: ours[k]["mean"] for k in ours},
        "ours_std": {k: ours[k]["std"] for k in ours},
        "ours_fid_mean_per_subject": fid_m,
        "ours_fid_pooled": pooled,
        "sota_reference": sota,
    }

    out_json = root / "STANDARD_SEVEN.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def fmt(v, digits=3):
        if v is None:
            return "—"
        return f"{v:.{digits}f}"

    md = []
    md.append("# Standard Seven Metrics (MindEye / ATM / CogCap protocol)\n\n")
    md.append(
        "Protocol: PixCorr, SSIM (skimage), AlexNet(2/5) **2-way**, Inception **2-way**, "
        "CLIP **2-way**, SwAV correlation distance. "
        f"Ours = HCMA 10-subject (`{args.tag}`), mean±std over {len(rows)} subjects.\n\n"
    )
    md.append("## Main table\n\n")
    md.append(
        "| Method | Scope | PixCorr↑ | SSIM↑ | AlexNet(2)↑ | AlexNet(5)↑ | Inception↑ | CLIP↑ | SwAV↓ |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    om = report["ours_mean"]
    os_ = report["ours_std"]
    md.append(
        f"| **Ours (HCMA)** | **10subj mean±std** | "
        f"**{om['pixcorr']:.3f}±{os_['pixcorr']:.3f}** | "
        f"**{om['ssim']:.3f}±{os_['ssim']:.3f}** | "
        f"**{om['alex2']:.3f}±{os_['alex2']:.3f}** | "
        f"**{om['alex5']:.3f}±{os_['alex5']:.3f}** | "
        f"**{om['inception']:.3f}±{os_['inception']:.3f}** | "
        f"**{om['clip']:.3f}±{os_['clip']:.3f}** | "
        f"**{om['swav']:.3f}±{os_['swav']:.3f}** |\n"
    )
    for s in sota:
        md.append(
            f"| {s['method']} | {s['scope']} | {fmt(s['pixcorr'])} | {fmt(s['ssim'])} | "
            f"{fmt(s['alex2'])} | {fmt(s['alex5'])} | {fmt(s['inception'])} | "
            f"{fmt(s['clip'])} | {fmt(s['swav'])} |\n"
        )
    md.append("\n## Per-subject (Ours)\n\n")
    md.append("| Sub | PixCorr | SSIM | A2 | A5 | Inc | CLIP | SwAV |\n|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        md.append(
            f"| {r['subject']} | {r['pixcorr']:.3f} | {r['ssim']:.3f} | {r['alex2']:.3f} | "
            f"{r['alex5']:.3f} | {r['inception']:.3f} | {r['clip']:.3f} | {r['swav']:.3f} |\n"
        )
    if pooled is not None:
        md.append(
            f"\n## Extra (not part of the classic seven)\n\n"
            f"- FID pooled (2000 vs 200 GT): **{pooled:.2f}**\n"
            f"- FID mean per-subject: **{fid_m:.2f}**\n"
        )
    md.append("\n## Sources\n\n")
    for s in sota:
        md.append(f"- {s['method']}: {s['source']}\n")
    md.append(
        "\nOurs CLIP/Alex/Inception 2-way from `erdc_twoway_metrics.py`; "
        "PixCorr/SSIM from `eval_paper_metrics.py`; SwAV from this script.\n"
    )

    out_md = root / "STANDARD_SEVEN_TABLE.md"
    out_md.write_text("".join(md), encoding="utf-8")
    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_md}")
    print(json.dumps(report["ours_mean"], indent=2))


if __name__ == "__main__":
    main()
