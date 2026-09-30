#!/usr/bin/env python3
"""Pooled (总) FID in the SOTA distribution-level sense.

Protocol (MindEye / ATM / torchmetrics FID style):
  - Fake: ALL subject generations concatenated (10 x 200 = 2000)
  - Real: unique THINGS-EEG2 test GT images (200)
  - Metric: FrechetInceptionDistance (Inception-v3, normalize=True)

Also reports mean of per-subject FIDs for reference (NOT the SOTA pooled number).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from tqdm import tqdm

BRAINIT = Path("/project/peilab/why/eeg-brainit")
sys.path.insert(0, str(BRAINIT / "scripts"))
from eval_atm_pipeline import list_test_images  # type: ignore


def load_batch(paths: list[Path], tfm, device: torch.device) -> torch.Tensor:
    return torch.cat(
        [tfm(Image.open(p).convert("RGB")).unsqueeze(0).to(device) for p in paths],
        dim=0,
    )


@torch.no_grad()
def fid_from_lists(
    real_paths: list[Path],
    fake_paths: list[Path],
    device: torch.device,
    batch_size: int,
    tag: str,
) -> float:
    fid = FrechetInceptionDistance(normalize=True).to(device)
    tfm = transforms.Compose(
        [transforms.Resize((299, 299), antialias=True), transforms.ToTensor()]
    )
    for start in tqdm(range(0, len(real_paths), batch_size), desc=f"FID-real[{tag}]"):
        end = min(start + batch_size, len(real_paths))
        fid.update(load_batch(real_paths[start:end], tfm, device), real=True)
    for start in tqdm(range(0, len(fake_paths), batch_size), desc=f"FID-fake[{tag}]"):
        end = min(start + batch_size, len(fake_paths))
        fid.update(load_batch(fake_paths[start:end], tfm, device), real=False)
    return float(fid.compute().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="hcma_10subj root")
    ap.add_argument("--tag", type=str, default="hcma_full_a40")
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--output-json", type=str, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    cache = Path("/project/peilab/why/cache/eeg-brainit")
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache / "open_clip"))

    root = Path(args.root)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    gt = list_test_images(Path(args.images_root))
    n_gt = len(gt)

    fake_paths: list[Path] = []
    per_subj = []
    for sub in sorted(root.glob("sub-*")):
        gdir = sub / "generation" / args.tag / "generated"
        if not (gdir / f"{n_gt - 1:03d}.png").is_file():
            raise FileNotFoundError(gdir)
        paths = [gdir / f"{i:03d}.png" for i in range(n_gt)]
        fake_paths.extend(paths)
        # reuse existing per-subject FID if present
        pm = sub / "metrics" / "paper_metrics.json"
        if pm.is_file():
            d = json.loads(pm.read_text())
            rows = d.get("results", d if isinstance(d, list) else [d])
            fid_s = None
            for r in rows:
                if isinstance(r, dict) and "fid" in r:
                    fid_s = float(r["fid"])
                    break
            if fid_s is not None:
                per_subj.append({"subject": sub.name, "n_real": n_gt, "n_fake": n_gt, "fid": fid_s, "source": "paper_metrics"})
                print(f"[OK] {sub.name} FID={fid_s:.2f} (from paper_metrics)")

    pooled = fid_from_lists(gt, fake_paths, device, args.batch_size, "pooled_all_subj")
    # also: tiled-GT variant (equal N) sometimes used when concatenating folders
    tiled = fid_from_lists(gt * (len(fake_paths) // n_gt), fake_paths, device, args.batch_size, "pooled_tiled_gt")

    import numpy as np

    report = {
        "protocol": {
            "metric": "torchmetrics.FrechetInceptionDistance(normalize=True)",
            "feature": "Inception-v3",
            "sota_primary": "pooled_fid_unique_gt",
            "pooled_fid_unique_gt": "fake=all subject gens; real=unique test GT (distribution-level, unequal N OK)",
            "pooled_fid_tiled_gt": "fake=all gens; real=GT repeated per subject (equal N sanity check)",
            "per_subject_mean_fid": "mean of per-subject FIDs (NOT the primary SOTA pooled number)",
        },
        "n_subjects": len(per_subj),
        "n_gt_unique": n_gt,
        "n_fake_total": len(fake_paths),
        "pooled_fid_unique_gt": pooled,
        "pooled_fid_tiled_gt": tiled,
        "per_subject_fid": per_subj,
        "mean_per_subject_fid": float(np.mean([x["fid"] for x in per_subj])),
        "std_per_subject_fid": float(np.std([x["fid"] for x in per_subj])),
        "tag": args.tag,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in [
        "pooled_fid_unique_gt", "pooled_fid_tiled_gt", "mean_per_subject_fid", "n_fake_total", "n_gt_unique"
    ]}, indent=2))
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
