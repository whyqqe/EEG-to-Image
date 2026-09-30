#!/usr/bin/env python3
"""Eval NeuroBOLT→BIT pipeline: retrieval + optional SDXL generation (sub-08).

Compares:
  A) direct_clip (S0)
  C) bridge_clip / bit_clip (S1/S3)
against ATM baseline and writes under --output-dir.
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
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from eeg_brainit.data.neurobolt_dataset import NeuroBoltBridgeDataset
from eeg_brainit.models.neurobolt_bridge import NeuroBoltBrainITPipeline


def retrieval_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    sim = pred @ target.T
    ranks = []
    top1 = top5 = 0
    n = sim.shape[0]
    for i in range(n):
        order = np.argsort(-sim[i])
        rank = int(np.where(order == i)[0][0]) + 1
        ranks.append(rank)
        if rank == 1:
            top1 += 1
        if rank <= 5:
            top5 += 1
    ranks = np.asarray(ranks)
    paired = float((pred * target).sum(1).mean())
    shuffled = float(
        (pred * target[np.random.RandomState(0).permutation(n)]).sum(1).mean()
    )
    return {
        "n": int(n),
        "top1": float(top1 / n),
        "top5": float(top5 / n),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(ranks.mean()),
        "paired_cos": paired,
        "shuffled_cos": shuffled,
        "cos_gap": paired - shuffled,
    }


@torch.no_grad()
def encode_all(
    model: NeuroBoltBrainITPipeline,
    tokens: np.ndarray,
    atm: np.ndarray,
    device: torch.device,
    keys: list[str],
    batch_size: int = 128,
) -> dict[str, np.ndarray]:
    model.eval()
    buckets: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    tok_t = torch.from_numpy(tokens.astype(np.float32))
    atm_t = torch.from_numpy(atm.astype(np.float32))
    for i in range(0, len(tok_t), batch_size):
        out = model(tok_t[i : i + batch_size].to(device), atm_t[i : i + batch_size].to(device))
        for k in keys:
            if k in out:
                buckets[k].append(F.normalize(out[k].float(), dim=-1).cpu().numpy())
    return {k: np.concatenate(v, 0).astype(np.float32) for k, v in buckets.items() if v}


def load_model(ckpt: Path, project: Path, device: torch.device) -> NeuroBoltBrainITPipeline:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    # Ensure heads present in weights are enabled.
    nb = cfg.setdefault("neurobolt", {})
    if any(k.startswith("bit.") for k in ck["model"].keys()):
        nb["use_bit"] = True
    if any(k.startswith("direct_head.") for k in ck["model"].keys()):
        nb.setdefault("direct", {})["enabled"] = True
    model = NeuroBoltBrainITPipeline.from_config(cfg, project_root=str(project)).to(device)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    print(f"[INFO] loaded {ckpt.name} missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="sub-08")
    parser.add_argument(
        "--neurobolt-dir",
        type=str,
        default="/project/peilab/why/cache/things_eeg2_b2/neurobolt",
    )
    parser.add_argument("--bridge-dir", type=str, default="outputs/atm_bridge")
    parser.add_argument(
        "--s0-ckpt",
        type=str,
        default="outputs/nb_direct_s0_sub08/checkpoints/nb_stage0_best.pt",
    )
    parser.add_argument(
        "--s1-ckpt",
        type=str,
        default="outputs/nb_bit_s1_sub08/checkpoints/nb_stage1_best.pt",
    )
    parser.add_argument(
        "--s3-ckpt",
        type=str,
        default="outputs/nb_bit_s3_sub08/checkpoints/nb_stage3_best.pt",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/eval/nb_pipeline_sub08")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--gen-sources", type=str, default="bit_clip,direct_clip,atm")
    parser.add_argument("--gen-steps", type=int, default=30)
    parser.add_argument("--gen-size", type=int, default=512)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project = ROOT
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = project / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", "/project/peilab/why/cache/eeg-brainit/hf")
    os.environ.setdefault(
        "HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bridge_dir = Path(args.bridge_dir)
    if not bridge_dir.is_absolute():
        bridge_dir = project / bridge_dir

    test_ds = NeuroBoltBridgeDataset(
        args.neurobolt_dir,
        bridge_dir,
        args.subject,
        split="test",
        teacher_img=project
        / "outputs/eval/atm_baseline/test_ViT-H-14_laion2b_s32b_b79k_features.npy",
    )
    tokens = np.stack([test_ds[i]["fmri_tokens"].numpy() for i in range(len(test_ds))])
    atm = np.stack([test_ds[i]["atm_emb"].numpy() for i in range(len(test_ds))])
    img = np.stack([test_ds[i]["clip_emb"].numpy() for i in range(len(test_ds))])
    print(f"[INFO] test tokens={tokens.shape} atm={atm.shape} img={img.shape}")

    report: dict = {
        "subject": args.subject,
        "chance_top1": 1.0 / float(len(test_ds)),
        "retrieval": {"raw_atm": retrieval_metrics(atm, img)},
        "embeds": {},
    }
    print(
        f"[INFO] raw_atm top1={report['retrieval']['raw_atm']['top1']*100:.2f}% "
        f"top5={report['retrieval']['raw_atm']['top5']*100:.2f}%"
    )
    emb_bank: dict[str, np.ndarray] = {"atm": atm}

    for tag, ckpt_arg, keys in (
        ("direct_s0", args.s0_ckpt, ["atm_emb", "direct_clip"]),
        ("bridge_s1", args.s1_ckpt, ["atm_emb", "bridge_clip"]),
        ("bit_s3", args.s3_ckpt, ["atm_emb", "bridge_clip", "bit_clip"]),
    ):
        ckpt = Path(ckpt_arg)
        if not ckpt.is_absolute():
            ckpt = project / ckpt
        if not ckpt.is_file():
            print(f"[WARN] skip {tag}: missing {ckpt}")
            continue
        model = load_model(ckpt, project, device)
        embeds = encode_all(model, tokens, atm, device, keys)
        for k, arr in embeds.items():
            m = retrieval_metrics(arr, img)
            report["retrieval"][f"{tag}_{k}"] = m
            print(f"[INFO] {tag}_{k} top1={m['top1']*100:.2f}% top5={m['top5']*100:.2f}%")
            np.save(out_dir / f"{args.subject}_{tag}_{k}_1024.npy", arr)
            if k in ("direct_clip", "bridge_clip", "bit_clip"):
                emb_bank[k if k != "bridge_clip" else "bridge_clip"] = arr
            if k == "direct_clip":
                emb_bank["direct_clip"] = arr
            if k == "bit_clip":
                emb_bank["bit_clip"] = arr
        del model
        torch.cuda.empty_cache()

    if not args.skip_generate:
        # Import SDXL helpers from sibling script.
        sys.path.insert(0, str(ROOT / "scripts"))
        from eval_atm_pipeline import generate_images_sdxl, image_metrics, list_test_images

        gt_paths = list_test_images(Path("/project/peilab/why/data/images_set"))
        report["generation"] = {}
        wanted = [s.strip() for s in args.gen_sources.split(",") if s.strip()]
        for src in wanted:
            if src not in emb_bank:
                report["generation"][src] = {"error": f"missing embeds for {src}"}
                print(f"[WARN] skip gen {src}")
                continue
            gen_dir = out_dir / "generated" / src
            try:
                paths = generate_images_sdxl(
                    emb_bank[src],
                    gen_dir,
                    device,
                    steps=args.gen_steps,
                    height=args.gen_size,
                    width=args.gen_size,
                    max_images=args.max_images,
                    seed=args.seed,
                    source_name=src,
                )
                metrics = image_metrics(paths, gt_paths, device)
                report["generation"][src] = metrics
                print(
                    f"[INFO] gen[{src}] pixcorr={metrics['pixcorr']:.4f} "
                    f"ssim={metrics['ssim']:.4f} clip={metrics['clip_cosine']:.4f}"
                )
            except Exception as exc:  # noqa: BLE001
                report["generation"][src] = {"error": str(exc)}
                print(f"[ERROR] gen[{src}] {exc}")

    # Attach ATM pipeline metrics if present for side-by-side.
    atm_m = project / "outputs/eval/atm_pipeline_sub08/metrics.json"
    if atm_m.is_file():
        try:
            report["atm_pipeline_ref"] = json.loads(atm_m.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass

    out_json = out_dir / "metrics.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("========== NEUROBOLT PIPELINE EVAL ==========")
    for k, m in report["retrieval"].items():
        print(
            f"{k:28s} Top-1={m['top1']*100:6.2f}% Top-5={m['top5']*100:6.2f}% "
            f"med={m['median_rank']:5.1f}"
        )
    print(f"[OK] wrote {out_json}")


if __name__ == "__main__":
    main()
