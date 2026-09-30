#!/usr/bin/env python3
"""Build concept prompts from strongest EEG→concept retrieval.

Critical finding: official NB checkpoint reports ~73% Top-1 under CPA-augmented
image gallery (image_test_aug), but only ~35% on clean RN50 features. For prompt
writing we use the CPA-aug test gallery (same 200 held-out concepts) so concept ID
matches the model's trained retrieval geometry, then optionally margin-gate to
avoid wrong-concept damage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT))
from module.dataset import EEGPreImageDataset  # noqa: E402
from module.eeg_encoder.model import EEGProject  # noqa: E402
from module.projector import ProjectorLinear  # noqa: E402

DEFAULT_CHANNELS = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]


def l2(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)).astype(np.float32)


def encode_eeg(ckpt_path: Path, subject: int, device: torch.device) -> np.ndarray:
    eeg_dir = str(NB_ROOT / "data/things_eeg/preprocessed_eeg")
    rn50_dir = str(NB_ROOT / "data/things_eeg/image_feature/RN50")
    ds = EEGPreImageDataset(
        [subject], eeg_dir, DEFAULT_CHANNELS, [0, 250],
        rn50_dir, "", False, [], True, False, None, False, False, False, False,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = EEGProject(
        feature_dim=int(ds.image_features.shape[-1]),
        eeg_sample_points=int(ds.num_sample_points),
        channels_num=int(ds.channels_num),
    ).to(device)
    eeg_proj = ProjectorLinear(int(ds.image_features.shape[-1]), 512).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    eeg_proj.load_state_dict(ckpt["eeg_projector_state_dict"])
    model.eval()
    eeg_proj.eval()
    outs = []
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=200, shuffle=False):
            raw = model(batch[0].to(device))
            outs.append(F.normalize(eeg_proj(raw), dim=-1).cpu().numpy())
    return l2(np.concatenate(outs, axis=0))


def project_gallery(img: np.ndarray, ckpt_path: Path, out_dim: int, device: torch.device) -> np.ndarray:
    # Accept (N,D), (N,1,D), (1,N,1,D), (R,N,K,D) → mean-pool to (N,D) concept prototypes
    x = np.asarray(img)
    if x.ndim == 4:
        # (reps, concepts, imgs_per, dim)
        x = x.mean(axis=(0, 2))
    elif x.ndim == 3:
        if x.shape[0] == 200:
            x = x.mean(axis=1)
        elif x.shape[1] == 200:
            x = x.mean(axis=0)
            if x.ndim == 3:
                x = x.mean(axis=1)
        else:
            raise ValueError(f"unexpected gallery shape {img.shape}")
    if x.ndim != 2 or x.shape[0] != 200:
        raise ValueError(f"gallery must be (200,D), got {x.shape} from {img.shape}")
    img = x.astype(np.float32)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    proj = ProjectorLinear(img.shape[-1], out_dim)
    proj.load_state_dict(ckpt["img_projector_state_dict"])
    proj.eval()
    with torch.no_grad():
        gal = F.normalize(proj(torch.from_numpy(img)), dim=-1).cpu().numpy()
    return l2(gal)


def build_prompts(names: list[str], margins: np.ndarray, gate: float, soft_gate: float) -> list[str]:
    prompts = []
    for name, m in zip(names, margins):
        if m < gate:
            prompts.append("")  # abstain: keep pure image condition
        elif m < soft_gate:
            prompts.append(f"a photo of {name}")
        else:
            prompts.append(f"a photo of {name}, highly detailed")
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg-embed-npy", type=str, default="", help="optional precomputed SSP-512; else encode from ckpt")
    ap.add_argument("--nb-ckpt", type=str, required=True)
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--gallery", type=str, default="cpa", choices=["cpa", "clean", "ensemble"])
    ap.add_argument("--rn50-image-test", type=str, default=str(NB_ROOT / "data/things_eeg/image_feature/RN50/image_test.npy"))
    ap.add_argument(
        "--cpa-image-test",
        type=str,
        default=str(
            NB_ROOT
            / "data/things_eeg/image_feature/RN50/GaussianBlur-GaussianNoise-LowResolution-Mosaic/test.npy"
        ),
    )
    ap.add_argument("--concept-phrases-test", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--margin-gate", type=float, default=0.02, help="below → empty prompt")
    ap.add_argument("--soft-gate", type=float, default=0.05, help="below → short prompt")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--top-k", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    ckpt_path = Path(args.nb_ckpt)

    phrases = json.loads(Path(args.concept_phrases_test).read_text())
    if args.eeg_embed_npy:
        eeg = l2(np.load(args.eeg_embed_npy).astype(np.float32))
    else:
        eeg = encode_eeg(ckpt_path, args.subject, device)
    assert len(phrases) == eeg.shape[0], (len(phrases), eeg.shape)

    clean = np.load(args.rn50_image_test)
    cpa = np.load(args.cpa_image_test)
    gal_clean = project_gallery(clean, ckpt_path, eeg.shape[-1], device)
    gal_cpa = project_gallery(cpa, ckpt_path, eeg.shape[-1], device)

    sim_clean = eeg @ gal_clean.T
    sim_cpa = eeg @ gal_cpa.T
    if args.gallery == "cpa":
        sim = sim_cpa
        gallery_name = "CPA-aug RN50 test gallery (NeuroBridge image_test_aug protocol)"
    elif args.gallery == "clean":
        sim = sim_clean
        gallery_name = "clean RN50 image_test"
    else:
        sim = 0.3 * sim_clean + 0.7 * sim_cpa
        gallery_name = "0.3*clean + 0.7*CPA"

    topk = args.top_k
    pred, always_names = [], []
    correct = 0
    margins = np.zeros(len(phrases), dtype=np.float32)
    for i in range(sim.shape[0]):
        idx = np.argsort(-sim[i])[: max(topk, 5)]
        names = [phrases[j] for j in idx]
        pred.append(names[:topk])
        always_names.append(names[0])
        if names[0] == phrases[i]:
            correct += 1
        srt = np.sort(sim[i])[::-1]
        margins[i] = float(srt[0] - srt[1]) if len(srt) > 1 else float(srt[0])

    top1 = correct / len(phrases)
    top5 = sum(
        1 for i in range(len(phrases)) if phrases[i] in [phrases[j] for j in np.argsort(-sim[i])[:5]]
    ) / len(phrases)
    top1_clean = float((sim_clean.argmax(1) == np.arange(len(phrases))).mean())
    top1_cpa = float((sim_cpa.argmax(1) == np.arange(len(phrases))).mean())

    prompts_always = [f"a photo of {n}, highly detailed" for n in always_names]
    prompts_gated = build_prompts(always_names, margins, args.margin_gate, args.soft_gate)
    oracle = [f"a photo of {p}, highly detailed" for p in phrases]

    # Adaptive gen knobs: prompt → slightly more denoise / less IP; abstain → NDA-SS defaults
    strength = np.full(len(phrases), 0.40, dtype=np.float32)
    ip_scale = np.full(len(phrases), 1.00, dtype=np.float32)
    for i, p in enumerate(prompts_gated):
        if p:
            strength[i] = 0.45
            ip_scale[i] = 0.90

    gated_cover = float(np.mean([bool(p) for p in prompts_gated]))
    gated_prec = float(
        np.mean([always_names[i] == phrases[i] for i, p in enumerate(prompts_gated) if p] or [0.0])
    )

    (out / "prompts_pred.json").write_text(json.dumps(prompts_always, indent=2), encoding="utf-8")
    (out / "prompts_gated.json").write_text(json.dumps(prompts_gated, indent=2), encoding="utf-8")
    (out / "prompts_oracle.json").write_text(json.dumps(oracle, indent=2), encoding="utf-8")
    (out / "pred_concepts.json").write_text(json.dumps(pred, indent=2), encoding="utf-8")
    np.save(out / "margins.npy", margins)
    np.save(out / "strength_gated.npy", strength)
    np.save(out / "ip_scale_gated.npy", ip_scale)

    report = {
        "method": "NB EEG → CPA-aug/clean gallery concept retrieval → prompt",
        "gallery": args.gallery,
        "gallery_detail": gallery_name,
        "nb_ckpt": str(ckpt_path),
        "eeg_source": args.eeg_embed_npy or "live_encode_from_ckpt",
        "concept_top1": top1,
        "concept_top5": top5,
        "concept_top1_clean_gallery": top1_clean,
        "concept_top1_cpa_gallery": top1_cpa,
        "margin_gate": args.margin_gate,
        "soft_gate": args.soft_gate,
        "gated_prompt_coverage": gated_cover,
        "gated_prompt_precision": gated_prec,
        "n": len(phrases),
        "note": "CPA gallery matches NeuroBridge train-time image_test_aug; still 200-way held-out concepts",
    }
    (out / "prompt_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
