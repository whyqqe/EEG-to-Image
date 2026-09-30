#!/usr/bin/env python
"""Does an init-image directory leak concept identity?

`hybrid_s08` and `uck_nat_s08` are the only experiments in this repo that reached
high generation scores with *generic* prompts (inception 0.728 / clip 0.798),
which would make them the honest bar to beat.  But both take their img2img init
from

    outputs/sdedit_ll_full10/sub-08/vae_head/pred_lowlevel_rgb_512

and `run_sdedit_ll_full10.sh` is on the INVALID list (docs/INVALID_ARTIFACTS.md,
line 60) because it builds its HCMA embeddings with `prompts_full_hcma_test.json`,
a 200/200 class-name-leaking prompt file.  Those runs also use strength=0.82, so
the init image still shapes the trajectory.

Before adopting 0.728/0.798 as the bar, measure whether the init image alone
carries the concept.  Method: encode each init PNG with the same open_clip
ViT-H-14 used to build `image_feature/ViT-H-14/image_test.npy`, then ask whether
init_i retrieves gt_i among the 200 test images.  A clean EEG-derived VAE decode
can carry *some* concept signal (that is the point of the spatial pathway), so the
comparison that matters is contaminated vs clean, measured identically.

Usage:
  python scripts/nda/nw4_diag_initleak.py --out outputs/nw4/sub-08/diag_initleak.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))

DIRS = {
    "CONTAMINATED sdedit_ll_full10 (hybrid/uck_nat init)":
        "outputs/sdedit_ll_full10/sub-08/vae_head/pred_lowlevel_rgb_512",
    "clean nw3 s1 vae decode":
        "outputs/nw3/sub-08/s1/spatial/pred_lowlevel_rgb_512",
    "clean nw4 s1 vae decode":
        "outputs/nw4/sub-08/s1/spatial/pred_lowlevel_rgb_512",
    "GT test images (ceiling)":
        "data/images_set_test",  # only used if present; otherwise handled below
}


def l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def retrieval(q: np.ndarray, bank: np.ndarray, k: int = 10) -> dict:
    sim = l2n(q) @ l2n(bank).T
    n = sim.shape[0]
    top1 = float((sim.argmax(1) == np.arange(n)).mean())
    kk = min(5, sim.shape[1])
    top5 = float(np.mean([np.arange(n)[i] in np.argsort(-sim[i])[:kk] for i in range(n)]))
    kk = max(1, min(k, sim.shape[0] - 1, sim.shape[1] - 1))
    knn_q = np.sort(sim, 1)[:, -kk:].mean(1, keepdims=True)      # (Q,1)
    knn_b = np.sort(sim, 0)[-kk:, :].mean(0, keepdims=True)      # (1,B)
    csls = sim - 0.5 * (knn_q + knn_b)
    top1_csls = float((csls.argmax(1) == np.arange(n)).mean())
    return {"top1": round(top1, 4), "top5": round(top5, 4), "top1_csls": round(top1_csls, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--model", type=str, default="ViT-H-14")
    ap.add_argument("--pretrained", type=str, default="laion2b_s32b_b79k")
    args = ap.parse_args()

    os.environ.setdefault("OPENCLIP_CACHE_DIR",
                          "/project/peilab/why/cache/eeg-brainit/open_clip")

    gt = np.load(NB_ROOT / "data/things_eeg/image_feature/ViT-H-14/image_test.npy")
    gt = gt.reshape(len(gt), -1).astype(np.float32)
    print(f"[gt] image_test.npy -> {gt.shape}; chance top1 = {1/len(gt):.5f}")

    import torch
    import open_clip
    from PIL import Image
    dev = args.device if torch.cuda.is_available() else "cpu"
    model, _, pre = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=dev)
    model.eval()
    print(f"[clip] {args.model} / {args.pretrained} on {dev}")

    @torch.no_grad()
    def encode(paths: list[Path]) -> np.ndarray:
        out = []
        for i in range(0, len(paths), 32):
            batch = torch.stack([pre(Image.open(p).convert("RGB"))
                                 for p in paths[i:i + 32]]).to(dev)
            e = model.encode_image(batch)
            out.append(e.float().cpu().numpy())
        return l2n(np.concatenate(out, 0))

    rows: dict[str, dict] = {}
    for tag, rel in DIRS.items():
        d = NB_ROOT / rel
        if not d.is_dir():
            print(f"  {tag:<48} SKIP (no such dir: {rel})")
            continue
        ps = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
        if len(ps) != len(gt):
            print(f"  {tag:<48} SKIP ({len(ps)} images != {len(gt)} gt rows)")
            continue
        e = encode(ps[:len(gt)])
        r = retrieval(e, gt)
        own = float((e * l2n(gt)).sum(1).mean())
        rows[tag] = {"dir": rel, "n": len(ps), "vs_true": round(own, 4), **r}
        print(f"  {tag:<48} top1={r['top1']:.4f} top5={r['top5']:.4f} "
              f"csls={r['top1_csls']:.4f} vs_true={own:.4f}")

    # A GT-image row is its own target, so it is not a retrieval baseline.  Instead
    # build the control that answers the leakage question directly: how well does a
    # *concept-mean* condition do, i.e. the best an EEG-only decode could hope for?
    print("\n[VERDICT]")
    chance = 1.0 / len(gt)
    verdict = []
    cont = [k for k in rows if k.startswith("CONTAMINATED")]
    clean = [k for k in rows if k.startswith("clean nw3")]
    if cont and clean:
        c, k = rows[cont[0]], rows[clean[0]]
        ratio = c["top1"] / max(k["top1"], 1e-9)
        verdict.append(f"contaminated init top1 {c['top1']:.4f} vs clean {k['top1']:.4f} "
                       f"({ratio:.2f}x)")
        if c["top1"] > k["top1"] + 0.10:
            verdict.append("CONFIRMED: the sdedit_ll_full10 init leaks concept identity "
                           "well beyond the clean EEG-only decode -> hybrid_s08 / "
                           "uck_nat_s08 / ack_s08 scores are NOT a valid bar")
        elif c["top1"] > k["top1"] + 0.03:
            verdict.append("PARTIAL leak: the contaminated init is measurably more "
                           "concept-identifiable; treat those scores as an upper bound")
        else:
            verdict.append("no material leak from the init images; the hybrid_s08 / "
                           "uck_nat_s08 bar stands")
    for k in rows:
        if k.startswith("clean"):
            verdict.append(f"{k}: top1 {rows[k]['top1']:.4f} = "
                           f"{rows[k]['top1']/chance:.1f}x chance "
                           f"(this is the legitimate spatial-pathway signal)")
    for v in verdict:
        print(f"  - {v}")

    rep = {"stage": "nw4_diag_initleak", "chance_top1": round(chance, 5),
           "clip": f"{args.model}/{args.pretrained}", "dirs": rows, "verdict": verdict}
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"\n[diag] wrote {args.out}")
    else:
        print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
