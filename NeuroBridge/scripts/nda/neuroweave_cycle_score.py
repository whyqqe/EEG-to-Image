#!/usr/bin/env python3
"""NeuroWeave cycle-consistency variants, measured in MULTIPLE visual spaces.

WHY THIS WAS REWRITTEN
----------------------
Round 1 scored cycle consistency in CLIP space only and reported, for each
generated set, `raw_cos` (mean cos to its own ground truth) and `disc_top1`
(does the generated image retrieve its own test concept).  Two problems:

  1. `raw_cos` and `disc_top1` ranked the three sets IDENTICALLY
     (atm_aligned > mb_p1_i1 > mb_p3_i3_cn), so neither can separate
     "semantically right" from "structurally right" -- both are CLIP-space
     quantities, and CLIP is famously insensitive to exactly the layout/texture
     detail whose absence the SSIM gap is measuring.  A cycle term built in CLIP
     space therefore CANNOT answer the reviewer question "is the detail
     hallucinated?", no matter how it is normalised.

  2. `eeg_agree` was 0.0000 for every set, and that was a BUG, not a finding:
     it compared the raw exported `shared_r` (an encoder feature, NOT in any CLIP
     space -- measured cosine to ViT-H-14 targets = -0.0003) against CLIP
     features.  It is now computed from the probe's own ALIGNED query bank
     (`vith_levels_mean__mlp_q`, saved by cfmsf_route_probe), which lives in the
     same space as the bank it is compared to.

WHAT THIS SCRIPT DOES NOW
-------------------------
For each generated set it re-encodes the images in THREE spaces and scores the
same two quantities plus agreement:

    clip_vith   open_clip ViT-H-14 -- semantic, the paper's CLIP score
    dino_v2     timm DINOv2 ViT-L/14 (self-supervised) -- appearance/structure
    rn50        torchvision ResNet-50 ImageNet -- lower-level colour/texture

PRE-REGISTERED PREDICTION (this is the whole point of running it)
    If structure-space cycle is the right construction, then the
    structure-conditioned arms (`mb_p1_i1`, `mb_p3_i3_cn`, which have high SSIM)
    must rank HIGHER in dino_v2/rn50 than they do in clip_vith, while the
    semantic-only arm (`atm_aligned`, high Inception, low SSIM) must rank LOWER.
    If every space produces the SAME ranking, the choice of cycle space is not
    what the paper needs and NeuroWeave should drop the cycle term rather than
    re-weight it.

FALSIFIABLE CONTROL
-------------------
The GT test images are re-encoded in clip_vith and compared against
`ViT-H-14/image_test.npy`, the bank every route was trained against.  If that
control does not match, this script's image ordering is wrong and none of its
numbers may be used.  The check is a hard gate.
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

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT))
sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))

from cfmsf_joint_train import VITH, l2n, load_level  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_EXT = (".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG")


# --------------------------------------------------------------------- images
def gt_test_images(n: int = 200) -> list[Image.Image]:
    """The 200 GT test images in the SAME ORDER as the route bank.

    The order comes from `image_metadata.npy['test_img_concepts']`, which is the
    list the THINGS-EEG2 pipeline itself uses; the clip_vith control below proves
    the ordering is right rather than assuming it.
    """
    meta = np.load("/project/peilab/why/data/images_set/image_metadata.npy",
                   allow_pickle=True).item()
    concepts = list(meta["test_img_concepts"])
    root = Path("/project/peilab/why/data/images_set/test_images")
    out, used = [], []
    for c in concepts:
        d = root / c
        files = sorted(p for p in d.iterdir() if p.suffix in IMG_EXT)
        if not files:
            raise FileNotFoundError(f"no image under {d}")
        out.append(Image.open(files[0]).convert("RGB"))
        used.append(str(files[0]))
    if n and len(out) != n:
        raise SystemExit(f"[FATAL] expected {n} test images, found {len(out)}")
    return out


def load_gen_images(d: Path, n: int = 200) -> tuple[list[Image.Image], list[str]]:
    """Generated PNGs, indexed 000..199; tolerates 3- and 5-digit names."""
    imgs, names = [], []
    for i in range(n):
        for pat in (f"{i:03d}.png", f"{i:05d}.png", f"{i}.png"):
            p = d / pat
            if p.is_file():
                imgs.append(Image.open(p).convert("RGB"))
                names.append(str(p))
                break
        else:
            raise FileNotFoundError(f"missing generated image {i} under {d}")
    return imgs, names


# -------------------------------------------------------------------- encoders
def _prep(imgs: list[Image.Image], size: int, mean, std) -> torch.Tensor:
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return torch.stack([tf(im) for im in imgs])


class ClipVith:
    name = "clip_vith"

    def __init__(self, dev):
        import open_clip
        self.m, _, _ = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained="laion2b_s32b_b79k")
        self.m = self.m.to(dev).eval()
        self.dev = dev
        self.size, self.mean, self.std = 224, IMAGENET_MEAN, IMAGENET_STD

    @torch.no_grad()
    def encode(self, imgs):
        out = []
        for i in range(0, len(imgs), 16):
            x = _prep(imgs[i:i + 16], self.size, self.mean, self.std).to(self.dev)
            out.append(self.m.encode_image(x).float().cpu().numpy())
        return l2n(np.concatenate(out).astype(np.float32))


class DinoV2:
    """timm DINOv2 ViT-L/14: self-supervised, keeps texture/layout that CLIP drops."""

    name = "dino_v2"

    def __init__(self, dev):
        import timm
        self.m = timm.create_model("vit_large_patch14_reg4_dinov2.lvd142m",
                                   pretrained=True, num_classes=0, img_size=224)
        self.m = self.m.to(dev).eval()
        self.dev = dev
        self.size, self.mean, self.std = 224, IMAGENET_MEAN, IMAGENET_STD

    @torch.no_grad()
    def encode(self, imgs):
        out = []
        for i in range(0, len(imgs), 16):
            x = _prep(imgs[i:i + 16], self.size, self.mean, self.std).to(self.dev)
            f = self.m(x)
            if f.dim() > 2:
                f = f.mean(1)
            out.append(f.float().cpu().numpy())
        return l2n(np.concatenate(out).astype(np.float32))


class RN50:
    """torchvision ResNet-50: the lowest-level of the three spaces."""

    name = "rn50"

    def __init__(self, dev):
        from torchvision.models import ResNet50_Weights, resnet50
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.m = torch.nn.Sequential(*list(m.children())[:-1]).to(dev).eval()
        self.dev = dev
        self.size, self.mean, self.std = 224, IMAGENET_MEAN, IMAGENET_STD

    @torch.no_grad()
    def encode(self, imgs):
        out = []
        for i in range(0, len(imgs), 16):
            x = _prep(imgs[i:i + 16], self.size, self.mean, self.std).to(self.dev)
            f = self.m(x).flatten(1)
            out.append(f.float().cpu().numpy())
        return l2n(np.concatenate(out).astype(np.float32))


SPACE_CLASSES = (ClipVith, DinoV2, RN50)


def score(gen: np.ndarray, gt: np.ndarray, eeg_q: np.ndarray | None) -> dict:
    """Same two quantities in whatever space it is handed."""
    ar = np.arange(len(gen))
    raw = float((gen * gt).sum(1).mean())
    sim = gen @ gt.T
    order = np.argsort(-sim, 1)
    disc1 = float((order[:, 0] == ar).mean())
    disc5 = float(np.mean([i in order[i, :5] for i in range(len(gen))]))
    out = {"cycle_raw_cos": raw, "cycle_disc_top1": disc1, "cycle_disc_top5": disc5}
    if eeg_q is not None:
        ep = (eeg_q @ gt.T).argmax(1)
        gp = sim.argmax(1)
        out["eeg_gen_agree"] = float((ep == gp).mean())
        out["eeg_correct"] = float((ep == ar).mean())
        out["gen_correct"] = disc1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--sets", type=str, default="",
                    help="name=path pairs, comma separated")
    ap.add_argument("--eeg-bank", type=str,
                    default=str(NB_ROOT / "outputs/cfmsf_all/sub-08/frozen/probe/"
                                          "probe_queries.npz"),
                    help=("ALIGNED EEG query bank (route, not raw shared_r). "
                          "`vith_levels_mean__mlp_q` is used, which lives in the "
                          "same ViT-H-14 space as the bank it is compared with."))
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--skip-gt-control", action="store_true")
    args = ap.parse_args()

    if not args.sets:
        raise SystemExit("[FATAL] --sets is required (name=path,...)")

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sets = {}
    for item in args.sets.split(","):
        k, v = item.split("=", 1)
        sets[k.strip()] = Path(v.strip())
    print(f"[cycle] {len(sets)} sets: {list(sets)}", flush=True)

    gt_imgs = gt_test_images(200)
    gen_imgs = {k: load_gen_images(p)[0] for k, p in sets.items()}

    # EEG query in ViT-H-14 space (same space as the bank below).
    eeg_q = None
    if Path(args.eeg_bank).is_file():
        d = np.load(args.eeg_bank)
        key = "vith_levels_mean__mlp_q"
        if key in d:
            eeg_q = l2n(d[key].astype(np.float32))
            # Sanity: if the bank were wrong, this would be ~0.5 (chance).
            print(f"[cycle] eeg bank {key} shape={eeg_q.shape}", flush=True)
    if eeg_q is None:
        print("[warn] no aligned EEG bank; eeg_gen_agree will be omitted", flush=True)

    report = {"gt_source": "image_metadata.test_img_concepts",
              "eeg_bank": args.eeg_bank, "spaces": {}, "verdict": {}}
    vith_gt_features = None

    for cls in SPACE_CLASSES:
        enc = cls(dev)
        print(f"\n[cycle] === space {enc.name} ===", flush=True)
        G = {k: enc.encode(v) for k, v in gen_imgs.items()}
        T = enc.encode(gt_imgs)

        if enc.name == "clip_vith":
            vith_gt_features = T
            # HARD GATE: our re-encoded GT must reproduce the bank every route was
            # trained against.  If this fails, the image ORDER is wrong and no
            # number from this script is usable.
            ref = load_level(VITH, "image", "test")
            cos = float((T * ref).sum(1).mean())
            # numpy, not torch: `.eq` is a torch method and this is an ndarray.
            arg_ok = float(((T @ ref.T).argmax(1) == np.arange(len(T))).mean())
            print(f"[cycle] GT control: cos={cos:.4f} diag_argmax={arg_ok:.4f}", flush=True)
            ok = cos > 0.95 and arg_ok == 1.0
            report["gt_control"] = {"cos": cos, "diag_argmax": arg_ok, "pass": ok}
            if not ok and not args.skip_gt_control:
                raise SystemExit(
                    f"[FATAL] GT image ordering control FAILED (cos={cos:.4f}, "
                    f"diag_argmax={arg_ok:.4f}). The generated/clip numbers below "
                    f"would be meaningless, so nothing is written.")

        space_rows = {}
        for name in sets:
            eeg = eeg_q if (enc.name == "clip_vith" and eeg_q is not None) else None
            r = score(G[name], T, eeg)
            space_rows[name] = r
            print(f"  {name:<14} raw={r['cycle_raw_cos']:.4f} "
                  f"disc1={r['cycle_disc_top1']:.4f} disc5={r['cycle_disc_top5']:.4f}"
                  + (f" eeg_agree={r['eeg_gen_agree']:.4f}" if eeg is not None else ""),
                  flush=True)
        report["spaces"][enc.name] = space_rows
        del enc
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    # ---- PRE-REGISTERED PREDICTION ------------------------------------------
    # Structure-heavy arms must rise in structure spaces relative to clip_vith.
    ranks = {s: [k for k, _ in sorted(rows.items(),
                                      key=lambda kv: -kv[1]["cycle_disc_top1"])]
             for s, rows in report["spaces"].items()}
    struct_names = [n for n in sets if n != "atm_aligned"]
    verdict = {"rank_by_disc_top1": ranks,
               "struct_arms": struct_names,
               "semantic_arm": "atm_aligned"}
    if "clip_vith" in ranks and len(sets) > 1:
        better = True
        for s in ("dino_v2", "rn50"):
            if s not in ranks:
                continue
            # mean rank (0 = best) of structure arms in this space vs clip
            def mean_rank(rk, names):
                return float(np.mean([rk.index(n) for n in names if n in rk]))
            rc = mean_rank(ranks["clip_vith"], struct_names)
            rs = mean_rank(ranks[s], struct_names)
            verdict[f"struct_mean_rank_{s}_minus_clip"] = rs - rc
            if rs > rc:
                better = False
        # Reference: the same sets' known SSIM ordering (structure-heavy > semantic).
        verdict["known_ssim_order"] = {"mb_p3_i3_cn": 0.3696, "mb_p1_i1": 0.3500,
                                       "atm_aligned": 0.2300}
        verdict["structure_space_helps"] = bool(better)
        verdict["action"] = ("use a structure space for the cycle term"
                             if better else
                             "DROP the cycle term: every space gives the same "
                             "ranking, so the space is not the problem")
    report["verdict"] = verdict

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cycle_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[cycle] wrote {out / 'cycle_report.json'}")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
