#!/usr/bin/env python3
"""SHB step 4: image-space verification / discrete hypothesis selection.

Mechanism 3 of the Structural Hypothesis Branch
-----------------------------------------------
The scalar ControlNet gate could only interpolate a fixed structure<->semantics
tradeoff curve (proven: even an ORACLE gate failed to move the frontier). What
it lacked was a DISCRETE choice over *content*. Here we generate several
structural hypotheses and pick one per sample using IMAGE-SIDE evidence only
(no ground-truth image is ever consulted):

  sem   : cos( CLIP_image(candidate), EEG->IP embed )        [semantic agreement]
  geo   : corr( DepthAnything(candidate), EEG geometry )     [realises EEG structure]
  cons  : mean depth-agreement with the other candidates     [robust medoid]

Composed rows are index-aligned symlink trees, so evaluating them costs the same
as evaluating any other row (no diffusion re-runs).

Outputs: generation/<tag>/generated + manifest_shb_sel.json + selection_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))


def l2(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-8)).astype(np.float32)


def zn(x: np.ndarray) -> np.ndarray:
    return ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32)


def pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.reshape(len(a), -1).astype(np.float64)
    b = b.reshape(len(b), -1).astype(np.float64)
    ac, bc = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    den = np.sqrt((ac * ac).sum(1) * (bc * bc).sum(1)).clip(1e-12)
    return (ac * bc).sum(1) / den


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=str, required=True,
                    help="comma list of 'tag=gen_dir' entries")
    ap.add_argument("--ip-embed", type=str, required=True, help="EEG->IP semantic embed (200,1024)")
    ap.add_argument("--geo-ref", type=str, required=True, help="EEG geometry ref map (200,R,R) from SHB")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--depth-low-res", type=int, default=64)
    ap.add_argument("--gen-size", type=int, default=512)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cands = []
    for item in args.candidates.split(","):
        item = item.strip()
        if not item:
            continue
        tag, gdir = item.split("=", 1)
        gdir = Path(gdir)
        if not (gdir / "199.png").is_file():
            print(f"[WARN] skip {tag}: missing {gdir}/199.png")
            continue
        cands.append((tag, gdir))
    if not cands:
        raise SystemExit("[FATAL] no usable candidates")
    n = 200
    print(f"[INFO] {len(cands)} candidates: {[c[0] for c in cands]}")

    emb = l2(np.load(args.ip_embed).astype(np.float32))
    geo_ref = np.load(args.geo_ref).astype(np.float32)          # (200,R,R)
    assert len(emb) == n and len(geo_ref) == n

    # ---------- semantic: CLIP ViT-H/14 (same space as the IP embed) ----------
    from eval_standard7 import Encoders  # type: ignore
    cache = out / "cache"
    enc = Encoders(device, cache)
    sem = np.zeros((len(cands), n), dtype=np.float32)
    for ci, (tag, gdir) in enumerate(cands):
        feats = enc.clip_features([gdir / f"{i:03d}.png" for i in range(n)])
        sem[ci] = (l2(feats) * emb).sum(axis=1)
        print(f"[OK] clip {tag}: mean cos={sem[ci].mean():.4f}")

    # ---------- structure: DepthAnything on each candidate ----------
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    from PIL import Image
    mid = "depth-anything/Depth-Anything-V2-Small-hf"
    processor = AutoImageProcessor.from_pretrained(mid)
    dmodel = AutoModelForDepthEstimation.from_pretrained(mid).to(device).eval()
    R = args.depth_low_res
    depth = np.zeros((len(cands), n, R, R), dtype=np.float32)
    with torch.no_grad():
        for ci, (tag, gdir) in enumerate(cands):
            for s in tqdm(range(0, n, args.batch_size), desc=f"depth[{tag}]", leave=False):
                ps = [gdir / f"{i:03d}.png" for i in range(s, min(s + args.batch_size, n))]
                imgs = [Image.open(p).convert("RGB") for p in ps]
                inp = processor(images=imgs, return_tensors="pt")
                inp = {k: v.to(device) for k, v in inp.items()}
                pf = dmodel(**inp).predicted_depth
                p = torch.nn.functional.interpolate(
                    pf.unsqueeze(1), size=(R, R), mode="bicubic", align_corners=False).squeeze(1)
                for j in range(p.shape[0]):
                    d = p[j].float().cpu().numpy()
                    depth[ci, s + j] = (d - d.min()) / (d.max() - d.min() + 1e-8)
    del dmodel
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # agreement with the EEG-implied geometry (resized to the candidate depth resolution)
    if geo_ref.shape[-1] != R:
        ref = torch.nn.functional.interpolate(
            torch.from_numpy(geo_ref)[:, None], size=(R, R), mode="bilinear", align_corners=False
        ).squeeze(1).numpy().astype(np.float32)
    else:
        ref = geo_ref
    geo = np.stack([pearson_rows(depth[ci], ref) for ci in range(len(cands))], axis=0)

    # consensus (medoid-like): mean agreement with the other candidates
    cons = np.zeros_like(geo)
    for ci in range(len(cands)):
        vals = [pearson_rows(depth[ci], depth[cj]) for cj in range(len(cands)) if cj != ci]
        cons[ci] = np.mean(np.stack(vals, 0), axis=0)

    np.save(out / "scores_sem.npy", sem)
    np.save(out / "scores_geo.npy", geo)
    np.save(out / "scores_cons.npy", cons)

    # ---------- policies + composition ----------
    policies = {
        "shb_sel_sem_c040_s086": zn(sem),
        "shb_sel_geo_c040_s086": 0.5 * zn(sem) + 0.5 * zn(geo),
        "shb_sel_cons_c040_s086": 0.5 * zn(sem) + 0.5 * zn(cons),
    }
    man = {"protocol": "standard7", "rows": [], "avg_rows": []}
    hist = {}
    for tag, score in policies.items():
        pick = score.argmax(axis=0)
        dst = out / "generation" / tag / "generated"
        dst.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            src = cands[int(pick[i])][1] / f"{i:03d}.png"
            link = dst / f"{i:03d}.png"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(src.resolve())
        counts = {cands[k][0]: int((pick == k).sum()) for k in range(len(cands))}
        hist[tag] = counts
        man["rows"].append({"tag": tag, "display": tag + " (SHB verification selection)",
                            "gen_dir": str(dst)})
        print(f"[row {tag}] picks={counts}")

    (out / "manifest_shb_sel.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    report = {
        "pipeline": "shb_select",
        "candidates": [c[0] for c in cands],
        "score_means": {
            "sem": {cands[ci][0]: float(sem[ci].mean()) for ci in range(len(cands))},
            "geo": {cands[ci][0]: float(geo[ci].mean()) for ci in range(len(cands))},
            "cons": {cands[ci][0]: float(cons[ci].mean()) for ci in range(len(cands))},
        },
        "selection_histograms": hist,
        "note": "image-side evidence only; no GT image used. sem=IP-embed agreement, "
                "geo=DepthAnything vs EEG geometry, cons=agreement with other candidates.",
    }
    (out / "selection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
