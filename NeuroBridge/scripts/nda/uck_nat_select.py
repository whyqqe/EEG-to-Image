#!/usr/bin/env python3
"""Render-verify selection over K generated candidates (Stage D).

For each sample i, pick
    k* = argmax_k  cos( cond(z_i),  CLIP(generated_image_{i,k}) )

Selection uses a DIFFERENT measurement than the generation posterior
(the rendered image is re-encoded). Writes:
  - selected/000.png .. 199.png
  - selected_idx.npy
  - select_report.json
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dirs", type=str, nargs="+", required=True,
                    help="K dirs each containing 000.png .. 199.png (or generated/ subdir)")
    ap.add_argument("--cond-npy", type=str, required=True,
                    help="(200,1024) EEG-derived SEMANTIC condition used as verifier query")
    ap.add_argument("--struct-npy", type=str, default="",
                    help="optional (200,1024) EEG-derived STRUCTURE condition "
                         "(e.g. predicted depth CLIP) for score-level fusion")
    ap.add_argument("--struct-weight", type=float, default=0.5,
                    help="weight beta on the structure route; 0 disables it")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--clip-model", type=str, default="ViT-H-14")
    ap.add_argument("--clip-pretrained", type=str, default="laion2b_s32b_b79k")
    ap.add_argument("--max-k", type=int, default=0,
                    help="if >0, only use the first max-k gen dirs")
    args = ap.parse_args()

    import torch
    from PIL import Image
    import open_clip
    from tqdm import tqdm

    gen_dirs = []
    for g in args.gen_dirs:
        p = Path(g)
        if (p / "generated").is_dir():
            p = p / "generated"
        gen_dirs.append(p)
    if args.max_k > 0:
        gen_dirs = gen_dirs[: args.max_k]
    K = len(gen_dirs)
    if K < 1:
        raise SystemExit("[FATAL] no gen dirs")

    # verify presence
    for gd in gen_dirs:
        miss = [i for i in range(200) if not (gd / f"{i:03d}.png").is_file()]
        if miss:
            raise SystemExit(f"[FATAL] {gd} missing {len(miss)} pngs e.g. {miss[0]}")

    cond = l2n(np.load(args.cond_npy).astype(np.float32))
    if len(cond) != 200:
        raise SystemExit(f"[FATAL] cond rows {len(cond)} != 200")

    struct = None
    if args.struct_npy and args.struct_weight > 0:
        struct = l2n(np.load(args.struct_npy).astype(np.float32))
        if len(struct) != 200:
            raise SystemExit(f"[FATAL] struct rows {len(struct)} != 200")
        if args.struct_npy == args.cond_npy:
            raise SystemExit("[FATAL] struct-npy equals cond-npy; routes must differ")
        print(f"[INFO] structure route ON: {args.struct_npy} weight={args.struct_weight}")

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained, device=dev)
    model = model.to(dev).eval()

    # encode all K x 200
    PHI = np.zeros((200, K, cond.shape[1]), dtype=np.float32)
    with torch.no_grad():
        for k, gd in enumerate(gen_dirs):
            feats = []
            for i in tqdm(range(200), desc=f"clip gen[{k}]"):
                x = preprocess(Image.open(gd / f"{i:03d}.png").convert("RGB")).unsqueeze(0).to(dev)
                feats.append(model.encode_image(x).float().cpu().numpy()[0])
            PHI[:, k] = l2n(np.stack(feats, 0))

    # scores: cos(cond_i, phi_{i,k})  [+ beta * cos(struct_i, phi_{i,k})]
    scores_sem = np.einsum("nd,nkd->nk", cond, PHI)  # 200 x K
    scores = scores_sem.copy()
    scores_struct = None
    if struct is not None:
        scores_struct = np.einsum("nd,nkd->nk", struct, PHI)
        scores = scores_sem + float(args.struct_weight) * scores_struct

    sel = scores.argmax(axis=1).astype(np.int64)
    margin = scores.max(axis=1) - np.partition(scores, -2, axis=1)[:, -2]
    sel_sem_only = scores_sem.argmax(axis=1).astype(np.int64)

    out = Path(args.out_dir)
    sel_dir = out / "selected"
    sel_dir.mkdir(parents=True, exist_ok=True)
    for i, k in enumerate(sel):
        src = gen_dirs[int(k)] / f"{i:03d}.png"
        shutil.copy2(src, sel_dir / f"{i:03d}.png")

    np.save(out / "selected_idx.npy", sel)
    np.save(out / "scores.npy", scores.astype(np.float32))
    np.save(out / "phi_selected.npy", PHI[np.arange(200), sel])

    # 2-way own-vs-other on selected (quick sanity)
    phi_s = l2n(PHI[np.arange(200), sel])
    sim = cond @ phi_s.T
    own = np.diag(sim)
    # random other
    rng = np.random.default_rng(0)
    other_idx = (np.arange(200) + rng.integers(1, 200, size=200)) % 200
    twoway = float((own > sim[np.arange(200), other_idx]).mean())

    # diversity of selected vs forcing k=0
    agree_k0 = float((sel == 0).mean())
    usage = {int(k): int((sel == k).sum()) for k in range(K)}
    usage_sem = {int(k): int((sel_sem_only == k).sum()) for k in range(K)}

    rep = {
        "pipeline": "uck_nat_select",
        "K": K,
        "gen_dirs": [str(g) for g in gen_dirs],
        "cond": args.cond_npy,
        "struct_cond": args.struct_npy if struct is not None else None,
        "struct_weight": float(args.struct_weight) if struct is not None else 0.0,
        "score_mode": ("semantic+structure" if struct is not None else "semantic"),
        "mean_score_selected": float(scores[np.arange(200), sel].mean()),
        "mean_score_k0": float(scores[:, 0].mean()),
        "mean_score_selected_semantic_only": float(scores_sem[np.arange(200), sel].mean()),
        "mean_margin": float(margin.mean()),
        "frac_agree_k0": agree_k0,
        "frac_fused_agrees_semantic_only": float((sel == sel_sem_only).mean()),
        "usage_per_k": usage,
        "usage_per_k_semantic_only": usage_sem,
        "verifier_2way_own_vs_random": twoway,
        "selected_dir": str(sel_dir),
    }
    if scores_struct is not None:
        rep["mean_score_struct_route"] = float(
            scores_struct[np.arange(200), sel].mean())
        rep["struct_route_2way"] = _twoway(struct, PHI, sel)
    (out / "select_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    np.save(out / "sel_idx_semantic_only.npy", sel_sem_only)
    print(json.dumps(rep, indent=2))
    print(f"[select] wrote {sel_dir}")


def _twoway(query: np.ndarray, PHI: np.ndarray, sel: np.ndarray) -> float:
    """Own-vs-random 2-way on the selected candidate for an arbitrary route."""
    q = l2n(query)
    phi = l2n(PHI[np.arange(len(sel)), sel])
    sim = q @ phi.T
    own = np.diag(sim)
    rng = np.random.default_rng(0)
    other = (np.arange(len(sel)) + rng.integers(1, len(sel), size=len(sel))) % len(sel)
    return float((own > sim[np.arange(len(sel)), other]).mean())


if __name__ == "__main__":
    main()
