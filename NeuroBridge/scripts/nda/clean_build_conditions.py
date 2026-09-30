#!/usr/bin/env python3
"""Build LEAK-FREE conditions + prompts for the HCMA decode stage.

THE PROTOCOL BUG THIS FIXES (audit finding P2)
---------------------------------------------
The shipped prompt files are e.g. "a photo of aircraft carrier, highly detailed,
natural lighting, true-to-class appearance" in the order the 200 TEST concepts
appear alphabetically. Those ARE the test-set concept labels, which the model
has never been trained on. Every shipped number therefore measures an ORACLE
protocol: the diffusion model is told the correct answer in text.

Can EEG supply that text instead? No. THINGS-EEG2 test concepts are disjoint
from the 1654 train concepts, so "predict the concept name from EEG" is not
defined for a test concept without its label. The honest deployable protocols
therefore are:

    free     : no text prompt at all        -- IP-adapter embeddings only
    neutral  : concept-free but flowery text -- isolates "does text help at all"
                                                   from "does the RIGHT concept help"
    oracle   : the GT concept name           -- REFERENCE ONLY, never a headline

The gap (neutral -> oracle) is the size of the leakage. It is reported, not hidden.

ALPHA ABLATION (audit finding: RAG degrades semantics)
-----------------------------------------------------
Measured 200-way: alpha=0 gives Top-1 0.350 while the shipped alpha=0.5 gives 0.215,
i.e. the RAG memory blend costs 13.5pp of retrieval accuracy and was never ablated.

Usage:
  python clean_build_conditions.py --intra-root ... --gallery-clip ... \
      --oracle-prompts ... --out-dir ... [--stage intra|inter] [--align-dir ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FREE_PROMPT = ""
NEUTRAL_PROMPT = "a photo, highly detailed, natural lighting"


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)


def diagnose(z: np.ndarray, gal: np.ndarray, k: int = 5) -> dict:
    """Label-free 200-way retrieval diagnostics (chance 0.005)."""
    Fn, Gn = l2(z), l2(gal)
    S = Fn @ Gn.T
    n = len(Fn)
    order = np.argsort(-S, axis=1)
    tgt = np.arange(n)[:, None]
    hit = order == tgt
    d = np.sum(S * np.eye(n), 1)
    off = (S.sum() - np.trace(S)) / (n * (n - 1))
    Pn = Fn @ Fn.T
    spread = (Pn.sum() - np.trace(Pn)) / (n * (n - 1))
    idx5 = order[:, :k]
    cnt = np.bincount(idx5.reshape(-1), minlength=n).astype(float)
    sd = cnt.std() + 1e-8
    return {
        "top1": round(float(hit[:, :1].any(1).mean()), 4),
        "top5": round(float(hit[:, :k].any(1).mean()), 4),
        "margin": round(float(d.mean() - off), 4),
        "spread": round(float(spread), 4),
        "hub_skew": round(float(((cnt - cnt.mean()) ** 3).mean() / sd**3), 3),
    }


def blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return l2((1.0 - alpha) * l2(a) + alpha * l2(b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intra-root", required=True, help="clean intra outputs, has train/ + memory/")
    ap.add_argument("--gallery-clip", required=True, help="(200,1024) ViT-H CLIP of the 200 test images")
    ap.add_argument("--oracle-prompts", required=True, help="existing GT-concept prompt json (reference only)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--align-dir", default="", help="flow_align.py output dir (stage=inter)")
    ap.add_argument("--stage", choices=["intra", "inter", "both"], default="intra")
    ap.add_argument("--cn-scale", type=float, default=0.32)
    ap.add_argument("--strength", type=float, default=0.86)
    ap.add_argument("--alphas", default="0.0,0.25,0.5")
    args = ap.parse_args()

    out = Path(args.out_dir).resolve()
    (out / "embeds").mkdir(parents=True, exist_ok=True)
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    intra = Path(args.intra_root).resolve()

    gal = l2(np.load(args.gallery_clip).astype(np.float32))
    n_samples = len(gal)
    oracle = [str(x) for x in json.loads(Path(args.oracle_prompts).read_text(encoding="utf-8"))]
    assert len(oracle) == n_samples, (len(oracle), n_samples)

    # ---------------- prompts (deployable vs reference) ----------------
    prompt_sets = {
        "free": [FREE_PROMPT] * n_samples,
        "neutral": [NEUTRAL_PROMPT] * n_samples,
        "oracle": oracle,  # REFERENCE ONLY (contains test labels)
    }
    for k, v in prompt_sets.items():
        (out / "prompts" / f"{k}.json").write_text(json.dumps(v, indent=2), encoding="utf-8")

    # ---------------- conditions ----------------
    z_vith = l2(np.load(intra / "train" / "z_decode_vith_test.npy").astype(np.float32))
    rag_p = intra / "memory" / "rag_soft5_test_clip_1024.npy"
    rag = l2(np.load(rag_p).astype(np.float32)) if rag_p.is_file() else None

    conds: dict[str, np.ndarray] = {}
    for a_ in args.alphas.split(","):
        al = float(a_)
        ai = int(round(al * 100))
        if al == 0.0 or rag is None:
            conds[f"a{ai:02d}"] = z_vith
        else:
            conds[f"a{ai:02d}"] = blend(rag, z_vith, al)

    # ---------------- alignment variants (stage=inter) ----------------
    align_variants: list[str] = []
    if args.stage in ("inter", "both"):
        adir = Path(args.align_dir).resolve() if args.align_dir else None
        if adir and (adir / "features").is_dir():
            zsf = l2(np.load(intra / "train" / "z_decode_vith_test.npy").astype(np.float32))
            for f in sorted((adir / "features").glob("*.npy")):
                z = l2(np.load(f).astype(np.float32))
                if z.shape != (n_samples, 1024):
                    print(f"[WARN] skip {f.name}: shape {z.shape}")
                    continue
                conds[f"al_{f.stem}"] = z
                align_variants.append(f.stem)

    # ---------------- write features + diagnostics ----------------
    diag: dict[str, dict] = {}
    for name, z in conds.items():
        assert z.shape == (n_samples, 1024), (name, z.shape)
        np.save(out / "embeds" / f"{name}.npy", z.astype(np.float32))
        diag[name] = diagnose(z, gal)

    # ---------------- rows ----------------
    rows: list[dict] = []

    def add(tag: str, cond: str, prompt: str, note: str, deployable: bool,
            structural: str = "clean") -> None:
        rows.append(
            {
                "tag": tag, "cond": cond, "embed": str(out / "embeds" / f"{cond}.npy"),
                "prompts": str(out / "prompts" / f"{prompt}.json"), "prompt_protocol": prompt,
                "structural": structural, "cn": args.cn_scale, "strength": args.strength,
                "gen_dir": str(out / "generation" / tag),
                "deployable": deployable, "note": note,
                "retrieval": diag[cond],
            }
        )

    if args.stage in ("intra", "both"):
        # P0: alpha ablation x deployable prompt protocol
        for ai in (0, 25, 50):
            for p in ("free", "neutral"):
                add(f"cl_a{ai:02d}_{p}", f"a{ai:02d}", p,
                    f"intra alpha={ai/100:.2f}, deployable prompt", True)
        # reference rows (flagged, not headline results)
        add("cl_ref_oracle_a00", "a00", "oracle", "ORACLE text (test labels) — reference only", False)
        add("cl_ref_oracle_a50", "a50", "oracle", "ORACLE text (test labels) — reference only", False)

    if args.stage in ("inter", "both"):
        for v in align_variants:
            for p in ("free", "neutral"):
                add(f"cl_inter_{v}_{p}", f"al_{v}", p,
                    f"inter LOSO alignment={v}, deployable prompt", True)

    (out / "conditions_manifest.json").write_text(
        json.dumps(
            {
                "stage": args.stage,
                "rows": rows,
                "n": n_samples,
                "prompt_protocols": {
                    "free": "no text (deployable)",
                    "neutral": "concept-free text (deployable)",
                    "oracle": "GT concept name — REFERENCE ONLY, contains test labels",
                },
                "leak_audit": {
                    "oracle_rows_are_not_results": True,
                    "gt_concepts_used_as_prompt": "oracle protocol only (flagged)",
                    "rag_alphas_tested": [float(x) for x in args.alphas.split(",")],
                    "note": "deployable rows use no test label and no test-set concept vocabulary",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (out / "rows.tsv").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write("\t".join([r["tag"], r["embed"], r["prompts"], str(r["cn"]),
                               str(r["strength"]), r["gen_dir"]]) + "\n")

    print(f"\n=== condition diagnostics (label-free 200-way, chance {1/n_samples:.3f}) ===")
    print(f"{'cond':<14}{'Top-1':>8}{'Top-5':>8}{'margin':>9}{'spread':>9}{'hub_skew':>10}")
    for name, d in sorted(diag.items(), key=lambda kv: -kv[1]["top1"]):
        print(f"{name:<14}{d['top1']:>8.3f}{d['top5']:>8.3f}{d['margin']:>9.4f}"
              f"{d['spread']:>9.4f}{d['hub_skew']:>10.2f}")
    print(f"\n[OK] {len(rows)} generation rows -> {out/'rows.tsv'}")
    n_ref = sum(1 for r in rows if not r["deployable"])
    print(f"     {len(rows)-n_ref} deployable rows, {n_ref} flagged reference rows (oracle text)")

    (out / "retrieval_diag.json").write_text(
        json.dumps({"conditions": diag, "n": n_samples}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
