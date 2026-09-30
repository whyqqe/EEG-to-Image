#!/usr/bin/env python3
"""GCC (Geometry-Calibrated Conditioning) — P0/P1 condition builder.

WHY THIS EXISTS
---------------
Two defects were measured offline (no GPU needed) before building this:

  D1  HCMA's flagship retrieval memory *degrades* the IP-Adapter conditioning.
      rag_soft5 alone:  200-way Top-1 = 0.060 / T5 = 0.190 (2-way 0.750)
      z_decode_vith:    200-way Top-1 = 0.350 / T5 = 0.615 (2-way 0.930)
      the blend actually used (alpha=0.5): Top-1 = 0.215  <- -13.5pp for free
      -> alpha was never ablated; alpha=0 was never generated.

  D2  the LOSO (cross-subject) conditioning is chosen without measuring
      retrieval. Of 9 available components the pipeline blends the 5th-best:
        z_s_c / z_s_f   2-way 0.925/0.910, T1 0.160   <- NEVER USED in blends
        blend_nda_cfm_f_a40 (used)  2-way 0.755, T1 0.055
        z_cfm_f                     2-way 0.700, T1 0.015   <- in the blend
      and hubness skewness is 3x higher cross-subject (5.2 vs 1.84).
      Subject-adaptive whitening (SAW) is label-free and gives
        z_s_f  T1 0.160 -> 0.235 (+47% rel), T5 0.450 -> 0.525
        z_cfm_c T1 0.010 -> 0.145 (14.5x), T5 0.060 -> 0.360 (6x)
      consistent with SATTC (CVPR'26): "subject-adaptive whitening is the
      main driver of cross-subject improvements".

  D3  the prompt carries the GROUND-TRUTH concept name (build_hcma_prompts.py
      with concepts_test.json == GT labels, 200/200). That is an oracle upper
      bound, not a deployable protocol, and it MASKS D1/D2: image metrics only
      moved 6pp (alex2) while the real EEG conditioning collapsed 6x.

This script builds, with NO training:
  * P0 intra rows: alpha in {0, .25, .5} x prompt in {oracle, pred, free}
  * P1 LOSO rows : the measured-best conditioning + SAW calibration
and emits a label-free retrieval diagnostic for every condition, so the
generation stage is justified by data instead of hope.

Labels used anywhere? Only inside the two reference protocols:
  prompts/oracle.json = GT concept (upper-bound reference, pre-existing)
  prompts/pred.json   = concept from *label-free* 200-way retrieval over the
                        200 test images (the standard THINGS-EEG2 protocol)
Everything else (embeds, SAW, weights) is label-free.

Outputs under --out-dir:
  embeds/<row>.npy            IP-Adapter conditioning (200,D), l2-normalized
  prompts/{oracle,pred,free}.json
  rows.tsv                    tag \t mode \t embed \t prompts \t cn \t strength
  conditions_manifest.json    rows for generate_* + eval_standard7
  retrieval_diag.json         2-way / Top-1 / Top-5 / hubness per condition
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------- scene rule
# copied verbatim from build_hcma_prompts.py so the pred-prompt template is
# identical to the oracle template and only the *concept string* differs.
SCENE_RULES = [
    (("ocean", "boat", "ship", "carrier", "fish", "seal", "whale", "calamari", "beaver"), "on water"),
    (("bench", "bike", "unicycle", "cart", "buggy", "road", "car"), "outdoors"),
    (("cake", "bread", "cheese", "sausage", "banana", "cashew", "bok choy", "basil", "bun"), "on a table"),
    (("cat", "dog", "cheetah", "antelope", "bug", "caterpillar", "grasshopper", "bat"), "in a natural setting"),
    (("basketball", "baseball", "balance beam", "baton"), "in a sports setting"),
]


def scene_for(concept: str) -> str:
    c = concept.lower()
    for keys, scene in SCENE_RULES:
        if any(k in c for k in keys):
            return scene
    return "in a clean studio setting"


def full_prompt(concept: str) -> str:
    return (
        f"a photo of a {concept}, clearly showing its shape, color, and "
        f"distinctive parts, {scene_for(concept)}, natural lighting"
    )


FREE_PROMPT = "a photo of an object, clearly showing its shape, color, and distinctive parts, natural lighting"


# ------------------------------------------------------------------- numerics
def l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)).astype(np.float32)


def whiten(z: np.ndarray, shrink: float = 0.1) -> np.ndarray:
    """Subject-adaptive whitening (SAW). shrink=0.1 keeps it stable at n=200,d=1024."""
    zc = z - z.mean(0, keepdims=True)
    cov = np.cov(zc, rowvar=False)
    d = cov.shape[0]
    cov = (1.0 - shrink) * cov + shrink * (np.trace(cov) / d) * np.eye(d)
    w, v = np.linalg.eigh(cov)
    w = np.clip(w, 1e-6, None)
    return (zc @ (v @ np.diag(w ** -0.5) @ v.T)).astype(np.float32)


def blend_rag(rag: np.ndarray, prior: np.ndarray, alpha: float) -> np.ndarray:
    """Same convention as ensemble_embeds.blend(): alpha = weight on RAG."""
    return l2(alpha * l2(rag) + (1.0 - alpha) * l2(prior))


# ------------------------------------------------------------------ retrieval
def topk_acc(sim: np.ndarray, k: int) -> float:
    idx = np.argsort(-sim, axis=1)[:, :k]
    return float((idx == np.arange(sim.shape[0])[:, None]).any(1).mean())


def two_way(sim: np.ndarray, seed: int = 0) -> float:
    n = sim.shape[0]
    rng = np.random.RandomState(seed)
    win = 0
    for i in range(n):
        j = rng.randint(n - 1)
        j = j if j < i else j + 1
        win += int(sim[i, i] > sim[i, j])
    return win / n


def hub_skew(sim: np.ndarray, k: int = 5) -> float:
    idx = np.argsort(-sim, axis=1)[:, :k]
    cnt = np.bincount(idx.ravel(), minlength=sim.shape[0]).astype(np.float64)
    cnt /= max(cnt.mean(), 1e-8)
    m, s = cnt.mean(), cnt.std()
    return float(((cnt - m) ** 3).mean() / max(s ** 3, 1e-8))


def diagnose(z: np.ndarray, gal: np.ndarray) -> dict:
    sim = l2(z) @ gal.T
    return {
        "two_way": round(two_way(sim), 4),
        "top1": round(topk_acc(sim, 1), 4),
        "top5": round(topk_acc(sim, 5), 4),
        "hub_skew5": round(hub_skew(sim), 3),
    }


# ----------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intra-root", required=True, help="outputs/intra_hcma_s/sub-08")
    ap.add_argument("--loso-embeds", required=True, help=".../inter_ll_full10/sub-08/inter_embeds/embeds")
    ap.add_argument("--gallery-clip", required=True, help="(200,1024) ViT-H CLIP of the 200 test images")
    ap.add_argument("--concept-phrases", required=True, help="(200,) GT concept names (labels: oracle ref only)")
    ap.add_argument("--oracle-prompts", required=True, help="pre-existing GT-prompt json (upper-bound ref)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cn-scale", type=float, default=0.25, help="best measured HCMA-S control")
    ap.add_argument("--strength", type=float, default=0.86, help="best measured HCMA-S strength")
    ap.add_argument("--shrink", type=float, default=0.1, help="SAW shrinkage")
    args = ap.parse_args()

    # resolve() so every emitted path is absolute: the generator/eval stages may
    # run from any cwd and a relative path would silently break them.
    out = Path(args.out_dir).resolve()
    (out / "embeds").mkdir(parents=True, exist_ok=True)
    (out / "prompts").mkdir(parents=True, exist_ok=True)

    intra = Path(args.intra_root).resolve()
    loso = Path(args.loso_embeds).resolve()

    gal = l2(np.load(args.gallery_clip))
    phrases = [str(x) for x in json.loads(Path(args.concept_phrases).read_text())]
    assert gal.shape[0] == len(phrases), (gal.shape, len(phrases))

    # ---------------- load intra components ----------------
    rag = l2(np.load(intra / "memory" / "rag_soft5_test_clip_1024.npy"))
    vith = l2(np.load(intra / "train" / "z_decode_vith_test.npy"))

    # ---------------- load LOSO components ----------------
    z_s_f = l2(np.load(loso / "z_s_f_test.npy"))
    z_s_c = l2(np.load(loso / "z_s_c_test.npy"))
    loso_a40 = l2(np.load(loso / "blend_nda_cfm_f_a40_test.npy"))

    # ---------------- prompts ----------------
    oracle_prompts = [str(x) for x in json.loads(Path(args.oracle_prompts).read_text())]
    assert len(oracle_prompts) == len(phrases), (len(oracle_prompts), len(phrases))

    # label-free predicted concepts: 200-way retrieval over the 200 test images,
    # using the strongest *available* EEG semantic embedding for each protocol.
    def pred_names(z: np.ndarray) -> tuple[list[str], float]:
        idx = (l2(z) @ gal.T).argmax(1)
        names = [phrases[j] for j in idx]
        acc = float(np.mean([names[i] == phrases[i] for i in range(len(phrases))]))
        return names, acc

    intra_names, intra_acc = pred_names(vith)      # best intra retrieval (T1 0.350)
    loso_names, loso_acc = pred_names(z_s_f)       # best LOSO retrieval (T1 0.160)

    prompt_sets = {
        "oracle": oracle_prompts,
        "pred_intra": [full_prompt(n) for n in intra_names],
        "pred_loso": [full_prompt(n) for n in loso_names],
        "free": [FREE_PROMPT] * len(phrases),
    }
    for k, v in prompt_sets.items():
        (out / "prompts" / f"{k}.json").write_text(json.dumps(v, indent=2), encoding="utf-8")

    # ---------------- build conditions ----------------
    conds: dict[str, np.ndarray] = {}
    diag: dict[str, dict] = {}

    # ---- P0 intra: alpha ablation x prompt protocol ----
    for a in (0.0, 0.25, 0.5):
        conds[f"p0_a{int(a*100):02d}"] = blend_rag(rag, vith, a)

    # ---- P1 LOSO: measured-best conditioning + SAW ----
    conds["p1_raw_a40"] = loso_a40
    conds["p1_sf"] = z_s_f
    conds["p1_sf_saw50"] = l2(0.5 * z_s_f + 0.5 * l2(whiten(z_s_f, args.shrink)))
    conds["p1_sf_saw100"] = l2(whiten(z_s_f, args.shrink))
    conds["p1_sc_saw50"] = l2(0.5 * z_s_c + 0.5 * l2(whiten(z_s_c, args.shrink)))

    for name, z in conds.items():
        assert z.shape == (len(phrases), 1024), (name, z.shape)
        np.save(out / "embeds" / f"{name}.npy", z.astype(np.float32))
        diag[name] = diagnose(z, gal)

    # ---------------- rows ----------------
    rows = []

    def add(tag: str, cond: str, prompts: str, structural: str) -> None:
        rows.append(
            {
                "tag": tag,
                "cond": cond,
                "embed": str(out / "embeds" / f"{cond}.npy"),
                "prompts": str(out / "prompts" / f"{prompts}.json"),
                "prompt_protocol": prompts,
                "structural": structural,
                "cn": args.cn_scale,
                "strength": args.strength,
                "gen_dir": str(out / "generation" / tag),
                "retrieval": diag[cond],
            }
        )

    # P0 — intra structure heads
    add("gcc_p0_or_a00", "p0_a00", "oracle", "intra")
    add("gcc_p0_or_a25", "p0_a25", "oracle", "intra")
    add("gcc_p0_pr_a50", "p0_a50", "pred_intra", "intra")
    add("gcc_p0_pr_a25", "p0_a25", "pred_intra", "intra")
    add("gcc_p0_pr_a00", "p0_a00", "pred_intra", "intra")
    add("gcc_p0_pf_a50", "p0_a50", "free", "intra")
    add("gcc_p0_pf_a00", "p0_a00", "free", "intra")

    # P1 — intra structure heads reused (controlled: only semantics change)
    add("gcc_p1_raw_a40", "p1_raw_a40", "pred_loso", "intra")
    add("gcc_p1_sf", "p1_sf", "pred_loso", "intra")
    add("gcc_p1_sf_saw50", "p1_sf_saw50", "pred_loso", "intra")
    add("gcc_p1_sf_saw100", "p1_sf_saw100", "pred_loso", "intra")
    add("gcc_p1_sc_saw50", "p1_sc_saw50", "pred_loso", "intra")

    (out / "conditions_manifest.json").write_text(
        json.dumps(
            {
                "rows": rows,
                "n": len(phrases),
                "prompt_concept_accuracy_vs_gt": {
                    "intra_pred": intra_acc,
                    "loso_pred": loso_acc,
                },
                "note": "retrieval fields are label-free 200-way diagnostics (chance 0.005)",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with (out / "rows.tsv").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(
                "\t".join(
                    [r["tag"], r["embed"], r["prompts"], f"{r['cn']}", f"{r['strength']}", r["gen_dir"]]
                )
                + "\n"
            )

    # ---------------- report ----------------
    ref = conds["p0_a00"]
    print("\n=== GCC conditioning diagnostics (label-free 200-way, chance 0.005) ===")
    print(f"{'condition':<18}{'2-way':>8}{'Top-1':>8}{'Top-5':>8}{'hub_skew':>10}")
    for name, d in sorted(diag.items(), key=lambda kv: -kv[1]["top1"]):
        print(f"{name:<18}{d['two_way']:>8.3f}{d['top1']:>8.3f}{d['top5']:>8.3f}{d['hub_skew5']:>10.2f}")
    print(f"\nprompt concept accuracy vs GT (label-free retrieval):")
    print(f"  intra (z_decode_vith Top-1) = {intra_acc:.3f}")
    print(f"  loso  (z_s_f         Top-1) = {loso_acc:.3f}")
    print(f"  oracle (GT)                 = 1.000  <- reference only, NOT deployable")
    print(f"\n[OK] {len(rows)} rows -> {out/'rows.tsv'}")

    Path(out / "retrieval_diag.json").write_text(
        json.dumps(
            {
                "conditions": diag,
                "prompt_concept_accuracy_vs_gt": {"intra_pred": intra_acc, "loso_pred": loso_acc, "oracle": 1.0},
                "reference_embed": "p0_a00 (= z_decode_vith, no RAG)",
                "sanity": {
                    "p0_a00": diag["p0_a00"],
                    "p0_a50": diag["p0_a50"],
                    "delta_top1_a00_minus_a50": round(diag["p0_a00"]["top1"] - diag["p0_a50"]["top1"], 4),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _ = ref


if __name__ == "__main__":
    main()
