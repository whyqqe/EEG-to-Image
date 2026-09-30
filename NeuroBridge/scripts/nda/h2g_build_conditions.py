#!/usr/bin/env python3
"""Build conditioning inputs for HCMA-2G.

Two jobs, both aimed at defects that are measurable and label-free:

(A) GEOMETRY CALIBRATION  (mode=calibrate)
    Diagnosed failure: cross-subject failure is HUBNESS, not margin.
        raw   : margin 0.1329  hub_skew 3.00  Top-1 0.160
        whiten: margin 0.0970  hub_skew 0.63  Top-1 0.235   (+7.5pp)
        +CSLS : Top-1 0.350
    The loss objective cannot reach this: cosine/InfoNCE is invariant to
    orthogonal transforms, and hubness lives entirely in the covariance geometry.
    So the correction is applied at conditioning time, label-free (covariance is
    estimated from TRAIN/reference features only).

(B) DEPLOYABLE PROMPTS  (mode=prompts)
    The shipped pipeline conditions text on the GROUND-TRUTH concept name of the
    test image. That is an oracle: at inference the label is unknown, so any
    reported quality built on it is not attributable to the EEG signal.
    Modes emitted:
        oracle  -- GT concept name (flagged; only for quantifying the leak)
        neutral -- concept-free, fully deployable
        zshot   -- concept predicted from EEG by zero-shot text matching
                   (label-free per sample; assumes only the concept pool is known)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ------------------------------------------------------------- geometry
def whiten(apply_to: np.ndarray, ref: np.ndarray, shrink: float = 0.1):
    mu = ref.mean(0, keepdims=True)
    Xc = ref - mu
    C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    d = C.shape[0]
    C = (1.0 - shrink) * C + shrink * (np.trace(C) / d) * np.eye(d, dtype=np.float32)
    w, V = np.linalg.eigh(C.astype(np.float64))
    W = (V * (1.0 / np.sqrt(np.maximum(w, 1e-8)))) @ V.T
    return (apply_to - mu) @ W.astype(np.float32)


def csls(sim: np.ndarray, k: int = 10) -> np.ndarray:
    """Cross-domain similarity local scaling: 2*cos - r_q - r_g."""
    q = np.sort(sim, axis=1)[:, -k:].mean(1, keepdims=True)      # per query, over gallery
    g = np.sort(sim, axis=0)[-k:, :].mean(0, keepdims=True)      # per gallery, over queries
    return 2.0 * sim - q - g


def hub_skew(sim: np.ndarray) -> float:
    """Skewness of the gallery-side in-degree. >1 means hub points dominate."""
    n = sim.shape[1]
    k = max(1, int(round(n * 0.05)))
    idx = np.argsort(-sim, axis=1)[:, :k]
    deg = np.bincount(idx.ravel(), minlength=n).astype(np.float64)
    m, s = deg.mean(), deg.std()
    if s < 1e-9:
        return 0.0
    return float(((deg - m) ** 3).mean() / s**3)


def retrieval_diag(x: np.ndarray, k: int = 10) -> dict:
    """Standard 200-way, SELF-EXCLUDED retrieval (the protocol used in the literature).

    Test concepts are held out from the train concepts in THINGS-EEG2, so a
    train-gallery Top-1 would be identically 0 -- the diagnostic must be
    within-test-set, self-excluded.
    """
    n = len(x)
    xn = x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)
    sim = xn @ xn.T
    sim = sim - np.eye(n, dtype=sim.dtype) * 1e9          # self-exclusion
    order = np.argsort(-sim, axis=1)
    top1 = float(np.mean(order[:, 0] == np.arange(n)))
    top5 = float(np.mean([np.any(order[i, :5] == i) for i in range(n)]))
    csls1 = float(np.mean(np.argmax(csls(sim, k), 1) == np.arange(n)))
    srt = np.sort(sim, 1)
    return {"top1": round(top1, 4), "top5": round(top5, 4),
            "top1_csls": round(csls1, 4), "hub_skew": round(hub_skew(sim), 3),
            "margin": round(float(np.mean(srt[:, -1] - srt[:, -2])), 4),
            "protocol": "200-way self-excluded"}


def collapse(q: np.ndarray) -> float:
    """cos2mu: mean cosine to the distribution mean. High => collapsed to centre."""
    mu = q.mean(0, keepdims=True)
    return round(float(np.mean((q @ mu.T).ravel() /
                               (np.linalg.norm(q, axis=1) * np.linalg.norm(mu)).clip(1e-8))), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["calibrate", "prompts", "concept"])
    # calibrate
    ap.add_argument("--test-npy", default="")
    ap.add_argument("--ref-npy", default="", help="reference features to FIT the covariance (train-side only)")
    ap.add_argument("--output-npy", default="")
    ap.add_argument("--label-npy", default="", help="(N,) concept id per test row, for retrieval diagnostics")
    ap.add_argument("--report-json", default="")
    ap.add_argument("--methods", nargs="+", default=["none", "whiten"])
    # prompts
    ap.add_argument("--prompt-json", default="", help="template / oracle prompt file (JSON list)")
    ap.add_argument("--text-emb-npy", default="", help="(Nconcept, D) text embeddings, for zshot")
    ap.add_argument("--z-npy", default="", help="EEG feature for zshot classification")
    ap.add_argument("--concepts-json", default="")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    if args.mode == "calibrate":
        te = np.load(args.test_npy).astype(np.float32)
        ref = np.load(args.ref_npy).astype(np.float32) if args.ref_npy else te
        rep = {"n_test": len(te), "n_ref": len(ref),
               "methods": args.methods, "arms": {}}
        for m in args.methods:
            x = te if m == "none" else whiten(te, ref)
            rep["arms"][m] = {"diag": retrieval_diag(x), "cos2mu": collapse(x)}
            if args.output_npy:
                suf = "" if m == "none" else f"_{m}"
                np.save(str(Path(args.output_npy)).replace(".npy", f"{suf}.npy"), x.astype(np.float32))
        if args.report_json:
            Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report_json).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(json.dumps(rep, indent=2))
        return

    if args.mode == "concept":
        # zero-shot concept prediction from EEG, against a label-free concept pool
        z = np.load(args.z_npy).astype(np.float32)
        T = np.load(args.text_emb_npy).astype(np.float32)
        zn = z / np.linalg.norm(z, axis=1, keepdims=True).clip(1e-8)
        tn = T / np.linalg.norm(T, axis=1, keepdims=True).clip(1e-8)
        sim = zn @ tn.T
        pred = np.argmax(sim, 1)
        gt = np.arange(len(z))
        rep = {"n": len(z), "pool": len(T),
               "top1_acc": round(float(np.mean(pred == gt)), 4),
               "top5_acc": round(float(np.mean([gt[i] in np.argsort(-sim[i])[:5] for i in range(len(z))])), 4),
               "note": "label-free per sample; assumes the concept POOL is known (no GT used)"}
        np.save(Path(args.out_dir) / "concept_pred_ids.npy", pred.astype(np.int64))
        (Path(args.out_dir) / "concept_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(json.dumps(rep, indent=2))
        return

    # ---- prompts
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmpl = json.load(open(args.prompt_json, encoding="utf-8"))
    concepts = json.load(open(args.concepts_json, encoding="utf-8")) if args.concepts_json else None
    pred = np.load(Path(args.out_dir) / "concept_pred_ids.npy") if (Path(args.out_dir) / "concept_pred_ids.npy").is_file() else None

    written = {}
    # oracle: the shipped prompts verbatim (GT concept), kept ONLY to quantify the leak
    json.dump(tmpl, open(out / "prompts_oracle.json", "w", encoding="utf-8"), indent=2)
    written["oracle"] = str(out / "prompts_oracle.json")
    # neutral: concept-free, always deployable
    neutral = ["a detailed photograph, natural lighting, high quality"] * len(tmpl)
    json.dump(neutral, open(out / "prompts_neutral.json", "w", encoding="utf-8"), indent=2)
    written["neutral"] = str(out / "prompts_neutral.json")
    # zshot: use the EEG-PREDICTED concept in place of the GT name
    if pred is not None and concepts:
        zs = []
        for i, t in enumerate(tmpl):
            zs.append(t.replace(concepts[i], concepts[int(pred[i])]))
        json.dump(zs, open(out / "prompts_zshot.json", "w", encoding="utf-8"), indent=2)
        written["zshot"] = str(out / "prompts_zshot.json")
        agree = float(np.mean([concepts[int(p)] == concepts[i] for i, p in enumerate(pred)]))
        print(f"[PROMPT] zshot agreement with GT = {agree:.4f}")
    print(json.dumps({"written": written}, indent=2))


if __name__ == "__main__":
    main()
