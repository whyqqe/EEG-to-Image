#!/usr/bin/env python3
"""Follow-up to the FAILED attribute gate.

WHAT THE GATE SHOWED
--------------------
    24 antonym attribute axes, kernel regression over train concepts:
        top-1 0.005  2-way 0.595  (cos to centroid 0.974 -> it collapsed to the mean)
    nearest TRAIN CONCEPT from an oracle query:
        top-1 0.425  2-way 0.930
So the concept gallery is not the problem -- its ceiling is high.  The attribute
code is the problem: 24 axes at std 0.07 cannot resolve 1654 concepts, so the
kernel degenerates to the dataset mean.  The attribute-prompt idea as designed is
DEAD, and this file does not try to revive it.

THE QUESTION THIS FILE ANSWERS
------------------------------
If retrieval over train concepts has a ceiling of top-1 0.425, how far is the
actual EEG query from that ceiling?  That decomposes the bottleneck cleanly:

    ceiling (oracle query -> nearest train concept)        0.425
    EEG query   -> nearest train concept                   ???   <- this file
    EEG query   -> direct regression (no retrieval)        0.245-0.290 (measured)

and it decides between two architectures:
  * if the EEG-retrieval number is close to 0.29 and far from 0.425, retrieval
    adds nothing and the direct regressor is already the right answer;
  * if retrieval beats the direct regressor, then a better QUERY (not a better
    gallery, not a bigger prompt vocabulary) is the lever worth building.

QUERY QUALITY IS A WEAKER REQUIREMENT THAN NAMING
-------------------------------------------------
"Name the test concept" needs log2(200)=7.64 bits and EEG has ~1.5.  "Land near
the right train concept" needs only that the nearest neighbour be SEMANTICALLY
close -- a much weaker requirement, and exactly what the oracle row above prices.

TRAIN-ONLY
    the ridge is fit on train rows, the gallery is train concepts, the per-concept
    means are train images.  The test EEG is read once, for scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NB_ROOT = Path(__file__).resolve().parents[2]


def l2n(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x).astype(np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def ranks(pred: np.ndarray, targ: np.ndarray) -> dict:
    p, t = l2n(pred), l2n(targ)
    s = p @ t.T
    n = len(s)
    idx = np.arange(n)
    perm = np.random.default_rng(0).permutation(n)
    ok = idx != perm
    return {"top1": float(np.mean(s.argmax(1) == idx)),
            "top5": float(np.mean([idx[i] in np.argsort(-s[i])[:5] for i in range(n)])),
            "twoway": float(np.mean(s[idx, idx][ok] > s[idx, perm][ok])),
            "cos_to_target": float((p * t).sum(1).mean())}


def ridge_fit(X, Y, lam):
    X = X.astype(np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    W = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ Y.astype(np.float64))
    return W.astype(np.float32), mu.astype(np.float32), sd.astype(np.float32)


def ridge_apply(m, X):
    W, mu, sd = m
    return ((X.astype(np.float32) - mu) / sd) @ W


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--z-source", type=str, default="shared_r")
    ap.add_argument("--z-cache-root", type=str, default=f"{NB_ROOT}/outputs/ocf/ss_parts")
    ap.add_argument("--clip-text-dir", type=str, default=f"{NB_ROOT}/outputs/nda_ss/sub-08/clip_text")
    ap.add_argument("--captions-jsonl", type=str,
                    default=f"{NB_ROOT}/outputs/g2/captions/captions_train.jsonl")
    ap.add_argument("--img-train", type=str,
                    default="/project/peilab/why/eeg-brainit/outputs/atm_bridge/clip_img_train_1024.npy")
    ap.add_argument("--img-test", type=str,
                    default="/project/peilab/why/eeg-brainit/outputs/atm_bridge/clip_img_test_1024.npy")
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--out-json", type=str, default=f"{NB_ROOT}/outputs/idg/retrieval_probe.json")
    args = ap.parse_args()

    ct = Path(args.clip_text_dir)
    tr_phr = json.loads((ct / "train" / "concept_phrases.json").read_text(encoding="utf-8"))
    tr_txt = l2n(np.load(ct / "train" / "text_concept_clip.npy").astype(np.float32))
    Ytr = np.load(args.img_train).astype(np.float32)
    Yte = np.load(args.img_test).astype(np.float32)

    Ztr = np.load(Path(args.z_cache_root) / f"sub-{args.subject:02d}" / f"{args.z_source}_train.npy")
    Zte = np.load(Path(args.z_cache_root) / f"sub-{args.subject:02d}" / f"{args.z_source}_test.npy")
    if Ztr.shape[0] != Ytr.shape[0]:
        raise SystemExit(f"[FATAL] z rows {Ztr.shape[0]} != image rows {Ytr.shape[0]}")

    lab_dir = [json.loads(l)["path"].rsplit("/", 1)[0].rsplit("/", 1)[-1]
               for l in Path(args.captions_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    concepts = [d.split("_", 1)[1].replace("_", " ") for d in lab_dir]
    index = {c: i for i, c in enumerate(tr_phr)}
    ci = np.asarray([index[c] for c in concepts])
    Ybar = np.zeros((len(tr_phr), Ytr.shape[1]), dtype=np.float32)
    cnt = np.zeros(len(tr_phr), dtype=np.int64)
    np.add.at(Ybar, ci, Ytr)
    np.add.at(cnt, ci, 1)
    Ybar /= np.clip(cnt, 1, None)[:, None]
    print(f"[retr] z {Ztr.shape} -> gallery {tr_txt.shape} ({int((cnt>0).sum())} concepts with images)")

    m = ridge_fit(Ztr, Ytr, args.lam)
    q = l2n(ridge_apply(m, Zte))          # the EEG query, in CLIP image space

    res: dict[str, dict] = {}
    res["direct_regression"] = ranks(q, Yte)           # no retrieval at all

    # --- retrieval over TRAIN CONCEPTS (text space), then that concept's image mean
    st = q @ tr_txt.T
    res["retr_concept_hard"] = ranks(Ybar[st.argmax(1)], Yte)
    for tau in (0.02, 0.05, 0.1):
        w = np.exp((st / tau) - (st / tau).max(1, keepdims=True))
        w /= w.sum(1, keepdims=True)
        res[f"retr_concept_soft_tau{tau}"] = ranks(w @ Ybar, Yte)

    # --- retrieval over TRAIN IMAGES (image space), the HCMA-style memory path
    si = q @ l2n(Ytr).T
    res["retr_image_hard"] = ranks(Ytr[si.argmax(1)], Yte)
    for tau in (0.02, 0.05, 0.1):
        w = np.exp((si / tau) - (si / tau).max(1, keepdims=True))
        w /= w.sum(1, keepdims=True)
        res[f"retr_image_soft_tau{tau}"] = ranks(w @ Ytr, Yte)
    for kk in (5, 16, 64):
        idx = np.argsort(-si, 1)[:, :kk]
        agg = np.stack([l2n(Ytr[idx[i]]).mean(0) for i in range(len(idx))])
        res[f"retr_image_topk{kk}"] = ranks(agg, Yte)

    # --- how good is the QUERY itself? rank of the query against train concepts,
    #     measured as: is the nearest train concept's image mean close to the
    #     TRUE test image embedding (a query-quality proxy that needs no labels)
    d_true = np.sum(q * l2n(Yte), 1)
    d_best = float(np.median([float(np.sum(Ybar[st.argmax(1)][i] * l2n(Yte[i]))) for i in range(len(Yte))]))

    print(f"\n{'='*90}\nEEG QUERY vs ORACLE QUERY (gallery ceiling was top-1 0.425 / 2-way 0.930)\n{'='*90}")
    print(f"{'method':<34}{'top1':>9}{'top5':>9}{'2way':>9}{'cos':>9}")
    for k, v in res.items():
        print(f"{k:<34}{v['top1']:>9.4f}{v['top5']:>9.4f}{v['twoway']:>9.4f}{v['cos_to_target']:>9.4f}")

    base = res["direct_regression"]
    best = max(res.items(), key=lambda kv: kv[1]["twoway"])
    verdict = {
        "ceiling_from_gate_top1": 0.425,
        "direct_top1": base["top1"],
        "best_retrieval_variant": best[0],
        "best_retrieval_top1": best[1]["top1"],
        "best_retrieval_twoway": best[1]["twoway"],
        "retrieval_beats_direct_top1": best[1]["top1"] > base["top1"] + 0.01,
        "retrieval_beats_direct_twoway": best[1]["twoway"] > base["twoway"] + 0.01,
        "gap_to_ceiling_top1": 0.425 - best[1]["top1"],
        "query_self_cos_to_true": float(d_true.mean()),
        "query_retrieved_concept_mean_cos": d_best,
    }
    print(f"\n{'='*90}\nVERDICT\n{'='*90}")
    for k, v in verdict.items():
        print(f"  {k}: {v}")
    print("\nREAD: if no retrieval variant beats direct_regression, retrieval is NOT the")
    print("lever and the direct regressor is the architecture. If one does, the lever is")
    print("QUERY QUALITY (how close the EEG query gets to the right train concept).")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(
        {"subject": args.subject, "z_source": args.z_source, "results": res,
         "verdict": verdict,
         "note": ("ridge and gallery are train-only; the test EEG is read once. The "
                  "gallery ceiling of 0.425 comes from idg_attr_gate.json (hard_name).")},
        indent=2), encoding="utf-8")
    print(f"\n[save] {args.out_json}")


if __name__ == "__main__":
    main()
