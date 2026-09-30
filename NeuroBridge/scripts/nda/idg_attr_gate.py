#!/usr/bin/env python3
"""THE GATE for the attribute-prompt idea: is the attribute code a sufficient
descriptor, and is it better than a concept name under concept disjointness?

WHY THIS IS THE ONLY BLOCKING MEASUREMENT
-----------------------------------------
Every other part of the IDG architecture has either been measured (shared_r,
dual-target orthogonality, band identifiability) or is a mechanical change
(FRLA).  The attribute vocabulary is the one interface that does not exist yet,
and the whole "bit-budget prompt" claim rests on it.

THE CLAIM UNDER TEST (stated before measuring)
---------------------------------------------
Naming one of 200 test concepts costs log2(200) = 7.64 bits, and EEG delivers
about 1.5 bits (measured: top-1 = 8% against the oracle concept target, with a
train/test concept intersection of exactly ZERO).  Antonym attribute axes are
DIRECTIONS in CLIP text space, so they are shared between train and test
concepts and are not capped by disjointness.  Therefore an attribute code should
describe a test concept better than a hard nearest-train-concept name does.

THE MEASUREMENT
---------------
Conditional mean by kernel regression over TRAIN concepts, in ATTRIBUTE space:

    mu_hat(c) = sum_i softmax( cos(c, c_i)/tau ) * ybar_i

with ybar_i the mean TRAIN IMAGE embedding of concept i, and c the attribute code
of the query.  Evaluated by top-1 / 2-way against REAL test CLIP image
embeddings.  This is a CEILING: c comes from the oracle test text embedding, i.e.
it assumes EEG predicts the attributes perfectly.  A high ceiling means the
attribute resolution is sufficient and the remaining problem is predicting c from
EEG (a separate, trainable head).  A low ceiling kills the idea outright.

BASELINES, all on the same test rows and the same 200 real image embeddings
    constant      the train concept mean  -- the floor; a constant scored cos
                  0.6147 on this data, which is why 2-way/top-1 are the metrics
                  used everywhere here and cosine is reported only as context
    hard_name     nearest TRAIN concept (oracle-text query) -- the self-prompt
                  analogue, and the thing that cannot work across disjointness
    attr_soft     the attributre kernel, at several code widths K
    random_code   a random query code -- the null for the kernel itself

LEAK-FREE
    the query code is built from an ORACLE test text embedding in this diagnostic
    only; nothing here is used to build the production prompt.  The train side
    (concept words, per-concept image means, axes) is train-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


NB_ROOT = Path(__file__).resolve().parents[2]

# Binary antonym pairs.  These are chosen to be CONCEPT-AGNOSTIC: every pair is a
# property axis, and neither pole is a class name, so the vocabulary transfers to
# unseen classes by construction.  Kept deliberately small and physical -- the
# measured EEG concept channel carries ~1.5 bits, so a large vocabulary would
# only add unidentifiable bits.
PAIRS: list[tuple[str, str]] = [
    ("a large object", "a small object"),
    ("a round object", "an angular object"),
    ("a long thin object", "a compact object"),
    ("a metallic object", "a wooden object"),
    ("a smooth object", "a textured object"),
    ("a bright object", "a dark object"),
    ("a colorful object", "a monochrome object"),
    ("a living animal", "a manufactured object"),
    ("a natural thing", "an artificial thing"),
    ("a heavy object", "a lightweight object"),
    ("a single object", "a group of objects"),
    ("an object with a handle", "an object without a handle"),
    ("an object with wheels", "an object without wheels"),
    ("an object with a flat surface", "an object with a curved surface"),
    ("an outdoor object", "an indoor object"),
    ("a soft object", "a rigid object"),
    ("a transparent object", "an opaque object"),
    ("a sharp object", "a blunt object"),
    ("a transparent-glass object", "a solid-material object"),
    ("an object that is worn", "an object that is held"),
    ("an object used for eating", "an object not used for eating"),
    ("a vehicle-like object", "a non-vehicle object"),
    ("an abstract shape", "a concrete object"),
    ("a vertical object", "a horizontal object"),
]


def l2n(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x).astype(np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def ranks(pred: np.ndarray, targ: np.ndarray) -> dict:
    p, t = l2n(pred), l2n(targ)
    s = p @ t.T
    r = (p @ l2n(targ.mean(0, keepdims=True)).T).mean()   # context only
    n = len(s)
    idx = np.arange(n)
    perm = np.random.default_rng(0).permutation(n)
    ok = idx != perm
    return {
        "top1": float(np.mean(s.argmax(1) == idx)),
        "top5": float(np.mean([idx[i] in np.argsort(-s[i])[:5] for i in range(n)])),
        "twoway": float(np.mean(s[idx, idx][ok] > s[idx, perm][ok])),
        "cos_to_target": float((p * t).sum(1).mean()),
        "cos_to_centroid": float(r),
    }


def encode_texts(texts: list[str], device: str) -> np.ndarray:
    import torch
    import open_clip

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", device=dev)
    model = model.to(dev).eval()
    tok = open_clip.get_tokenizer("ViT-H-14")
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 256):
            t = tok(texts[i:i + 256]).to(dev)
            out.append(model.encode_text(t).float().cpu())
    return torch.cat(out, 0).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-text-dir", type=str,
                    default=f"{NB_ROOT}/outputs/nda_ss/sub-08/clip_text")
    ap.add_argument("--captions-jsonl", type=str,
                    default=f"{NB_ROOT}/outputs/g2/captions/captions_train.jsonl")
    ap.add_argument("--img-train", type=str,
                    default="/project/peilab/why/eeg-brainit/outputs/atm_bridge/clip_img_train_1024.npy")
    ap.add_argument("--img-test", type=str,
                    default="/project/peilab/why/eeg-brainit/outputs/atm_bridge/clip_img_test_1024.npy")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--out-json", type=str, default=f"{NB_ROOT}/outputs/idg/attr_gate.json")
    args = ap.parse_args()

    ct = Path(args.clip_text_dir)
    tr_phr = json.loads((ct / "train" / "concept_phrases.json").read_text(encoding="utf-8"))
    te_phr = json.loads((ct / "test" / "concept_phrases.json").read_text(encoding="utf-8"))
    tr_txt = l2n(np.load(ct / "train" / "text_concept_clip.npy").astype(np.float32))
    te_txt = l2n(np.load(ct / "test" / "text_concept_clip.npy").astype(np.float32))
    Ytr = np.load(args.img_train).astype(np.float32)
    Yte = np.load(args.img_test).astype(np.float32)
    print(f"[gate] train concepts {tr_txt.shape} test concepts {te_txt.shape} "
          f"| train imgs {Ytr.shape} test imgs {Yte.shape}")

    # ---- disjointness, re-verified rather than assumed
    inter = set(tr_phr) & set(te_phr)
    print(f"[gate] train/test concept name intersection = {len(inter)} "
          f"{sorted(inter)[:5] if inter else '(disjoint)'}")
    if inter:
        raise SystemExit("[FATAL] concepts are not disjoint; the whole premise fails")

    # ---- per-TRAIN-concept image means (train images only)
    lab_dir: list[str] = []
    for line in Path(args.captions_jsonl).read_text(encoding="utf-8").splitlines():
        if line.strip():
            p = json.loads(line)["path"]
            lab_dir.append(p.rsplit("/", 1)[0].rsplit("/", 1)[-1])
    concepts = [d.split("_", 1)[1].replace("_", " ") for d in lab_dir]
    if len(concepts) != Ytr.shape[0]:
        raise SystemExit(f"[FATAL] captions {len(concepts)} != image rows {Ytr.shape[0]}")
    index = {c: i for i, c in enumerate(tr_phr)}
    missing = sorted({c for c in concepts if c not in index})
    if missing:
        raise SystemExit(f"[FATAL] {len(missing)} captions concepts not in the gallery: {missing[:5]}")
    ci = np.asarray([index[c] for c in concepts])
    Ybar = np.zeros((len(tr_phr), Ytr.shape[1]), dtype=np.float32)
    cnt = np.zeros(len(tr_phr), dtype=np.int64)
    np.add.at(Ybar, ci, Ytr)
    np.add.at(cnt, ci, 1)
    have = cnt > 0
    Ybar[have] /= cnt[have, None]
    print(f"[gate] per-concept image means: {int(have.sum())}/{len(tr_phr)} concepts covered "
          f"({cnt[have].mean():.1f} imgs/concept)")

    # ---- attribute axes: CLIP-TEXT directions of the antonym pairs
    poles = encode_texts([p for pair in PAIRS for p in pair], args.device)
    D = l2n(poles[0::2] - poles[1::2])                     # (K,1024)
    K_all = D.shape[0]
    print(f"[gate] {K_all} antonym axes encoded")

    # how much concept variance do these axes carry, and do they transfer?
    def code(x: np.ndarray, k: int) -> np.ndarray:
        return l2n(x @ D[:k].T)

    # reliability per axis = |correlation with the concept embedding| spread,
    # computed on TRAIN concepts only, and checked on TEST concepts
    tr_c, te_c = code(tr_txt, K_all), code(te_txt, K_all)
    ax = []
    for j in range(K_all):
        a = float(tr_c[:, j].std())
        b = float(te_c[:, j].std())
        ax.append({"pair": f"{PAIRS[j][0]} / {PAIRS[j][1]}",
                   "train_std": a, "test_std": b, "ratio": b / max(a, 1e-8)})
    rep_axes = sorted(ax, key=lambda d: -d["test_std"])
    print(f"\n{'='*96}\nAXIS TRANSFER (a shared direction should keep its spread on test concepts)\n{'='*96}")
    for d in rep_axes[:10]:
        print(f"  test_std {d['test_std']:.4f}  train_std {d['train_std']:.4f}  "
              f"ratio {d['ratio']:.3f}   {d['pair']}")

    # ---- the kernel, with tau chosen on TRAIN concept halves (never on test)
    def kernel_mu(cq: np.ndarray, k: int, tau: float) -> np.ndarray:
        ctr = code(tr_txt, k)
        sim = (cq @ ctr.T) / tau
        w = np.exp(sim - sim.max(1, keepdims=True))
        w /= w.sum(1, keepdims=True)
        return w @ Ybar

    rs = np.random.default_rng(0)
    h1, h2 = rs.permutation(len(tr_phr))[: len(tr_phr) // 2], rs.permutation(len(tr_phr))[len(tr_phr) // 2:]
    best_tau = {}
    print(f"\n{'='*96}\nTAU SELECTION (train-concept halves only, then frozen)\n{'='*96}")
    for k in (2, 4, 6, 8, 12, K_all):
        bv, bt = -1.0, 0.1
        for tau in (0.02, 0.05, 0.1, 0.2, 0.35, 0.5):
            cq = code(tr_txt[h1], k)
            sim = (cq @ code(tr_txt[h2], k).T) / tau
            w = np.exp(sim - sim.max(1, keepdims=True))
            w /= w.sum(1, keepdims=True)
            mu = w @ Ybar[h2]
            s = l2n(mu) @ l2n(Ytr[h1]).T
            v = float(np.mean([i in np.argsort(-s[i])[:1] for i in range(len(s))]))
            if v > bv:
                bv, bt = v, tau
        best_tau[k] = bt
        print(f"  K={k:<3} best tau={bt:<5} held-out top-1={bv:.4f}")

    # ---- the comparison, all on the 200 real test image embeddings
    print(f"\n{'='*96}\nGATE  (query code is the ORACLE test text -> this is a CEILING)\n{'='*96}")
    print(f"{'method':<22}{'top1':>9}{'top5':>9}{'2way':>9}{'cos':>9}{'cos_cent':>11}")
    res: dict[str, dict] = {}

    const = np.tile(Ybar[have].mean(0), (len(Yte), 1))
    res["constant"] = ranks(const, Yte)

    # hard nearest TRAIN concept -- the self-prompt analogue
    sn = te_txt @ tr_txt.T
    res["hard_name"] = ranks(Ybar[sn.argmax(1)], Yte)

    for k in (2, 4, 6, 8, 12, K_all):
        res[f"attr_soft_K{k}"] = ranks(kernel_mu(code(te_txt, k), k, best_tau[k]), Yte)

    rand_code = l2n(rs.normal(size=(len(Yte), K_all)).astype(np.float32))
    res["random_code"] = ranks(kernel_mu(rand_code, K_all, best_tau[K_all]), Yte)

    for m, v in res.items():
        print(f"{m:<22}{v['top1']:>9.4f}{v['top5']:>9.4f}{v['twoway']:>9.4f}"
              f"{v['cos_to_target']:>9.4f}{v['cos_to_centroid']:>11.4f}")

    # ---- verdict
    best_attr = max((m for m in res if m.startswith("attr_soft")),
                    key=lambda m: res[m]["twoway"])
    verdict = {
        "best_attribute_variant": best_attr,
        "attr_beats_constant_twoway": res[best_attr]["twoway"] > res["constant"]["twoway"] + 0.01,
        "attr_beats_hardname_twoway": res[best_attr]["twoway"] > res["hard_name"]["twoway"] + 0.01,
        "attr_beats_hardname_top1": res[best_attr]["top1"] > res["hard_name"]["top1"] + 0.01,
        "random_code_is_null": res["random_code"]["twoway"] < 0.55,
        "gain_twoway_vs_constant": res[best_attr]["twoway"] - res["constant"]["twoway"],
        "gain_twoway_vs_hardname": res[best_attr]["twoway"] - res["hard_name"]["twoway"],
        "gain_top1_vs_hardname": res[best_attr]["top1"] - res["hard_name"]["top1"],
    }
    print(f"\n{'='*96}\nVERDICT\n{'='*96}")
    for k, v in verdict.items():
        print(f"  {'PASS' if (v is True or (isinstance(v, (int, float)) and v > 0)) else 'FAIL'}"
              f"  {k}: {v}")

    out = {"pairs": [f"{a} / {b}" for a, b in PAIRS], "K_all": K_all,
           "axis_transfer": rep_axes, "best_tau": best_tau,
           "results": res, "verdict": verdict,
           "concept_intersection": sorted(inter),
           "note": ("query codes come from the ORACLE test text embedding, so every "
                    "attr_soft number is a CEILING, not an achievable result. The "
                    "trainable step is predicting the code from EEG."),
           }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[save] {args.out_json}")


if __name__ == "__main__":
    main()
