#!/usr/bin/env python3
"""CF-MSF route-quality probe (sub-08): WHICH TARGET should a route align to?

WHY THIS EXISTS
---------------
The first CF-MSF run (job 581546) measured the following 200-way Top-1 on the
official THINGS-EEG2 test split (one image per concept, 200 disjoint concepts):

    img 29.0 | text 15.0 | depth 21.0 | edge 31.0
    uniform+CSLS fusion 40.0   (+Sinkhorn 50.0)   <- project's best intra retrieval

The fusion rules were NOT the bottleneck: even a perfect decision rule cannot
lift a 15-31% route set to the 73-86% the literature reports for intra-subject
200-way.  The bottleneck is per-route ALIGNMENT QUALITY.  So this probe holds the
decision side fixed (same fusion code, same split, same head, same hyperparams as
`cfmsf_train.py`) and varies ONE thing: what a route aligns to.

WHAT IS VARIED (and why each arm is here)
-----------------------------------------
1. TARGET SPACE.  The current `img` route aligns to CLIP ViT-H-14 image features
   (verified: cos == 1.0000 row-wise against data/things_eeg/image_feature/
   ViT-H-14/image_train.npy, so "CLIP-image-1024" and "ViT-H-14 image" are the
   SAME object and the current route is already the strong backbone).
   The arms that are NOT yet used anywhere in the project:
     * the four THINGS-EEG2 perturbation feature banks (GaussianBlur,
       LowResolution, Mosaic, GaussianNoise) -- precomputed in the repo for the
       same 16,540 train / 200 test images, and the closest thing available to
       the "multi-level blur" target that the 86%-Top-1 retrieval systems
       (arXiv 2605.23996: 8-blur + EVNet + InfoNCE) report as their main lever;
     * a MEAN of the five levels (a genuine multi-level target, still 1024-d);
     * a CONCAT of four levels (4096-d) -- tests whether the route benefits from
       being asked to preserve several spatial scales AT ONCE rather than in a
       mixture.  This is the arm that needs `Head(out_dim=...)`;
     * RN50 image features (weaker backbone, different geometry) -- reference for
       "is the gain from the target's level or from its backbone?".
2. ESTIMATOR.  Two heads per target, so a weak result cannot be blamed on the
   head: the same gallery-NCE MLP used by the real run, and a closed-form RIDGE
   (alpha chosen on val_b).  Ridge is the linear ceiling of the target; if the
   MLP is far below ridge the head is undertrained, and if ridge is far below the
   MLP the target needs a nonlinear map.

LEAK-FREE
---------
Gradients and ridge fits use `fit` concepts only; alpha and epoch selection use
`val_b` concepts only (leakfree/split.json).  The 200 test concepts are read once,
at the end, for reporting.  Nothing is selected on a test number.

OUTPUT
------
`route_probe.json` (full table + the fusion of the best-k routes) and a printed
table.  The script never overwrites another run's directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cfmsf_train import (  # noqa: E402
    SELECTORS, SELECTOR_KEY, build_concept_mean, train_route)
from ocf_train import build_concept_bank, l2n  # noqa: E402
import leakfree as LF  # noqa: E402

VITH = "data/things_eeg/image_feature/ViT-H-14"
RN50 = "data/things_eeg/image_feature/RN50"
LEVELS = ["image", "GaussianBlur", "LowResolution", "Mosaic", "GaussianNoise"]
# the combined corruption (blur+noise+lowres+mosaic at once); this is a directory
# under the backbone, not an entry in LEVELS, because it was precomputed as one bank
MIXLEVEL = "GaussianBlur-GaussianNoise-LowResolution-Mosaic"


def load_pair(train_rel: str, test_rel: str) -> tuple[np.ndarray, np.ndarray]:
    """(train, test) instance features, L2-normalised, flattened to rows.

    train.npy is (concepts, images, D) -> (16540, D); test.npy is (200, 1, D).
    The plain image level lives at `<backbone>/image_{train,test}.npy` while each
    perturbation level is a directory `<backbone>/<Level>/{train,test}.npy`, so the
    two paths are passed in explicitly rather than guessed.
    """
    tr_p, te_p = NB_ROOT / train_rel, NB_ROOT / test_rel
    d = np.load(tr_p, mmap_mode="r").shape[-1]
    tr = np.load(tr_p).astype(np.float32).reshape(-1, d)
    te = np.load(te_p).astype(np.float32).reshape(200, -1)
    return l2n(tr), l2n(te)


def level_paths(backbone: str, level: str) -> tuple[str, str]:
    """(train_rel, test_rel) for one backbone/level bank.  Split out so the raw and the
    L2-normalised loaders cannot disagree about where a level lives -- the plain image
    level is a FILE pair while every perturbation level is a DIRECTORY pair."""
    if level == "image":
        return f"{backbone}/image_train.npy", f"{backbone}/image_test.npy"
    return f"{backbone}/{level}/train.npy", f"{backbone}/{level}/test.npy"


def load_pair_raw(train_rel: str, test_rel: str) -> tuple[np.ndarray, np.ndarray]:
    """As `load_pair` but WITHOUT normalising, so an operator can decide for itself.

    This matters for bit-exactness: the legacy target definitions apply `l2n` to
    features that `load_pair` has already normalised, so the legacy arrays are
    double-normalised.  A bank operator that normalises only once differs from them in
    the last float32 bits, which the equivalence check in `build_targets` catches.  The
    fix is to keep blocks raw here and normalise once, inside the operator, at exactly
    the point the legacy code does.
    """
    tr_p, te_p = NB_ROOT / train_rel, NB_ROOT / test_rel
    d = np.load(tr_p, mmap_mode="r").shape[-1]
    tr = np.load(tr_p).astype(np.float32).reshape(-1, d)
    te = np.load(te_p).astype(np.float32).reshape(200, -1)
    return tr, te


def level_pair(backbone: str, level: str) -> tuple[np.ndarray, np.ndarray]:
    return load_pair(*level_paths(backbone, level))


def _block(ref: str, cond: Path) -> tuple[np.ndarray, np.ndarray]:
    """`ref` -> (train, test) RAW instance features for ONE feature block.

    RAW, not normalised: the operators normalise, and they must do it at exactly the
    point the legacy definitions do, or the bank and the legacy routes disagree in the
    last float32 bits (see `load_pair_raw`).

    `ref` is a declarative string so the whole target bank can be specified without
    loading anything: a route is (bank, operator) and a bank is a list of refs.
    Two kinds exist -- backbone/level (the precomputed THINGS-EEG2 perturbation banks)
    and clip/<name> (the generation-side CLIP condition banks).
    """
    kind, _, rest = ref.partition(":")
    if kind in ("vith", "rn50"):
        return load_pair_raw(*level_paths(VITH if kind == "vith" else RN50, rest))
    if kind == "clip":
        return (np.load(cond / f"{rest}_train.npy").astype(np.float32),
                np.load(cond / f"{rest}_test.npy").astype(np.float32))
    raise SystemExit(f"[FATAL] unknown block ref {ref!r}")


def bank_blocks(bank: str) -> list[str]:
    v5 = [f"vith:{lv}" for lv in LEVELS]
    r5 = [f"rn50:{lv}" for lv in LEVELS]
    c3 = ["clip:clip_img1024", "clip:clip_depth1024", "clip:clip_edge1024"]
    c6 = c3 + ["clip:clip_img1280", "clip:clip_depth1280", "clip:clip_edge1280"]
    banks = {"V5": v5, "V6": v5 + [f"vith:{MIXLEVEL}"], "R5": r5,
             "VR": v5 + r5, "C3": c3, "C6": c6}
    if bank not in banks:
        raise SystemExit(f"[FATAL] unknown bank {bank!r}; have {sorted(banks)}")
    return banks[bank]


# Which aggregation operators each bank gets.  The axis being tested is "does the
# FORM of the aggregation matter, or only the fact that several levels are present?".
# `mean`/`cat` are already known good (they are the legacy top routes); the others are
# the untested alternatives.  Operator count per bank is deliberately uneven: applying
# an operator to a bank whose blocks are near-duplicates (e.g. concat over 10 blocks
# that share the same geometry) buys nothing but costs a 10240-d head, so the wide
# banks get fewer operators.
BANK_OPS = {
    "V5": ["mean", "cat", "max", "had", "res", "wcat", "wres"],
    "V6": ["cat", "wcat", "res"],
    "R5": ["mean", "cat", "wcat"],
    "VR": ["mean", "cat", "wcat"],
    "C3": ["mean", "cat", "wcat"],
    "C6": ["cat", "wcat"],
}


def whiten_fit(B_tr: np.ndarray, B_te: np.ndarray, fit_i: np.ndarray,
               eps: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Per-block whitening whose statistics come from `fit` rows ONLY.

    Fit on `fit_i` (the gradient split) and never on val or test, so the transform
    cannot carry any selection or test information.  The same transform is then
    applied to train and test blocks alike -- it is a global linear map, so it
    cannot leak which test image is which.

    The eigen-floor is relative (`eps * max eigenvalue`) rather than absolute, so
    the transform does not silently become a no-op when the blocks arrive at a
    different scale from the ones it was tuned on.
    """
    X = B_tr[fit_i].astype(np.float64)
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    C = (Xc.T @ Xc) / max(len(X) - 1, 1)
    w, V = np.linalg.eigh(C)
    top = float(w.max()) if w.size else 0.0
    w = np.clip(w, eps * top if top > 0 else eps, None)
    W = (V / np.sqrt(w)).astype(np.float32)
    mu32 = mu.astype(np.float32)
    return (B_tr - mu32) @ W, (B_te - mu32) @ W


def _agg(op: str, blocks: list[tuple[np.ndarray, np.ndarray]],
         fit_i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply one aggregation operator to a bank's blocks.

    Every operator except `wcat`/`wres` is a pure function of the blocks, so it has
    no fitted parameters and cannot leak.  The two whitening operators fit on `fit_i`
    (see `whiten_fit`).

    Blocks arrive RAW and are normalised here, once.  That placement is not cosmetic:
    the legacy route definitions normalise, then `cat`/`mean` normalise AGAIN, so an
    operator that normalises zero extra times produces arrays that differ from the
    legacy ones in the last float32 bits.  `build_targets` asserts equality against the
    legacy arrays, so this is checked rather than assumed.
    """
    blocks = [(l2n(a), l2n(b)) for a, b in blocks]
    if op == "mean":
        return (l2n(np.mean([b[0] for b in blocks], 0)),
                l2n(np.mean([b[1] for b in blocks], 0)))
    if op == "cat":
        # The extra `l2n` is NOT a typo: the legacy definition is
        # `concat([l2n(vith[lv][0]) ...])` where `vith[lv]` is ALREADY normalised, so
        # the legacy array is double-normalised per block.  Reproducing it here is what
        # makes `V5_cat` bit-identical to `vith_cat5`.  Dropping it changes only the
        # last float32 bits, but it would break the equivalence assertion -- and that
        # assertion is the only thing standing between a definition typo and a bank of
        # routes that silently are not the ones the reported numbers came from.
        return (np.concatenate([l2n(b[0]) for b in blocks], 1),
                np.concatenate([l2n(b[1]) for b in blocks], 1))
    if op == "max":
        return (np.maximum.reduce([b[0] for b in blocks]),
                np.maximum.reduce([b[1] for b in blocks]))
    if op == "had":
        tr, te = np.ones_like(blocks[0][0]), np.ones_like(blocks[0][1])
        for a, b in blocks:
            tr, te = tr * a, te * b
        return l2n(tr), l2n(te)
    if op in ("res", "wres"):
        # "what every level agrees on" + "what only this level says".  This is the
        # target-space counterpart of the project's semantic/structural split, and it
        # tests whether the useful part of a multi-level target is the COMMON component
        # (robust identity) or the PER-LEVEL residual (scale-specific structure).
        if op == "wres":
            blocks = [whiten_fit(a, b, fit_i) for a, b in blocks]
        m_tr = l2n(np.mean([b[0] for b in blocks], 0))
        m_te = l2n(np.mean([b[1] for b in blocks], 0))
        return (np.concatenate([m_tr] + [b[0] - m_tr for b in blocks], 1),
                np.concatenate([m_te] + [b[1] - m_te for b in blocks], 1))
    if op == "wcat":
        w = [whiten_fit(a, b, fit_i) for a, b in blocks]
        return (np.concatenate([x[0] for x in w], 1),
                np.concatenate([x[1] for x in w], 1))
    raise SystemExit(f"[FATAL] unknown operator {op!r}")


def plan_banks(banks: str) -> dict[str, tuple[str, list[str]]]:
    """name -> (op, block refs).  Declarative: loads nothing, so it can be used to
    validate a user-editable route list before spending any GPU time."""
    want = [s.strip() for s in banks.split(",") if s.strip()]
    out: dict[str, tuple[str, list[str]]] = {}
    for bk in want:
        refs = bank_blocks(bk)
        for op in BANK_OPS[bk]:
            out[f"{bk}_{op}"] = (op, refs)
    return out


def build_bank(banks: str, cond: Path, fit_i: np.ndarray) -> dict:
    """Materialise the systematic bank.  Blocks are loaded once and reused across the
    operators of the same bank -- eight operators over five 1024-d blocks would
    otherwise re-read the same 67 MB files eight times.

    Two steps happen here, in this order, and the order matters:

    1. VERIFY.  The operator output is compared bit-for-bit against the legacy arrays
       for the two routes that exist in both families (`V5_mean` == `vith_levels_mean`,
       `V5_cat` == `vith_cat5`).  This runs BEFORE any post-processing, so the check
       tests the operator code itself rather than the processing on top of it.
    2. NORMALISE ROWS.  Every bank route is returned with unit-norm rows.

    Step 2 is a correctness fix, not cosmetics.  Retrieval scores `q . t` with `q`
    already unit-norm, so a target row's SCALE acts as a per-column bias on the argmax.
    The legacy `cat`/`mean` routes happen to have uniform row norms (`cat5` rows are
    all sqrt(5), `mean` rows are all 1), which is why the reported numbers were never
    distorted -- but `wcat`/`wres` come out of a whitening with row norms ranging over
    ~55 and would have been ranked partly by their norm.  Normalising also matters for
    `build_concept_mean`: it AVERAGES the train targets, so unnormalised rows would let
    the largest-norm instances dominate the concept gallery.

    Legacy routes are deliberately left untouched -- they are what jobs 581602/581704
    were measured on, and since their row norms are uniform, normalising them would not
    have changed a single reported metric.
    """
    plan = plan_banks(banks)
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (op, refs) in plan.items():
        blocks = []
        for r in refs:
            if r not in cache:
                cache[r] = _block(r, cond)
            blocks.append(cache[r])
        raw[name] = _agg(op, blocks, fit_i)

    pairs = [(n, o) for n, o in (("V5_mean", "vith_levels_mean"),
                                 ("V5_cat", "vith_cat5")) if n in raw]
    if pairs:
        # built ONCE, outside the loop: `build_legacy_targets` reads ~1 GB of npy
        legacy = build_legacy_targets(cond)
        for new, old in pairs:
            for part in (0, 1):
                if not np.array_equal(raw[new][part], legacy[old][part]):
                    raise SystemExit(
                        f"[FATAL] bank operator disagrees with the legacy definition: "
                        f"{new} != {old} (part {part}), max|delta|="
                        f"{np.abs(raw[new][part] - legacy[old][part]).max():.3e}. The "
                        f"legacy arrays are what jobs 581602/581704 were measured on, so "
                        f"a mismatch makes every legacy-vs-bank comparison meaningless.")
        del legacy
    print(f"[bank] {len(raw)} routes from {banks}; operator equivalence verified "
          f"bit-exactly against the legacy definitions where both exist")
    for name, (a, b) in raw.items():
        raw[name] = (l2n(a), l2n(b))
    return raw


def build_legacy_targets(cond: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The pre-existing target family, definitions UNCHANGED.

    Split out of `build_targets` so the bank's equivalence check can obtain the legacy
    arrays without re-entering `build_targets` (which would recurse).  Anything edited
    in here changes what jobs 581602/581704 measured.
    """
    t: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    t["img_clip"] = (l2n(np.load(cond / "clip_img1024_train.npy").astype(np.float32)),
                     l2n(np.load(cond / "clip_img1024_test.npy").astype(np.float32)))
    t["depth_clip"] = (l2n(np.load(cond / "clip_depth1024_train.npy").astype(np.float32)),
                       l2n(np.load(cond / "clip_depth1024_test.npy").astype(np.float32)))
    t["edge_clip"] = (l2n(np.load(cond / "clip_edge1024_train.npy").astype(np.float32)),
                      l2n(np.load(cond / "clip_edge1024_test.npy").astype(np.float32)))

    vith = {}
    for lv in LEVELS:
        vith[lv] = level_pair(VITH, lv)
        t[f"vith_{lv.lower()}"] = vith[lv]
    t["rn50_image"] = level_pair(RN50, "image")
    # the combined corruption bank (blur+noise+lowres+mosaic at once)
    t["vith_mixall"] = level_pair(VITH, MIXLEVEL)

    # multi-level MEAN: average the five L2-normalised levels, renormalise.
    mean_tr = l2n(np.mean([vith[lv][0] for lv in LEVELS], axis=0))
    mean_te = l2n(np.mean([vith[lv][1] for lv in LEVELS], axis=0))
    t["vith_levels_mean"] = (mean_tr, mean_te)

    # multi-level CONCAT.  A subset of 3 levels keeps the head small while still
    # forcing several scales into one vector.
    for tag, lvs in (("cat3", ["image", "GaussianBlur", "LowResolution"]),
                     ("cat5", LEVELS)):
        t[f"vith_{tag}"] = (np.concatenate([l2n(vith[lv][0]) for lv in lvs], 1),
                            np.concatenate([l2n(vith[lv][1]) for lv in lvs], 1))
    return t


def build_targets(args, fit_i: np.ndarray | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """name -> (train_instance_features, test_bank_features), all L2-normalised.

    Two families are returned together:

    * the LEGACY names (`img_clip`, `vith_cat5`, ...) from `build_legacy_targets`.
      Their definitions are untouched and reproduce the arrays used by jobs
      581602/581704 bit-for-bit, so any number already reported stays comparable.
    * the SYSTEMATIC bank (`V5_cat`, `VR_wcat`, ...) from `--banks`, which varies the
      aggregation operator over fixed block sets.

    `fit_i` may be None only for route-name validation, where the whitening operators
    are skipped rather than fitted on the wrong rows.
    """
    cond = NB_ROOT / args.cond_cache
    t = build_legacy_targets(cond)

    banks = getattr(args, "banks", "")
    if banks:
        if fit_i is None:
            # Name-validation path.  The whitening operators need `fit_i`, so they
            # cannot be materialised here; their NAMES are what the caller is checking,
            # and fitting a whiten transform on the wrong rows would be a silent leak.
            names = [f"{n}({'skipped' if op in ('wcat', 'wres') else 'ready'})"
                     for n, (op, _) in plan_banks(banks).items()]
            print(f"[bank] --banks={banks}: {len(names)} route names validated "
                  f"without materialising -> {names}")
        else:
            t.update(build_bank(banks, cond, fit_i))
    return t


def ridge_route(ztr: np.ndarray, tgt_tr: np.ndarray, zte: np.ndarray, tgt_te: np.ndarray,
                gal: np.ndarray, cid: np.ndarray, fit_i: np.ndarray, val_i: np.ndarray,
                alphas: list[float]) -> dict:
    """Closed-form ridge z -> target.  alpha selected on val_b concept top-1."""
    X = ztr[fit_i].astype(np.float64)
    Y = tgt_tr[fit_i].astype(np.float64)
    G = np.eye(X.shape[1]) * 1.0
    XtX = X.T @ X
    XtY = X.T @ Y
    best = {"alpha": None, "val_top1": -1.0}
    Ws: dict[float, np.ndarray] = {}
    for a in alphas:
        W = np.linalg.solve(XtX + a * G, XtY)
        Ws[a] = W
        qv = l2n((ztr[val_i] @ W).astype(np.float32))
        top1 = float((qv @ l2n(gal).T).argmax(1).__eq__(cid[val_i]).mean())
        if top1 > best["val_top1"]:
            best = {"alpha": a, "val_top1": top1}
    W = Ws[best["alpha"]]
    qt = l2n((zte @ W).astype(np.float32))
    sim = qt @ l2n(tgt_te).T
    return {"ridge": True, "alpha": best["alpha"], "val_top1": best["val_top1"],
            "test200": metrics_200(sim), "q_test": qt,
            "best": {"val_top1": best["val_top1"], "val_top1_shuffled": float("nan"),
                     "val_inst_cos": float("nan"), "epoch": -1}}


def csls(sim: np.ndarray, k: int = 10) -> np.ndarray:
    q = np.sort(sim, 1)[:, -k:].mean(1, keepdims=True)
    b = np.sort(sim, 0)[-k:, :].mean(0, keepdims=True)
    return 2.0 * sim - q - b


def metrics_200(sim: np.ndarray) -> dict:
    n = sim.shape[0]
    order = np.argsort(-sim, 1)
    return {
        "top1": float((order[:, 0] == np.arange(n)).mean()),
        "top5": float(np.mean([i in order[i, :5] for i in range(n)])),
        "mean_rank": float(np.mean([np.where(order[i] == i)[0][0] + 1 for i in range(n)])),
        "top1_csls": float((csls(sim).argmax(1) == np.arange(n)).mean()),
    }


def sinkhorn(sim: np.ndarray, iters: int = 50, tau: float = 0.07) -> np.ndarray:
    log_k = sim.astype(np.float64) / max(tau, 1e-6)
    log_k -= log_k.max(1, keepdims=True)
    k = np.exp(log_k)
    for _ in range(iters):
        k /= np.clip(k.sum(1, keepdims=True), 1e-12, None)
        k /= np.clip(k.sum(0, keepdims=True), 1e-12, None)
    return k.argmax(1).astype(np.int64)


class Cfg:
    """the subset of cfmsf_train's args that train_route reads"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--test-subject", type=int, default=8)
    ap.add_argument("--z-root", type=str, default=str(NB_ROOT / "outputs/ocf/intra_z"))
    ap.add_argument("--cond-cache", type=str, default="outputs/gem/cond_cache")
    ap.add_argument("--clip-text-dir", type=str,
                    default=str(NB_ROOT / "outputs/nda_ss/sub-08/clip_text"),
                    help=("concept-name -> CLIP-text bank and the per-row concept ids. "
                          "This is a DATASET-level asset, not a subject one: the 1654 "
                          "THINGS concept names and the 16540 caption rows are the same "
                          "for every subject, and a CLIP text embedding of a concept "
                          "name does not depend on who was being recorded. It is stored "
                          "under sub-08 only because that is where it was first built, "
                          "and every other pipeline in the project (run_tdm_all.sh, "
                          "run_gem_intra.sh) references it the same way. The probe "
                          "verifies that the bank is 1654 concepts and that the derived "
                          "row labels match the dataset's own concept ordering, so a "
                          "wrong path fails loudly instead of silently mislabelling."))
    ap.add_argument("--captions-jsonl", type=str,
                    default=str(NB_ROOT / "outputs/g2/captions/captions_train.jsonl"))
    ap.add_argument("--split-json", type=str,
                    default=str(NB_ROOT / "outputs/leakfree/split.json"))
    ap.add_argument("--only", type=str, default="",
                    help="comma list to restrict the target arms (debug / smoke)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--inst-weight", type=float, default=0.2)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--ridge-alphas", type=str, default="1e-2,1e-1,1,10,100")
    ap.add_argument("--fuse-topk", type=str, default="4",
                    help=("comma list of k values to also report for fusion. Every k is "
                          "evaluated from the SAME saved score matrices, so sweeping is "
                          "free and the 10-subject run can be re-analysed without "
                          "retraining. Default 4 reproduces the legacy behaviour."))
    ap.add_argument("--fuse-by", type=str, default="val_top1", choices=list(SELECTORS),
                    help=("the val_b statistic that ranks routes for fusion. 'val_top1' is "
                          "the legacy rule; the selection audit measured it as the "
                          "second-WORST of seven candidates, missing the best route in 12 "
                          "of 20 (subject, arm) pairs. Every selector is ALSO evaluated "
                          "regardless of this flag -- the ranking is free once the score "
                          "matrices exist -- so a 10-subject run can be re-ranked without "
                          "retraining."))
    ap.add_argument("--select-by", type=str, default="val_top1", choices=list(SELECTORS),
                    help="the val_b statistic that picks each head's checkpoint")
    ap.add_argument("--banks", type=str, default="",
                    help=("systematic target banks to add, e.g. 'V5,VR,C6'. Legacy route "
                          "names are always built and are bit-identical to the ones "
                          "behind the reported numbers; the banks are additive."))
    ap.add_argument("--ridge-dim-cap", type=int, default=6144,
                    help=("skip the closed-form ridge when the target dim exceeds this. "
                          "Ridge solves a (D+1)x(D+1) system, so D=10240 costs ~1e12 flops "
                          "and ~4 GB of W matrices PER ROUTE for what is only a linear "
                          "ceiling diagnostic. Skipped routes record why instead of "
                          "silently reporting a number computed a different way."))
    args = ap.parse_args()

    out = Path(args.out)
    (out / "heads").mkdir(parents=True, exist_ok=True)
    sid = f"{args.test_subject:02d}"
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ztr = l2n(np.load(Path(args.z_root) / f"sub-{sid}" / "shared_r_train.npy").astype(np.float32))
    zte = l2n(np.load(Path(args.z_root) / f"sub-{sid}" / "shared_r_test.npy").astype(np.float32))
    split = LF.load(args.split_json)
    fit_i = LF.rows_for(split, "fit", len(ztr))
    val_i = LF.rows_for(split, "val_b", len(ztr))
    assert not (set(fit_i.tolist()) & set(val_i.tolist())), "fit/val overlap"

    _, cid, phrases = build_concept_bank(Path(args.clip_text_dir), Path(args.captions_jsonl))
    n_cls = len(phrases)
    if len(cid) != len(ztr):
        raise SystemExit(f"[FATAL] cid {len(cid)} vs z {len(ztr)}")
    # Label-convention invariant. The joint trainer labels a row with the DATASET's
    # `object_idx` (index // 10), while this probe labels it with the concept id
    # recovered from the caption path. Those are two independent conventions, and if
    # they ever diverged every downstream fusion would compare mislabelled routes
    # while still producing plausible-looking numbers. Verified to hold for the
    # 1654-concept THINGS layout; asserted here so a future asset change cannot pass
    # silently (and so the same guarantee carries to every subject).
    _expected = np.arange(len(cid), dtype=np.int64) // 10
    if not np.array_equal(cid, _expected):
        n_bad = int((cid != _expected).sum())
        raise SystemExit(
            f"[FATAL] caption-derived concept ids disagree with the dataset convention "
            f"index//10 on {n_bad} of {len(cid)} rows -- the caption file and the concept "
            f"bank are out of step (check {args.captions_jsonl} vs "
            f"{args.clip_text_dir}/train/concept_phrases.json)")

    targets = build_targets(args, fit_i)
    if args.only:
        keep = [s.strip() for s in args.only.split(",") if s.strip()]
        targets = {k: v for k, v in targets.items() if k in keep}
        missing = [k for k in keep if k not in targets]
        if missing:
            raise SystemExit(f"[FATAL] unknown arms {missing}")
    print(f"[probe] sub-{sid} z={ztr.shape} concepts={n_cls} fit={len(fit_i)} "
          f"val_b={len(val_i)} arms={list(targets)}")

    cfg = Cfg()
    for k in ("epochs", "batch_size", "lr", "weight_decay", "tau", "inst_weight",
              "depth", "drop", "seed"):
        setattr(cfg, k, getattr(args, k))
    # opt-in selector; every legacy invocation leaves this at `val_top1` so the heads
    # it produces stay bit-identical to the ones behind the reported numbers
    cfg.select_by = args.select_by

    alphas = [float(s) for s in args.ridge_alphas.split(",") if s.strip()]
    report: dict = {"subject": f"sub-{sid}", "z_root": args.z_root,
                    "n_concepts": n_cls, "fit": int(len(fit_i)), "val_b": int(len(val_i)),
                    "params": {k: getattr(args, k) for k in
                               ("epochs", "lr", "tau", "inst_weight", "batch_size",
                                "weight_decay", "seed")},
                    "targets": {}}
    sims: dict[str, dict[str, np.ndarray]] = {}

    for name, (tgt_tr, tgt_te) in targets.items():
        gal = build_concept_mean(tgt_tr, cid, n_cls)
        q_te = q_tr = None
        # ---------- MLP gallery-NCE head (identical to the real CF-MSF run) ----------
        info, head = train_route(name, ztr, tgt_tr, gal, cid, fit_i, val_i, cfg,
                                 out / "heads", dev, out_dim=tgt_tr.shape[1])
        with torch.no_grad():
            q_te = l2n(head(torch.from_numpy(zte).to(dev)).cpu().numpy().astype(np.float32))
            q_tr = l2n(head(torch.from_numpy(ztr).to(dev)).cpu().numpy().astype(np.float32))
        sim_mlp = q_te @ tgt_te.T
        # ---------- ridge (linear ceiling of the same target) ----------
        # Skipped above the dim cap: the solve is O(D^3) and holds one W per alpha, so
        # D=10240 would cost ~1e12 flops and ~4 GB for a diagnostic.  Recording the
        # skip (rather than silently changing the estimator) keeps the "MLP vs ridge =
        # how nonlinear is this target" ratio honest.
        if tgt_tr.shape[1] > args.ridge_dim_cap:
            rid = None
            rid_skip = (f"dim {tgt_tr.shape[1]} > --ridge-dim-cap "
                        f"{args.ridge_dim_cap}: O(D^3) solve + one W per alpha")
            sim_rid = None
            m_rid = None
        else:
            rid = ridge_route(ztr, tgt_tr, zte, tgt_te, gal, cid, fit_i, val_i, alphas)
            sim_rid = rid["q_test"] @ tgt_te.T
            m_rid = metrics_200(sim_rid)
            rid_skip = None

        m_mlp = metrics_200(sim_mlp)
        report["targets"][name] = {
            "dim": int(tgt_tr.shape[1]),
            "mlp": {"val_top1": info["best"]["val_top1"],
                    "val_top1_shuffled": info["best"]["val_top1_shuffled"],
                    "val_inst_cos": info["best"]["val_inst_cos"],
                    "val_two_way": info["best"].get("val_two_way"),
                    "mini_top1": info["best"].get("mini_top1"),
                    "mini_csls": info["best"].get("mini_csls"),
                    "best_epoch": info["best"]["epoch"], **m_mlp},
            "selection": info.get("selection"),
            "ridge": ({"alpha": rid["alpha"], "val_top1": rid["val_top1"], **m_rid}
                      if rid is not None else None),
            "ridge_skipped": rid_skip,
        }
        sims[name] = {"mlp": sim_mlp, "mlp_csls": csls(sim_mlp),
                      "q_mlp_test": q_te, "tgt_test": tgt_te}
        if sim_rid is not None:
            sims[name].update({"ridge": sim_rid, "ridge_csls": csls(sim_rid),
                               "ridge_q": rid["q_test"]})
        print(f"[{name:<20} d={tgt_tr.shape[1]:<5}] "
              f"mlp val={info['best']['val_top1']:.4f} "
              f"tw={info['best'].get('val_two_way', float('nan')):.4f} "
              f"test={m_mlp['top1']:.4f}"
              f"/{m_mlp['top5']:.4f} csls={m_mlp['top1_csls']:.4f}"
              + ("" if m_rid is None else
                 f"  |  ridge(a={rid['alpha']:g}) val={rid['val_top1']:.4f} "
                 f"test={m_rid['top1']:.4f}/{m_rid['top5']:.4f} "
                 f"csls={m_rid['top1_csls']:.4f}"))

    # ---------------- fusion over the best-k routes (selected on val_b) -------------
    # TWO independent sweeps are recorded from the same saved score matrices:
    #   * `--fuse-by` (which val statistic ranks the routes) -- the audit measured
    #     `val_top1` as the second-worst of seven rules and as missing the best route in
    #     12/20 pairs, so the ranking statistic is itself a variable;
    #   * every k in `--fuse-topk`, because "how many routes to sum" was never swept and
    #     the audit showed raw summation can DEGRADE below the best single route.
    # Both are free (the scores are already computed), so a 10-subject run is
    # re-analysable without retraining anything.
    fusions: dict = {}
    k_list = [int(s) for s in args.fuse_topk.split(",") if s.strip()]
    ests = [e for e in ("mlp", "ridge")
            if all(e in sims[n] for n in targets)]
    # --fuse-by first (so it is the one printed and read by hand), then every other
    # selector, so the report prices all of them.  Ridge only has `val_top1`: it is a
    # closed-form fit whose alpha was chosen by that statistic, and inventing the others
    # for it would invite comparing a number computed a different way.
    fusion_bys = [args.fuse_by] + [s for s in SELECTORS if s != args.fuse_by]
    for by in fusion_bys:
        for est in ests:
            if est == "ridge" and by != "val_top1":
                continue
            rank = sorted(
                targets,
                key=lambda k: -float(
                    report["targets"][k][est].get(SELECTOR_KEY[by])
                    if report["targets"][k][est].get(SELECTOR_KEY[by]) is not None
                    else -1.0))
            for k in k_list:
                picked = rank[:k]
                if len(picked) < 2:
                    continue
                acc = np.zeros_like(sims[picked[0]][est])
                acc_c = np.zeros_like(acc)
                for p in picked:
                    acc += sims[p][est]
                    acc_c += sims[p][f"{est}_csls"]
                key = f"{est}|by={by}|k={k}"
                fusions[key] = {
                    "estimator": est, "ranked_by": by, "k": k, "routes": picked,
                    "raw": {**metrics_200(acc),
                            "sinkhorn_top1": float((sinkhorn(acc) == np.arange(200)).mean())},
                    "csls": {**metrics_200(acc_c),
                             "sinkhorn_top1": float((sinkhorn(acc_c) == np.arange(200)).mean())},
                }
                print(f"[fuse {key:<26}] csls={fusions[key]['csls']['top1']:.4f} "
                      f"csls+sink={fusions[key]['csls']['sinkhorn_top1']:.4f}")
    report["fusion"] = fusions
    report["reference_cfmsf_job"] = {
        "job": "581546", "primary": "F_all_csls", "top1": 0.40, "top1_plus_sinkhorn": 0.50,
        "NOTE": "the number this probe has to beat on the SAME 200 test concepts"}
    (out / "route_probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Keep each arm's query embeddings + test bank so a follow-up fusion (or the
    # generation stage) can be built without retraining the heads.  Ridge is omitted
    # for the routes that exceeded the dim cap -- savez would otherwise need a
    # placeholder array that a downstream reader could mistake for a real result.
    payload = {}
    for k, v in sims.items():
        payload[f"{k}__mlp_q"] = v["q_mlp_test"]
        payload[f"{k}__bank"] = v["tgt_test"]
        if "ridge_q" in v:
            payload[f"{k}__ridge_q"] = v["ridge_q"]
    np.savez_compressed(out / "probe_queries.npz", **payload)
    print(f"[probe] wrote {out}")


if __name__ == "__main__":
    main()
