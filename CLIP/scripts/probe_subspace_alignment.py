#!/usr/bin/env python
"""Diagnose WHICH axis is broken: the representation (Axis 1) or the mapping (Axis 2).

The problem being diagnosed
---------------------------
`calibration.saw_whiten` documents in its own docstring that whitening a 200-sample
query set in a 512-dim embedding is rank-deficient: "~313 of the 512 eigenvalues are
numerical zero". It patches that with `shrink`. The architectural reading of the same
fact is different: if the informative subspace is ~16-dimensional and `d_embed` is 512,
then EVERY Axis-2 mechanism -- whitening, Procrustes recovery, any learned subject map --
is estimating a `d_embed^2`-sized object from 200 samples in a space that only has
`r^2` real degrees of freedom. A `shrink` constant cannot fix a 32x oversizing.

Four measurements, each separating one hypothesis from another
--------------------------------------------------------------
1. VARIANCE CONCENTRATION -- how many dimensions does the query set actually use? If
   the answer is ~16, the 512-wide head is not storing 512 things and the calibration
   is paying for dimensions that do not exist.

2. SUBSPACE OVERLAP between the EEG cloud and the IMAGE cloud. This is the measurement
   the others depend on, and getting it wrong invalidates them.

   It matters because of a subtlety: the natural test -- "project BOTH clouds onto the
   EEG's top-r directions and retrieve" -- is only fair if the image cloud's variance
   actually lies in those directions. If the EEG's top variance directions are
   SUBJECT-NUISANCE that the image targets have no counterpart for, then projecting the
   gallery through them DESTROYS the gallery, and a low retrieval score would be read as
   "the representation is bad" when it really says "the query covariance is the wrong
   coordinate system". Those two have opposite redesigns, so the overlap is measured
   rather than assumed. Reported as (a) the fraction of each cloud's variance captured by
   the other's top-r subspace, and (b) the principal angles between the two subspaces.

3. ORACLE MAPS -- upper bounds on the whole family of "learn a global alignment"
   fixes. Three of them, because they bound different families:
     * linear, leave-one-out : what a supervised LINEAR alignment could do for a NEW query
     * linear, fit-on-all   : what it could do on the queries it was fit to. If this is
                              far above the LOO version, the map has capacity but does not
                              transfer, i.e. the pairing is not linearly recoverable.
     * MLP, leave-one-out   : the same bound for a NONLINEAR alignment. If the MLP LOO
                              beats the linear LOO materially, a nonlinear alignment head
                              is worth building; if it does not, no alignment head will
                              help and the fix has to be in the representation.
   All three are given the labels of the other queries, so no label-free method can beat
   them. Whitening is a LINEAR map and an oracle fit absorbs it exactly, which is why the
   oracle is computed on the projected (unwhitened) representation: it measures the
   SUBSPACE, not the whitening.

Run:  python scripts/probe_subspace_alignment.py --ckpts <c1> <c2> --target-subject 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import calibration, config, evaluate  # noqa: E402
from samclip.data import things_eeg  # noqa: E402
from samclip.data.targets import load_target_stack  # noqa: E402
from samclip.models import build_model  # noqa: E402


def extract(ckpt_path: Path, target_subject: int, mvnn: str) -> dict:
    """Frozen-encoder features for the held-out subject (CPU, no GPU needed)."""
    import torch
    from torch.utils.data import DataLoader
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    channels = (config.CHANNELS_OCCIPITO_PARIETAL
                if cfg.get("channel_set") == "occipital17" else None)
    img = cfg.get("image", {}) or {}
    _, test = things_eeg.load_subject_std(target_subject, channels, mvnn=mvnn)
    targets_te = load_target_stack(img.get("feature_set", "clip_h14_multilevel"),
                                   img.get("layers"), "test")
    model = build_model(cfg, targets_te.shape[2], targets_te.shape[-1])
    model.load_state_dict(ckpt["model"])
    model.eval()
    loader = DataLoader(things_eeg.TestDataset(np.asarray(test), targets_te),
                        batch_size=200, shuffle=False, collate_fn=things_eeg.collate)
    with torch.no_grad():
        return evaluate.extract_features(model, loader, torch.device("cpu"))


# ------------------------------------------------------------------ geometry
def pc_basis(x: np.ndarray, r: int) -> np.ndarray:
    """Top-r principal directions of `x` (right singular vectors of the centered data).

    The mean is removed only to DEFINE which directions are principal; the projection
    below does not subtract it, so this stays a pure change of basis on the raw vectors.
    """
    xc = np.asarray(x, dtype=np.float64) - np.asarray(x, dtype=np.float64).mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    return vt[:min(r, vt.shape[0])].T


def var_frac_in(x: np.ndarray, b: np.ndarray) -> float:
    """Fraction of `x`'s centered variance that lies inside `b`'s span."""
    xc = np.asarray(x, dtype=np.float64)
    xc = xc - xc.mean(0, keepdims=True)
    return float((np.linalg.norm(xc @ b, axis=1) ** 2).sum()
                 / max((np.linalg.norm(xc, axis=1) ** 2).sum(), 1e-12))


def rung(q: np.ndarray, g: np.ndarray, k: int = 10) -> dict:
    qn = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    gn = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-8)
    return {"raw": calibration.report_with_scores(qn @ gn.T)["top1"],
            "csls": calibration.report_with_scores(
                calibration.csls_scores(q, g, k=k))["top1"]}


def oracle_linear(q: np.ndarray, g: np.ndarray, ridge: float = 1e-2) -> tuple[float, float]:
    """(LOO top1, fit-on-all top1) for a ridge map q -> g.

    `ridge` is relative to the mean squared entry so it means the same thing at every r.
    The fit-on-all number is the important companion: a large gap between it and the LOO
    number says the map has the capacity to align these vectors but the alignment does
    not TRANSFER to an unseen query -- i.e. the pairing is not the thing being learned.
    """
    n = q.shape[0]
    qn = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    gn = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-8)
    X = np.hstack([qn, np.ones((n, 1))])
    lam = ridge * float((X ** 2).mean())
    Ginv = np.linalg.inv(X.T @ X + lam * np.eye(X.shape[1]))
    W_all = Ginv @ (X.T @ gn)
    fit_all = 100.0 * float(np.mean(
        np.argmax(gn @ (X @ W_all).T, axis=0) == np.arange(n)))
    hit = 0
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        W = Ginv @ (X[keep].T @ gn[keep])
        if int(np.argmax(gn @ (X[i] @ W))) == i:
            hit += 1
    return 100.0 * hit / n, fit_all


def oracle_mlp(q: np.ndarray, g: np.ndarray, hidden: int = 64, steps: int = 300,
               seed: int = 0) -> float:
    """Leave-one-out bound for a NONLINEAR (1-hidden-layer) alignment."""
    import torch
    n = q.shape[0]
    qn = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    gn = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-8)
    X = torch.tensor(qn, dtype=torch.float32)
    Y = torch.tensor(gn, dtype=torch.float32)
    hit = 0
    for i in range(n):
        keep = torch.ones(n, dtype=torch.bool)
        keep[i] = False
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(),
                                  torch.nn.Linear(hidden, Y.shape[1]))
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        for _ in range(steps):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(X[keep]), Y[keep])
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = net(X[i])
            if int(torch.argmax(Y @ pred)) == i:
                hit += 1
    return 100.0 * hit / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--target-subject", type=int, required=True)
    ap.add_argument("--mvnn", default="test")
    ap.add_argument("--ranks", type=int, nargs="*", default=[8, 16, 32, 64, 199])
    ap.add_argument("--csls-k", type=int, default=10)
    ap.add_argument("--mlp-oracle", action="store_true",
                    help="also run the nonlinear LOO oracle (200 small fits; slow)")
    args = ap.parse_args()

    for c in args.ckpts:
        p = Path(c)
        tag = f"{p.parent.parent.name}/{p.parent.name}"
        feats = extract(p, args.target_subject, args.mvnn)
        q, g = np.asarray(feats["eeg"], dtype=np.float64), np.asarray(feats["img"], dtype=np.float64)

        print(f"\n{'=' * 100}\n[{tag}]  n={q.shape[0]}  d_embed={q.shape[1]}\n{'=' * 100}")
        base = rung(q, g, args.csls_k)
        full = rung(calibration.saw_whiten(q, shrink=0.1)[0], g, args.csls_k)
        print(f"  achieved rungs        raw {base['raw']:6.2f}   +CSLS {base['csls']:6.2f}   "
              f"full-whiten raw {full['raw']:6.2f} +CSLS {full['csls']:6.2f}")

        print(f"\n  1+2. subspace geometry")
        print(f"  {'r':>5} | {'eeg var':>8} {'img var':>8} | {'img var in':>11} {'eeg var in':>11} | "
              f"{'proj raw':>9} {'proj+csls':>10} {'whit':>7} {'whit+csls':>10}")
        for r in args.ranks:
            bq, bg = pc_basis(q, r), pc_basis(g, r)
            vt = var_frac_in(q, bq)
            vi = var_frac_in(g, bg)
            ov_q = var_frac_in(q, bg)          # eeg variance captured by the IMAGE basis
            ov_i = var_frac_in(g, bq)          # image variance captured by the EEG basis
            zq, zg = q @ bq, g @ bq
            rp = rung(zq, zg, args.csls_k)
            zw = calibration.saw_whiten(zq, shrink=0.0)[0]
            rw = rung(zw, zg, args.csls_k)
            print(f"  {r:>5} | {vt:>8.3f} {vi:>8.3f} | {ov_i:>11.3f} {ov_q:>11.3f} | "
                  f"{rp['raw']:>9.2f} {rp['csls']:>10.2f} {rw['raw']:>7.2f} {rw['csls']:>10.2f}")

        print(f"\n  3. oracle alignment (given the OTHER queries' labels)")
        print(f"  {'r':>5} | {'linear LOO':>11} {'linear all':>11} | {'MLP LOO':>9}")
        for r in args.ranks:
            bq = pc_basis(q, r)
            zq, zg = q @ bq, g @ bq
            loo, allf = oracle_linear(zq, zg)
            mlp = oracle_mlp(zq, zg) if args.mlp_oracle else float("nan")
            mlp_s = "     n/a " if np.isnan(mlp) else f"{mlp:>8.2f}"
            print(f"  {r:>5} | {loo:>11.2f} {allf:>11.2f} | {mlp_s}")
        print("     Read `linear LOO` against the achieved rungs above. If a SUPERVISED map, "
              "with labels for 199 of 200 queries,")
        print("     cannot beat the label-free rung, then no global alignment head can close "
              "the gap and the fix is the representation.")


if __name__ == "__main__":
    main()
