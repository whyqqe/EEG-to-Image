#!/usr/bin/env python
"""Which target space can this EEG actually carry? A closed-form ridge probe over
every candidate alignment target, with a constant-predictor floor for the spatial ones.

Why this exists
---------------
The structural tower trained against SDXL VAE latents and a monocular depth map
collapsed, and the numbers say it was not a tuning problem:

  * depth, across-sample variance / total variance = 0.0068 (the target sits at 0.63);
  * depth, mean pairwise cosine between predictions   = 0.9991;
  * depth, r(pred_i, GT_i) = +0.6485 against +0.6540 for the *constant* predictor that
    emits the training-set mean map -- the learned head was 0.0055 WORSE than a
    constant whose value it already contained;
  * VAE latents, instance Top-1 = 1.00% against 0.50% chance, mean rank 85 of 200.

L1's Bayes-optimal answer for a high-entropy target under limited capacity is the
conditional mean. So the heads were asked for a quantity the input does not contain,
and they answered correctly. Before spending another GPU run on a structural target,
this asks the prior question at the cost of one matrix factorisation: for each
candidate target space, how much of it is linearly decodable from the EEG, and how
much of *that* is above what a constant predictor already gets for free?

Why ridge, and why one factorisation
------------------------------------
Ridge is closed form, so there is no training and no overfitting to argue about, and
the expensive part depends only on the EEG (X), never on the target (Y). One SVD
therefore serves every target in the list. `probe_layers.py` uses the same trick to
scan CLIP's *depth*; this scans *spaces*, which is the axis the collapse lives on.

Reading the numbers -- the two families of target are not scored the same way
----------------------------------------------------------------------------
Vector targets (CLIP, DINOv2) are scored by 200-way retrieval, and their floor is
chance: a constant prediction is the same vector for every query, so it cannot rank
anything. top1 0.50 is chance and the standard error on 200 trials is ~2.8 points.

Spatial targets (VAE latent, depth, blurry RGB) are scored by retrieval AND by
per-sample Pearson r, and they need a floor to compare against, because a constant
map already correlates with every real map. That floor is reported as
`r_constant_to_gt`: the mean r that a predictor returning the fit-set mean target
achieves against each held-out target. `r_pred_to_gt` must clear it to mean anything.
Reporting r without that floor is exactly how a collapsed head reads as +0.6485 and
looks healthy.

Usage
-----
    # on a GPU node (see slurm/nwret_probe_targets.sbatch)
    python scripts/nwret/probe_targets.py --subject 8 --targets clip_block26 clip_pooled dino_l vae depth

    # a quick CPU pass: one fit slot and the 17-channel subset is ~1500 x 4250
    python scripts/nwret/probe_targets.py --subject 8 --channels occipito_parietal --fit-slots 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nwret import config
from nwret.data import concept_split, load_subject
from nwret.metrics import mean_rank, retrieval_report

# ----------------------------------------------------------------------------- roots
# Subject-independent caches, and they are the reason this is affordable: every
# target below is a property of the STIMULUS, not of the recording, so one build
# serves all ten subjects. `--target-root` defaults to the sub-08 tree only because
# that is where the existing structural caches were first written.
DEFAULT_TARGET_ROOT = config.OUTPUTS / "sub08" / "patch_dual_targets"
# The DINOv2 bank was built by `eeg-thor-a/scripts/build_ftmsf_teachers.py`, whose
# `list_images()` is sorted(categories) then sorted(images) under the SAME
# `images_root` this project uses. That ordering is asserted equal to our own
# `train_index.json` / `test_index.json` (16540 and 200 rows, zero mismatches), which
# is what makes these arrays loadable here without a rebuild.
DEFAULT_DINO_DIR = Path("/project/peilab/why/eeg-thor-a/outputs/ftmsf_teachers")


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def pearson_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-row Pearson r between two (n, D) arrays."""
    p = p - p.mean(axis=1, keepdims=True)
    q = q - q.mean(axis=1, keepdims=True)
    num = (p * q).sum(axis=1)
    den = np.linalg.norm(p, axis=1) * np.linalg.norm(q, axis=1) + 1e-12
    return num / den


# ----------------------------------------------------------------------------- targets
# Each loader returns (train, test, kind) with
#   train : (n_concepts, n_slots, D)   -- D flattened; the ridge target is linear
#   test  : (n_test, D)
#   kind  : "vector" | "spatial", which decides how the result is scored.
def _three_d(a: np.ndarray, n_conc: int, n_slots: int) -> np.ndarray:
    """Accept either (n_conc, n_slots, D) or a flattened (n_conc*n_slots, D)."""
    if a.ndim == 3:
        return a
    if a.ndim == 2 and a.shape[0] == n_conc * n_slots:
        return a.reshape(n_conc, n_slots, a.shape[1])
    raise ValueError(f"cannot read {a.shape} as ({n_conc}, {n_slots}, D)")


def load_clip(features_dir: Path, key: str, n_conc: int, n_slots: int) -> tuple:
    tr = np.load(features_dir / "train" / f"{key}.npy")
    te = np.load(features_dir / "test" / f"{key}.npy")
    tr = _three_d(tr, n_conc, n_slots)
    if te.ndim == 3:
        te = te[:, 0]
    return tr.astype(np.float32), te.astype(np.float32), "vector"


def load_dino(dino_dir: Path, key: str, n_conc: int, n_slots: int) -> tuple:
    tr = np.load(dino_dir / f"{key}_train.npy")
    te = np.load(dino_dir / f"{key}_test.npy")
    tr = _three_d(tr, n_conc, n_slots)
    if te.ndim == 3:
        te = te[:, 0]
    return tr.astype(np.float32), te.astype(np.float32), "vector"


def load_spatial(path: Path, n_conc: int, n_slots: int, is_test: bool) -> np.ndarray:
    a = np.load(path)
    a = a.astype(np.float32).reshape(a.shape[0], -1)
    if is_test:
        return a
    return _three_d(a, n_conc, n_slots)


TARGETS: dict[str, dict] = {
    # The shipped semantic target, and the reference line every other row is read
    # against: val 17.33 / test 27.50 from outputs/sub08/probe_layers.json.
    "clip_block26": dict(kind="clip", key="block26", layer_kind="block 26"),
    "clip_pooled": dict(kind="clip", key="_pooled", layer_kind="final (pooled)"),
    # The structural candidate this probe exists to price.
    "dino_l": dict(kind="dino", key="dino", layer_kind="DINOv2-L-reg4"),
    "siglip": dict(kind="dino", key="siglip", layer_kind="SigLIP-SO400M"),
    # The two targets the collapsed tower used, scored with their floor.
    "vae": dict(kind="spatial", rel="vae_cache/train_vae_latents_f16.npy",
                rel_test="vae_cache/test_vae_latents_f16.npy", layer_kind="SDXL VAE latent"),
    "depth": dict(kind="spatial", rel="gt_depth/train_depth_64.npy",
                  rel_test="gt_depth/test_depth_64.npy", layer_kind="monocular depth"),
}


def build_target(name: str, spec: dict, a, n_conc: int, n_slots: int):
    if spec["kind"] == "clip":
        return load_clip(Path(a.features), spec["key"], n_conc, n_slots)
    if spec["kind"] == "dino":
        return load_dino(Path(a.dino_dir), spec["key"], n_conc, n_slots)
    tr = load_spatial(Path(a.target_root) / spec["rel"], n_conc, n_slots, is_test=False)
    te = load_spatial(Path(a.target_root) / spec["rel_test"], n_conc, n_slots, is_test=True)
    return tr, te, "spatial"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--channels", default="occipito_parietal",
                    choices=["all", "occipito_parietal"])
    ap.add_argument("--targets", nargs="+", default=["clip_block26", "clip_pooled", "dino_l", "vae", "depth"])
    ap.add_argument("--features", default=str(config.OUTPUTS / "features" / "clip_h14_layers"))
    ap.add_argument("--dino-dir", default=str(DEFAULT_DINO_DIR))
    ap.add_argument("--target-root", default=str(DEFAULT_TARGET_ROOT))
    ap.add_argument("--val-concepts", type=int, default=150)
    ap.add_argument("--split-seed", type=int, default=2025)
    ap.add_argument("--lams", type=float, nargs="+", default=[1e2, 1e3, 1e4, 1e5])
    ap.add_argument("--fit-slots", type=int, default=0,
                    help="how many image slots per concept to FIT on (0 = all 10). The "
                         "held-out validation still sweeps every slot, so this only "
                         "trims the fit side; the ridge solution is what is being "
                         "estimated, so fewer rows is a variance cost, not a bias one.")
    ap.add_argument("--rank", type=int, default=0, help="truncate the EEG SVD (0 = all)")
    ap.add_argument("--fit-limit", type=int, default=0, help="cap fit concepts (smoke only)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"[probe] device={dev}  subject=sub-{a.subject:02d}")

    unknown = [t for t in a.targets if t not in TARGETS]
    if unknown:
        raise SystemExit(f"unknown target(s) {unknown}; known: {sorted(TARGETS)}")

    n_conc, n_slots, n_test = (config.N_TRAIN_CONCEPTS, config.N_IMAGES_PER_CONCEPT,
                               config.N_TEST_CONCEPTS)

    # ---------------- EEG
    channels = None if a.channels == "all" else config.CHANNELS_OCCIPITO_PARIETAL
    tr_eeg, te_eeg = load_subject(a.subject, channels)
    n_ch = tr_eeg.shape[2]
    print(f"[probe] EEG train {tr_eeg.shape} test {te_eeg.shape} ({n_ch} channels)")

    split = concept_split(a.val_concepts, a.split_seed)
    fit_c, val_c = split.fit_concepts, split.val_concepts
    if a.fit_limit:
        fit_c = fit_c[:a.fit_limit]
        print(f"[probe] --fit-limit {a.fit_limit} -> {len(fit_c)} fit concepts")

    # Fit on a subset of slots if asked, validate on all of them. Mirrors
    # probe_layers.py's reasoning in reverse: there the concern was that a flat
    # reshape of the VAL split silently collapses a 150-way retrieval into 15
    # concepts, so the val side keeps an explicit slot axis here too.
    fit_slots = list(range(n_slots)) if a.fit_slots <= 0 else list(range(min(a.fit_slots, n_slots)))
    X_fit = torch.from_numpy(
        tr_eeg[fit_c][:, fit_slots].reshape(-1, n_ch * tr_eeg.shape[-1])).float()
    X_val_all = torch.from_numpy(
        tr_eeg[val_c].reshape(len(val_c), n_slots, -1)).float()
    X_te = torch.from_numpy(te_eeg[:, 0].reshape(n_test, -1)).float()

    mu = X_fit.mean(0, keepdim=True)
    sd = X_fit.std(0, keepdim=True).clamp_min(1e-6)
    X_fit = ((X_fit - mu) / sd).to(dev)
    X_val_all = ((X_val_all - mu) / sd).to(dev)
    X_te = ((X_te - mu) / sd).to(dev)
    print(f"[probe] fit {tuple(X_fit.shape)} (slots {fit_slots})  "
          f"val {tuple(X_val_all.shape)} ({len(val_c)}-way x {n_slots} slots)  "
          f"test {tuple(X_te.shape)}")

    # ---------------- one factorisation, reused for every target
    t0 = time.time()
    U, S, Vh = torch.linalg.svd(X_fit, full_matrices=False)
    V = Vh.T
    if a.rank:
        k = min(a.rank, S.numel())
        U, S, V = U[:, :k], S[:k], V[:, :k]
        print(f"[probe] truncated to rank {k}")
    print(f"[probe] SVD {time.time() - t0:.0f}s  s[0]={float(S[0]):.2e} s[-1]={float(S[-1]):.2e}")

    A_val_all = X_val_all @ V
    A_te = X_te @ V

    def predict(A: torch.Tensor, B: torch.Tensor, lam: float) -> np.ndarray:
        w = (S / (S * S + lam)).unsqueeze(-1)
        return (A * w.squeeze(-1).unsqueeze(0)) @ B

    results: list[dict] = []

    for name in a.targets:
        spec = TARGETS[name]
        try:
            Ytr, Yte, kind = build_target(name, spec, a, n_conc, n_slots)
        except FileNotFoundError as e:
            print(f"[skip] {name}: {e}")
            continue
        D = Ytr.shape[-1]
        Yte = Yte.reshape(Yte.shape[0], -1) if Yte.ndim > 2 else Yte

        if kind == "vector":
            Y_fit = torch.from_numpy(l2n(Ytr[fit_c][:, fit_slots].reshape(-1, D))).float().to(dev)
            Y_val = l2n(Ytr[val_c].reshape(len(val_c), n_slots, D))
            Y_te = l2n(Yte.reshape(n_test, -1))
        else:
            # Spatial targets are NOT l2-normalised: their magnitude across the map is
            # the signal (a blank map and a structured one differ in exactly that), and
            # normalising per sample would score the recovered layout while hiding
            # whether the head learned to vary at all -- which is the thing that
            # collapsed.
            Y_fit = torch.from_numpy(Ytr[fit_c][:, fit_slots].reshape(-1, D)).float().to(dev)
            Y_val = Ytr[val_c].reshape(len(val_c), n_slots, D)
            Y_te = Yte.reshape(n_test, -1)

        t_l = time.time()
        B = U.T @ Y_fit
        best_lam, best_val, per_lam = None, -1.0, []
        for lam in a.lams:
            t1 = []
            for s in range(n_slots):
                Yh = predict(A_val_all[:, s], B, lam).cpu().numpy()
                t1.append(retrieval_report(Yh, Y_val[:, s])["top1"])
            v_top1 = float(np.mean(t1))
            per_lam.append({"lam": lam, "val_top1": v_top1, "val_top1_std": float(np.std(t1))})
            if v_top1 > best_val:
                best_val, best_lam = v_top1, lam

        Yh_te = predict(A_te, B, best_lam).cpu().numpy()
        te = retrieval_report(Yh_te, Y_te)
        te["mean_rank"] = mean_rank(Yh_te, Y_te)
        row = {"target": name, "kind": kind, "dim": D, "layer_kind": spec["layer_kind"],
               "lam": best_lam, "val_top1": best_val,
               "val_top1_std": per_lam[[p["lam"] for p in per_lam].index(best_lam)]["val_top1_std"],
               "test_top1": te["top1"], "test_top5": te["top5"],
               "test_mean_rank": te["mean_rank"], "per_lam": per_lam}

        if kind == "spatial":
            # The floor, computed the same way as the head's score but from the best
            # constant available to it: the fit-set mean target. A predictor that
            # cannot beat this has learned nothing the mean did not already say.
            mu_t = Y_fit.mean(0).cpu().numpy()
            r_pred, r_const, spread = [], [], []
            for s in range(n_slots):
                Yh = predict(A_val_all[:, s], B, best_lam).cpu().numpy()
                gt = Y_val[:, s]
                r_pred.append(pearson_rows(Yh, gt))
                r_const.append(pearson_rows(np.broadcast_to(mu_t, gt.shape), gt))
                spread.append(Yh.std(axis=0).mean() / (gt.std(axis=0).mean() + 1e-12))
            r_pred, r_const = np.concatenate(r_pred), np.concatenate(r_const)
            row.update({
                "val_r_pred_to_gt": float(r_pred.mean()),
                "val_r_constant_to_gt": float(r_const.mean()),
                "val_r_margin": float(r_pred.mean() - r_const.mean()),
                # Across-sample variance ratio: the collapse detector. Compare with the
                # target's own ratio, which is the ceiling a perfect predictor reaches.
                "val_pred_var_ratio": float(np.mean(spread)),
            })

        results.append(row)
        msg = (f"[probe] {name:>13} D={D:>6} lam*={best_lam:<8.0e} "
               f"val {best_val:6.2f}|{row['val_top1_std']:4.2f} "
               f"test {te['top1']:6.2f} top5 {te['top5']:6.2f} rank {te['mean_rank']:5.1f}")
        if kind == "spatial":
            msg += (f" | r {row['val_r_pred_to_gt']:+.4f} vs floor "
                    f"{row['val_r_constant_to_gt']:+.4f} (margin {row['val_r_margin']:+.4f})")
        print(msg + f"  ({time.time() - t_l:.0f}s)", flush=True)

    if not results:
        raise SystemExit("no targets could be evaluated")

    out = Path(a.out) if a.out else (config.OUTPUTS / f"sub{a.subject:02d}" / "probe_targets.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "subject": a.subject, "channels": a.channels, "n_channels": n_ch,
        "n_fit_concepts": len(fit_c), "n_val_concepts": len(val_c), "fit_slots": fit_slots,
        "features": str(a.features), "dino_dir": str(a.dino_dir),
        "target_root": str(a.target_root),
        "row_order": ("verified: eeg-thor-a build_ftmsf_teachers.list_images(images_root/"
                      "training_images) equals our train_index.json row-for-row "
                      "(16540 rows) and likewise the 200 test rows"),
        "selection": "val Top-1 (all slots averaged); test scored once at the selected lambda",
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2, default=float))

    # ---------------- report
    print()
    print("=" * 104)
    print("TARGET-SPACE PROBE  (closed-form ridge, val-selected, 200-way)")
    print("=" * 104)
    print(f"  {'target':>13} {'kind':>8} {'dim':>7} {'val top1':>9} {'test top1':>10} "
          f"{'test top5':>10} {'rank':>7}")
    for r in results:
        print(f"  {r['target']:>13} {r['kind']:>8} {r['dim']:>7} {r['val_top1']:>9.2f} "
              f"{r['test_top1']:>10.2f} {r['test_top5']:>10.2f} {r['test_mean_rank']:>7.1f}")
    print(f"  chance: top1 0.50, top5 2.50, mean rank 100.5;  SE on top1 ~2.8 points")
    print()
    print(f"  {'target':>13} {'r(pred,gt)':>12} {'r(const,gt)':>13} {'margin':>9} {'pred var ratio':>15}")
    any_spatial = False
    for r in results:
        if r["kind"] != "spatial":
            continue
        any_spatial = True
        print(f"  {r['target']:>13} {r['val_r_pred_to_gt']:>12.4f} "
              f"{r['val_r_constant_to_gt']:>13.4f} {r['val_r_margin']:>+9.4f} "
              f"{r['val_pred_var_ratio']:>15.4f}")
    if any_spatial:
        print("  (a spatial target is only decoded if the margin is positive and the")
        print("   variance ratio is well above 0: the collapsed run measured 0.0068)")

    # ---------------- verdict
    print()
    by = {r["target"]: r for r in results}
    ref = by.get("clip_block26") or by.get("clip_pooled")
    if ref and "dino_l" in by:
        d = by["dino_l"]["val_top1"] - ref["val_top1"]
        se = np.hypot(by["dino_l"]["val_top1_std"], ref["val_top1_std"]) / np.sqrt(n_slots)
        print(f"  DINOv2-L vs the shipped semantic target: val {d:+.2f} +-{se:.2f} (SE)")
        if d > 2 * se:
            print("  -> DINOv2-L is the more EEG-decodable space. A structural tower asked")
            print("     to predict it is asking a question this input answers.")
        elif d < -2 * se:
            print("  -> DINOv2-L is LESS decodable than the CLIP space we already use.")
            print("     Moving the structural tower onto it trades a measured collapse for")
            print("     a target that is also not there; do not run the structural redesign.")
        else:
            print("  -> the two are inside the noise of each other on this probe, so the")
            print("     target swap is not justified by decodability alone. The remaining")
            print("     argument for DINOv2 is that it encodes geometry rather than texture,")
            print("     which is a claim about WHAT is recovered, not how much.")
    for r in results:
        if r["kind"] != "spatial":
            continue
        ok = r["val_r_margin"] > 0.02 and r["val_pred_var_ratio"] > 0.20
        print(f"  {r['target']}: {'PASSES' if ok else 'FAILS'} the collapse gate "
              f"(margin {r['val_r_margin']:+.4f}, var ratio {r['val_pred_var_ratio']:.4f})"
              + ("" if ok else "  <- do not rebuild this head; the linear ceiling is the mean"))
    print(f"\n[probe] wrote {out}")


if __name__ == "__main__":
    main()
