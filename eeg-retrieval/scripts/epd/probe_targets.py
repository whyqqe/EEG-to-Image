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
    # on a GPU node (see slurm/epd_probe_targets.sbatch)
    python scripts/epd/probe_targets.py --subject 8 --targets clip_block26 clip_pooled dino_l vae depth

    # a quick CPU pass: one fit slot and the 17-channel subset is ~1500 x 4250
    python scripts/epd/probe_targets.py --subject 8 --channels occipito_parietal --fit-slots 0
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
from epd import config
from epd.data import concept_split, load_subject
from epd.metrics import mean_rank, retrieval_report

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


def load_feature_cache(cache: Path, source: str, n_conc: int, n_slots: int,
                       n_test: int) -> tuple[np.ndarray, np.ndarray]:
    """Load a per-(concept, slot) feature cache written by `extract_probe_features.py`.

    Two files per source (`..._train.npy` / `..._test.npy`), because the two splits do
    not share a slot count and therefore cannot live in one array.

    Shapes are asserted rather than inferred. The ridge is fitted on a (concept, slot)
    layout and evaluated on an explicit slot axis, and a cache written by a
    `--limit-concepts` smoke run would otherwise produce a probe over a fraction of
    the concepts while still reporting itself as 200-way.
    """
    d = Path(cache)
    tr_p, te_p = d / f"X_{source}_train.npy", d / f"X_{source}_test.npy"
    for p in (tr_p, te_p):
        if not p.is_file():
            avail = sorted(q.name for q in d.glob("X_*.npy"))
            raise FileNotFoundError(f"{p} not found; available: {avail}")
    tr = np.load(tr_p)
    te = np.load(te_p)
    if tr.ndim != 3 or te.ndim != 3:
        raise SystemExit(f"{tr_p} is {tr.shape} and {te_p} is {te.shape}; both must be "
                         f"(n_concepts, n_slots, D)")
    if tr.shape[:2] != (n_conc, n_slots):
        raise SystemExit(f"{tr_p} holds train features {tr.shape}, but this protocol "
                         f"needs ({n_conc}, {n_slots}, D). Re-run "
                         f"extract_probe_features.py without --limit-concepts.")
    if te.shape[0] != n_test:
        raise SystemExit(f"{te_p} holds {te.shape[0]} test concepts, expected {n_test}")
    return tr, te


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--channels", default="occipito_parietal",
                    choices=["all", "occipito_parietal"])
    ap.add_argument("--targets", nargs="+", default=["clip_block26", "clip_pooled", "dino_l", "vae", "depth"])
    ap.add_argument("--features", default=str(config.OUTPUTS / "features" / "clip_h14_layers"))
    ap.add_argument("--feature-cache", default="",
                    help="directory of X_*.npy from extract_probe_features.py. When set, "
                         "these REPLACE the flattened EEG as the ridge input, which is "
                         "what turns this from 'can the EEG carry the target' into 'did "
                         "the trained trunk keep it'.")
    ap.add_argument("--feature-source", default="",
                    help="which X_<...>.npy inside --feature-cache to use, e.g. "
                         "struct_grid / trained_struct_grid")
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
    ap.add_argument("--center-spatial", action="store_true",
                    help="subtract the FIT-set per-pixel mean from every spatial target "
                         "before fitting, so the ridge is asked for the instance-specific "
                         "residual instead of the residual plus a shared mean layout. "
                         "Makes the reported floor identically 0, which turns the margin "
                         "into a direct reading of `r_pred`. See the block in `main` for "
                         "why depth and VAE differ by so much under the uncentered form.")
    ap.add_argument("--fit-limit", type=int, default=0, help="cap fit concepts (smoke only)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"[probe] device={dev}  subject=sub-{a.subject:02d}")

    # Any CLIP ViT-H-14 block can be probed by name, not just the two the pipeline
    # ships. `outputs/features/clip_h14_layers/{train,test}/block{01..32}.npy` already
    # holds all 32 layers, so "which depth should a tower align to" is a zero-cost
    # question -- and it is the question that decides whether the structural branch
    # should target CLIP's TEXTURE-like early layers or its semantic late ones.
    # Requiring a hand-written TARGETS entry per layer would mean editing code to ask
    # a question the cache already answers.
    def resolve(name: str) -> dict | None:
        if name in TARGETS:
            return TARGETS[name]
        # `depthNN` / `vaeNN`: the same shipped spatial target, area-averaged to an
        # NN x NN grid. Exists to separate "EEG cannot carry spatial layout at all"
        # from "EEG cannot carry FINE layout", which imply opposite designs. The
        # coarse target is a deterministic linear functional of the fine one, so a
        # difference in decodability is attributable to scale alone -- no re-encoding,
        # no GPU, and the same images.
        for fam, rel in (("depth", "gt_depth/{split}_depth_64.npy"),
                         ("vae", "vae_cache/{split}_vae_latents_f16.npy")):
            if name.startswith(fam) and name[len(fam):].isdigit():
                s = int(name[len(fam):])
                if s in (4, 8, 16, 32):
                    pre = "coarse"
                    return dict(
                        kind="spatial",
                        rel=f"../../struct_targets/{pre}/train_{fam}_{s}.npy",
                        rel_test=f"../../struct_targets/{pre}/test_{fam}_{s}.npy",
                        layer_kind=f"{fam} @ {s}x{s} (area-averaged)",
                    )
        p = Path(a.features) / "train" / f"{name.replace('clip_', '', 1)}.npy"
        if name.startswith("clip_") and p.is_file():
            key = name.replace("clip_", "", 1)
            return dict(kind="clip", key=key, layer_kind=f"block {key}")
        return None

    resolved = {t: resolve(t) for t in a.targets}
    unknown = [t for t, s in resolved.items() if s is None]
    if unknown:
        raise SystemExit(f"unknown target(s) {unknown}; known: {sorted(TARGETS)} "
                         f"(plus any `clip_blockNN` with a cached layer)")

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
    if a.feature_cache:
        if not a.feature_source:
            raise SystemExit("--feature-cache requires --feature-source (which X_*.npy)")
        ftr, fte = load_feature_cache(Path(a.feature_cache), a.feature_source,
                                      n_conc, n_slots, n_test)
        print(f"[probe] ridge input = cached trunk features "
              f"{Path(a.feature_cache) / f'X_{a.feature_source}_train.npy'} "
              f"{ftr.shape}")
        X_fit = torch.from_numpy(
            ftr[fit_c][:, fit_slots].reshape(-1, ftr.shape[-1])).float()
        X_val_all = torch.from_numpy(ftr[val_c].reshape(len(val_c), n_slots, -1)).float()
        X_te = torch.from_numpy(fte[:n_test, 0].reshape(n_test, -1)).float()
    else:
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
        spec = resolved[name]
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

        # ---- optional per-pixel centering of a spatial target ----------------
        # Why this is a separate measurement rather than a redefinition
        # --------------------------------------------------------------
        # `pearson_rows` centres each ROW, so the reported floor is the correlation
        # between the *shape* of the fit-set mean target and the *shape* of each GT.
        # For depth that floor is high (+0.53 at 64x64, +0.61 at 4x4) for a reason
        # that has nothing to do with EEG: every depth map of a COCO scene shares a
        # "far at the top, near at the bottom" layout, so the mean map is already
        # shape-similar to each individual map. VAE latents have no such shared
        # layout, which is why their floor is only +0.10..+0.17.
        #
        # The uncentered margin then answers "is the prediction more shape-similar
        # to each GT than the mean map is", and it is a hard bar for a target whose
        # mean is itself a good shape. It is NOT the same question as "does the EEG
        # carry the target's instance-specific content", and those two questions
        # imply different designs:
        #
        #   * uncentered fails, centered passes -> the shared layout is what the
        #     metric rewards, but the EEG does carry the per-concept deviation.
        #     A structural branch is viable, and its output must be the DEVIATION
        #     (the mean map is a free constant any decoder has as a bias).
        #   * both fail -> the target is not carried at any scale and no head or
        #     injection mechanism can fix it. Drop the branch.
        #
        # So this centres on the FIT set's per-pixel mean only. Centering must not
        # see the val or test rows or the measurement stops being out-of-sample, and
        # with it centred the constant predictor becomes the zero vector, whose row
        # Pearson r is exactly 0 by construction (`pearson_rows` subtracts the row
        # mean and the zero row has none). The margin therefore collapses to
        # `r_pred` itself, which is the clean "how much of the residual is in the
        # EEG" number rather than a difference of two shape similarities.
        if a.center_spatial and kind == "spatial":
            mu_fit = Y_fit.mean(0, keepdim=True)          # (1, D), fit rows only
            Y_fit = Y_fit - mu_fit
            mu_np = mu_fit.cpu().numpy()
            Y_val = Y_val - mu_np
            Y_te = Y_te - mu_np
            print(f"[probe] {name}: target centred on the fit-set mean "
                  f"(||mean||={float(mu_fit.norm()):.3f})")

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
        # Which tensor the ridge was actually fitted on. A probe result is only
        # meaningful next to the input it used, and the two possible inputs here
        # (raw flattened EEG vs a trunk's cached features) answer different questions
        # from numericallly similar-looking tables.
        "ridge_input": ("raw_eeg" if not a.feature_cache
                        else f"{a.feature_source} @ {a.feature_cache}"),
        "feature_cache": str(a.feature_cache),
        "feature_source": a.feature_source,
        "ridge_input_dim": int(X_fit.shape[1]),
        "spatial_centered": bool(a.center_spatial),
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
