#!/usr/bin/env python
"""MGA-0 -- does the TIME axis carry a cross-subject-alignable structure the static cloud lacks?

WHY THIS IS NOT THE STATIC PROBE AGAIN. `scripts/probe_gauge.py` (v2) already tested the *static*
question -- is the subject term a group action, and is a global map locally resolvable? -- and it
answered **GROUP SUFFICES: global maps already compose** (`outputs/probe/gauge/gauge_probe_v2.json`).
That matters for the theory: our encoder is JOINTLY trained, so its subjects are already pushed into
a shared frame, and the "orientation gap" of arXiv 2604.08579 -- which is a phenomenon of
INDEPENDENTLY pretrained encoders (DINOv2 vs MiniLM) -- does not directly apply to us. So the
gauge-fixing half of the Manifold Gauge Alignment theory is weakened HERE, and the honest open
question is the other half:

    does the temporal trajectory carry information that the static per-concept mean does not, and
    that is ALIGNABLE across subjects?

WHAT IS AND IS NOT BEING TESTED. This probe does NOT re-run retrieval (no new operator, no labels
consumed beyond the concept identity every retrieval number already uses). It tests a necessary
condition for any temporal method to be worth building: a temporal descriptor must transfer ACROSS
subjects at least as well as the static one, on held-out concepts. If a descriptor does not even
transfer, no downstream decoder can use it.

THE DESCRIPTORS (one encoder, one fold, same 200 concepts for every subject):
  * ``static``  = E(x)                       -- the deployment object (repetition-averaged trial)
  * ``early``   = E(x with t >= T/2 zeroed)  -- the encoder's read of the pre-500 ms half
  * ``late``    = E(x with t <  T/2 zeroed)  -- the post-500 ms half
  * ``dyn``     = early - late               -- the temporal CONTRAST (the trajectory's endpoint
                                                difference; a linear proxy for the manifold flow)
  * ``dyn_scr`` = the same contrast on a TIME-SCRAMBLED trial (one fixed permutation of the time
                                                axis) -- destroys the ERP while keeping the marginal

THE TEST (paired, self-calibrating). For every ordered subject pair (i -> j) and every descriptor,
fit an orthogonal Procrustes map on HALF the concepts and read the held-out half:
  * ``resid``  = ||D_i[hold] @ Q - D_j[hold]|| / ||D_j[hold]||   (lower is better)
  * ``nn_acc`` = after mapping, does nearest-neighbour in D_j recover the SAME concept?
Both arms see the same held-out concepts, so the estimation-noise floor cancels in the difference.

PRE-REGISTERED VERDICT (the theory is killed if this fails):
  MGA is WELL POSED iff   mean(static_resid - dyn_resid) > 0.02  (dyn transfers better by >= 2%
                          relative) AND dyn beats dyn_scr AND the paired t > 3 over pairs.
  Otherwise the time axis is either noise or already redundant with the static cloud, and the
  temporal branch of the theory is abandoned for this encoder WITHOUT spending any training.

Writes ``outputs/probe/mga0/mga0_summary.json`` (one entry per fold + aggregate).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT_DIR = os.path.join(ROOT, "outputs", "probe", "mga0")


def _procrustes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(x.T @ y)
    return u @ vt


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-12))


def _build_model(ckpt_path: str, device):
    from samclip import config
    from samclip.models import build_model
    from samclip.data.targets import load_target_stack

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    channel_set = cfg.get("channel_set", "all63")
    channels = (config.CHANNELS_OCCIPITO_PARIETAL if channel_set == "occipital17" else None)
    mvnn = "test" if cfg.get("mvnn", "off") != "off" else "off"
    img = cfg.get("image", {}) or {}
    targets_te = load_target_stack(img.get("feature_set", "clip_h14_multilevel"),
                                   img.get("layers"), "test")
    model = build_model(cfg, targets_te.shape[2], targets_te.shape[-1]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, channels, mvnn, int(ckpt.get("epoch", -1))


def _embed(model, x: np.ndarray, device, batch: int = 200) -> np.ndarray:
    """(N, Ch, T) -> (N, d) RAW (pre-SMN) embeddings, frozen encoder."""
    out = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            xb = torch.as_tensor(np.ascontiguousarray(x[i:i + batch]),
                                 dtype=torch.float32, device=device)
            out.append(model.embed_eeg(xb).cpu().numpy())
    return np.concatenate(out, axis=0)


def _descriptors(model, x: np.ndarray, device, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """The temporal descriptors for one subject's (C, Ch, T) averaged trials.

    TWO KINDS OF TEMPORAL CONTRAST, and the difference matters:

      * ``dyn`` = E(first half alone) - E(second half alone). Supported on a HALF the trial, so
        its dominant direction is the encoder's response to the MASKING intervention -- the same
        for every concept, hence trivially alignable but non-discriminative. It is kept
        deliberately as a POSITIVE CONTROL for the failure mode the smoke hit (low transfer
        residual, low NN accuracy).
      * ``rev`` = E(x) - E(x with the time axis REVERSED). Full support, so no masking artefact:
        the only thing that changed is temporal ORDER. This is the descriptor the theory actually
        asks about -- if temporal order carries concept information, ``rev`` must beat its
        order-destroying control.
      * ``scr`` = E(x) - E(x with a random time PERMUTATION). Same full support as ``rev`` but the
        order is destroyed, so ``rev`` vs ``scr`` isolates order from mere spectral/marginal change.
    """
    C, Ch, T = x.shape
    half = T // 2
    m_early = np.ones((1, 1, T), dtype=x.dtype); m_early[..., half:] = 0.0
    m_late = np.ones((1, 1, T), dtype=x.dtype); m_late[..., :half] = 0.0
    static = _embed(model, x, device)
    early = _embed(model, x * m_early, device)
    late = _embed(model, x * m_late, device)
    rev = static - _embed(model, x[:, :, ::-1], device)      # full-support order contrast
    perm = rng.permutation(T)                                # ONE fixed permutation per subject
    scr = static - _embed(model, x[:, :, perm], device)
    return {"static": static, "dyn": early - late, "rev": rev, "scr": scr}


def _pair_transfer(di: np.ndarray, dj: np.ndarray, rng: np.random.Generator,
                   n_split: int) -> dict:
    """Held-out Procrustes transfer i -> j, plus the no-map baseline and NN accuracy."""
    C, d = di.shape
    res: dict[str, list[float]] = {}
    nn: dict[str, list[float]] = {}
    nomap = []
    for _ in range(n_split):
        idx = rng.permutation(C)
        fit, hold = idx[: C // 2], idx[C // 2:]
        q = _procrustes(di[fit], dj[fit])
        mapped = di[hold] @ q
        res.setdefault("mapped", []).append(_rel(mapped, dj[hold]))
        nomap.append(_rel(di[hold], dj[hold]))
        # nearest-neighbour concept recovery in j-space
        d2 = ((mapped[:, None, :] - dj[hold][None, :, :]) ** 2).sum(-1)
        nn.setdefault("mapped", []).append(float((d2.argmin(1) == np.arange(hold.size)).mean()))
    return {"resid": float(np.mean(res["mapped"])), "nn_acc": float(np.mean(nn["mapped"])),
            "nomap_resid": float(np.mean(nomap))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default=os.path.join(
        ROOT, "outputs/stage1/g3/sub*_k20_seed2025/last.pt"))
    ap.add_argument("--n-split", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N folds")
    ap.add_argument("--reps", type=int, default=0,
                    help="0 = use the repetition-averaged trial; >0 = average this many reps first")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpts = sorted(glob.glob(args.ckpts))
    if args.limit:
        ckpts = ckpts[: args.limit]
    if not ckpts:
        raise SystemExit(f"no checkpoint matched {args.ckpts}")
    if device.type != "cuda":
        print("[mga0] WARNING: running on CPU (no GPU visible)")

    from samclip import config
    from samclip.data import things_eeg

    agg: dict[str, list[float]] = {}
    folds_out = []
    for ck in ckpts:
        name = os.path.basename(os.path.dirname(ck))
        fold_subj = int(name.split("_")[0].replace("sub", ""))
        model, channels, mvnn, epoch = _build_model(ck, device)
        rng = np.random.default_rng(args.seed)

        # One encoder, every subject embedded by it: the fold's 9 sources are in-domain and the
        # held-out subject is exactly the deployment case (this encoder never saw its trials).
        subjects = config.all_subjects()
        desc: dict[int, dict[str, np.ndarray]] = {}
        for s in subjects:
            reps = things_eeg.load_test_reps(s, channels, mvnn=mvnn)     # (C, R, Ch, T)
            reps = np.asarray(reps, dtype=np.float32)
            if args.reps and args.reps < reps.shape[1]:
                reps = reps[:, : args.reps]
            x = reps.mean(axis=1)                                        # (C, Ch, T)
            desc[s] = _descriptors(model, x, device, rng)

        # Pairs that matter: source -> held-out target (the real cross-subject setting).
        pairs = [(s, fold_subj) for s in subjects if s != fold_subj]
        per_desc: dict[str, list[tuple[float, float, float]]] = {}
        for s, t in pairs:
            for tag in ("static", "dyn", "rev", "scr"):
                r = _pair_transfer(desc[s][tag], desc[t][tag], rng, args.n_split)
                per_desc.setdefault(tag, []).append((r["resid"], r["nn_acc"], r["nomap_resid"]))

        fold_row = {"ckpt": ck, "target": fold_subj, "epoch": epoch,
                    "n_pairs": len(pairs)}
        for tag, vals in per_desc.items():
            a = np.asarray(vals, float)
            fold_row[f"{tag}_resid"] = float(a[:, 0].mean())
            fold_row[f"{tag}_nn_acc"] = float(a[:, 1].mean())
            fold_row[f"{tag}_nomap_resid"] = float(a[:, 2].mean())
            agg.setdefault(f"{tag}_resid", []).append(fold_row[f"{tag}_resid"])
            agg.setdefault(f"{tag}_nn_acc", []).append(fold_row[f"{tag}_nn_acc"])
        folds_out.append(fold_row)
        print(f"  [{name}] resid static {fold_row['static_resid']:.4f} rev {fold_row['rev_resid']:.4f} "
              f"scr {fold_row['scr_resid']:.4f} dyn(mask) {fold_row['dyn_resid']:.4f} "
              f"| NN static {fold_row['static_nn_acc']:.3f} rev {fold_row['rev_nn_acc']:.3f} "
              f"scr {fold_row['scr_nn_acc']:.3f} dyn {fold_row['dyn_nn_acc']:.3f}")

    static = np.asarray(agg["static_resid"], float)
    rev = np.asarray(agg["rev_resid"], float)
    scr = np.asarray(agg["scr_resid"], float)
    dyn = np.asarray(agg["dyn_resid"], float)
    # PRIMARY criterion is NN concept recovery (residual alone rewards a trivially-alignable
    # concept-independent direction -- which is exactly what the masked `dyn` is).
    nn_s = np.asarray(agg["static_nn_acc"], float)
    nn_r = np.asarray(agg["rev_nn_acc"], float)
    nn_c = np.asarray(agg["scr_nn_acc"], float)
    nn_d = np.asarray(agg["dyn_nn_acc"], float)
    nn_gain = nn_r - nn_s                                    # rev vs static
    nn_gain_t = float(nn_gain.mean() / (nn_gain.std(ddof=1) / np.sqrt(nn_gain.size))) \
        if nn_gain.size > 1 and nn_gain.std(ddof=1) > 0 else float("nan")
    order_gain = nn_r - nn_c                                 # order (rev) vs destroyed-order (scr)
    order_gain_t = float(order_gain.mean() / (order_gain.std(ddof=1) / np.sqrt(order_gain.size))) \
        if order_gain.size > 1 and order_gain.std(ddof=1) > 0 else float("nan")
    mask_artifact = bool(nn_d.mean() < nn_s.mean() and dyn.mean() < static.mean())

    well_posed = bool(nn_gain.mean() > 0.01 and nn_gain_t > 3.0
                      and order_gain.mean() > 0.0 and order_gain_t > 3.0)
    if well_posed:
        reading = ("MGA WELL POSED: the temporal-ORDER contrast transfers across subjects with "
                   "better held-out concept recovery than the static mean AND than its "
                   "order-destroying control -> build the temporal branch.")
    elif mask_artifact:
        reading = ("TIME ADDS NO ALIGNABLE STRUCTURE. The masked half-contrast has a LOWER transfer "
                   "residual but WORSE concept recovery, i.e. it is dominated by a "
                   "concept-INDEPENDENT masking direction -- trivially alignable, non-discriminative. "
                   "The full-support order contrast fails to beat the static mean. Temporal branch "
                   "abandoned for this encoder with zero training spent.")
    else:
        reading = ("TIME ADDS NO ALIGNABLE STRUCTURE: the temporal-order contrast does not beat "
                   "the static mean on held-out concept recovery (or does not beat its "
                   "order-destroying control). Temporal branch abandoned for this encoder with "
                   "zero training spent.")
    summary = {
        "n_folds": len(ckpts),
        "static_resid_mean": float(static.mean()),
        "rev_resid_mean": float(rev.mean()),
        "scr_resid_mean": float(scr.mean()),
        "dyn_masked_resid_mean": float(dyn.mean()),
        "static_nn_acc_mean": float(nn_s.mean()),
        "rev_nn_acc_mean": float(nn_r.mean()),
        "scr_nn_acc_mean": float(nn_c.mean()),
        "dyn_masked_nn_acc_mean": float(nn_d.mean()),
        "nn_gain_rev_minus_static": float(nn_gain.mean()),
        "nn_gain_rev_minus_static_t": nn_gain_t,
        "order_gain_rev_minus_scr": float(order_gain.mean()),
        "order_gain_rev_minus_scr_t": order_gain_t,
        "masked_contrast_is_alignable_but_nondiscriminative": mask_artifact,
        "folds": folds_out,
        "verdict": {"mga_well_posed": well_posed, "reading": reading},
    }
    p = os.path.join(OUT_DIR, "mga0_summary.json")
    with open(p, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[mga0] folds={len(ckpts)}")
    print(f"[mga0] NN concept recovery: static {nn_s.mean():.4f}  rev {nn_r.mean():.4f}  "
          f"scr {nn_c.mean():.4f}  dyn(mask) {nn_d.mean():.4f}")
    print(f"[mga0] NN gain (rev-static) {nn_gain.mean():+.4f} (t={nn_gain_t:.2f})  "
          f"order gain (rev-scr) {order_gain.mean():+.4f} (t={order_gain_t:.2f})")
    print(f"[mga0] wrote {p}")
    print(f"[mga0] VERDICT: {reading}")


if __name__ == "__main__":
    main()
