#!/usr/bin/env python
"""C3 probe — retrieval under the per-concept FISHER (Mahalanobis) metric.

THE HYPOTHESIS (docs debate, 2026-10-06). Each concept is not a point but a Gaussian
`N(mu_c, Sigma_c)` over its R unlabelled test repetitions. `Sigma_c` is SUBJECT-SPECIFIC (each
subject has its own noise ellipsoid), so using `Sigma_c^{-1}` as the retrieval metric is a
subject adaptation that needs no labels, no reference metric, and never touches the concept
INDEX. It therefore cannot hit either horn of the two-horn bound (marginal isotropy kills
label-free coordinate search; a reference metric IS the answer key under index-aligned eval).

WHY THIS REDUCES TO MAHALANOBIS AND NOT TO BURES. The gallery is a POINT per concept (one frozen
image embedding), and the Bures-Wasserstein / Fisher-Rao distance between a Gaussian and a point
degenerates to `||mu - g||^2 + Tr(Sigma)`, in which `Tr(Sigma)` is constant per query ROW and
cannot change that row's ranking. So with point galleries the Gaussian-manifold metric IS the
per-concept Mahalanobis metric. The probe measures exactly that, plus the controls that separate
its two ingredients (centring vs the per-concept metric).

WHAT EACH ROW IS
  "point cosine + CSLS"        the SCORE-like baseline on the mean embedding
  "point + centre + CSLS"      + the query-cloud mean removed (the measured strongest rung)
  "cloud (shipped T2)"         our deployed order-2 operator, for reference
  "fisher l=<\lam>"            centre + per-concept Sigma_c^{-1} (shrunk toward the pooled
                               within-concept covariance by \lam) + CSLS
  "fisher diag l=<\lam>"       the same with a per-DIMENSION variance (well-conditioned floor)
  "pooled mahalanobis"         one shared Sigma for all concepts (the control that isolates the
                               value of PER-CONCEPT over GLOBAL covariance)
  "fisher (concept-shuffled)"  the control: Sigma_c assigned to the WRONG concept. A real
                               per-concept geometry must lose most of its gain here.
  "identity metric + CSLS"     M = I, which reduces exactly to cosine (asserted bit-for-bit)

LEGITIMACY. No reference metric, no gallery labels, no concept-index correspondence. The only
inputs are the target subject's own repetitions. `plan_acc` is not applicable (no FGW plan is
solved), which is why the concept-shuffled control is the load-bearing one here.

Run:
  python scripts/probe_c3_fisher.py --root outputs/stage1/g3 --subjects 1 --seeds 2025
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from samclip import calibration, config, evaluate  # noqa: E402
from samclip.data import things_eeg  # noqa: E402
from samclip.data.targets import load_target_stack  # noqa: E402
from samclip.models import build_model  # noqa: E402
import run_eval  # noqa: E402  (its main() is guarded; helpers are reused so channels/mvnn
#                              /feature-set cannot drift from the evaluation path)


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def _inv_spd(sig: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Symmetric inverse with an eigenvalue floor (the covariances here are rank-deficient)."""
    sig = (sig + sig.T) / 2.0
    w, v = np.linalg.eigh(sig)
    w = np.clip(w, floor * max(float(w.max()), 1e-12), None)
    return (v / w) @ v.T


def _mahal_scores(q: np.ndarray, g: np.ndarray, M: np.ndarray) -> np.ndarray:
    """s[c,j] = -(q_c - g_j)^T M_c (q_c - g_j), with q,g unit-norm and M either (C,d,d) or (d,d).

    M = I reduces to cosine up to a constant (asserted by the caller), so the metric is the
    ONLY thing that changes between the "identity" and "fisher" rows.
    """
    if M.ndim == 2:
        A = M @ g.T                                     # (d, C)
        t1 = np.einsum("ci,ij,cj->c", q, M, q)
        t2 = np.einsum("dj,jd->j", A, g)                # (C,): g_j^T M g_j, shared across rows
        cross = A.T @ q.T                               # (C_g, C_q): q_c^T M g_j
        return -(t1[:, None] + t2[None, :] - 2.0 * cross.T)
    Am = np.einsum("cij,jd->cid", M, g.T)               # A_m[c] = M_c @ g^T -> (C, d, C)
    t1 = np.einsum("ci,cij,cj->c", q, M, q)
    t2 = np.einsum("cdj,jd->cj", Am, g)
    cross = np.einsum("cd,cdj->cj", q, Am)
    return -(t1[:, None] + t2 - 2.0 * cross)


def _csls_on_scores(s: np.ndarray, k: int = 10) -> np.ndarray:
    """The same hubness correction `calibration.csls_scores` applies, on a PRECOMPUTED score
    matrix (higher = better). Needed because the Fisher rows are scores, not embeddings, so the
    correction has to be applied after the metric rather than to the dot product."""
    if s.size == 0:
        return s
    kk = max(1, min(int(k), s.shape[1]))
    kq = max(1, min(int(k), s.shape[0]))
    r_g = np.sort(s, axis=1)[:, -kk:].mean(axis=1, keepdims=True)
    r_q = np.sort(s, axis=0)[-kq:, :].mean(axis=0, keepdims=True)
    return 2.0 * s - r_g - r_q


def _rows_for_fold(ckpt_path: Path, target_subject: int, args, device) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt["cfg"]
    channel_set = model_cfg.get("channel_set", "all63")
    channels = (config.CHANNELS_OCCIPITO_PARIETAL if channel_set == "occipital17" else None)
    img = model_cfg.get("image", {}) or {}
    feature_set = img.get("feature_set", "clip_h14_multilevel")
    layers = img.get("layers")
    mvnn = args.mvnn if args.mvnn is not None else \
        ("test" if model_cfg.get("mvnn", "off") != "off" else "off")

    _, test = run_eval._load_fold_arrays(target_subject, channels, mvnn)
    targets_te = load_target_stack(feature_set, layers, "test")
    model = build_model(model_cfg, targets_te.shape[2], targets_te.shape[-1]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = DataLoader(things_eeg.TestDataset(test, targets_te), batch_size=200,
                        shuffle=False, collate_fn=things_eeg.collate)
    feats = evaluate.extract_features(model, loader, device)
    reps = things_eeg.load_test_reps(target_subject, channels, mvnn=mvnn)
    z_reps = np.asarray(evaluate.embed_reps(model, reps, device), dtype=np.float64)
    g = _unit(np.asarray(feats["img"], dtype=np.float64))
    C, R, d = z_reps.shape

    q_mean = z_reps.mean(axis=1)                        # (C, d)
    rows: dict[str, dict] = {}

    def emit(name, scores):
        rows[name] = calibration.report_with_scores(scores)

    # --- baselines -------------------------------------------------------------------
    qn = _unit(q_mean)
    emit("point cosine + CSLS",
         calibration.csls_scores(qn, g, k=args.csls_k))
    qc, _ = calibration.center_queries(q_mean)
    qc = _unit(qc)
    emit("point + centre + CSLS",
         calibration.csls_scores(qc, g, k=args.csls_k))
    sc_cloud, _ = calibration.rep_cloud_scores(z_reps, feats["img"], k=args.csls_k,
                                               rho=args.rho)
    emit("cloud (shipped T2)", sc_cloud)

    # --- the metric family -----------------------------------------------------------
    # Covariances are computed on the CENTRED repetitions (the same centring the query gets),
    # in the aligned space the encoder produced. `lam` shrinks toward the pooled within-concept
    # covariance; the diagonal variant is the well-conditioned floor when d >> R.
    zc = z_reps - z_reps.mean(axis=1, keepdims=True)
    Sig = np.einsum("crk,crl->ckl", zc, zc) / max(R - 1, 1)          # (C, d, d)
    Sig_pool = Sig.mean(axis=0)
    Sig_diag = np.diagonal(Sig, axis1=1, axis2=2)                    # (C, d)
    Sig_pool_diag = Sig_diag.mean(axis=0)

    for lam in args.lam:
        S = (1.0 - lam) * Sig + lam * Sig_pool[None]
        M = np.stack([_inv_spd(S[c]) for c in range(C)], axis=0)
        emit(f"fisher l={lam:g}", _csls_on_scores(_mahal_scores(qc, g, M), k=args.csls_k))
        # diagonal (per-dimension) metric
        vd = (1.0 - lam) * Sig_diag + lam * Sig_pool_diag[None]
        Md = np.stack([np.diag(1.0 / np.clip(vd[c], 1e-6 * vd[c].max(), None))
                       for c in range(C)], axis=0)
        emit(f"fisher diag l={lam:g}", _csls_on_scores(_mahal_scores(qc, g, Md), k=args.csls_k))

    # --- controls --------------------------------------------------------------------
    lam0 = float(args.lam[0])
    S = (1.0 - lam0) * Sig + lam0 * Sig_pool[None]
    M = np.stack([_inv_spd(S[c]) for c in range(C)], axis=0)
    emit("identity metric + CSLS", _csls_on_scores(_mahal_scores(
        qc, g, np.broadcast_to(np.eye(d), (C, d, d))), k=args.csls_k))
    emit("pooled mahalanobis", _csls_on_scores(_mahal_scores(
        qc, g, _inv_spd(Sig_pool)), k=args.csls_k))

    perm = np.random.default_rng(0).permutation(C)
    emit("fisher (concept-shuffled)", _csls_on_scores(_mahal_scores(
        qc, g, M[perm]), k=args.csls_k))

    return {"ckpt": str(ckpt_path), "target_subject": int(target_subject),
            "channel_set": channel_set, "mvnn": mvnn, "R": int(R), "d": int(d),
            "n_queries": int(C), "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "outputs/stage1/g3"),
                    help="dir holding <sub>_k20_seed<seed>/last.pt")
    ap.add_argument("--subjects", nargs="+", type=int, default=[1])
    ap.add_argument("--seeds", nargs="+", type=int, default=[2025])
    ap.add_argument("--mvnn", default=None)
    ap.add_argument("--csls-k", type=int, default=10)
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--lam", nargs="+", type=float, default=[0.5, 0.7, 0.9])
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/eval/c3_fisher"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    for s in args.subjects:
        for seed in args.seeds:
            ck = Path(args.root) / f"sub{s:02d}_k20_seed{seed}/last.pt"
            out = outdir / f"sub{s:02d}_seed{seed}.json"
            if not ck.exists():
                print(f"[c3] [skip] no ckpt {ck}")
                continue
            if out.exists() and out.stat().st_mtime > ck.stat().st_mtime:
                print(f"[c3] [skip] done {out.name}")
                continue
            print(f"[c3] === sub{s:02d}/seed{seed} ===")
            payload = _rows_for_fold(ck, s, args, device)
            out.write_text(json.dumps(payload, indent=2))
            for name, r in payload["rows"].items():
                print(f"    {name:<30s} top1 {r['top1']:6.2f}  top5 {r['top5']:6.2f}")
            print(f"[c3] wrote {out}")


if __name__ == "__main__":
    main()
