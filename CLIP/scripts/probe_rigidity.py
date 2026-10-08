#!/usr/bin/env python
"""probe_rigidity -- is the cross-subject misalignment RIGID (as Procrustes assumes)?

THE QUESTION. `coordinate_recovery` / `subspace_soft_recovery` both fit a map in a restricted
class: orthogonal (a rigid motion of R^d), optionally regularised toward the identity. That is
the right model iff subject s's concept cloud is an isometric copy of subject t's, i.e. iff the
only thing separating two subjects is a rotation/reflection. Biology says otherwise: electrode
geometry, skull conductivity, and cortical folding differ per subject, so the map that carries
one subject's concept manifold onto another is a smooth-but-NON-RIGID re-embedding.

If that is true, rigid alignment has an IRREDUCIBLE BIAS -- a residual that no number of
landmarks can remove, because the true map is not in the model class. That is exactly the
signature the G3 grid shows for the recovery rung (+3.62 +- 1.72 Top-1, FLAT: uncorrelated with
encoder quality AND with landmark rate over 30 runs). This probe measures the bias directly.

WHAT IS MEASURED, on frozen features from the banked checkpoints.

  1. PER-SUBJECT rigidity gap. For each subject, fit `z_e -> z_i` (a) by orthogonal
     Procrustes, (b) by an unconstrained linear map, (c) by a smooth nonlinear map (RBF random
     features + ridge). Report `residual_orth / residual_smooth` as the RIGIDITY GAP. A gap of
     1.0 means "rigid is the right model class"; a large gap means the model class is wrong and
     the encoder is paying for it.

  2. THE SHARED-FRAME TEST, which is the actual cross-subject statement. Fit ONE map shared
     across all 10 subjects, and compare its per-subject residual to the per-subject fits. If
     the shared map is nearly as good, subjects differ only by a global change of frame. If it
     is much worse, the distortion is SUBJECT-SPECIFIC -- which is the quantity that bounds
     anything fitted on source subjects and applied to a held-out one.

  3. THE RESIDUAL SPECTRUM. SVD of the per-subject orthogonal residual. LOW-RANK residual
     means the non-rigid part lives in a few directions and is therefore learnable with few
     parameters (this is the case that justifies a low-dimensional corrective flow). A DIFFUSE
     residual means no such parameterisation exists at this sample size and the fix has to be
     structural (optimal transport / Gromov-Wasserstein) instead.

Run (login node, CPU, ~2 min):
    python scripts/probe_rigidity.py --subjects 1 2 3 4 5 6 7 8 9 10 --seed 2025 \
        --stage1-root outputs/stage1/g3 --out outputs/probe/rigidity.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _extract(ckpt_path: Path, subject: int, device, mvnn=None):
    """`(z_e, z_i, concepts)` for one subject's test split, on the checkpoint's own geometry."""
    import torch
    from torch.utils.data import DataLoader

    from samclip import config, evaluate
    from samclip.data import things_eeg
    from samclip.data.targets import load_target_stack
    from samclip.models import build_model

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    channel_set = cfg.get("channel_set", "all63")
    channels = (config.CHANNELS_OCCIPITO_PARIETAL
                if channel_set == "occipital17" else None)
    img_cfg = cfg.get("image", {}) or {}
    feature_set = img_cfg.get("feature_set", "clip_h14_multilevel")
    layers = img_cfg.get("layers")
    route_mvnn = mvnn if mvnn is not None else \
        ("test" if cfg.get("mvnn", "off") != "off" else "off")

    _, test = things_eeg.load_subject_std(subject, channels, mvnn=route_mvnn)
    targets = load_target_stack(feature_set, layers, "test")
    model = build_model(cfg, targets.shape[2], targets.shape[-1]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    loader = DataLoader(things_eeg.TestDataset(np.asarray(test), targets),
                        batch_size=200, shuffle=False,
                        collate_fn=things_eeg.collate)
    feats = evaluate.extract_features(model, loader, device)
    return (feats["eeg"].astype(np.float64), feats["img"].astype(np.float64),
            feats.get("concept"))


def _ridge(A: np.ndarray, B: np.ndarray, lam: float) -> np.ndarray:
    d = A.shape[1]
    return np.linalg.solve(A.T @ A + lam * np.eye(d), A.T @ B)


def _resid(P: np.ndarray, Q: np.ndarray) -> float:
    """Normalised residual: ||P - Q||_F / ||Q||_F."""
    return float(np.linalg.norm(P - Q) / max(np.linalg.norm(Q), 1e-12))


def _orth_procrustes(A: np.ndarray, B: np.ndarray):
    """Best orthogonal `R` with `A R ~ B`, plus the residual. Requires equal dims."""
    u, s, vt = np.linalg.svd(A.T @ B)
    r = u @ vt
    return r, _resid(A @ r, B)


def _rbf_features(X: np.ndarray, D: int, gamma: float, seed: int = 0):
    """Random Fourier features for an RBF kernel -- a generic smooth-map basis."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(X.shape[1], D)) / max(np.sqrt(gamma), 1e-6)
    b = rng.uniform(0.0, 2.0 * np.pi, size=D)
    return np.sqrt(2.0 / D) * np.cos(X @ w + b)


def _zeromean(x: np.ndarray):
    return x - x.mean(axis=0, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 11)))
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--stage1-root", default="outputs/stage1/g3")
    ap.add_argument("--rff", type=int, default=1024, help="random Fourier feature count")
    ap.add_argument("--lam", type=float, default=1e-2, help="ridge for the smooth fit")
    ap.add_argument("--out", default="outputs/probe/rigidity.json")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out: dict = {"seed": args.seed, "stage1_root": args.stage1_root, "per_subject": {}}
    store: dict[int, tuple] = {}

    for s in args.subjects:
        ck = Path(args.stage1_root) / f"sub{s:02d}_k20_seed{args.seed}" / "last.pt"
        if not ck.is_file():
            print(f"[rigid] sub{s:02d}: missing {ck}")
            continue
        cache = Path("outputs/probe") / f"rigid_feats_sub{s:02d}_seed{args.seed}.npz"
        if cache.is_file():
            z = np.load(cache)
            ze, zi, con = z["z_e"].astype(np.float64), z["z_i"].astype(np.float64), z["con"]
        else:
            ze, zi, con = _extract(ck, s, device)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, z_e=ze, z_i=zi, con=con if con is not None else np.array([]))
        store[s] = (ze, zi, con)
        print(f"[rigid] sub{s:02d}: z_e {ze.shape}  z_i {zi.shape}")

    if not store:
        raise SystemExit("no features extracted")

    d_e = next(iter(store.values()))[0].shape[1]
    d_i = next(iter(store.values()))[1].shape[1]
    out["d_eeg"], out["d_img"] = int(d_e), int(d_i)
    equal_dims = d_e == d_i

    # ---------------------------------------------------------------- per subject
    for s, (ze, zi, _) in sorted(store.items()):
        # Centre both clouds first: an intercept is not a "non-rigidity", and leaving the
        # means in would make every map look non-rigid for a reason with no bearing on the
        # model class (the deployment operator moment-matches before it fits).
        A, B = _zeromean(ze), _zeromean(zi)
        rec: dict = {}
        if equal_dims:
            _, r_orth = _orth_procrustes(A, B)
            rec["resid_orth"] = r_orth
        Wl = _ridge(A, B, lam=1e-6)
        rec["resid_linear"] = _resid(A @ Wl, B)
        F = _rbf_features(A, args.rff, gamma=1.0 / max(A.shape[1], 1))
        Wr = _ridge(np.hstack([A, F]), B, lam=args.lam)
        rec["resid_smooth"] = _resid(np.hstack([A, F]) @ Wr, B)
        if equal_dims:
            rec["rigidity_gap"] = rec["resid_orth"] / max(rec["resid_smooth"], 1e-12)
        # residual spectrum after the ORTHOGONAL fit: is the non-rigid part low-rank?
        if equal_dims:
            u, sv, vt = np.linalg.svd(A.T @ B)
            E = A @ (u @ vt) - B
            svv = np.linalg.svd(E, compute_uv=False)
            e = svv ** 2
            rec["resid_energy_1d"] = float(e[0] / max(e.sum(), 1e-12))
            rec["resid_energy_8d"] = float(e[:8].sum() / max(e.sum(), 1e-12))
            rec["resid_energy_16d"] = float(e[:16].sum() / max(e.sum(), 1e-12))
        # intrinsic dimension of the concept cloud: participation ratio of the spectrum
        sc = np.linalg.svd(A, compute_uv=False) ** 2
        rec["concept_dim_partratio"] = float((sc.sum() ** 2) / max((sc ** 2).sum(), 1e-12))
        out["per_subject"][f"sub{s:02d}"] = rec

    # ------------------------------------------------- shared frame vs per-subject
    # The cross-subject quantity: fit ONE map on all subjects, then look at its per-subject
    # residual. `ratio > 1` means a shared frame does not exist and the held-out subject's
    # distortion is its own.
    Aall = np.vstack([_zeromean(store[s][0]) for s in sorted(store)])
    Ball = np.vstack([_zeromean(store[s][1]) for s in sorted(store)])
    Wall = _ridge(Aall, Ball, lam=1e-6)
    shared = {}
    for s in sorted(store):
        A, B = _zeromean(store[s][0]), _zeromean(store[s][1])
        Ws = _ridge(A, B, lam=1e-6)
        shared[f"sub{s:02d}"] = {
            "resid_shared_linear": _resid(A @ Wall, B),
            "resid_per_subject_linear": _resid(A @ Ws, B),
        }
    out["shared_frame"] = shared

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))

    # -------------------------------------------------------------------- report
    print("\n" + "=" * 84)
    print("1. PER-SUBJECT RIGIDITY GAP   (resid_orth / resid_smooth; 1.0 = rigid is correct)")
    print("-" * 84)
    print(f"{'subject':<9}{'orth':>9}{'linear':>9}{'smooth':>9}{'gap':>8}   "
          f"{'resid energy':>20}{'concept dim':>12}")
    print(f"{'':<9}{'':>9}{'':>9}{'':>9}{'':>8}   {'1d':>6}{'8d':>7}{'16d':>7}")
    gaps, dims = [], []
    for s, r in sorted(out["per_subject"].items()):
        gap = r.get("rigidity_gap")
        if gap is not None:
            gaps.append(gap)
        dims.append(r["concept_dim_partratio"])
        print(f"{s:<9}{r.get('resid_orth', float('nan')):>9.3f}{r['resid_linear']:>9.3f}"
              f"{r['resid_smooth']:>9.3f}{(gap if gap is not None else float('nan')):>8.2f}   "
              f"{r.get('resid_energy_1d', float('nan')):>6.2f}"
              f"{r.get('resid_energy_8d', float('nan')):>7.2f}"
              f"{r.get('resid_energy_16d', float('nan')):>7.2f}{r['concept_dim_partratio']:>12.1f}")
    print("-" * 84)
    if gaps:
        print(f"mean rigidity gap = {np.mean(gaps):.2f}   (min {min(gaps):.2f}, max {max(gaps):.2f})")
    print(f"mean intrinsic concept dimension (participation ratio) = {np.mean(dims):.1f}")

    print("\n2. SHARED FRAME vs PER-SUBJECT MAP  (ratio > 1 => distortion is subject-specific)")
    print("-" * 84)
    ratios = []
    for s, r in sorted(out["shared_frame"].items()):
        rr = r["resid_shared_linear"] / max(r["resid_per_subject_linear"], 1e-12)
        ratios.append(rr)
        print(f"  {s}   shared {r['resid_shared_linear']:.3f}   "
              f"per-subject {r['resid_per_subject_linear']:.3f}   ratio {rr:.2f}x")
    print("-" * 84)
    print(f"mean shared/per-subject ratio = {np.mean(ratios):.2f}x")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
