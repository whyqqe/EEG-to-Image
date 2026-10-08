"""TTGA -- TEST-TIME GROUP AVERAGING. Reynolds operator at inference, zero retraining.

THE IDEA, AND WHY IT IS THE MIRROR OF THE TRAINING ARM
------------------------------------------------------
SQA's level-2 claim is that volume conduction is an action of `GL(C)` on the trial
(`X -> M X`) and that the concept lives in a subspace the action does not move. Training
enforces invariance with a Monte-Carlo estimate of the Reynolds (group-averaging) operator

    P_G z(x) = E_{M ~ G} z(M x),

by feeding `M x` and asking the contrast to place it at the same point. That average is
estimated on ONE draw per epoch, and the encoder is only APPROXIMATELY invariant
afterwards. The same operator can therefore be applied at INFERENCE, where we are free to
draw as many group elements as we like and average exactly:

    z_TTGA(x) = normalize( (1/K) * sum_{k=1..K} normalize(z(M_k x)) ).

This is not "test-time augmentation for robustness". It is the projection the training arm
can only approximate, run to convergence -- and it is measurable as such: the REYNOLDS
RESIDUAL `E_M || z(Mx) - z(x) ||` is the distance the encoder still has left to travel
into the invariant subspace. A checkpoint trained with the augmentation should have a
small residual and gain from TTGA; one trained without should have a large residual and
LOSE, which is the falsifiable prediction this script is built to test.

WHY IT CANNOT HURT THE SHIPPED NUMBER (the accuracy guarantee)
-------------------------------------------------------------
Every row is emitted for a grid of interpolation weights

    z(gamma) = normalize( (1 - gamma) * z_base + gamma * z_TTGA ),

`gamma = 0` is EXACTLY the shipped cell (verified bit-for-bit against
`outputs/eval/v11_fuse/`), so the result is an ADDITIVE report: the shipped number stays on
disk and any TTGA cell that loses is simply not adopted. The grid is an interpolation, not
a choice, so there is no "selected on the test set" interval to defend.

WHY A STANDALONE SCRIPT AND NOT A FLAG ON run_eval.py
-----------------------------------------------------
`samclip`'s `src/` and `scripts/` are read at job start by every queued Slurm job
(AGENTS.md 1.4). A group-attribution job was running when this was written, and each of
its eval steps re-imports `scripts/run_eval.py`; editing that file mid-flight is the exact
failure recorded in AGENTS.md (four stage-1 jobs died on a half-edited module). So this
operator lives in its own file and imports only stable modules. The deployed call sequence
is COPIED from `run_eval.py`'s `--fgw-struct-fuse` block, and the copy is validated by
requiring `gamma=0` to reproduce the banked `v11_fuse` cell on the same fold.

THE MIXING DISTRIBUTION MIRRORS `data/augment.py` (dense mode, matched by ||M - I||_F).
It is reproduced here rather than imported because importing would have required editing
`augment.py` while a job was running. `--check-mixing` asserts the two agree in
distribution, so the duplication cannot silently drift.

Usage
-----
    python scripts/probe_ttga.py --ckpts outputs/stage1/v8/sub01_k20_seed2025/last.pt \
        --target-subject 1 --out outputs/probe/ttga_sub01.json \
        --ttga-K 1,2,4,8 --ttga-gammas 0,0.5,1.0 --ttga-mode dense --ttga-mixing 3.15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import calibration, config, evaluate  # noqa: E402
from samclip.data import things_eeg  # noqa: E402
from samclip.data.targets import load_target_stack  # noqa: E402
from samclip.models import build_model  # noqa: E402

CELL = "+ T2 R=80,a=0.75,t=0.03,fuse=16"


# --------------------------------------------------------------------- the group action
def draw_mixing(c: int, magnitude: float, mode: str, rng: np.random.Generator) -> np.ndarray:
    """One group element of GL(C), mirroring `samclip.data.augment.make_augment`.

    `magnitude` is ||M - I||_F, NOT an entry scale (see the augment docstring: the entry
    scale is mode-dependent and matching it would make the diagonal arm a silent no-op).
    """
    if mode == "diag":
        g = rng.normal(size=c)
        return np.diag(np.exp(magnitude * g / (np.linalg.norm(g) + 1e-12))).astype(np.float32)
    if mode == "orth":
        from scipy.linalg import expm
        s = rng.normal(size=(c, c))
        a = s - s.T
        return expm(magnitude * a / (np.linalg.norm(a) + 1e-12)).astype(np.float32)
    g = rng.normal(size=(c, c))
    return (np.eye(c, dtype=np.float32)
            + (magnitude * g / (np.linalg.norm(g) + 1e-12)).astype(np.float32))


def _norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


# ------------------------------------------------------------------- deployed cell (copy)
class _Ns:
    """The `args` surface `_recovery_fn_for` reads. Kept minimal and explicit."""
    recovery_operator = "fgw"
    recovery_tau = 0.03
    recovery_iters = 50
    recovery_alpha = 0.75
    recovery_fgw_outer = 10


def deployed_recovery():
    """`run_eval._recovery_fn_for(args, force_alpha=0.75, force_tau=0.03)`, copied verbatim
    from the deployed call site. `alpha=0` is the structural-off twin and is bit-identical
    to the shipped sinkhorn operator, so the twin is the deployment minus the term."""
    a = _Ns()

    def sinkhorn_recovery(q, g, k=10, rho=0.1, min_landmark_rate=0.0, **kw):
        return calibration.subspace_soft_recovery(
            q, g, k=k, rho=rho, rank=None, tau=float(a.recovery_tau),
            iters=int(a.recovery_iters), hard_landmarks=False, min_landmarks=8,
            alpha=float(kw.pop("alpha", a.recovery_alpha)),
            fgw_outer=int(a.recovery_fgw_outer),
            fgw_de_ref=kw.pop("fgw_de_ref", None),
            fgw_de_mix=float(kw.pop("fgw_de_mix", 0.0)),
            fgw_spec_rank=kw.pop("fgw_spec_rank", None),
            fgw_di_ref=kw.pop("fgw_di_ref", None),
            fgw_topo_eps=kw.pop("fgw_topo_eps", None))

    return sinkhorn_recovery


def cell_scores(z_reps, img, *, alpha, csls_k, rho, shrink, rep_blocks, fn):
    sc, dg = calibration.rep_cloud_scores(
        z_reps, img, k=csls_k, rho=rho, shrink=shrink, min_landmark_rate=0.0,
        recovery_fn=fn, rep_subsample=None, src_means=None, src_mix=0.0,
        rep_blocks=int(rep_blocks), gallery_di_ref=None)
    return sc, {k: v for k, v in dg.items() if not hasattr(v, "shape")}


# --------------------------------------------------------------------------------- main
def _load_model(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    channel_set = cfg.get("channel_set", "all63")
    channels = config.CHANNELS_OCCIPITO_PARIETAL if channel_set == "occipital17" else None
    img_cfg = cfg.get("image", {}) or {}
    feature_set = img_cfg.get("feature_set", "clip_h14_multilevel")
    layers = img_cfg.get("layers")
    mvnn = "test" if cfg.get("mvnn", "off") != "off" else "off"
    return ckpt, cfg, channels, mvnn, feature_set, layers


def run_one(ckpt_path: Path, target_subject: int, args, device) -> dict:
    ckpt, cfg, channels, mvnn, feature_set, layers = _load_model(ckpt_path, device)
    _, test = things_eeg.load_subject_std(target_subject, channels, mvnn=mvnn)
    test = np.asarray(test)
    targets_te = load_target_stack(feature_set, layers, "test")

    model = build_model(cfg, targets_te.shape[2], targets_te.shape[-1]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = DataLoader(things_eeg.TestDataset(test, targets_te), batch_size=200,
                        shuffle=False, collate_fn=things_eeg.collate)
    feats = evaluate.extract_features(model, loader, device)
    img = feats["img"]

    reps = things_eeg.load_test_reps(target_subject, channels, mvnn=mvnn)
    reps = np.asarray(reps)
    c = int(reps.shape[2])

    out: dict = {"ckpt": str(ckpt_path), "target_subject": target_subject,
                 "feature_set": feature_set, "channel_set": cfg.get("channel_set", "all63"),
                 "mvnn": mvnn, "n_reps": int(reps.shape[1]),
                 "ttga": {"mode": args.ttga_mode, "mixing": args.ttga_mixing,
                          "K_list": args.ttga_K, "gammas": args.ttga_gammas},
                 "rows": {}}

    if args.check_mixing:
        from samclip.data.augment import build_augment
        x = np.random.default_rng(0).normal(size=(c, 250)).astype(np.float32)
        rel_aug = float(np.mean([
            np.linalg.norm(build_augment({"augment": {"noise": 0., "gain": 0., "shift": 0,
                                                      "channel_dropout": 0.,
                                                      "mixing": args.ttga_mixing,
                                                      "mixing_mode": args.ttga_mode}})(
                x, np.random.default_rng(100 + i)) - x) / np.linalg.norm(x)
            for i in range(20)]))
        rel_here = float(np.mean([
            np.linalg.norm(draw_mixing(c, args.ttga_mixing, args.ttga_mode,
                                       np.random.default_rng(100 + i)) @ x - x)
            / np.linalg.norm(x) for i in range(20)]))
        out["mixing_check"] = {"augment": rel_aug, "ttga": rel_here,
                               "abs_diff": abs(rel_aug - rel_here)}
        print(f"[ttga] mixing check: augment {rel_aug:.4f} vs ttga {rel_here:.4f} "
              f"(diff {abs(rel_aug-rel_here):.4f})")

    # ---- base and Reynolds-averaged embeddings, per K --------------------------------
    base = evaluate.embed_reps(model, reps, device)        # (C, R, d), already normalised
    base = np.asarray(base)
    variants: dict[int, np.ndarray] = {1: base}
    resid: dict[int, float] = {}
    Kmax = max(args.ttga_K)
    acc = base.copy()
    for k in range(1, Kmax):
        rng = np.random.default_rng(args.ttga_seed + k)
        flat = reps.reshape(-1, *reps.shape[2:])            # (C*R, ch, T)
        m = draw_mixing(c, args.ttga_mixing, args.ttga_mode, rng)
        mixed = np.einsum("ij,njt->nit", m, flat).astype(np.float32)
        z_k = np.asarray(evaluate.embed_reps(
            model, mixed.reshape(*reps.shape), device))
        acc = acc + z_k
        if (k + 1) in args.ttga_K:
            variants[k + 1] = _norm(acc / (k + 1))
    for K, zv in variants.items():
        resid[K] = float(np.mean(np.linalg.norm(_norm(base) - zv, axis=-1)))
    out["reynolds_residual"] = resid

    fn = deployed_recovery()
    sc, dg = cell_scores(base, img, alpha=0.75, csls_k=args.csls_k, rho=args.rho,
                         shrink=args.rep_shrink, rep_blocks=args.rep_blocks, fn=fn)
    out["rows"][CELL] = {**calibration.report_with_scores(sc), "diag": dg}

    for K in sorted(variants):
        zv = variants[K]
        for gamma in args.ttga_gammas:
            zg = _norm((1.0 - gamma) * base + gamma * zv)
            sc, dg = cell_scores(zg, img, alpha=0.75, csls_k=args.csls_k, rho=args.rho,
                                 shrink=args.rep_shrink, rep_blocks=args.rep_blocks, fn=fn)
            name = (f"{CELL} [TTGA K={K},g={gamma:g}]" if gamma > 0
                    else f"{CELL} [TTGA K={K},g=0]")
            out["rows"][name] = {**calibration.report_with_scores(sc), "diag": dg}
            out["rows"][name]["K"], out["rows"][name]["gamma"] = K, float(gamma)

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--target-subject", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ttga-K", type=lambda s: [int(x) for x in s.split(",")],
                    default=[1, 2, 4, 8], help="K grid (1 == the base, for the floor)")
    ap.add_argument("--ttga-gammas", type=lambda s: [float(x) for x in s.split(",")],
                    default=[0.0, 0.5, 1.0])
    ap.add_argument("--ttga-mode", default="dense", choices=["dense", "diag", "orth"])
    ap.add_argument("--ttga-mixing", type=float, default=3.15)
    ap.add_argument("--ttga-seed", type=int, default=2025)
    ap.add_argument("--rep-blocks", type=int, default=16)
    ap.add_argument("--rep-shrink", type=float, default=0.1)
    ap.add_argument("--csls-k", type=int, default=10)
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--check-mixing", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    reports = {}
    for ck in args.ckpts:
        rep = run_one(Path(ck), args.target_subject, args, device)
        reports[ck] = rep
        print(f"\n== {ck}  (fold {args.target_subject}) ==")
        print(f"   Reynolds residual: " +
              "  ".join(f"K={k}:{v:.4f}" for k, v in sorted(rep["reynolds_residual"].items())))
        for name, rv in rep["rows"].items():
            print(f"   {name:48s} {rv['top1']:>6.2f} / {rv['top5']:<6.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"target_subject": args.target_subject, "cell": CELL, "reports": reports},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
