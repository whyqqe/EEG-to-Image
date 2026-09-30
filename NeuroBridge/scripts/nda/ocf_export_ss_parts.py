#!/usr/bin/env python3
"""Export the DECORRELATED parts of the HCMA encoder, and test whether they
should be routed to different towers.

FACT (read from the code, not assumed)
--------------------------------------
`SharedSpecificEncoder` (design string in its checkpoint: "MindCross-shared+
specific + MindBridge-calibrate") is NOT a semantic/structural dual tower.  It is
a CROSS-SUBJECT module:

    r = self.shared(x)                      # EEGProjectWide,   1024-d  (shared)
    s[mask] = self.specific[subj](flat)     # SubjectEmbedder,  1024-d  (per-subject)
    fused = self.fuse(s, r) = mlp(cat(s,r)) + r
    out  = self.adapters[subj](fused) if present
    z_eeg_proj = ProjectorLinear(1024 -> 512)(out)               # what the pipeline keeps

`forward(..., return_parts=True)` already returns `(out, s, r)`, and pretraining
adds `diff_loss(s, r) = MSE(s * r, 0)` with `lambda_diff = 0.1`, i.e. the encoder
was TRAINED to make `s` and `r` orthogonal.  But `nda_ss_encode.py` keeps only
`eeg_projector(fused)` and discards both parts.

WHY THAT MATTERS
----------------
  * the semantic readout needs subject INVARIANCE (a new subject's EEG must land
    in the same CLIP space)  -> that is what `r` (shared) was built for,
    and it is what diff_loss protects;
  * the layout readout needs subject-SPECIFIC geometry (retinotopy, head
    geometry, cortical folding -- where things are) -> that is what `s`
    (specific) carries, and it is exactly the component diff_loss REMOVED from
    the shared part.
So the encoder's own training objective says: shared -> semantics, specific ->
geometry.  The pipeline instead feeds the same `fused` to everything, and
`ResFuse(s, r) = mlp(...) + r` biases `fused` TOWARD the invariant component,
which is the wrong prior for geometry.  This is a candidate explanation for the
weak layout channel (cos(pred, GT) = 0.261).

WHAT THIS SCRIPT DOES
---------------------
 1. re-exports `fused`, `s`, `r` and `z_eeg_proj` for sub-08, and VERIFIES the
    re-export reproduces the stored `z_eeg_proj_*.npy` -- if it does not, the
    wrong checkpoint was picked and every downstream number would be garbage;
 2. fits ONE ridge regression per candidate representation, onto (A) the CLIP
    image target and (B) the low-frequency VAE latent, using TRAIN rows only;
 3. scores each on TEST rows with top-1 / 2-way / cosine.

The prediction, stated in advance so the result can falsify it:
    semantics:  r (shared)      >= fused  >= s (specific)
    layout:     s (specific)    >  fused  >  r (shared)
If `s` does NOT beat `fused` on layout, the routing idea is dead and the OCF
architecture must keep the learned-tower design instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT))
sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))

DEFAULT_CHANNELS = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]


def l2n(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x).astype(np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def radius_grid(h: int, w: int) -> np.ndarray:
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    return np.sqrt(fy ** 2 + fx ** 2)


def export_parts(args) -> dict[str, np.ndarray]:
    from module.dataset import EEGPreImageDataset  # noqa: E402
    from module.projector import ProjectorLinear  # noqa: E402
    from ss_modules import SharedSpecificEncoder  # noqa: E402

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    subjects = [int(s) for s in ck["subjects"]]
    img_dim, feature_dim = int(ck["img_dim"]), int(ck["feature_dim"])
    eeg_len, n_ch = int(ck["eeg_sample_points"]), int(ck["channels_num"])
    if args.subject not in subjects:
        raise SystemExit(f"[FATAL] subject {args.subject} not in ckpt subjects {subjects}; "
                         f"calibrate it first with calibrate_ss_new_subject.py")
    print(f"[parts] ckpt phase={ck.get('phase')} epoch={ck.get('epoch')} "
          f"lambda_diff={ck.get('lambda_diff')} subjects={subjects}")

    model = SharedSpecificEncoder(
        subject_ids=subjects, feature_dim=img_dim, eeg_sample_points=eeg_len,
        channels_num=n_ch, n_extra_blocks=int(ck.get("n_extra_blocks", 1)),
        use_adapter=True,
    ).to(dev)
    proj = ProjectorLinear(img_dim, feature_dim).to(dev)
    model.load_state_dict(ck["model_state_dict"])
    proj.load_state_dict(ck["eeg_projector_state_dict"])
    model.eval()
    proj.eval()

    eeg_dir = f"{NB_ROOT}/data/things_eeg/preprocessed_eeg"
    rn50_dir = f"{NB_ROOT}/data/things_eeg/image_feature/RN50"
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    got: dict[str, list[np.ndarray]] = {}
    for train_flag, tag in ((True, "train"), (False, "test")):
        ds = EEGPreImageDataset(
            [args.subject], eeg_dir, DEFAULT_CHANNELS, [0, 250],
            rn50_dir, "", False, [], True, False, None, train_flag, False, False, False,
        )
        acc: dict[str, list[np.ndarray]] = {}
        with torch.no_grad():
            for batch in DataLoader(ds, batch_size=512, shuffle=False):
                eeg, _img, _t, sid, *_ = batch
                eeg, sid = eeg.to(dev), sid.to(dev)
                out, s, r = model(eeg, sid, return_parts=True)
                z = proj(out)
                for k, v in (("fused", out), ("shared_r", r), ("specific_s", s),
                             ("z_eeg_proj", z)):
                    acc.setdefault(k, []).append(v.float().cpu().numpy())
        arrs = {k: np.concatenate(v, 0).astype(np.float32) for k, v in acc.items()}
        # VERIFY against the stored export -- wrong checkpoint => garbage downstream
        ref_p = Path(args.stored_dir) / f"z_eeg_proj_{tag}.npy"
        rep = {}
        if ref_p.is_file():
            ref = np.load(ref_p).astype(np.float32)
            rep["shape_ok"] = bool(ref.shape == arrs["z_eeg_proj"].shape)
            if rep["shape_ok"]:
                a, b = l2n(arrs["z_eeg_proj"]), l2n(ref)
                rep["cos_mean"] = float((a * b).sum(1).mean())
                rep["cos_min"] = float((a * b).sum(1).min())
                rep["max_abs_diff"] = float(np.abs(arrs["z_eeg_proj"] - ref).max())
            print(f"[parts] VERIFY {tag}: {rep}")
            if rep.get("cos_mean", 0) < 0.99:
                raise SystemExit(
                    f"[FATAL] re-exported z_eeg_proj does not reproduce {ref_p} "
                    f"(cos {rep.get('cos_mean')}). Wrong checkpoint/seed/dataset order "
                    f"-- every downstream comparison would be invalid.")
        for k, v in arrs.items():
            np.save(cache / f"{k}_{tag}.npy", v)
            got.setdefault(k, []).append(v)
        (cache / f"verify_{tag}.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"[parts] {tag}: " + " ".join(f"{k}{v.shape}" for k, v in arrs.items()))
    return got


def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form ridge on standardised X.  (d,d) solve, no (n,n) anywhere."""
    X = X.astype(np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    XtX = Xs.T @ Xs
    XtY = Xs.T @ Y.astype(np.float64)
    W = np.linalg.solve(XtX + lam * np.eye(XtX.shape[0]), XtY)
    return W.astype(np.float32), mu.astype(np.float32), sd.astype(np.float32)


def ridge_apply(model: tuple[np.ndarray, np.ndarray, np.ndarray], X: np.ndarray) -> np.ndarray:
    W, mu, sd = model
    return ((X.astype(np.float32) - mu) / sd) @ W


def rank_metrics(pred: np.ndarray, targ: np.ndarray) -> dict:
    """2-way identification and top-1, on cosine ranking.  Scale invariant, so it
    does not reward a channel merely for having a better gain."""
    p, t = l2n(pred), l2n(targ)
    s = p @ t.T
    n = len(s)
    idx = np.arange(n)
    r = np.random.default_rng(0).permutation(n)
    ok = idx != r
    return {
        "top1": float(np.mean(s.argmax(1) == idx)),
        "top5": float(np.mean([idx[i] in np.argsort(-s[i])[:5] for i in range(n)])),
        "twoway": float(np.mean(s[idx, idx][ok] > s[idx, r][ok])),
        "cos_mean": float((p * t).sum(1).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--checkpoint", type=str,
                    default=f"{NB_ROOT}/outputs/nda_ss/sub-08/ss/checkpoint_ss_calib_best.pth")
    ap.add_argument("--stored-dir", type=str, default=f"{NB_ROOT}/outputs/hcma_10subj/sub-08/zret")
    ap.add_argument("--cache-dir", type=str, default="",
                    help="defaults to <NB_ROOT>/outputs/ocf/ss_parts/sub-<SS>")
    ap.add_argument("--targets-dir", type=str, default=f"{NB_ROOT}/outputs/g2/targets")
    ap.add_argument("--clip-train", type=str,
                    default="/project/peilab/why/eeg-brainit/outputs/atm_bridge/clip_img_train_1024.npy")
    ap.add_argument("--clip-test", type=str,
                    default="/project/peilab/why/eeg-brainit/outputs/atm_bridge/clip_img_test_1024.npy")
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--n-pca", type=int, default=512)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--out-json", type=str, default=f"{NB_ROOT}/outputs/ocf/ss_parts_probe.json")
    args = ap.parse_args()

    if not args.cache_dir:
        args.cache_dir = f"{NB_ROOT}/outputs/ocf/ss_parts/sub-{args.subject:02d}"
    if not args.stored_dir or args.stored_dir.endswith("sub-08/zret"):
        args.stored_dir = f"{NB_ROOT}/outputs/hcma_10subj/sub-{args.subject:02d}/zret"

    cache = Path(args.cache_dir)
    need = cache / "shared_r_train.npy"
    if need.is_file():
        print(f"[parts] using cache {cache}")
        tr = {k: np.load(cache / f"{k}_train.npy") for k in
              ("fused", "shared_r", "specific_s", "z_eeg_proj")}
        te = {k: np.load(cache / f"{k}_test.npy") for k in
              ("fused", "shared_r", "specific_s", "z_eeg_proj")}
    else:
        g = export_parts(args)
        tr = {k: v[0] for k, v in g.items()}
        te = {k: v[1] for k, v in g.items()}

    tr["cat"] = np.concatenate([tr["shared_r"], tr["specific_s"]], 1)
    te["cat"] = np.concatenate([te["shared_r"], te["specific_s"]], 1)

    # ---------------- task A: semantics (CLIP image space)
    Ctr = np.load(args.clip_train).astype(np.float32)
    Cte = np.load(args.clip_test).astype(np.float32)
    print(f"\n{'='*84}\nTASK A  semantics: ridge(rep) -> CLIP-1024, train-only fit, scored on test\n{'='*84}")
    print(f"{'rep':<14}{'dim':>6}{'top1':>9}{'top5':>9}{'2way':>9}{'cos':>9}")
    rep_a: dict[str, dict] = {}
    for k in ("z_eeg_proj", "fused", "shared_r", "specific_s", "cat"):
        W = ridge_fit(tr[k], Ctr, args.lam)
        m = rank_metrics(ridge_apply(W, te[k]), Cte)
        rep_a[k] = {"dim": int(tr[k].shape[1]), **m}
        print(f"{k:<14}{tr[k].shape[1]:>6}{m['top1']:>9.4f}{m['top5']:>9.4f}{m['twoway']:>9.4f}{m['cos_mean']:>9.4f}")

    # ---------------- task B: layout (low-frequency VAE latent)
    # PCA is fit on TRAIN latents only; 2-way is a ranking metric evaluated in
    # that same reduced space for both prediction and target, so it is a fair
    # comparison across representations.
    Str = np.load(f"{args.targets_dir}/perc_struct_train.npy").astype(np.float32).reshape(16540, -1)
    Ste = np.load(f"{args.targets_dir}/perc_struct_test.npy").astype(np.float32).reshape(200, -1)
    mu = Str.mean(0, keepdims=True)
    Str_c = Str - mu
    # PCA via SVD on a train subsample of columns: use eigendecomposition of the
    # (d,d) covariance instead, which is 16384^2 = 1.07e9 -- too big.  So reduce
    # with a random projection-free approach: SVD on (n,d) with n=16540 is fine
    # but memory heavy; use the top-k right singular vectors from a Gram trick on
    # 4096 sampled rows.
    rs = np.random.default_rng(0)
    sub = rs.choice(len(Str_c), 4096, replace=False)
    A = Str_c[sub]
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt[: args.n_pca].T
    Str_r, Ste_r = Str_c @ V, (Ste - mu) @ V
    print(f"\n{'='*84}\nTASK B  layout: ridge(rep) -> LF-VAE latent (PCA {args.n_pca} on train)\n{'='*84}")
    print(f"{'rep':<14}{'dim':>6}{'top1':>9}{'top5':>9}{'2way':>9}{'cos':>9}")
    rep_b: dict[str, dict] = {}
    for k in ("z_eeg_proj", "fused", "shared_r", "specific_s", "cat"):
        W = ridge_fit(tr[k], Str_r, args.lam)
        m = rank_metrics(ridge_apply(W, te[k]), Ste_r)
        rep_b[k] = {"dim": int(tr[k].shape[1]), **m}
        print(f"{k:<14}{tr[k].shape[1]:>6}{m['top1']:>9.4f}{m['top5']:>9.4f}{m['twoway']:>9.4f}{m['cos_mean']:>9.4f}")

    # ---------------- vertical check: does shared->sem and specific->layout hold?
    verdict = {
        "semantics_shared_beats_specific": rep_a["shared_r"]["twoway"] > rep_a["specific_s"]["twoway"],
        "layout_specific_beats_shared": rep_b["specific_s"]["twoway"] > rep_b["shared_r"]["twoway"],
        "semantics_shared_beats_fused": rep_a["shared_r"]["twoway"] >= rep_a["fused"]["twoway"] - 1e-4,
        "layout_specific_beats_fused": rep_b["specific_s"]["twoway"] > rep_b["fused"]["twoway"] + 1e-4,
        "layout_cat_beats_fused": rep_b["cat"]["twoway"] > rep_b["fused"]["twoway"] + 1e-4,
    }
    print(f"\n{'='*84}\nVERDICT (the prediction was stated in advance in the docstring)\n{'='*84}")
    for k, v in verdict.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")

    out = {"subject": args.subject, "checkpoint": args.checkpoint,
           "semantics": rep_a, "layout": rep_b, "verdict": verdict,
           "dim_pca": args.n_pca, "lam": args.lam,
           "note": ("ridge is train-only; the PCA basis is fit on TRAIN LF latents only; "
                    "2-way is evaluated in that common reduced space for every rep, so "
                    "the comparison across representations is apples-to-apples.")}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[save] {args.out_json}")


if __name__ == "__main__":
    main()
