"""NW4 Stage 2 — closed-form manifold projection (replaces nw3's 20-step sampler).

Why nw3's S2 had to go
----------------------
nw3 refined `z_sem` with a 20-step denoiser started from N(0, I).  Measured via
`nw3_cond_audit`: the condition bank's row-to-row redundancy (`offdiag` cosine)
went 0.101 (S1) -> 0.567 (S2) -> 0.699 (S3, i.e. the vector actually fed to the
generator), with 1.0 = a constant vector.  Sampling from the prior dominates the
condition, so the generator was asked to render a near-constant "something
plausible" -- which is exactly the measured result: Incep 0.5110 (init) ->
0.5236 (best of 6 arms), i.e. +0.013 over chance for the whole generation grid.

What replaces it
----------------
A sparse, closed-form projection that CANNOT leave the manifold:

    sim  = u @ B.T                 B = real train bank (trial rows)
    idx  = topK(sim)               K small (default 2)
    base = l2n(softmax(sim[idx]/tau) @ B[idx])   <- convex combination of REAL rows
    r    = u - (u . base) base     <- component orthogonal to the base
    z    = l2n(base + gamma * r)   <- gamma preserves the trial-specific residual

`base` is a convex combination of real bank rows, so it lies on the manifold by
construction -- no sampler, no prior entropy competing with the condition.  The
residual keeps per-trial individuality that a mean would destroy (that was nw3's
S3 failure: `offdiag` 0.699, i.e. fusion collapsed to the mean).

Every linear map here is fitted on `fit` rows only; `val_b` picks alpha; the test
bank is touched once, for reporting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))

VITH_LEVELS = ["image", "GaussianBlur", "LowResolution", "Mosaic", "GaussianNoise"]
ATTR_FIELDS = ["overall", "subject", "detail"]


def l2n(x: np.ndarray) -> np.ndarray:
    """L2-normalise.  Deliberately keeps float32: these arrays are (16540, 5120) and
    promoting them to float64 costs ~2.7 GB across the five banks.  float32 norms on
    a few thousand dims are far from overflow, and every solver that needs precision
    (`fit_ridge`) casts to float64 itself."""
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


# --------------------------------------------------------------------------
# diagnostics: the numbers that must IMPROVE vs nw3's S2/S3
# --------------------------------------------------------------------------
def row2mean(z: np.ndarray) -> float:
    zn = l2n(z)
    mu = l2n(zn.mean(0, keepdims=True))
    return float((zn * mu).sum(1).mean())


def offdiag(z: np.ndarray) -> float:
    s = l2n(z) @ l2n(z).T
    n = len(s)
    return float(s[~np.eye(n, dtype=bool)].mean())


def erank(z: np.ndarray) -> float:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.mean(0, keepdims=True)
    sv = np.linalg.svd(z, compute_uv=False)
    p = sv ** 2
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def retrieval(q: np.ndarray, bank: np.ndarray, k: int = 10) -> dict:
    """200-way top-1/top5 vs the test bank (row i <-> concept i), plus CSLS."""
    qn, bn = l2n(q), l2n(bank)
    sim = qn @ bn.T
    n = len(qn)
    order = np.argsort(-sim, 1)
    # knn_q is (Q, 1), knn_b is (1, B) -- do NOT transpose knn_b (see nw4_s1_train).
    k = max(1, min(k, sim.shape[0] - 1, sim.shape[1] - 1))
    knn_q = np.sort(sim, 1)[:, -k:].mean(1, keepdims=True)
    knn_b = np.sort(sim, 0)[-k:, :].mean(0, keepdims=True)
    csls = sim - 0.5 * (knn_q + knn_b)
    return {"top1": float((order[:, 0] == np.arange(n)).mean()),
            "top5": float(np.mean([i in order[i, :5] for i in range(n)])),
            "top1_csls": float((csls.argmax(1) == np.arange(n)).mean())}


# --------------------------------------------------------------------------
# closed-form building blocks
# --------------------------------------------------------------------------
def fit_ridge(X: np.ndarray, Y: np.ndarray, alphas: list[float],
              val_x: np.ndarray | None = None, val_gal: np.ndarray | None = None,
              val_cid: np.ndarray | None = None) -> dict:
    """Closed-form ridge X->Y.  alpha chosen on val_b concept top-1 when provided."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    XtX = X.T @ X
    XtY = X.T @ Y
    I = np.eye(X.shape[1])
    if val_x is None or val_gal is None:
        a = alphas[len(alphas) // 2]
        return {"W": np.linalg.solve(XtX + a * I, XtY), "alpha": a, "val_top1": float("nan")}
    g = l2n(val_gal)
    best = {"W": None, "alpha": None, "val_top1": -1.0}
    for a in alphas:
        W = np.linalg.solve(XtX + a * I, XtY)
        q = l2n(val_x @ W)
        t1 = float((q @ g.T).argmax(1).__eq__(val_cid).mean())
        if t1 > best["val_top1"]:
            best = {"W": W, "alpha": a, "val_top1": t1}
    return best


def manifold_project(u: np.ndarray, bank: np.ndarray, k: int, tau: float,
                     gamma: float) -> tuple[np.ndarray, dict]:
    """Sparse convex projection onto real bank rows + orthogonal residual."""
    un, bn = l2n(u), l2n(bank)
    sim = un @ bn.T
    kk = max(1, min(k, sim.shape[1]))
    part = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
    val = np.take_along_axis(sim, part, 1)
    w = np.exp((val - val.max(1, keepdims=True)) / max(tau, 1e-6))
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
    base = l2n((w[:, :, None] * bn[part]).sum(1))
    resid = un - (un * base).sum(1, keepdims=True) * base
    z = l2n(base + gamma * resid)
    diag = {"nn_top1_cos": float(np.sort(sim, 1)[:, -1].mean()),
            "proj_entropy": float(-(w * np.log(np.maximum(w, 1e-12))).sum(1).mean()),
            "resid_frac": float(np.linalg.norm(resid, axis=1).mean())}
    return z.astype(np.float32), diag


def concept_means(rows: np.ndarray, cid: np.ndarray, n_cls: int) -> np.ndarray:
    acc = np.zeros((n_cls, rows.shape[1]), dtype=np.float64)
    cnt = np.bincount(cid, minlength=n_cls).astype(np.float64)
    np.add.at(acc, cid, rows.astype(np.float64))
    return l2n(acc / np.maximum(cnt, 1)[:, None])


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", type=str, required=True, help="NW4 S1 output dir (has best.pth)")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--test-subject", type=int, default=8)
    ap.add_argument("--z-root", type=str, default=str(NB_ROOT / "outputs/ocf/intra_z"))
    ap.add_argument("--cond-cache", type=str, default=str(NB_ROOT / "outputs/gem/cond_cache"))
    ap.add_argument("--g2-targets", type=str, default=str(NB_ROOT / "outputs/g2/targets"))
    ap.add_argument("--split-json", type=str, default=str(NB_ROOT / "outputs/leakfree/split.json"))
    # bank spec: name=kind:ref   kind in {clip:<name>, vith, attr}
    ap.add_argument("--banks", type=str,
                    default="img=clip:clip_img1024,"
                            "attr=attr:all,"
                            "vith=vith:cat5,"
                            "depth=clip:clip_depth1024,"
                            "edge=clip:clip_edge1024")
    ap.add_argument("--emit", type=str, default="img,attr,depth,edge",
                    help="which projected conditions to write for the generator")
    ap.add_argument("--vith-levels", type=str, default=",".join(VITH_LEVELS))
    ap.add_argument("--attr-fields", type=str, default=",".join(ATTR_FIELDS))
    ap.add_argument("--alphas", type=str, default="0.1,1,10,100,1000")
    ap.add_argument("--proj-k", type=int, default=2)
    ap.add_argument("--proj-tau", type=float, default=0.07)
    ap.add_argument("--proj-gamma", type=float, default=0.20)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--allow-cpu", type=int, default=0)
    ap.add_argument("--json-out", type=str, default="")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import leakfree as LF  # noqa: E402
    from nw4_s1_train import FactorizedEncoder  # noqa: E402

    from dev_guard import pick_device  # noqa: E402
    dev = pick_device(args.device, allow_cpu=bool(args.allow_cpu))
    sid = f"{args.test_subject:02d}"
    s1d, out = Path(args.s1), Path(args.out)
    (out / "conds").mkdir(parents=True, exist_ok=True)
    alphas = [float(x) for x in args.alphas.split(",")]

    ztr = l2n(np.load(Path(args.z_root) / f"sub-{sid}" / "shared_r_train.npy").astype(np.float32))
    zte = l2n(np.load(Path(args.z_root) / f"sub-{sid}" / "shared_r_test.npy").astype(np.float32))
    n_flat = len(ztr)
    cc, g2 = Path(args.cond_cache), Path(args.g2_targets)

    # ---- concept ids (THINGS-EEG2 train is concept-major, verified by nw4_preflight) ----
    reps = n_flat // 1654
    if reps * 1654 != n_flat:
        raise SystemExit(f"[FATAL] cannot infer reps from n_flat={n_flat}")
    cid_tr = np.arange(n_flat, dtype=np.int64) // reps
    n_cls = 1654
    print(f"[nw4-s2] {n_cls} concepts x {reps} reps = {n_flat}")

    # ---- build the requested banks ----
    def build(spec: str) -> tuple[np.ndarray, np.ndarray]:
        kind, _, ref = spec.partition(":")
        if kind == "clip":
            return (np.load(cc / f"{ref}_train.npy").astype(np.float32),
                    np.load(cc / f"{ref}_test.npy").astype(np.float32))
        if kind == "attr":
            fs = [f.strip() for f in ref.split("+")] if ref != "all" else \
                 [f.strip() for f in args.attr_fields.split(",")]
            return (np.concatenate([np.load(g2 / f"sem_{f}_train.npy").astype(np.float32)
                                    for f in fs], 1),
                    np.concatenate([np.load(g2 / f"sem_{f}_test.npy").astype(np.float32)
                                    for f in fs], 1))
        if kind == "vith":
            lv = [x.strip() for x in args.vith_levels.split(",")]
            vroot = NB_ROOT / "data/things_eeg/image_feature/ViT-H-14"
            tr, te = [], []
            for one in lv:
                p_tr = (vroot / f"{one}_train.npy") if one == "image" else (vroot / one / "train.npy")
                p_te = (vroot / f"{one}_test.npy") if one == "image" else (vroot / one / "test.npy")
                d = np.load(p_tr, mmap_mode="r").shape[-1]
                tr.append(l2n(np.load(p_tr).astype(np.float32).reshape(-1, d)))
                te.append(l2n(np.load(p_te).astype(np.float32).reshape(200, -1)))
            return np.concatenate(tr, 1), np.concatenate(te, 1)
        raise SystemExit(f"[FATAL] unknown bank kind '{kind}' in '{spec}'")

    banks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for item in args.banks.split(","):
        name, _, spec = item.partition("=")
        name, spec = name.strip(), spec.strip()
        banks[name] = build(spec)
        print(f"[bank] {name:<6} {spec:<24} train {banks[name][0].shape} "
              f"test {banks[name][1].shape}")

    # ---- the IP-Adapter entry manifold: every emitted condition lands here ----
    anchor_name = "img"
    B_anchor = l2n(banks[anchor_name][0])          # (n_flat, 1024)
    B_anchor_te = l2n(banks[anchor_name][1])

    # ---- encode the S1 trunk (the pre-head latent) ----
    ckpt = s1d / "best.pth"
    if not ckpt.is_file():
        raise SystemExit(f"[FATAL] missing {ckpt}")
    sd = torch.load(ckpt, map_location=dev, weights_only=False)
    dims = {k: sd["state_dict"][k].shape[0] for k in
            ("head_img.weight", "head_vith.weight", "head_attr.weight")}
    model = FactorizedEncoder(z_dim=ztr.shape[1], dim_img=dims["head_img.weight"],
                              dim_vith=dims["head_vith.weight"],
                              dim_attr=dims["head_attr.weight"]).to(dev)
    model.load_state_dict(sd["state_dict"])
    model.eval()

    def trunk(z: np.ndarray) -> np.ndarray:
        # cast here, not upstream: `l2n`/ridge outputs may be float64, and the model
        # weights are float32 -- torch raises "mat1 and mat2 must have the same dtype"
        # rather than promoting, which is exactly how this job first died.
        with torch.no_grad():
            h = model.trunk(torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32)).to(dev))
        return h.float().cpu().numpy().astype(np.float32)

    htr, hte = trunk(ztr), trunk(zte)
    print(f"[trunk] train {htr.shape} test {hte.shape}")

    split = LF.load(args.split_json)
    fit_i = LF.rows_for(split, "fit", n_flat)
    val_i = LF.rows_for(split, "val_b", n_flat)
    val_cid_raw = cid_tr[val_i]
    uq = {c: i for i, c in enumerate(sorted(set(val_cid_raw.tolist())))}
    val_cid = np.asarray([uq[c] for c in val_cid_raw.tolist()], dtype=np.int64)

    report: dict = {"stage": "nw4_s2_projection", "subject": sid,
                    "s1": str(s1d), "proj": {"k": args.proj_k, "tau": args.proj_tau,
                                             "gamma": args.proj_gamma},
                    "banks": {k: {"train": list(v[0].shape), "test": list(v[1].shape)}
                              for k, v in banks.items()},
                    "conditions": {}}

    for name, (B, B_te) in banks.items():
        if name in ("vith", "attr"):
            # bring the space into the anchor manifold with a bank->bank map fitted on
            # TRAIN rows only; no EEG, no test information is involved.
            P = fit_ridge(B[fit_i], B_anchor[fit_i], alphas)["W"]
            tgt_tr, tgt_te = B @ P, B_te @ P
        else:
            tgt_tr, tgt_te = l2n(B), l2n(B_te)

        # gallery for val alpha selection: concept means of the target over val rows
        gal_val = concept_means(tgt_tr[val_i], val_cid, len(uq))
        r = fit_ridge(htr[fit_i], tgt_tr[fit_i], alphas,
                      val_x=htr[val_i], val_gal=gal_val, val_cid=val_cid)
        u_te = l2n(hte @ r["W"])
        u_tr_val = l2n(htr[val_i] @ r["W"])

        # tune gamma on val_b (the only knob the projection adds): pick the gamma whose
        # projected val rows best retrieve their own concept mean
        best_g, best_v = args.proj_gamma, -1.0
        for g in (0.0, 0.1, 0.2, 0.4, 0.8):
            zv, _ = manifold_project(u_tr_val, tgt_tr, args.proj_k, args.proj_tau, g)
            v = float((l2n(zv) @ l2n(gal_val).T).argmax(1).__eq__(val_cid).mean())
            if v > best_v:
                best_g, best_v = g, v
        z_te, pdiag = manifold_project(u_te, tgt_tr, args.proj_k, args.proj_tau, best_g)

        rec = {"alpha": r["alpha"], "ridge_val_top1": r["val_top1"],
               "gamma": best_g, "gamma_val_top1": best_v,
               "dim_in": int(tgt_tr.shape[1]), "dim_out": int(z_te.shape[1]),
               "raw_row2mean": round(row2mean(u_te), 4),
               "raw_offdiag": round(offdiag(u_te), 4),
               "proj_row2mean": round(row2mean(z_te), 4),
               "proj_offdiag": round(offdiag(z_te), 4),
               "proj_erank": round(erank(z_te), 1),
               **{k: round(v, 4) for k, v in pdiag.items()}}
        # retrieval in the condition's OWN space and in the anchor space
        rec["own_test"] = retrieval(z_te, tgt_te)
        rec["raw_test"] = retrieval(u_te, tgt_te)
        z_anchor, _ = manifold_project(u_te, B_anchor, args.proj_k, args.proj_tau, best_g)
        rec["anchor_test"] = retrieval(z_anchor, B_anchor_te)
        rec["anchor_raw_test"] = retrieval(u_te, B_anchor_te)
        report["conditions"][name] = rec
        print(f"[cond] {name:<6} alpha={r['alpha']:<7g} gamma={best_g:<4g} "
              f"offdiag raw->proj {rec['raw_offdiag']:.4f}->{rec['proj_offdiag']:.4f} "
              f"erank={rec['proj_erank']:.1f} "
              f"own_top1 raw->proj {rec['raw_test']['top1']:.4f}->{rec['own_test']['top1']:.4f} "
              f"anchor_top1 {rec['anchor_raw_test']['top1']:.4f}"
              f"->{rec['anchor_test']['top1']:.4f} "
              f"(csls {rec['anchor_test']['top1_csls']:.4f})")

        if name in [x.strip() for x in args.emit.split(",")]:
            np.save(out / "conds" / f"{name}_test.npy", z_anchor.astype(np.float32))

    # ---- A3 control: the dense mean of the emitted conditions (= nw3's S3 failure) ----
    emit = [x.strip() for x in args.emit.split(",")]
    stack = np.stack([l2n(np.load(out / "conds" / f"{n}_test.npy").astype(np.float32))
                      for n in emit])
    fused = l2n(stack.mean(0).astype(np.float32))
    np.save(out / "conds" / "fused_test.npy", fused.astype(np.float32))
    report["fused_control"] = {"members": emit, "row2mean": round(row2mean(fused), 4),
                               "offdiag": round(offdiag(fused), 4),
                               "erank": round(erank(fused), 1),
                               "test": retrieval(fused, B_anchor_te)}
    print(f"[fused] dense mean of {emit}: offdiag={report['fused_control']['offdiag']:.4f} "
          f"erank={report['fused_control']['erank']:.1f} "
          f"top1={report['fused_control']['test']['top1']:.4f}")
    print("[judge] nw3 baseline to beat: S3 offdiag=0.6994, S2 offdiag=0.5668; "
          "target row2mean>=0.30 and offdiag<=0.45")

    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[nw4-s2] wrote {out / 'report.json'}")


if __name__ == "__main__":
    main()
