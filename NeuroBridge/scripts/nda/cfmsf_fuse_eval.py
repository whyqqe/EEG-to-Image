#!/usr/bin/env python3
"""CF-MSF Stage B: concept-factorized multi-route score fusion + ablations.

For each route r, build S^r = cos(q_r, bank_r) over the shared 200-way index,
optionally apply CSLS / whitening, then fuse:

    S = Σ_r  w_r · τ_r^{-1} · Ŝ^r

Primary result uses UNIFORM weights (CORTIVA: uniform ≥ learned). Ablations:
  - single routes
  - leave-one-route-out
  - CSLS on/off
  - Sinkhorn (balanced assignment) on/off  [transductive; declared]
  - paired vs shuffled concept-address control (neural-address check)
  - temperature grid
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocf_train import l2n  # noqa: E402


def csls(sim: np.ndarray, k: int = 10) -> np.ndarray:
    """Cross-domain similarity local scaling: 2*cos − r_q − r_g."""
    kk = min(k, sim.shape[1] - 1, sim.shape[0] - 1)
    if kk < 1:
        return sim
    q = np.sort(sim, axis=1)[:, -kk:].mean(1, keepdims=True)
    g = np.sort(sim, axis=0)[-kk:, :].mean(0, keepdims=True)
    return 2.0 * sim - q - g


def whiten_rows(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Simple per-batch ZCA-ish whitening on the query side (row features)."""
    x = x - x.mean(0, keepdims=True)
    cov = (x.T @ x) / max(len(x) - 1, 1)
    u, s, _ = np.linalg.svd(cov, full_matrices=False)
    w = u @ np.diag(1.0 / np.sqrt(s + eps)) @ u.T
    return l2n(x @ w)


def sinkhorn(sim: np.ndarray, iters: int = 50, tau: float = 0.07) -> np.ndarray:
    log_k = sim.astype(np.float64) / max(tau, 1e-6)
    log_k = log_k - log_k.max(axis=1, keepdims=True)
    k = np.exp(log_k)
    for _ in range(iters):
        k = k / np.clip(k.sum(axis=1, keepdims=True), 1e-12, None)
        k = k / np.clip(k.sum(axis=0, keepdims=True), 1e-12, None)
    return k.argmax(axis=1).astype(np.int64)


def metrics(sim: np.ndarray, use_sinkhorn: bool = False, sink_tau: float = 0.07) -> dict:
    n = sim.shape[0]
    if use_sinkhorn:
        pred = sinkhorn(sim, tau=sink_tau)
        # top5 from raw sim still (sinkhorn is permutation)
        order = np.argsort(-sim, axis=1)
        top5 = float(np.mean([i in order[i, :5] for i in range(n)]))
        top1 = float((pred == np.arange(n)).mean())
    else:
        order = np.argsort(-sim, axis=1)
        top1 = float((order[:, 0] == np.arange(n)).mean())
        top5 = float(np.mean([i in order[i, :5] for i in range(n)]))
    # hubness: skewness of gallery in-degree at top-5%
    k = max(1, int(round(n * 0.05)))
    deg = np.bincount(np.argsort(-sim, axis=1)[:, :k].ravel(), minlength=n).astype(np.float64)
    m, s = deg.mean(), deg.std()
    hub = float((((deg - m) / (s + 1e-9)) ** 3).mean()) if s > 1e-9 else 0.0
    return {"top1": top1, "top5": top5, "hub_skew": hub,
            "mean_rank": float(np.mean([np.where(order[i] == i)[0][0] + 1 for i in range(n)]))}


def load_route(conds: Path, name: str, bank: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = l2n(np.load(conds / f"q_{name}_test.npy").astype(np.float32))
    if len(q) != len(bank):
        raise SystemExit(f"[FATAL] {name}: q {len(q)} vs bank {len(bank)}")
    return q, bank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-out", type=str, required=True,
                    help="directory written by cfmsf_train.py")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--cond-cache", type=str, default=str(NB_ROOT / "outputs/gem/cond_cache"))
    ap.add_argument("--clip-text-dir", type=str, default="")
    ap.add_argument("--routes", type=str, default="img,text,depth,edge")
    ap.add_argument("--csls-k", type=int, default=10)
    ap.add_argument("--sink-tau", type=float, default=0.07)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_out = Path(args.train_out)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    conds = train_out / "conds"
    cond = Path(args.cond_cache)

    # subject from train report
    tr = json.loads((train_out / "train_report.json").read_text())
    sid = tr["subject"].split("-")[1]
    if not args.clip_text_dir:
        args.clip_text_dir = str(NB_ROOT / f"outputs/nda_ss/sub-{sid}/clip_text")

    banks = {
        "img":   l2n(np.load(cond / "clip_img1024_test.npy").astype(np.float32)),
        "text":  l2n(np.load(Path(args.clip_text_dir) / "test" / "text_concept_clip.npy").astype(np.float32)),
        "depth": l2n(np.load(cond / "clip_depth1024_test.npy").astype(np.float32)),
        "edge":  l2n(np.load(cond / "clip_edge1024_test.npy").astype(np.float32)),
    }
    routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    for r in routes:
        if not (conds / f"q_{r}_test.npy").is_file():
            raise SystemExit(f"[FATAL] missing {conds}/q_{r}_test.npy")

    # build per-route score matrices
    scores: dict[str, np.ndarray] = {}
    qs: dict[str, np.ndarray] = {}
    for r in routes:
        q, bank = load_route(conds, r, banks[r])
        # optional query whitening (SATTC-style geometric expert, light version)
        qw = whiten_rows(q)
        scores[r] = (qw @ l2n(bank).T).astype(np.float64)
        qs[r] = q
        print(f"[route {r}] raw {metrics(scores[r])}")

    def fuse(names: list[str], use_csls: bool, weights: dict[str, float] | None = None,
             temps: dict[str, float] | None = None) -> np.ndarray:
        w = weights or {n: 1.0 for n in names}
        t = temps or {n: 1.0 for n in names}
        acc = None
        for n in names:
            s = scores[n]
            if use_csls:
                s = csls(s, k=args.csls_k)
            term = (w[n] / max(t[n], 1e-6)) * s
            acc = term if acc is None else acc + term
        return acc

    results: dict = {"protocol": {
        "task": "200-way zero-shot retrieval",
        "subject": tr["subject"],
        "n": 200,
        "chance_top1": 0.005,
        "chance_top5": 0.025,
        "note": ("Sinkhorn uses the public square structure of the 200-way bank "
                 "(one unique concept per image); reported as a SEPARATE arm."),
        "train_report": str(train_out / "train_report.json"),
    }, "arms": {}}

    def add(tag: str, sim: np.ndarray, **extra):
        row = {**metrics(sim, use_sinkhorn=False), **extra}
        row_s = metrics(sim, use_sinkhorn=True, sink_tau=args.sink_tau)
        results["arms"][tag] = row
        results["arms"][tag + "+sinkhorn"] = {**row_s, "transductive": True,
                                              "parent": tag}
        print(f"[{tag}] top1={row['top1']:.4f} top5={row['top5']:.4f} "
              f"rank={row['mean_rank']:.2f} hub={row['hub_skew']:.2f}  |  "
              f"+sink top1={row_s['top1']:.4f}")

    # ---- single routes (raw + CSLS) ----
    for r in routes:
        add(f"R_{r}_raw", scores[r], routes=[r], csls=False)
        add(f"R_{r}_csls", csls(scores[r], args.csls_k), routes=[r], csls=True)

    # ---- full fusion ----
    add("F_all_raw", fuse(routes, use_csls=False), routes=routes, csls=False, weights="uniform")
    add("F_all_csls", fuse(routes, use_csls=True), routes=routes, csls=True, weights="uniform")

    # ---- leave-one-route-out ----
    for drop in routes:
        keep = [r for r in routes if r != drop]
        if not keep:
            continue
        add(f"F_wo_{drop}_csls", fuse(keep, use_csls=True),
            routes=keep, csls=True, dropped=drop)

    # ---- semantic-only vs structure-only ----
    sem = [r for r in routes if r in ("img", "text")]
    struct = [r for r in routes if r in ("depth", "edge")]
    if sem:
        add("F_sem_csls", fuse(sem, use_csls=True), routes=sem, csls=True)
    if struct:
        add("F_struct_csls", fuse(struct, use_csls=True), routes=struct, csls=True)

    # ---- neural-address control: shuffle gallery columns of img route ----
    # If "concept identity" is what the head recovers, destroying the column
    # alignment of the concept-matched bank should collapse accuracy to chance.
    # (We shuffle the TEST bank columns — a destructive control, not a training ablation.)
    rng = np.random.default_rng(args.seed)
    if "img" in scores:
        perm = rng.permutation(200)
        shuf = scores["img"][:, perm]
        add("CTRL_img_bank_shuffled", csls(shuf, args.csls_k),
            routes=["img"], csls=True,
            NOTE="gallery columns shuffled; should fall near chance if identity matters")

    # ---- temperature sensitivity (uniform routes, CSLS on) ----
    temp_grid = {}
    for tau in (0.5, 1.0, 2.0):
        sim = fuse(routes, use_csls=True, temps={r: tau for r in routes})
        m = metrics(sim)
        temp_grid[str(tau)] = m
    results["temperature_grid_csls"] = temp_grid

    # ---- primary pick ----
    primary = "F_all_csls"
    results["primary"] = primary
    results["primary_metrics"] = results["arms"][primary]
    results["primary_plus_sinkhorn"] = results["arms"][primary + "+sinkhorn"]

    # summary table
    print("\n===== CF-MSF summary =====")
    print(f"{'arm':<28}{'top1':>8}{'top5':>8}{'rank':>8}{'hub':>8}")
    for k, v in results["arms"].items():
        if k.endswith("+sinkhorn"):
            continue
        print(f"{k:<28}{v['top1']:>8.4f}{v['top5']:>8.4f}{v['mean_rank']:>8.2f}{v['hub_skew']:>8.2f}")
    print("--- sinkhorn variants ---")
    for k, v in results["arms"].items():
        if k.endswith("+sinkhorn"):
            print(f"{k:<28}{v['top1']:>8.4f}{v['top5']:>8.4f}{v['mean_rank']:>8.2f}")

    (out / "fuse_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    # also dump the primary score matrix for downstream generation use
    np.save(out / "S_primary_csls.npy", fuse(routes, use_csls=True))
    np.save(out / "pred_idx_primary.npy",
            fuse(routes, use_csls=True).argmax(1).astype(np.int64))
    np.save(out / "pred_idx_primary_sinkhorn.npy",
            sinkhorn(fuse(routes, use_csls=True), tau=args.sink_tau))
    print(f"[cfmsf-fuse] wrote {out}")


if __name__ == "__main__":
    main()
