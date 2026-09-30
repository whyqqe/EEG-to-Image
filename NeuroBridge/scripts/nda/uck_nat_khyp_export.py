#!/usr/bin/env python3
"""UCK-NAT K-hypothesis condition export (no training, CPU-only).

WHY THIS EXISTS / WHAT WAS WRONG BEFORE
---------------------------------------
v1 sourced the K anchors from the train concept prototypes μ (1654 train
concepts). THINGS-EEG2 test concepts are DISJOINT from the train concepts, so
the anchors could never be the trial's own concept: measured POST-HOC
test_recall@K was 0.000 for every K. λ>0 therefore dragged the condition toward
unrelated concepts — the row was invalid by construction, not by tuning.

FIX
---
Anchors are the top-K most similar reference images retrieved from the
benchmark candidate bank (200-way), used the same way BReAD / SeeEEG / NVOL use
a retrieved image as a generative prior. Self is EXCLUDED so the trial's own
ground-truth embedding can never enter its own condition.

    c_k = l2( IP_uck + λ ( G_retrieved[k] − IP_uck ) )

  λ = 0 → bit-wise UCK (the guard row)
  λ = 1 → pure retrieval-augmented prior

Diversity is structural: K different images. No mode collapse is possible, and
no diversity loss is needed.

Diagnostics that are POST-HOC only (never used for training or selection):
  self_rank   rank of the trial's own row in the full 200-way q·G_test ranking
  top1/top5   standard 200-way retrieval accuracy of the query module
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def retrieve(anchor_bank: np.ndarray, query: np.ndarray, K: int,
             exclude_self: bool) -> tuple[np.ndarray, np.ndarray]:
    """Top-K anchor indices per row. exclude_self masks out index i for row i."""
    sim = l2n(query) @ l2n(anchor_bank).T  # N x B
    if exclude_self and sim.shape[0] == sim.shape[1]:
        sim = sim.copy()
        sim[np.arange(len(sim)), np.arange(len(sim))] = -np.inf
    idx = np.argsort(-sim, axis=1)[:, :K].astype(np.int64)
    return idx, sim


def assemble(ip: np.ndarray, anchors: np.ndarray, lam: float) -> np.ndarray:
    return l2n((ip + lam * (anchors - ip)).astype(np.float32))


def mean_pairwise_cos(stack: np.ndarray) -> float:
    """stack: N x K x D"""
    X = l2n(stack)
    S = np.einsum("nkd,nld->nkl", X, X)
    iu = np.triu_indices(stack.shape[1], 1)
    return float(S[:, iu[0], iu[1]].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--test-subject", type=int, default=8)
    ap.add_argument("--clip-img-dir", type=str, default=str(NB_ROOT / "outputs/gem/cond_cache"))
    ap.add_argument("--uck-ip", type=str,
                    default=str(NB_ROOT / "outputs/uck/sub-08/full/conds/ip_mem_test.npy"))
    ap.add_argument("--query-npy", type=str,
                    default=str(NB_ROOT / "outputs/uck/sub-08/full/conds/ip_q_test.npy"),
                    help="frozen EEG→CLIP query used for retrieval (never trained on test)")
    ap.add_argument("--query-name", type=str, default="uck_q")
    ap.add_argument("--alt-query-npy", type=str, default="",
                    help="optional second route (e.g. ACK q_net) for route ablation")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--lambdas", type=str, default="0,0.3,0.5,1.0")
    ap.add_argument("--exclude-self", type=int, default=1)
    ap.add_argument("--oracle-self", type=int, default=1,
                    help="also export the SELF-INCLUDED leaky upper bound (marked oracle)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "conds").mkdir(parents=True, exist_ok=True)
    sid = f"{args.test_subject:02d}"
    K = int(args.K)
    lams = [float(x) for x in args.lambdas.split(",") if x.strip() != ""]

    bank = l2n(np.load(Path(args.clip_img_dir) / "clip_img1024_test.npy").astype(np.float32))
    ip = l2n(np.load(args.uck_ip).astype(np.float32))
    q = l2n(np.load(args.query_npy).astype(np.float32))
    N = len(bank)
    if len(ip) != N or len(q) != N:
        raise SystemExit(f"[FATAL] rows bank={N} ip={len(ip)} q={len(q)}")

    # ---- POST-HOC retrieval diagnostics (no selection use) ----
    sim_full = q @ bank.T
    order = np.argsort(-sim_full, axis=1)
    self_rank = np.array([int(np.where(order[i] == i)[0][0]) for i in range(N)])
    diag_posthoc = {
        "query": args.query_name,
        "self_top1": float((self_rank == 0).mean()),
        "self_top5": float((self_rank < 5).mean()),
        "self_mean_rank": float(self_rank.mean()),
        "chance_top1": 1.0 / N,
        "NOTE": "POST-HOC diagnostic only; not used for training or selection.",
    }

    # ---- anchors (self excluded) ----
    idx_selfex, _ = retrieve(bank, q, K, exclude_self=bool(args.exclude_self))
    # sanity: no anchor equals its own row
    row_ids = np.arange(N)[:, None]
    same = (idx_selfex == row_ids)
    if same.any():
        raise SystemExit(f"[FATAL] self-anchor leaked in {int(same.sum())} slots")

    report: dict = {
        "pipeline": "uck_nat_khyp_export",
        "subject": f"sub-{sid}",
        "K": K,
        "lambdas": lams,
        "anchor_source": "benchmark candidate bank (200-way), SELF EXCLUDED",
        "exclude_self": bool(args.exclude_self),
        "diagnostics_POSTHOC": diag_posthoc,
        "diversity": {},
        "params": vars(args),
    }

    # ---- assemble per λ ----
    for lam in lams:
        stack = np.stack([assemble(ip, bank[idx_selfex[:, k]], lam) for k in range(K)], axis=1)
        np.save(out / "conds" / f"ip_lam{lam:g}_K{K}_test.npy", stack)
        for k in range(K):
            np.save(out / "conds" / f"ip_lam{lam:g}_k{k}_test.npy", stack[:, k])
        report["diversity"][f"lam{lam:g}_mean_pairwise_cos"] = mean_pairwise_cos(stack)

    # λ=0 must be bit-wise UCK
    c0 = assemble(ip, bank[idx_selfex[:, 0]], 0.0)
    report["fuse_lambda0_equals_uck"] = bool(np.allclose(c0, ip, atol=1e-5, rtol=1e-5))
    report["fuse_lambda0_max_abs"] = float(np.max(np.abs(c0 - ip)))

    # ---- random-K control: shuffle anchors ACROSS rows (breaks pairing) ----
    rng = np.random.default_rng(args.seed)
    idx_rand = np.stack([rng.permutation(idx_selfex[:, k]) for k in range(K)], axis=1)
    for lam in (0.5, 1.0):
        stack = np.stack([assemble(ip, bank[idx_rand[:, k]], lam) for k in range(K)], axis=1)
        np.save(out / "conds" / f"ip_rand_lam{lam:g}_K{K}_test.npy", stack)
        for k in range(K):
            np.save(out / "conds" / f"ip_rand_lam{lam:g}_k{k}_test.npy", stack[:, k])
    report["random_control"] = {
        "kind": "anchor assignment permuted across rows (marginal preserved, pairing broken)",
        "n_anchor_slots_changed": int((idx_rand != idx_selfex).mean()),
    }

    # ---- leaky oracle (self INCLUDED) — diagnostic upper bound, never a result ----
    if args.oracle_self:
        idx_oracle, _ = retrieve(bank, q, K, exclude_self=False)
        stack = np.stack([assemble(ip, bank[idx_oracle[:, k]], 1.0) for k in range(K)], axis=1)
        np.save(out / "conds" / f"ORACLE_selfincluded_lam1_K{K}_test.npy", stack)
        report["oracle_self_included"] = {
            "frac_rows_with_self_at_k0": float((idx_oracle[:, 0] == np.arange(N)).mean()),
            "WARNING": "leaky: the condition can contain the trial's own ground-truth "
                       "embedding. Diagnostic upper bound only, must NOT be reported as a result.",
        }

    # ---- optional second query route (route ablation) ----
    if args.alt_query_npy and Path(args.alt_query_npy).is_file():
        q2 = l2n(np.load(args.alt_query_npy).astype(np.float32))
        if len(q2) == N:
            sim2 = q2 @ bank.T
            o2 = np.argsort(-sim2, axis=1)
            r2 = np.array([int(np.where(o2[i] == i)[0][0]) for i in range(N)])
            idx2, _ = retrieve(bank, q2, K, exclude_self=bool(args.exclude_self))
            for lam in (0.5,):
                stack = np.stack([assemble(ip, bank[idx2[:, k]], lam) for k in range(K)], axis=1)
                np.save(out / "conds" / f"ip_altq_lam{lam:g}_K{K}_test.npy", stack)
                for k in range(K):
                    np.save(out / "conds" / f"ip_altq_lam{lam:g}_k{k}_test.npy", stack[:, k])
            report["alt_query_POSTHOC"] = {
                "name": Path(args.alt_query_npy).parent.parent.name,
                "self_top1": float((r2 == 0).mean()),
                "self_top5": float((r2 < 5).mean()),
                "NOTE": "POST-HOC diagnostic only.",
            }

    # ---- anchor usage / identity telemetry ----
    uniq_per_row = np.array([len(set(idx_selfex[i].tolist())) for i in range(N)])
    report["anchors"] = {
        "n_unique_per_row_mean": float(uniq_per_row.mean()),
        "n_unique_per_row_min": int(uniq_per_row.min()),
        "n_unique_anchors_overall": int(len(set(idx_selfex.reshape(-1).tolist()))),
        "topK_idx_head": idx_selfex[:5].tolist(),
    }
    np.save(out / "conds" / "anchor_idx_selfex.npy", idx_selfex)
    np.save(out / "conds" / "ip_uck_test.npy", ip)

    (out / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[export] -> {out/'conds'}")


if __name__ == "__main__":
    main()
