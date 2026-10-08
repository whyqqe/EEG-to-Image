#!/usr/bin/env python
"""SBA feasibility check — does a subject-invariant, concept-discriminative TEMPORAL anchor
exist in the raw EEG?  ZERO training. This is a measurement on banked data, not a model.

WHY THIS NEEDS NO TRAINING. The question "does the temporal factor carry a cross-subject
concept signal?" is a property of the DATA. The full SBA architecture (a temporal-factor
encoder + alignment) would need training, but only AFTER this gate says the anchor exists.
This probe is the gate.

THE GENERATIVE MODEL UNDER TEST. Each concept's grand-average ERP is
    X_c^(s) = A_s S_c + noise,
where A_s is a SUBJECT-SPECIFIC spatial mixing (volume conduction) and S_c is a shared
source-space spatiotemporal pattern. The SBA premise is a mathematical statement about the SVD
of X_c^(s): right-multiplying by nothing, the RIGHT singular subspace (time) is invariant to
A_s (left multiplication by an invertible map does not change the row space), while the LEFT
singular subspace (channels) is not. So if SBA's anchor exists, we must measure:
  * temporal factor  : cross-subject CONSISTENT  (invariance)
  * temporal factor  : still CONCEPT-DISCRIMINATIVE (otherwise it is a shared template, useless)
  * and temporal >> spatial on BOTH, with the mix stress test as the smoking gun.

REPRESENTATIONS (per subject, per concept; grand-average over the 80 repetitions):
  temporal_svd : top-k RIGHT singular vectors of the (63, 250) ERP  -> (k, T), the time basis
  gfp          : global field power  ||X[:, t]||_2 over channels      -> (T,), a time course
  spatial_svd  : top-k LEFT singular vectors                          -> (k, Ch), topography
  spatial_amp  : channel amplitude profile (mean over time)           -> (Ch,)
  full         : vec(X)                                               -> (Ch*T,)

MEASUREMENTS
  A   same-concept cross-subject affinity     (invariance: higher is better)
  B   different-concept cross-subject affinity (the discriminability floor)
  A-B margin                                  (a factor is useful iff A high AND A-B large)
  ID  cross-subject concept-identity top-1    (the practical version; chance = 1/200 = 0.5%)
  mix  : apply a random invertible spatial mixing to one subject; re-measure A

PRE-REGISTERED VERDICT (fixed before running):
  SBA ANCHOR EXISTS  iff  ID(temporal) > max(2%, 3x chance)  AND  A(temporal) - A(spatial) > 0.10
                          AND  A(temporal) - B(temporal) > 0.10
  SBA FAILS          iff  ID(temporal) <= 2%  (temporal factor is invariant but not
                          discriminative)  OR  A(temporal) ~ A(spatial) (no symmetry breaking)

Run:
  python scripts/probe_sba_anchor.py --subjects 1 2 3 4 5 6 7 8 9 10 --out outputs/probe/sba_anchor.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from samclip import config  # noqa: E402


def _grand_avg_erp(subject: int, t_slice: slice | None = None) -> np.ndarray:
    """(C, Ch, T) grand-average ERP for one subject, per-channel z-scored across concepts+time.

    The raw file is (C, 1, R, Ch, T). Averaging over R is the standard deployment query. The
    per-channel z-score uses THIS subject's own statistics only, which is legitimate (it is the
    subject's own scale, no labels) and makes the subjects comparable in amplitude before any
    cross-subject affinity is computed.
    """
    raw = np.load(config.subject_dir(subject) / "test.npy", mmap_mode="r")
    blk = raw[:, 0]                                              # (C, R, Ch, T) memmap view
    X = np.asarray(blk.mean(axis=1), dtype=np.float64)          # (C, Ch, T): mean over reps
    if t_slice is not None:
        X = X[:, :, t_slice]
    mu = X.mean(axis=(0, 2), keepdims=True)
    sd = X.std(axis=(0, 2), keepdims=True) + 1e-9
    return (X - mu) / sd


def _topk_subspaces(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-concept top-k left/right singular vectors. X: (C, Ch, T). Returns U (C,Ch,k), V (C,k,T)."""
    U, _, Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :, :k], Vt[:, :k, :]


def _subspace_affinity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Affinity between orthonormal row-bases A (C,k,d) and B (C,k,d) for ALL concept pairs:
    mean squared principal cosines. Returns a (C,C) matrix; entry [i,j] compares concept i of A
    to concept j of B. 1.0 = identical subspace, 0 = orthogonal."""
    G = np.einsum("aid,bjd->abij", A, B)                     # (C,C,k,k): G[a,b] = A_a B_b^T
    sv = np.linalg.svd(G, compute_uv=False)                  # (C,C,k) principal cosines
    return (sv ** 2).mean(axis=-1)


def _same_diff_id(sim_fn, reps: dict[int, dict]):
    """Cross-subject same-concept affinity (A), different-concept affinity (B), and concept-ID
    top-1, given a similarity function `sim_fn(subject_a, subject_b)` -> (C, C) matrix."""
    subs = sorted(reps)
    A_vals, B_vals, id_hits, id_tot = [], [], 0, 0
    rng = np.random.default_rng(0)
    C = reps[subs[0]]["n_concepts"]
    for i in range(len(subs)):
        for j in range(i + 1, len(subs)):
            S = sim_fn(subs[i], subs[j])                     # (C, C)
            A_vals.append(np.diag(S))
            off = rng.integers(0, C, size=2000)
            off2 = rng.integers(0, C, size=2000)
            keep = off != off2
            B_vals.append(S[off[keep], off2[keep]])
            id_hits += int((S.argmax(axis=1) == np.arange(C)).sum())
            id_tot += C
    return (float(np.mean(np.concatenate(A_vals))),
            float(np.mean(np.concatenate(B_vals))),
            float(100.0 * id_hits / id_tot))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 11)))
    ap.add_argument("--k", type=int, default=5, help="subspace dimension for the SVD factors")
    ap.add_argument("--t-start", type=int, default=0, help="first time sample (250 Hz)")
    ap.add_argument("--t-end", type=int, default=250, help="last time sample (exclusive)")
    ap.add_argument("--views", default="temporal_svd,gfp,spatial_svd,spatial_amp,full",
                    help="comma list of views to compute (cheaper when you only need the temporal ones)")
    ap.add_argument("--out", default=str(ROOT / "outputs/probe/sba_anchor.json"))
    args = ap.parse_args()

    sl = slice(int(args.t_start), int(args.t_end))
    k = int(args.k)
    print("=" * 92)
    print(f"SBA feasibility — temporal vs spatial anchor, subjects {args.subjects}, k={k}, "
          f"window [{sl.start}:{sl.stop}] of 250")
    print("=" * 92)

    reps: dict[int, dict] = {}
    for s in args.subjects:
        X = _grand_avg_erp(s, sl)
        U, V = _topk_subspaces(X, k)
        gfp = np.linalg.norm(X, axis=1)                      # (C, T)
        amp = X.mean(axis=2)                                 # (C, Ch)
        reps[s] = {"U": U, "V": V, "gfp": gfp, "amp": amp, "full": X,
                   "n_concepts": X.shape[0]}
        print(f"  loaded sub-{s:02d}: X {X.shape}, gfp {gfp.shape}")

    want = [v.strip() for v in str(args.views).split(",") if v.strip()]
    all_views = {
        "temporal_svd": lambda a, b: _subspace_affinity(reps[a]["V"], reps[b]["V"]),
        "gfp": lambda a, b: _pair_corr(reps[a]["gfp"], reps[b]["gfp"]),
        "spatial_svd": lambda a, b: _subspace_affinity(reps[a]["U"], reps[b]["U"]),
        "spatial_amp": lambda a, b: _pair_corr(reps[a]["amp"], reps[b]["amp"]),
        "full": lambda a, b: _pair_corr(reps[a]["full"].reshape(len(reps[a]["full"]), -1),
                                        reps[b]["full"].reshape(len(reps[b]["full"]), -1)),
    }
    views = {k: all_views[k] for k in want if k in all_views}

    results = {}
    print(f"\n{'view':<14s} {'A(same)':>8s} {'B(diff)':>8s} {'margin':>7s} {'ID top-1':>9s}")
    for name, fn in views.items():
        A, B, ID = _same_diff_id(fn, reps)
        results[name] = {"A_same": A, "B_diff": B, "margin": A - B, "id_top1": ID}
        print(f"{name:<14s} {A:8.4f} {B:8.4f} {A-B:7.4f} {ID:8.2f}%")

    # ---- mix stress test: apply a random invertible spatial mixing to ONE subject ----------
    s0 = sorted(reps)[0]
    rng = np.random.default_rng(1)
    Cho = reps[s0]["full"].shape[1]
    A_mix = np.eye(Cho) + 0.5 * rng.standard_normal((Cho, Cho))
    print(f"\n[mix] applied random invertible {Cho}x{Cho} spatial mixing to sub-{s0:02d} "
          f"(cond={np.linalg.cond(A_mix):.1f}); re-measuring affinity to one other subject")
    s1 = sorted(reps)[1]
    Xm = np.einsum("ij,cjt->cit", A_mix, reps[s0]["full"])
    Um, Vm = _topk_subspaces(Xm, k)
    a_spat_clean = np.diag(_subspace_affinity(reps[s0]["U"], reps[s1]["U"])).mean()
    a_spat_mixed = np.diag(_subspace_affinity(Um, reps[s1]["U"])).mean()
    a_temp_clean = np.diag(_subspace_affinity(reps[s0]["V"], reps[s1]["V"])).mean()
    a_temp_mixed = np.diag(_subspace_affinity(Vm, reps[s1]["V"])).mean()
    print(f"    spatial_svd same-concept affinity : clean {a_spat_clean:.4f} -> mixed {a_spat_mixed:.4f}"
          f"  (drop {a_spat_clean - a_spat_mixed:+.4f})")
    print(f"    temporal_svd same-concept affinity: clean {a_temp_clean:.4f} -> mixed {a_temp_mixed:.4f}"
          f"  (drop {a_temp_clean - a_temp_mixed:+.4f})")
    print("    -> the mathematical claim is that the TEMPORAL drop is ~0 and the SPATIAL drop is large.")

    # ---- pre-registered verdict --------------------------------------------------------
    t = results["temporal_svd"]
    sp = results.get("spatial_svd")
    chance = 100.0 / reps[s0]["n_concepts"]
    if sp is None:
        verdict = ("SBA ANCHOR: temporal invariance measured; spatial view not requested "
                   "(--views), so the comparative criterion is not evaluated here")
        exists = failed = None
    else:
        exists = (t["id_top1"] > max(2.0, 3 * chance) and (t["A_same"] - sp["A_same"]) > 0.10
                  and t["margin"] > 0.10)
        failed = (t["id_top1"] <= 2.0) or abs(t["A_same"] - sp["A_same"]) <= 0.02
        verdict = ("SBA ANCHOR EXISTS" if exists else
                   "SBA FAILS (temporal factor not cross-subject usable)" if failed else
                   "INCONCLUSIVE — read the table")
    print(f"\n  chance concept-ID = {chance:.2f}% ; verdict: {verdict}")
    results["_meta"] = {"subjects": args.subjects, "k": k, "window": [sl.start, sl.stop],
                        "chance_id": chance, "verdict": verdict,
                        "mix_stress": {"spatial_clean": a_spat_clean, "spatial_mixed": a_spat_mixed,
                                       "temporal_clean": a_temp_clean, "temporal_mixed": a_temp_mixed}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"[sba] wrote {out}")


def _pair_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Full (C, C) correlation matrix between rows of a (C, d) and rows of b (C, d)."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


if __name__ == "__main__":
    main()
