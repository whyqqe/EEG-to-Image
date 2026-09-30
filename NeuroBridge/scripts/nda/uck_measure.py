#!/usr/bin/env python3
"""Validity measurements for a UCK subject directory. Never raises."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def rowcos(x: np.ndarray) -> float:
    z = l2(x)
    g = z @ z.T
    n = len(z)
    return float((g.sum() - np.trace(g)) / max(n * (n - 1), 1))


def retrieval(q: np.ndarray, bank: np.ndarray) -> dict:
    sim = l2(q) @ l2(bank).T
    n = len(sim)
    order = np.argsort(-sim, axis=1)
    top1 = order[:, 0]
    return {
        "n": n,
        "chance_row_identity": 1.0 / max(n, 1),
        "row_identity_top1": float((top1 == np.arange(n)).mean()),
        "row_identity_top5": float(np.mean([i in order[i, :5] for i in range(n)])),
        "mean_top1_cos": float(sim[np.arange(n), top1].mean()),
        "mean_true_cos": float(sim[np.arange(n), np.arange(n)].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, required=True,
                    help="subject dir that contains full/conds/")
    ap.add_argument("--bank", type=str, required=True,
                    help="TEST CLIP-image encode_image() bank (200x1024)")
    ap.add_argument("--report", type=str, default="")
    args = ap.parse_args()

    root = Path(args.out_dir)
    full = root / "full" / "conds"
    rep: dict = {"out_dir": str(root), "errors": {}}
    try:
        T = l2(np.load(args.bank).astype(np.float32))
        centre = l2(T.mean(0, keepdims=True))
        c = float((centre * T).sum(1).mean())
        rep["centreline"] = {"cos_mean_to_each": c}
        print(f"[measure] centreline {c:+.4f}")
    except Exception as e:  # noqa: BLE001
        T = None
        rep["errors"]["centreline"] = f"{type(e).__name__}: {e}"
        print(f"[measure] centreline FAILED: {e}")

    rows = {}
    for name in ("ip_mem_test", "ip_q_test", "ip_mem_image_test", "ip_mem_noise_test"):
        p = full / f"{name}.npy"
        if not p.is_file():
            continue
        x = np.load(p).astype(np.float32)
        rec = {"shape": list(x.shape), "rowcos": rowcos(x), "std": float(x.std(0).mean())}
        if T is not None and len(x) == len(T) and x.shape[1] == T.shape[1]:
            rec["vs_true"] = retrieval(x, T)
            rec["vs_centreline"] = float((l2(x) * T).sum(1).mean()) - rep.get(
                "centreline", {}).get("cos_mean_to_each", 0.0)
        rows[name] = rec
        print(f"[measure] {name}: rowcos={rec['rowcos']:.3f} "
              f"true_cos={rec.get('vs_true', {}).get('mean_true_cos', float('nan')):.3f}")
    rep["rows"] = rows

    try:
        a_p, b_p = full / "ip_mem_test.npy", full / "ip_mem_noise_test.npy"
        if a_p.is_file() and b_p.is_file():
            a, b = l2(np.load(a_p)), l2(np.load(b_p))
            agree = float((a * b).sum(1).mean())
            rep["full_vs_noise_rowcos"] = agree
            if agree > 0.95:
                rep["arm_agreement_verdict"] = "BROKEN"
                print(f"[measure] [FAIL] full vs noise {agree:.4f} -- control is a no-op")
            else:
                rep["arm_agreement_verdict"] = "ok"
                print(f"[measure] [ok] full vs noise {agree:.4f}")
    except Exception as e:  # noqa: BLE001
        rep["errors"]["arm"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    dst = Path(args.report) if args.report else root / "measure.json"
    dst.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"[measure] wrote {dst}")


if __name__ == "__main__":
    main()
