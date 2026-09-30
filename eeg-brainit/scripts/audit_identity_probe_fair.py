#!/usr/bin/env python3
"""Fair identity audit: fresh linear probe on frozen clip_emb.

Compares checkpoint dirs (e.g. abs_only_v2 vs abs_id_v1) under identical protocol:
  - For each LOSO fold checkpoint: encode train-subjects (held-in), fit LogisticRegression
  - Report probe acc (lower = less identity leakage) + chance
Does NOT use the co-trained probe_id head.

Also reports absolute retrieval on the held-out test subject for sanity.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("HOME", "/project/peilab/why/cache/eeg-brainit/xdg-home")
os.environ.setdefault("HF_HOME", "/project/peilab/why/cache/eeg-brainit/hf")

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_brainit.data.things_eeg2_adapt import ThingsEEG2SubjectDataset, collate_batch
from eeg_brainit.models.aria import ARIAEncoder
from eeg_brainit.utils.config import ensure_dirs

ALL_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 11)]


def l2_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def retrieval_top1(queries: np.ndarray, gallery: np.ndarray) -> float:
    q, g = l2_np(queries), l2_np(gallery)
    order = np.argsort(-(q @ g.T), axis=1)
    ranks = np.argmax(order == np.arange(order.shape[0])[:, None], axis=1)
    return float((ranks < 1).mean())


@torch.no_grad()
def encode_subject(model, subject: str, split: str, args, device, max_samples: int) -> np.ndarray:
    ds = ThingsEEG2SubjectDataset(
        ROOT / args.eeg_root,
        ROOT / args.atm_bridge_dir,
        subject,
        split=split,
        max_samples=max_samples,
        seed=args.seed,
    )
    model.eval()
    outs = []
    bs = args.batch_size
    for i0 in range(0, len(ds), bs):
        samples = [ds[i] for i in range(i0, min(i0 + bs, len(ds)))]
        batch = collate_batch(samples)
        emb = model(batch["eeg"].to(device))["clip_emb"].float().cpu().numpy()
        outs.append(emb)
    return np.concatenate(outs, 0)


def load_fold_model(ckpt_path: Path, device: torch.device) -> ARIAEncoder:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    args_d = ck.get("args") or {}
    nz = int(args_d.get("nz", 256))
    clip_dim = int(args_d.get("clip_dim", 1024))
    model = ARIAEncoder(n_subjects=10, clip_dim=clip_dim, nz=nz).to(device)
    # anchors buffer size differs from default; set before / strip from state
    state = dict(ck["model"])
    anchors = ck.get("anchors")
    if anchors is None and "anchors" in state:
        anchors = state["anchors"]
    state.pop("anchors", None)
    state.pop("anchors_ready", None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if anchors is not None:
        model.set_anchors(torch.as_tensor(anchors, device=device).float())
    model.eval()
    return model


def fair_probe(x: np.ndarray, y: np.ndarray, seed: int) -> float:
    """Train/test split by sample; return holdout accuracy."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    n_te = max(int(0.25 * len(y)), 1)
    te, tr = idx[:n_te], idx[n_te:]
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="lbfgs"),
    )
    clf.fit(x[tr], y[tr])
    return float(clf.score(x[te], y[te]))


def audit_run(name: str, ckpt_dir: Path, args, device, gallery: np.ndarray, test_subs: list[str]) -> dict:
    rows = []
    for test_sub in test_subs:
        ck = ckpt_dir / f"aria_{test_sub}.pt"
        if not ck.is_file():
            print(f"[SKIP] missing {ck}", flush=True)
            continue
        print(f"[{name}] fold={test_sub}", flush=True)
        model = load_fold_model(ck, device)
        train_subs = [s for s in ALL_SUBJECTS if s != test_sub]
        feats, labels = [], []
        for sub in train_subs:
            emb = encode_subject(model, sub, "train", args, device, args.max_per_sub)
            sid = int(sub.replace("sub-", "")) - 1
            feats.append(emb)
            labels.append(np.full(len(emb), sid, dtype=np.int64))
        x = np.concatenate(feats, 0)
        y = np.concatenate(labels, 0)
        # remap labels to 0..S-1 contiguous
        uniq = sorted(set(y.tolist()))
        remap = {u: i for i, u in enumerate(uniq)}
        y_m = np.array([remap[v] for v in y], dtype=np.int64)
        probe = fair_probe(x, y_m, args.seed + uniq[0])
        chance = 1.0 / len(uniq)
        te = encode_subject(model, test_sub, "test", args, device, 0)
        abs_top1 = retrieval_top1(te, gallery)
        row = {
            "test_subject": test_sub,
            "identity_probe_fresh": probe,
            "chance": chance,
            "n_probe": int(len(y)),
            "absolute_top1": abs_top1,
        }
        rows.append(row)
        print(
            f"  probe={probe*100:.1f}% chance={chance*100:.1f}% abs={abs_top1*100:.1f}%",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    probe_m = float(np.mean([r["identity_probe_fresh"] for r in rows]))
    abs_m = float(np.mean([r["absolute_top1"] for r in rows]))
    return {
        "name": name,
        "ckpt_dir": str(ckpt_dir),
        "folds": rows,
        "summary": {
            "identity_probe_fresh_mean": probe_m,
            "identity_probe_fresh_std": float(np.std([r["identity_probe_fresh"] for r in rows])),
            "chance": float(np.mean([r["chance"] for r in rows])),
            "absolute_top1_mean": abs_m,
            "absolute_top1_std": float(np.std([r["absolute_top1"] for r in rows])),
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eeg-root", default="data/processed/things-eeg2")
    p.add_argument("--atm-bridge-dir", default="outputs/atm_bridge")
    p.add_argument("--output-dir", default="outputs/aria/identity_audit_v1")
    p.add_argument("--runs", nargs="+", default=[
        "abs_only:outputs/aria/abs_only_v2/checkpoints",
        "abs_id:outputs/aria/abs_id_v1/checkpoints",
    ], help="name:ckpt_dir pairs")
    p.add_argument("--max-per-sub", type=int, default=400, help="samples/subject for probe fit")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-folds", type=int, default=0, help="0=all 10")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = ROOT / args.output_dir
    ensure_dirs(out)
    gallery = np.load(ROOT / args.atm_bridge_dir / "clip_img_test_1024.npy").astype(np.float32)
    test_subs = ALL_SUBJECTS[: args.max_folds] if args.max_folds > 0 else list(ALL_SUBJECTS)

    print(
        f"[INFO] fair identity audit device={device} max_per_sub={args.max_per_sub} "
        f"folds={len(test_subs)}",
        flush=True,
    )
    results = []
    for spec in args.runs:
        if ":" not in spec:
            raise SystemExit(f"bad --runs entry {spec}, need name:ckpt_dir")
        name, cdir = spec.split(":", 1)
        results.append(audit_run(name, ROOT / cdir, args, device, gallery, test_subs))

    # pairwise deltas vs first run
    compare = []
    if len(results) >= 2:
        base = results[0]["summary"]
        for r in results[1:]:
            compare.append({
                "baseline": results[0]["name"],
                "this": r["name"],
                "delta_probe": r["summary"]["identity_probe_fresh_mean"] - base["identity_probe_fresh_mean"],
                "delta_abs": r["summary"]["absolute_top1_mean"] - base["absolute_top1_mean"],
            })

    card = {
        "status": "ok",
        "protocol": "fresh LogisticRegression identity probe on frozen clip_emb",
        "runs": results,
        "compare": compare,
    }
    (out / "JOB_COMPLETE.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

    print("\n[SUMMARY]")
    for r in results:
        s = r["summary"]
        print(
            f"  {r['name']:12s} probe={s['identity_probe_fresh_mean']*100:.1f}% "
            f"(chance≈{s['chance']*100:.1f}%) abs={s['absolute_top1_mean']*100:.1f}%",
            flush=True,
        )
    for c in compare:
        print(
            f"  Δ {c['this']} vs {c['baseline']}: "
            f"probe={c['delta_probe']*100:+.1f}pp abs={c['delta_abs']*100:+.1f}pp",
            flush=True,
        )
    print("[OK]", out / "JOB_COMPLETE.json")


if __name__ == "__main__":
    main()
