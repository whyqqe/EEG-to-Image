#!/usr/bin/env python3
"""LOSO v2: Subject-Context EEG (SC-EEG) + closed-form K-shot offset.

Why v2 (from v1 3-fold evidence)
--------------------------------
v1 trained OK (loss↓) but the M×K grid showed:
  • full proj/FiLM K-shot FT (50 steps) *hurt* Top-1 vs zero-shot
  • SC-FiLM alone ≈ no gain
Root cause: tiny-K InfoNCE overfits the projector; FiLM subject prior alone is weak.

v2 changes (same LOSO protocol, no test-label leakage)
------------------------------------------------------
1) Keep SC-FiLM pretraining (stronger context use: lower null-profile dropout).
2) At test, SC-EEG (M unlabeled) does *embedding recenter* toward gallery mean
   (and optional light ridge-SAW) — label-free subject shift correction.
3) K-shot uses *closed-form CLIP-space residual offset* from K labeled pairs
   (optional tiny bias-only FT). No full projector fine-tuning.
4) Same grid: M∈{0,50,100} × K∈{0,1,5,10,20}; ATM ceiling reference.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("HOME", "/project/peilab/why/cache/eeg-brainit/xdg-home")
os.environ.setdefault("HF_HOME", "/project/peilab/why/cache/eeg-brainit/hf")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_brainit.data.things_eeg2_adapt import ThingsEEG2SubjectDataset, collate_batch
from eeg_brainit.models.atm_backbone import info_nce
from eeg_brainit.models.subject_context import SCConditionedEEGEncoder, SubjectContextBank
from eeg_brainit.utils.config import ensure_dirs


ALL_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 11)]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l2_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def retrieval_metrics(queries: np.ndarray, gallery: np.ndarray) -> dict[str, float]:
    q = l2_np(queries)
    g = l2_np(gallery)
    sim = q @ g.T
    n = sim.shape[0]
    gt = np.arange(n)
    order = np.argsort(-sim, axis=1)
    ranks = np.argmax(order == gt[:, None], axis=1)
    top1 = float((ranks < 1).mean())
    top5 = float((ranks < 5).mean())
    top1_idx = order[:, 0]
    counts = np.bincount(top1_idx, minlength=gallery.shape[0]).astype(np.float64)
    if counts.std() < 1e-8:
        hub = 0.0
    else:
        hub = float(((counts - counts.mean()) ** 3).mean() / (counts.std() ** 3 + 1e-8))
    return {"top1": top1, "top5": top5, "hubness_skew": hub}


def atm_ceiling(subject: str, atm_bridge_dir: Path, gallery: np.ndarray) -> dict[str, float]:
    path = atm_bridge_dir / f"{subject}_test_eeg_1024.npy"
    if not path.is_file():
        return {"top1": float("nan"), "top5": float("nan"), "hubness_skew": float("nan")}
    return retrieval_metrics(np.load(path).astype(np.float32), gallery)


def sc_recenter(q: np.ndarray, calib: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Label-free subject shift: move query cloud toward gallery centroid."""
    mu_c = calib.mean(axis=0, keepdims=True)
    mu_g = gallery.mean(axis=0, keepdims=True)
    return l2_np(q - mu_c + mu_g)


def sc_ridge_saw(q: np.ndarray, calib: np.ndarray, eps: float = 1e-2) -> np.ndarray:
    """Mild ZCA on queries using unlabeled calib (heavy diagonal loading)."""
    mu = calib.mean(axis=0, keepdims=True)
    xc = calib - mu
    d = xc.shape[1]
    cov = (xc.T @ xc) / max(len(calib) - 1, 1) + eps * np.eye(d, dtype=np.float64)
    # diagonal-dominant inverse sqrt via eigh
    w, v = np.linalg.eigh(cov)
    w = np.maximum(w, eps)
    p = (v * (1.0 / np.sqrt(w))) @ v.T
    return l2_np(((q - mu) @ p.T).astype(np.float32))


def kshot_ridge_offset(
    eeg_k: np.ndarray, img_k: np.ndarray, alpha: float = 1.0
) -> np.ndarray:
    """Closed-form residual: mean(y - eeg) in CLIP space."""
    return (alpha * (l2_np(img_k) - l2_np(eeg_k)).mean(axis=0)).astype(np.float32)


def choose_alpha_loo(
    eeg_k: np.ndarray, img_k: np.ndarray, alphas: list[float]
) -> float:
    """Pick alpha by leave-one-out cosine on the K support pairs (K>=2)."""
    if len(eeg_k) < 2:
        return 1.0
    eeg_k = l2_np(eeg_k)
    img_k = l2_np(img_k)
    best_a, best_s = 1.0, -1e9
    for a in alphas:
        scores = []
        for i in range(len(eeg_k)):
            mask = np.ones(len(eeg_k), dtype=bool)
            mask[i] = False
            off = a * (img_k[mask] - eeg_k[mask]).mean(axis=0)
            pred = l2_np(eeg_k[i : i + 1] + off[None, :])[0]
            scores.append(float(pred @ img_k[i]))
        s = float(np.mean(scores))
        if s > best_s:
            best_s, best_a = s, a
    return float(best_a)


def train_sc_encoder(model, train_subs, bank, args, device, tag: str) -> SCConditionedEEGEncoder:
    dss = [
        ThingsEEG2SubjectDataset(
            ROOT / args.eeg_root,
            ROOT / args.atm_bridge_dir,
            sub,
            split="train",
            max_samples=args.max_per_sub,
            seed=args.seed,
        )
        for sub in train_subs
    ]
    loader = DataLoader(
        ConcatDataset(dss),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_batch,
        num_workers=0,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    model.train()
    for ep in range(1, args.epochs + 1):
        loss_sum = n = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            use_null = random.random() < args.null_profile_p
            if use_null:
                out = model(batch["eeg"], use_null_profile=True)
            else:
                ctx = bank.sample_batch(batch["subject_id"], m=args.ctx_m, device=device)
                out = model(batch["eeg"], ctx_eeg=ctx)
            loss = info_nce(out["clip_emb"], batch["clip_img"], temp=args.temp)
            if args.lambda_atm > 0 and "atm_emb" in batch:
                loss = loss + args.lambda_atm * F.mse_loss(
                    out["clip_emb"], F.normalize(batch["atm_emb"], dim=-1)
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss) * batch["eeg"].size(0)
            n += batch["eeg"].size(0)
        print(f"[{tag} ep{ep:02d}] loss={loss_sum / max(n, 1):.4f}", flush=True)
    return model


@torch.no_grad()
def encode_eeg_indices(
    model: SCConditionedEEGEncoder,
    subject: str,
    indices: np.ndarray,
    profile: torch.Tensor | None,
    args,
    device,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode selected train indices → (eeg_emb, clip_img)."""
    ds = ThingsEEG2SubjectDataset(
        ROOT / args.eeg_root,
        ROOT / args.atm_bridge_dir,
        subject,
        split="train",
        max_samples=0,
        seed=args.seed,
    )
    model.eval()
    embs, imgs = [], []
    prof = profile.to(device) if profile is not None else None
    idxs = [int(i) for i in indices]
    bs = 64
    for i0 in range(0, len(idxs), bs):
        chunk = idxs[i0 : i0 + bs]
        samples = [ds[j] for j in chunk]
        batch = collate_batch(samples)
        x = batch["eeg"].to(device)
        if prof is None:
            out = model(x, use_null_profile=True)
        else:
            out = model(x, profile=prof)
        embs.append(out["clip_emb"].cpu().numpy())
        imgs.append(batch["clip_img"].numpy())
    return np.concatenate(embs, 0), np.concatenate(imgs, 0)


@torch.no_grad()
def encode_test(
    model: SCConditionedEEGEncoder,
    subject: str,
    profile: torch.Tensor | None,
    args,
    device,
) -> np.ndarray:
    ds = ThingsEEG2SubjectDataset(
        ROOT / args.eeg_root,
        ROOT / args.atm_bridge_dir,
        subject,
        split="test",
        max_samples=0,
        seed=args.seed,
    )
    model.eval()
    embs = []
    prof = profile.to(device) if profile is not None else None
    bs = 64
    for i0 in range(0, len(ds), bs):
        samples = [ds[i] for i in range(i0, min(i0 + bs, len(ds)))]
        batch = collate_batch(samples)
        x = batch["eeg"].to(device)
        if prof is None:
            out = model(x, use_null_profile=True)
        else:
            out = model(x, profile=prof)
        embs.append(out["clip_emb"].cpu().numpy())
    return np.concatenate(embs, 0)


def pick_support_indices(n: int, k: int, m: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    n_cls = n // 10
    if k > 0:
        cls = rng.choice(n_cls, size=min(k, n_cls), replace=False)
        k_idx = []
        for c in cls:
            rep = int(rng.randint(0, 10))
            k_idx.append(int(c) * 10 + rep)
        while len(k_idx) < k:
            j = int(rng.randint(0, n))
            if j not in k_idx:
                k_idx.append(j)
        k_idx = np.array(k_idx[:k], dtype=np.int64)
    else:
        k_idx = np.array([], dtype=np.int64)
    ban = set(int(i) for i in k_idx)
    cand = [i for i in range(n) if i not in ban]
    if m > 0:
        m_idx = rng.choice(cand, size=min(m, len(cand)), replace=False).astype(np.int64)
    else:
        m_idx = np.array([], dtype=np.int64)
    return k_idx, m_idx


def adapt_bias_only(
    model: SCConditionedEEGEncoder,
    subject: str,
    k_idx: np.ndarray,
    profile: torch.Tensor | None,
    args,
    device,
) -> nn.Parameter:
    """Learn a single CLIP-space bias with cosine loss (few steps, tiny capacity)."""
    bias = nn.Parameter(torch.zeros(args.clip_dim, device=device))
    if len(k_idx) == 0:
        return bias
    eeg_k, img_k = encode_eeg_indices(model, subject, k_idx, profile, args, device)
    x = torch.from_numpy(eeg_k).to(device)
    y = F.normalize(torch.from_numpy(img_k).to(device), dim=-1)
    opt = torch.optim.Adam([bias], lr=args.adapt_lr)
    steps = min(args.adapt_steps, max(5, 3 * len(k_idx)))
    for _ in range(steps):
        pred = F.normalize(x + bias, dim=-1)
        loss = (1.0 - (pred * y).sum(dim=-1)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return bias


def apply_sc_calib(q: np.ndarray, calib: np.ndarray, gallery: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none" or calib is None or len(calib) == 0:
        return l2_np(q)
    if mode == "recenter":
        return sc_recenter(q, calib, gallery)
    if mode == "saw":
        return sc_ridge_saw(q, calib, eps=1e-2)
    if mode == "recenter_saw":
        return sc_ridge_saw(sc_recenter(q, calib, gallery), calib, eps=1e-2)
    raise ValueError(mode)


def eval_setting(
    model: SCConditionedEEGEncoder,
    subject: str,
    bank: SubjectContextBank,
    gallery: np.ndarray,
    m: int,
    k: int,
    args,
    device,
    seed: int,
) -> dict:
    n_train = bank.n[subject]
    k_idx, m_idx = pick_support_indices(n_train, k=k, m=m, seed=seed)

    # FiLM profile from unlabeled SC-EEG
    profile = None
    if m > 0 and args.use_film_profile:
        ctx = torch.from_numpy(np.asarray(bank.eeg[subject][m_idx], dtype=np.float32)).to(device)
        model.eval()
        with torch.no_grad():
            profile = model.build_profile_from_eeg(ctx)

    q = encode_test(model, subject, profile, args, device)

    calib = None
    if m > 0:
        calib, _ = encode_eeg_indices(model, subject, m_idx, profile, args, device)
        q = apply_sc_calib(q, calib, gallery, args.sc_mode)

    alpha_used = 0.0
    if k > 0:
        eeg_k, img_k = encode_eeg_indices(model, subject, k_idx, profile, args, device)
        if args.kshot_mode == "ridge":
            alphas = [float(a) for a in args.alpha_grid.split(",") if a.strip()]
            alpha_used = choose_alpha_loo(eeg_k, img_k, alphas) if len(eeg_k) >= 2 else alphas[len(alphas) // 2]
            # if SC calib exists, apply same calib to support eeg before offset
            if calib is not None:
                eeg_k = apply_sc_calib(eeg_k, calib, gallery, args.sc_mode)
            off = kshot_ridge_offset(eeg_k, img_k, alpha=alpha_used)
            q = l2_np(q + off[None, :])
        elif args.kshot_mode == "bias":
            bias = adapt_bias_only(model, subject, k_idx, profile, args, device)
            q = l2_np(q + bias.detach().cpu().numpy()[None, :])
            alpha_used = float(bias.detach().norm().cpu())
        else:
            raise ValueError(args.kshot_mode)

    metrics = retrieval_metrics(q, gallery)
    metrics.update(
        {
            "m": m,
            "k": k,
            "n_k": int(len(k_idx)),
            "n_m": int(len(m_idx)),
            "alpha": float(alpha_used),
            "sc_mode": args.sc_mode,
            "kshot_mode": args.kshot_mode,
        }
    )
    return metrics


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def _sub_seed(subject: str) -> int:
    return int(subject.replace("sub-", ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg-root", default="data/processed/things-eeg2")
    parser.add_argument("--atm-bridge-dir", default="outputs/atm_bridge")
    parser.add_argument("--output-dir", default="outputs/subject_context_kshot/loso_v2")
    parser.add_argument("--encoders", default="atm_style")
    parser.add_argument("--subjects", default=",".join(ALL_SUBJECTS))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-per-sub", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temp", type=float, default=0.07)
    parser.add_argument("--lambda-atm", type=float, default=0.15)
    parser.add_argument("--clip-dim", type=int, default=1024)
    parser.add_argument("--profile-dim", type=int, default=256)
    parser.add_argument("--ctx-m", type=int, default=32)
    parser.add_argument("--null-profile-p", type=float, default=0.1)
    parser.add_argument("--m-list", default="0,50,100")
    parser.add_argument("--k-list", default="0,1,5,10,20")
    parser.add_argument("--sc-mode", default="recenter", choices=["none", "recenter", "saw", "recenter_saw"])
    parser.add_argument("--kshot-mode", default="ridge", choices=["ridge", "bias"])
    parser.add_argument("--alpha-grid", default="0.25,0.5,1.0,1.5,2.0")
    parser.add_argument("--use-film-profile", type=int, default=1)
    parser.add_argument("--adapt-lr", type=float, default=5e-3)
    parser.add_argument("--adapt-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-folds", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / args.output_dir
    ensure_dirs(out_dir, out_dir / "checkpoints", out_dir / "folds")
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    m_list = parse_int_list(args.m_list)
    k_list = parse_int_list(args.k_list)
    gallery = np.load(ROOT / args.atm_bridge_dir / "clip_img_test_1024.npy").astype(np.float32)
    print(
        f"[INFO] device={device} sc={args.sc_mode} kshot={args.kshot_mode} "
        f"film={args.use_film_profile} M={m_list} K={k_list}",
        flush=True,
    )

    all_results = {}
    for enc_name in encoders:
        fold_rows = []
        test_subs = subjects[: args.max_folds] if args.max_folds > 0 else subjects
        for test_sub in test_subs:
            train_subs = [s for s in subjects if s != test_sub]
            print(f"\n[FOLD] encoder={enc_name} test={test_sub} train={len(train_subs)}", flush=True)
            bank = SubjectContextBank(ROOT / args.eeg_root, train_subs + [test_sub], split="train")
            model = SCConditionedEEGEncoder(
                backbone=enc_name,
                clip_dim=args.clip_dim,
                profile_dim=args.profile_dim,
            ).to(device)
            model = train_sc_encoder(
                model, train_subs, bank, args, device, tag=f"{enc_name}-{test_sub}"
            )
            ck = out_dir / "checkpoints" / f"{enc_name}_{test_sub}.pt"
            torch.save(
                {"model": model.state_dict(), "encoder": enc_name, "test_sub": test_sub, "args": vars(args)},
                ck,
            )

            ceiling = atm_ceiling(test_sub, ROOT / args.atm_bridge_dir, gallery)
            settings = {}
            for m in m_list:
                for k in k_list:
                    key = f"m{m}_k{k}"
                    settings[key] = eval_setting(
                        model,
                        test_sub,
                        bank,
                        gallery,
                        m=m,
                        k=k,
                        args=args,
                        device=device,
                        seed=args.seed + _sub_seed(test_sub) * 10007 + m * 17 + k,
                    )
            row = {
                "test_subject": test_sub,
                "encoder": enc_name,
                "atm_ceiling": ceiling,
                "settings": settings,
            }
            fold_rows.append(row)
            (out_dir / "folds" / f"{enc_name}_{test_sub}.json").write_text(
                json.dumps(row, indent=2), encoding="utf-8"
            )
            z = settings.get("m0_k0", {})
            best = max(settings.items(), key=lambda kv: kv[1]["top1"])
            print(
                f"[EVAL {test_sub}] zero={z.get('top1', 0)*100:.2f}% "
                f"best={best[0]}:{best[1]['top1']*100:.2f}% "
                f"atm={ceiling['top1']*100:.2f}%",
                flush=True,
            )

        keys = list(fold_rows[0]["settings"].keys())
        summary = {}
        for key in keys:
            t1 = [r["settings"][key]["top1"] for r in fold_rows]
            t5 = [r["settings"][key]["top5"] for r in fold_rows]
            hub = [r["settings"][key]["hubness_skew"] for r in fold_rows]
            summary[key] = {
                "top1": {"mean": float(np.mean(t1)), "std": float(np.std(t1)), "per_subject": t1},
                "top5": {"mean": float(np.mean(t5)), "std": float(np.std(t5)), "per_subject": t5},
                "hubness_skew": {"mean": float(np.mean(hub)), "std": float(np.std(hub))},
            }
        ceil1 = [r["atm_ceiling"]["top1"] for r in fold_rows]
        ceil5 = [r["atm_ceiling"]["top5"] for r in fold_rows]
        summary["atm_ceiling"] = {
            "top1": {"mean": float(np.nanmean(ceil1)), "std": float(np.nanstd(ceil1)), "per_subject": ceil1},
            "top5": {"mean": float(np.nanmean(ceil5)), "std": float(np.nanstd(ceil5)), "per_subject": ceil5},
            "hubness_skew": {"mean": float("nan"), "std": float("nan")},
        }
        all_results[enc_name] = {"folds": fold_rows, "summary": summary}
        print(f"\n[SUMMARY {enc_name}]")
        for key, st in summary.items():
            print(
                f"  {key:16s} top1={st['top1']['mean']*100:.2f}±{st['top1']['std']*100:.2f}% "
                f"top5={st['top5']['mean']*100:.2f}%",
                flush=True,
            )

    (out_dir / "metrics.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    card = {}
    for enc, blob in all_results.items():
        card[enc] = {
            k: {
                "top1_mean": v["top1"]["mean"],
                "top1_std": v["top1"]["std"],
                "top5_mean": v["top5"]["mean"],
            }
            for k, v in blob["summary"].items()
        }
    (out_dir / "JOB_COMPLETE.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "protocol": "SC-EEG recenter + ridge K-shot LOSO v2",
                "results": card,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("[OK]", out_dir / "JOB_COMPLETE.json")


if __name__ == "__main__":
    main()
