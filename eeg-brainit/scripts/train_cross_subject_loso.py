#!/usr/bin/env python3
"""Cross-subject LOSO EEG→image retrieval + label-free test-time calibration.

Plan executed:
  1) Train frozen-CLIP-aligned EEG encoders on 9 subjects (leave-one-out)
  2) Encode held-out subject test EEG (no labels used for calibration stats)
  3) Apply calibration ladder: cosine → CW → SAW → CSLS → Ada-CSLS → PoE → Phys-PoE
  4) Aggregate 10-fold metrics + hubness

Differentiation vs SATTC: physiology-aware query prior from channel×band energies.
Public stack: OpenCLIP ViT-H/14 targets (via atm_bridge) + optional ATM distill.
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
from eeg_brainit.models.atm_backbone import AtmStyleEEGEncoder, info_nce
from eeg_brainit.models.cross_subject_tta import run_calibration_suite
from eeg_brainit.models.eeg2fmri import TemporalEEGEncoder
from eeg_brainit.models.physics_prior_lora import PhysicsFeatureExtractor, build_subspace_specs
from eeg_brainit.utils.config import ensure_dirs


ALL_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 11)]


class TemporalCLIPEncoder(nn.Module):
    """Second public-style backbone for multi-encoder claim."""

    def __init__(self, n_channels: int = 63, clip_dim: int = 1024):
        super().__init__()
        self.enc = TemporalEEGEncoder(n_channels=n_channels, d_model=512, dropout=0.2, depth=3)
        self.proj = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, clip_dim))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.enc(x.float())
        return {"clip_emb": F.normalize(self.proj(h), dim=-1)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_encoder(name: str, clip_dim: int) -> nn.Module:
    if name == "atm_style":
        return AtmStyleEEGEncoder(n_channels=63, seq_len=250, clip_dim=clip_dim)
    if name == "temporal":
        return TemporalCLIPEncoder(n_channels=63, clip_dim=clip_dim)
    raise ValueError(name)


def train_encoder(model, subjects, args, device, tag: str) -> nn.Module:
    dss = [
        ThingsEEG2SubjectDataset(
            ROOT / args.eeg_root,
            ROOT / args.atm_bridge_dir,
            sub,
            split="train",
            max_samples=args.max_per_sub,
            seed=args.seed,
        )
        for sub in subjects
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
            out = model(batch["eeg"])
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
        print(f"[{tag} ep{ep:02d}] loss={loss_sum/max(n,1):.4f}", flush=True)
    return model


@torch.no_grad()
def encode_split(model, subject: str, split: str, args, device) -> tuple[np.ndarray, np.ndarray | None]:
    ds = ThingsEEG2SubjectDataset(
        ROOT / args.eeg_root,
        ROOT / args.atm_bridge_dir,
        subject,
        split=split,
        max_samples=0,
        seed=args.seed,
    )
    # For test split, img may be missing; we only need eeg embeddings
    model.eval()
    embs = []
    eeg_raw = []
    bs = 64
    # manual loop over indices for mmap safety
    n = len(ds)
    for i0 in range(0, n, bs):
        samples = [ds[i] for i in range(i0, min(i0 + bs, n))]
        batch = collate_batch(samples)
        x = batch["eeg"].to(device)
        embs.append(model(x)["clip_emb"].cpu().numpy())
        eeg_raw.append(batch["eeg"].numpy())
    return np.concatenate(embs, 0), np.concatenate(eeg_raw, 0)


def subspace_energy_np(eeg: np.ndarray) -> np.ndarray:
    """Compute label-free channel×band energies via PhysicsFeatureExtractor (CPU)."""
    specs = build_subspace_specs()
    feat = PhysicsFeatureExtractor(specs, out_dim=32, sfreq=250.0).eval()
    energies = []
    with torch.no_grad():
        for i0 in range(0, len(eeg), 64):
            x = torch.from_numpy(eeg[i0 : i0 + 64].astype(np.float32))
            # use bank only: absolute mean over projected feat dims as energy proxy
            xb = feat.bank(x)  # (B,C,F,T)
            # per-spec energy
            row = []
            for spec in specs:
                f_idx = feat.band_to_idx[spec.band_hz]
                e = xb[:, list(spec.channel_idx), f_idx, :].abs().mean(dim=(1, 2))
                row.append(e)
            energies.append(torch.stack(row, dim=1).numpy())
    return np.concatenate(energies, 0).astype(np.float32)


def aggregate(fold_rows: list[dict], method: str) -> dict:
    vals = {k: [] for k in ("top1", "top5", "hubness_skew")}
    for row in fold_rows:
        m = row["methods"][method]
        for k in vals:
            vals[k].append(m[k])
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "per_subject": v} for k, v in vals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg-root", default="data/processed/things-eeg2")
    parser.add_argument("--atm-bridge-dir", default="outputs/atm_bridge")
    parser.add_argument("--output-dir", default="outputs/cross_subject_tta/loso_v1")
    parser.add_argument("--encoders", default="atm_style,temporal")
    parser.add_argument("--subjects", default=",".join(ALL_SUBJECTS))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-per-sub", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temp", type=float, default=0.07)
    parser.add_argument("--lambda-atm", type=float, default=0.15)
    parser.add_argument("--clip-dim", type=int, default=1024)
    parser.add_argument("--csls-k", type=int, default=10)
    parser.add_argument("--poe-beta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-folds", type=int, default=0, help="0=all subjects")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / args.output_dir
    ensure_dirs(out_dir, out_dir / "checkpoints", out_dir / "folds")
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    gallery = np.load(ROOT / args.atm_bridge_dir / "clip_img_test_1024.npy").astype(np.float32)
    print(f"[INFO] device={device} subjects={subjects} encoders={encoders}", flush=True)

    all_results = {}
    for enc_name in encoders:
        fold_rows = []
        test_subs = subjects[: args.max_folds] if args.max_folds > 0 else subjects
        for test_sub in test_subs:
            train_subs = [s for s in subjects if s != test_sub]
            print(f"\n[FOLD] encoder={enc_name} test={test_sub} train={len(train_subs)}", flush=True)
            model = build_encoder(enc_name, args.clip_dim).to(device)
            model = train_encoder(model, train_subs, args, device, tag=f"{enc_name}-{test_sub}")
            ck = out_dir / "checkpoints" / f"{enc_name}_{test_sub}.pt"
            torch.save({"model": model.state_dict(), "encoder": enc_name, "test_sub": test_sub}, ck)

            q_test, eeg_test = encode_split(model, test_sub, "test", args, device)
            # unlabeled calib: use held-out train EEG embeddings (labels unused)
            q_calib, _ = encode_split(model, test_sub, "train", args, device)
            # subsample calib for SAW covariance stability / speed
            rng = np.random.RandomState(args.seed)
            if len(q_calib) > 2000:
                idx = rng.choice(len(q_calib), 2000, replace=False)
                q_calib = q_calib[idx]
            energy = subspace_energy_np(eeg_test)

            methods = run_calibration_suite(
                q_test,
                gallery,
                calib_queries=q_calib,
                subspace_energy=energy,
                csls_k=args.csls_k,
                beta=args.poe_beta,
            )
            row = {"test_subject": test_sub, "encoder": enc_name, "methods": methods}
            fold_rows.append(row)
            (out_dir / "folds" / f"{enc_name}_{test_sub}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            best = max(methods.items(), key=lambda kv: kv[1]["top1"])
            print(
                f"[EVAL {test_sub}] cosine={methods['cosine']['top1']*100:.2f}% "
                f"poe={methods['sattc_like_poe']['top1']*100:.2f}% "
                f"phys={methods.get('phys_poe',{}).get('top1',0)*100:.2f}% "
                f"best={best[0]}:{best[1]['top1']*100:.2f}%",
                flush=True,
            )

        method_names = list(fold_rows[0]["methods"].keys())
        summary = {m: aggregate(fold_rows, m) for m in method_names}
        all_results[enc_name] = {"folds": fold_rows, "summary": summary}
        print(f"\n[SUMMARY {enc_name}]")
        for m, st in summary.items():
            print(f"  {m:16s} top1={st['top1']['mean']*100:.2f}±{st['top1']['std']*100:.2f}% "
                  f"top5={st['top5']['mean']*100:.2f}% hub={st['hubness_skew']['mean']:.3f}")

    (out_dir / "metrics.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    # compact complete card
    card = {}
    for enc, blob in all_results.items():
        card[enc] = {
            m: {"top1_mean": v["top1"]["mean"], "top5_mean": v["top5"]["mean"], "hub_mean": v["hubness_skew"]["mean"]}
            for m, v in blob["summary"].items()
        }
    (out_dir / "JOB_COMPLETE.json").write_text(json.dumps({"status": "ok", "results": card}, indent=2), encoding="utf-8")
    print("[OK]", out_dir / "JOB_COMPLETE.json")


if __name__ == "__main__":
    main()
