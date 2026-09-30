#!/usr/bin/env python3
"""Physics residual adapter v2: pretrained on multi-subject, then K-shot personalize.

Protocol (high success probability):
  1) Freeze ATM embeddings as authoritative f_θ (34%+ ceiling on THINGS-EEG2)
  2) Pretrain PhysicsResidualAdapter on meta-train subjects:
        clip' = normalize(ATM + s * Δ_phys(raw EEG))
     with InfoNCE to image CLIP + gate entropy + mild ATM identity regularizer
  3) K-shot personalize on held-out subject; evaluate 200-way retrieval

Baselines:
  - ATM teacher only
  - ATM + vanilla residual (pretrained + K-shot)
  - ATM + physics residual (pretrained zero-shot / K-shot)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("HF_HOME", "/project/peilab/why/cache/eeg-brainit/hf")
os.environ.setdefault("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub")
os.environ.setdefault("HOME", "/project/peilab/why/cache/eeg-brainit/xdg-home")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_brainit.data.things_eeg2_adapt import ThingsEEG2SubjectDataset, collate_batch
from eeg_brainit.models.physics_residual_adapter import (
    PhysicsResidualAdapter,
    VanillaResidualAdapter,
    gate_entropy,
    info_nce,
    retrieval_topk,
)
from eeg_brainit.utils.config import ensure_dirs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loaders(subjects, args, split="train"):
    dss = [
        ThingsEEG2SubjectDataset(
            ROOT / args.eeg_root,
            ROOT / args.atm_bridge_dir,
            sub,
            split=split,
            max_samples=args.max_per_sub,
            seed=args.seed,
        )
        for sub in subjects
    ]
    ds = ConcatDataset(dss)
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_batch,
        num_workers=0,
    )


def adapter_loss(out, batch, args, use_gate_reg: bool) -> dict[str, torch.Tensor]:
    losses = {}
    losses["nce"] = info_nce(out["clip_emb"], batch["clip_img"], temp=args.temp)
    # keep residual small unless helpful (identity bias toward ATM)
    losses["res"] = (out["delta"] ** 2).mean()
    total = args.lambda_nce * losses["nce"] + args.lambda_res * losses["res"]
    if use_gate_reg and "gates" in out:
        # encourage peaked gates (minimize entropy)
        losses["gate_H"] = gate_entropy(out["gates"])
        total = total + args.lambda_gate * losses["gate_H"]
    losses["total"] = total
    return losses


def train_adapter(model, loader, args, device, epochs: int, tag: str, use_gate_reg: bool):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    history = []
    model.train()
    for ep in range(1, epochs + 1):
        loss_sum = n = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["eeg"], batch["atm_emb"])
            losses = adapter_loss(out, batch, args, use_gate_reg=use_gate_reg)
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(losses["total"]) * batch["eeg"].size(0)
            n += batch["eeg"].size(0)
        row = {"epoch": ep, "train_loss": loss_sum / max(n, 1)}
        history.append(row)
        extra = ""
        if hasattr(model, "gates"):
            g = model.gates().detach().cpu()
            top = int(g.argmax())
            extra = f" scale={float(model.res_scale):.3f} top_gate={model.specs[top].name}:{float(g[top]):.3f}"
        elif hasattr(model, "res_scale"):
            extra = f" scale={float(model.res_scale):.3f}"
        print(f"[{tag} {ep:03d}] loss={row['train_loss']:.4f}{extra}")
    return history


@torch.no_grad()
def eval_model(model, subject: str, args, device, gallery: np.ndarray) -> dict[str, float]:
    eeg = np.load(ROOT / args.eeg_root / subject / "test_eeg.npy")
    atm = np.load(ROOT / args.atm_bridge_dir / f"{subject}_test_eeg_1024.npy")
    if model is not None:
        model.eval()
    preds = []
    bs = 64
    for i in range(0, len(eeg), bs):
        x = torch.from_numpy(np.asarray(eeg[i : i + bs], dtype=np.float32).copy()).to(device)
        a = torch.from_numpy(np.asarray(atm[i : i + bs], dtype=np.float32).copy()).to(device)
        if model is None:
            preds.append(F.normalize(a, dim=-1).cpu())
        else:
            preds.append(model(x, a)["clip_emb"].cpu())
    pred = torch.cat(preds, 0)
    return retrieval_topk(pred, torch.from_numpy(gallery.astype(np.float32)))


def kshot_adapt(model, subject: str, args, device):
    ds = ThingsEEG2SubjectDataset(
        ROOT / args.eeg_root,
        ROOT / args.atm_bridge_dir,
        subject,
        split="train",
        max_samples=0,
        seed=args.seed,
    )
    rng = random.Random(args.seed + 17)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    support = [ds[i] for i in idxs[: args.k_shot]]
    batch = collate_batch(support)
    batch = {k: v.to(device) for k, v in batch.items()}
    # personalize all adapter params (small module)
    opt = torch.optim.SGD(model.parameters(), lr=args.inner_lr)
    model.train()
    for _ in range(args.inner_steps):
        out = model(batch["eeg"], batch["atm_emb"])
        losses = adapter_loss(out, batch, args, use_gate_reg=isinstance(model, PhysicsResidualAdapter))
        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        opt.step()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg-root", default="data/processed/things-eeg2")
    parser.add_argument("--atm-bridge-dir", default="outputs/atm_bridge")
    parser.add_argument("--output-dir", default="outputs/physics_prior_adapt/v2_sub-08")
    parser.add_argument("--meta-train-subjects", default="sub-01,sub-02,sub-03,sub-04,sub-05,sub-06,sub-07,sub-09")
    parser.add_argument("--meta-test-subject", default="sub-08")
    parser.add_argument("--clip-dim", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--max-per-sub", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--k-shot", type=int, default=50)
    parser.add_argument("--inner-steps", type=int, default=30)
    parser.add_argument("--inner-lr", type=float, default=1e-2)
    parser.add_argument("--temp", type=float, default=0.07)
    parser.add_argument("--lambda-nce", type=float, default=1.0)
    parser.add_argument("--lambda-res", type=float, default=1e-3)
    parser.add_argument("--lambda-gate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / args.output_dir
    ensure_dirs(out_dir, out_dir / "checkpoints")

    train_subs = [s.strip() for s in args.meta_train_subjects.split(",") if s.strip()]
    test_sub = args.meta_test_subject
    print(f"[INFO] v2 device={device} train={train_subs} test={test_sub} k={args.k_shot}")

    gallery = np.load(ROOT / args.atm_bridge_dir / "clip_img_test_1024.npy")
    loader = make_loaders(train_subs, args)

    # --- ATM only ceiling ---
    atm_only = eval_model(None, test_sub, args, device, gallery)
    print(f"[EVAL] atm_only {atm_only}")

    # --- Physics adapter pretrain ---
    phys = PhysicsResidualAdapter(clip_dim=args.clip_dim, rank=args.rank).to(device)
    hist_p = train_adapter(phys, loader, args, device, args.pretrain_epochs, "phys-pre", use_gate_reg=True)
    torch.save({"model": phys.state_dict(), "args": vars(args), "history": hist_p}, out_dir / "checkpoints" / "physics_pretrained.pt")
    phys_zs = eval_model(phys, test_sub, args, device, gallery)
    print(f"[EVAL] physics_pretrained_zeroshot {phys_zs}")

    phys_ft = copy.deepcopy(phys)
    kshot_adapt(phys_ft, test_sub, args, device)
    phys_ks = eval_model(phys_ft, test_sub, args, device, gallery)
    print(f"[EVAL] physics_pretrained_kshot {phys_ks}")
    torch.save({"model": phys_ft.state_dict(), "gates": phys_ft.gates().detach().cpu().tolist()}, out_dir / "checkpoints" / "physics_kshot.pt")

    # --- Vanilla residual ablation ---
    van = VanillaResidualAdapter(clip_dim=args.clip_dim).to(device)
    hist_v = train_adapter(van, loader, args, device, args.pretrain_epochs, "van-pre", use_gate_reg=False)
    torch.save({"model": van.state_dict(), "history": hist_v}, out_dir / "checkpoints" / "vanilla_pretrained.pt")
    van_zs = eval_model(van, test_sub, args, device, gallery)
    print(f"[EVAL] vanilla_pretrained_zeroshot {van_zs}")
    van_ft = copy.deepcopy(van)
    kshot_adapt(van_ft, test_sub, args, device)
    van_ks = eval_model(van_ft, test_sub, args, device, gallery)
    print(f"[EVAL] vanilla_pretrained_kshot {van_ks}")

    gates = phys_ft.gates().detach().cpu()
    gate_map = {s.name: float(gates[i]) for i, s in enumerate(phys_ft.specs)}
    top_gates = sorted(gate_map.items(), key=lambda x: -x[1])[:5]

    metrics = {
        "protocol": "atm_frozen + physics_residual_adapter_pretrain + kshot",
        "meta_test_subject": test_sub,
        "meta_train_subjects": train_subs,
        "k_shot": args.k_shot,
        "adapter_params": phys.trainable_param_count(),
        "results": {
            "atm_only": atm_only,
            "physics_pretrained_zeroshot": phys_zs,
            "physics_pretrained_kshot": phys_ks,
            "vanilla_pretrained_zeroshot": van_zs,
            "vanilla_pretrained_kshot": van_ks,
        },
        "gate_map": gate_map,
        "top_gates": top_gates,
        "res_scale_physics": float(phys_ft.res_scale.detach()),
        "res_scale_vanilla": float(van_ft.res_scale.detach()),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "JOB_COMPLETE.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "atm_only_top1": atm_only["top1"],
                "physics_zs_top1": phys_zs["top1"],
                "physics_kshot_top1": phys_ks["top1"],
                "vanilla_zs_top1": van_zs["top1"],
                "vanilla_kshot_top1": van_ks["top1"],
                "delta_vs_atm": phys_ks["top1"] - atm_only["top1"],
                "delta_vs_vanilla": phys_ks["top1"] - van_ks["top1"],
                "top_gates": top_gates,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("[OK] wrote", out_dir / "metrics.json")
    print("[SUMMARY]", json.dumps(json.loads((out_dir / "JOB_COMPLETE.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
