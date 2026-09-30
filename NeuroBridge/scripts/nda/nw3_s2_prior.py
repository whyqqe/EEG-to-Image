#!/usr/bin/env python3
"""NeuroWeave v3 Stage-2: Per-modality prior refinement (M3).

Lightweight residual denoiser: given noisy CLIP target z_I^t and EEG semantic
condition z_sem, predict noise. Sampling at test time projects z_sem onto the
real CLIP image/text/depth/edge manifold (fixes support mismatch C).

Trained on TRAIN concepts only (instance CLIP targets from gem/cond_cache).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NB_ROOT = Path("/project/peilab/why/NeuroBridge")
sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))
import leakfree as LF  # noqa: E402

MODALITIES = ("img", "txt", "depth", "edge")


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-8)


class PriorDenoiser(nn.Module):
    """ε_θ(z_t, t, z_sem) — ATM-style lightweight prior."""

    def __init__(self, dim: int = 1024, hidden: int = 1024, t_dim: int = 64):
        super().__init__()
        self.t_emb = nn.Sequential(nn.Linear(1, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.net = nn.Sequential(
            nn.Linear(dim + dim + t_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, z_sem: torch.Tensor) -> torch.Tensor:
        te = self.t_emb(t.unsqueeze(-1).float())
        return self.net(torch.cat([z_t, z_sem, te], dim=-1))


def cosine_schedule(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """ᾱ_t in (0,1) for t in [0,1]."""
    return torch.cos((t + s) / (1 + s) * np.pi * 0.5).clamp(min=1e-4) ** 2


@torch.no_grad()
def sample(model: PriorDenoiser, z_sem: torch.Tensor, steps: int = 20) -> torch.Tensor:
    """DDIM-ish ancestral sampling from N(0,I) conditioned on z_sem."""
    b, d = z_sem.shape
    z = torch.randn(b, d, device=z_sem.device, dtype=z_sem.dtype)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=z_sem.device)
    for i in range(steps):
        t = ts[i].expand(b)
        t_next = ts[i + 1].expand(b)
        a = cosine_schedule(t).unsqueeze(-1)
        a_n = cosine_schedule(t_next).unsqueeze(-1)
        eps = model(z, t, z_sem)
        x0 = (z - (1 - a).sqrt() * eps) / a.sqrt().clamp(min=1e-4)
        x0 = F.normalize(x0, dim=-1)
        z = a_n.sqrt() * x0 + (1 - a_n).sqrt() * eps
    return F.normalize(z, dim=-1)


def load_targets(cache: Path, modality: str) -> tuple[np.ndarray, np.ndarray]:
    key = {"img": "clip_img1024", "txt": "clip_img1024",  # txt uses image as proxy if no flat text
           "depth": "clip_depth1024", "edge": "clip_edge1024"}[modality]
    tr = np.load(cache / f"{key}_train.npy").astype(np.float32)
    te = np.load(cache / f"{key}_test.npy").astype(np.float32)
    return l2n(tr), l2n(te)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--s1-dir", type=str, required=True, help="S1 out with z_sem exports + checkpoint")
    ap.add_argument("--z-root", type=str, default=str(NB_ROOT / "outputs/ocf/intra_z"))
    ap.add_argument("--test-subject", type=int, default=8)
    ap.add_argument("--clip-cache", type=str, default=str(NB_ROOT / "outputs/gem/cond_cache"))
    ap.add_argument("--clip-text-dir", type=str, default=str(NB_ROOT / "outputs/nda_ss/sub-08/clip_text"))
    ap.add_argument("--split-json", type=str, default=str(NB_ROOT / "outputs/leakfree/split.json"))
    ap.add_argument("--modalities", type=str, default="img,txt,depth,edge")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sample-steps", type=int, default=20)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "conds").mkdir(exist_ok=True)
    sid = f"{args.test_subject:02d}"
    s1 = Path(args.s1_dir)

    # Condition = S1 semantic projection of train/test z
    sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))
    from nw3_s1_train import FactorizedEncoder, l2t  # noqa: E402

    blob = torch.load(s1 / "best.pth", map_location=dev, weights_only=False)
    enc = FactorizedEncoder(z_dim=blob["z_dim"]).to(dev)
    enc.load_state_dict(blob["model"])
    enc.eval()
    ztr = np.load(Path(args.z_root) / f"sub-{sid}" / "shared_r_train.npy").astype(np.float32)
    zte = np.load(Path(args.z_root) / f"sub-{sid}" / "shared_r_test.npy").astype(np.float32)
    ztr = ztr / np.linalg.norm(ztr, axis=-1, keepdims=True).clip(min=1e-8)
    zte = zte / np.linalg.norm(zte, axis=-1, keepdims=True).clip(min=1e-8)
    with torch.no_grad():
        z_sem_tr = []
        for s in range(0, len(ztr), 512):
            o = enc(torch.from_numpy(ztr[s:s + 512]).to(dev))
            z_sem_tr.append(l2t(o["z_sem"]).cpu().numpy())
        z_sem_tr = np.concatenate(z_sem_tr, 0).astype(np.float32)
        z_sem_te = l2t(enc(torch.from_numpy(zte).to(dev))["z_sem"]).cpu().numpy().astype(np.float32)

    split = LF.load(args.split_json)
    fit_i = LF.rows_for(split, "fit", len(z_sem_tr))
    val_i = LF.rows_for(split, "val_b", len(z_sem_tr))

    mods = [m.strip() for m in args.modalities.split(",") if m.strip()]
    report = {"modalities": {}, "sample_steps": args.sample_steps}

    # text targets: concept phrases mapped to train rows via captions
    txt_bank = l2n(np.load(Path(args.clip_text_dir) / "train" / "text_concept_clip.npy").astype(np.float32))
    phrases = json.loads((Path(args.clip_text_dir) / "train" / "concept_phrases.json").read_text(encoding="utf-8"))
    caps = [json.loads(l) for l in (NB_ROOT / "outputs/g2/captions/captions_train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    index = {p: i for i, p in enumerate(phrases)}
    cid = []
    for d in caps:
        c = Path(d.get("path") or "").parent.name.split("_", 1)[1].replace("_", " ")
        cid.append(index[c])
    cid = np.asarray(cid, dtype=np.int64)
    txt_tr = txt_bank[cid]

    for mod in mods:
        ckpt = out / f"prior_{mod}.pth"
        if mod == "txt":
            tgt_tr, tgt_te_proxy = txt_tr, None
        else:
            tgt_tr, tgt_te_proxy = load_targets(Path(args.clip_cache), mod)

        if args.resume and ckpt.is_file() and (out / "conds" / f"z_{mod}_test.npy").is_file():
            print(f"[s2] skip {mod} (exists)")
            model = PriorDenoiser().to(dev)
            model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model"])
        else:
            model = PriorDenoiser().to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
            best_loss = 1e9
            for ep in range(args.epochs):
                model.train()
                order = np.random.permutation(fit_i)
                run, n = 0.0, 0
                for s in range(0, len(order), args.batch_size):
                    ix = order[s:s + args.batch_size]
                    x0 = torch.from_numpy(tgt_tr[ix]).to(dev)
                    cond = torch.from_numpy(z_sem_tr[ix]).to(dev)
                    t = torch.rand(len(ix), device=dev)
                    a = cosine_schedule(t).unsqueeze(-1)
                    eps = torch.randn_like(x0)
                    z_t = a.sqrt() * x0 + (1 - a).sqrt() * eps
                    pred = model(z_t, t, cond)
                    loss = F.mse_loss(pred, eps)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    run += float(loss.detach().cpu())
                    n += 1
                # val
                model.eval()
                with torch.no_grad():
                    ix = val_i
                    x0 = torch.from_numpy(tgt_tr[ix]).to(dev)
                    cond = torch.from_numpy(z_sem_tr[ix]).to(dev)
                    t = torch.rand(len(ix), device=dev)
                    a = cosine_schedule(t).unsqueeze(-1)
                    eps = torch.randn_like(x0)
                    z_t = a.sqrt() * x0 + (1 - a).sqrt() * eps
                    vloss = float(F.mse_loss(model(z_t, t, cond), eps).cpu())
                print(f"[s2:{mod}] ep={ep:02d} train={run/max(n,1):.4f} val={vloss:.4f}")
                if vloss < best_loss:
                    best_loss = vloss
                    torch.save({"model": model.state_dict(), "mod": mod, "val_loss": vloss, "epoch": ep}, ckpt)
            model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model"])
            report["modalities"][mod] = {"best_val_loss": best_loss}

        # sample test conditions
        model.eval()
        with torch.no_grad():
            z_out = sample(model, torch.from_numpy(z_sem_te).to(dev), steps=args.sample_steps)
            z_np = z_out.cpu().numpy().astype(np.float32)
        np.save(out / "conds" / f"z_{mod}_test.npy", z_np)
        # manifold proximity: cosine to nearest train target mean (diagnostic)
        if mod != "txt":
            g = tgt_tr.mean(0)
            g = g / (np.linalg.norm(g) + 1e-8)
            cos = float((z_np * g).sum(1).mean())
        else:
            cos = float((z_np * txt_bank.mean(0) / (np.linalg.norm(txt_bank.mean(0)) + 1e-8)).sum(1).mean())
        report["modalities"].setdefault(mod, {})["test_mean_cos_to_train_centroid"] = cos
        print(f"[s2:{mod}] exported {z_np.shape} centroid_cos={cos:.4f}")

    # also save raw z_sem as baseline
    np.save(out / "conds" / "z_sem_test.npy", z_sem_te)
    (out / "s2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[s2] done")


if __name__ == "__main__":
    main()
