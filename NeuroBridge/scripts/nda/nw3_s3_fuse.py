#!/usr/bin/env python3
"""NeuroWeave v3 Stage-3: Cross-modal semantic fusion (M4 / N5).

Takes refined per-modality CLIP conditions {img, txt, depth, edge} and learns a
cross-modal attention fusion → z_fused. Trained with InfoNCE against image
gallery + MSE to image CLIP (held-in fit/val_b).

At test time exports: z_img, z_txt, z_depth, z_edge, z_fused  (5 IP conditions).
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

MODS = ("img", "txt", "depth", "edge")


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-8)


def l2t(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


class CrossModalFusion(nn.Module):
    def __init__(self, dim: int = 1024, n_heads: int = 4, n_mods: int = 4):
        super().__init__()
        self.mod_emb = nn.Parameter(torch.randn(n_mods, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        # xs: (B, M, D)
        h = xs + self.mod_emb.unsqueeze(0)
        a, _ = self.attn(h, h, h, need_weights=False)
        h = self.norm1(h + a)
        h = self.norm2(h + self.ff(h))
        # mean pool + residual of image token
        fused = self.out(h.mean(1) + h[:, 0])
        return fused


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--s2-dir", type=str, required=True)
    ap.add_argument("--s1-dir", type=str, required=True)
    ap.add_argument("--z-root", type=str, default=str(NB_ROOT / "outputs/ocf/intra_z"))
    ap.add_argument("--test-subject", type=int, default=8)
    ap.add_argument("--clip-cache", type=str, default=str(NB_ROOT / "outputs/gem/cond_cache"))
    ap.add_argument("--g-img", type=str, default=str(NB_ROOT / "outputs/uck/shared/g_img_concept.npy"))
    ap.add_argument("--captions-dir", type=str, default=str(NB_ROOT / "outputs/g2/captions"))
    ap.add_argument("--clip-text-dir", type=str, default=str(NB_ROOT / "outputs/nda_ss/sub-08/clip_text"))
    ap.add_argument("--split-json", type=str, default=str(NB_ROOT / "outputs/leakfree/split.json"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.07)
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
    s2 = Path(args.s2_dir)
    sid = f"{args.test_subject:02d}"

    # Rebuild train-side modality conditions by re-sampling priors (deterministic seed)
    sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))
    from nw3_s1_train import FactorizedEncoder, l2t as s1_l2t  # noqa: E402
    from nw3_s2_prior import PriorDenoiser, sample, cosine_schedule  # noqa: E402

    blob = torch.load(Path(args.s1_dir) / "best.pth", map_location=dev, weights_only=False)
    enc = FactorizedEncoder(z_dim=blob["z_dim"]).to(dev)
    enc.load_state_dict(blob["model"])
    enc.eval()
    ztr = np.load(Path(args.z_root) / f"sub-{sid}" / "shared_r_train.npy").astype(np.float32)
    ztr = ztr / np.linalg.norm(ztr, axis=-1, keepdims=True).clip(min=1e-8)
    with torch.no_grad():
        zs = []
        for s in range(0, len(ztr), 512):
            zs.append(s1_l2t(enc(torch.from_numpy(ztr[s:s + 512]).to(dev))["z_sem"]).cpu().numpy())
        z_sem_tr = np.concatenate(zs, 0).astype(np.float32)

    # For fusion training we use TEACHER targets (true CLIP) as inputs on train,
    # and S2-sampled conditions at test. This matches CogCapPro's align-then-fuse.
    img_tr = l2n(np.load(Path(args.clip_cache) / "clip_img1024_train.npy").astype(np.float32))
    depth_tr = l2n(np.load(Path(args.clip_cache) / "clip_depth1024_train.npy").astype(np.float32))
    edge_tr = l2n(np.load(Path(args.clip_cache) / "clip_edge1024_train.npy").astype(np.float32))
    phrases = json.loads((Path(args.clip_text_dir) / "train" / "concept_phrases.json").read_text(encoding="utf-8"))
    txt_bank = l2n(np.load(Path(args.clip_text_dir) / "train" / "text_concept_clip.npy").astype(np.float32))
    caps = [json.loads(l) for l in (Path(args.captions_dir) / "captions_train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    index = {p: i for i, p in enumerate(phrases)}
    cid = np.asarray([
        index[Path(d.get("path") or "").parent.name.split("_", 1)[1].replace("_", " ")]
        for d in caps
    ], dtype=np.int64)
    txt_tr = txt_bank[cid]
    g_img = l2n(np.load(args.g_img).astype(np.float32))

    split = LF.load(args.split_json)
    fit_i = LF.rows_for(split, "fit", len(z_sem_tr))
    val_i = LF.rows_for(split, "val_b", len(z_sem_tr))

    model = CrossModalFusion().to(dev)
    ckpt = out / "fusion.pth"
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    g_img_t = torch.from_numpy(g_img).to(dev)

    if not (args.resume and ckpt.is_file() and (out / "conds" / "z_fused_test.npy").is_file()):
        best = -1.0
        for ep in range(args.epochs):
            model.train()
            order = np.random.permutation(fit_i)
            run, n = 0.0, 0
            for s in range(0, len(order), args.batch_size):
                ix = order[s:s + args.batch_size]
                stack = np.stack([img_tr[ix], txt_tr[ix], depth_tr[ix], edge_tr[ix]], 1)
                x = torch.from_numpy(stack).to(dev)
                c = torch.from_numpy(cid[ix]).to(dev)
                fused = model(x)
                logits = (l2t(fused) @ l2t(g_img_t).T) / args.tau
                loss = F.cross_entropy(logits, c) + 0.5 * F.mse_loss(l2t(fused), torch.from_numpy(img_tr[ix]).to(dev))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                run += float(loss.detach().cpu())
                n += 1
            model.eval()
            with torch.no_grad():
                stack = np.stack([img_tr[val_i], txt_tr[val_i], depth_tr[val_i], edge_tr[val_i]], 1)
                fused = model(torch.from_numpy(stack).to(dev))
                pred = (l2t(fused) @ l2t(g_img_t).T).argmax(1)
                top1 = float((pred == torch.from_numpy(cid[val_i]).to(dev)).float().mean().cpu())
            print(f"[s3] ep={ep:02d} loss={run/max(n,1):.4f} val_top1={top1:.4f}")
            if top1 > best:
                best = top1
                torch.save({"model": model.state_dict(), "val_top1": top1, "epoch": ep}, ckpt)
        (out / "s3_report.json").write_text(json.dumps({"best_val_top1": best}, indent=2), encoding="utf-8")
    else:
        print("[s3] resume existing fusion")

    model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model"])
    model.eval()

    # Test: load S2-sampled modality conditions, fuse
    stacks = []
    exports = {}
    for m in MODS:
        p = s2 / "conds" / f"z_{m}_test.npy"
        if not p.is_file():
            raise SystemExit(f"[FATAL] missing {p}")
        exports[m] = np.load(p).astype(np.float32)
        stacks.append(exports[m])
    X = np.stack(stacks, 1)  # (200,4,1024)
    with torch.no_grad():
        fused = l2t(model(torch.from_numpy(X).to(dev))).cpu().numpy().astype(np.float32)
    for m, arr in exports.items():
        np.save(out / "conds" / f"z_{m}_test.npy", l2n(arr))
    np.save(out / "conds" / "z_fused_test.npy", fused)
    # primary IP embed = fused (single-channel path for struct_inject)
    np.save(out / "conds" / "ip_primary_test.npy", fused)
    print(f"[s3] exported fused {fused.shape} + 4 modalities")


if __name__ == "__main__":
    main()
