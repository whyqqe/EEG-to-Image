#!/usr/bin/env python3
"""NeuroWeave v3 Stage-1: Task-Factorized Encoder (M2).

Two streams from frozen EEG embedding z (shared_r):
  z_sem  — InfoNCE (train gallery) + MSE to concept-mean CLIP  (fixes A/B)
  z_spa  — L1 to SDXL VAE latent + depth + reconstruct z       (fixes D + BrainAE-style)

Leak-free: fit gradients, val_b selection, test export only.
Never regresses TEST VAE latents / depth as training targets.
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
from torch.cuda.amp import GradScaler, autocast

NB_ROOT = Path("/project/peilab/why/NeuroBridge")
sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))
import leakfree as LF  # noqa: E402


def l2n(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-8)
    return x / n


def l2t(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


class FactorizedEncoder(nn.Module):
    """Semantic stream + spatial stream (depth + VAE) + z-reconstructor."""

    def __init__(self, z_dim: int = 1024, hidden: int = 1024, ip_dim: int = 1024):
        super().__init__()
        self.sem = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, ip_dim),
        )
        self.fc = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 128 * 8 * 8),
        )
        self.up = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 48, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(48, 32, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
        )
        self.depth_head = nn.Conv2d(32, 1, 3, padding=1)
        self.vae_head = nn.Conv2d(32, 4, 3, padding=1)
        nn.init.zeros_(self.vae_head.weight)
        nn.init.zeros_(self.vae_head.bias)
        # BrainAE-style: reconstruct EEG embedding from spatial field (anti-collapse).
        self.recon = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, hidden), nn.GELU(),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        z_sem = self.sem(z)
        field = self.up(self.fc(z).view(-1, 128, 8, 8))
        return {
            "z_sem": z_sem,
            "F": field,
            "depth": torch.sigmoid(self.depth_head(field)).squeeze(1),
            "vae": self.vae_head(field),
            "z_hat": self.recon(field),
        }


def gallery_nce(q: torch.Tensor, gallery: torch.Tensor, cid: torch.Tensor, tau: float) -> torch.Tensor:
    logits = (l2t(q) @ l2t(gallery).T) / tau
    return F.cross_entropy(logits, cid)


def retrieval_top1(q: torch.Tensor, gallery: torch.Tensor, cid: torch.Tensor) -> float:
    pred = (l2t(q) @ l2t(gallery).T).argmax(1)
    return float((pred == cid).float().mean().cpu())


def build_concept_bank(clip_text_dir: Path) -> tuple[np.ndarray, list[str]]:
    phrases = json.loads((clip_text_dir / "train" / "concept_phrases.json").read_text(encoding="utf-8"))
    bank = l2n(np.load(clip_text_dir / "train" / "text_concept_clip.npy").astype(np.float32))
    return bank, phrases


def concept_ids_from_captions(captions_dir: Path, phrases: list[str], split: str) -> np.ndarray:
    path = captions_dir / f"captions_{split}.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    index = {p: i for i, p in enumerate(phrases)}

    def concept_of(d: dict) -> str:
        p = d.get("path") or d.get("image") or ""
        return Path(p).parent.name.split("_", 1)[1].replace("_", " ")

    concepts = [concept_of(d) for d in rows]
    missing = sorted({c for c in concepts if c not in index})
    if missing:
        raise SystemExit(f"[FATAL] {len(missing)} concepts missing from gallery (e.g. {missing[:3]})")
    return np.asarray([index[c] for c in concepts], dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--test-subject", type=int, default=8)
    ap.add_argument("--z-root", type=str, default=str(NB_ROOT / "outputs/ocf/intra_z"))
    ap.add_argument("--captions-dir", type=str, default=str(NB_ROOT / "outputs/g2/captions"))
    ap.add_argument("--clip-text-dir", type=str, default=str(NB_ROOT / "outputs/nda_ss/sub-08/clip_text"))
    ap.add_argument("--g-img", type=str, default=str(NB_ROOT / "outputs/uck/shared/g_img_concept.npy"))
    ap.add_argument("--vae-cache", type=str, default=str(NB_ROOT / "outputs/sdedit_ll_full10/shared/vae_cache"))
    ap.add_argument("--depth-train", type=str, default=str(NB_ROOT / "outputs/uck/shared/gt_depth/train_depth_64.npy"))
    ap.add_argument("--depth-test", type=str, default=str(NB_ROOT / "outputs/hcma_s_full10/shared/gt_depth/test_depth_64.npy"))
    ap.add_argument("--split-json", type=str, default=str(NB_ROOT / "outputs/leakfree/split.json"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--w-nce-txt", type=float, default=1.0)
    ap.add_argument("--w-nce-img", type=float, default=1.0)
    ap.add_argument("--w-mse", type=float, default=0.5, help="MSE to concept-mean CLIP (anchor)")
    ap.add_argument("--w-depth", type=float, default=1.0)
    ap.add_argument("--w-vae", type=float, default=1.0)
    ap.add_argument("--w-recon", type=float, default=0.2, help="EEG-embedding reconstruction")
    ap.add_argument("--scaling-factor", type=float, default=0.13025)
    ap.add_argument("--decode-rgb", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--export-only", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "conds").mkdir(exist_ok=True)
    (out / "spatial").mkdir(exist_ok=True)
    sid = f"{args.test_subject:02d}"

    ztr = l2n(np.load(Path(args.z_root) / f"sub-{sid}" / "shared_r_train.npy").astype(np.float32))
    zte = l2n(np.load(Path(args.z_root) / f"sub-{sid}" / "shared_r_test.npy").astype(np.float32))
    g_txt, phrases = build_concept_bank(Path(args.clip_text_dir))
    g_img = l2n(np.load(args.g_img).astype(np.float32))
    assert g_txt.shape[0] == g_img.shape[0] == 1654
    cid_tr = concept_ids_from_captions(Path(args.captions_dir), phrases, "train")
    # Test concepts are DISJOINT from the 1654-train gallery by protocol —
    # never map them to gallery ids (that would raise "missing from gallery").
    test_names = set()
    for line in (Path(args.captions_dir) / "captions_test.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = json.loads(line).get("path") or ""
        test_names.add(Path(p).parent.name.split("_", 1)[1].replace("_", " "))
    leak = sorted(n for n in test_names if n in set(phrases))
    if leak:
        raise SystemExit(f"[FATAL] {len(leak)} test concepts in train gallery phrases")
    print(f"[audit] gallery {len(phrases)} train concepts, test {len(test_names)}, intersection 0")

    vtr = np.load(Path(args.vae_cache) / "train_vae_latents_f16.npy", mmap_mode="r")
    dtr = np.load(args.depth_train, mmap_mode="r")
    assert len(vtr) == len(ztr) == len(dtr) == len(cid_tr)

    split = LF.load(args.split_json)
    fit_i = LF.rows_for(split, "fit", len(ztr))
    val_i = LF.rows_for(split, "val_b", len(ztr))
    if set(fit_i.tolist()) & set(val_i.tolist()):
        raise SystemExit("[FATAL] fit/val_b overlap")

    # VAE normalization from fit only
    chunk = np.asarray(vtr[fit_i[:: max(1, len(fit_i) // 2048)]], dtype=np.float32)
    v_mean = chunk.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    v_std = chunk.std(axis=(0, 2, 3), keepdims=True).astype(np.float32).clip(1e-3)

    g_txt_t = torch.from_numpy(g_txt).to(dev)
    g_img_t = torch.from_numpy(g_img).to(dev)
    # concept-mean targets for MSE anchor (= gallery rows)
    model = FactorizedEncoder(z_dim=ztr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=dev.type == "cuda")
    history: list[dict] = []
    best = {"score": -1.0, "epoch": -1}
    ckpt = out / "best.pth"

    if args.resume and ckpt.is_file() and (out / "s1_report.json").is_file() and not args.export_only:
        print(f"[s1] resume checkpoint present; jump to export (use --export-only 0 --resume 0 to retrain)")
        # fall through to export if conds missing
        if (out / "conds" / "z_sem_test.npy").is_file() and (out / "spatial" / "pred_vae_test_scaled.npy").is_file():
            print("[s1] exports already present; done")
            return

    if not args.export_only and not (args.resume and ckpt.is_file()):
        print(f"[s1] sub-{sid} fit={len(fit_i)} val_b={len(val_i)} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        for ep in range(args.epochs):
            model.train()
            order = np.random.permutation(fit_i)
            run, nstep = 0.0, 0
            for s in range(0, len(order), args.batch_size):
                ix = order[s : s + args.batch_size]
                zb = torch.from_numpy(ztr[ix]).to(dev)
                cb = torch.from_numpy(cid_tr[ix].astype(np.int64)).to(dev)
                vb = torch.from_numpy(np.asarray(vtr[ix], dtype=np.float32)).to(dev)
                vb = (vb - torch.from_numpy(v_mean).to(dev)) / torch.from_numpy(v_std).to(dev)
                db = torch.from_numpy(np.asarray(dtr[ix], dtype=np.float32)).to(dev)
                opt.zero_grad(set_to_none=True)
                with autocast(enabled=dev.type == "cuda"):
                    o = model(zb)
                    loss = (
                        args.w_nce_txt * gallery_nce(o["z_sem"], g_txt_t, cb, args.tau)
                        + args.w_nce_img * gallery_nce(o["z_sem"], g_img_t, cb, args.tau)
                        + args.w_mse * F.mse_loss(l2t(o["z_sem"]), g_img_t[cb])  # absolute anchor
                        + args.w_depth * F.l1_loss(o["depth"], db)
                        + args.w_vae * F.l1_loss(o["vae"], vb)
                        + args.w_recon * F.mse_loss(l2t(o["z_hat"]), zb)
                    )
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                run += float(loss.detach().cpu())
                nstep += 1

            model.eval()
            with torch.no_grad():
                zv = torch.from_numpy(ztr[val_i]).to(dev)
                cv = torch.from_numpy(cid_tr[val_i].astype(np.int64)).to(dev)
                ov = model(zv)
                t1_txt = retrieval_top1(ov["z_sem"], g_txt_t, cv)
                t1_img = retrieval_top1(ov["z_sem"], g_img_t, cv)
                dv = torch.from_numpy(np.asarray(dtr[val_i], dtype=np.float32)).to(dev)
                a = ov["depth"].reshape(len(val_i), -1)
                b = dv.reshape(len(val_i), -1)
                a = a - a.mean(1, keepdim=True)
                b = b - b.mean(1, keepdim=True)
                dpear = float(((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp(min=1e-8)).mean().cpu())
                # selection: semantic top1 + depth pearson (both held-in)
                score = 0.5 * (t1_txt + t1_img) + 0.3 * max(dpear, 0.0)
            row = {
                "epoch": ep, "loss": run / max(nstep, 1),
                "val_text_top1": t1_txt, "val_img_top1": t1_img,
                "val_depth_pearson": dpear, "score": score,
            }
            history.append(row)
            print(f"[s1] ep={ep:02d} loss={row['loss']:.4f} txt={t1_txt:.3f} img={t1_img:.3f} dpear={dpear:.3f} score={score:.3f}")
            if score > best["score"]:
                best = dict(row)
                torch.save({
                    "model": model.state_dict(),
                    "v_mean": v_mean, "v_std": v_std,
                    "z_dim": ztr.shape[1], "epoch": ep, "score": score,
                    "args": vars(args),
                }, ckpt)

        (out / "s1_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (out / "s1_report.json").write_text(json.dumps({
            "best": best, "n_params": sum(p.numel() for p in model.parameters()),
            "fit": len(fit_i), "val_b": len(val_i), "subject": sid,
            "note": "task-factorized: NCE+MSE semantic, L1 VAE/depth spatial, z-recon anti-collapse",
        }, indent=2), encoding="utf-8")

    # ---- export test ----
    if not ckpt.is_file():
        raise SystemExit(f"[FATAL] missing checkpoint {ckpt}")
    blob = torch.load(ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(blob["model"])
    v_mean = blob["v_mean"]
    v_std = blob["v_std"]
    model.eval()
    with torch.no_grad():
        ot = model(torch.from_numpy(zte).to(dev))
        z_sem = l2t(ot["z_sem"]).cpu().numpy().astype(np.float32)
        depth = ot["depth"].cpu().numpy().astype(np.float32)
        vae = ot["vae"].cpu().numpy().astype(np.float32)
        vae = vae * v_std + v_mean  # de-normalize to scaled latents
    np.save(out / "conds" / "z_sem_test.npy", z_sem)
    np.save(out / "spatial" / "pred_depth_test_64.npy", depth)
    np.save(out / "spatial" / "pred_vae_test_scaled.npy", vae)

    # depth RGB for ControlNet
    from PIL import Image
    ddir = out / "spatial" / "pred_depth_rgb_512"
    ddir.mkdir(parents=True, exist_ok=True)
    for i in range(len(depth)):
        d = depth[i]
        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
        rgb = (np.stack([d, d, d], -1) * 255).astype(np.uint8)
        Image.fromarray(rgb).resize((512, 512), Image.Resampling.BICUBIC).save(ddir / f"{i:03d}.png")

    if args.decode_rgb:
        try:
            sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))
            from train_eeg_vae_head import decode_latents, resolve_vae  # type: ignore
            hub = Path(__import__("os").environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
            vae_m = resolve_vae(hub, dev)
            rgb_dir = out / "spatial" / "pred_lowlevel_rgb_512"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            for s in range(0, len(vae), 8):
                chunk_t = torch.from_numpy(vae[s:s + 8]).to(dev)
                imgs = decode_latents(vae_m, chunk_t, args.scaling_factor)
                for j, im in enumerate(imgs):
                    im.save(rgb_dir / f"{s + j:03d}.png")
            del vae_m
            print(f"[s1] decoded lowlevel RGB -> {rgb_dir}")
        except Exception as e:
            print(f"[WARN] VAE decode failed: {e}")

    print(f"[s1] exported z_sem {z_sem.shape} depth {depth.shape} vae {vae.shape}")


if __name__ == "__main__":
    main()
