#!/usr/bin/env python3
"""UCK -- Unified Concept Kernel.

One neural identity, two views of the same query, one spatial field.

    z = intra shared_r                         (already leak-free)
    q = MLP(z)                                 (ONE query)
    IP = mem(q, G_img)                         (train-concept image gallery)
    F  = Up(z)                                 (shared 64x64)
    depth = sigmoid(W_d F)                     (ControlNet)
    vae   = W_v F                              (img2img init)

Hard constraints this file will not violate:
  * never regress test-photo encode_image() (UGE measured that below the centreline)
  * never fuse the condition toward CLIP-text (l_frow / fuse_in)
  * gallery is the 1654 TRAIN concepts, asserted disjoint from the 200 test ones
  * checkpoint selection is held-in train concepts only (leakfree val_b)
  * --noise-arm silences z at train, val and export through ONE function
  * disk: test-side exports only; no last.pth, no activations, no train preds
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

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocf_train import GENERIC_PROMPT, build_concept_bank, l2n, l2t  # noqa: E402
import leakfree as LF  # noqa: E402


# ---------------------------------------------------------------- helpers
def gallery_nce(q: torch.Tensor, gallery: torch.Tensor, cid: torch.Tensor, tau: float) -> torch.Tensor:
    """InfoNCE of q against a concept gallery. Positive = row's train concept.

    `ocf_train.multi_pos_nce` is in-batch (targ must be Bxd, mask is BxB). Passing
    the 1654-row gallery there is a shape error, not a softer loss.
    """
    return F.cross_entropy((l2t(q) @ l2t(gallery).T) / tau, cid)


def memory(q: torch.Tensor, bank: torch.Tensor, tau: float, k: int = 16) -> torch.Tensor:
    """Differentiable top-k soft retrieval. Same geometry as g2f.

    Full softmax over a flat similarity profile collapses to the bank mean
    (measured: mem_to_ip 0.64 vs a constant 0.61). top-k keeps per-row mass.
    """
    sim = l2t(q) @ l2t(bank).T
    topv, topi = sim.topk(min(k, sim.shape[1]), dim=-1)
    return l2t((torch.softmax(topv / tau, dim=-1).unsqueeze(-1) * l2t(bank)[topi]).sum(1))


def grad_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    px = pred[:, :, 1:] - pred[:, :, :-1]
    py = pred[:, 1:, :] - pred[:, :-1, :]
    gx = gt[:, :, 1:] - gt[:, :, :-1]
    gy = gt[:, 1:, :] - gt[:, :-1, :]
    return F.l1_loss(px, gx) + F.l1_loss(py, gy)


def rowcos(a: np.ndarray) -> float:
    z = l2n(np.asarray(a, dtype=np.float64).reshape(len(a), -1))
    s = z @ z.T
    n = len(z)
    return float((s.sum() - np.trace(s)) / max(n * (n - 1), 1))


def retrieval_top1(q: torch.Tensor, bank: torch.Tensor, cid: torch.Tensor) -> float:
    pred = (l2t(q) @ l2t(bank).T).argmax(-1)
    return float((pred == cid).float().mean())


def depth_to_rgb(depth: np.ndarray):
    from PIL import Image
    d = depth.astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    u8 = (d * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([u8, u8, u8], axis=-1))


def inputs(z: torch.Tensor, noise: bool, scale: torch.Tensor) -> torch.Tensor:
    """THE only place z is read. --noise-arm must go through here."""
    if not noise:
        return z
    return torch.randn_like(z) * scale


# ---------------------------------------------------------------- model
class UCK(nn.Module):
    def __init__(self, z_dim: int, hidden: int = 1024, ip_dim: int = 1024, spatial: bool = True):
        super().__init__()
        self.spatial = spatial
        self.q_net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, ip_dim),
        )
        if spatial:
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

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        q = self.q_net(z)
        out: dict[str, torch.Tensor] = {"q": q}
        if self.spatial:
            field = self.up(self.fc(z).view(-1, 128, 8, 8))
            out["F"] = field
            out["depth"] = torch.sigmoid(self.depth_head(field)).squeeze(1)
            out["vae"] = self.vae_head(field)
        return out


def build_g_img(clip_img: np.ndarray, cid: np.ndarray, n_cls: int) -> np.ndarray:
    g = np.zeros((n_cls, clip_img.shape[1]), dtype=np.float32)
    cnt = np.zeros((n_cls,), dtype=np.int64)
    # clip_img may be mmap; accumulate in chunks
    bs = 2048
    for i in range(0, len(cid), bs):
        sl = slice(i, i + bs)
        x = np.asarray(clip_img[sl], dtype=np.float32)
        x = l2n(x)
        for row, c in zip(x, cid[sl]):
            g[int(c)] += row
            cnt[int(c)] += 1
    cnt = np.clip(cnt, 1, None)
    return l2n(g / cnt[:, None])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--test-subject", type=int, default=8)
    ap.add_argument("--z-root", type=str, default=str(NB_ROOT / "outputs/ocf/intra_z"))
    ap.add_argument("--captions-dir", type=str, default=str(NB_ROOT / "outputs/g2/captions"))
    ap.add_argument("--clip-text-dir", type=str,
                    default=str(NB_ROOT / "outputs/nda_ss/sub-08/clip_text"))
    ap.add_argument("--clip-img-dir", type=str, default=str(NB_ROOT / "outputs/gem/cond_cache"))
    ap.add_argument("--vae-cache", type=str,
                    default=str(NB_ROOT / "outputs/sdedit_ll_full10/shared/vae_cache"))
    ap.add_argument("--depth-train", type=str, default="")
    ap.add_argument("--depth-test", type=str, default="")
    ap.add_argument("--split-json", type=str, default=str(NB_ROOT / "outputs/leakfree/split.json"))
    ap.add_argument("--gallery-cache", type=str, default=str(NB_ROOT / "outputs/uck/shared"))
    ap.add_argument("--views", type=str, default="both", choices=["both", "text", "img"])
    ap.add_argument("--mem-bank", type=str, default="concept", choices=["concept", "image", "both"])
    ap.add_argument("--spatial", type=int, default=1)
    ap.add_argument("--noise-arm", type=int, default=0)
    ap.add_argument("--export-only", type=int, default=0)
    ap.add_argument("--decode-rgb", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--mem-k", type=int, default=16)
    ap.add_argument("--w-text", type=float, default=1.0)
    ap.add_argument("--w-img", type=float, default=1.0)
    ap.add_argument("--w-depth", type=float, default=1.0)
    ap.add_argument("--w-vae", type=float, default=1.0)
    ap.add_argument("--lambda-grad", type=float, default=0.5)
    ap.add_argument("--scaling-factor", type=float, default=0.13025)
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
    (out / "prompts").mkdir(exist_ok=True)
    sid = f"{args.test_subject:02d}"
    noise = bool(args.noise_arm)
    print(f"[uck] sub-{sid} views={args.views} mem={args.mem_bank} "
          f"spatial={args.spatial} noise={int(noise)} out={out}")

    ztr = np.load(Path(args.z_root) / f"sub-{sid}" / "shared_r_train.npy").astype(np.float32)
    zte = np.load(Path(args.z_root) / f"sub-{sid}" / "shared_r_test.npy").astype(np.float32)
    ztr = l2n(ztr)
    zte = l2n(zte)

    cd = Path(args.captions_dir)
    g_text, cid_np, phrases = build_concept_bank(
        Path(args.clip_text_dir), cd / "captions_train.jsonl")
    caps_te = [json.loads(l) for l in (cd / "captions_test.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    if len(cid_np) != len(ztr) or len(caps_te) != len(zte):
        raise SystemExit(f"[FATAL] rows z {len(ztr)}/{len(zte)} vs cap/cid "
                         f"{len(cid_np)}/{len(caps_te)}")

    def concept_of(p: str) -> str:
        return Path(p).parent.name.split("_", 1)[1].replace("_", " ")

    con_te = [concept_of(c["path"]) for c in caps_te]
    inter = {str(p).strip().lower() for p in phrases} & {c.strip().lower() for c in con_te}
    if inter:
        raise SystemExit(f"[FATAL] {len(inter)} test concepts in the train gallery")
    print(f"[audit] gallery {len(phrases)} train concepts, test {len(set(con_te))}, intersection 0")

    gcache = Path(args.gallery_cache)
    gcache.mkdir(parents=True, exist_ok=True)
    g_img_p = gcache / "g_img_concept.npy"
    clip_tr = np.load(Path(args.clip_img_dir) / "clip_img1024_train.npy", mmap_mode="r")
    if clip_tr.shape[0] != len(cid_np) or clip_tr.shape[1] != 1024:
        raise SystemExit(f"[FATAL] clip_img train {clip_tr.shape} vs cid {len(cid_np)}")
    if g_img_p.is_file() and g_img_p.stat().st_size > 1000:
        g_img = l2n(np.load(g_img_p).astype(np.float32))
        if g_img.shape != (len(phrases), 1024):
            g_img = build_g_img(clip_tr, cid_np, len(phrases))
            np.save(g_img_p, g_img)
    else:
        print("[uck] building concept-mean CLIP-image gallery (once)")
        g_img = build_g_img(clip_tr, cid_np, len(phrases))
        np.save(g_img_p, g_img)
        (gcache / "g_img_concept.json").write_text(json.dumps({
            "n": int(len(phrases)), "source": "mean of train encode_image per concept",
            "leak": "train concepts only",
        }, indent=2), encoding="utf-8")
    print(f"[uck] G_text {g_text.shape}  G_img {g_img.shape}")

    dtr = dte = None
    vtr = vte = None
    spatial = bool(args.spatial)
    if spatial:
        dp_tr = Path(args.depth_train) if args.depth_train else Path()
        dp_te = Path(args.depth_test) if args.depth_test else Path()
        if dp_tr.is_file() and dp_te.is_file():
            dtr = np.load(dp_tr, mmap_mode="r")
            dte = np.load(dp_te, mmap_mode="r")
            if len(dtr) != len(ztr) or len(dte) != len(zte):
                print(f"[WARN] depth rows {len(dtr)}/{len(dte)} != z; depth loss off")
                dtr = dte = None
        else:
            print("[WARN] depth npy missing; depth loss off (existing maps used at decode)")
        vp_tr = Path(args.vae_cache) / "train_vae_latents_f16.npy"
        vp_te = Path(args.vae_cache) / "test_vae_latents_f16.npy"
        if vp_tr.is_file() and vp_te.is_file():
            vtr = np.load(vp_tr, mmap_mode="r")
            vte = np.load(vp_te, mmap_mode="r")
            if len(vtr) != len(ztr):
                print(f"[WARN] vae rows {len(vtr)} != z; vae loss off")
                vtr = vte = None
        else:
            print("[WARN] vae cache missing; vae loss off")
        if dtr is None and vtr is None:
            print("[WARN] no spatial target; --spatial forced off")
            spatial = False

    split = LF.load(args.split_json)
    fit_i = LF.rows_for(split, "fit", len(ztr))
    val_i = LF.rows_for(split, "val_b", len(ztr))
    if set(fit_i.tolist()) & set(val_i.tolist()):
        raise SystemExit("[FATAL] fit/val_b overlap")

    G_text = torch.from_numpy(l2n(g_text)).to(dev)
    G_img = torch.from_numpy(l2n(g_img)).to(dev)
    img_bank = None
    if args.mem_bank in ("image", "both"):
        img_bank = torch.from_numpy(l2n(np.asarray(clip_tr, dtype=np.float32))).to(dev)

    z_scale = torch.from_numpy(ztr[fit_i].std(0).astype(np.float32).clip(1e-3)).to(dev)

    v_mean = v_std = None
    if spatial and vtr is not None:
        # cache is already scaled (encode mean * scaling_factor)
        chunk = np.asarray(vtr[fit_i[:: max(1, len(fit_i) // 2048)]], dtype=np.float32)
        v_mean = torch.tensor(chunk.mean(axis=(0, 2, 3)), device=dev, dtype=torch.float32).view(1, 4, 1, 1)
        v_std = torch.tensor(chunk.std(axis=(0, 2, 3)).clip(1e-3), device=dev, dtype=torch.float32).view(1, 4, 1, 1)

    model = UCK(z_dim=ztr.shape[1], spatial=spatial).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[uck] params {n_par/1e6:.2f}M spatial={spatial} fit={len(fit_i)} val={len(val_i)}")

    ckpt_p = out / "best.pth"
    if args.export_only:
        if not ckpt_p.is_file():
            raise SystemExit(f"[FATAL] --export-only needs {ckpt_p}")
        model.load_state_dict(torch.load(ckpt_p, map_location=dev, weights_only=False)["state_dict"])
    elif args.resume and (out / "conds" / "ip_mem_test.npy").is_file() and (out / "report.json").is_file():
        print("[uck] resume: exports exist, skipping train")
        model.load_state_dict(torch.load(ckpt_p, map_location=dev, weights_only=False)["state_dict"])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best = -1e9
        history = []
        bs = args.batch_size
        for ep in range(1, args.epochs + 1):
            model.train()
            order = np.random.permutation(fit_i)
            run = 0.0
            nstep = 0
            for s in range(0, len(order) - bs + 1, bs):
                ix = order[s:s + bs]
                zb = inputs(torch.from_numpy(ztr[ix]).to(dev), noise, z_scale)
                cid = torch.from_numpy(cid_np[ix].astype(np.int64)).to(dev)
                o = model(zb)
                loss = zb.new_zeros(())
                if args.views in ("both", "text"):
                    loss = loss + args.w_text * gallery_nce(o["q"], G_text, cid, args.tau)
                if args.views in ("both", "img"):
                    loss = loss + args.w_img * gallery_nce(o["q"], G_img, cid, args.tau)
                if spatial:
                    if dtr is not None:
                        db = torch.from_numpy(np.asarray(dtr[ix], dtype=np.float32)).to(dev)
                        loss = loss + args.w_depth * (F.l1_loss(o["depth"], db)
                                                      + args.lambda_grad * grad_loss(o["depth"], db))
                    if vtr is not None and v_mean is not None:
                        vb = torch.from_numpy(np.asarray(vtr[ix], dtype=np.float32)).to(dev)
                        vn = (vb - v_mean) / v_std
                        loss = loss + args.w_vae * F.l1_loss(o["vae"], vn)
                if not torch.isfinite(loss):
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                run += float(loss.item())
                nstep += 1

            model.eval()
            with torch.no_grad():
                zv = inputs(torch.from_numpy(ztr[val_i]).to(dev), noise, z_scale)
                ov = model(zv)
                cv = torch.from_numpy(cid_np[val_i].astype(np.int64)).to(dev)
                t1_txt = retrieval_top1(ov["q"], G_text, cv)
                t1_img = retrieval_top1(ov["q"], G_img, cv)
                dpear = 0.0
                if spatial and dtr is not None:
                    dv = torch.from_numpy(np.asarray(dtr[val_i], dtype=np.float32)).to(dev)
                    a = ov["depth"].reshape(len(val_i), -1)
                    b = dv.reshape(len(val_i), -1)
                    a = a - a.mean(1, keepdim=True)
                    b = b - b.mean(1, keepdim=True)
                    dpear = float(((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-8)).mean())
            score = t1_txt + t1_img + 0.1 * dpear
            row = {"epoch": ep, "loss": run / max(nstep, 1), "val_text_top1": t1_txt,
                   "val_img_top1": t1_img, "val_depth_pearson": dpear, "score": score}
            history.append(row)
            print(f"[uck ep {ep}] loss={row['loss']:.4f} txt={t1_txt:.3f} "
                  f"img={t1_img:.3f} depth_r={dpear:.3f}")
            if score > best:
                best = score
                torch.save({"epoch": ep, "state_dict": model.state_dict(),
                            "z_dim": ztr.shape[1], "spatial": spatial,
                            "views": args.views, "metrics": row}, ckpt_p)

        model.load_state_dict(torch.load(ckpt_p, map_location=dev, weights_only=False)["state_dict"])
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # ============================================================ export (test only)
    model.eval()
    with torch.no_grad():
        zt = inputs(torch.from_numpy(zte).to(dev), noise, z_scale)
        ot = model(zt)
        q = ot["q"]
        ip_c = memory(q, G_img, args.tau, args.mem_k)
        ip_q = l2t(q)
        np.save(out / "conds" / "ip_mem_test.npy", ip_c.cpu().numpy().astype(np.float32))
        np.save(out / "conds" / "ip_q_test.npy", ip_q.cpu().numpy().astype(np.float32))
        if img_bank is not None:
            ip_i = memory(q, img_bank, args.tau, args.mem_k)
            np.save(out / "conds" / "ip_mem_image_test.npy", ip_i.cpu().numpy().astype(np.float32))

        # noise twin from THE SAME weights, silenced z -- not a second training run
        if not noise:
            zn = inputs(torch.from_numpy(zte).to(dev), True, z_scale)
            on = model(zn)
            ip_n = memory(on["q"], G_img, args.tau, args.mem_k)
            np.save(out / "conds" / "ip_mem_noise_test.npy", ip_n.cpu().numpy().astype(np.float32))

        sim = (l2t(q) @ l2t(G_text).T)
        topi = sim.topk(5, dim=-1).indices.cpu().numpy()
        five = [", ".join(phrases[int(j)] for j in row) for row in topi]
        (out / "prompts" / "prompts_five.json").write_text(
            json.dumps(five, indent=1), encoding="utf-8")
        (out / "prompts" / "prompts_generic.json").write_text(
            json.dumps([GENERIC_PROMPT] * len(zte), indent=1), encoding="utf-8")
        (out / "prompts" / "prompts_empty.json").write_text(
            json.dumps([""] * len(zte), indent=1), encoding="utf-8")

        spat = out / "spatial"
        if spatial and "depth" in ot and dtr is not None:
            spat.mkdir(exist_ok=True)
            depth = ot["depth"].cpu().numpy().astype(np.float32)
            np.save(spat / "pred_depth_test_64.npy", depth)
            rgb_d = spat / "pred_depth_rgb_512"
            rgb_d.mkdir(exist_ok=True)
            for i, d in enumerate(depth):
                depth_to_rgb(d).resize((512, 512)).save(rgb_d / f"{i:03d}.png")
        if spatial and "vae" in ot and vtr is not None:
            spat.mkdir(exist_ok=True)
            vae_pred = ot["vae"]
            if v_mean is not None:
                vae_pred = vae_pred * v_std + v_mean
            vae_np = vae_pred.cpu().numpy().astype(np.float32)
            # cache is already scaled; ATM consumes this file as-is
            np.save(spat / "pred_vae_test_scaled.npy", vae_np)
            if args.decode_rgb:
                try:
                    from train_eeg_vae_head import decode_latents, resolve_vae
                    import os
                    hub = Path(os.environ.get(
                        "HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
                    vae = resolve_vae(hub, dev)
                    rgb_v = spat / "pred_lowlevel_rgb_512"
                    rgb_v.mkdir(exist_ok=True)
                    for s in range(0, len(vae_np), 8):
                        chunk = torch.from_numpy(vae_np[s:s + 8]).to(dev)
                        imgs = decode_latents(vae, chunk, args.scaling_factor)
                        for j, im in enumerate(imgs):
                            im.save(rgb_v / f"{s + j:03d}.png")
                    del vae
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                except Exception as e:  # noqa: BLE001
                    print(f"[WARN] VAE RGB decode failed: {type(e).__name__}: {e}")

    ip = np.load(out / "conds" / "ip_mem_test.npy")
    rep = {
        "pipeline": "uck",
        "subject": f"sub-{sid}",
        "views": args.views,
        "mem_bank": args.mem_bank,
        "spatial_trained": spatial,
        "noise_arm": noise,
        "n_params": n_par,
        "ip_mem_rowcos": rowcos(ip),
        "ip_mem_std": float(ip.std(0).mean()),
        "gallery": len(phrases),
        "note": "IP is train-gallery soft retrieval, not encode_image regression",
    }
    if (out / "conds" / "ip_mem_noise_test.npy").is_file() and not noise:
        a, b = l2n(ip), l2n(np.load(out / "conds" / "ip_mem_noise_test.npy"))
        rep["full_vs_noise_rowcos"] = float((a * b).sum(1).mean())
    (out / "report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    # never keep a last.pth; best.pth is the only weight file
    last = out / "last.pth"
    if last.is_file():
        last.unlink()


if __name__ == "__main__":
    main()
