#!/usr/bin/env python3
"""NAT -- Neural Address then Translate.

Same identity as UCK (intra shared_r), two simultaneous readouts:

    z  = intra shared_r                         (frozen, leak-free)
    μ  = fit-only concept means of z            (neural concept book)
    α  = top-k softmax(z · μ / τ)               (address in neural space)
    IP = α @ G_img                              (CLIP is a dictionary)
    F  = Up(z)                                  (structure from the SAME z)

Hard constraints:
  * never project z into CLIP before addressing (that is UCK's q_net)
  * never regress encode_image() of the test photo
  * never fuse the condition toward CLIP-text
  * gallery / book are the 1654 TRAIN concepts, disjoint from the 200 test ones
  * residual r = z - αμ is a diagnostic readout, not the default structure path
  * --noise-arm silences z for BOTH IP and F through ONE function
  * disk: test-side exports only; no last.pth, no train preds, no raw EEG
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
from uck_train import grad_loss, rowcos  # noqa: E402


def inputs(z: torch.Tensor, noise: bool, scale: torch.Tensor) -> torch.Tensor:
    if not noise:
        return z
    return torch.randn_like(z) * scale


def concept_means(z: np.ndarray, cid: np.ndarray, n_cls: int, rows: np.ndarray):
    mu = np.zeros((n_cls, z.shape[1]), dtype=np.float64)
    cnt = np.zeros((n_cls,), dtype=np.int64)
    for i in rows:
        mu[int(cid[i])] += z[i]
        cnt[int(cid[i])] += 1
    ok = cnt > 0
    mu[ok] /= cnt[ok, None]
    return l2n(mu.astype(np.float32)), cnt


def loo_top1(z: np.ndarray, cid: np.ndarray, n_cls: int,
             rows: np.ndarray | None = None) -> dict:
    """Leave-one-trial-out n_cls-way top1 in neural space. No CLIP.

    Prototypes are built from all rows; accuracy is reported on `rows`
    (default: all). A scored trial is never in its own prototype.
    """
    z = l2n(np.asarray(z, dtype=np.float32))
    cid = np.asarray(cid, dtype=np.int64)
    n, d = z.shape
    sums = np.zeros((n_cls, d), dtype=np.float64)
    cnt = np.zeros((n_cls,), dtype=np.int64)
    for i in range(n):
        sums[cid[i]] += z[i]
        cnt[cid[i]] += 1
    proto = sums / np.clip(cnt, 1, None)[:, None]
    proto = l2n(proto.astype(np.float32))
    sim = z @ proto.T
    own_sum = sums[cid]
    own_cnt = cnt[cid].astype(np.float64)[:, None]
    p = (own_sum - z.astype(np.float64)) / np.clip(own_cnt - 1.0, 1.0, None)
    p = l2n(p.astype(np.float32))
    sim[np.arange(n), cid] = (z * p).sum(1)
    singleton = cnt[cid] <= 1
    sim[np.arange(n)[singleton], cid[singleton]] = -1e9
    pred = sim.argmax(1)
    hit = pred == cid
    if rows is None:
        rows = np.arange(n)
    rows = np.asarray(rows, dtype=np.int64)
    return {
        "n": int(len(rows)),
        "n_cls": int(n_cls),
        "top1": float(hit[rows].mean()),
        "chance": 1.0 / max(n_cls, 1),
    }


def translate(z: torch.Tensor, mu: torch.Tensor, g_img: torch.Tensor,
              tau: float, k: int):
    """Address in neural space, translate the top-k mass into CLIP-image."""
    sim = l2t(z) @ l2t(mu).T
    kk = min(k, sim.shape[1])
    topv, topi = sim.topk(kk, dim=-1)
    w = torch.softmax(topv / tau, dim=-1)
    ip = l2t((w.unsqueeze(-1) * l2t(g_img)[topi]).sum(1))
    proto = (w.unsqueeze(-1) * l2t(mu)[topi]).sum(1)
    return ip, proto, topi, w


class SpatialHead(nn.Module):
    """Same 64x64 field as UCK. Depth only -- VAE branch is dropped."""

    def __init__(self, z_dim: int, hidden: int = 1024):
        super().__init__()
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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        field = self.up(self.fc(z).view(-1, 128, 8, 8))
        return torch.sigmoid(self.depth_head(field)).squeeze(1)


class NAT(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.full = SpatialHead(z_dim)
        self.res = SpatialHead(z_dim)

    def forward(self, z: torch.Tensor, mu: torch.Tensor, g_img: torch.Tensor,
                tau: float, k: int) -> dict[str, torch.Tensor]:
        ip, proto, topi, w = translate(z, mu, g_img, tau, k)
        r = l2t(z - proto.detach())
        ip_res, _, _, _ = translate(r, mu, g_img, tau, k)
        return {
            "ip": ip,
            "ip_res": ip_res,
            "proto": proto,
            "r": r,
            "depth": self.full(z),
            "depth_res": self.res(r),
            "topi": topi,
            "w": w,
        }


def depth_to_rgb(depth: np.ndarray):
    from PIL import Image
    d = depth.astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    u8 = (d * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([u8, u8, u8], axis=-1))


def save_depth_rgb(depth: np.ndarray, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for i, d in enumerate(depth):
        depth_to_rgb(d).resize((512, 512)).save(dst / f"{i:03d}.png")


def depth_pearson(pred: torch.Tensor, gt: torch.Tensor) -> float:
    a = pred.reshape(pred.shape[0], -1)
    b = gt.reshape(gt.shape[0], -1)
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    return float(((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-8)).mean())


def retrieval_top1(z: torch.Tensor, bank: torch.Tensor, cid: torch.Tensor) -> float:
    pred = (l2t(z) @ l2t(bank).T).argmax(-1)
    return float((pred == cid).float().mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--test-subject", type=int, default=8)
    ap.add_argument("--z-root", type=str, default=str(NB_ROOT / "outputs/ocf/intra_z"))
    ap.add_argument("--captions-dir", type=str, default=str(NB_ROOT / "outputs/g2/captions"))
    ap.add_argument("--clip-text-dir", type=str,
                    default=str(NB_ROOT / "outputs/nda_ss/sub-08/clip_text"))
    ap.add_argument("--clip-img-dir", type=str, default=str(NB_ROOT / "outputs/gem/cond_cache"))
    ap.add_argument("--depth-train", type=str, default="")
    ap.add_argument("--depth-test", type=str, default="")
    ap.add_argument("--split-json", type=str, default=str(NB_ROOT / "outputs/leakfree/split.json"))
    ap.add_argument("--gallery-cache", type=str, default=str(NB_ROOT / "outputs/uck/shared"))
    ap.add_argument("--spatial", type=int, default=1)
    ap.add_argument("--noise-arm", type=int, default=0)
    ap.add_argument("--export-only", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--mem-k", type=int, default=16)
    ap.add_argument("--w-depth", type=float, default=1.0)
    ap.add_argument("--w-res", type=float, default=1.0)
    ap.add_argument("--lambda-grad", type=float, default=0.5)
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
    (out / "proto").mkdir(exist_ok=True)
    sid = f"{args.test_subject:02d}"
    noise = bool(args.noise_arm)
    print(f"[nat] sub-{sid} spatial={args.spatial} noise={int(noise)} out={out}")

    ztr = l2n(np.load(Path(args.z_root) / f"sub-{sid}" / "shared_r_train.npy").astype(np.float32))
    zte = l2n(np.load(Path(args.z_root) / f"sub-{sid}" / "shared_r_test.npy").astype(np.float32))

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
    print(f"[audit] book {len(phrases)} train concepts, test {len(set(con_te))}, intersection 0")

    gcache = Path(args.gallery_cache)
    g_img_p = gcache / "g_img_concept.npy"
    if not (g_img_p.is_file() and g_img_p.stat().st_size > 1000):
        raise SystemExit(f"[FATAL] missing CLIP dictionary {g_img_p}")
    g_img = l2n(np.load(g_img_p).astype(np.float32))
    if g_img.shape != (len(phrases), 1024):
        raise SystemExit(f"[FATAL] G_img {g_img.shape} vs phrases {len(phrases)}")

    split = LF.load(args.split_json)
    fit_i = LF.rows_for(split, "fit", len(ztr))
    val_i = LF.rows_for(split, "val_b", len(ztr))
    if set(fit_i.tolist()) & set(val_i.tolist()):
        raise SystemExit("[FATAL] fit/val_b overlap")

    # leakfree is concept-disjoint: fit has 1489, val_a+val_b hold out 165.
    # Empty fit prototypes are expected, not a data error. Test-time address
    # uses ALL 1654 train EEG means (test 200 are still disjoint). Val
    # metrics never use a prototype that contains the scored trial (LOO).
    n_cls = len(phrases)
    mu_fit, cnt_fit = concept_means(ztr, cid_np, n_cls, fit_i)
    mu_all, cnt_all = concept_means(ztr, cid_np, n_cls, np.arange(len(ztr)))
    n_empty_fit = int((cnt_fit == 0).sum())
    n_empty_all = int((cnt_all == 0).sum())
    print(f"[nat] fit book {int((cnt_fit > 0).sum())}/{n_cls} "
          f"(empty={n_empty_fit}, expected val_a+val_b=165)  "
          f"all-train empty={n_empty_all}")
    if n_empty_all > 0:
        raise SystemExit(f"[FATAL] {n_empty_all} train concepts have no EEG at all")
    if n_empty_fit not in (0, 165):
        print(f"[WARN] empty fit prototypes {n_empty_fit} != 165; check split")
    np.save(out / "proto" / "mu_fit.npy", mu_fit)
    np.save(out / "proto" / "mu_all.npy", mu_all)
    np.save(out / "proto" / "cnt_fit.npy", cnt_fit.astype(np.int64))

    loo = loo_top1(ztr, cid_np, n_cls)
    print(f"[nat] LOO 1654-way top1={loo['top1']:.4f} chance={loo['chance']:.6f}")
    mu_np = mu_all

    dtr = dte = None
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
            print("[WARN] depth npy missing; --spatial forced off")
            dtr = dte = None
        if dtr is None:
            spatial = False

    G_img = torch.from_numpy(g_img).to(dev)
    G_text = torch.from_numpy(l2n(g_text)).to(dev)
    mu = torch.from_numpy(mu_np).to(dev)
    z_scale = torch.from_numpy(ztr[fit_i].std(0).astype(np.float32).clip(1e-3)).to(dev)

    model = NAT(z_dim=ztr.shape[1]).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[nat] params {n_par/1e6:.2f}M spatial={spatial} fit={len(fit_i)} val={len(val_i)}")

    ckpt_p = out / "best.pth"
    if args.export_only:
        if not ckpt_p.is_file():
            raise SystemExit(f"[FATAL] --export-only needs {ckpt_p}")
        model.load_state_dict(torch.load(ckpt_p, map_location=dev, weights_only=False)["state_dict"])
    elif args.resume and (out / "conds" / "ip_nat_test.npy").is_file() and (out / "report.json").is_file():
        print("[nat] resume: exports exist, skipping train")
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
                o = model(zb, mu, G_img, args.tau, args.mem_k)
                loss = zb.new_zeros(())
                if spatial and dtr is not None:
                    db = torch.from_numpy(np.asarray(dtr[ix], dtype=np.float32)).to(dev)
                    loss = loss + args.w_depth * (
                        F.l1_loss(o["depth"], db) + args.lambda_grad * grad_loss(o["depth"], db))
                    loss = loss + args.w_res * (
                        F.l1_loss(o["depth_res"], db) + args.lambda_grad * grad_loss(o["depth_res"], db))
                if not torch.isfinite(loss):
                    continue
                if float(loss.item()) == 0.0 and not spatial:
                    # non-parametric address; nothing to train
                    break
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                run += float(loss.item())
                nstep += 1

            model.eval()
            with torch.no_grad():
                zv = inputs(torch.from_numpy(ztr[val_i]).to(dev), noise, z_scale)
                cv = torch.from_numpy(cid_np[val_i].astype(np.int64)).to(dev)
                t1_neu = retrieval_top1(zv, mu, cv)
                t1_clip = retrieval_top1(zv, G_img, cv)
                t1_txt = retrieval_top1(zv, G_text, cv)
                dpear = dpear_r = 0.0
                if spatial and dtr is not None:
                    ov = model(zv, mu, G_img, args.tau, args.mem_k)
                    dv = torch.from_numpy(np.asarray(dtr[val_i], dtype=np.float32)).to(dev)
                    dpear = depth_pearson(ov["depth"], dv)
                    dpear_r = depth_pearson(ov["depth_res"], dv)
            score = dpear
            row = {"epoch": ep, "loss": run / max(nstep, 1),
                   "val_neural_top1": t1_neu, "val_clipimg_top1": t1_clip,
                   "val_cliptext_top1": t1_txt, "val_depth_pearson": dpear,
                   "val_depth_res_pearson": dpear_r, "score": score}
            history.append(row)
            print(f"[nat ep {ep}] loss={row['loss']:.4f} neu={t1_neu:.3f} "
                  f"clip={t1_clip:.3f} depth_r={dpear:.3f} res_r={dpear_r:.3f}")
            if score > best or (not spatial and ep == 1):
                best = score
                torch.save({"epoch": ep, "state_dict": model.state_dict(),
                            "z_dim": ztr.shape[1], "metrics": row}, ckpt_p)
            if not spatial:
                break

        model.load_state_dict(torch.load(ckpt_p, map_location=dev, weights_only=False)["state_dict"])
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # ============================================================ export (test only)
    model.eval()
    with torch.no_grad():
        zt = inputs(torch.from_numpy(zte).to(dev), noise, z_scale)
        ot = model(zt, mu, G_img, args.tau, args.mem_k)
        np.save(out / "conds" / "ip_nat_test.npy", ot["ip"].cpu().numpy().astype(np.float32))
        np.save(out / "conds" / "ip_res_test.npy", ot["ip_res"].cpu().numpy().astype(np.float32))

        if not noise:
            zn = inputs(torch.from_numpy(zte).to(dev), True, z_scale)
            on = model(zn, mu, G_img, args.tau, args.mem_k)
            np.save(out / "conds" / "ip_noise_test.npy", on["ip"].cpu().numpy().astype(np.float32))
        else:
            on = None

        zv = torch.from_numpy(ztr[val_i]).to(dev)
        cv = torch.from_numpy(cid_np[val_i].astype(np.int64)).to(dev)
        # raw z vs CLIP gallery: should stay near 0 (wrong space)
        val_clip = retrieval_top1(zv, G_img, cv)
        loo_val = loo_top1(ztr, cid_np, n_cls, rows=val_i)
        val_cids = np.unique(cid_np[val_i])
        remap = {int(c): i for i, c in enumerate(val_cids.tolist())}
        cid82 = np.asarray([remap[int(c)] for c in cid_np[val_i]], dtype=np.int64)
        loo_val82 = loo_top1(ztr[val_i], cid82, len(val_cids))
        val_neu = loo_val["top1"]
        val82 = loo_val82["top1"]

        spat = out / "spatial"
        if spatial:
            spat.mkdir(exist_ok=True)
            depth = ot["depth"].cpu().numpy().astype(np.float32)
            np.save(spat / "pred_depth_test_64.npy", depth)
            save_depth_rgb(depth, spat / "pred_depth_rgb_512")
            depth_r = ot["depth_res"].cpu().numpy().astype(np.float32)
            np.save(spat / "pred_depth_res_64.npy", depth_r)
            save_depth_rgb(depth_r, spat / "pred_depth_res_rgb_512")
            if on is not None:
                depth_n = on["depth"].cpu().numpy().astype(np.float32)
                np.save(spat / "pred_depth_noise_64.npy", depth_n)
                save_depth_rgb(depth_n, spat / "pred_depth_noise_rgb_512")

        (out / "prompts" / "prompts_generic.json").write_text(
            json.dumps([GENERIC_PROMPT] * len(zte), indent=1), encoding="utf-8")
        (out / "prompts" / "prompts_empty.json").write_text(
            json.dumps([""] * len(zte), indent=1), encoding="utf-8")

    ip = np.load(out / "conds" / "ip_nat_test.npy")
    ip_r = np.load(out / "conds" / "ip_res_test.npy")
    clip_te = np.load(Path(args.clip_img_dir) / "clip_img1024_test.npy").astype(np.float32)
    clip_te = l2n(clip_te)
    ip_n = l2n(ip)
    vs_true = float((ip_n * clip_te).sum(1).mean()) if len(ip) == len(clip_te) else float("nan")
    vs_true_r = float((l2n(ip_r) * clip_te).sum(1).mean()) if len(ip_r) == len(clip_te) else float("nan")
    centre = float((l2n(clip_te.mean(0, keepdims=True)) * clip_te).sum(1).mean())

    with torch.no_grad():
        # nearest-train-concept CLIP-image vs the true test photo (not 200-way acc)
        topi = ot["topi"][:, 0].cpu().numpy()
        nn_g = g_img[topi]
        nn_vs_true = float((l2n(nn_g) * clip_te).sum(1).mean()) if len(nn_g) == len(clip_te) else float("nan")

    rep = {
        "pipeline": "nat",
        "subject": f"sub-{sid}",
        "spatial_trained": spatial,
        "noise_arm": noise,
        "n_params": n_par,
        "book": len(phrases),
        "loo_train_1654": loo,
        "val_b_loo_1654": loo_val,
        "val_b_loo_82": loo_val82,
        "val_b_neural_1654_top1": val_neu,
        "val_b_neural_82_top1": val82,
        "val_b_clipimg_1654_top1": val_clip,
        "ip_nat_rowcos": rowcos(ip),
        "ip_nat_vs_true": vs_true,
        "ip_res_vs_true": vs_true_r,
        "centreline": centre,
        "nn_train_concept_vs_true": nn_vs_true,
        "note": "IP is neural-address translated through G_img, not MLP(z) and not encode_image",
    }
    if (out / "conds" / "ip_noise_test.npy").is_file() and not noise:
        a, b = l2n(ip), l2n(np.load(out / "conds" / "ip_noise_test.npy"))
        rep["full_vs_noise_rowcos"] = float((a * b).sum(1).mean())
    (out / "report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    last = out / "last.pth"
    if last.is_file():
        last.unlink()


if __name__ == "__main__":
    main()
