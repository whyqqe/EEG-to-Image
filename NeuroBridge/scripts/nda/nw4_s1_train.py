"""NW4 Stage 1 — task-factorized encoder with OPEN-SET, TRIAL-LEVEL semantics.

What changed vs nw3_s1_train, and the measurement that forced each change
-----------------------------------------------------------------------
nw3 aligned `z_sem` to the concept-name text bank with a 1654-way NCE.  Measured
fact (nw4_preflight_facts / nw4_diag_attr):

  * the concept-name bank has within-concept cos == 1.0000 EXACTLY -- all 10 reps
    of a concept collapse to one vector, so the target carries ZERO within-concept
    variance.  EEG fluctuates trial to trial and the target does not, so NCE can
    only teach "which of 1654 buckets", never a transferable function.
  * open-set attribute captions (overall/subject/detail) have within-concept
    cos ~= 0.62-0.64 with 16137/16540 DISTINCT strings -> real per-trial target.
  * closed-form ridge probe on unseen concepts: attr_all 0.0750 vs concept_name
    0.0150 top-1 => attributes transfer 5x better.
  * `background` is NOT the class-orthogonal signal it looks like: its
    within-concept cos (0.475) EXCEEDS its between-concept cos (0.626) -- it
    cannot separate concepts at all.  Dropped by default.

So this stage uses:
  * INSTANCE-level NCE (positive = the row's own trial target, not a bucket id),
  * three heads onto three measured-good spaces:
        img  : clip_img1024      (1024)  <- the space IP-Adapter consumes
        vith : vith_cat5         (5120)  <- strongest target (multi-level)
        attr : attr_all          (4096)  <- open-set attributes (overall+subject+detail)
  * an MSE anchor pinning `head_img` onto the real clip_img manifold,
  * the spatial stream kept from nw3 (depth + VAE + EEG reconstruction).

Everything is leak-free: gradients on `fit`, selection on `val_b`, test export only.
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

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))

# levels used for the multi-level (vith_cat5) target; order fixes the concat layout
VITH_LEVELS = ["image", "GaussianBlur", "LowResolution", "Mosaic", "GaussianNoise"]
ATTR_FIELDS = ["overall", "subject", "detail"]  # 'background' excluded: not separable


def l2n(x: np.ndarray) -> np.ndarray:
    """L2-normalise, always returning float32.

    The explicit cast matters: these arrays are fed straight into float32 torch
    modules, and torch raises "mat1 and mat2 must have the same dtype" instead of
    promoting a float64 input -- the failure mode that killed the first nw4 run.
    """
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-8)
    return (x / n).astype(np.float32)


def l2t(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


class FactorizedEncoder(nn.Module):
    """Semantic stream (3 heads) + spatial stream (depth + VAE) + z-reconstructor."""

    def __init__(self, z_dim: int = 1024, hidden: int = 1024,
                 dim_img: int = 1024, dim_vith: int = 5120, dim_attr: int = 3072):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.head_img = nn.Linear(hidden, dim_img)
        self.head_vith = nn.Linear(hidden, dim_vith)
        self.head_attr = nn.Linear(hidden, dim_attr)
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
        self.recon = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, hidden), nn.GELU(),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(z)
        field = self.up(self.fc(z).view(-1, 128, 8, 8))
        return {
            "z_img": self.head_img(h),
            "z_vith": self.head_vith(h),
            "z_attr": self.head_attr(h),
            "F": field,
            "depth": torch.sigmoid(self.depth_head(field)).squeeze(1),
            "vae": self.vae_head(field),
            "z_hat": self.recon(field),
        }


def inst_nce(q: torch.Tensor, tgt_all: torch.Tensor, idx: torch.Tensor, tau: float) -> torch.Tensor:
    """In-batch InfoNCE with the row's OWN trial target as the positive.

    This is the core of A0/A1.  The positive is `tgt_all[idx[i]]`, i.e. the target
    of the very trial the EEG came from -- NOT a concept bucket id.  Because the
    target bank varies within a concept (verified: cos 0.74 not 1.00), the loss
    forces a real regression, not a 1654-way classification.
    """
    qn = l2t(q)
    t = tgt_all[idx]                       # (B, D) already L2-normalised
    logits = (qn @ t.T) / tau
    labels = torch.arange(len(qn), device=qn.device)
    return F.cross_entropy(logits, labels)


def concept_gallery(rows: np.ndarray, cid: np.ndarray, n_cls: int) -> np.ndarray:
    """L2-normalised concept means from per-row features."""
    d = rows.shape[1]
    acc = np.zeros((n_cls, d), dtype=np.float64)
    cnt = np.bincount(cid, minlength=n_cls).astype(np.float64)
    np.add.at(acc, cid, rows.astype(np.float64))
    return l2n((acc / np.maximum(cnt, 1)[:, None]).astype(np.float32))


def batch_rsa(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """RSA within a batch: corr( cos(z_i,z_j), cos(T_i,T_j) ) over all i < j.

    THE AXIS NOTHING WAS OPTIMISING.  Scoring every condition we have a generation
    score for (`nw4_diag_rsa.py`) gives

        corr(RSA,  inception) = +0.967
        corr(vs_cl, inception) = +0.687

    yet every term in this loss optimises something else.  `inst_nce` is satisfied once
    the single best neighbour is right and is blind to whether row i is *more* similar
    to j than to k; the anchor pins each row to its own target and likewise says nothing
    about the geometry BETWEEN rows; `w_cpt` collapses to concept means, which is the
    degenerate structure that made the closed-set design fail.  So the encoder was free
    to scramble the pairwise ordering, and it did: the raw bank scored RSA 0.136 against
    a real-bank reference of 1.000, i.e. barely above a constant's 0.

    This term makes that ordering the objective.  Cosine is scale-invariant, so the
    term depends only on the shape of the two similarity trees and cannot be gamed by
    inflating the embedding norm.  Excluding the diagonal matters: self-similarity is a
    constant 1 and would otherwise inflate the correlation for free.
    """
    z = F.normalize(z, dim=-1)
    t = F.normalize(t, dim=-1)
    b = z.shape[0]
    if b < 4:
        return z.new_zeros(())
    cz, ct = z @ z.t(), t @ t.t()
    m = torch.triu(torch.ones(b, b, dtype=torch.bool, device=z.device), diagonal=1)
    a, c = cz[m], ct[m]
    a = a - a.mean()
    c = c - c.mean()
    return -(a @ c) / (a.norm() * c.norm()).clamp_min(1e-8)


def np_rsa(z: np.ndarray, t: np.ndarray) -> float:
    """The same statistic in numpy, for checkpoint selection."""
    zn, tn = l2n(z), l2n(t)
    cz, ct = zn @ zn.T, tn @ tn.T
    iu = np.triu_indices(len(cz), k=1)
    a, b = cz[iu], ct[iu]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def mini_metrics(q: np.ndarray, gallery: np.ndarray, cid: np.ndarray, k: int = 10,
                 true_tgt: np.ndarray | None = None) -> dict:
    """Concept retrieval on a small (val_b) gallery, with CSLS to see through hubs."""
    qn, gn = l2n(q), l2n(gallery)
    sim = qn @ gn.T
    top1 = float((sim.argmax(1) == cid).mean())
    # CSLS: subtract each row's mean similarity to its k nearest gallery items.
    # knn_q is (Q, 1) and knn_g is (1, G); their SUM broadcasts to (Q, G), so
    # knn_g must NOT be transposed (transposing broke the (G,1)+(Q,1) add).
    k = max(1, min(k, sim.shape[1] - 1))
    knn_q = np.sort(sim, 1)[:, -k:].mean(1, keepdims=True)
    knn_g = np.sort(sim, 0)[-k:, :].mean(0, keepdims=True)
    csls = sim - 0.5 * (knn_q + knn_g)
    out = {"mini_top1": top1,
           "mini_csls": float((csls.argmax(1) == cid).mean()),
           "mini_cos": float((qn * gn[cid]).sum(1).mean())}
    if true_tgt is not None:
        # the metric the gate and the generator actually respond to
        out["mini_rsa"] = np_rsa(qn, true_tgt)
    return out


def load_vith_cat5(split: str, n_flat: int) -> np.ndarray:
    """concat of the 5 L2-normalised ViT-H-14 levels -> (n_flat, 5120)."""
    vroot = NB_ROOT / "data/things_eeg/image_feature/ViT-H-14"
    blocks = []
    for lv in VITH_LEVELS:
        p = (vroot / f"{lv}_{split}.npy") if lv == "image" else (vroot / lv / f"{split}.npy")
        if not p.is_file():
            raise SystemExit(f"[FATAL] missing vith level {p}")
        d = np.load(p, mmap_mode="r").shape[-1]
        a = np.load(p).astype(np.float32).reshape(-1, d)
        blocks.append(l2n(a))
    return np.concatenate(blocks, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--test-subject", type=int, default=8)
    ap.add_argument("--z-root", type=str, default=str(NB_ROOT / "outputs/ocf/intra_z"))
    ap.add_argument("--cond-cache", type=str, default=str(NB_ROOT / "outputs/gem/cond_cache"))
    ap.add_argument("--g2-targets", type=str, default=str(NB_ROOT / "outputs/g2/targets"))
    ap.add_argument("--clip-text-dir", type=str,
                    default=str(NB_ROOT / "outputs/nda_ss/sub-08/clip_text"))
    ap.add_argument("--vae-cache", type=str,
                    default=str(NB_ROOT / "outputs/sdedit_ll_full10/shared/vae_cache"))
    ap.add_argument("--depth-train", type=str,
                    default=str(NB_ROOT / "outputs/uck/shared/gt_depth/train_depth_64.npy"))
    ap.add_argument("--depth-test", type=str,
                    default=str(NB_ROOT / "outputs/hcma_s_full10/shared/gt_depth/test_depth_64.npy"))
    ap.add_argument("--split-json", type=str,
                    default=str(NB_ROOT / "outputs/leakfree/split.json"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--w-inst-img", type=float, default=1.0)
    ap.add_argument("--w-inst-vith", type=float, default=1.0)
    ap.add_argument("--w-inst-attr", type=float, default=1.0)
    ap.add_argument("--w-cpt", type=float, default=0.0,
                    help="concept-level aux NCE (0 = off; the space is degenerate alone)")
    ap.add_argument("--w-anchor", type=float, default=400.0,
                    help="MSE pinning head_img onto the real clip_img manifold. Must "
                         "dominate the NCE terms: InfoNCE is mathematically invariant to "
                         "a per-row shift along a direction common to all rows, so it "
                         "discards the large common component CLIP embeddings share "
                         "(rowcos 0.628) -- which is most of what IP-Adapter reads. "
                         "Weights 3xNCE (3.0) vs 0.5 let the head drift to vs_true 0.257, "
                         "BELOW the 0.620 a constant vector scores.")
    ap.add_argument("--w-spa", type=float, default=0.0,
                    help="weight on batch_rsa, the structure-preserving term. Off by "
                         "default so existing runs reproduce; the RSA axis is the one "
                         "that predicts generation (corr +0.967), so turning it on is "
                         "the principled way to move the metric that matters.")
    ap.add_argument("--w-depth", type=float, default=1.0)
    ap.add_argument("--w-vae", type=float, default=1.0)
    ap.add_argument("--w-recon", type=float, default=0.2)
    ap.add_argument("--scaling-factor", type=float, default=0.13025)
    ap.add_argument("--select-metric", type=str, default="img_rsa",
                    choices=["img_rsa", "img_vs_cl", "img_mini_csls", "img_mini_top1"],
                    help="checkpoint selector. `img_vs_cl` (default) maximises the net "
                         "cosine to each val row's true trial target minus the constant "
                         "floor, i.e. the quantity that tracks generation quality. "
                         "`img_mini_csls` is the retrieval-based selector the first nw4 "
                         "run used; it is kept for direct comparison and is expected to "
                         "be worse, since retrieval accuracy and vs_true are "
                         "anti-correlated in this regime.")
    ap.add_argument("--decode-rgb", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--allow-cpu", type=int, default=0,
                    help="opt out of the GPU fail-fast guard (slow, for debugging)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--export-only", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # NOT `torch.device(args.device if torch.cuda.is_available() else "cpu")`: the GPU
    # nodes have mixed driver versions, and that expression silently turns a driver
    # mismatch into a 10-20x CPU run (one S1 epoch: 75 s on dgx-13, >10 min on dgx-09
    # without finishing).  See dev_guard.py.
    from dev_guard import pick_device  # noqa: E402
    dev = pick_device(args.device, allow_cpu=bool(args.allow_cpu))
    out = Path(args.out)
    (out / "conds").mkdir(parents=True, exist_ok=True)
    (out / "spatial").mkdir(parents=True, exist_ok=True)
    sid = f"{args.test_subject:02d}"

    import leakfree as LF  # noqa: E402

    ztr = l2n(np.load(Path(args.z_root) / f"sub-{sid}" / "shared_r_train.npy").astype(np.float32))
    zte = l2n(np.load(Path(args.z_root) / f"sub-{sid}" / "shared_r_test.npy").astype(np.float32))
    n_flat = len(ztr)
    cc, g2 = Path(args.cond_cache), Path(args.g2_targets)

    meta = json.loads((Path(args.clip_text_dir) / "train" / "meta.json").read_text(encoding="utf-8"))
    reps = int(meta["n_flat"]) // int(meta["n_concepts"])
    n_cls = int(meta["n_concepts"])
    if n_flat != int(meta["n_flat"]):
        raise SystemExit(f"[FATAL] eeg rows {n_flat} != meta n_flat {meta['n_flat']}")
    cid_tr = np.arange(n_flat, dtype=np.int64) // reps
    print(f"[audit] {n_cls} train concepts x {reps} reps = {n_flat} rows; "
          f"test concepts DISJOINT by protocol")

    # ---- targets: trial-level (n_flat rows), each L2-normalised ----
    t_img = l2n(np.load(cc / "clip_img1024_train.npy").astype(np.float32))
    t_img_te = l2n(np.load(cc / "clip_img1024_test.npy").astype(np.float32))
    t_vith = load_vith_cat5("train", n_flat)
    t_vith_te = load_vith_cat5("test", 200)
    t_attr = np.concatenate(
        [l2n(np.load(g2 / f"sem_{f}_train.npy").astype(np.float32)) for f in ATTR_FIELDS], 1)
    t_attr_te = np.concatenate(
        [l2n(np.load(g2 / f"sem_{f}_test.npy").astype(np.float32)) for f in ATTR_FIELDS], 1)
    for nm, a, b in (("img", t_img, t_img_te), ("vith", t_vith, t_vith_te),
                     ("attr", t_attr, t_attr_te)):
        if a.shape[0] != n_flat or b.shape[0] != 200:
            raise SystemExit(f"[FATAL] {nm} target rows {a.shape[0]}/{b.shape[0]}")
        print(f"[target] {nm:<5} train {a.shape} test {b.shape}")

    vtr = np.load(Path(args.vae_cache) / "train_vae_latents_f16.npy", mmap_mode="r")
    dtr = np.load(args.depth_train, mmap_mode="r")
    if not (len(vtr) == len(ztr) == len(dtr) == n_flat):
        raise SystemExit("[FATAL] spatial target row mismatch")

    split = LF.load(args.split_json)
    fit_i = LF.rows_for(split, "fit", n_flat)
    val_i = LF.rows_for(split, "val_b", n_flat)
    if set(fit_i.tolist()) & set(val_i.tolist()):
        raise SystemExit("[FATAL] fit/val_b overlap")

    chunk = np.asarray(vtr[fit_i[:: max(1, len(fit_i) // 2048)]], dtype=np.float32)
    v_mean = chunk.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    v_std = chunk.std(axis=(0, 2, 3), keepdims=True).astype(np.float32).clip(1e-3)

    T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)  # noqa: E731
    ti_img, ti_vith, ti_attr = T(t_img), T(t_vith), T(t_attr)
    # concept-level gallery for the optional aux loss (train concepts -> cid space)
    cg_img = T(concept_gallery(t_img, cid_tr, n_cls))
    # val_b gallery: concept means over val rows (val concepts are disjoint from fit)
    val_cid = cid_tr[val_i]
    uq = {c: i for i, c in enumerate(sorted(set(val_cid.tolist())))}
    val_target = np.asarray([uq[c] for c in val_cid.tolist()], dtype=np.int64)
    gv_img = T(concept_gallery(t_img[val_i], val_target, len(uq)))
    gv_vith = T(concept_gallery(t_vith[val_i], val_target, len(uq)))
    gv_attr = T(concept_gallery(t_attr[val_i], val_target, len(uq)))

    model = FactorizedEncoder(z_dim=ztr.shape[1], dim_img=t_img.shape[1],
                              dim_vith=t_vith.shape[1], dim_attr=t_attr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=dev.type == "cuda")
    ckpt = out / "best.pth"
    history: list[dict] = []
    best = {"score": -1.0, "epoch": -1}

    if args.export_only:
        if not ckpt.is_file():
            raise SystemExit(f"[FATAL] --export-only needs {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["state_dict"])
    elif args.resume and (out / "conds" / "z_img_test.npy").is_file() and (out / "report.json").is_file():
        print("[nw4-s1] resume: exports exist, skipping train")
        model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["state_dict"])
    else:
        print(f"[nw4-s1] sub-{sid} fit={len(fit_i)} val_b={len(val_i)} "
              f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        for ep in range(args.epochs):
            model.train()
            order = np.random.permutation(fit_i)
            run, nstep = 0.0, 0
            for s in range(0, len(order), args.batch_size):
                ix = order[s : s + args.batch_size]
                if len(ix) < 2:
                    continue
                zb = torch.from_numpy(ztr[ix]).to(dev)
                ib = torch.from_numpy(ix.astype(np.int64)).to(dev)
                vb = torch.from_numpy(np.asarray(vtr[ix], dtype=np.float32)).to(dev)
                vb = (vb - T(v_mean)) / T(v_std)
                db = torch.from_numpy(np.asarray(dtr[ix], dtype=np.float32)).to(dev)
                cb = torch.from_numpy(cid_tr[ix].astype(np.int64)).to(dev)
                opt.zero_grad(set_to_none=True)
                with autocast(enabled=dev.type == "cuda"):
                    o = model(zb)
                    # SCALE WARNING -- read before touching these weights.
                    # `F.mse_loss` defaults to reduction="mean", which divides by the
                    # number of ELEMENTS.  For a per-VECTOR residual over D=1024
                    # dimensions on unit-norm vectors that is (2-2cos)/1024, i.e. the
                    # term is silently 1024x weaker than its weight suggests.  With
                    # w_anchor=20 the effective weight was 0.0195 against three NCE
                    # terms summing to 3.0, so the anchor was inert: raising it from
                    # 0.5 to 20 changed the epoch-0 loss by 0.034 instead of ~39
                    # (predicted (20-0.5)*2/1024 = 0.038, measured 0.034 -- that match
                    # is how the bug was identified).  Consequence: the head never
                    # learned the common component CLIP embeddings share, and the
                    # exported condition scored vs_true 0.257 against a CONSTANT
                    # vector's 0.620.
                    # Cosine distance is used instead: it is O(1) per row, cannot be
                    # rescaled by the ambient dimension, and is exactly the quantity
                    # the downstream gate measures.  L1 on the (B,64,64) / (B,4,64,64)
                    # field outputs is fine -- there the per-element mean IS the
                    # natural scale.
                    loss = (
                        args.w_inst_img * inst_nce(o["z_img"], ti_img, ib, args.tau)
                        + args.w_inst_vith * inst_nce(o["z_vith"], ti_vith, ib, args.tau)
                        + args.w_inst_attr * inst_nce(o["z_attr"], ti_attr, ib, args.tau)
                        + args.w_spa * batch_rsa(o["z_img"], ti_img[ib])
                        + args.w_anchor * (1.0 - F.cosine_similarity(
                            o["z_img"], ti_img[ib], dim=-1)).mean()
                        + args.w_depth * F.l1_loss(o["depth"], db)
                        + args.w_vae * F.l1_loss(o["vae"], vb)
                        + args.w_recon * (1.0 - F.cosine_similarity(
                            o["z_hat"], zb, dim=-1)).mean()
                    )
                    if args.w_cpt > 0:
                        a = (l2t(o["z_img"]) @ cg_img.T) / args.tau
                        loss = loss + args.w_cpt * F.cross_entropy(a, cb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                run += float(loss.detach().cpu())
                nstep += 1

            model.eval()
            with torch.no_grad():
                ov = model(torch.from_numpy(ztr[val_i]).to(dev))
                m_img = mini_metrics(ov["z_img"].float().cpu().numpy(),
                                     gv_img.float().cpu().numpy(), val_target,
                                     true_tgt=ti_img[val_i].float().cpu().numpy())
                m_vith = mini_metrics(ov["z_vith"].float().cpu().numpy(),
                                      gv_vith.float().cpu().numpy(), val_target)
                m_attr = mini_metrics(ov["z_attr"].float().cpu().numpy(),
                                      gv_attr.float().cpu().numpy(), val_target)
                dv = torch.from_numpy(np.asarray(dtr[val_i], dtype=np.float32)).to(dev)
                a2 = ov["depth"].reshape(len(val_i), -1)
                b2 = dv.reshape(len(val_i), -1)
                a2 = a2 - a2.mean(1, keepdim=True)
                b2 = b2 - b2.mean(1, keepdim=True)
                dpear = float(((a2 * b2).sum(1) /
                               (a2.norm(dim=1) * b2.norm(dim=1)).clamp(min=1e-8)).mean().cpu())
                # THE metric that predicts generation quality.  `nw4_diag_vstrue.py`
                # showed the generation score tracks cos(head_out, that trial's TRUE
                # clip_img), and that this is ANTI-correlated with retrieval accuracy:
                # the most discriminative checkpoint had vs_true 0.257 while a constant
                # vector scored 0.620, and the generator responded to the latter.
                # Selecting on CSLS therefore picked the worst checkpoint for
                # generation.  This is the mean cosine to each val row's own target.
                tv = ti_img[val_i].float()
                vt_img = float((F.normalize(ov["z_img"].float(), dim=-1) * tv).sum(-1).mean().cpu())
                tv_v = ti_vith[val_i].float()
                vt_vith = float((F.normalize(ov["z_vith"].float(), dim=-1) * tv_v).sum(-1).mean().cpu())
                tv_a = ti_attr[val_i].float()
                vt_attr = float((F.normalize(ov["z_attr"].float(), dim=-1) * tv_a).sum(-1).mean().cpu())
                # the floor this must beat: a constant (the train-bank mean) scores
                # ~0.62 on the same metric, so vs_cl is the honest net signal.
                cmean = F.normalize(ti_img.mean(0, keepdim=True), dim=-1).float()
                floor = float((cmean * tv).sum(-1).mean().cpu())
            row = {"epoch": ep, "loss": run / max(nstep, 1), "depth_pearson": dpear,
                   "img_vs_true": vt_img, "vith_vs_true": vt_vith, "attr_vs_true": vt_attr,
                   "img_vs_cl": vt_img - floor, "vs_cl_floor": floor}
            for nm, m in (("img", m_img), ("vith", m_vith), ("attr", m_attr)):
                for k, v in m.items():
                    row[f"{nm}_{k}"] = v
            # Primary selector.  Default `img_vs_cl` (net semantic signal the adapter can
            # read); `img_mini_csls` is kept selectable because it is the metric the
            # earlier runs used, so the two can be compared directly.
            if args.select_metric == "img_mini_csls":
                row["score"] = float(row["img_mini_csls"])
            elif args.select_metric == "img_mini_top1":
                row["score"] = float(row["img_mini_top1"])
            elif args.select_metric == "img_rsa":
                # default: the axis that predicts generation (corr +0.967) rather than
                # vs_cl (+0.687) or retrieval, which was measured to be anti-correlated
                row["score"] = float(row.get("img_mini_rsa", 0.0))
            else:
                row["score"] = float(row["img_vs_cl"])
            history.append(row)
            print(f"[nw4-s1] ep{ep:02d} loss={row['loss']:.4f} "
                  f"img(csls={row['img_mini_csls']:.4f} t1={row['img_mini_top1']:.4f} "
                  f"vs_cl={row['img_vs_cl']:+.4f}) "
                  f"vith(csls={row['vith_mini_csls']:.4f} vs_cl={vt_vith - floor:+.4f}) "
                  f"attr(csls={row['attr_mini_csls']:.4f}) dpear={dpear:.4f} "
                  f"[sel={args.select_metric}]")
            if row["score"] > best["score"]:
                best = {"score": row["score"], "epoch": ep}
                torch.save({"state_dict": model.state_dict(), "args": vars(args),
                            "best": best, "history": history}, ckpt)

        (out / "s1_history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
        print(f"[nw4-s1] best {best}")
        if not ckpt.is_file():
            raise SystemExit("[FATAL] no checkpoint saved")
        model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["state_dict"])

    # ---------------- export test conditions ----------------
    model.eval()
    with torch.no_grad():
        ot = model(torch.from_numpy(zte).to(dev))
        z_img = l2t(ot["z_img"]).float().cpu().numpy().astype(np.float32)
        z_vith = l2t(ot["z_vith"]).float().cpu().numpy().astype(np.float32)
        z_attr = l2t(ot["z_attr"]).float().cpu().numpy().astype(np.float32)
        depth = ot["depth"].float().cpu().numpy().astype(np.float32)
        vae = ot["vae"].float().cpu().numpy().astype(np.float32)
        vae = vae * v_std + v_mean
    np.save(out / "conds" / "z_img_test.npy", z_img)
    np.save(out / "conds" / "z_vith_test.npy", z_vith)
    np.save(out / "conds" / "z_attr_test.npy", z_attr)
    np.save(out / "conds" / "z_img_gt_test.npy", t_img_te)   # reference, for audits
    np.save(out / "spatial" / "pred_depth_test_64.npy", depth)
    np.save(out / "spatial" / "pred_vae_test_scaled.npy", vae)

    from PIL import Image
    ddir = out / "spatial" / "pred_depth_rgb_512"
    ddir.mkdir(parents=True, exist_ok=True)
    for i in range(len(depth)):
        d = depth[i]
        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
        Image.fromarray((np.stack([d, d, d], -1) * 255).astype(np.uint8)).resize(
            (512, 512), Image.Resampling.BICUBIC).save(ddir / f"{i:03d}.png")

    if args.decode_rgb:
        try:
            from train_eeg_vae_head import decode_latents, resolve_vae  # type: ignore
            import os
            hub = Path(os.environ.get("HF_HUB_CACHE",
                                      "/project/peilab/why/cache/eeg-brainit/hf/hub"))
            vae_m = resolve_vae(hub, dev)
            rgb_dir = out / "spatial" / "pred_lowlevel_rgb_512"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            for s in range(0, len(vae), 8):
                imgs = decode_latents(vae_m, torch.from_numpy(vae[s:s + 8]).to(dev),
                                      args.scaling_factor)
                for j, im in enumerate(imgs):
                    im.save(rgb_dir / f"{s + j:03d}.png")
            del vae_m
            print(f"[nw4-s1] decoded lowlevel RGB -> {rgb_dir}")
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] VAE decode failed: {type(e).__name__}: {e}")

    rep = {"stage": "nw4_s1", "subject": sid, "best": best,
           "targets": {"img": list(t_img.shape), "vith": list(t_vith.shape),
                       "attr": list(t_attr.shape), "attr_fields": ATTR_FIELDS},
           "objective": "instance-level NCE on 3 spaces + manifold anchor (A0+A1)",
           "history": history}
    (out / "report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("best", "targets", "objective")}, indent=2))


if __name__ == "__main__":
    main()
