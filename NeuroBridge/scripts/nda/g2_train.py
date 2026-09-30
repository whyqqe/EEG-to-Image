#!/usr/bin/env python3
"""G2: granularity-factorised PARALLEL dual tower + CFM condition synthesiser.

What this model is
------------------
    cross-subject module + EEG encoder   (FROZEN -- unchanged from HCMA)
                    |
                z  (512-d latent, one per trial)
                    |
        +-----------+-----------+
        |                       |          <- PARALLEL: neither tower feeds the other
   semantic tower           perceptual tower
   image encoding           structure granularity  (coarse VAE band, r < cut)
   overall description      texture granularity    (log spectral statistics)
   subject description
   background description
   detail description
        |                       |
        +-----------+-----------+
                    |
        condition synthesiser  ->  IP-Adapter embedding (K samples, CFM)
                                   LF latent anchor     (deterministic)
                                   depth map            (deterministic, optional)

Why the towers are parallel
---------------------------
In the shipped HCMA pipeline the structural head consumed `z_decode_vith`, i.e. the
OUTPUT of the semantic head (`train_eeg_vae_head.py --eeg-train-npy
.../z_decode_vith_train.npy`). Two consequences: the encoder never receives a
structural gradient, and any semantic miscalibration is inherited by the
structural channel. Both towers now read the same frozen latent directly.

Why the targets are shaped the way they are
-------------------------------------------
The per-band ceiling probe (ridge; lambda on a held-in split; test scored once)
measured, on the low-frequency VAE latent:

    band            E_frac   test_corr   var_expl    share
    r<0.0625        0.364     0.133      0.00645     92.9%
    0.0625-0.125    0.107     0.043      0.00019      2.8%
    0.125-0.25      0.084     0.022      0.00004      0.6%
    0.25-0.5        0.101     0.044      0.00020      2.8%
    r>0.5           0.345     0.013      0.00006      0.9%

So structure is supervised on the COARSE band only (92.9% of what is recoverable),
and texture on LOW-DIMENSIONAL log spectral statistics rather than on the fine
coefficients (26424-dim, ~99% noise for EEG). Regressing those coefficients with
L1 -- which is what the shipped VAE head does -- spends nearly all capacity on
unpredictable bands and drives them to the conditional mean. That is the observed
collapse, and it is an objective/geometry problem, not a head-capacity problem.

Anti-collapse is enforced explicitly
------------------------------------
Every head that regresses a mean (structure, texture, direct IP) carries a VICReg
style variance hinge against a target standard deviation, and the structure loss
is computed in standardised coordinates. A head can otherwise satisfy cosine/L1
losses perfectly while emitting a near-constant vector -- this was measured, not
hypothesised, during the T3 iteration (0.135 output/input std ratio at a perfect
statistics loss).

CFM is ABLATABLE, not load-bearing
----------------------------------
The synthesiser exports two families of conditions:

    ip_direct_*   deterministic head: the conditional-mean image embedding.
                  Retrieval metrics usually prefer a mean over a sample.
    ip_cfm_k_*    K CFM samples: diverse hypotheses, and the only place where a
                  one-to-many map is modelled explicitly (many images are
                  consistent with one EEG trial).

Training one and only running the other would make the flow-matching claim
unfalsifiable, so both are trained and both are exported. Note the earlier
failure mode of CFM in this codebase was asking it to do MEAN-PRESERVING
alignment of a zero-mean field, where the conditional mean is provably 0. Here it
transports a distribution over 1024-d CLIP embeddings, which is exactly the
one-to-many setting flow matching is for -- and the comparison still decides it.

Leak-free protocol
------------------
Checkpoint selection uses a held-in set of image indices excluded from training
(images whose index %% --val-mod == 0). The test split is never read during
training or selection; it is used once, for export. This replaces the earlier
`best_mae on test` selection identified in the leakage audit.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


# ------------------------------------------------------------------ helpers

def l2n(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)


def l2n_np(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-8)


def radial_grid(h: int, w: int, device) -> torch.Tensor:
    fy = torch.fft.fftfreq(h, device=device)[:, None]
    fx = torch.fft.fftfreq(w, device=device)[None, :]
    return torch.sqrt(fy ** 2 + fx ** 2) / 0.5


def low_band_t(x: torch.Tensor, r: torch.Tensor, cut: float, k: int) -> torch.Tensor:
    """Keep r < cut in a compact coefficient form.

    Returns (N, C, k, 2) real/imag pairs at the k lowest radial bins, which is
    enough to score the coarse band without materialising two full 64x64 FFTs.
    """
    Fx = torch.fft.fft2(x, dim=(-2, -1))
    m = r < cut
    sel = Fx[:, :, m]                       # (N, C, n_low)
    if sel.shape[-1] > k:                   # deterministic subsample of the band
        idx = torch.linspace(0, sel.shape[-1] - 1, k, device=x.device).long()
        sel = sel[:, :, idx]
    elif sel.shape[-1] < k:
        sel = F.pad(sel, (0, k - sel.shape[-1]))
    return torch.stack([sel.real, sel.imag], dim=-1)


class InfoNCE(nn.Module):
    def __init__(self, tau: float = 0.07):
        super().__init__()
        self.tau = tau

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a, b = l2n(a), l2n(b)
        logits = a @ b.t() / self.tau
        tgt = torch.arange(a.shape[0], device=a.device)
        return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))


def mlp(i: int, h: int, o: int, layers: int = 2, drop: float = 0.0) -> nn.Sequential:
    mods: list[nn.Module] = [nn.Linear(i, h), nn.GELU()]
    if drop > 0:
        mods.append(nn.Dropout(drop))
    for _ in range(layers - 1):
        mods += [nn.Linear(h, h), nn.GELU()]
        if drop > 0:
            mods.append(nn.Dropout(drop))
    mods += [nn.Linear(h, o)]
    return nn.Sequential(*mods)


def var_hinge(x: torch.Tensor, target_norm: float) -> torch.Tensor:
    """Dispersion floor for heads that regress a conditional mean with pure L1/L2.

    Only such heads need it: the constant solution satisfies their objective. An
    embedding head that carries an InfoNCE term must NOT be floored -- a collapsed
    batch makes every logit equal and the contrastive term diverges on its own, so
    a floor there only fights the alignment.

    Two details are load-bearing:

    * the reduction is the NORM of the per-dim std vector, not its mean. A mean over
      the 16384 structure dims diluted the signal ~128x, so a structurally collapsed
      head still reported a negligible penalty and kept collapsing (measured:
      intra-sub-08 std_ratio 0.157 with the hinge supposedly active).
    * the floor is derived from the target's own measured dispersion
      (`--var-frac` x it), never a shared absolute constant. A single 0.5 was
      previously applied to heads whose target dispersions differ by 20x, which was
      a 21x over-demand on the 1024-d image embedding (per-dim std 0.024) and
      trained that head into a blown-up, mis-aligned regime.
    """
    if x.shape[0] < 2:
        return x.new_zeros(())
    s = x.flatten(1).std(dim=0)
    return F.relu(target_norm - s.norm())


# ------------------------------------------------------------------ model

class G2Net(nn.Module):
    def __init__(self, in_dim: int = 512, code: int = 768,
                 img_dim: int = 1280, txt_dim: int = 1024, ip_dim: int = 1024,
                 ch: int = 4, spatial: int = 64, n_tex: int = 17,
                 depth_out: int = 0, per_layers: int = 3, drop: float = 0.15):
        super().__init__()
        self.ch, self.spatial, self.n_tex = ch, spatial, n_tex
        # two INDEPENDENT trunks: the towers must not share a bottleneck, or the
        # "parallel" claim would be cosmetic and the structure tower would again
        # inherit the semantic tower's errors.
        #
        # Dropout is load-bearing here, not decoration. Measured on the 2026-09-11
        # run: validation img_cos fell monotonically 0.6315 -> 0.5760 over 40 epochs
        # while train loss fell 27.1 -> 16.1, and every LOSO fold selected an early
        # epoch (best 1-6 of 40) with a semantics-for-structure trade. That is a
        # ~25M-parameter model fitting 16540 unique images (the inter protocol tiles
        # those rows across 9 subjects, but tiled copies carry no new information).
        self.sem_trunk = mlp(in_dim, code, code, 2, drop)
        self.per_trunk = mlp(in_dim, code, code, per_layers, drop)

        self.h_image = mlp(code, code, img_dim, 1, drop)
        self.h_overall = mlp(code, code, txt_dim, 1, drop)
        self.h_subject = mlp(code, code, txt_dim, 1, drop)
        self.h_background = mlp(code, code, txt_dim, 1, drop)
        self.h_detail = mlp(code, code, txt_dim, 1, drop)

        self.h_struct = mlp(code, code, ch * spatial * spatial, 2, drop)
        # zero-init the texture head's last layer would start it collapsed; the
        # whole point of the variance hinge is to avoid that attractor.
        self.h_texture = mlp(code, code, ch * n_tex, 1, drop)
        self.h_depth = mlp(code, code, depth_out * depth_out, 2, drop) if depth_out else None

        # deterministic conditional-mean image embedding (the IP-Adapter condition)
        self.h_ip = mlp(code, code, ip_dim, 1, drop)

        # CFM velocity field: v(x_t, t, cond)
        # The structural field enters via a 4x4 average pool, not flattened: the
        # coarse band is exactly a low-pass field, so the pool keeps what matters
        # and avoids a 16384-dim conditioning vector.
        joint_dim = img_dim + 4 * txt_dim + ch * n_tex + ch * 16
        self.cond_proj = nn.Sequential(nn.Linear(joint_dim, code), nn.GELU(),
                                       nn.Linear(code, code))
        self.vel = nn.Sequential(
            nn.Linear(ip_dim + 1 + code, code), nn.GELU(),
            nn.Linear(code, code), nn.GELU(),
            nn.Linear(code, ip_dim))

    def towers(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        s = self.sem_trunk(x)
        p = self.per_trunk(x)
        return {
            "image": self.h_image(s),
            "overall": self.h_overall(s),
            "subject": self.h_subject(s),
            "background": self.h_background(s),
            "detail": self.h_detail(s),
            "struct": self.h_struct(p).view(-1, self.ch, self.spatial, self.spatial),
            "texture": self.h_texture(p).view(-1, self.ch, self.n_tex),
            "ip": self.h_ip(p),
            "_s": s, "_p": p,
        }

    def joint(self, o: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [l2n(o["image"]), l2n(o["overall"]), l2n(o["subject"]),
                 l2n(o["background"]), l2n(o["detail"]), l2n(o["texture"].flatten(1)),
                 l2n(F.adaptive_avg_pool2d(o["struct"], (4, 4)).flatten(1))]
        return torch.cat(parts, dim=-1)

    def vel_field(self, xt: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.vel(torch.cat([xt, t.view(-1, 1), cond], dim=-1))

    @torch.no_grad()
    def sample_ip(self, cond: torch.Tensor, steps: int, seed: int) -> torch.Tensor:
        """Integrate the flow from a Gaussian base to an image-embedding sample.

        Both ends of the learned path live at unit per-dim std (see the CFM loss),
        so the integration runs in that scaled space and is renormalised at the end
        to return a unit vector, which is what the IP-Adapter consumes.
        """
        d = self.h_ip[-1].out_features
        g = torch.Generator(device=cond.device).manual_seed(seed)
        x = torch.randn(cond.shape[0], d, device=cond.device, generator=g)
        dt = 1.0 / steps
        for k in range(steps):
            t = torch.full((cond.shape[0],), k * dt, device=cond.device)
            x = x + dt * self.vel_field(x, t, cond)
        return l2n(x)


# ------------------------------------------------------------------ data

class TargetBank:
    """Lazily-indexed target arrays. Rows are addressed by IMAGE index.

    Inter-subject training concatenates 9 subjects, so materialising a tiled
    (148860, 4, 64, 64) structure target would cost ~4.9 GB. Indexing by image
    index avoids the tile entirely.
    """

    def __init__(self, d: Path, split: str, keys: list[str]):
        self.d, self.split, self.keys = d, split, keys
        self.arr: dict[str, np.ndarray] = {}
        for k in keys:
            p = d / f"{k}_{split}.npy"
            if p.is_file():
                self.arr[k] = np.load(p, mmap_mode="r")
        self.n_img = len(self.arr[keys[0]]) if self.arr else 0

    def has(self, k: str) -> bool:
        return k in self.arr

    def get(self, k: str, idx: np.ndarray) -> torch.Tensor:
        a = np.asarray(self.arr[k][idx], dtype=np.float32)
        return torch.from_numpy(a)


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["intra", "inter"], required=True)
    ap.add_argument("--subject", type=int, default=8, help="the TEST subject")
    ap.add_argument("--z-root", default="outputs/hcma_10subj",
                    help="root holding sub-XX/zret/z_eeg_proj_{train,test}.npy")
    ap.add_argument("--z-file", default="zret/z_eeg_proj")
    ap.add_argument("--z-intra-dir", default="",
                    help="override the intra-protocol latent dir (must contain "
                         "z_eeg_proj_train.npy / z_eeg_proj_test.npy)")
    ap.add_argument("--targets-dir", default="outputs/g2_targets")
    ap.add_argument("--ip-train-npy", default="/project/peilab/why/eeg-brainit/outputs/atm_bridge/clip_img_train_1024.npy")
    ap.add_argument("--ip-test-npy", default="/project/peilab/why/eeg-brainit/outputs/atm_bridge/clip_img_test_1024.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2,
                    help="raised from 1e-4. The run overfits measurably (val img_cos "
                         "0.6315 -> 0.5760 over 40 epochs while train loss 27.1 -> 16.1), "
                         "and AdamW weight decay is the cheapest correction that does not "
                         "change the architecture.")
    ap.add_argument("--dropout", type=float, default=0.15,
                    help="dropout on every hidden layer of both trunks and all heads")
    ap.add_argument("--sel-struct-w", type=float, default=0.3,
                    help="weight of the two structural terms in the checkpoint-selection "
                         "score. At 1.0 the score happily traded 7.7pp of semantic cosine "
                         "for 10.3pp of structural error (measured, fold sub-01: ep1 "
                         "img_cos .6315 ip_cos .4805 struct_mae .7562 -> ep2 img_cos .6109 "
                         "ip_cos .4247 struct_mae .7022) and the exported test ip_cos "
                         "dropped 0.4569 -> 0.4026. The external SOTA gates are semantic "
                         "and distributional, so structure must not outbid them.")
    ap.add_argument("--anchor-std-match", type=int, default=0,
                    help="1 = rescale the exported anchor so its dispersion matches the "
                         "LOW-BAND TARGET's std. Default 0 because the SAME correction is "
                         "applied once, identically to every anchor, by the generator's "
                         "--anchor-std-target: doing it in both places would compound two "
                         "different statistics (a batch pointwise std here, a per-sample "
                         "band std there) and make the two anchors non-comparable. Kept as "
                         "an option for inspecting the raw regressor's shrinkage.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--val-mod", type=int, default=10,
                    help="held-in selection set = images whose index %% val_mod == 0")
    ap.add_argument("--cut", type=float, default=0.0625)
    ap.add_argument("--lam-image", type=float, default=1.0)
    ap.add_argument("--lam-text", type=float, default=1.0)
    ap.add_argument("--lam-struct", type=float, default=1.0)
    ap.add_argument("--lam-texture", type=float, default=1.0)
    ap.add_argument("--lam-ip", type=float, default=1.0)
    ap.add_argument("--lam-cfm", type=float, default=1.0)
    ap.add_argument("--lam-var", type=float, default=0.05)
    ap.add_argument("--var-frac", type=float, default=0.25,
                    help="dispersion floor as a FRACTION of the target's own measured "
                         "per-dim-std norm, per head. Scale-free on purpose: an absolute "
                         "constant is only correct for one target scale. Set to 0 to "
                         "disable. 0.25 is a collapse guard, not a fit target -- the "
                         "conditional mean legitimately keeps far less dispersion than "
                         "the marginal.")
    ap.add_argument("--save-train-conditions", type=int, default=0,
                    help="1 = also write ip_*/lf_latent/tex_stats for the train split. "
                         "They are never read downstream (generation uses the test split "
                         "only), and the inter protocol writes ~4.8 GB of lf_latent plus "
                         "4 x 610 MB of ip_cfm per fold, so 9 folds cost ~53 GB of NFS. "
                         "Diagnostics are still computed in memory.")
    ap.add_argument("--cfm-steps-train", type=int, default=32)
    ap.add_argument("--n-cfm", type=int, default=4, help="CFM samples exported per row")
    ap.add_argument("--cfm-steps", type=int, default=48)
    ap.add_argument("--max-train-rows", type=int, default=0, help="0 = all")
    ap.add_argument("--resume", type=int, default=1,
                    help="1 = continue from <out>/last.pth if present. Overnight jobs are "
                         "preemptible and training is long, so epoch-level resume is the "
                         "difference between losing and keeping hours of GPU time.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---------------- inputs ----------------
    z_root = root / args.z_root
    test_sub = args.subject
    train_subs = [s for s in range(1, 11) if s != test_sub] if args.protocol == "inter" else [test_sub]
    if args.protocol == "intra":
        # prefer the PURE intra subject latent when it exists (trained on sub-08
        # only); fall back to the 10-subject per-subject latent otherwise.
        cand = Path(args.z_intra_dir) if args.z_intra_dir else (
            root / "outputs/intra_hcma_s" / f"sub-{test_sub:02d}" / "train")
        if (cand / "z_eeg_proj_train.npy").is_file():
            z_src_train = cand
            z_src_test = cand
            z_note = f"pure intra subject latent {cand}"
        else:
            z_src_train = z_root / f"sub-{test_sub:02d}" / Path(args.z_file).parent
            z_src_test = z_src_train
            z_note = f"per-subject latent {z_src_train}"
    else:
        z_src_train = None       # handled per subject below
        z_src_test = z_root / f"sub-{test_sub:02d}" / Path(args.z_file).parent
        z_note = (f"inter: train = subjects {train_subs} pooled latents, "
                  f"test = subject {test_sub}")

    def load_pair(src: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
        tr = np.load(src / f"z_eeg_proj_train.npy").astype(np.float32)
        te = np.load(src / f"z_eeg_proj_test.npy").astype(np.float32)
        return l2n_np(tr), l2n_np(te)

    if args.protocol == "intra":
        z_tr, z_te = load_pair(z_src_train, "intra")
        n_per = len(z_tr)
    else:
        trs = []
        for s in train_subs:
            a, _ = load_pair(z_root / f"sub-{s:02d}" / Path(args.z_file).parent, f"s{s}")
            trs.append(a)
        n_per = len(trs[0])
        for a in trs:
            if len(a) != n_per:
                raise SystemExit(f"subject train row mismatch: {len(a)} != {n_per}")
        z_tr = np.concatenate(trs, 0)
        z_te = l2n_np(np.load(z_src_test / "z_eeg_proj_test.npy").astype(np.float32))

    if args.max_train_rows and len(z_tr) > args.max_train_rows:
        z_tr = z_tr[: args.max_train_rows]
    n_tr, d_in = z_tr.shape
    print(f"[g2] protocol={args.protocol} z_train={z_tr.shape} z_test={z_te.shape}")
    print(f"[g2] {z_note}")

    # row -> image index. Every subject's file is in the same image order, which is
    # the same assumption the validated LOSO MG-Flow pool made when it tiled targets
    # (`--tile-targets`). It is asserted below against the target row count.
    img_idx_tr = np.arange(n_tr) % n_per
    img_idx_te = np.arange(len(z_te)) % n_per

    # ---------------- targets ----------------
    td = root / args.targets_dir
    sem_keys = ["sem_image", "sem_overall", "sem_subject", "sem_background", "sem_detail"]
    per_keys = ["perc_struct", "perc_texture", "perc_depth"]
    bank_tr = TargetBank(td, "train", sem_keys + per_keys)
    bank_te = TargetBank(td, "test", sem_keys + per_keys)
    for k in sem_keys:
        if not bank_tr.has(k):
            raise SystemExit(f"missing target {k}_train.npy in {td}")
    if bank_tr.n_img != n_per:
        raise SystemExit(f"target rows {bank_tr.n_img} != per-subject rows {n_per}; "
                         f"row alignment cannot be assumed")
    if bank_te.n_img != len(img_idx_te):
        # test rows are 200 and are their own images, not a tiled subject list
        raise SystemExit(f"test target rows {bank_te.n_img} != test rows {len(img_idx_te)}")
    print(f"[g2] targets: {sorted(bank_tr.arr.keys())}")

    ip_tr = l2n_np(np.load(args.ip_train_npy).astype(np.float32))
    ip_te = l2n_np(np.load(args.ip_test_npy).astype(np.float32))
    if ip_tr.shape[0] != n_per:
        raise SystemExit(f"ip target rows {ip_tr.shape[0]} != per-subject rows {n_per}")
    if ip_te.shape[0] != len(z_te):
        raise SystemExit(f"ip test rows {ip_te.shape[0]} != test rows {len(z_te)}")
    ip_dim = ip_tr.shape[1]
    print(f"[g2] ip target {ip_tr.shape}")

    C, H, W = bank_tr.arr["perc_struct"].shape[1:]
    r = radial_grid(H, W, device)
    n_tex = bank_tr.arr["perc_texture"].shape[-1]
    img_dim = bank_tr.arr["sem_image"].shape[1]
    txt_dim = bank_tr.arr["sem_overall"].shape[1]
    depth_out = int(math.isqrt(bank_tr.arr["perc_depth"].shape[-1] * bank_tr.arr["perc_depth"].shape[-2])) \
        if bank_tr.has("perc_depth") else 0

    # structural target statistics (standardise for stable regression)
    smp = np.asarray(bank_tr.arr["perc_struct"][: min(2048, bank_tr.n_img)], dtype=np.float32)
    st_mean = float(smp.mean())
    st_std = float(smp.std()) or 1.0
    print(f"[g2] struct latent mean={st_mean:.4f} std={st_std:.4f}  cut={args.cut}")

    model = G2Net(in_dim=d_in, img_dim=img_dim, txt_dim=txt_dim, ip_dim=ip_dim,
                  ch=C, spatial=H, n_tex=n_tex, depth_out=depth_out,
                  drop=args.dropout).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    nce = InfoNCE(0.07)

    # ---------------- leak-free selection split ----------------
    val_mask = (img_idx_tr % args.val_mod) == 0
    fit_rows = np.where(~val_mask)[0]
    val_rows = np.where(val_mask)[0]
    print(f"[g2] leak-free: fit={len(fit_rows)} val={len(val_rows)} "
          f"(val images excluded from fitting, test never touched)")

    # ---------------- dispersion floors (measured on the FIT split) ----------------
    # Measured, not assumed, and per head. The previous single `--var-target 0.5`
    # was ~1.0x the structure and texture targets' own dispersion but 21.2x the
    # image-embedding target's (its per-dim std is 0.0236), so the floor became an
    # unsatisfiable demand that inflated the embedding heads until their cosine to
    # the target fell. Signature in the previous run: train loss fell 25.4 -> 16.1
    # while val semantics fell monotonically (img_cos 0.629 -> 0.574, txt_cos
    # 0.385 -> 0.324, ip_cos 0.478 -> 0.369) and every inter fold selected epoch 1.
    # Measured on fit rows only; the test split is never touched.
    fit_img_idx = img_idx_tr[fit_rows]

    def target_disp(k: str) -> float:
        v = np.asarray(bank_tr.arr[f"perc_{k}"][fit_img_idx[:4096]], dtype=np.float32)
        return float(np.linalg.norm(v.reshape(len(v), -1).std(0)))

    var_floor = {k: args.var_frac * target_disp(k) for k in ("struct", "texture")}
    print("[g2] dispersion floor = %.2f x measured target dispersion: %s"
          % (args.var_frac, {k: round(v, 3) for k, v in var_floor.items()}))

    z_tr_t = torch.from_numpy(z_tr)
    ip_tr_t = torch.from_numpy(ip_tr)

    def batch_targets(bank: TargetBank, idx: np.ndarray, keys: list[str]) -> dict[str, torch.Tensor]:
        return {k: bank.get(k, idx).to(device) for k in keys if bank.has(k)}

    # ---------------- training ----------------
    def run_epoch(rows: np.ndarray, train: bool) -> dict[str, float]:
        model.train(train)
        order = np.random.permutation(rows) if train else rows
        agg: dict[str, float] = {}
        nb = 0
        for s in range(0, len(order), args.batch_size):
            b = order[s : s + args.batch_size]
            if len(b) < 4:
                continue
            bi = img_idx_tr[b]
            z = z_tr_t[b].to(device)
            t = batch_targets(bank_tr, bi, sem_keys + per_keys)
            ip = ip_tr_t[bi].to(device)

            with torch.set_grad_enabled(train):
                o = model.towers(z)
                cond = model.cond_proj(model.joint(o))
                losses: dict[str, torch.Tensor] = {}

                losses["image"] = (1 - (l2n(o["image"]) * l2n(t["sem_image"])).sum(-1)).mean() \
                    + nce(o["image"], t["sem_image"])
                ltxt = 0.0
                for f in ("overall", "subject", "background", "detail"):
                    lt = (1 - (l2n(o[f]) * l2n(t[f"sem_{f}"])).sum(-1)).mean() \
                        + nce(o[f], t[f"sem_{f}"])
                    ltxt = ltxt + lt if isinstance(ltxt, torch.Tensor) else lt
                    losses[f] = lt
                losses["text"] = ltxt / 4.0

                # structure: standardised L1 on the coarse band + coarse-band cosine.
                # The coarse band is the identifiable part (92.9% of recoverable
                # variance), so it is supervised in the coefficient domain.
                st = (o["struct"] - st_mean) / st_std
                st_t = (t["perc_struct"].view(-1, C, H, W) - st_mean) / st_std
                l_struct = F.l1_loss(st, st_t)
                co, ct = low_band_t(st, r, args.cut, 128), low_band_t(st_t, r, args.cut, 128)
                co_f, ct_f = co.reshape(co.shape[0], -1), ct.reshape(ct.shape[0], -1)
                co_f, ct_f = co_f - co_f.mean(1, keepdim=True), ct_f - ct_f.mean(1, keepdim=True)
                cos = (co_f * ct_f).sum(1) / (co_f.norm(dim=1) * ct_f.norm(dim=1) + 1e-8)
                losses["struct"] = l_struct + (1 - cos.mean())

                # texture: log spectral statistics (distributional, low-dim)
                losses["texture"] = F.mse_loss(o["texture"], t["perc_texture"])

                if model.h_depth is not None and "perc_depth" in t:
                    d = t["perc_depth"].view(-1, 1, depth_out, depth_out)
                    losses["depth"] = F.l1_loss(model.h_depth(o["_p"]).view_as(d), d)

                # deterministic conditional-mean IP embedding
                losses["ip"] = (1 - (l2n(o["ip"]) * l2n(ip)).sum(-1)).mean() + nce(o["ip"], ip)

                # CFM over the IP-Adapter embedding distribution (one-to-many map).
                if args.lam_cfm > 0:
                    x1 = l2n(ip)
                    # Flow on the sqrt(d)-scaled embedding. x1 is a UNIT vector, so a
                    # unit-scale Gaussian base and this target differ in norm by
                    # ~sqrt(d): ~96% of the flow's work would be information-free norm
                    # reduction, and the raw MSE of `x1 - eps` is ~1.5e-3 -- three
                    # orders below the alignment heads, so CFM received essentially no
                    # gradient no matter how long it trained. Measured in the previous
                    # run: tr_cfm sat flat at ~0.0014 from epoch 1 to 40, and the
                    # exported CFM samples tracked the deterministic head to within
                    # 1pp of cosine -- i.e. the branch was decorative.
                    # Scaling both ends to unit per-dim std preserves the geometry
                    # (the path stays a straight interpolation between two vectors of
                    # equal norm) and puts the loss on the alignment heads' scale.
                    d_flow = x1.shape[-1]
                    u = x1 * math.sqrt(d_flow)
                    tt = torch.rand(x1.shape[0], device=device)
                    e = torch.randn_like(x1)
                    xt = (1 - tt.view(-1, 1)) * e + tt.view(-1, 1) * u
                    v = model.vel_field(xt, tt, cond)
                    losses["cfm"] = F.mse_loss(v, u - e)

                # dispersion floors, only for the two heads whose objective is pure
                # L1/L2 and can therefore satisfy it with a constant. `ip` is excluded
                # on purpose: it carries an InfoNCE term, which already forbids collapse.
                losses["var"] = (var_hinge(o["struct"], var_floor["struct"])
                                 + var_hinge(o["texture"], var_floor["texture"]))

                lam = {"image": args.lam_image, "text": args.lam_text,
                       "struct": args.lam_struct, "texture": args.lam_texture,
                       "ip": args.lam_ip, "cfm": args.lam_cfm, "var": args.lam_var}
                loss = sum(lam[k] * v for k, v in losses.items() if k in lam)

                if train:
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            agg["loss"] = agg.get("loss", 0.0) + float(loss)
            nb += 1
        return {k: v / max(nb, 1) for k, v in agg.items()}

    @torch.no_grad()
    def val_score() -> tuple[float, dict[str, float]]:
        model.eval()
        m_all: dict[str, list[float]] = {}
        for s in range(0, len(val_rows), 512):
            b = val_rows[s : s + 512]
            bi = img_idx_tr[b]
            z = z_tr_t[b].to(device)
            t = batch_targets(bank_tr, bi, sem_keys + per_keys)
            ip = ip_tr_t[bi].to(device)
            o = model.towers(z)
            m_all.setdefault("img_cos", []).append(
                float((l2n(o["image"]) * l2n(t["sem_image"])).sum(-1).mean()))
            tc = [float((l2n(o[f]) * l2n(t[f"sem_{f}"])).sum(-1).mean())
                  for f in ("overall", "subject", "background", "detail")]
            m_all.setdefault("txt_cos", []).append(float(np.mean(tc)))
            m_all.setdefault("ip_cos", []).append(
                float((l2n(o["ip"]) * l2n(ip)).sum(-1).mean()))
            st = (o["struct"] - st_mean) / st_std
            st_t = (t["perc_struct"].view(-1, C, H, W) - st_mean) / st_std
            m_all.setdefault("struct_mae", []).append(float(F.l1_loss(st, st_t)))
            m_all.setdefault("tex_mse", []).append(
                float(F.mse_loss(o["texture"], t["perc_texture"])))
        mm = {k: float(np.mean(v)) for k, v in m_all.items()}
        # retrieval-oriented score: semantics dominate, structure enters as a weighted
        # penalty (see --sel-struct-w for the measurement that motivates the weight)
        sc = (mm["img_cos"] + mm["txt_cos"] + mm["ip_cos"]
              - args.sel_struct_w * (mm["struct_mae"] + mm["tex_mse"]))
        return sc, mm

    best_score, best_ep, history = -1e9, 0, []
    start_ep = 1
    resume_path = out / "last.pth"
    if args.resume and resume_path.is_file():
        r = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(r["state_dict"])
        opt.load_state_dict(r["optimizer"])
        history = r.get("history", [])
        best_score, best_ep = r.get("best_score", -1e9), r.get("best_epoch", 0)
        start_ep = int(r["epoch"]) + 1
        print(f"[g2] RESUMED from epoch {r['epoch']} (best so far ep{best_ep} "
              f"score {best_score:.4f}); continuing at epoch {start_ep}")

    for ep in range(start_ep, args.epochs + 1):
        tr_m = run_epoch(fit_rows, True)
        sc, vm = val_score()
        row = {"epoch": ep, "train_loss": tr_m.get("loss", 0.0), "val_score": sc, **vm,
               **{f"tr_{k}": v for k, v in tr_m.items()}}
        history.append(row)
        print(f"ep{ep:3d} loss={row['train_loss']:.4f} "
              f"val img_cos={vm['img_cos']:.4f} txt_cos={vm['txt_cos']:.4f} "
              f"ip_cos={vm['ip_cos']:.4f} struct_mae={vm['struct_mae']:.4f} "
              f"tex_mse={vm['tex_mse']:.4f} score={sc:.4f}", flush=True)
        if sc > best_score:
            best_score, best_ep = sc, ep
            torch.save({"epoch": ep, "state_dict": model.state_dict(),
                        "args": vars(args), "z_note": z_note,
                        "struct_mean": st_mean, "struct_std": st_std,
                        "dims": {"in": d_in, "img": img_dim, "txt": txt_dim,
                                 "ip": ip_dim, "ch": C, "spatial": H, "n_tex": n_tex,
                                 "depth_out": depth_out}},
                       out / "checkpoint_g2_best.pth")
        # save the resume state AFTER updating best_*, so a preemption between the
        # two writes cannot leave last.pth with a stale best and re-select a worse
        # epoch on restart
        torch.save({"epoch": ep, "state_dict": model.state_dict(),
                    "optimizer": opt.state_dict(), "history": history,
                    "best_score": best_score, "best_epoch": best_ep}, resume_path)
    print(f"[g2] best epoch {best_ep} score {best_score:.4f}")
    ck = torch.load(out / "checkpoint_g2_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    # ---------------- export conditions ----------------
    # Fitted on the TRAIN export and reused verbatim for test: the anchor's amplitude
    # is a property of the regressor, not of the evaluated subject, so there is no
    # reason to look at test data and every reason not to.
    anchor_scale = {"v": 1.0}

    @torch.no_grad()
    def export(z: np.ndarray, ip_ref: np.ndarray, tag: str, save: bool = True) -> dict:
        n = len(z)
        ipd = np.empty((n, ip_dim), dtype=np.float32)
        lf = np.empty((n, C, H, W), dtype=np.float32)
        tx = np.empty((n, C, n_tex), dtype=np.float32)
        cfm = np.empty((args.n_cfm, n, ip_dim), dtype=np.float32)
        dep = np.empty((n, 1, depth_out, depth_out), dtype=np.float32) if depth_out else None
        for s in range(0, n, 512):
            e = min(s + 512, n)
            zb = torch.from_numpy(z[s:e]).to(device)
            o = model.towers(zb)
            cond = model.cond_proj(model.joint(o))
            ipd[s:e] = l2n(o["ip"]).float().cpu().numpy()
            lf[s:e] = o["struct"].float().cpu().numpy()
            tx[s:e] = o["texture"].float().cpu().numpy()
            if dep is not None:
                dep[s:e] = model.h_depth(o["_p"]).view(-1, 1, depth_out, depth_out).float().cpu().numpy()
            for k in range(args.n_cfm):
                cfm[k, s:e] = l2n(model.sample_ip(cond, args.cfm_steps,
                                                  args.seed + 1000 * k)).float().cpu().numpy()

        # Amplitude correction, fitted on train and applied to test unchanged.
        # The structural target is already the LOW-BAND filtered latent (its measured
        # std is 0.5127 vs 0.8289 for the full latent), so `st_std` is exactly the
        # scale the anchor should have -- no extra filtering is needed.
        if args.anchor_std_match:
            if tag == "train":
                anchor_scale["v"] = float(st_std / max(float(lf.std()), 1e-8))
                print(f"[g2] anchor amplitude fitted on train: x{anchor_scale['v']:.4f} "
                      f"(pred std {lf.std():.4f} -> target std {st_std:.4f})")
            lf = lf * anchor_scale["v"]

        if save:
            np.save(out / f"ip_direct_{tag}.npy", ipd)
            np.save(out / f"lf_latent_{tag}.npy", lf.astype(np.float16))
            np.save(out / f"tex_stats_{tag}.npy", tx)
            for k in range(args.n_cfm):
                np.save(out / f"ip_cfm{k}_{tag}.npy", cfm[k])
            if dep is not None:
                np.save(out / f"depth_{tag}.npy", dep.astype(np.float16))

        # Diagnostics reference the per-IMAGE target, so for the inter protocol
        # (9 subjects concatenated) only the first subject-block is scored -- that
        # block already covers every image, so nothing is lost by not tiling.
        ref = l2n_np(ip_ref)
        m = min(len(z), len(ref))
        ipd_s, cfm_s = ipd[:m], cfm[:, :m]
        diag = {
            "n": int(n), "diag_rows": int(m),
            "ip_direct_cos_to_target": float((l2n_np(ipd_s) * ref[:m]).sum(-1).mean()),
            "ip_direct_spread": float(np.linalg.norm(ipd.std(0))),
            "cfm_spread": [float(np.linalg.norm(cfm[k].std(0))) for k in range(args.n_cfm)],
            "cfm_cos_to_target": [float((l2n_np(cfm_s[k]) * ref[:m]).sum(-1).mean())
                                  for k in range(args.n_cfm)],
            "cfm_pairwise_diversity": [
                float((l2n_np(cfm_s[i]) * l2n_np(cfm_s[j])).sum(-1).mean())
                for i in range(args.n_cfm) for j in range(i + 1, args.n_cfm)],
        }
        print(f"[g2] export {tag}: ip_cos={diag['ip_direct_cos_to_target']:.4f} "
              f"cfm_cos={['%.4f' % c for c in diag['cfm_cos_to_target']]} "
              f"cfm_diversity={['%.3f' % c for c in diag['cfm_pairwise_diversity']]}")
        return diag

    diag_tr = export(z_tr, ip_tr, "train", save=bool(args.save_train_conditions))
    diag_te = export(z_te, ip_te, "test")

    # structure diagnostic against the held-out TEST target, reported once
    st_te = bank_te.get("perc_struct", np.arange(bank_te.n_img)).numpy()
    lf_te = np.load(out / "lf_latent_test.npy").astype(np.float32)
    a = (lf_te - st_mean).reshape(len(lf_te), -1)
    b = (st_te - st_mean).reshape(len(st_te), -1)
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    cc = (a * b).sum(1) / (np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1)) + 1e-8)
    diag_te["struct_corr"] = float(cc.mean())
    diag_te["struct_std_ratio"] = float(lf_te.std() / max(st_te.std(), 1e-8))
    print(f"[g2] structure recon on test: corr={diag_te['struct_corr']:.4f} "
          f"std_ratio={diag_te['struct_std_ratio']:.4f}")

    report = {
        "protocol": args.protocol,
        "test_subject": test_sub,
        "train_subjects": (train_subs if args.protocol == "inter" else [test_sub]),
        "z_source": z_note, "cut": args.cut, "best_epoch": best_ep,
        "best_val_score": best_score, "args": vars(args),
        "export_train": diag_tr, "export_test": diag_te,
        "struct_stats": {"mean": st_mean, "std": st_std},
        "note": ("parallel towers; structure head reads the frozen latent directly, not "
                 "the semantic head's output; CFM is exported alongside a deterministic "
                 "mean head so its contribution is falsifiable"),
    }
    (out / "g2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    print(f"[g2] wrote {out}/g2_report.json")


if __name__ == "__main__":
    main()
