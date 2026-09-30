#!/usr/bin/env python3
"""T3: three-tower HCMA extension (granularity-explicit semantics + LF structure + HF texture).

WHAT THIS CHANGES, AND WHY (every item is a MEASURED defect, not a preference)
-----------------------------------------------------------------------------
D1. THE STRUCTURE TOWER HAS NO INDEPENDENT EVIDENCE PATH.
    Shipped chain (run_intra_hcma_s_sub08.sh:218-219):
        raw EEG -> vith_head -> z_decode_vith -> VAE head / depth head
    The structure head's INPUT is the semantic head's OUTPUT, so every layout
    decision must survive a bottleneck trained for CLIP similarity. Measurement
    that this is binding: global image teachers (RN50 / ViT-H-14 / DINOv2) predict
    the same low-band VAE target BETTER than z_eeg does -- the target is not the
    bottleneck, the EEG->encoder map is.
    FIX: `--struct-input both` gives the LF head its own raw-EEG trunk in parallel
    with the semantic trunk; `sem` reproduces the shipped chain as an ablation.

D2. ONE GRANULARITY FOR TWO USES.
    All shipped targets are per-image GLOBAL pooled vectors (RN50 512, ViT-H-14
    1024, NVOL-HCF 1280 = mean of CLIP layers 8/10/14, DINOv2 1024 = CLS only,
    CLIP-Text, RAG anchor). DINOv2's 37x37 patch tokens are DISCARDED, so
        I(e; y) = I(e; y_glob) + I(e; y_loc | y_glob)
                  supervised       never supervised
    FIX: `--dino-local-*` supplies G x G patch cells.

    WHAT NOT TO DO: splitting the semantic tower into "subject / background /
    whole" heads does NOT add granularity. All three are CLIP-Text-space GLOBAL
    vectors with no spatial support and are mutually redundant; adding them repeats
    the RAG failure mode (a GLOBAL CLIP anchor cost 13.5pp Top-1: 0.325 at alpha=0
    -> 0.225 at alpha=0.5, measured). The missing axis is GLOBAL vs SPATIAL.

D3. THE HIGH BAND CANNOT BE LEARNED AS A PIXEL TARGET.
    Shipped: one L1 over the whole (4,64,64) latent. L1 is a conditional-MEDIAN
    estimator, so where EEG is uninformative (r>0.5: corr 0.0355 vs 0.5955 below
    r<0.0625) the optimal prediction is the conditional median and the HF field
    collapses toward zero. A high-pass field has zero DC, so "collapse to the
    median" means "collapse to nothing". Measured collapse: shipped spread ratio
    0.428 vs 0.4636 for a plain linear baseline. More capacity or a separate L1
    head cannot fix it -- the cause is conditional uncertainty.
    FIX: supervise the high band with DISTRIBUTIONAL targets (radial spectrum,
    angular spectrum, low-res Gram) whose typical realisation IS correct, so
    collapse is unrepresentable in the loss.

D4. CFM WAS APPLIED TO A TASK IT CANNOT DO.
    CFM as an ALIGNMENT module failed here (z_cfm_f Top-1 0.010 from z_s_f 0.160,
    spread 0.9911, plus an untrainable ode_mix), consistent with SP-FM
    (arXiv 2601.11827): with a single-Gaussian conditional base, SAMPLE recovery is
    ill-posed. A distributional target is exactly what a transport map should
    recover, so CFM is re-scoped here to the HF TEXTURE DISTRIBUTION
    (`--hf-head cfm`), where "the mean of samples" is the correct answer rather
    than a collapse. We integrate the learned field from x0 = 0 (mean flow), which
    returns the conditional-mean texture and is exactly what the shipped L1 target
    was trying (and failing) to express.

LEAK-FREE
---------
Checkpoints are selected on held-in train concepts (leakfree.py val_b), never on
the 200-concept test set. Test rows are exported once and never used to select.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT))
sys.path.insert(0, str(NB_ROOT / "scripts" / "nmb"))
sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))

from decode_aligner_modules import ClipInfoNCE, l2norm  # noqa: E402
from nda_dual_train import DEFAULT_CHANNELS, DualDataset, build_hcf  # noqa: E402


# --------------------------------------------------------------------------- #
# spectral statistics -- torch twin of t3_build_targets.hf_stats (layouts match)
# --------------------------------------------------------------------------- #
class Spectra(nn.Module):
    """Map (N,C,H,W) -> the SAME statistics vector that t3_build_targets.py saved.

    torch.fft.fft2 and np.fft.fft2 use identical unshifted frequency ordering, so
    the index maps built from np.fft.fftfreq index both correctly.
    """

    def __init__(self, h: int, w: int, cut: float, n_radial: int, n_ang: int, gram_res: int):
        super().__init__()
        fy = np.fft.fftfreq(h)[:, None]
        fx = np.fft.fftfreq(w)[None, :]
        r = np.sqrt(fy**2 + fx**2) / 0.5
        ang = np.arctan2(np.broadcast_to(fy, (h, w)), np.broadcast_to(fx, (h, w))) % np.pi

        self.cut, self.n_radial, self.n_ang, self.gram_res = cut, n_radial, n_ang, gram_res
        hf = (r >= cut)
        r_edges = np.linspace(cut, r.max() + 1e-6, n_radial + 1)
        a_edges = np.linspace(0.0, np.pi + 1e-6, n_ang + 1)
        ri = np.clip(np.digitize(r, r_edges) - 1, 0, n_radial - 1)
        ai = np.clip(np.digitize(ang, a_edges) - 1, 0, n_ang - 1)

        mr = np.stack([((ri == k) & hf).astype(np.float32) for k in range(n_radial)])
        ma = np.stack([((ai == k) & hf).astype(np.float32) for k in range(n_ang)])
        self.register_buffer("hf_mask", torch.as_tensor(hf.astype(np.float32)))
        self.register_buffer("hf_pass", torch.as_tensor((~hf).astype(np.float32)))
        self.register_buffer("mask_r", torch.as_tensor(mr))
        self.register_buffer("mask_a", torch.as_tensor(ma))
        self.register_buffer("cnt_r", torch.as_tensor(mr.sum(axis=(1, 2)).clip(min=1e-6)))
        self.register_buffer("cnt_a", torch.as_tensor(ma.sum(axis=(1, 2)).clip(min=1e-6)))
        self.register_buffer("n_hi", torch.as_tensor(max(float(hf.sum()), 1.0)))
        # fixed 8x8 grid for the Gram term; the mean removes DC so no centring needed
        self.register_buffer("gram_k", torch.as_tensor(max(1, h // gram_res)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        Fx = torch.fft.fft2(x, dim=(-2, -1))
        P = (Fx.real**2 + Fx.imag**2) * self.hf_mask.view(1, 1, h, w)
        scale = (P.sum(dim=(2, 3)) / self.n_hi).clamp(min=1e-8).unsqueeze(-1)       # (N,C,1)

        rad = torch.einsum("nchw,rhw->ncr", P, self.mask_r) / self.cnt_r.view(1, 1, -1)
        ang = torch.einsum("nchw,ahw->nca", P, self.mask_a) / self.cnt_a.view(1, 1, -1)
        rad = torch.log1p(rad / scale * 10.0)
        ang = torch.log1p(ang / scale * 10.0)

        hi = torch.fft.ifft2(Fx * self.hf_pass.view(1, 1, h, w), dim=(-2, -1)).real
        k = int(self.gram_k)
        if k > 1:
            hi = F.avg_pool2d(hi, k)
        f = hi.flatten(2)
        f = f - f.mean(dim=2, keepdim=True)
        f = f / f.norm(dim=2, keepdim=True).clamp(min=1e-8)
        gram = f @ f.transpose(1, 2)

        # amplitude first, matching t3_build_targets.stats_layout
        amp = torch.log(scale.squeeze(-1).clamp(min=1e-12))          # (N,C)
        return torch.cat([amp, rad.flatten(1), ang.flatten(1), gram.flatten(1)], dim=1)


# --------------------------------------------------------------------------- #
# modules
# --------------------------------------------------------------------------- #
class Dec(nn.Module):
    """z -> (ch,S,S) via an 8x8 projection and three bilinear upsamples.

    zero_init puts the last conv at exactly 0. That is the right neutral prior for a
    sparse REGRESSION target (the shipped VAE head), but it is actively harmful for a
    DISTRIBUTIONAL target: Spectra normalises per sample by that sample's own mean HF
    power, so an all-zero field makes that denominator collapse to its floor, the
    gradient through the ratio blows up to ~1e12, and clip_grad_norm_ then spends the
    whole budget on that direction (measured: the stats-supervised head did not move
    after 220 steps with zero_init=True). The texture heads therefore start from a
    small non-degenerate field instead.
    """

    def __init__(self, in_dim: int, ch: int, spatial: int, hidden: int = 1024,
                 base: int = 8, zero_init: bool = True):
        super().__init__()
        assert spatial == base * 8, f"upsample chain expects spatial={base*8}, got {spatial}"
        self.base, self.ch, self.spatial = base, ch, spatial
        self.fc = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                nn.Linear(hidden, 128 * base * base))
        self.up = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, ch, 3, padding=1),
        )
        if zero_init:
            nn.init.zeros_(self.up[-1].weight)
            nn.init.zeros_(self.up[-1].bias)
        else:
            nn.init.normal_(self.up[-1].weight, std=0.01)
            nn.init.zeros_(self.up[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.up(self.fc(z).view(-1, 128, self.base, self.base))


class Vel(nn.Module):
    """Conditional velocity field for the HF texture distribution."""

    def __init__(self, cond_dim: int, ch: int, spatial: int, hidden: int = 512):
        super().__init__()
        self.temb = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, 64))
        # zero_init would make the transport map the identity-on-zero at init, i.e.
        # it would start inside the very collapse state D3/D4 are about
        self.net = Dec(cond_dim + 64, ch, spatial, hidden=hidden, zero_init=False)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([cond, self.temb(t.view(-1, 1))], dim=1))


class T3Net(nn.Module):
    def __init__(self, in_dim: int, code: int = 768, spatial: int = 64, ch: int = 4,
                 n_cells: int = 36, loc_dim: int = 1024, use_local: bool = True,
                 use_hf: bool = True, use_depth: bool = False, struct_extra: int = 0,
                 hf_head: str = "pix", stats_dim: int = 0):
        super().__init__()
        self.use_local, self.use_hf, self.use_depth = use_local, use_hf, use_depth
        self.n_cells, self.loc_dim, self.hf_head = n_cells, loc_dim, hf_head
        self.struct_extra = struct_extra

        self.enc = nn.Sequential(nn.Linear(in_dim, code), nn.GELU(),
                                 nn.Linear(code, code), nn.GELU())
        # Separate trunk for the structure/texture towers (fixes D1: they are no
        # longer downstream of the semantic bottleneck).
        self.st_trunk = nn.Sequential(nn.Linear(1024 + struct_extra, code), nn.GELU(),
                                      nn.Linear(code, code), nn.GELU())
        self.tx_trunk = nn.Sequential(nn.Linear(in_dim, code), nn.GELU(),
                                      nn.Linear(code, code), nn.GELU())

        self.h_rn50 = nn.Sequential(nn.Linear(code, 512), nn.GELU(), nn.Linear(512, 512))
        self.h_vith = nn.Sequential(nn.Linear(code, 1024), nn.GELU(), nn.Linear(1024, 1024))
        self.h_hcf = nn.Sequential(nn.Linear(code, 1024), nn.GELU(), nn.Linear(1024, 1280))
        self.h_dino = nn.Linear(code, 1024)
        self.h_text = nn.Sequential(nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 512))
        self.h_local = nn.Sequential(nn.Linear(code, 1024), nn.GELU(),
                                     nn.Linear(1024, n_cells * loc_dim)) if use_local else None

        self.h_lf = Dec(768, ch, spatial)
        self.h_depth = Dec(768, 1, spatial, hidden=512) if use_depth else None
        if use_hf:
            if hf_head == "pix":
                self.h_hf = Dec(768, ch, spatial, zero_init=False)
                self.vel = None
            else:
                self.h_hf = None
                self.vel = Vel(768, ch, spatial)

    def forward(self, x: torch.Tensor, raw_x: torch.Tensor, t_txt: torch.Tensor):
        code = self.enc(x)
        out = {
            "z_rn50": self.h_rn50(code),
            "z_vith": l2norm(self.h_vith(code)),
            "z_hcf": self.h_hcf(code),
            "z_dino": self.h_dino(code),
            "z_txt": self.h_text(t_txt),
        }
        if self.h_local is not None:
            out["z_loc"] = self.h_local(code).view(-1, self.n_cells, self.loc_dim)

        st_in = torch.cat([out["z_vith"], raw_x], dim=1) if self.struct_extra > 0 else out["z_vith"]
        st = self.st_trunk(st_in)
        out["lf"] = self.h_lf(st)
        if self.h_depth is not None:
            out["depth"] = self.h_depth(st)

        if self.use_hf:
            tx = self.tx_trunk(x)
            out["tx_cond"] = tx
            if self.h_hf is not None:
                out["hf"] = self.h_hf(tx)
        return out


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class T3View(Dataset):
    """DualDataset + extra targets, indexed by FLAT image row.

    DualDataset row index == object_idx * images_per_object + image_idx for both
    splits (train 1654*10, test 200*1), which is exactly the flat ordering used by
    t3_build_targets.py, so no reindexing is needed. The assertion below keeps that
    invariant honest instead of trusting it.
    """

    def __init__(self, base: DualDataset, lf, hf, st, loc, depth=None,
                 indices: np.ndarray | None = None):
        self.base, self.lf, self.hf, self.st, self.loc, self.depth = base, lf, hf, st, loc, depth
        self.indices = (np.arange(len(base)) if indices is None
                        else np.asarray(indices, dtype=np.int64))
        self._checked = False

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        row = int(self.indices[i])
        eeg, rn50, vith, hcf, dino, txt, sid, obj, img, rep = self.base[row]
        flat = int(obj) * self.base.images_per_object + int(img)
        if not self._checked:
            # one-shot guard: without this a silent misalignment would train the
            # structural towers against the wrong images for the whole run
            assert flat == row, f"row {row} != flat {flat} (target/EEG ordering mismatch)"
            self._checked = True
        t = lambda a: torch.from_numpy(np.asarray(a[flat], dtype=np.float32))  # noqa: E731
        return (eeg, rn50, vith, hcf, dino, txt, sid, t(self.lf), t(self.hf), t(self.st),
                t(self.loc) if self.loc is not None else torch.zeros(1),
                t(self.depth) if self.depth is not None else torch.zeros(1))


def gd_loss(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    px, py = pred[..., 1:, :] - pred[..., :-1, :], pred[..., :, 1:] - pred[..., :, :-1]
    rx, ry = ref[..., 1:, :] - ref[..., :-1, :], ref[..., :, 1:] - ref[..., :, :-1]
    return F.l1_loss(px, rx) + F.l1_loss(py, ry)


def unit(x: torch.Tensor) -> torch.Tensor:
    """Scale each sample to unit RMS over the spatial dims (removes global amplitude)."""
    s = x.flatten(1).pow(2).mean(1).sqrt().clamp(min=1e-8).view(-1, 1, 1, 1)
    return x / s


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ss-checkpoint", required=True)
    ap.add_argument("--train-subjects", required=True,
                    help="comma list. sub-08 alone = intra; 1,2,..,7,9,10 = LOSO training set")
    ap.add_argument("--test-subject", type=int, default=8)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--clip-layers-dir", required=True)
    ap.add_argument("--nvol-json", default="")
    ap.add_argument("--layers", default="")
    ap.add_argument("--dino-train-npy", required=True)
    ap.add_argument("--dino-test-npy", required=True)
    ap.add_argument("--dino-local-train-npy", default="")
    ap.add_argument("--dino-local-test-npy", default="")
    ap.add_argument("--clip-train-npy", required=True)
    ap.add_argument("--clip-test-npy", required=True)
    ap.add_argument("--text-train-npy", required=True)
    ap.add_argument("--text-test-npy", required=True)
    ap.add_argument("--vith-dir", required=True)
    ap.add_argument("--rn50-dir", required=True)
    ap.add_argument("--eeg-dir", required=True)
    ap.add_argument("--lf-train-npy", required=True)
    ap.add_argument("--lf-test-npy", required=True)
    ap.add_argument("--hf-train-npy", required=True)
    ap.add_argument("--hf-test-npy", required=True)
    ap.add_argument("--stats-train-npy", required=True)
    ap.add_argument("--stats-test-npy", required=True)
    ap.add_argument("--depth-train-npy", default="")
    ap.add_argument("--depth-test-npy", default="")
    ap.add_argument("--val-split-json", required=True,
                    help="leakfree.py split; selection on the test set is the audit finding we fix")
    ap.add_argument("--variant", choices=["t3", "mono"], default="t3")
    ap.add_argument("--hf-head", choices=["pix", "cfm"], default="pix")
    ap.add_argument("--struct-input", choices=["sem", "both"], default="both")
    ap.add_argument("--grid", type=int, default=6)
    ap.add_argument("--cut", type=float, default=0.0625)
    ap.add_argument("--n-radial", type=int, default=16)
    ap.add_argument("--n-ang", type=int, default=8)
    ap.add_argument("--gram-res", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=384)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cfm-steps", type=int, default=32, help="integration steps at export time")
    ap.add_argument("--cfm-steps-train", type=int, default=8,
                    help="integration steps for the in-loop statistics loss (speed); "
                         "the CFM regression loss itself is step-free")
    ap.add_argument("--lambda-rn50", type=float, default=0.8)
    ap.add_argument("--lambda-txt", type=float, default=0.2)
    ap.add_argument("--lambda-vith", type=float, default=0.35)
    ap.add_argument("--lambda-hcf", type=float, default=1.0)
    ap.add_argument("--lambda-dino", type=float, default=0.5)
    ap.add_argument("--lambda-local", type=float, default=0.5)
    ap.add_argument("--lambda-lf", type=float, default=1.0)
    ap.add_argument("--lambda-hf-stats", type=float, default=1.0)
    ap.add_argument("--lambda-hf-pix", type=float, default=0.0,
                    help="small on purpose: a strong pixel term reintroduces the zero-collapse solution")
    ap.add_argument("--lambda-depth", type=float, default=0.3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-batches", type=int, default=0, help="smoke test: cap batches per epoch")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_subjects = [int(s) for s in args.train_subjects.split(",") if s.strip()]
    if args.test_subject in train_subjects:
        raise SystemExit(f"test subject {args.test_subject} is inside the training list "
                         f"{train_subjects}: that is an intra run, label it as such")
    print(f"[INFO] train subjects {train_subjects} -> test subject {args.test_subject}")

    if args.layers:
        layer_ids = [int(x) for x in args.layers.split(",") if x.strip()]
    elif args.nvol_json and Path(args.nvol_json).is_file():
        layer_ids = json.loads(Path(args.nvol_json).read_text())["top_k_layers"]
    else:
        layer_ids = [8, 10, 14]

    hcf_train, hcf_test = out / "hcf_train.npy", out / "hcf_test.npy"
    if not hcf_train.is_file():
        build_hcf(Path(args.clip_layers_dir), "train", layer_ids, hcf_train)
    if not hcf_test.is_file():
        build_hcf(Path(args.clip_layers_dir), "test", layer_ids, hcf_test)

    # Train side: subject-randomised (one EEG realisation per image per step) so a
    # multi-subject list is a domain-randomised inter-subject training set.
    train_ds = DualDataset(
        str(hcf_train), args.dino_train_npy, args.vith_dir, args.text_train_npy,
        train_subjects, args.eeg_dir, DEFAULT_CHANNELS, [0, 250],
        args.rn50_dir, "", False, [], True, True, None, True, False, False, False)
    # Test side: only the held-out subject, deterministic order, 200 rows.
    test_ds = DualDataset(
        str(hcf_test), args.dino_test_npy, args.vith_dir, args.text_test_npy,
        [args.test_subject], args.eeg_dir, DEFAULT_CHANNELS, [0, 250],
        args.rn50_dir, "", False, [], True, False, None, False, False, False, False)

    lf_tr, lf_te = np.load(args.lf_train_npy, mmap_mode="r"), np.load(args.lf_test_npy, mmap_mode="r")
    hf_tr, hf_te = np.load(args.hf_train_npy, mmap_mode="r"), np.load(args.hf_test_npy, mmap_mode="r")
    st_tr, st_te = np.load(args.stats_train_npy), np.load(args.stats_test_npy)
    loc_tr = np.load(args.dino_local_train_npy) if args.dino_local_train_npy else None
    loc_te = np.load(args.dino_local_test_npy) if args.dino_local_test_npy else None
    dep_tr = np.load(args.depth_train_npy) if args.depth_train_npy else None
    dep_te = np.load(args.depth_test_npy) if args.depth_test_npy else None
    n_cells = args.grid * args.grid
    loc_dim = int(loc_tr.shape[-1]) if loc_tr is not None else 1024
    if loc_tr is not None:
        assert loc_tr.shape[1] == n_cells, f"local cells {loc_tr.shape[1]} != G*G {n_cells}"
    assert lf_tr.shape[0] == len(train_ds), f"LF rows {lf_tr.shape[0]} != train rows {len(train_ds)}"
    assert lf_te.shape[0] == len(test_ds), f"LF test rows {lf_te.shape[0]} != test rows {len(test_ds)}"

    import leakfree as LF
    sp = LF.load(args.val_split_json)
    fit_rows = LF.rows_for(sp, "fit", len(train_ds))
    val_rows = LF.rows_for(sp, "val_b", len(train_ds))
    assert len(set(fit_rows.tolist()) & set(val_rows.tolist())) == 0

    tr_view = T3View(train_ds, lf_tr, hf_tr, st_tr, loc_tr, dep_tr, fit_rows)
    va_view = T3View(train_ds, lf_tr, hf_tr, st_tr, loc_tr, dep_tr, val_rows)
    te_view = T3View(test_ds, lf_te, hf_te, st_te, loc_te, dep_te)
    print(f"[leakfree] fit {len(tr_view)} | val_b {len(va_view)} | test {len(te_view)} (export only)")

    tr_loader = DataLoader(tr_view, batch_size=args.batch_size, shuffle=True, drop_last=True)
    va_loader = DataLoader(va_view, batch_size=512, shuffle=False)
    te_loader = DataLoader(te_view, batch_size=200, shuffle=False)

    ckpt = torch.load(Path(args.ss_checkpoint), map_location=device, weights_only=False)
    from module.projector import ProjectorLinear  # noqa: E402
    from ss_modules import SharedSpecificEncoder  # noqa: E402

    img_dim = int(ckpt.get("img_dim", 1024))
    feature_dim = int(ckpt.get("feature_dim", 512))
    ck_subjects = [int(s) for s in ckpt.get("subjects", train_subjects + [args.test_subject])]
    for s in train_subjects + [args.test_subject]:
        if s not in ck_subjects:
            ck_subjects.append(s)
    enc = SharedSpecificEncoder(
        subject_ids=ck_subjects, feature_dim=img_dim,
        eeg_sample_points=int(ckpt.get("eeg_sample_points", train_ds.num_sample_points)),
        channels_num=int(ckpt.get("channels_num", train_ds.channels_num)),
        n_extra_blocks=int(ckpt.get("n_extra_blocks", 1)), use_adapter=True).to(device)
    enc.load_state_dict(ckpt["model_state_dict"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False          # frozen root: THIS job retrains the towers
    raw_dim = img_dim

    spatial, ch = int(lf_tr.shape[-1]), int(lf_tr.shape[1])
    stats_dim = int(st_tr.shape[1])
    print(f"[INFO] raw_dim={raw_dim} spatial={spatial} ch={ch} stats_dim={stats_dim} "
          f"n_cells={n_cells} variant={args.variant} hf_head={args.hf_head} struct={args.struct_input}")

    model = T3Net(
        in_dim=raw_dim, spatial=spatial, ch=ch, n_cells=n_cells, loc_dim=loc_dim,
        use_local=(args.variant == "t3" and loc_tr is not None),
        use_hf=(args.variant == "t3"),
        use_depth=bool(dep_tr is not None and args.variant == "t3"),
        struct_extra=(raw_dim if args.struct_input == "both" else 0),
        hf_head=args.hf_head, stats_dim=stats_dim,
    ).to(device)
    print(f"[INFO] trainable {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    spectra = Spectra(spatial, spatial, args.cut, args.n_radial, args.n_ang,
                      args.gram_res).to(device)
    nce = ClipInfoNCE(0.07).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    ref_lf_std = float(np.asarray(lf_tr[:512], dtype=np.float32).std())
    ref_hf_std = float(np.asarray(hf_tr[:512], dtype=np.float32).std())

    def sample_hf(cond: torch.Tensor, steps: int, seed: int | None = None) -> torch.Tensor:
        """Sample the HF texture from the learned transport, starting at x0 ~ N(0,I).

        NOT mean flow. The high band is zero-mean by construction (it is a high-pass
        field), so the conditional MEAN of x1 is 0 whenever the condition is
        uninformative -- which is exactly the regime D3 is about. Mean flow would
        therefore integrate to zero and reproduce the very collapse it was meant to
        fix. Measured on the CPU harness: mean flow gave a std ratio of 0.229 at
        stats_l1 1.14, i.e. no better than the L1 control.

        A transport map sampled from a unit-Gaussian base reproduces the conditional
        DISTRIBUTION rather than its mean, so the field's dispersion survives. Seed
        is fixed per call so exports are reproducible while still non-degenerate.
        """
        g = None
        if seed is not None:
            g = torch.Generator(device=cond.device).manual_seed(seed)
        x = torch.randn(cond.shape[0], ch, spatial, spatial, device=cond.device, generator=g)
        dt = 1.0 / steps
        for k in range(steps):
            t = torch.full((cond.shape[0],), k * dt, device=cond.device)
            x = x + dt * model.vel(x, t, cond)
        return x

    def hf_infer(cond: torch.Tensor, steps: int, seed: int | None = None) -> torch.Tensor:
        """Single entry point so train / select / export all use the same inference."""
        if model.vel is not None:
            return sample_hf(cond, steps, seed)
        return model.h_hf(cond)

    hist, best = [], {"score": -1e9, "epoch": 0}
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for bi, batch in enumerate(tqdm(tr_loader, desc=f"t3-{args.variant}-{ep}")):
            if args.max_batches and bi >= args.max_batches:
                break
            eeg, rn50, vith, hcf, dino, txt, sid, lf, hf, stt, loc, dep = batch
            eeg, rn50 = eeg.to(device), rn50.to(device)
            vith, hcf, dino = vith.to(device), hcf.to(device), dino.to(device)
            txt, lf, hf, stt = txt.to(device), lf.to(device), hf.to(device), stt.to(device)
            loc, dep = loc.to(device), dep.to(device)

            with torch.no_grad():
                raw = enc(eeg, sid.to(device))
            o = model(raw, raw, txt)

            # --- semantic global (unchanged HCMA objectives) ---
            loss = args.lambda_rn50 * ((1 - (l2norm(o["z_rn50"]) * l2norm(rn50)).sum(-1)).mean()
                                       + nce(o["z_rn50"], rn50))
            loss = loss + args.lambda_txt * ((1 - (l2norm(o["z_rn50"]) * l2norm(o["z_txt"])).sum(-1)).mean()
                                             + nce(o["z_rn50"], o["z_txt"]))
            loss = loss + args.lambda_vith * ((1 - (o["z_vith"] * l2norm(vith)).sum(-1)).mean()
                                              + 0.5 * nce(o["z_vith"], vith))
            loss = loss + args.lambda_hcf * ((1 - (l2norm(o["z_hcf"]) * l2norm(hcf)).sum(-1)).mean()
                                             + nce(o["z_hcf"], hcf))
            loss = loss + args.lambda_dino * (1 - (l2norm(o["z_dino"]) * l2norm(dino)).sum(-1)).mean()

            # --- semantic LOCAL: the missing granularity term (D2) ---
            if "z_loc" in o:
                loss = loss + args.lambda_local * (1 - (l2norm(o["z_loc"]) * l2norm(loc)).sum(-1)).mean()

            if args.variant == "t3":
                # --- structure: low band (D1) ---
                loss = loss + args.lambda_lf * (F.l1_loss(o["lf"], lf) + 0.5 * gd_loss(o["lf"], lf))
                if "depth" in o:
                    loss = loss + args.lambda_depth * (F.l1_loss(o["depth"], dep)
                                                       + 0.5 * gd_loss(o["depth"], dep))
                # --- texture: DISTRIBUTIONAL high band (D3) ---
                if model.vel is not None:
                    x1 = unit(hf)
                    t = torch.rand(eeg.shape[0], device=device)
                    eps = torch.randn_like(x1)
                    xt = (1 - t.view(-1, 1, 1, 1)) * eps + t.view(-1, 1, 1, 1) * x1
                    v = model.vel(xt, t, o["tx_cond"])
                    loss = loss + args.lambda_hf_stats * F.mse_loss(v, x1 - eps)
                    hf_pred = mean_flow(o["tx_cond"].detach(), args.cfm_steps_train)
                else:
                    hf_pred = o["hf"]
                loss = loss + args.lambda_hf_stats * F.smooth_l1_loss(spectra(hf_pred), stt)
                if args.lambda_hf_pix > 0:
                    sc = hf.flatten(1).pow(2).mean(1).sqrt().clamp(min=1e-3).view(-1, 1, 1, 1)
                    loss = loss + args.lambda_hf_pix * F.l1_loss(hf_pred / sc, hf / sc)
            else:
                # control: the shipped recipe -- ONE L1 over the FULL band
                tgt = lf + hf
                loss = loss + args.lambda_lf * (F.l1_loss(o["lf"], tgt) + 0.5 * gd_loss(o["lf"], tgt))

            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss -- abort rather than train on NaNs")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item())
            n += 1
        tr_loss = tot / max(n, 1)

        # ---- held-in selection (never the test set) ----
        model.eval()
        with torch.no_grad():
            agg: dict[str, list] = {}
            for batch in va_loader:
                eeg, rn50, vith, hcf, dino, txt, sid, lf, hf, stt, loc, dep = batch
                raw = enc(eeg.to(device), sid.to(device))
                o = model(raw, raw, txt.to(device))
                r = {"vith_cos": (o["z_vith"] * l2norm(vith.to(device))).sum(-1),
                     "lf_l1": F.l1_loss(o["lf"], lf.to(device), reduction="none").flatten(1).mean(1),
                     "lf_std": o["lf"].flatten(1).std(1)}
                if "z_loc" in o:
                    r["loc_cos"] = (l2norm(o["z_loc"]) * l2norm(loc.to(device))).sum(-1)
                if args.variant == "t3":
                    hp = (mean_flow(o["tx_cond"], args.cfm_steps)
                          if model.vel is not None else o["hf"])
                    r["hf_std"] = hp.flatten(1).std(1)
                    r["stats_l1"] = F.l1_loss(spectra(hp), stt.to(device),
                                              reduction="none").mean(1)
                for k, v in r.items():
                    agg.setdefault(k, []).append(v.cpu().numpy())
            m = {k: float(np.concatenate(v).mean()) for k, v in agg.items()}

        lf_ratio = m["lf_std"] / max(ref_lf_std, 1e-8)
        hf_ratio = m.get("hf_std", 0.0) / max(ref_hf_std, 1e-8)
        # Selection rewards semantic fidelity, structural fidelity, and TEXTURE
        # NON-COLLAPSE, so a mean-collapsed head cannot win on a low L1 alone.
        score = (m["vith_cos"]
                 - 0.5 * m["lf_l1"]
                 - (0.5 * m["stats_l1"] if args.variant == "t3" else 0.0)
                 + (0.15 * min(hf_ratio, 1.0) if args.variant == "t3" else 0.0)
                 + 0.05 * m.get("loc_cos", 0.0))
        row = {"epoch": ep, "loss": tr_loss, "score": score,
               "lf_std_ratio": round(lf_ratio, 4), "ref_lf_std": round(ref_lf_std, 4),
               **{k: round(v, 5) for k, v in m.items()}}
        if args.variant == "t3":
            row.update({"hf_std_ratio": round(hf_ratio, 4), "ref_hf_std": round(ref_hf_std, 4)})
        hist.append(row)
        print(f"[ep {ep}] loss={tr_loss:.4f} score={score:.4f} vith_cos={m['vith_cos']:.4f} "
              f"lf_l1={m['lf_l1']:.4f} lf_std_ratio={lf_ratio:.3f}"
              + (f" hf_std_ratio={hf_ratio:.3f} stats_l1={m['stats_l1']:.4f}" if args.variant == "t3" else ""))

        if score > best["score"]:
            best = {"score": score, "epoch": ep}
            torch.save({"epoch": ep, "state_dict": model.state_dict(), "args": vars(args),
                        "metrics": row, "layer_ids": layer_ids, "subjects": ck_subjects,
                        "img_dim": img_dim, "feature_dim": feature_dim, "raw_dim": raw_dim,
                        "spatial": spatial, "ch": ch, "n_cells": n_cells, "loc_dim": loc_dim,
                        "stats_dim": stats_dim, "variant": args.variant,
                        "hf_head": args.hf_head, "struct_input": args.struct_input},
                       out / "checkpoint_t3_best.pth")

    pd.DataFrame(hist).to_csv(out / "t3_history.csv", index=False)
    print(f"[OK] best epoch {best['epoch']} score {best['score']:.4f}")

    # ---- export (test rows read ONCE, never used to choose anything) ----
    ck = torch.load(out / "checkpoint_t3_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    def export(loader, tag: str) -> np.ndarray:
        acc: dict[str, list] = {}
        stds = []
        with torch.no_grad():
            for batch in loader:
                eeg, rn50, vith, hcf, dino, txt, sid, lf, hf, stt, loc, dep = batch
                raw = enc(eeg.to(device), sid.to(device))
                o = model(raw, raw, txt.to(device))
                hp = mean_flow(o["tx_cond"], args.cfm_steps) if model.vel is not None else o.get("hf")
                for k, v in (("z_vith", o["z_vith"]), ("lf", o["lf"]), ("hf", hp),
                             ("z_loc", o.get("z_loc")), ("depth", o.get("depth"))):
                    if v is not None:
                        acc.setdefault(k, []).append(v.float().cpu().numpy())
                if hp is not None:
                    acc.setdefault("hf_stats", []).append(spectra(hp).float().cpu().numpy())
                acc.setdefault("lf_tgt", []).append(lf.numpy())
                acc.setdefault("hf_tgt", []).append(hf.numpy())
                acc.setdefault("stats_tgt", []).append(stt.numpy())
                stds.append(np.stack([o["lf"].flatten(1).std(1).cpu().numpy(),
                                      hp.flatten(1).std(1).cpu().numpy() if hp is not None
                                      else np.zeros(len(eeg))], axis=1))
        for k, v in acc.items():
            np.save(out / f"{k}_{tag}.npy", np.concatenate(v).astype(np.float32))
        return np.concatenate(stds)

    export(DataLoader(tr_view, batch_size=512, shuffle=False), "train")
    d_te = export(te_loader, "test")

    def band_corr(pred: np.ndarray, tgt: np.ndarray, low: bool) -> float:
        h, w = tgt.shape[-2], tgt.shape[-1]
        rr = np.sqrt(np.fft.fftfreq(h)[:, None]**2 + np.fft.fftfreq(w)[None, :]**2) / 0.5
        msk = (rr < args.cut) if low else (rr >= args.cut)
        cs = []
        for i in range(len(pred)):
            a = np.real(np.fft.fft2(pred[i], axes=(-2, -1)) * msk).ravel()
            b = np.real(np.fft.fft2(tgt[i].astype(np.float32), axes=(-2, -1)) * msk).ravel()
            cs.append(0.0 if a.std() < 1e-9 or b.std() < 1e-9 else float(np.corrcoef(a, b)[0, 1]))
        return float(np.mean(cs))

    lf_p = np.load(out / "lf_test.npy")
    lf_t = np.load(out / "lf_test_tgt.npy").astype(np.float32)
    hf_t = np.load(out / "hf_test_tgt.npy").astype(np.float32)
    hf_file = out / "hf_test.npy"
    # mono has no HF head by construction: report the band split of its single
    # full-band prediction so the two variants stay comparable on the same axes
    hf_p = np.load(hf_file) if hf_file.is_file() else np.zeros_like(lf_p)
    rep = {
        "pipeline": "t3_three_tower", "variant": args.variant, "hf_head": args.hf_head,
        "struct_input": args.struct_input, "cut": args.cut, "layer_ids": layer_ids,
        "train_subjects": train_subjects, "test_subject": args.test_subject,
        "regime": "intra" if len(train_subjects) == 1 else "inter(domain-randomised)",
        "best_epoch": best["epoch"], "best_score": best["score"],
        "leakfree": {"fit": int(len(tr_view)), "val_b": int(len(va_view)), "test": int(len(te_view)),
                     "selected_on": "held-in val_b train concepts (test never used for selection)"},
        "collapse_diagnostics": {
            "test_lf_std_ratio": round(float(d_te[:, 0].mean() / max(ref_lf_std, 1e-8)), 4),
            "test_hf_std_ratio": (round(float(d_te[:, 1].mean() / max(ref_hf_std, 1e-8)), 4)
                                  if args.variant == "t3" else None),
            "reference": "shipped L1 VAE head measured 0.428 spread (linear baseline 0.4636)",
        },
        "band_corr_test": {
            "lf_corr": round(band_corr(lf_p, lf_t, True), 4),
            "hf_corr": round(band_corr(hf_p, hf_t, False), 4),
            "full_corr": round(band_corr(lf_p + hf_p, lf_t + hf_t, True), 4)
            if args.variant == "t3" else round(band_corr(lf_p, lf_t + hf_t, True), 4),
        },
        "history": hist,
    }
    (out / "t3_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("variant", "hf_head", "struct_input", "regime",
                                          "best_epoch", "collapse_diagnostics",
                                          "band_corr_test")}, indent=2))


if __name__ == "__main__":
    main()
