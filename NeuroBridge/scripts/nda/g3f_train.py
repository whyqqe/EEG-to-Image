#!/usr/bin/env python3
"""G3F: leak-free dual tower with CENTERED DISCRIMINATIVE READOUT and a
SELF-PROMPT that is structurally incapable of leaking the test class name.

WHAT CHANGED VS g2f_train.py, AND WHY (every change is tied to a measurement)
---------------------------------------------------------------------------
A. DELETE the CFM conditioner (cond_proj / vel / sample_ip / cfm_loss).
   Three independent disconfirmations:
     * training-side: the sampled output had spread 0.9911 (near-constant) and
       top-1 0.015, i.e. WORSE than its own input (z_s_f 0.160);
     * mechanism: the training target is the MEAN flow field (x1 - x0), while
       inference integrates a sample. Under one-to-many conditioning those two
       disagree by construction, so sampling returns the conditional mean and
       the conditional spread collapses;
     * `ode_mix` never entered the training graph at all -- an untrainable knob.
     * image-side (g2f run, sub-08, identical prompt/anchor/seed):
         g2f_cfm   CLIP 0.600  Alex2 0.581  FID 223.48   <- worst row of the run
         g2f_mem   CLIP 0.784  Alex2 0.733  FID 178.44
   Its job (synthesising the IP condition) is done better by h_fuse under a
   discriminative loss.

B. DELETE h_texture and its target (17-d spectral statistics of the VAE latent).
   Band-ceiling measurement (outputs/band_probe_ceiling.json, ridge, scored once
   on test, lambda chosen on held-in train):
       image-latent band   energy_frac   test_corr   var_expl
       0.00-0.0625             0.3639      0.1332     0.006452
       0.0625-0.125            0.1067      0.0426     0.000194
       0.125-0.25              0.0836      0.0219     0.000040
       0.25-0.5                0.1009      0.0442     0.000197
       0.5-2.0                 0.3450      0.0134     0.000062
       fine/coarse = 0.0765, total_var_expl = 0.00695
   92.9% of the explainable variance sits in the lowest band. The texture head
   was regressing an UNIDENTIFIABLE component, so its gradient could only add
   noise to the shared trunk.

C. DELETE h_ip (the "direct" head) and make h_fuse the SOLE IP producer.
   Measured at training time on g2f:
       head      cos-to-constant-target   constant baseline   2-way
       h_ip            0.613-0.634            0.6147         0.640-0.710
       h_fuse          0.237-0.314              --           0.870-0.955
   A pure L2/cosine head under one-to-many conditioning has the conditional MEAN
   as its optimum, and because CLIP image/IP embeddings share a strong common
   direction (a CONSTANT train centroid scores 0.6147, random pairs 0.378),
   predicting the mean is nearly optimal for the loss while carrying no
   instance information. h_ip was not failing; it was correctly solving a
   useless problem, and its gradient kept pulling the trunk toward the mean --
   the opposite direction from discriminability. Keeping both heads meant the
   generated IP condition was contaminated by that mean component.

D. CENTERED RESIDUAL TARGETS. Each head now predicts
       delta_hat = h(x),   full = l2( mu_k + delta_hat )
   with mu_k the TRAIN-split mean of target k. The regression term acts on the
   residual, so capacity is not spent re-learning the common direction that eats
   the whole cosine budget, while all discriminative losses act on the FULL
   vector. A two-sided var_band keeps the residual from collapsing back to zero
   (a one-sided floor alone let g2_train's structural head drift to 6.1x the
   target dispersion, uncorrectable by a scale-invariant cosine loss).

E. InfoNCE on h_image as well. h_image regressed OpenCLIP-1280 under a pure
   cosine loss, i.e. exactly the mechanism that collapsed h_ip. It is now scored
   discriminatively in its own space. NOTE: this is a fix we can measure, not an
   assumption -- h_image was not separately instrumented in the g2f run.

F. SELF-PROMPT (the innovation). HCMA's prompt was the GROUND-TRUTH test
   concept name for 200/200 rows, which no EEG-derived model can produce: the
   200 test concepts are disjoint from the train vocabulary (verified --
   intersection is empty). g2f measured the size of that leak directly, holding
   prompt as the only variable on sub-08:
       hcma_deploy (generic prompt)  CLIP 0.736  Inception 0.675  FID 179.98
       hcma_oracle (GT class name)   CLIP 0.948  Inception 0.909  FID 149.22
       delta                        +21.2pp      +23.4pp        -30.76
   and, from the low-level split,
       hcma_deploy -> g2f_mem        CLIP +6.3pp   PixCorr +0.000   FID -7.8
       hcma_oracle (same prompt) -> sdedit_ll_sub08 (RGB init)
                                     CLIP -0.3pp   PixCorr +0.091   FID -3.4
   i.e. PROMPT buys high-level semantics and FID; the LOW-LEVEL PATHWAY buys
   PixCorr/SSIM. The two are separable, and this run tests them separately.

   G3F replaces the oracle with a prompt retrieved from a gallery of the 1654
   TRAIN concepts, indexed by an EEG-predicted concept distribution:
       logits = h_concept(z) @ train_concept_bank.T        (1654-way)
       name   = concept_phrases[argmax logits]
       prompt = "a photo of a {name}"
   LEAK-FREE BY CONSTRUCTION: the gallery contains only train concepts, and the
   train/test concept intersection is empty, so the emitted string cannot name a
   test class even if the EEG prediction is wrong. A margin gate falls back to
   the generic prompt when the top-1/top-2 margin is below a threshold set on
   TRAIN-val rows only.

HARD LEAK-FREE GUARANTEES
-------------------------
  * test EEG is never read during training or checkpoint selection;
  * the memory bank, the concept gallery and the prompt gallery are built ONLY
    from train images / train concepts / train captions;
  * checkpoint selection uses a held-out slice of TRAIN target indices;
  * prompt construction reads `text_concept_clip.npy` (train concepts) and
    `concept_phrases.json` only -- never `sem_concept_tmpl_test.npy`, which IS
    the oracle target and is used solely as a diagnostic when asked for.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NB_ROOT = Path(__file__).resolve().parents[2]

GRANS = ("overall", "subject", "background", "detail")
GENERIC_PROMPT = ("a photo of an object, clearly showing its shape, color, and "
                  "distinctive parts, natural lighting")


# ---------------------------------------------------------------- utils

def l2t(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-8)


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def mlp(i: int, h: int, o: int, layers: int = 2, drop: float = 0.0) -> nn.Sequential:
    mods: list[nn.Module] = [nn.Linear(i, h), nn.GELU()]
    if drop > 0:
        mods.append(nn.Dropout(drop))
    for _ in range(layers - 1):
        mods += [nn.Linear(h, h), nn.GELU()]
        if drop > 0:
            mods.append(nn.Dropout(drop))
    mods.append(nn.Linear(h, o))
    return nn.Sequential(*mods)


def build_concept_bank(clip_text_dir: Path, captions_jsonl: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Train-only concept gallery + per-train-row labels, alignment VERIFIED.

    `concept_phrases.json` and `text_concept_clip.npy` share row order (both
    1654 train concepts). Row labels come from the image directory names of the
    train caption file, so the mapping is checked rather than assumed: every
    train directory must convert to a phrase present in the gallery.
    """
    phrases = json.loads((clip_text_dir / "train" / "concept_phrases.json").read_text(encoding="utf-8"))
    bank = l2n(np.load(clip_text_dir / "train" / "text_concept_clip.npy").astype(np.float32))
    if bank.shape[0] != len(phrases):
        raise SystemExit(f"[FATAL] concept bank {bank.shape} vs phrases {len(phrases)}")
    index = {c: i for i, c in enumerate(phrases)}

    dirs: list[str] = []
    for line in captions_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = json.loads(line)["path"]
        dirs.append(p.rsplit("/", 1)[0].rsplit("/", 1)[-1])
    # "00001_aardvark" -> "aardvark"; verified: 0 test concepts land in the vocab
    concepts = [d.split("_", 1)[1].replace("_", " ") for d in dirs]
    missing = sorted({c for c in concepts if c not in index})
    if missing:
        raise SystemExit(f"[FATAL] {len(missing)} train concepts absent from the gallery, e.g. {missing[:5]}")
    labels = np.asarray([index[c] for c in concepts], dtype=np.int64)
    return bank, labels, phrases


class G3FNet(nn.Module):
    """Two towers, centered discriminative readout, single IP producer.

    No CFM, no texture head, no direct IP head (see the module docstring).
    """

    def __init__(self, in_dim: int = 512, code: int = 768, img_dim: int = 1280,
                 txt_dim: int = 1024, ip_dim: int = 1024, ch: int = 4,
                 spatial: int = 64, per_layers: int = 3, drop: float = 0.15):
        super().__init__()
        self.ch, self.spatial, self.ip_dim = ch, spatial, ip_dim

        self.sem_trunk = mlp(in_dim, code, code, 2, drop)
        self.per_trunk = mlp(in_dim, code, code, per_layers, drop)

        # semantic tower: residual heads over a train-mean base (see docstring D)
        self.h_image = mlp(code, code, img_dim, 2, drop)
        for g in GRANS:
            setattr(self, f"h_{g}", mlp(code, code, txt_dim, 1, drop))
        # dedicated concept head: single source for the CE classifier AND the
        # self-prompt retrieval (docstring F)
        self.h_concept = mlp(code, code, txt_dim, 2, drop)
        # the ONLY IP producer; ingests trunk + all granularity predictions
        self.h_fuse = mlp(code + 4 * txt_dim + txt_dim + img_dim, code, ip_dim, 2, drop)

        # perceptual tower: low-frequency latent only
        self.h_struct = mlp(code, code, ch * spatial * spatial, 2, drop)

        # train-split means; filled by set_means, kept as buffers so they move
        # with .to(dev) and are saved/restored with the state dict
        self.register_buffer("mu_image", torch.zeros(img_dim), persistent=True)
        for g in GRANS:
            self.register_buffer(f"mu_{g}", torch.zeros(txt_dim), persistent=True)
        self.register_buffer("mu_concept", torch.zeros(txt_dim), persistent=True)
        self.register_buffer("mu_ip", torch.zeros(ip_dim), persistent=True)

        # Zero-init the output layer of every residual head, so training starts
        # exactly at `full = l2(mu)` and residuals grow only as the discriminative
        # losses demand. Without this the untrained residuals are O(1) random and
        # l2(mu + res) swings to a random direction on step 0, which wastes the
        # first epochs and can park the trunk in a bad basin.
        for head in [self.h_image, self.h_concept] + [getattr(self, f"h_{g}") for g in GRANS]:
            last = [m for m in head.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def set_means(self, mu: dict[str, np.ndarray]) -> None:
        with torch.no_grad():
            self.mu_image.copy_(torch.from_numpy(mu["image"]))
            for g in GRANS:
                getattr(self, f"mu_{g}").copy_(torch.from_numpy(mu[g]))
            self.mu_concept.copy_(torch.from_numpy(mu["concept"]))
            self.mu_ip.copy_(torch.from_numpy(mu["ip"]))

    def sem_heads(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        s = self.sem_trunk(x)
        res = {g: getattr(self, f"h_{g}")(s) for g in GRANS}
        res_img = self.h_image(s)
        res_con = self.h_concept(s)
        # full vectors = l2(mean + residual); every loss acts on these
        full = {g: l2t(self.mu_dict[g] + res[g]) for g in GRANS}
        f_img = l2t(self.mu_image + res_img)
        f_con = l2t(self.mu_concept + res_con)
        fuse = l2t(self.mu_ip + self.h_fuse(torch.cat(
            [s] + [full[g] for g in GRANS] + [f_con, f_img], -1)))
        return {"image": f_img, "concept": f_con, "fused": fuse,
                "_res": res, "_res_image": res_img, "_res_concept": res_con,
                "_s": s, **full}

    def per_heads(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        p = self.per_trunk(x)
        return {"struct": self.h_struct(p).view(-1, self.ch, self.spatial, self.spatial),
                "_p": p}

    # differentiable top-k soft memory (unchanged from g2f; see its docstring)
    def memory(self, q: torch.Tensor, bank: torch.Tensor, tau: float, k: int = 16) -> torch.Tensor:
        sim = l2t(q) @ l2t(bank).T
        topv, topi = sim.topk(min(k, sim.shape[1]), dim=-1)
        return l2t((torch.softmax(topv / tau, dim=-1).unsqueeze(-1) * l2t(bank)[topi]).sum(1))

    @property
    def mu_dict(self) -> dict[str, torch.Tensor]:
        return {g: getattr(self, f"mu_{g}") for g in GRANS}


def var_band(pred: torch.Tensor, target: torch.Tensor, lo: float = 0.4,
             hi: float = 2.0) -> torch.Tensor:
    """Two-sided anti-degeneracy, band derived from the target's own dispersion."""
    tn = target.std(0).norm()
    pn = pred.std(0).norm()
    return (F.relu(lo * tn - pn) + F.relu(pn - hi * tn)) / (tn + 1e-8)


def multi_pos_nce(pred: torch.Tensor, targ: torch.Tensor, tix: torch.Tensor,
                  tau: float) -> torch.Tensor:
    """In-batch InfoNCE with multi-positive masking (see g2f_train.py)."""
    logits = (l2t(pred) @ l2t(targ).T) / tau
    pos = tix[:, None] == tix[None, :]
    lse_pos = torch.logsumexp(torch.where(pos, logits, torch.full_like(logits, -1e9)), dim=-1)
    return (torch.logsumexp(logits, dim=-1) - lse_pos).mean()


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-subjects", type=int, nargs="+", required=True)
    ap.add_argument("--test-subject", type=int, required=True)
    ap.add_argument("--z-root", type=str, default=str(NB_ROOT / "outputs/hcma_10subj"))
    ap.add_argument("--targets-dir", type=str, default=str(NB_ROOT / "outputs/g2/targets"))
    ap.add_argument("--clip-text-dir", type=str, default=str(NB_ROOT / "outputs/nda_ss/sub-08/clip_text"))
    ap.add_argument("--captions-jsonl", type=str, default=str(NB_ROOT / "outputs/g2/captions/captions_train.jsonl"))
    ap.add_argument("--ip-train-npy", type=str,
                    default="/project/peilab/why/eeg-brainit/outputs/atm_bridge/clip_img_train_1024.npy")
    ap.add_argument("--ip-test-npy", type=str,
                    default="/project/peilab/why/eeg-brainit/outputs/atm_bridge/clip_img_test_1024.npy")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=26)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--tau-nce", type=float, default=0.07)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--var-frac", type=float, default=0.40)
    ap.add_argument("--w-res", type=float, default=0.4)
    ap.add_argument("--w-img", type=float, default=0.5)
    ap.add_argument("--w-ip", type=float, default=1.0)
    ap.add_argument("--w-mem", type=float, default=1.0)
    ap.add_argument("--w-nce", type=float, default=1.0)
    ap.add_argument("--w-cls", type=float, default=1.0)
    ap.add_argument("--w-fuse-cls", type=float, default=0.3)
    ap.add_argument("--w-struct", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--gate-quantile", type=float, default=0.5,
                    help="margin quantile on TRAIN-val rows below which the self-prompt "
                         "falls back to the generic prompt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    (out / "conds").mkdir(parents=True, exist_ok=True)
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    T = Path(args.targets_dir)
    stag = f"sub-{args.test_subject:02d}"

    # -------------------------------------------------- train-only concept gallery
    cbank_np, clab_np, phrases = build_concept_bank(
        Path(args.clip_text_dir), Path(args.captions_jsonl))
    n_cls = cbank_np.shape[0]
    print(f"[g3f] concept gallery {cbank_np.shape} labels {clab_np.shape} "
          f"({len(set(clab_np.tolist()))} distinct)")

    # -------------------------------------------------- frozen targets
    ip_bank_np = np.load(args.ip_train_npy).astype(np.float32)          # (16540,1024)
    ip_test_np = np.load(args.ip_test_npy).astype(np.float32)           # (200,1024)
    bank = torch.from_numpy(l2n(ip_bank_np)).to(dev)
    n_bank = bank.shape[0]
    cbank = torch.from_numpy(cbank_np).to(dev)
    clab = torch.from_numpy(clab_np).to(dev)

    keys = ("image", "concept", "overall", "subject", "background", "detail")
    tgt: dict[str, torch.Tensor] = {}
    for k in keys:
        src = T / f"sem_{k}_train.npy"
        if not src.is_file():
            src = T / f"sem_concept_tmpl_train.npy" if k == "concept" else src
        tgt[k] = torch.from_numpy(np.load(src).astype(np.float32))
    tgt["struct"] = torch.from_numpy(np.load(T / "perc_struct_train.npy").astype(np.float32))
    tgt["ip"] = torch.from_numpy(l2n(ip_bank_np))

    # -------------------------------------------------- gather train rows
    zs = []
    for s in args.train_subjects:
        z = np.load(f"{args.z_root}/sub-{s:02d}/zret/z_eeg_proj_train.npy").astype(np.float32)
        if z.shape[0] != n_bank:
            raise SystemExit(f"[FATAL] sub-{s:02d} z rows {z.shape[0]} != bank {n_bank}")
        zs.append(z)
    Ztr = np.concatenate(zs, 0)
    tidx = np.tile(np.arange(n_bank), len(args.train_subjects))
    n_tr = Ztr.shape[0]

    # held-out slice of TRAIN target indices -> leak-free selection AND the
    # margin threshold for the self-prompt gate
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_bank)
    val_t = np.zeros(n_bank, dtype=bool)
    val_t[perm[:int(n_bank * args.val_frac)]] = True
    is_val = val_t[tidx]
    tr_sel = np.where(~is_val)[0]
    va_sel = np.where(is_val)[0]
    va_sel = va_sel[np.arange(len(va_sel)) % max(1, len(args.train_subjects)) == 0]
    if args.limit_train > 0:
        tr_sel = tr_sel[:args.limit_train]
    if args.limit_val > 0:
        va_sel = va_sel[:args.limit_val]

    # TRAIN-split means over the training rows only (never the val slice, never test)
    mu = {k: tgt[k][tidx[tr_sel]].mean(0).cpu().numpy().astype(np.float32) for k in keys}
    mu["ip"] = tgt["ip"][tidx[tr_sel]].mean(0).cpu().numpy().astype(np.float32)
    # the same vector the g2f run used as its constant baseline; reported as a floor
    print("[g3f] ||mu_ip|| = %.4f (constant-condition cos floor)" % float(np.linalg.norm(mu["ip"])))

    tgt = {k: v.to(dev) for k, v in tgt.items()}
    clab = clab.to(dev)
    Zte_t = torch.from_numpy(
        np.load(f"{args.z_root}/{stag}/zret/z_eeg_proj_test.npy").astype(np.float32)).to(dev)
    ite_np = l2n(ip_test_np)
    Ztr_t = torch.from_numpy(Ztr)
    print(f"[g3f] train rows {n_tr} (trainsel {len(tr_sel)}, valsel {len(va_sel)}) "
          f"| bank {n_bank} | concepts {n_cls} | test {tuple(Zte_t.shape)}")

    model = G3FNet(drop=args.dropout).to(dev)
    model.set_means(mu)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[g3f] params {n_par/1e6:.2f}M device {dev}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = max(1, len(tr_sel) // args.batch_size)
    total_steps = args.epochs * steps
    if total_steps >= 20:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.25)
    else:
        sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=total_steps)

    best = {"score": -1e9, "epoch": -1}
    last_path, best_path = out / "last.pth", out / "best.pth"
    start_ep = 0
    if args.resume and last_path.is_file():
        try:
            ck = torch.load(last_path, map_location=dev, weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            start_ep = ck["epoch"] + 1
            best = ck.get("best", best)
            print(f"[g3f] resumed at epoch {start_ep} (best {best})")
        except Exception as e:                                     # noqa: BLE001
            print(f"[g3f] resume failed ({e}); starting fresh")

    def evaluate(sel: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
        model.eval()
        acc: dict[str, list[float]] = {}
        margins: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(sel), 1024):
                rows = sel[i:i + 1024]
                x = Ztr_t[rows].to(dev)
                ti = torch.from_numpy(tidx[rows]).to(dev)
                sem = model.sem_heads(x)
                per = model.per_heads(x)
                qf = sem["fused"]
                tgtip = tgt["ip"][ti]
                lg = sem["concept"] @ cbank.T
                top2 = lg.topk(2, dim=-1).values
                margins.append((top2[:, 0] - top2[:, 1]).float().cpu().numpy())
                mem = model.memory(qf, bank, args.tau)
                v = {
                    "image": (sem["image"] * tgt["image"][ti]).sum(-1).mean(),
                    "fused_to_ip": (qf * tgtip).sum(-1).mean(),
                    "mem_to_ip": (mem * tgtip).sum(-1).mean(),
                    "nce": multi_pos_nce(qf, tgtip, ti, args.tau_nce),
                    "cls_top1": (lg.argmax(1) == clab[ti]).float().mean(),
                    "cls_top5": (lg.topk(5, dim=-1).indices == clab[ti][:, None]).any(-1).float().mean(),
                    "struct": (l2t(per["struct"].flatten(1)) *
                               l2t(tgt["struct"][ti].flatten(1))).sum(-1).mean(),
                    "struct_std": per["struct"].std(0).norm() / (tgt["struct"].std(0).norm() + 1e-8),
                    "res_norm": sem["_res"]["subject"].std(0).norm() /
                                (tgt["subject"].std(0).norm() + 1e-8),
                }
                for k, t in v.items():
                    acc.setdefault(k, []).append(float(t.detach()))
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        # semantic objectives dominate; structure is weighted down because
        # g2_train let it outbid semantics during selection
        m["score"] = (0.30 * m["mem_to_ip"] + 0.15 * m["fused_to_ip"] + 0.20 * m["cls_top1"]
                      + 0.10 * m["image"] + 0.20 * m["struct"] - 0.15 * m["nce"])
        model.train()
        return m, np.concatenate(margins, 0)

    history: list[dict] = []
    val_margin = np.zeros(0)
    for ep in range(start_ep, args.epochs):
        model.train()
        perm2 = rng.permutation(len(tr_sel))
        run: dict[str, float] = {}
        nb = 0
        t0 = time.time()
        for b in range(steps):
            rows = tr_sel[perm2[b * args.batch_size:(b + 1) * args.batch_size]]
            if len(rows) < 2:
                continue
            x = Ztr_t[rows].to(dev)
            ti = torch.from_numpy(tidx[rows]).to(dev)
            sem = model.sem_heads(x)
            per = model.per_heads(x)
            ip_t = tgt["ip"][ti]

            # 1. centered residual regression over the granularities + image
            l_img = (1 - (sem["image"] * tgt["image"][ti]).sum(-1)).mean()
            l_res = sum((1 - (sem[g] * tgt[g][ti]).sum(-1)).mean() for g in GRANS) / len(GRANS)

            # 2. single IP producer + differentiable memory
            l_ip = (1 - (sem["fused"] * ip_t).sum(-1)).mean()
            mem = model.memory(sem["fused"], bank, args.tau)
            l_mem = (1 - (mem * ip_t).sum(-1)).mean()

            # 3. discriminability (the term that actually buys it)
            l_nce = (multi_pos_nce(sem["fused"], ip_t, ti, args.tau_nce)
                     + multi_pos_nce(mem, ip_t, ti, args.tau_nce)
                     + multi_pos_nce(sem["image"], tgt["image"][ti], ti, args.tau_nce))

            # 4. concept supervision: the head that drives the self-prompt, and a
            #    lighter copy on the fused condition so IP itself is discriminative
            lg = sem["concept"] @ cbank.T
            l_cls = F.cross_entropy(lg / args.tau, clab[ti])
            l_fcls = F.cross_entropy((sem["fused"] @ cbank.T) / args.tau, clab[ti])

            # 5. perceptual tower (low-frequency latent only)
            l_struct = (1 - (l2t(per["struct"].flatten(1)) *
                             l2t(tgt["struct"][ti].flatten(1))).sum(-1)).mean()

            # 6. two-sided anti-degeneracy on every predicted block
            hinges = (var_band(per["struct"], tgt["struct"][ti])
                      + var_band(sem["fused"], ip_t)
                      + var_band(sem["image"], tgt["image"][ti])
                      + var_band(sem["concept"], tgt["concept"][ti])
                      + sum(var_band(sem[g], tgt[g][ti]) for g in GRANS) / len(GRANS))

            loss = (args.w_img * l_img + args.w_res * l_res + args.w_ip * l_ip
                    + args.w_mem * l_mem + args.w_nce * l_nce + args.w_cls * l_cls
                    + args.w_fuse_cls * l_fcls + args.w_struct * l_struct + 0.5 * hinges)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            for k, v in (("loss", loss), ("img", l_img), ("res", l_res), ("ip", l_ip),
                         ("mem", l_mem), ("nce", l_nce), ("cls", l_cls),
                         ("fcls", l_fcls), ("struct", l_struct)):
                run[k] = run.get(k, 0.0) + float(v.detach())
            nb += 1

        tr_m = {k: v / max(nb, 1) for k, v in run.items()}
        va_m, val_margin = evaluate(va_sel)
        rec = {"epoch": ep, "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1),
               **{f"tr_{k}": round(v, 4) for k, v in tr_m.items()},
               **{f"va_{k}": round(v, 4) for k, v in va_m.items()}}
        history.append(rec)
        print(f"[ep{ep}] " + " ".join(f"{k}={va_m[k]:.4f}" for k in
                                      ("mem_to_ip", "fused_to_ip", "nce", "cls_top1",
                                       "cls_top5", "image", "struct", "struct_std",
                                       "res_norm", "score")))
        if va_m["score"] > best["score"]:
            best = {"score": va_m["score"], "epoch": ep,
                    **{f"va_{k}": v for k, v in va_m.items()}}
            torch.save({"model": model.state_dict(), **best}, best_path)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best": best}, last_path)

    # -------------------------------------------------- export (test rows only)
    if best_path.is_file():
        ck = torch.load(best_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"[g3f] loaded best epoch {ck.get('epoch')} score {ck.get('score'):.4f}")
    model.eval()
    outs: dict[str, list[np.ndarray]] = {}
    logits_te: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, Zte_t.shape[0], 256):
            x = Zte_t[i:i + 256]
            sem = model.sem_heads(x)
            per = model.per_heads(x)
            qf = sem["fused"]
            vals = {"ip_fused": qf, "ip_mem": model.memory(qf, bank, args.tau),
                    "concept": sem["concept"], "image": sem["image"],
                    "lf_latent": per["struct"]}
            for g in GRANS:
                vals[f"gran_{g}"] = sem[g]
            for k, v in vals.items():
                outs.setdefault(k, []).append(v.float().cpu().numpy())
            logits_te.append((sem["concept"] @ cbank.T).float().cpu().numpy())

    for k, v in outs.items():
        a = np.concatenate(v, 0).astype(np.float32)
        if k.startswith(("ip_", "concept", "image", "gran_")):
            a = l2n(a)
        np.save(out / "conds" / f"{k}_test.npy", a)
    LOG = np.concatenate(logits_te, 0)

    # -------------------------------------------------- SELF-PROMPT (leak-free)
    # threshold from TRAIN-val margins only; the gallery holds TRAIN concepts only
    thr = float(np.quantile(val_margin, args.gate_quantile)) if val_margin.size else 0.0
    top1 = LOG.argmax(1)
    top2v = np.sort(LOG, 1)[:, -2]
    margin = LOG[np.arange(len(LOG)), top1] - top2v
    self_prompts = [f"a photo of a {phrases[i]}" for i in top1]
    gated = [self_prompts[i] if margin[i] >= thr else GENERIC_PROMPT for i in range(len(margin))]
    (out / "prompts" / "prompts_self.json").write_text(
        json.dumps(self_prompts, indent=1), encoding="utf-8")
    (out / "prompts" / "prompts_selfgate.json").write_text(
        json.dumps(gated, indent=1), encoding="utf-8")
    (out / "prompts" / "prompts_generic.json").write_text(
        json.dumps([GENERIC_PROMPT] * len(margin), indent=1), encoding="utf-8")
    (out / "prompts" / "selfprompt_debug.json").write_text(json.dumps({
        "gate_threshold": thr, "gate_quantile": args.gate_quantile,
        "val_margin_mean": float(val_margin.mean()) if val_margin.size else None,
        "test_margin_mean": float(margin.mean()),
        "n_gated_to_generic": int((margin < thr).sum()),
        "top1_concepts": [phrases[i] for i in top1],
        "unique_top1": int(len(set(top1.tolist()))),
        "note": ("gallery = 1654 TRAIN concepts; test concepts are disjoint from it, "
                 "so the emitted string cannot name a test class."),
    }, indent=2), encoding="utf-8")

    # -------------------------------------------------- diagnostics
    def disc(a: np.ndarray) -> dict[str, float]:
        s = l2n(a) @ ite_np.T
        n = len(s)
        r = np.random.default_rng(0).permutation(n)
        ok = np.arange(n) != r
        return {"top1": float(np.mean([i in np.argsort(-s[i])[:1] for i in range(n)])),
                "top5": float(np.mean([i in np.argsort(-s[i])[:5] for i in range(n)])),
                "twoway": float(np.mean(s[np.arange(n), np.arange(n)][ok] >
                                        s[np.arange(n), r][ok]))}

    # concept retrieval accuracy against the ORACLE concept target is a DIAGNOSTIC
    # ONLY (it needs test concepts, which we never use to build anything).
    oracle_ok = None
    cte_path = T / "sem_concept_tmpl_test.npy"
    if cte_path.is_file():
        cte = l2n(np.load(cte_path).astype(np.float32))
        sim = outs["concept"][0] @ cte.T
        oracle_ok = {"concept_top1_to_oracle": float(np.mean(np.argmax(sim, 1) == np.arange(len(sim)))),
                     "note": "diagnostic only; never used to build the prompt"}

    report = {"protocol": "g3f", "test_subject": stag, "train_subjects": args.train_subjects,
              "params_m": round(n_par / 1e6, 3), "best": best, "history": history,
              "n_concepts": int(n_cls), "n_bank": int(n_bank), "mu_ip_norm": float(np.linalg.norm(mu["ip"])),
              "gate_threshold": thr, "n_gated_to_generic": int((margin < thr).sum()),
              "prompt_unique_top1": int(len(set(top1.tolist()))),
              "constant_baseline_cos_to_ip": float((l2n(np.tile(mu["ip"], (len(ite_np), 1))) * ite_np).sum(-1).mean())}
    # only 1024-d conditions live in the CLIP image/IP space; h_image is 1280-d
    for k in ("ip_fused", "ip_mem", "concept"):
        report[f"{k}_disc"] = disc(outs[k][0])
    if oracle_ok:
        report.update(oracle_ok)
    report["note"] = ("cos_to_ip is NOT a usable quality metric (a constant scores "
                      f"{report['constant_baseline_cos_to_ip']:.4f}); use *_disc.")
    (out / "g3f_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[g3f] " + json.dumps({k: v for k, v in report.items()
                                 if k.endswith("_disc") or k in
                                 ("constant_baseline_cos_to_ip", "concept_top1_to_oracle",
                                  "gate_threshold", "n_gated_to_generic")}, indent=2))
    print(f"[g3f] done -> {out}")


if __name__ == "__main__":
    main()
