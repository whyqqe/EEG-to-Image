#!/usr/bin/env python
"""NW-v6 S1: the STH-Align alignment stack -- one shared trunk, three projection heads,
jointly trained, with an uncertainty-weighted multi-positive objective.

WHY THIS EXISTS
---------------
The generator is not the problem.  A5 -- our operator, our semantic bank, plus
GROUND-TRUTH depth/edge CLIP rows -- reaches incep 0.8400 / clip 0.9123, which is at or
slightly above CogCapPro's published sub-08 (0.831 / 0.903).  Our fully deployable best
reaches 0.7440 / 0.8525.  The entire interval between those two is the structure branch:

    arm                                    incep     clip
    E1   no structure at all               0.7302   0.8122   <- floor
    E2   + variance-restored linear ridge  0.7363   0.8215   <- 5.5% of the GT gain
    G3   + trained head (per-modality MLP) 0.7483   0.8314   <- 16.5%
    A5   + GT depth/edge                   0.8400   0.9123   <- the target
    A9   + GT image/depth/edge             0.9811   0.9921   <- mechanism ceiling

So the question is only how much trial-specific structure our EEG->depth/edge map can
carry.  We measure that with dev_corr (mean cosine between predicted and true deviation
from the GT TRAIN centroid, mean removed from both sides), which is the quantity the
seven-arm gain table tracks.

WHAT WAS ALREADY LEARNED THE HARD WAY
-------------------------------------
The first head was a plain 1024->1024->1024 MLP (3.1M parameters) trained with in-batch
InfoNCE.  It overfit inside 9 epochs -- training loss fell to 0.014 while validation
dev_corr decayed 0.2537 -> 0.1404 monotonically -- and at its best it did not even match
the linear ridge (0.3211 vs 0.3620 on test).  A regularisation grid then showed the
ordering is INVERTED with respect to capacity:

    h256_r512_do0.4   461k params   val dev_corr 0.3424   test 0.3731
    h512_r0_do0.3    1315k params   val dev_corr 0.3151
    h1024_r0_do0.0   3153k params   val dev_corr 0.3036

Fewer parameters, more dropout, and a low-rank output bottleneck -- predicting R
coefficients in the top-R principal directions of the training deviations -- is what
works.  Two consequences that shape this script:

  * we are NOT capacity-limited, we are information-limited, so this script changes the
    OBJECTIVE and the INPUT FEATURES rather than growing the network;
  * the winner is a configuration, not an architecture, so the grid below is anchored on
    h256/r512/do0.4 and varies the things that were never tried.

THE FOUR UPGRADES, AND WHY EACH ONE
-----------------------------------
This transposes CogCapPro's encoder-side contributions, which is where that work's gains
actually live (its generator is the same SDXL-Turbo + multi-branch IP-Adapter + empty
prompt + CFG 0 stack we already run; its reported +25.9% Top-1 / +10.6% Top-5 are
retrieval numbers).  We already have the two generator-side pieces -- asymmetric
multi-branch injection (our cc3 layout) and text dropped at inference (empty prompt).
We had none of the four encoder-side ones.

  (1) STH-Align: ONE SHARED TRUNK, THREE PROJECTION HEADS, JOINTLY TRAINED.
      We currently fit image, depth and edge independently.  CogCapPro maps modalities
      into a unified image space through a shared backbone plus multimodal projection
      heads.  Joint training lets the trunk learn the EEG->CLIP structure that is common
      to all three, and each head only has to learn the residual.  It is also 3x cheaper:
      one training run now serves all three modalities.

  (2) SCM-LOSS: A MULTI-POSITIVE OBJECTIVE, BECAUSE THE TASK IS ONE-TO-MANY.
      The training bank is 16540 rows = 1654 concepts x 10 images, and this script
      VERIFIES at runtime that the rows are block-ordered by concept (same-concept GT
      image-CLIP rows sit at cosine ~0.75, different-concept at ~0.40 against a random-
      pair baseline of ~0.39).  Hard-positive InfoNCE therefore tells the model that two
      images of the same concept are wrong answers, which is a mislabelled problem, not a
      hard one.  The multi-positive form puts every same-concept row in the numerator:

          L = (1-w) * -log( exp(s_ii) / sum_{j!=i} exp(s_ij) )
            +     w  * -log( sum_{j: c_j=c_i} exp(s_ij) / sum_{j!=i} exp(s_ij) )

      w is gridded, including w=0 so the hard-positive control runs under identical code.
      The risk is real and the gate catches it: a multi-positive objective can buy
      concept-level accuracy by giving up trial specificity, and dev_corr is precisely a
      trial-specificity measure.

  (3) FUSION: FEED THE SHARED AND SUBJECT-SPECIFIC PARTS TOGETHER.
      The HCMA encoder emits `shared_r` (built to be subject-invariant, protected by an
      explicit decorrelation loss) and `specific_s` (subject-specific).  Our pipeline kept
      only `shared_r`.  Re-exporting both and probing them (ocf_export_ss_parts.py) gives:

          layout retrieval (top-5)   fused 0.35   shared_r 0.355   specific_s 0.225
                                     cat(shared,specific) 0.38   <- best
          semantics retrieval (top-1) shared_r 0.29  cat 0.285  fused 0.225  specific 0.155

      Note what this does and does not say.  `specific_s` alone is the WORST at layout, so
      the appealing story -- "geometry lives in the private part" -- is FALSE and was
      rightly rejected (verdict recorded in outputs/ocf/ss_parts_probe_sub08.json).  But
      concatenating still beats every single representation at layout, so the two parts
      carry complementary information and the fusion is worth the two extra columns.

  (4) UNCERTAINTY WEIGHTING ACROSS MODALITIES.
      Joint training needs a way to balance three losses whose scales differ.  Rather than
      a hand-tuned constant we learn a per-modality log-variance (Kendall-style):

          L = sum_m [ (1/(2 sigma_m^2)) * L_m + log sigma_m ],  sigma_m = exp(clamp(ls_m))

      clamped to +-4 so no single modality can be silenced; the plain mean is kept as a
      control in the grid.

THE GATE, AND WHY IT IS NOT A VETO
----------------------------------
A linear ridge is fitted here too, on the same fit rows with alpha chosen on the same
validation rows and scored by the identical function, so the comparison is like for like.
Configuration and variant are chosen PER MODALITY on validation dev_corr, using held-out
CONCEPTS (fit/val_a/val_b all live inside the 16540 training rows).  Nothing is selected
on test; test numbers are printed only after the choice is made.

Banks are emitted unconditionally, because embedding metrics have already misled us once:
dev_corr 0.352 bought about 5% of the GT gain while dev_corr 1.0 bought +0.128, so a head
that looks good in embedding space can be worthless in pixels and only generation settles
it.  The tier is a prediction; the generation arms are the verdict.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

NB_ROOT = Path("/project/peilab/why/NeuroBridge")


# --------------------------------------------------------------------------- utils
def l2n(x: np.ndarray) -> np.ndarray:
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def energy_split(x: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    x = l2n(x)
    m = x.mean(0, keepdims=True)
    d = x - m
    return (float((m ** 2).sum()), float((d ** 2).sum(1).mean()), l2n(m), l2n(d))


def dev_metrics(pred: np.ndarray, tgt: np.ndarray) -> dict:
    """dev_corr / dev_top1, with the batch mean removed from BOTH sides.

    This must match nwv4_a5_gap.py exactly, because that script produced the numbers this
    stack is judged against.  Without the mean removal the same ridge bank reads 0.72
    instead of 0.36: the shared component -- which is not what the structure branch is for
    -- dominates the cosine, and a predicted offset is not trial information.
    """
    p, t = l2n(pred), l2n(tgt)
    p = l2n(p - p.mean(0, keepdims=True))
    t = l2n(t - t.mean(0, keepdims=True))
    S = p @ t.T
    return {"dev_top1": float((S.argmax(1) == np.arange(len(t))).mean()),
            "dev_corr": float((p * t).sum(1).mean())}


def solve_scale_for_share(m_unit: np.ndarray, dev: np.ndarray, target_share: float) -> float:
    """cond = l2n(m + a*dev); find a so mean_i |cond_i - mean(cond)|^2 == target_share.

    This is the variance-restoration operator from nw5c.  Least squares buys its error
    reduction by predicting the mean, so the trial-specific component -- the branch's
    entire value -- arrives crushed; restoring it to the GT share is what took retrieval
    from 3x/11x to 26x/35x chance at identical weights and identical EEG.
    """
    lo, hi = 0.0, 200.0
    for _ in range(70):
        a = 0.5 * (lo + hi)
        c = l2n(m_unit + dev * a)
        d = c - c.mean(0, keepdims=True)
        if float((d ** 2).sum(1).mean()) < target_share:
            lo = a
        else:
            hi = a
    return 0.5 * (lo + hi)


def principal_basis(d: np.ndarray, fit_i: np.ndarray, rank: int) -> np.ndarray:
    """Top-`rank` principal directions of the FIT-row deviations, (1024, rank)."""
    x = d[fit_i].astype(np.float64)
    x = x - x.mean(0, keepdims=True)
    c = (x.T @ x) / max(len(x) - 1, 1)
    w, v = np.linalg.eigh(c)
    return v[:, np.argsort(w)[::-1][:rank]].astype(np.float32)


def ridge_control(ztr, ytr_dev, zte, fit_i, val_i, alphas):
    """Deviation-targeted ridge, alpha on val dev_corr.  Both banks are needed: val_i
    indexes the TRAIN bank, test is a separate 200-row bank."""
    zf = np.concatenate([ztr, np.ones((len(ztr), 1), np.float32)], 1)
    zf_te = np.concatenate([zte, np.ones((len(zte), 1), np.float32)], 1)
    best = None
    for a in alphas:
        g = zf[fit_i].T @ zf[fit_i]
        if a > 0:
            pen = np.eye(g.shape[0], dtype=np.float32) * a
            pen[-1, -1] = 0.0
            g = g + pen
        w = np.linalg.solve(g, zf[fit_i].T @ ytr_dev[fit_i])
        v = dev_metrics(zf[val_i] @ w, ytr_dev[val_i])
        if best is None or v["dev_corr"] > best[1]["dev_corr"]:
            best = (a, v, w)
    a, v, w = best
    return a, v, zf_te @ w


# ------------------------------------------------------------------- feature loading
def load_features(kind: str, sid: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (train, test, description) for the requested EEG representation.

    The `cat` path mixes two files written by different scripts, so the shared half is
    VERIFIED against the copy stored beside the specific half.  A silent mismatch here
    would corrupt every downstream number, and the two exports differ in the last decimal
    of an unrelated warm-start flag.
    """
    intra = NB_ROOT / "outputs/ocf/intra_z" / f"sub-{sid:02d}"
    chab = NB_ROOT / "outputs/chab" / f"sub-{sid:02d}/z_warm0" / f"sub-{sid:02d}"
    sh_tr = l2n(np.load(intra / "shared_r_train.npy").astype(np.float32))
    sh_te = l2n(np.load(intra / "shared_r_test.npy").astype(np.float32))

    if kind == "shared":
        return sh_tr, sh_te, "shared_r (1024-d, subject-invariant)"

    if not (chab / "shared_r_train.npy").is_file():
        raise SystemExit(f"[FATAL] {chab}/shared_r_train.npy missing (needed for kind={kind})")
    ref_tr = l2n(np.load(chab / "shared_r_train.npy").astype(np.float32))
    cos = float((ref_tr * sh_tr).sum(1).mean())
    if cos < 0.999:
        raise SystemExit(
            f"[FATAL] shared_r mismatch between intra_z and z_warm0 (cos {cos:.4f}). "
            f"They are different encoders; concatenating their parts would mix spaces.")

    if kind == "fused":
        return (l2n(np.load(chab / "fused_train.npy").astype(np.float32)),
                l2n(np.load(chab / "fused_test.npy").astype(np.float32)),
                f"fused (1024-d; shared_r cos {cos:.4f})")

    if kind == "cat":
        sp_tr = l2n(np.load(chab / "specific_s_train.npy").astype(np.float32))
        sp_te = l2n(np.load(chab / "specific_s_test.npy").astype(np.float32))
        return (np.concatenate([sh_tr, sp_tr], 1), np.concatenate([sh_te, sp_te], 1),
                f"cat = shared_r (+) specific_s (2048-d; shared cos {cos:.4f})")

    raise SystemExit(f"[FATAL] unknown feature kind '{kind}'")


# -------------------------------------------------------------------------- the stack
def build_stack(torch, d_in, hidden, ranks, dropout):
    """Shared trunk + one projection head per modality, with an optional per-modality
    low-rank output bottleneck (predict R coefficients in that modality's principal
    directions rather than 1024 free numbers).  The bottleneck is not cosmetic: it is the
    single change that took the head from below the linear ridge to above it."""
    n_mod = len(ranks)

    class Stack(torch.nn.Module):
        def __init__(self):
            super().__init__()
            layers = [torch.nn.Linear(d_in, hidden), torch.nn.LayerNorm(hidden),
                      torch.nn.GELU()]
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            layers += [torch.nn.Linear(hidden, hidden), torch.nn.LayerNorm(hidden),
                       torch.nn.GELU()]
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            self.trunk = torch.nn.Sequential(*layers)
            self.out_dims = [r if r else 1024 for r in ranks]
            self.heads = torch.nn.ModuleList(
                [torch.nn.Linear(hidden, d) for d in self.out_dims])
            self.log_sigma = torch.nn.Parameter(torch.zeros(n_mod))
            for i, r in enumerate(ranks):
                if r:
                    self.register_buffer(f"basis_{i}", torch.zeros(1024, r))

        def forward(self, x):
            h = self.trunk(x)
            outs = []
            for i, head in enumerate(self.heads):
                o = head(h)
                if self.out_dims[i] != 1024:
                    o = o @ getattr(self, f"basis_{i}").T
                outs.append(o)
            return outs

    return Stack()


def train_one(torch, Ztr, Zte, DTR, fit_i, val_i, val_a_i, concepts, cfg, whiten,
              args, log, tag):
    """One (config, variant).  Returns per-modality predictions on test/val/val_a.

    All three modalities are trained together in one model -- that is upgrade (1).
    """
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    labs = list(DTR.keys())
    n_mod = len(labs)

    ranks, bases = [], {}
    for m in labs:
        if cfg["rank"]:
            b = principal_basis(DTR[m], fit_i, cfg["rank"])
            bases[m] = torch.tensor(b, device=dev)
            ranks.append(cfg["rank"])
        else:
            ranks.append(0)

    model = build_stack(torch, Ztr.shape[1], cfg["hidden"], ranks, cfg["dropout"]).to(dev)
    for i, m in enumerate(labs):
        if cfg["rank"]:
            getattr(model, f"basis_{i}").copy_(bases[m])
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

    std = {m: (DTR[m][fit_i].std(0, keepdims=True) + 1e-6) for m in labs}

    def make_tgt(m, idx_np):
        t = DTR[m][idx_np]
        if whiten:
            t = t / std[m]
        return torch.tensor(l2n(t), device=dev)

    Z = torch.tensor(Ztr, device=dev)
    fit_t = torch.tensor(fit_i, device=dev)
    Yfit = {m: make_tgt(m, fit_i) for m in labs}
    std_t = {m: torch.tensor(std[m], device=dev) for m in labs}
    val_t = torch.tensor(val_i, device=dev)
    con = torch.tensor(concepts, device=dev)

    n_fit = len(fit_i)
    spe = max(1, n_fit // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * spe)
    eye_cache = {}

    def scm_loss(pred, tgt, sel_con):
        """Multi-positive InfoNCE.  w=0 reproduces the hard-positive form exactly, so the
        control and the treatment share a code path and differ only in `w`."""
        p = torch.nn.functional.normalize(pred, dim=-1)
        logits = (p @ tgt.T) / cfg["tau"]
        b = logits.shape[0]
        eye = eye_cache.get(b)
        if eye is None:
            eye = torch.eye(b, dtype=torch.bool, device=dev)
            eye_cache[b] = eye
        log_denom = torch.logsumexp(logits.masked_fill(eye, -float("inf")), dim=1)
        hard = log_denom - torch.diagonal(logits)
        w = cfg["scm"]
        if w <= 0:
            return hard.mean()
        pos = (sel_con[:, None] == sel_con[None, :]) & ~eye
        # a batch can miss a concept's partner entirely; those rows fall back to hard
        has = pos.any(1)
        log_num = torch.full_like(hard, -float("inf"))
        if has.any():
            ln = torch.logsumexp(logits.masked_fill(~pos, -float("inf")), dim=1)
            log_num = torch.where(has, ln, torch.diagonal(logits))
        multi = log_denom - log_num
        return ((1.0 - w) * hard + w * multi).mean()

    best = {m: {"corr": -1.0, "epoch": -1, "vm": None, "state": None} for m in labs}
    bad = 0
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(n_fit)
        tot, nb = 0.0, 0
        for s in range(spe):
            sel = perm[s * args.batch:(s + 1) * args.batch]
            if len(sel) < 8:
                continue
            inputs = Z[fit_t[sel]]
            outs = model(inputs)
            sel_con = con[fit_t[sel]]
            losses = [scm_loss(outs[i], Yfit[m][sel], sel_con) for i, m in enumerate(labs)]

            if cfg["unc"]:
                ls = model.log_sigma.clamp(-4.0, 4.0)
                loss = sum(torch.exp(-2 * ls[i]) * losses[i] + ls[i]
                           for i in range(n_mod))
            else:
                loss = sum(losses)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()); nb += 1

        model.eval()
        with torch.no_grad():
            pv = model(Z[val_t])
        improved = False
        for i, m in enumerate(labs):
            o = pv[i]
            if whiten:
                o = o * std_t[m]
            o = o.cpu().numpy()
            o = o / (np.linalg.norm(o, axis=1, keepdims=True) + 1e-8)
            vm = dev_metrics(o, DTR[m][val_i])
            if vm["dev_corr"] > best[m]["corr"]:
                best[m] = {"corr": vm["dev_corr"], "epoch": ep, "vm": vm,
                           "state": {k: v.detach().clone()
                                     for k, v in model.state_dict().items()}}
                improved = True
        bad = 0 if improved else bad + 1
        if args.verbose and ((ep + 1) % 20 == 0 or ep == 0):
            log(f"      {tag} ep {ep:>3} loss {tot/max(nb,1):.4f} " + " ".join(
                f"{labs[i][:4]}_corr {best[labs[i]]['corr']:.4f}@{best[labs[i]]['epoch']}"
                for i in range(n_mod)))
        if bad >= args.patience:
            break

    out = {"params": n_par, "stopped_epoch": ep, "per_mod": {}}
    for i, m in enumerate(labs):
        model.load_state_dict(best[m]["state"])
        model.eval()

        def predict(arr):
            with torch.no_grad():
                o = model(torch.tensor(arr, device=dev))[i]
            if whiten:
                o = o * std_t[m]
            return o.cpu().numpy()

        out["per_mod"][m] = {"test": predict(Zte), "val": predict(Ztr[val_i]),
                             "val_a": predict(Ztr[val_a_i]), "vm": best[m]["vm"],
                             "epoch": best[m]["epoch"]}
    ls = model.log_sigma.detach().clamp(-4.0, 4.0).cpu().numpy()
    out["learned_log_sigma"] = {m: float(ls[i]) for i, m in enumerate(labs)}
    log(f"  {tag}: {n_par/1e3:.0f}k params, stopped ep {ep}, " + " ".join(
        f"{labs[i][:4]} {best[labs[i]]['vm']['dev_corr']:.4f}@{best[labs[i]]['epoch']}"
        for i in range(n_mod)))
    return out


# ------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--stag", type=str, default="sub-08")
    ap.add_argument("--cond-cache", type=str, default=str(NB_ROOT / "outputs/gem/cond_cache"))
    ap.add_argument("--split-json", type=str, default=str(NB_ROOT / "outputs/leakfree/split.json"))
    ap.add_argument("--out-dir", type=str, default=str(NB_ROOT / "outputs/nw6_s08/conds"))
    ap.add_argument("--modalities", type=str, default="img,depth,edge")
    ap.add_argument("--features", type=str, default="cat,shared")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--verbose", type=int, default=1)
    ap.add_argument("--gate-margin", type=float, default=1.10)
    ap.add_argument("--rank-margin", type=float, default=1.5)
    ap.add_argument("--rank-corr-floor", type=float, default=0.98)
    ap.add_argument("--holdout-concepts", type=int, default=100)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    def log(s: str) -> None:
        print(s, flush=True)

    import torch
    torch.manual_seed(args.seed)

    sid, stag = args.subject, args.stag
    cc, outdir = Path(args.cond_cache), Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    labs = args.modalities.split(",")
    feats = args.features.split(",")

    sp = json.loads(Path(args.split_json).read_text())
    fit_i = np.asarray(sp["fit_rows"], dtype=int)
    val_i = np.asarray(sp.get("val_b_rows") or sp["val_a_rows"], dtype=int)
    val_a_i = np.asarray(sp["val_a_rows"], dtype=int)

    # ---- 16540 train rows are block-ordered by concept (10 images each) -----------
    # VERIFIED at runtime rather than trusted: SCM-Loss depends on it, and a wrong
    # grouping would silently turn same-concept negatives into false negatives.
    Gtr_img = l2n(np.load(cc / "clip_img1024_train.npy").astype(np.float32))
    n_tr = len(Gtr_img)
    if n_tr % 10:
        raise SystemExit(f"[FATAL] train bank has {n_tr} rows, not a multiple of 10")
    concepts = (np.arange(n_tr) // 10).astype(np.int64)
    S = Gtr_img[:200] @ Gtr_img[:200].T
    blocks_in, blocks_out = [], []
    for b in range(6):
        g = slice(b * 10, (b + 1) * 10)
        sub = S[g, g]
        blocks_in.append(sub[~np.eye(10, dtype=bool)].mean())
        oth = np.r_[0:b * 10, (b + 1) * 10:200].astype(int)
        blocks_out.append(S[g][:, oth].mean())
    rng = np.random.default_rng(0)
    rnd = rng.permutation(n_tr)[:200]
    Sr = Gtr_img[rnd] @ Gtr_img[rnd].T
    rand_in = float(Sr[np.triu_indices(200, 1)].mean())
    ci, co = float(np.mean(blocks_in)), float(np.mean(blocks_out))
    log(f"[concepts] block-of-10 within-concept cos {ci:.4f}  between {co:.4f}  "
        f"random-pair baseline {rand_in:.4f}")
    if ci < co + 0.1:
        raise SystemExit(
            f"[FATAL] train rows do not look block-ordered by concept "
            f"(within {ci:.4f} vs between {co:.4f}). SCM-Loss would be wrong here; "
            f"refusing to proceed.")

    # ---- targets: GT rows minus the GT TRAIN centroid ----------------------------
    TGT, gt_meta = {}, {}
    for lab in labs:
        Gtr = l2n(np.load(cc / f"clip_{lab}1024_train.npy").astype(np.float32))
        Gte = l2n(np.load(cc / f"clip_{lab}1024_test.npy").astype(np.float32))
        m_tr = Gtr.mean(0, keepdims=True).astype(np.float32)
        TGT[lab] = (Gtr - m_tr).astype(np.float32)
        e_m, e_d, _, _ = energy_split(Gte)
        gt_meta[lab] = {"gt_energy_dev": e_d, "gt_energy_mean": e_m, "m_tr": m_tr,
                        "Gte": Gte}

    # ---- linear ridge control, same rows, same scoring ---------------------------
    log("[align] linear ridge control (deviation target, alpha on val dev_corr)")
    lin = {}
    for kind in feats:
        ztr, zte, desc = load_features(kind, sid)
        for lab in labs:
            a, v, pte = ridge_control(ztr, TGT[lab], zte, fit_i, val_i,
                                      [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
            lin[(kind, lab)] = {"alpha": a, "val": v, "test_pred": pte, "feat": desc,
                                "dim": int(ztr.shape[1])}
            log(f"  {kind:<7} {lab:<6} dim {ztr.shape[1]:<5} alpha {a:g}  "
                f"val dev_corr {v['dev_corr']:.4f}  val dev_top1 {v['dev_top1']*200:.1f}x  "
                f"test dev_corr {dev_metrics(pte, gt_meta[lab]['Gte'])['dev_corr']:.4f}")

    # ---- the grid: anchored on the winning config, varying what was never tried ---
    CFGS = [
        {"name": "cat_h256r512d4_s0.0", "feat": "cat", "hidden": 256, "rank": 512, "dropout": 0.4, "wd": 1e-2, "lr": 1e-3, "tau": 0.1, "scm": 0.0, "unc": 1},
        {"name": "cat_h256r512d4_s0.3", "feat": "cat", "hidden": 256, "rank": 512, "dropout": 0.4, "wd": 1e-2, "lr": 1e-3, "tau": 0.1, "scm": 0.3, "unc": 1},
        {"name": "cat_h256r512d4_s0.6", "feat": "cat", "hidden": 256, "rank": 512, "dropout": 0.4, "wd": 1e-2, "lr": 1e-3, "tau": 0.1, "scm": 0.6, "unc": 1},
        {"name": "cat_h256r512d4_s0.3_nu", "feat": "cat", "hidden": 256, "rank": 512, "dropout": 0.4, "wd": 1e-2, "lr": 1e-3, "tau": 0.1, "scm": 0.3, "unc": 0},
        {"name": "shr_h256r512d4_s0.0", "feat": "shared", "hidden": 256, "rank": 512, "dropout": 0.4, "wd": 1e-2, "lr": 1e-3, "tau": 0.1, "scm": 0.0, "unc": 1},
        {"name": "shr_h256r512d4_s0.3", "feat": "shared", "hidden": 256, "rank": 512, "dropout": 0.4, "wd": 1e-2, "lr": 1e-3, "tau": 0.1, "scm": 0.3, "unc": 1},
        {"name": "cat_h512r256d2_s0.3", "feat": "cat", "hidden": 512, "rank": 256, "dropout": 0.2, "wd": 1e-2, "lr": 1e-3, "tau": 0.1, "scm": 0.3, "unc": 1},
        {"name": "cat_h128r768d4_s0.3", "feat": "cat", "hidden": 128, "rank": 768, "dropout": 0.4, "wd": 1e-2, "lr": 1e-3, "tau": 0.1, "scm": 0.3, "unc": 1},
    ]
    run_cfgs = [c for c in CFGS if c["feat"] in feats]
    variants = [0, 1]

    results = {}
    for cfg in run_cfgs:
        ztr, zte, desc = load_features(cfg["feat"], sid)
        for wh in variants:
            tag = f"{cfg['name']}/{'whit' if wh else 'raw'}"
            t0 = time.time()
            torch.manual_seed(args.seed)
            try:
                r = train_one(torch, ztr, zte, TGT, fit_i, val_i, val_a_i, concepts,
                              cfg, wh, args, log, tag)
            except Exception as e:                                   # noqa: BLE001
                log(f"  {tag}: FAILED ({type(e).__name__}: {e})")
                continue
            r["variant"] = "whitened" if wh else "raw"
            r["config"] = cfg["name"]
            r["feat"] = cfg["feat"]
            r["secs"] = time.time() - t0
            results[(cfg["name"], wh)] = r

    # ---- per-modality selection, gate, emission ----------------------------------
    report = {"subject": sid, "stag": stag, "features": feats,
              "concept_check": {"within": ci, "between": co, "random": rand_in},
              "linear": {f"{k[0]}|{k[1]}": {"alpha": v["alpha"], "val": v["val"],
                                            "feat": v["feat"], "dim": v["dim"]}
                         for k, v in lin.items()},
              "head": {}, "gate": {}, "grid": {}, "emitted": {}}

    for lab in labs:
        ranked = []
        for (cname, wh), r in results.items():
            pm = r["per_mod"][lab]
            ranked.append((pm["vm"]["dev_corr"], cname, wh, r, pm))
        ranked.sort(key=lambda t: -t[0])
        lk = max((k for k in lin if k[1] == lab), key=lambda k: lin[k]["val"]["dev_corr"])
        lv = lin[lk]["val"]
        report["grid"][lab] = [
            {"config": c, "variant": "whitened" if w else "raw", "feat": r["feat"],
             "val_dev_corr": pm["vm"]["dev_corr"], "val_dev_top1": pm["vm"]["dev_top1"],
             "epoch": pm["epoch"], "params": r["params"], "secs": r["secs"],
             "log_sigma": r["learned_log_sigma"].get(lab)}
            for _, c, w, r, pm in ranked]
        if not ranked:
            report["gate"][lab] = {"tier": "fail", "passed": False, "strong": False}
            log(f"[align] {lab}: every config failed - no bank emitted")
            continue

        _, cname, wh, r, pm = ranked[0]
        vm = pm["vm"]
        tm = dev_metrics(pm["test"], gt_meta[lab]["Gte"])
        lt = dev_metrics(lin[lk]["test_pred"], gt_meta[lab]["Gte"])
        g_top = vm["dev_top1"] / max(lv["dev_top1"], 1e-9)
        g_cor = vm["dev_corr"] / max(lv["dev_corr"], 1e-9)
        if g_top >= args.gate_margin and g_cor >= args.gate_margin:
            tier = "strong"
        elif g_top >= args.rank_margin and g_cor >= args.rank_corr_floor:
            tier = "rank"
        else:
            tier = "fail"
        report["head"][lab] = {"config": cname, "variant": "whitened" if wh else "raw",
                               "feat": r["feat"], "val": vm, "test": tm, "linear_feat": lk[0],
                               "linear_val": lv, "linear_test": lt,
                               "val_gain_top1": g_top, "val_gain_corr": g_cor,
                               "epoch": pm["epoch"], "params": r["params"],
                               "log_sigma": r["learned_log_sigma"].get(lab)}
        report["gate"][lab] = {"tier": tier, "passed": tier != "fail",
                               "strong": tier == "strong",
                               "margin_required": args.gate_margin,
                               "rank_margin": args.rank_margin,
                               "rank_corr_floor": args.rank_corr_floor,
                               "val_gain_top1": g_top, "val_gain_corr": g_cor}
        log(f"[align] {lab}: BEST {cname}/{report['head'][lab]['variant']} "
            f"[{r['feat']}] ({r['params']/1e3:.0f}k, ep {pm['epoch']})  "
            f"val dev_corr {vm['dev_corr']:.4f} vs linear "
            f"{lv['dev_corr']:.4f} [{lk[0]}] (x{g_cor:.2f})  "
            f"val dev_top1 {vm['dev_top1']*200:.1f}x vs {lv['dev_top1']*200:.1f}x "
            f"(x{g_top:.2f})  -> {tier.upper()}")
        log(f"        test dev_corr {tm['dev_corr']:.4f} vs linear {lt['dev_corr']:.4f}   "
            f"test dev_top1 {tm['dev_top1']*200:.1f}x vs {lt['dev_top1']*200:.1f}x")

        # Emitted unconditionally: this bank is an experiment for the generation arms to
        # judge, not something to be settled in embedding space.
        m = gt_meta[lab]["m_tr"]
        a = solve_scale_for_share(m, l2n(pm["test"]), gt_meta[lab]["gt_energy_dev"])
        cond = l2n(m + l2n(pm["test"]) * a)
        p = outdir / f"joint_{lab}1024_{stag}_test.npy"
        np.save(p, cond)
        e_m, e_d, _, _ = energy_split(cond)
        cm = dev_metrics(cond, gt_meta[lab]["Gte"])
        report["emitted"][lab] = {"path": str(p), "scale": float(a), "tier": tier,
                                  "energy_mean": e_m, "energy_dev": e_d,
                                  "config": cname, "variant": report["head"][lab]["variant"],
                                  "feat": r["feat"], **cm}
        log(f"        emitted {p.name} (x{a:.2f}; energy dev {e_d:.3f} vs GT "
            f"{gt_meta[lab]['gt_energy_dev']:.3f}; bank dev_corr {cm['dev_corr']:.4f})")

    report["any_passed"] = any(v["passed"] for v in report["gate"].values())
    report["any_strong"] = any(v["strong"] for v in report["gate"].values())
    out = Path(args.out) if args.out else (outdir / f"joint_{stag}_report.json")
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    log(f"[align] wrote {out}   tiers: "
        + ", ".join(f"{k}={v['tier']}" for k, v in report["gate"].items()))

    log("")
    log("[align] grid, ranked by val dev_corr per modality "
        "(linear control shown first in each block)")
    for lab in labs:
        lk = max((k for k in lin if k[1] == lab), key=lambda k: lin[k]["val"]["dev_corr"])
        lv = lin[lk]["val"]
        log(f"  -- {lab}: linear [{lk[0]}, {lin[lk]['dim']}-d] val_corr {lv['dev_corr']:.4f} "
            f"val_top1 {lv['dev_top1']*200:.1f}x   <-- the bar to beat")
        bestc = report["head"].get(lab, {}).get("config")
        bestv = report["head"].get(lab, {}).get("variant")
        for r in report["grid"].get(lab, []):
            mk = " ***" if (r["config"] == bestc and r["variant"] == bestv) else ""
            sig = r.get("log_sigma")
            sigs = f" ls={sig:+.2f}" if sig is not None else ""
            log(f"     {r['config']:<22} {r['variant']:<8} [{r['feat']:<6}] "
                f"val_corr {r['val_dev_corr']:.4f} ({r['val_dev_corr']/lv['dev_corr']:.2f}x) "
                f" val_top1 {r['val_dev_top1']*200:>5.1f}x  ep {r['epoch']:>3}  "
                f"{r['params']/1e3:>5.0f}k{sigs}{mk}")


if __name__ == "__main__":
    main()
