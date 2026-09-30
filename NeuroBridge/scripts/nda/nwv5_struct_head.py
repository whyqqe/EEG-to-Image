#!/usr/bin/env python
"""NW-v5 S1: the structural head, trained on the DEVIATION and gated against linear.

WHY THIS EXISTS
---------------
A5 (cc3 layout, no ControlNet, band anchor sigma 3.0, our semantic bank plus GROUND-TRUTH
depth/edge CLIP rows) is the arm we want:

    pixcorr 0.1980  ssim 0.3756  incep 0.8400  clip 0.9123  alex2 0.8248  alex5 0.9245

Its deployable twin E2 differs only in the two structure branches and reaches incep
0.7363 / clip 0.8215.  Splitting every bank into the part shared by all 200 trials and the
part that varies between them (nwv4_a5_gap.py) gives the whole story:

    target  bank                %E(mean)  %E(dev)  cos(m)  dev_corr  dev_top1
    depth   GT (A5)               0.568    0.432   1.000    1.000     200x
    depth   ridge varest (E2)     0.575    0.425   0.993    0.358      29x
    edge    GT (A5)               0.532    0.468   1.000    1.000     200x
    edge    ridge varest (E2)     0.522    0.478   0.991    0.345      44x
    image   a_hi SEM (shipped)    0.376    0.624   0.976    0.317      49x

Two facts decide what to build.

  (1) Variance restoration is already CORRECT and adds no information.  Matching the GT
      energy split to within 0.007 leaves dev_top1 at 29x/44x and dev_corr at 0.358/0.345
      -- both unchanged from the un-restored ridge.  It changes presentation, not content.

  (2) Re-targeting the linear fit at the deviation changes NOTHING about the content
      either: dev_corr stays 0.3599 (depth) and 0.3485 (edge) to four decimals.  Only the
      validation criterion improves.  So the linear map already extracts everything a
      linear map can, and the bottleneck is extraction, not presentation, not the loss.

The gain curve in dev_corr is the reason this is worth doing, and also the reason it might
fail.  Seven arms, one crossed variable (the structure branch):

    dev_corr  0.001 (C1 constant)      -> incep 0.6990   BELOW the no-structure floor
    dev_corr  0.002 (C3 shuffled)      -> incep 0.6829   BELOW
    dev_corr  0.016 (A8/E3, our UCK)   -> incep 0.7077   BELOW the floor 0.7116
    dev_corr  ----  (no structure)     -> incep 0.7116 / 0.7302
    dev_corr  0.352 (E2 ridge)         -> incep 0.7363   about 5% of the GT gain
    dev_corr  0.354 (C2 ridge)         -> incep 0.7082   zero
    dev_corr  1.000 (A5 GT)            -> incep 0.8400   +0.128 over the floor

Below 0.1 the branch is HARMFUL.  At 0.35 it buys about a twentieth of what the GT rows
buy.  Nothing has ever been generated between 0.35 and 1.0, so the shape of the rise is
inference, not measurement -- but the jump from 0.35 to 1.0 has to happen somewhere, and
a nonlinear head is the only mechanism left that could produce it.

WHAT IS TRAINED
---------------
Input is the same leak-free EEG feature every other head in this project consumes
(outputs/ocf/intra_z/sub-XX/shared_r_*.npy, 1024-d, averaged over the 4 EEG repetitions of
each image).  Targets are the GT CLIP rows for depth and edge MINUS the GT TRAIN centroid,
which is the only place a centroid may come from.

    d_i = GT_i - mean(GT_train)

Loss is InfoNCE over the batch, so the objective is "rank this trial's target above the
other 255 trials' targets" rather than "be close in L2" -- and by construction the target
carries no shared component, so there is no mean for the model to hide in.

THE OVERFITTING PROBLEM, AND WHAT THIS SCRIPT DOES ABOUT IT
-----------------------------------------------------------
The first version of this head was a plain 1024->1024->1024 MLP, about 2M parameters.  It
overfit immediately and unmistakably: training InfoNCE fell to 0.014 (near-perfect in-batch
ranking) while validation dev_corr went 0.2537 -> 0.1404 monotonically, and the BEST epoch
was the very first one evaluated.  At its best it scored 0.25/0.26 against the linear
ridge's 0.325/0.299 -- i.e. a 2M-parameter MLP did not even match a linear map on deviation
direction, because it spent its capacity memorising trials.

So the search below is a small regularisation grid rather than one architecture, scored on
the quantity that actually tracks generation gain (val dev_corr), with every epoch
evaluated so a very-early optimum can be found:

    hidden      512 / 1024
    dropout     0.0 / 0.2 / 0.4
    weight decay 1e-4 / 1e-2
    rank        the head predicts R coefficients and decodes through the top-R principal
                directions of the training deviations.  This is the important one: the
                deviation space is plausibly low-rank, and a rank-128 bottleneck cuts the
                parameter count by 8x while making the prediction a smooth combination of
                the directions that actually carry variance.
    whiten      targets divided by their per-dimension std over the fit rows, un-whitened
                at prediction time (CLIP dimensions differ wildly in scale)

Selection is per modality on validation dev_corr, using held-out CONCEPTS.  Nothing is
selected on test.  Test numbers are reported only after the choice is made.

THE GATE
--------
The linear ridge is fitted here too, on the same fit rows with alpha chosen on the same
validation rows, scored with the identical function, so the comparison is like for like.
A head is tiered 'strong' if both dev_top1 and dev_corr beat linear by a margin, 'rank' if
it substantially out-ranks linear while not losing on dev_corr, else 'fail'.

Head banks are emitted UNCONDITIONALLY regardless of tier.  The tier is a prediction, not
a decision: embedding metrics have already misled us once (dev_corr 0.352 bought ~5% of
the GT gain while dev_corr 1.0 bought +0.128), so the head's verdict comes from the
generation arms pointed at it.  If those arms also fail, the honest reading is that this
EEG carries no more structure information than a linear map reaches, the structure branch
should be dropped rather than tuned again, and the effort belongs on the semantic branch
(A9 - A5 = +0.141 incep, the same order of magnitude, with far more machinery already
built).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

NB_ROOT = Path("/project/peilab/why/NeuroBridge")


def l2n(x: np.ndarray) -> np.ndarray:
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def energy_split(x: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    x = l2n(x)
    m = x.mean(0, keepdims=True)
    d = x - m
    return (float((m ** 2).sum()), float((d ** 2).sum(1).mean()), l2n(m), l2n(d))


def dev_metrics(pred: np.ndarray, tgt: np.ndarray) -> dict:
    """Both metrics strip the batch mean from BOTH sides.

    This has to match nwv4_a5_gap.py exactly, because that script produced the numbers
    this head is judged against.  Skipping the mean removal is not cosmetic: on the same
    ridge bank dev_corr reads 0.72 instead of 0.36, because the shared component (which is
    not what the structure branch is for) dominates the cosine.  A predicted offset is not
    information about the trial and must not score as if it were.
    """
    p, t = l2n(pred), l2n(tgt)
    p = l2n(p - p.mean(0, keepdims=True))
    t = l2n(t - t.mean(0, keepdims=True))
    S = p @ t.T
    return {"dev_top1": float((S.argmax(1) == np.arange(len(t))).mean()),
            "dev_corr": float((p * t).sum(1).mean())}


def solve_scale_for_share(m_unit: np.ndarray, dev: np.ndarray, target_share: float) -> float:
    """cond = l2n(m + a*dev); find a so that mean_i |cond_i - mean(cond)|^2 == target."""
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


def ridge_baseline(ztr, ytr_dev, zte, fit_i, val_i, alphas):
    """Deviation-targeted ridge.  Alpha on val dev_corr.

    Returns (alpha, val_metrics, train_predictions, test_predictions).  Both banks are
    needed: val_i indexes the TRAIN bank, test is a separate 200-row bank.
    """
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
    return a, v, zf @ w, zf_te @ w


def principal_basis(d, fit_i, rank):
    """Top-`rank` principal directions of the fit-row deviations, (1024, rank)."""
    x = d[fit_i].astype(np.float64)
    x = x - x.mean(0, keepdims=True)
    c = (x.T @ x) / max(len(x) - 1, 1)
    w, v = np.linalg.eigh(c)
    v = v[:, np.argsort(w)[::-1][:rank]]
    return v.astype(np.float32)


def build_model(torch, d_in, d_hid, d_out, dropout):
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            layers = [torch.nn.Linear(d_in, d_hid), torch.nn.LayerNorm(d_hid),
                      torch.nn.GELU()]
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            layers += [torch.nn.Linear(d_hid, d_hid), torch.nn.LayerNorm(d_hid),
                       torch.nn.GELU()]
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            self.trunk = torch.nn.Sequential(*layers)
            self.out = torch.nn.Linear(d_hid, d_out)

        def forward(self, x):
            return self.out(self.trunk(x))

    return Head()


def train_one(torch, ztr, zte, dtr, fit_i, val_i, val_a_i, cfg, whiten, args, log, tag):
    """Train one (config, variant) and return the best-by-val-dev_corr predictions."""
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rank = cfg["rank"]
    basis = None
    if rank:
        basis = principal_basis(dtr, fit_i, rank)
        basis_t = torch.tensor(basis, device=dev)                       # (1024, R)

    model = build_model(torch, ztr.shape[1], cfg["hidden"],
                        rank if rank else 1024, cfg["dropout"]).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

    std = dtr[fit_i].std(0, keepdims=True) + 1e-6

    def make_targets(idx_np):
        t = dtr[idx_np]
        if whiten:
            t = t / std
        return torch.tensor(l2n(t), device=dev)

    def decode(raw):
        return raw @ basis_t.T if rank else raw

    Z = torch.tensor(ztr, device=dev)
    fit_t = torch.tensor(fit_i, device=dev)
    Yfit = make_targets(fit_i)
    Yval = make_targets(val_i)
    std_t = torch.tensor(std, device=dev)
    val_t = torch.tensor(val_i, device=dev)

    n_fit = len(fit_i)
    spe = max(1, n_fit // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * spe)

    best = {"corr": -1.0, "epoch": -1, "vm": None, "state": None}
    bad = 0
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(n_fit)
        tot, nb = 0.0, 0
        for s in range(spe):
            sel = perm[s * args.batch:(s + 1) * args.batch]
            if len(sel) < 8:
                continue
            p = torch.nn.functional.normalize(decode(model(Z[fit_t[sel]])), dim=-1)
            loss = torch.nn.functional.cross_entropy(
                p @ Yfit[sel].T / cfg["tau"], torch.arange(len(sel), device=dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()); nb += 1

        model.eval()
        with torch.no_grad():
            pv = decode(model(Z[val_t]))
        if whiten:
            pv = pv * std_t
        vm = dev_metrics(pv.cpu().numpy(), dtr[val_i])
        if vm["dev_corr"] > best["corr"]:
            best = {"corr": vm["dev_corr"], "epoch": ep, "vm": vm,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
        if args.verbose and ((ep + 1) % 10 == 0 or ep == 0):
            log(f"      {tag} ep {ep:>3} loss {tot/max(nb,1):.4f} "
                f"val_corr {vm['dev_corr']:.4f} val_top1 {vm['dev_top1']*200:.1f}x "
                f"best {best['corr']:.4f}@{best['epoch']}")
        if bad >= args.patience:
            break

    model.load_state_dict(best["state"])
    model.eval()

    def predict(arr):
        with torch.no_grad():
            p = decode(model(torch.tensor(arr, device=dev)))
        if whiten:
            p = p * std_t
        return p.cpu().numpy()

    log(f"  {tag}: {n_par/1e3:.0f}k params, stopped ep {ep}, "
        f"best val_corr {best['corr']:.4f}@{best['epoch']} "
        f"(val_top1 {best['vm']['dev_top1']*200:.1f}x)")
    return {"test": predict(zte), "val": predict(ztr[val_i]),
            "val_a": predict(ztr[val_a_i]), "vm": best["vm"],
            "epoch": best["epoch"], "params": n_par}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--stag", type=str, default="sub-08")
    ap.add_argument("--z-root", type=str, default=str(NB_ROOT / "outputs/ocf/intra_z"))
    ap.add_argument("--cond-cache", type=str, default=str(NB_ROOT / "outputs/gem/cond_cache"))
    ap.add_argument("--split-json", type=str, default=str(NB_ROOT / "outputs/leakfree/split.json"))
    ap.add_argument("--out-dir", type=str, default=str(NB_ROOT / "outputs/nw5_s08/conds"))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--modalities", type=str, default="depth,edge")
    ap.add_argument("--verbose", type=int, default=1)
    ap.add_argument("--gate-margin", type=float, default=1.10)
    ap.add_argument("--rank-margin", type=float, default=1.5)
    ap.add_argument("--rank-corr-floor", type=float, default=0.98)
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

    sp = json.loads(Path(args.split_json).read_text())
    fit_i = np.asarray(sp["fit_rows"], dtype=int)
    val_i = np.asarray(sp.get("val_b_rows") or sp["val_a_rows"], dtype=int)
    val_a_i = np.asarray(sp["val_a_rows"], dtype=int)

    zdir = Path(args.z_root) / f"sub-{sid:02d}"
    ztr = l2n(np.load(zdir / "shared_r_train.npy").astype(np.float32))
    zte = l2n(np.load(zdir / "shared_r_test.npy").astype(np.float32))
    log(f"[head] sub-{sid}  EEG {ztr.shape} -> {zte.shape}  fit={len(fit_i)} "
        f"val={len(val_i)} val_a={len(val_a_i)}  modalities={labs}")

    # CRITICAL: fit_i / val_i / val_a_i index the TRAIN bank (that is what the leak-free
    # split defines -- fit_rows 14890 and val_b_rows 820 both live inside the 16540 train
    # rows), so validation is scored on held-out CONCEPTS and never touches test.
    TGT, gt_meta = {}, {}
    for lab in labs:
        Gtr = l2n(np.load(cc / f"clip_{lab}1024_train.npy").astype(np.float32))
        Gte = l2n(np.load(cc / f"clip_{lab}1024_test.npy").astype(np.float32))
        m_tr = Gtr.mean(0, keepdims=True).astype(np.float32)
        TGT[lab] = (Gtr - m_tr).astype(np.float32)
        e_m, e_d, _, _ = energy_split(Gte)
        gt_meta[lab] = {"gt_energy_dev": e_d, "gt_energy_mean": e_m,
                        "m_tr": m_tr, "Gte": Gte}

    # ---- linear baseline, same fit rows and same val rows ----------------------
    alphas = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    lin, lin_pred = {}, {}
    log("[head] linear ridge baseline (deviation target, alpha on val dev_corr)")
    for lab in labs:
        a, v, _ptr, pte = ridge_baseline(ztr, TGT[lab], zte, fit_i, val_i, alphas)
        lin[lab], lin_pred[lab] = {"alpha": a, "val": v}, pte
        log(f"  {lab}: alpha {a:g}  val_dev_corr {v['dev_corr']:.4f}  "
            f"val_dev_top1 {v['dev_top1']*200:.1f}x")

    # ---- the regularisation grid ---------------------------------------------
    # The plain 2M-parameter MLP overfit inside 9 epochs and never matched the linear
    # map, so capacity is the thing being varied, not depth.
    CFGS = [
        {"name": "h1024_r0_do0.0",   "hidden": 1024, "rank": 0,   "dropout": 0.0, "wd": 1e-4, "lr": 1.2e-3, "tau": 0.07},
        {"name": "h512_r0_do0.3",    "hidden": 512,  "rank": 0,   "dropout": 0.3, "wd": 1e-2, "lr": 1.2e-3, "tau": 0.07},
        {"name": "h512_r128_do0.1",  "hidden": 512,  "rank": 128, "dropout": 0.1, "wd": 1e-2, "lr": 1.2e-3, "tau": 0.07},
        {"name": "h512_r256_do0.1",  "hidden": 512,  "rank": 256, "dropout": 0.1, "wd": 1e-2, "lr": 1.2e-3, "tau": 0.07},
        {"name": "h256_r512_do0.2",  "hidden": 256,  "rank": 512, "dropout": 0.2, "wd": 1e-2, "lr": 1e-3,  "tau": 0.1},
        # the three below bracket the best config found in the smoke run: a rank bottleneck
        # plus a small trunk beat the plain 3.1M-parameter MLP by 0.026 in val dev_corr
        {"name": "h128_r512_do0.2",  "hidden": 128,  "rank": 512, "dropout": 0.2, "wd": 1e-2, "lr": 1e-3,  "tau": 0.1},
        {"name": "h256_r512_do0.4",  "hidden": 256,  "rank": 512, "dropout": 0.4, "wd": 1e-2, "lr": 1e-3,  "tau": 0.1},
        {"name": "h256_r768_do0.2",  "hidden": 256,  "rank": 768, "dropout": 0.2, "wd": 1e-2, "lr": 1e-3,  "tau": 0.1},
    ]
    variants = [0, 1]

    results = {}
    for lab in labs:
        log(f"[head] ==== {lab}: {len(CFGS)} configs x {len(variants)} variants ====")
        for cfg in CFGS:
            for wh in variants:
                tag = f"{cfg['name']}/{'whit' if wh else 'raw'}"
                t0 = time.time()
                torch.manual_seed(args.seed)
                try:
                    r = train_one(torch, ztr, zte, TGT[lab], fit_i, val_i, val_a_i,
                                  cfg, wh, args, log, tag)
                except Exception as e:                              # noqa: BLE001
                    log(f"  {tag}: FAILED ({type(e).__name__}: {e})")
                    continue
                r["variant"] = "whitened" if wh else "raw"
                r["config"] = cfg["name"]
                r["secs"] = time.time() - t0
                results[(lab, cfg["name"], wh)] = r

    # ---- per-modality selection and the gate ---------------------------------
    report = {"subject": sid, "stag": stag, "linear": lin, "head": {}, "gate": {},
              "grid": {}}
    emitted = {}
    for lab in labs:
        grid = {k: v for k, v in results.items() if k[0] == lab}
        ranked = sorted(grid.items(), key=lambda kv: -kv[1]["vm"]["dev_corr"])
        report["grid"][lab] = [
            {"config": v["config"], "variant": v["variant"],
             "val_dev_corr": v["vm"]["dev_corr"], "val_dev_top1": v["vm"]["dev_top1"],
             "epoch": v["epoch"], "params": v["params"], "secs": v["secs"]}
            for k, v in ranked]
        if not ranked:
            report["gate"][lab] = {"tier": "fail", "passed": False, "strong": False}
            log(f"[head] {lab}: every config failed - no bank emitted")
            continue
        (bk, best) = ranked[0]
        vm = best["vm"]
        tm = dev_metrics(best["test"], gt_meta[lab]["Gte"])
        lin_te = dev_metrics(lin_pred[lab], gt_meta[lab]["Gte"])
        g_top = vm["dev_top1"] / max(lin[lab]["val"]["dev_top1"], 1e-9)
        g_cor = vm["dev_corr"] / max(lin[lab]["val"]["dev_corr"], 1e-9)
        if g_top >= args.gate_margin and g_cor >= args.gate_margin:
            tier = "strong"
        elif g_top >= args.rank_margin and g_cor >= args.rank_corr_floor:
            tier = "rank"
        else:
            tier = "fail"
        report["head"][lab] = {
            "config": best["config"], "variant": best["variant"], "val": vm,
            "test": tm, "linear_test": lin_te, "val_gain_top1": g_top,
            "val_gain_corr": g_cor, "epoch": best["epoch"], "params": best["params"]}
        report["gate"][lab] = {"tier": tier, "passed": tier != "fail",
                               "strong": tier == "strong",
                               "margin_required": args.gate_margin,
                               "rank_margin": args.rank_margin,
                               "rank_corr_floor": args.rank_corr_floor,
                               "val_gain_top1": g_top, "val_gain_corr": g_cor}
        log(f"[head] {lab}: BEST {best['config']}/{best['variant']} "
            f"({best['params']/1e3:.0f}k params, epoch {best['epoch']})  "
            f"val dev_corr {vm['dev_corr']:.4f} (linear {lin[lab]['val']['dev_corr']:.4f}, "
            f"x{g_cor:.2f})  val dev_top1 {vm['dev_top1']*200:.1f}x "
            f"(linear {lin[lab]['val']['dev_top1']*200:.1f}x, x{g_top:.2f})  -> {tier.upper()}")
        log(f"       test dev_corr {tm['dev_corr']:.4f} vs linear "
            f"{lin_te['dev_corr']:.4f}   test dev_top1 {tm['dev_top1']*200:.1f}x vs "
            f"{lin_te['dev_top1']*200:.1f}x")

        # Emitted unconditionally: the head bank is an EXPERIMENT to be measured by the
        # generation metrics, not something to be judged in embedding space.
        m = gt_meta[lab]["m_tr"]
        a = solve_scale_for_share(m, l2n(best["test"]), gt_meta[lab]["gt_energy_dev"])
        cond = l2n(m + l2n(best["test"]) * a)
        p = outdir / f"head_{lab}1024_{stag}_test.npy"
        np.save(p, cond)
        e_m, e_d, _, _ = energy_split(cond)
        cm = dev_metrics(cond, gt_meta[lab]["Gte"])
        emitted[lab] = {"path": str(p), "scale": float(a), "tier": tier,
                        "energy_mean": e_m, "energy_dev": e_d,
                        "dev_top1": cm["dev_top1"], "dev_corr": cm["dev_corr"]}
        log(f"  emitted {p.name}  (rescaled x{a:.2f}; energy dev {e_d:.3f} vs GT "
            f"{gt_meta[lab]['gt_energy_dev']:.3f}; bank dev_corr {cm['dev_corr']:.4f})")

    report["emitted"] = emitted
    report["any_passed"] = any(v["passed"] for v in report["gate"].values())
    report["any_strong"] = any(v["strong"] for v in report["gate"].values())
    out = Path(args.out) if args.out else (outdir / f"head_{stag}_report.json")
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    log(f"[head] wrote {out}   tiers: "
        + ", ".join(f"{k}={v['tier']}" for k, v in report["gate"].items()))

    # The grid table is the real result when everything fails: it says whether the linear
    # map is a ceiling or just a better-behaved point in the same space.
    log("")
    log("[head] regularisation grid (ranked by val dev_corr; linear baseline shown first)")
    for lab in labs:
        log(f"  -- {lab}: linear ridge val dev_corr {lin[lab]['val']['dev_corr']:.4f} "
            f"(test {dev_metrics(lin_pred[lab], gt_meta[lab]['Gte'])['dev_corr']:.4f})")
        for r in report["grid"].get(lab, []):
            log(f"     {r['config']:<16} {r['variant']:<8} val_corr {r['val_dev_corr']:.4f}  "
                f"val_top1 {r['val_dev_top1']*200:>5.1f}x  ep {r['epoch']:>3}  "
                f"{r['params']/1e3:>5.0f}k  {r['secs']:>4.0f}s")


if __name__ == "__main__":
    main()
