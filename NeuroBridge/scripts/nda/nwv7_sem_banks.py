#!/usr/bin/env python
"""NW-v7: rebuild the IMAGE/SEMANTIC condition bank -- the measured bottleneck.

THE DIAGNOSIS THIS RESPONDS TO (NW-v6 2x2, job 589824, cancelled mid-M6)
-----------------------------------------------------------------------
Holding the generator fixed and crossing image x structure:

    arm                              image          structure     incep
    A5_band_cc3                      SEM (a_hi)     GT            0.8400
    W5_jimg_gtstruct_s30             JOINT img      GT            0.8521
    W4_gtimg_jstruct_s30             GT             JOINT         0.9557
    A9_orc_cc3                       GT             GT            0.9811
    W2_joint_struct_s30 (deployable) SEM/JOINT      JOINT         ~0.744

With image pinned at GT, our predicted structure recovers 82% of the oracle
structure gap (W4 - A5 = +0.116 of a possible A9 - A5 = +0.141).  With structure
pinned at GT, swapping SEM for JOINT image moves only +0.012.  The remaining gap
to CogCapPro-like numbers is therefore the IMAGE branch -- by about 4.5x.

The encoder-side stack of NW-v6 (shared trunk, SCM-Loss, fusion, uncertainty) made
every modality STRONG on validation and did NOT move deployable generation.  SCM
was a zero result (depth/edge preferred w=0).  Fusion helped val, not test --
`specific_s` carries same-session structure that inflates held-out-concept val
inside the training session.  That path is closed; this script opens the image one.

WHAT WAS ALSO RULED OUT
-----------------------
Self-conditioning by re-injecting CLIP(pass-1 generations) as the image IP-Adapter
condition.  Measured against GT CLIP-img (ViT-H-14 encode_image):

    bank / source              top1    rowcos-to-GT
    SEM a_hi                   0.180   0.571
    CLIP(W2 gens)              0.155   0.396     <- WORSE than SEM
    CLIP(G3 gens)              0.125   0.395
    CLIP(M6v6 gens)            0.210   0.443     <- still worse than SEM
    CLIP(W4 gens) [GT image]   0.820   0.602     <- only with GT image input

Pass-1 CLIP is farther from GT than the EEG-derived SEM bank already is.  Reusing
it as a condition would move us away from the target.  Closed.

WHAT THIS SCRIPT BUILDS
-----------------------
Three families of image banks, all 1024-d, all variance-restored onto the GT TRAIN
mean so they sit on the same manifold the IP-Adapter was trained for (the operator
that took structure from "harmful" to "useful" in NW-v5):

  (A) IDENTITY TRANSPLANT from stronger retrieval banks we already have but never
      fed to this generator.  cfmsf q_img has top-1 0.305 (61x chance) vs a_hi's
      0.180 (36x), but sits off-manifold (E[dev]=0.85 vs GT 0.61, rowcos 0.31).
      Transplant = GT_train_mean + a * l2n(bank - bank.mean), with `a` chosen so
      the emitted bank matches the GT test energy split.  Same for uge.

  (B) BLENDS of cfmsf direction with a_hi (on-manifold) before the same restore.
      Retrieval identity from (A), concentration prior from a_hi.

  (C) A DEDICATED image head -- the NW-v6 joint trunk compromised image for
      structure (JOINT top1 40x but lost generation when paired with predicted
      structure).  Image-only, cat features, low-rank + dropout, deviation target,
      optional SCM.  Selected on val_b concepts by image-only val_dev_corr.

Nothing here touches the generator.  The generation arms in run_nwv7_s08.sh pair
each bank with the already-verified G3 structure head (and a JOINT-structure
control) under the locked cc3 + band-anchor + Turbo+CFG0 operator.
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


def energy_split(x: np.ndarray) -> tuple[float, float]:
    x = l2n(x)
    m = x.mean(0, keepdims=True)
    d = x - m
    return float((m ** 2).sum()), float((d ** 2).sum(1).mean())


def dev_metrics(pred: np.ndarray, tgt: np.ndarray) -> dict:
    p, t = l2n(pred), l2n(tgt)
    p = l2n(p - p.mean(0, keepdims=True))
    t = l2n(t - t.mean(0, keepdims=True))
    S = p @ t.T
    return {"dev_top1": float((S.argmax(1) == np.arange(len(t))).mean()),
            "dev_corr": float((p * t).sum(1).mean())}


def full_metrics(pred: np.ndarray, tgt: np.ndarray) -> dict:
    p, t = l2n(pred), l2n(tgt)
    S = p @ t.T
    em, ed = energy_split(p)
    d = dev_metrics(p, t)
    return {"top1": float((S.argmax(1) == np.arange(len(t))).mean()),
            "rowcos": float((p * t).sum(1).mean()),
            "energy_mean": em, "energy_dev": ed, **d}


def solve_scale(m_unit: np.ndarray, dev: np.ndarray, target_share: float) -> float:
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


def transplant(src: np.ndarray, m_tr: np.ndarray, target_ed: float) -> tuple[np.ndarray, float]:
    """GT-train-mean + scaled source deviation, restored to target energy share."""
    dev = l2n(src - src.mean(0, keepdims=True))
    a = solve_scale(m_tr, dev, target_ed)
    return l2n(m_tr + dev * a), float(a)


def principal_basis(d: np.ndarray, fit_i: np.ndarray, rank: int) -> np.ndarray:
    x = d[fit_i].astype(np.float64)
    x = x - x.mean(0, keepdims=True)
    c = (x.T @ x) / max(len(x) - 1, 1)
    w, v = np.linalg.eigh(c)
    return v[:, np.argsort(w)[::-1][:rank]].astype(np.float32)


def load_cat(sid: int) -> tuple[np.ndarray, np.ndarray]:
    intra = NB_ROOT / f"outputs/ocf/intra_z/sub-{sid:02d}"
    chab = NB_ROOT / f"outputs/chab/sub-{sid:02d}/z_warm0/sub-{sid:02d}"
    sh_tr = l2n(np.load(intra / "shared_r_train.npy").astype(np.float32))
    sh_te = l2n(np.load(intra / "shared_r_test.npy").astype(np.float32))
    ref = l2n(np.load(chab / "shared_r_train.npy").astype(np.float32))
    cos = float((ref * sh_tr).sum(1).mean())
    if cos < 0.999:
        raise SystemExit(f"[FATAL] shared_r mismatch cos={cos:.4f}")
    sp_tr = l2n(np.load(chab / "specific_s_train.npy").astype(np.float32))
    sp_te = l2n(np.load(chab / "specific_s_test.npy").astype(np.float32))
    return np.concatenate([sh_tr, sp_tr], 1), np.concatenate([sh_te, sp_te], 1)


def train_dedicated(torch, ztr, zte, ytr_dev, gte, fit_i, val_i, concepts, args, log):
    """Image-only head.  Capacity is not shared with depth/edge."""
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    CFGS = [
        {"name": "h256r512d4_s0.0", "hidden": 256, "rank": 512, "dropout": 0.4, "wd": 1e-2, "lr": 1e-3, "tau": 0.1, "scm": 0.0},
        {"name": "h256r512d4_s0.5", "hidden": 256, "rank": 512, "dropout": 0.4, "wd": 1e-2, "lr": 1e-3, "tau": 0.1, "scm": 0.5},
        {"name": "h512r256d3_s0.3", "hidden": 512, "rank": 256, "dropout": 0.3, "wd": 1e-2, "lr": 1e-3, "tau": 0.1, "scm": 0.3},
        {"name": "h128r768d4_s0.5", "hidden": 128, "rank": 768, "dropout": 0.4, "wd": 1e-2, "lr": 1e-3, "tau": 0.1, "scm": 0.5},
    ]
    best = None
    grid = []
    Z = torch.tensor(ztr, device=dev)
    fit_t = torch.tensor(fit_i, device=dev)
    val_t = torch.tensor(val_i, device=dev)
    con = torch.tensor(concepts, device=dev)
    Yfit_raw = torch.tensor(l2n(ytr_dev[fit_i]), device=dev)
    std = ytr_dev[fit_i].std(0, keepdims=True) + 1e-6
    Yfit_w = torch.tensor(l2n(ytr_dev[fit_i] / std), device=dev)
    std_t = torch.tensor(std, device=dev)

    for cfg in CFGS:
        for whiten in (0, 1):
            tag = f"{cfg['name']}/{'whit' if whiten else 'raw'}"
            basis = principal_basis(ytr_dev, fit_i, cfg["rank"])
            B = torch.tensor(basis, device=dev)
            layers = [torch.nn.Linear(ztr.shape[1], cfg["hidden"]), torch.nn.LayerNorm(cfg["hidden"]),
                      torch.nn.GELU(), torch.nn.Dropout(cfg["dropout"]),
                      torch.nn.Linear(cfg["hidden"], cfg["hidden"]), torch.nn.LayerNorm(cfg["hidden"]),
                      torch.nn.GELU(), torch.nn.Dropout(cfg["dropout"]),
                      torch.nn.Linear(cfg["hidden"], cfg["rank"])]
            model = torch.nn.Sequential(*layers).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
            n_fit = len(fit_i)
            spe = max(1, n_fit // args.batch)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * spe)
            Yfit = Yfit_w if whiten else Yfit_raw
            eye_cache = {}
            best_c, best_state, best_ep, bad = -1.0, None, -1, 0
            torch.manual_seed(args.seed)
            for ep in range(args.epochs):
                model.train()
                perm = rng.permutation(n_fit)
                for s in range(spe):
                    sel = perm[s * args.batch:(s + 1) * args.batch]
                    if len(sel) < 8:
                        continue
                    pred = model(Z[fit_t[sel]]) @ B.T
                    p = torch.nn.functional.normalize(pred, dim=-1)
                    tgt = Yfit[sel]
                    logits = (p @ tgt.T) / cfg["tau"]
                    bsz = logits.shape[0]
                    eye = eye_cache.get(bsz)
                    if eye is None:
                        eye = torch.eye(bsz, dtype=torch.bool, device=dev)
                        eye_cache[bsz] = eye
                    log_denom = torch.logsumexp(logits.masked_fill(eye, -float("inf")), dim=1)
                    hard = log_denom - torch.diagonal(logits)
                    w = cfg["scm"]
                    if w > 0:
                        sel_c = con[fit_t[sel]]
                        pos = (sel_c[:, None] == sel_c[None, :]) & ~eye
                        has = pos.any(1)
                        log_num = torch.where(
                            has,
                            torch.logsumexp(logits.masked_fill(~pos, -float("inf")), dim=1),
                            torch.diagonal(logits))
                        multi = log_denom - log_num
                        loss = ((1 - w) * hard + w * multi).mean()
                    else:
                        loss = hard.mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sched.step()
                model.eval()
                with torch.no_grad():
                    o = model(Z[val_t]) @ B.T
                    if whiten:
                        o = o * std_t
                    o = o.cpu().numpy()
                vm = dev_metrics(o, ytr_dev[val_i])
                if vm["dev_corr"] > best_c:
                    best_c, best_ep, bad = vm["dev_corr"], ep, 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    best_vm = vm
                else:
                    bad += 1
                if bad >= args.patience:
                    break
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                ote = model(torch.tensor(zte, device=dev)) @ B.T
                if whiten:
                    ote = ote * std_t
                ote = ote.cpu().numpy()
            tm = full_metrics(ote, gte)
            n_par = sum(p.numel() for p in model.parameters())
            row = {"config": cfg["name"], "variant": "whitened" if whiten else "raw",
                   "val": best_vm, "test_raw": tm, "epoch": best_ep, "params": n_par,
                   "pred": ote}
            grid.append({k: v for k, v in row.items() if k != "pred"})
            log(f"  {tag}: val_corr {best_vm['dev_corr']:.4f}@{best_ep}  "
                f"test top1 {tm['top1']*200:.1f}x rowcos {tm['rowcos']:.4f}  "
                f"({n_par/1e3:.0f}k)")
            if best is None or best_vm["dev_corr"] > best["val"]["dev_corr"]:
                best = row
    return best, grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--stag", type=str, default="sub-08")
    ap.add_argument("--out-dir", type=str, default=str(NB_ROOT / "outputs/nw7_s08/conds"))
    ap.add_argument("--cond-cache", type=str, default=str(NB_ROOT / "outputs/gem/cond_cache"))
    ap.add_argument("--split-json", type=str, default=str(NB_ROOT / "outputs/leakfree/split.json"))
    ap.add_argument("--sem-ahi", type=str,
                    default=str(NB_ROOT / "outputs/nw4_10s/arms/a_hi/conds/sub-08/cal_test.npy"))
    ap.add_argument("--cfmsf", type=str,
                    default=str(NB_ROOT / "outputs/cfmsf_s08/train/conds/q_img_test.npy"))
    ap.add_argument("--uge", type=str,
                    default=str(NB_ROOT / "outputs/uge/sub-08/full/conds/ip_img_test.npy"))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--skip-train", type=int, default=0)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    def log(s: str) -> None:
        print(s, flush=True)

    sid, stag = args.subject, args.stag
    cc, outdir = Path(args.cond_cache), Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    Gtr = l2n(np.load(cc / "clip_img1024_train.npy").astype(np.float32))
    Gte = l2n(np.load(cc / "clip_img1024_test.npy").astype(np.float32))
    m_tr = l2n(Gtr.mean(0, keepdims=True))
    _, target_ed = energy_split(Gte)
    log(f"[sem] GT test energy mean={energy_split(Gte)[0]:.3f} dev={target_ed:.3f}")

    banks = {}
    # ---- (A) identity transplants ------------------------------------------------
    for name, path in [("cfmsf", args.cfmsf), ("uge", args.uge), ("ahi", args.sem_ahi)]:
        p = Path(path)
        if not p.is_file():
            log(f"[sem] WARN missing {name}: {p}")
            continue
        src = np.load(p).astype(np.float32)
        cond, a = transplant(src, m_tr, target_ed)
        banks[f"sem_{name}_varest"] = cond
        log(f"[sem] {name}_varest x{a:.2f}  {full_metrics(cond, Gte)}")

    # ---- (B) blends --------------------------------------------------------------
    if "sem_cfmsf_varest" in banks and Path(args.sem_ahi).is_file():
        cf = l2n(np.load(args.cfmsf).astype(np.float32))
        ah = l2n(np.load(args.sem_ahi).astype(np.float32))
        for w in (0.3, 0.5, 0.7):
            mix = l2n(w * cf + (1.0 - w) * ah)
            cond, a = transplant(mix, m_tr, target_ed)
            banks[f"sem_blend_c{int(w*10)}"] = cond
            log(f"[sem] blend cfmsf*{w:.1f}+ahi  x{a:.2f}  {full_metrics(cond, Gte)}")

    # ---- (C) dedicated image head ------------------------------------------------
    report = {"subject": sid, "stag": stag, "banks": {}, "dedicated": None}
    if not args.skip_train:
        import torch
        sp = json.loads(Path(args.split_json).read_text())
        fit_i = np.asarray(sp["fit_rows"], dtype=int)
        val_i = np.asarray(sp.get("val_b_rows") or sp["val_a_rows"], dtype=int)
        concepts = (np.arange(len(Gtr)) // 10).astype(np.int64)
        # verify block ordering
        S = Gtr[:200] @ Gtr[:200].T
        wi = float(np.mean([S[b*10:(b+1)*10, b*10:(b+1)*10][~np.eye(10, dtype=bool)].mean()
                            for b in range(6)]))
        wo = float(np.mean([S[b*10:(b+1)*10][:, np.r_[0:b*10, (b+1)*10:200]].mean()
                            for b in range(6)]))
        log(f"[sem] concept blocks within {wi:.4f} between {wo:.4f}")
        if wi < wo + 0.1:
            raise SystemExit("[FATAL] train rows not block-ordered by concept")
        ztr, zte = load_cat(sid)
        ytr_dev = (Gtr - Gtr.mean(0, keepdims=True)).astype(np.float32)
        log("[sem] dedicated image-only head (cat features, deviation target)")
        t0 = time.time()
        best, grid = train_dedicated(torch, ztr, zte, ytr_dev, Gte, fit_i, val_i,
                                     concepts, args, log)
        cond, a = transplant(best["pred"], m_tr, target_ed)
        banks["sem_dedicated"] = cond
        report["dedicated"] = {
            "config": best["config"], "variant": best["variant"],
            "val": best["val"], "test_before_varest": best["test_raw"],
            "test_after_varest": full_metrics(cond, Gte),
            "scale": a, "params": best["params"], "secs": time.time() - t0,
            "grid": grid,
        }
        log(f"[sem] dedicated WINNER {best['config']}/{best['variant']}  "
            f"varest x{a:.2f}  {report['dedicated']['test_after_varest']}")

    # ---- emit --------------------------------------------------------------------
    for name, arr in banks.items():
        p = outdir / f"{name}_{stag}_test.npy"
        np.save(p, arr.astype(np.float32))
        m = full_metrics(arr, Gte)
        report["banks"][name] = {"path": str(p), **m}
        log(f"[sem] wrote {p.name}  top1 {m['top1']*200:.1f}x  "
            f"dev_corr {m['dev_corr']:.4f}  rowcos {m['rowcos']:.4f}  "
            f"Edev {m['energy_dev']:.3f}")

    # rank by a generation-proxy: 0.5*z(top1) + 0.5*z(rowcos), both vs a_hi baseline
    if "sem_ahi_varest" in report["banks"]:
        base = report["banks"]["sem_ahi_varest"]
        ranked = []
        for n, m in report["banks"].items():
            score = 0.5 * (m["top1"] / max(base["top1"], 1e-9)) + 0.5 * (m["rowcos"] / max(base["rowcos"], 1e-9))
            ranked.append((score, n, m))
        ranked.sort(key=lambda t: -t[0])
        report["ranked"] = [{"name": n, "score": s, **m} for s, n, m in ranked]
        log("[sem] ranked (0.5 top1-gain + 0.5 rowcos-gain vs ahi_varest):")
        for s, n, m in ranked:
            log(f"  {n:<22} score {s:.3f}  top1 {m['top1']*200:>5.1f}x  "
                f"rowcos {m['rowcos']:.4f}  dev_corr {m['dev_corr']:.4f}")

    out = Path(args.out) if args.out else (outdir / f"sem_{stag}_report.json")
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    log(f"[sem] wrote {out}")


if __name__ == "__main__":
    main()
