"""POLARIS on CogCapPro: one LOSO fold end to end.

Stages, in order:

  1. train the source subjects (stage 1 modality branches, stage 2 joint, stage 3 fusion)
     with the repaired multi-positive loss, plus L_spec and L_aug if weighted in
  2. pick the checkpoint on the *source* subjects' validation split (never the target's)
  3. fit the deployment recovery on the held-out subject's **unlabelled train split**
     against the 16540 training conditions, then freeze it
  4. report the ladder on the held-out subject's 200-way test split

Step 3 is the part that has to be right for the numbers to mean anything. The gallery is
the training split, which is disjoint from the 200 test stimuli, so a map fitted this way
cannot have memorised the answers; the anchors are mutual nearest neighbours, so the
correspondence it is fitted on is discovered rather than supplied.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cogcap import config                                        # noqa: E402
from cogcap.data import LOSOData                                 # noqa: E402
from cogcap.losses import (                                      # noqa: E402
    clip_loss_multi_positive,
    rotation_aug_loss,
    sample_modality_mask,
    spectral_flatness_loss,
)
from cogcap.model import CogCapPro                               # noqa: E402
from cogcap import recover as R                                  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 9, 10])
    p.add_argument("--target", type=int, default=8)
    p.add_argument("--modalities", default=None)
    p.add_argument("--out-dir", default=None)

    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--stage1-epochs", type=int, default=20)
    p.add_argument("--stage3-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)

    # POLARIS training-side knobs
    p.add_argument("--sea", dest="sea", action="store_true", default=True,
                   help="sensor-space subject operator (design doc §3.1)")
    p.add_argument("--no-sea", dest="sea", action="store_false")
    p.add_argument("--subject-wise", default="time", choices=["time", "none"],
                   help="upstream's per-subject time-axis Linear, or nothing")
    p.add_argument("--repair-topk", dest="repair_topk", action="store_true", default=True,
                   help="union instead of upstream's intersection (design doc §2.4)")
    p.add_argument("--no-repair-topk", dest="repair_topk", action="store_false")
    p.add_argument("--top-k", type=int, default=config.TOP_K)
    p.add_argument("--spec-weight", type=float, default=0.1, help="L_spec weight")
    p.add_argument("--aug-weight", type=float, default=0.1, help="L_aug weight")
    p.add_argument("--mask-count", type=int, default=1)
    p.add_argument("--fusion-detach", action="store_true", default=False)

    # recovery knobs
    p.add_argument("--recovery", dest="recovery", action="store_true", default=True)
    p.add_argument("--no-recovery", dest="recovery", action="store_false")
    p.add_argument("--saw", dest="saw", action="store_true", default=True)
    p.add_argument("--no-saw", dest="saw", action="store_false")
    p.add_argument("--rho", type=float, default=0.1)
    p.add_argument("--rec-k", type=int, default=10)
    p.add_argument("--min-votes", type=int, default=2)
    p.add_argument("--max-landmarks", type=int, default=160)
    p.add_argument("--coverage-rank", type=int, default=64)
    p.add_argument("--pair-mode", default="mnn", choices=["mnn", "oracle"])
    p.add_argument("--h1", dest="h1", action="store_true", default=True,
                   help="also fit the single sensor-space operator (AB-1)")
    p.add_argument("--no-h1", dest="h1", action="store_false")
    p.add_argument("--h1-steps", type=int, default=300)

    # smoke / plumbing
    p.add_argument("--limit-samples", type=int, default=0)
    p.add_argument("--recovery-rows", type=int, default=0)
    p.add_argument("--feature-suffix", default="",
                   help="e.g. _smoke, to read truncated condition arrays")
    p.add_argument("--device", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def resolve_out(args) -> Path:
    if args.out_dir:
        return Path(args.out_dir)
    tag = f"polaris_sub{args.target:02d}"
    extra = []
    if not args.sea:
        extra.append("nosea")
    if not args.repair_topk:
        extra.append("noRepair")
    if args.spec_weight:
        extra.append(f"spec{args.spec_weight:g}")
    if args.aug_weight:
        extra.append(f"aug{args.aug_weight:g}")
    if args.limit_samples:
        extra.append("smoke")
    return config.COGCAP_OUT / (tag + ("_" + "_".join(extra) if extra else ""))


def normalise_targets(mod: dict, modalities) -> dict:
    """CogCapPro normalises its conditioning targets before projecting
    (`training/module.py:203-207`); the same choice here keeps the `mod_z` side identical."""
    return {m: F.normalize(mod[m], dim=1) for m in modalities}


def run_epoch(model, data, opt, sched, args, device, modalities, epoch: int, stage: str,
              rng, log):
    model.train()
    tot, nb = 0.0, 0
    diag_acc = {}
    for bi, batch in enumerate(data.batches(
            "train", args.batch_size, shuffle=True, seed=args.seed * 1000 + epoch)):
        eeg = batch["eeg"].to(device)
        subj = batch["subj"].to(device)
        stim = batch["stim"].to(device)
        mod = {m: batch["mod"][m].to(device) for m in modalities}
        mod_n = normalise_targets(mod, modalities)

        mask = sample_modality_mask(len(modalities), args.mask_count, device) if model.use_fusion else set()
        out = model(eeg, [mod_n[m] for m in modalities], subject_ids=subj,
                    mask_modalities=mask)
        z, mod_z, ls = out["z"], out["mod_z"], out["logit_scale"]

        keys = list(modalities) + (["fusion"] if model.use_fusion else [])
        if stage == "stage1":
            keys = list(modalities)
        elif stage == "stage3":
            keys = ["fusion"] if model.use_fusion else list(modalities)

        loss = 0.0
        for k in keys:
            lk, _, d = clip_loss_multi_positive(
                z[k], mod_z[k], ls, stim, top_k=args.top_k,
                cos_batch=config.COS_BATCH, repair_topk=args.repair_topk,
                want_diag=(bi == 0))
            loss = loss + lk
            if bi == 0 and d:
                for kk, vv in d.items():
                    diag_acc[f"{k}/{kk}"] = vv
        loss = loss / max(len(keys), 1)

        if args.spec_weight:
            zmean = torch.stack([z[m] for m in modalities], 0).mean(0)
            lspec = spectral_flatness_loss(zmean, stim, subj, rank=args.coverage_rank)
            if lspec is not None:
                loss = loss + args.spec_weight * lspec

        if args.aug_weight:
            def fwd(x, ids):
                zs = model.brain(x, ids)
                return torch.stack(zs, 0).mean(0)
            laug = rotation_aug_loss(fwd, eeg, subj, stim, subj, eeg.shape[1], generator=rng)
            if laug is not None:
                loss = loss + args.aug_weight * laug

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched is not None:
            sched.step()
        tot += float(loss.item())
        nb += 1
        if bi % 20 == 0:
            log(f"  {stage} e{epoch} b{bi} loss {float(loss.item()):.4f}")

    return tot / max(nb, 1), diag_acc


@torch.no_grad()
def validate(model, data, args, device, modalities) -> dict:
    model.eval()
    zs, gs = {m: [] for m in modalities}, {m: [] for m in modalities}
    for batch in data.batches("val", 512, shuffle=False):
        eeg = batch["eeg"].to(device)
        out = model(eeg, [batch["mod"][m].to(device) for m in modalities], subject_ids=batch["subj"].to(device))
        for m in modalities:
            zs[m].append(out["z"][m].float().cpu())
            gs[m].append(batch["mod"][m].float())
    n_val = data.test_eeg.shape[0]
    # val rows are [subject-major, stimulus]; reduce to per-stimulus mean over subjects
    n_sub = data.S
    res = {}
    for m in modalities:
        z = torch.cat(zs[m], 0).reshape(n_sub, n_val, -1).mean(0)
        g = torch.cat(gs[m], 0).reshape(n_sub, n_val, -1)[0]
        s = F.normalize(z, dim=1) @ F.normalize(g, dim=1).t()
        order = s.argsort(dim=1, descending=True)
        tgt = torch.arange(n_val)
        res[m] = {
            "top1": float((order[:, 0] == tgt).float().mean() * 100),
            "top5": float((order[:, :5] == tgt[:, None]).any(1).float().mean() * 100),
        }
    return res


# ============================================================ recovery
@torch.no_grad()
def moment_affine(q: torch.Tensor, g: torch.Tensor, eps: float = 1e-5):
    scale = (g.std(0, unbiased=False) + eps) / (q.std(0, unbiased=False) + eps)
    shift = g.mean(0) - q.mean(0) * scale
    return scale, shift


@torch.no_grad()
def run_recovery(model, data, args, device, modalities, log) -> dict:
    model.eval()
    brain = model.brain
    rep = {}

    fit_eeg = data.recovery_eeg
    test_eeg = data.test_eeg
    # The gallery is the training conditions, so the fitting rows cannot exceed it. In a
    # full run both sides are the 16540 training stimuli and this is slack; it only binds
    # under --limit-samples, where the two caches are truncated independently.
    fit_eeg = fit_eeg[: data.n_stim]
    if args.limit_samples:
        fit_eeg = fit_eeg[: args.limit_samples]
        test_eeg = test_eeg[: min(args.limit_samples, test_eeg.shape[0])]

    # ---- SW: sensor-space subject-adaptive whitening
    if args.saw:
        w, mu = R.fit_saw(fit_eeg.to(device))
        fit_in = R.apply_saw(fit_eeg.to(device), w, mu)
        test_in = R.apply_saw(test_eeg.to(device), w, mu)
        log(f"[rec  ] SW fitted on {tuple(fit_eeg.shape)} unlabelled rows")
    else:
        fit_in, test_in = fit_eeg.to(device), test_eeg.to(device)

    # ---- gallery = the 16540 TRAINING conditions (disjoint from the 200 test stimuli)
    n_fit = fit_in.shape[0]
    gallery = {m: data.train_mod[m][:n_fit].to(device) for m in modalities}

    zq = R.branch_reps(brain, fit_in, modalities)
    zt = R.branch_reps(brain, test_in, modalities)
    baseline = R.evaluate_retrieval(zt, {m: data.test_mod[m].to(device) for m in modalities},
                                    modalities, torch.arange(test_in.shape[0]), config.TEST_WAY)
    rep["baseline_no_recovery"] = baseline
    log(f"[rec  ] baseline (SAW only) top1 " +
        " ".join(f"{m}={baseline[m]['cosine']['top1']:.2f}" for m in modalities))

    # ---- moment matching: orthogonal maps cannot express a translation
    scale, shift = {}, {}
    for m in modalities:
        s, h = moment_affine(zq[m], gallery[m])
        scale[m], shift[m] = s, h
        zq[m] = zq[m] * s + h
        zt[m] = zt[m] * s + h

    # ---- AS: anchors
    rows, votes, per_branch = R.vote_anchors(zq, gallery, modalities, k=args.rec_k,
                                             min_votes=args.min_votes)
    if rows.numel() == 0:
        log("[rec  ] no anchors passed the vote gate; recovery skipped")
        rep["anchors"] = {"n": 0}
        return rep

    cols = torch.zeros(rows.numel(), dtype=torch.long, device=device)
    if args.pair_mode == "oracle":
        cols = rows.clone()                       # CEILING ONLY: uses the true correspondence
    else:
        # majority-of-branches destination for the voted rows
        dest = []
        for m in modalities:
            s = R.csls_scores(F.normalize(zq[m], dim=1), F.normalize(gallery[m], dim=1),
                              k=args.rec_k)
            dest.append(s[rows].argmax(dim=1))
        cols = torch.stack(dest, 0).mode(dim=0).values
    rows, coverage = R.leverage_select(rows, zq["image" if "image" in modalities else modalities[0]],
                                       gallery["image" if "image" in modalities else modalities[0]],
                                       per_branch, args.coverage_rank, args.max_landmarks)
    cols = cols[: rows.numel()] if cols.numel() >= rows.numel() else cols

    # ---- RP: per-branch closed-form recovery
    maps = R.fit_per_branch(zq, gallery, rows, cols, modalities, rho=args.rho)
    defect = R.branch_defect(maps)

    rec_test = {}
    for m in modalities:
        r, mx, my = maps[m]
        rec_test[m] = R.apply_recovery(zt[m], r, mx, my)
    ladder = R.evaluate_retrieval(rec_test, {m: data.test_mod[m].to(device) for m in modalities},
                                 modalities, torch.arange(test_in.shape[0]), config.TEST_WAY)
    rep["per_branch"] = ladder

    # ---- H1: one operator in 63-d sensor space
    h1 = None
    if args.h1:
        try:
            rows_h = rows[: min(rows.numel(), 512)]
            cols_h = cols[: rows_h.numel()]
            op, hist = R.fit_sensor_operator(
                brain, fit_in, gallery, rows_h, cols_h, modalities,
                steps=args.h1_steps, log_every=max(1, args.h1_steps // 4))
            fixed = R.branch_reps(brain, op(test_in), modalities)
            for m in modalities:
                fixed[m] = fixed[m] * scale[m] + shift[m]
            h1 = {
                "loss_first": hist[0], "loss_last": hist[-1],
                "n_params": int((fit_in.shape[1] * (fit_in.shape[1] - 1)) // 2),
                "test": R.evaluate_retrieval(fixed, {m: data.test_mod[m].to(device) for m in modalities},
                                            modalities, torch.arange(test_in.shape[0]),
                                            config.TEST_WAY),
            }
            rep["h1_sensor_operator"] = h1
            log(f"[h1   ] loss {hist[0]:.5f} -> {hist[-1]:.5f} | top1 " +
                " ".join(f"{m}={h1['test'][m]['cosine']['top1']:.2f}" for m in modalities))
        except Exception as exc:                                   # noqa: BLE001
            log(f"[h1   ] failed: {type(exc).__name__}: {exc}")
            rep["h1_sensor_operator"] = {"error": f"{type(exc).__name__}: {exc}"}

    d = data.diagnostics()
    rep["anchors"] = {
        "n": int(rows.numel()),
        "vote_gate": args.min_votes,
        "coverage_min_frac": coverage,
        "defect": defect,
        "pair_mode": args.pair_mode,
        "rho": args.rho,
        "r_eff_target": args.coverage_rank,
        "m_anchor": int(rows.numel()),
        "identifiable": bool(rows.numel() >= args.coverage_rank),
    }
    rep["data"] = d
    return rep


def main() -> None:
    args = parse_args()
    modalities = ([m for m in args.modalities.split(",") if m] if args.modalities
                  else config.default_modalities())
    out = resolve_out(args)
    out.mkdir(parents=True, exist_ok=True)
    logf = out / "run.log"
    logf.write_text("")

    def log(msg: str) -> None:
        print(msg, flush=True)
        with logf.open("a") as fh:
            fh.write(msg + "\n")

    log(f"[info ] POLARIS/CogCapPro LOSO sources={args.sources} target={args.target} "
        f"modalities={modalities}")
    log(f"[info ] sea={args.sea} subject_wise={args.subject_wise} repair_topk={args.repair_topk} "
        f"spec={args.spec_weight} aug={args.aug_weight} recovery={args.recovery} saw={args.saw}")

    if args.dry_run:
        log("[ok   ] dry run: config parsed")
        (out / "dry_run.json").write_text(json.dumps(vars(args), indent=2, default=str))
        return

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log(f"[info ] device={device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.time()
    data = LOSOData(args.sources, args.target, modalities, limit=args.limit_samples,
                    seed=args.seed, target_recovery_rows=args.recovery_rows,
                    feature_suffix=args.feature_suffix)
    log(f"[data ] {data.diagnostics()}")

    model = CogCapPro(
        modalities, c_num=config.N_CHANNELS, timesteps=config.TIMESTEPS,
        subject_wise=args.subject_wise, n_subjects=len(args.sources),
        use_sea=args.sea, fusion=True, fusion_detach=args.fusion_detach,
    ).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[model] trainable params {n_par:,} | sea={model.sea is not None}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, data.train_eeg.shape[0] // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warm = max(1, steps_per_epoch * args.warmup_epochs)

    def lr_lambda(step):
        if step < warm:
            return step / warm
        prog = (step - warm) / max(total_steps - warm, 1)
        return 0.5 * (1 + np.cos(np.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    rng = torch.Generator(device=device).manual_seed(args.seed)

    stage3_start = max(args.epochs - args.stage3_epochs, args.stage1_epochs)
    history, best = [], None
    for epoch in range(args.epochs):
        if epoch < args.stage1_epochs:
            stage = "stage1"
        elif epoch >= stage3_start:
            stage = "stage3"
        else:
            stage = "stage2"
        loss, diag = run_epoch(model, data, opt, sched, args, device, modalities, epoch,
                               stage, rng, log)
        entry = {"epoch": epoch, "stage": stage, "loss": loss}
        if diag:
            entry["loss_diag"] = diag
        val = validate(model, data, args, device, modalities)
        entry["val"] = val
        history.append(entry)
        score = float(np.mean([val[m]["top1"] for m in modalities]))
        if best is None or score > best["score"]:
            best = {"score": score, "epoch": epoch, "val": val}
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "best": best, "history": history}, out / "best.pt")
        log(f"[epoch] {epoch:3d} {stage} loss {loss:.4f} "
            f"val_top1 " + " ".join(f"{m}={val[m]['top1']:.2f}" for m in modalities))

    torch.save({"model": model.state_dict(), "args": vars(args), "history": history},
               out / "last.pt")

    # LOSO protocol: the last epoch, not a target-selected best (design doc §7.4)
    ckpt = torch.load(out / "last.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    result = {
        "args": vars(args), "out_dir": str(out),
        "modalities": modalities, "trainable_params": n_par,
        "history": history, "best_val": best,
        "train_seconds": time.time() - t0,
        "protocol": {
            "fold": f"sources {args.sources} -> target {args.target}",
            "selection": "LAST epoch (source-subject protocol)",
            "test_way": config.TEST_WAY,
            "recovery_fit": "held-out subject's UNLABELLED train split vs 16540 training "
                            "conditions (disjoint from the 200 test stimuli)",
            "anchors": "mutual nearest neighbours, voted across branches (label-free)",
        },
    }

    if args.recovery:
        log("[rec  ] fitting deployment recovery on unlabelled target data")
        rep = run_recovery(model, data, args, device, modalities, log)
        result["recovery"] = rep
        for tag in ("baseline_no_recovery", "per_branch", "h1_sensor_operator"):
            if tag in rep and isinstance(rep[tag], dict) and "image" in rep.get(tag, {}):
                log(f"[rec  ] {tag}: " + " ".join(
                    f"{m}={rep[tag][m]['cosine']['top1']:.2f}" for m in modalities))

    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    log(f"[done ] wrote {out/'result.json'} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
