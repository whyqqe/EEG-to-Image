#!/usr/bin/env python
"""Measure the ACTUAL influence of each loss term on the shared EEG encoder.

A loss term whose *value* looks healthy can still be inert: what shapes the
representation is ``w * dL/dtheta``, not ``w * L``. A weight of 0.9 on a term whose
gradient is 1e-4 relative to the others is decoration, and it will read as "the
mechanism is on" in every log.

This script loads a trained checkpoint, rebuilds one real training batch, and reports
for each term (a) its weighted value and (b) the norm of its gradient w.r.t. the model
parameters, relative to the largest term.

It also reports the MMD kernel's bandwidth geometry, because an RBF-MMD over L2
normalised embeddings is only sensitive to subject shift when the bandwidth is
commensurate with the actual between-subject displacement.

Run:  python scripts/diag_term_influence.py --ckpt outputs/stage1/v3-sub-08/epoch049.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from samclip import config as C  # noqa: E402
from samclip.data import things_eeg  # noqa: E402
from samclip.losses import (clip_alignment_loss, cross_subject_loss, hsic_subject,  # noqa: E402
                            mmd_subject, vicreg_terms)
from samclip.losses.invariance import _median_sigma, _rbf  # noqa: E402
from samclip.models.samclip import build_model  # noqa: E402
from samclip.train import (LossWeights, Trainer, build_loaders, build_targets)  # noqa: E402


def grad_norm(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> float:
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    total = 0.0
    for g in gs:
        if g is not None:
            total += float(g.detach().pow(2).sum())
    return total ** 0.5


def kernel_geometry(z: torch.Tensor, subject: torch.Tensor) -> dict:
    """Within- vs between-subject displacement on the normalised sphere."""
    z = torch.nn.functional.normalize(z, dim=-1).detach()
    d2 = torch.cdist(z, z, p=2) ** 2
    sub = subject.view(-1, 1)
    same = (sub == sub.T)
    eye = torch.eye(z.shape[0], dtype=torch.bool)
    same = same & ~eye
    return {
        "median_pair_dist": float(d2.median().clamp_min(0).sqrt()),
        "within_subj_dist": float(d2[same].mean().clamp_min(0).sqrt()),
        "between_subj_dist": float(d2[~same & ~eye].mean().clamp_min(0).sqrt()),
        "signal_ratio": float((d2[~same & ~eye].mean() / d2[same].mean().clamp_min(1e-12))),
    }


def mmd_at_sigmas(z: torch.Tensor, subject: torch.Tensor, sigmas) -> float:
    z = torch.nn.functional.normalize(z, dim=-1).detach()
    subs = subject.unique()
    terms = []
    for i in range(subs.numel()):
        for j in range(i + 1, subs.numel()):
            a = z[subject == subs[i]]
            b = z[subject == subs[j]]
            xx = yy = xy = 0.0
            for sg in sigmas:
                xx = xx + _rbf(a, a, sg).mean()
                yy = yy + _rbf(b, b, sg).mean()
                xy = xy + _rbf(a, b, sg).mean()
            terms.append(float((xx + yy - 2.0 * xy).clamp_min(0.0) / len(sigmas)))
    return sum(terms) / max(1, len(terms))


def mmd_linear(z: torch.Tensor, subject: torch.Tensor) -> torch.Tensor:
    """Linear-kernel MMD, i.e. exactly the mean subject displacement squared.

    ``MMD^2`` with ``k(x, y) = <x, y>`` is ``||E_a[x] - E_b[y]||^2`` -- no bandwidth,
    no self-pair diagonal, and it targets the group MEANS, which is the only scale at
    which the subject shift exists. Candidate replacement for the RBF form.
    """
    z = F.normalize(z, dim=-1)
    subs = subject.unique()
    if subs.numel() < 2:
        return z.new_zeros(())
    mus = torch.stack([z[subject == s].mean(dim=0) for s in subs])
    d2 = torch.cdist(mus, mus, p=2) ** 2
    off = ~torch.eye(mus.shape[0], dtype=torch.bool, device=z.device)
    return d2[off].mean()


def mmd_linear_norm(z: torch.Tensor, subject: torch.Tensor) -> torch.Tensor:
    """`mmd_linear` divided by the mean within-subject scatter, so the value is a ratio.

    Dimensionless and directly interpretable: 1.0 means "subject means are as far apart
    as the samples inside a subject are from their own mean". Candidate replacement
    whose weight means the same thing at every stage of training.
    """
    z = F.normalize(z, dim=-1)
    subs = subject.unique()
    if subs.numel() < 2:
        return z.new_zeros(())
    mus = torch.stack([z[subject == s].mean(dim=0) for s in subs])
    within = torch.stack([(z[subject == s] - mus[i]).pow(2).sum(dim=1).mean()
                          for i, s in enumerate(subs)]).mean()
    return mmd_linear(z, subject) / within.clamp_min(1e-6)


def mean_displacements(z: torch.Tensor, subject: torch.Tensor) -> list[float]:
    """``||mu_a - mu_b||`` per subject pair -- the quantity a useful term must resolve.

    This is NOT the median pairwise distance. On a normalised sphere in high dimension
    every pair sits near sqrt(2) (concentration of measure), so the median pairwise
    distance says nothing about how far apart the group MEANS are. Measured here: the
    per-sample spread is ~1.41 while the subject-mean displacement is ~0.20, a factor
    of 7. Any per-sample kernel method with a bandwidth wide enough to see a whole
    subject sees the two subjects as the same cloud, and one narrow enough to separate
    means sees each subject's own samples as mutually foreign.
    """
    z = torch.nn.functional.normalize(z, dim=-1).detach()
    mus = {int(s): z[subject == s].mean(dim=0) for s in subject.unique()}
    ks = sorted(mus)
    out = []
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            out.append(float((mus[ks[i]] - mus[ks[j]]).norm()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--seed-bank", action="store_true",
                    help="run a full pass over the loader to rebuild the EMA prototype "
                         "bank before measuring `proto`/`anchor`. The bank lives on the "
                         "Trainer and is NOT written to the checkpoint, so without this "
                         "the prototype terms are measured against an empty bank and "
                         "their gradient share is meaningless.")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cpu")
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = dict(blob["cfg"])
    n_subjects = int(cfg["n_subjects"])

    subj = cfg.get("source_subjects") or [s for s in C.all_subjects()
                                          if s != int(cfg["target_subject"])]
    channels = (C.CHANNELS_OCCIPITO_PARIETAL
                if cfg.get("channel_set", "all63") == "occipital17" else None)
    data = things_eeg.load_loso(subj, int(cfg["target_subject"]), channels,
                               mvnn=cfg.get("mvnn", "off"))
    targets_tr, _ = build_targets(cfg)

    cfg_eff = dict(cfg)
    cfg_eff["n_subjects"] = data.n_subjects
    cfg_eff["n_channels"] = data.tr_eeg[0].shape[-2]
    cfg_eff["n_timepoints"] = data.tr_eeg[0].shape[-1]
    model = build_model(cfg_eff, targets_tr.shape[2], targets_tr.shape[-1]).to(device)
    model.load_state_dict(blob["model"])
    model.train()

    trainer = Trainer(model=model, cfg=cfg, device=device, n_subjects=n_subjects,
                      weights=LossWeights.from_cfg(cfg))
    had = trainer.load_criterion_state(blob.get("crit"))

    if trainer.weights.proto > 0:
        from samclip.losses.regularizers import PrototypeEMA
        n_classes = (data.n_concepts if str(cfg.get("prototype_level", "concept"))
                     == "concept" else data.n_concepts * data.n_images)
        trainer.prototype = PrototypeEMA(n_classes, int(cfg.get("d_embed", 512)))

    cfg_run = dict(cfg)
    cfg_run["num_workers"] = 0
    loader, _ = build_loaders(cfg_run, data, targets_tr)

    print(f"[diag] ckpt={args.ckpt}  epoch={blob.get('epoch')}  crit_restored={had}")
    print(f"[diag] d_model={cfg_eff['d_model']} d_embed={cfg_eff['d_embed']} "
          f"batch_stimuli={cfg_run.get('batch_stimuli')} "
          f"subjects_per_stimulus={cfg_run.get('subjects_per_stimulus')}")
    print(f"[diag] weights={trainer.weights}\n")

    if args.seed_bank and trainer.prototype is not None:
        import time
        t0 = time.time()
        n = 0
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            with torch.no_grad():
                model.eval()
                z = model.encode_eeg(batch["eeg"])
                model.train()
                trainer.prototype.update(z, trainer.prototype_group(batch))
            n += 1
        filled = int((trainer.prototype.count >= trainer.prototype.min_updates).sum())
        print(f"[diag] prototype bank seeded on {n} steps in {time.time() - t0:.0f}s: "
              f"{filled}/{trainer.prototype.n_classes} slots >= min_updates\n")

    params = [p for p in model.parameters() if p.requires_grad]
    acc: dict[str, list[float]] = {}
    geoms: list[dict] = []

    for bi, batch in enumerate(loader):
        if bi >= args.batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch["eeg"], batch["target"],
                    subject_ids=batch.get("subject"), training=True)
        z_e, z_i = out["z_eeg"], out["z_img"]
        subject = batch["subject"]
        stimulus = batch.get("stimulus")

        raw_terms = {
            "img": clip_alignment_loss(z_e, z_i, trainer.crit_img),
            "cross": cross_subject_loss(z_e, stimulus, trainer.crit_cross),
            "dec": hsic_subject(z_e, subject, n_subjects),
            "mmd": mmd_subject(z_e, subject),
            # candidates, measured at weight 1.0 so their raw signal strength is visible
            "mmd_lin": mmd_linear(z_e, subject),
            "mmd_linN": mmd_linear_norm(z_e, subject),
        }
        reg = vicreg_terms(out["z_eeg_raw"],
                           cov_weight=float(cfg.get("vicreg_cov_weight", 0.04)))
        raw_terms["var"] = reg["var"]
        raw_terms["cov"] = reg["cov"]

        wmap = {"img": trainer.weights.img, "cross": trainer.weights.cross,
                "dec": trainer.weights.dec, "mmd": trainer.weights.mmd,
                "var": trainer.weights.reg, "cov": trainer.weights.reg,
                "mmd_lin": 1.0, "mmd_linN": 1.0}

        if trainer.prototype is not None and trainer.weights.proto > 0:
            group = trainer.prototype_group(batch)
            raw_terms["proto"] = trainer.prototype.proto_contrast(z_e, group)
            raw_terms["anchor"] = trainer.prototype.image_anchor_contrast(z_i, group)
            wmap["proto"] = trainer.weights.proto
            wmap["anchor"] = trainer.weights.proto

        for name, term in raw_terms.items():
            w = wmap[name]
            gn = grad_norm(w * term, params)
            acc.setdefault(name, []).append((float(term.detach()), w, gn))

        geoms.append(kernel_geometry(z_e, subject))
        print(f"[diag] batch {bi}: rows={z_e.shape[0]} "
              f"subjects={subject.unique().numel()} "
              f"mmd={float(raw_terms['mmd'].detach()):.6f}")

    print("\n" + "=" * 84)
    print("term influence  (gradient norm w.r.t. ALL trainable parameters, on one batch)")
    print("=" * 84)
    print(f"{'term':>7} {'value':>10} {'weight':>8} {'w*value':>10} "
          f"{'|w*dL/dth|':>13} {'rel':>7} {'share':>8}")
    print("-" * 84)
    rows = {k: (sum(x[0] for x in v) / len(v),
                v[0][1],
                sum(x[2] for x in v) / len(v)) for k, v in acc.items()}
    top = max(r[2] for r in rows.values())
    tot = sum(r[2] for r in rows.values())
    for k, (val, w, gn) in sorted(rows.items(), key=lambda kv: -kv[1][2]):
        print(f"{k:>7} {val:>10.4f} {w:>8.3f} {val * w:>10.4f} "
              f"{gn:>13.3e} {gn / top:>7.3f} {gn / tot:>8.3%}")

    print("\n" + "=" * 84)
    print("MMD kernel geometry on L2-normalised embeddings")
    print("=" * 84)
    for i, g in enumerate(geoms):
        print(f"batch {i}: median pair dist {g['median_pair_dist']:.4f} | "
              f"within-subj {g['within_subj_dist']:.4f} | "
              f"between-subj {g['between_subj_dist']:.4f} | "
              f"between/within var ratio {g['signal_ratio']:.4f}")
    med = sum(g["median_pair_dist"] for g in geoms) / len(geoms)
    print(f"\nmedian heuristic sigma (what mmd_subject uses as its centre): {med:.4f}")
    print("  -> the code's grid is (0.5, 1, 2, 4) x sigma, i.e. "
          f"({0.5 * med:.4f}, {med:.4f}, {2 * med:.4f}, {4 * med:.4f})")

    # One fixed batch, reused for the whole kernel-sensitivity study so the numbers are
    # comparable. `training=False` so dropout does not add a second noise source on top
    # of the bandwidth effect being measured.
    batch = next(iter(loader))
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    out = model(batch["eeg"], batch["target"],
                subject_ids=batch.get("subject"), training=False)
    z_e, subject = out["z_eeg"], batch["subject"]

    disp = mean_displacements(z_e, subject)
    print(f"\nsubject-mean displacement ||mu_a - mu_b|| over {len(disp)} pairs: "
          f"min {min(disp):.4f}  median {sorted(disp)[len(disp) // 2]:.4f}  "
          f"max {max(disp):.4f}")
    print("  -> an RBF kernel resolves the subject shift only when sigma is within a")
    print("     small multiple of THIS number, not of the median pairwise distance")

    print("\nMMD vs bandwidth (single kernel, log grid):")
    import math
    grid = [10 ** (-3.0 + 3.5 * i / 39) for i in range(40)]
    curve = [(sg, mmd_at_sigmas(z_e, subject, (sg,))) for sg in grid]
    best = max(curve, key=lambda kv: kv[1])
    for sg in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]:
        v = dict(curve).get(sg)
        if v is None:
            v = mmd_at_sigmas(z_e, subject, (sg,))
        bar = "#" * int(round(v * 100))
        print(f"  sigma {sg:>7.4f}  MMD {v:>9.6f}  {bar}")
    print(f"  PEAK: sigma {best[0]:.4f} -> MMD {best[1]:.6f}")

    print("\nMMD value under alternative bandwidth grids:")
    for label, g in [
        ("default  (0.5,1,2,4)x med", (0.5 * med, med, 2 * med, 4 * med)),
        ("matched to mean displacement",
         (best[0] * 0.5, best[0], best[0] * 2.0, best[0] * 4.0)),
    ]:
        print(f"  {label:<30} MMD = {mmd_at_sigmas(z_e, subject, g):.6f}")


if __name__ == "__main__":
    main()
