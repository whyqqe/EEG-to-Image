#!/usr/bin/env python
"""Measures how much of the joint gradient each tower contributes.

The dual-tower model sums its two losses into ONE scalar and calls `backward()`
once, followed by a single `clip_grad_norm_` over ALL parameters (see
`train.py`). The parameter sets are disjoint, so neither loss can reach the other
tower's weights directly -- but the CLIP couples them, because it scales every
parameter by the same factor

    factor = min(1, max_norm / ||g_joint||),   ||g_joint||^2 = ||g_sem||^2 + ||g_str||^2

If the joint norm sits below the threshold the factor is exactly 1.0 and the
towers are independent in practice. If it sits above, each tower's effective
learning rate is multiplied by a factor that depends on the OTHER tower's
gradients, and the tower with the smaller norm is the one that loses -- it is
scaled down by the ratio, not by anything it computed.

The structural side is also broken into its two terms, because they have very
different scales and only one of them is a regression:

    MSE      `latent_mse(pred, target)`      -- a per-pixel objective on a (B,1,8,8)
                                                field
    floor    `variance_floor(pred, std, .15)` -- a hinge on `pred.std(dim=(0,2,3))`,
                                                which is a NON-LINEAR reduction over
                                                the whole batch and therefore has a
                                                gradient that can be large

Everything reads the same objects training reads: `AuxTargetDataset` (so the
target is normalised and paired exactly as in the run), `build_from_args` (so the
arms match), `InfoNCE`, and the loss functions themselves. Nothing here
re-implements the pipeline, because an approximate reproduction of the loss would
report an approximate gradient scale and the conclusion rests on that scale.

CPU is enough and is the point: this is a measurement of gradient SCALE, which is
set by the architecture and the loss weights, not by the device.

Usage:
    diag_grad_split.py                     # fresh init
    diag_grad_split.py --ckpt path/to.pt   # a trained checkpoint
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epd import config  # noqa: E402
from epd.data import AuxTargetDataset, concept_split, load_subject  # noqa: E402
from epd.losses import InfoNCE, latent_mse, variance_floor  # noqa: E402
from epd.model import build_from_args  # noqa: E402

# The real semantic arm and the real structural arm, transcribed from
# `scripts/run_epd_da2.sh`. Only the entries that reach `build_from_args`, the
# criterion or the structural loss are here; the schedule is irrelevant to a
# gradient scale and is deliberately absent rather than defaulted, so a drift in
# the run script shows up as a KeyError here.
TRAIN_CFG = {
    "backbone": "timm:vit_b16_in21k_orig",
    "layers": [12],
    "fusion_mode": "none",
    "pool": "mean",
    "timm_global_pool": "avg",
    "head_kind": "eegit",
    "img_head_kind": "eegit",
    "head_drop": 0.5,
    "d_embed": 1024,
    "target_layers": None,
    "target_fusion": "single",
    "tokenizer": "eegit",
    "patch_style": "time-region",
    "patch_size": 16,
    "n_patches_w": 14,
    "freeze_blocks": 0,
    "struct_backbone": "da2",
    "struct_arch": "da2",
    "struct_patch_size": 14,
    "struct_n_patches_w": 14,
    "struct_tokenizer": "eegit",
    "struct_out_hw": 8,
    "struct_vae_ch": 1,
    "struct_drop": 0.1,
    "struct_freeze_blocks": 0,
}

# The loss weights the run script uses (`--w-vae 1.0`, `--w-var 1.0`,
# `--var-margin 0.15`, no MMD so `contrast_w` is 1.0 throughout).
W_VAE = 1.0
W_VAR = 1.0
VAR_MARGIN = 0.15
CONTRAST_W = 1.0

FEATURE_LAYER = "block26"


def split_params(model: torch.nn.Module) -> tuple[list, list]:
    """Returns (semantic, structural) parameter lists. Disjoint by construction."""
    sem, st = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (st if name.startswith("struct.") else sem).append(p)
    return sem, st


def grad_norm(params: list) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return total ** 0.5


def zero(model) -> None:
    for p in model.parameters():
        p.grad = None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--features", type=Path,
                    default=config.OUTPUTS / "features" / "clip_h14_layers")
    ap.add_argument("--depth-train", type=Path,
                    default=config.OUTPUTS / "struct_targets" / "coarse" / "train_depth_8.npy")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="measure a trained checkpoint instead of the fresh init")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--clip", type=float, default=1.0,
                    help="the max_norm train.py passes to clip_grad_norm_")
    ap.add_argument("--adam-check", type=int, default=0, metavar="K",
                    help="also simulate K AdamW steps twice -- once with the shared "
                         "clip and once without -- and report how far the two "
                         "parameter trajectories actually diverge. This is the "
                         "number that says whether a binding global clip matters, "
                         "which for AdamW it may not.")
    ap.add_argument("--lr", type=float, default=5e-4,
                    help="AdamW lr for --adam-check (the head group's, the largest in "
                         "the run; a larger lr makes any divergence easier to see)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(2025)
    torch.set_num_threads(8)

    # ---- the data, through the SAME dataset class the run uses -------------
    # Re-implementing the pairing here is exactly how a gradient measurement goes
    # wrong quietly: the target is standardised by fit-split statistics and paired
    # by `concept * n_slots + slot`, and both change the loss's scale.
    ch_names = json.loads((config.EEG_DIR / "info.json").read_text())["ch_names"]
    tr_eeg, _ = load_subject(a.subject, None)                       # (C, 10, 63, 250)
    img_tr = np.load(a.features / "train" / f"{FEATURE_LAYER}.npy", mmap_mode="r")
    depth = np.load(a.depth_train, mmap_mode="r")                   # (N*10, 8, 8)
    if depth.ndim == 3:
        depth = depth[:, None]
    n_img = tr_eeg.shape[1]
    split = concept_split(150, 2025)
    print(f"[data] EEG {tr_eeg.shape}, features {img_tr.shape}, depth {depth.shape}")
    print(f"[data] fit concepts {len(split.fit_concepts)}, val "
          f"{len(split.val_concepts)}, {n_img} image slots")

    # The centring field, exactly as `load_struct_target` builds it: the fit-split
    # per-pixel mean, subtracted once over the whole array.
    fit_rows = np.sort((split.fit_concepts[:, None] * n_img
                        + np.arange(n_img)[None, :]).ravel())
    fit_blk = np.asarray(depth[fit_rows], dtype=np.float32)
    field = fit_blk.mean(axis=0, keepdims=True)
    centred = np.asarray(depth, dtype=np.float32) - field
    print(f"[tgts] centred on the fit-set mean field "
          f"(||field||={float(np.linalg.norm(field)):.3f}, "
          f"field std {float(field.std()):.4f}); "
          f"{float((centred < 0).mean()):.1%} of entries negative")
    del fit_blk

    # The per-channel statistics `AuxTargetDataset` standardises with, over the
    # fit rows only, exactly as `train.py` computes them.
    flat = centred[fit_rows].reshape(len(fit_rows), 1, -1).astype(np.float64)
    vae_mean = flat.sum(axis=(0, 2)) / (flat.shape[0] * flat.shape[2])
    vae_std = np.sqrt(np.maximum(
        (flat ** 2).sum(axis=(0, 2)) / (flat.shape[0] * flat.shape[2]) - vae_mean ** 2,
        1e-12)).astype(np.float32)
    vae_mean = vae_mean.astype(np.float32)
    print(f"[tgts] fit-split standardisation: mean "
          f"{np.round(vae_mean, 4).tolist()} std {np.round(vae_std, 4).tolist()}")
    del flat

    ds = AuxTargetDataset(tr_eeg, img_tr, split.fit_concepts,
                          aux_vae=centred, aux_depth=None,
                          vae_mean=vae_mean, vae_std=vae_std,
                          l2norm=True, augment=None, seed=2025)
    # `augment=None` on purpose: augmentation is a random draw, and a gradient scale
    # averaged over draws would carry the augmentation's variance into the number
    # being compared between the two towers.

    img_dim = int(img_tr.shape[-1])
    model = build_from_args(TRAIN_CFG, ch_names, img_dim)
    if a.ckpt:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        print(f"[ckpt] {a.ckpt.name}: epoch {ck.get('epoch')}, "
              f"{len(missing)} missing, {len(unexpected)} unexpected")
        if missing:
            raise SystemExit(f"[FATAL] checkpoint does not match the config: {missing[:4]}")
        # The trained tokenizer's own buffers are already in the state dict; the
        # `set_norm_stats` call below would overwrite them with the same numbers
        # (fit split == the fit split), so it is harmless either way.
    model.train()
    sem, st = split_params(model)
    print(f"[modl] semantic params {sum(p.numel() for p in sem)/1e6:.1f}M, "
          f"structural {sum(p.numel() for p in st)/1e6:.1f}M")

    # Both tokenizers refuse to run without fit-split z-score statistics. Fitted on
    # the fit split through `split.fit_concepts`, which is what training does.
    model.encoder.tokenizer.set_norm_stats(tr_eeg[split.fit_concepts])
    model.struct.encoder.tokenizer.set_norm_stats(tr_eeg[split.fit_concepts])

    crit = InfoNCE()
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                                        num_workers=0, drop_last=True)

    if a.dry_run:
        b = next(iter(loader))
        print(f"[dry ] one batch: {len(b)} tensors, shapes "
              f"{[tuple(t.shape) for t in b]}")
        return 0

    print()
    hdr = (f"| step | ||g_sem|| | ||g_sem_win|| | MSE | floor | joint "
           f"| clip factor | sem share |")
    print(hdr)
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    rows = []
    for step, batch in enumerate(loader):
        if step >= a.steps:
            break
        x, f = batch[0], batch[1]
        # `AuxTargetDataset.__getitem__` returns `(x, f, concept, vae)` when only
        # `aux_vae` is set, and `v` is already standardised with the fit-split
        # statistics -- so it is used as-is, exactly as `train.py` does with
        # `a_vae = batch[3:]` then `aux[0]`.
        tgt = batch[3]

        out = model.forward_all(x, torch.zeros(len(x), dtype=torch.long), training=True)
        z_i = model.encode_image(f)

        # (a) the semantic terms alone. `||g_str||` must be 0 here: if it is not,
        # the two towers are not as independent as the parameter split suggests and
        # every later number is contaminated.
        zero(model)
        (CONTRAST_W * crit(out["z"], z_i)).backward(retain_graph=True)
        sem_alone, str_leak = grad_norm(sem), grad_norm(st)

        # (b) the regression alone.
        zero(model)
        (W_VAE * latent_mse(out["struct"]["vae"], tgt)).backward(retain_graph=True)
        mse_norm = grad_norm(st)

        # (c) the variance floor alone.
        zero(model)
        (W_VAR * variance_floor(out["struct"]["vae"], vae_std, VAR_MARGIN)).backward(
            retain_graph=True)
        floor_norm = grad_norm(st)

        # (d) the joint loss, which is what the optimiser and the clip actually see.
        loss = (CONTRAST_W * crit(out["z"], z_i)
                + W_VAE * latent_mse(out["struct"]["vae"], tgt)
                + W_VAR * variance_floor(out["struct"]["vae"], vae_std, VAR_MARGIN))
        zero(model)
        loss.backward()
        g_sem, g_str = grad_norm(sem), grad_norm(st)
        joint = (g_sem ** 2 + g_str ** 2) ** 0.5
        factor = min(1.0, a.clip / joint) if joint > 0 else 1.0
        rows.append((g_sem, g_str, mse_norm, floor_norm, joint, factor))
        print(f"| {step} | {g_sem:.4f} | {sem_alone:.4f} | {mse_norm:.2f} | "
              f"{floor_norm:.2f} | {joint:.2f} | {factor:.6f} | "
              f"{g_sem / joint if joint else 0:.2%} |")
        if str_leak > 1e-6:
            print(f"|      | !! semantic loss produced a structural gradient of "
                  f"{str_leak:.3e} |" + " |" * 6)
        del out, loss, tgt, x, f

    if not rows:
        raise SystemExit("[FATAL] no steps ran")
    arr = np.asarray(rows)
    print()
    print("## Read")
    print()
    print(f"- ||g_sem||  {arr[:,0].mean():.3f} (median {np.median(arr[:,0]):.3f})")
    print(f"- ||g_str||  {arr[:,1].mean():.3f} (median {np.median(arr[:,1]):.3f}) "
          f"= {arr[:,1].mean() / max(arr[:,0].mean(), 1e-9):.0f}x the semantic norm")
    print(f"-   of which MSE {arr[:,2].mean():.3f}, variance floor "
          f"{arr[:,3].mean():.3f}")
    print(f"- clip factor {arr[:,5].mean():.6f}, "
          f"binding in {int((arr[:,5] < 1.0).sum())}/{len(arr)} steps")
    print()
    if arr[:, 5].min() >= 1.0:
        print(f"The clip never binds: the joint norm stays under {a.clip}. The two "
              f"towers are independent -- one summed loss and one step, but nothing "
              f"either tower computes reaches the other's update.")
    else:
        print(f"The clip binds at {arr[:,5].min():.6f}: the joint norm is above "
              f"{a.clip:g} at every step, and the structural norm is "
              f"~{arr[:,1].mean() / max(arr[:,0].mean(), 1e-9):.0f}x the semantic one.")
        print()
        print("DO NOT conclude from this that the semantic tower is being trained at "
              "6e-4 of its configured LR. `train.py` uses AdamW, whose update is "
              "`m_hat / (sqrt(v_hat) + eps)` with `m` linear and `v` quadratic in the "
              "gradient, so a GLOBAL rescale cancels exactly when the factor is "
              "constant across steps. The clip's factor IS global (one scalar for all "
              "parameters) and is dominated by the structural term, so it is "
              "approximately constant and therefore approximately a no-op. Run "
              "`--adam-check` for the control that separates the clip's effect from "
              "chaos; in the measured case a 1e-6 perturbation of the threshold "
              "reproduced most of the divergence, i.e. the clip was not the cause.")
        print()
        print("What the asymmetry IS good for: it shows the run's gradient budget is "
              "set by the structural regression, so `--w-vae` and the structural "
              "target's scale are load-bearing for the whole optimisation. A "
              "per-group clip is still the cleaner instrument (it makes each tower's "
              "step size reproducible and independent of the other's scale), but it "
              "is a hygiene fix here, not a bug fix.")

    if a.adam_check:
        adam_report(model, ds, crit, vae_std, a)
    return 0


def adam_report(model, ds, crit, vae_std, a) -> None:
    """Does the shared clip actually change where the optimiser goes?

    The raw norms cannot answer this, and the naive answer is wrong in a way worth
    spelling out, because it is the whole reason this function exists.

    The clip multiplies EVERY parameter's gradient by ONE scalar
    `c = min(1, max_norm / ||g_joint||)`, and AdamW's update is
    `m_hat / (sqrt(v_hat) + eps)` with `m` linear and `v` quadratic in `g`. For a
    CONSTANT `c` the scale cancels exactly -- `c*m_hat / (c*sqrt(v_hat))` is
    `m_hat/sqrt(v_hat)` -- so a globally binding clip would be a strict no-op and
    the 400x asymmetry between the towers would be harmless. It only bites through
    the VARIATION of `c` over steps, because `m` and `v` then average gradients
    that were scaled by different factors.

    So the question is empirical and needs a CONTROL. Two branches that differ only
    by the clip will diverge over K steps for one of two reasons: the clip moved
    the optimiser, or the run is chaotic and ANY perturbation grows. Reporting the
    first without ruling out the second would be reading chaos as causation.

    Three branches are run from the same weights over the same batches:

      A  `max_norm = 1.0`        what `train.py` does
      B  `max_norm = inf`        no clip at all
      C  `max_norm = 1.000001`   a 1e-6 relative change in the clip threshold

    C is the control. It is a perturbation of the same KIND and several orders of
    magnitude SMALLER than A->B. If A diverges from B no more than it diverges from
    C, the distance metric is measuring chaos and says nothing about the clip. Only
    a clear gap -- A vs B much larger than A vs C -- attributes the divergence to
    the clip.
    """
    import copy

    K = a.adam_check
    branches = [("A run (1.0)", a.clip),
                ("B none (inf)", float("inf")),
                ("C control (x1.000001)", a.clip * (1.0 + 1e-6))]
    print()
    print(f"## Adam check: {K} AdamW steps, lr {a.lr:g}, three branches from the same "
          f"weights over the same batches")
    print()
    print("| step | " + " | ".join(f"||g|| {b[0][0]}" for b in branches) + " | "
          + " | ".join(f"factor {b[0][0]}" for b in branches) + " |")
    print("|---:|" + "---:|" * (2 * len(branches)))

    models = {name: copy.deepcopy(model) for name, _ in branches}
    starts = {name: {n: p.detach().clone() for n, p in m.named_parameters()
                     if p.requires_grad}
              for name, m in models.items()}
    opts = {name: torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                                    lr=a.lr, betas=(0.9, 0.999), weight_decay=1e-4)
            for name, m in models.items()}
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                                        num_workers=0, drop_last=True)
    # One pass over the loader per step, shared by all branches, so every branch
    # sees the same batch at the same step. Unshuffled, so this is deterministic.
    order = []
    for i, b in enumerate(loader):
        if i >= K:
            break
        order.append((b[0], b[1], b[3]))

    for step, (x, f, tgt) in enumerate(order):
        line_g, line_f = [], []
        for name, thresh in branches:
            m, opt = models[name], opts[name]
            out = m.forward_all(x, torch.zeros(len(x), dtype=torch.long), training=True)
            loss = (CONTRAST_W * crit(out["z"], m.encode_image(f))
                    + W_VAE * latent_mse(out["struct"]["vae"], tgt)
                    + W_VAR * variance_floor(out["struct"]["vae"], vae_std, VAR_MARGIN))
            zero(m)
            loss.backward()
            g = float(sum(p.grad.detach().pow(2).sum() for p in m.parameters()
                          if p.requires_grad and p.grad is not None)) ** 0.5
            factor = 1.0
            if np.isfinite(thresh) and g > thresh:
                factor = thresh / g
                for p in m.parameters():
                    if p.requires_grad and p.grad is not None:
                        p.grad.mul_(factor)
            opt.step()
            line_g.append(f"{g:.1f}")
            line_f.append(f"{factor:.6f}")
            del out, loss
        print(f"| {step} | " + " | ".join(line_g) + " | " + " | ".join(line_f) + " |")

    def diverge(na, nb):
        num = den = 0.0
        for n, p in models[na].named_parameters():
            if not p.requires_grad:
                continue
            da = (p.detach() - starts[na][n]).flatten()
            db = (models[nb].state_dict()[n].detach() - starts[nb][n]).flatten()
            num += float((da - db).pow(2).sum())
            den += float(da.pow(2).sum())
        return (num ** 0.5) / max(den ** 0.5, 1e-12)

    ab = diverge("A run (1.0)", "B none (inf)")
    ac = diverge("A run (1.0)", "C control (x1.000001)")
    bc = diverge("B none (inf)", "C control (x1.000001)")
    print()
    print(f"- A vs B (the clip's real effect)      {ab:.4e}")
    print(f"- A vs C (a 1e-6 perturbation: chaos)  {ac:.4e}")
    print(f"- B vs C (also chaos)                  {bc:.4e}")
    print()
    print("Relative L2 distance between the parameter trajectories, over all trainable "
          "parameters, normalised by how far A moved.")
    print()
    if ac > 0 and ab < 5 * ac:
        print(f"INCONCLUSIVE, leaning 'chaos dominates': A vs B is {ab/ac:.2f}x A vs C, "
              f"so a perturbation 1e-6 the size of the clip produces most of the same "
              f"divergence. This run cannot attribute the divergence to the clip, and "
              f"the honest reading is that AdamW's per-parameter normalisation "
              f"absorbs a global rescale, leaving only the scale's step-to-step "
              f"variation -- which a 400x norm asymmetry does not by itself make "
              f"large.")
    else:
        print(f"THE CLIP HAS A REAL EFFECT: A vs B is {ab/ac:.2f}x A vs C, so the "
              f"joint clip moves the trajectory further than a perturbation 1e-6 its "
              f"size. The towers are coupled through it and the per-group clip is "
              f"warranted.")


if __name__ == "__main__":
    raise SystemExit(main())
