"""Where does the discriminative signal die: in z_inv, in the projector, or in the target space?

The two earlier probes established the boundary conditions:

* the EEG<->CLIP pairing is correct (a linear ridge reaches 5 sigma on single-trial
  top1), so the targets are not mismatched;
* the CLIP teacher spaces are strongly anisotropic -- mean off-diagonal cosine
  +0.391 with 39.7% of the energy in one shared direction, against +0.004 and 0.9%
  for DINOv2.

An anisotropic target space turns any *shared* component of the prediction into a
fixed bias toward the space's hub directions, and retrieval is then decided by that
bias rather than by the per-trial signal.  This script measures how much shared
component survives at each stage and how much discriminative margin is left, so the
fix can be aimed at the stage that actually destroys it.

Stages measured, for the same real batch:
    z_inv          encoder output
    centred(z_inv) the same minus its batch mean, which is the component the shared
                   direction does not occupy
    p_img          the CLIP projector's output, i.e. what the retrieval term sees

`margin` is the quantity that decides retrieval, and it is reported in the target
space so it is comparable to the gallery's own statistics: mean over rows of
cos(pred, own target) - mean over the other targets.  A margin far below the target
space's spread of off-diagonal cosines cannot out-rank the hub bias.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from loso import paths
from loso.data import eeg as eeg_mod
from loso.data import things
from loso.data.targets import TargetStore
from loso.models.eeg_encoder import EEGEncoder, EncoderConfig
from loso.models.heads import AlignmentHeads, HeadConfig


def energy_split(z: torch.Tensor) -> tuple[float, float, float]:
    """||mean||, mean per-dim std, and the shared-direction energy fraction."""
    z = z.float()
    m = z.mean(dim=0)
    var = z.var(dim=0, unbiased=False)
    msq = float((m ** 2).sum())
    vs = float(var.sum())
    return float(m.norm()), float(var.sqrt().mean()), msq / max(msq + vs, 1e-12)


def describe(name: str, z: torch.Tensor) -> None:
    m, sd, frac = energy_split(z)
    centred = z - z.mean(dim=0, keepdim=True)
    u = torch.nn.functional.normalize(z.float(), dim=-1)
    cos = u @ u.t()
    n = len(z)
    off = cos[~torch.eye(n, dtype=torch.bool)]
    print(f"  {name:16s} |row|={z.float().norm(dim=-1).mean():7.3f} "
          f"||mean||={m:7.3f} per_dim_std={sd:.4f} shared={frac * 100:5.1f}% "
          f"| ||centred||={centred.float().norm(dim=-1).mean():7.3f} "
          f"offdiag_cos={off.mean():+.4f}")


def margin(pred: torch.Tensor, keys: torch.Tensor, pos: torch.Tensor) -> float:
    """cos(pred, its correct key) - mean cos over the other keys.

    `pos[b]` is the column of the correct key for row `b`, so the same function
    serves the batch-internal case (`pos = arange`) and the gallery case
    (`pos = slots`).
    """
    p = torch.nn.functional.normalize(pred.float(), dim=-1)
    t = torch.nn.functional.normalize(keys.float(), dim=-1)
    sim = p @ t.t()
    own = sim.gather(1, pos.view(-1, 1)).squeeze(1)
    return float((own - (sim.sum(dim=1) - own) / (sim.shape[1] - 1)).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-subject", default="sub-08")
    ap.add_argument("--ckpt", default="",
                    help="checkpoint to inspect in addition to the initialisation")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--d-inv", type=int, default=512)
    args = ap.parse_args()

    train_subjects, test_subject = things.loso_split(args.test_subject)
    n_subjects = len(train_subjects)
    store = TargetStore("train", names=("clip_image", "clip_text_caption"))
    cfg = EncoderConfig(pretrained_subjects=n_subjects, d_inv=args.d_inv)
    head_cfg = HeadConfig(n_subjects=n_subjects, d_model=cfg.d_model,
                          d_inv=cfg.d_inv, d_sub=cfg.d_sub,
                          n_time_patches=cfg.n_tokens)
    model = EEGEncoder(cfg).eval()
    heads = AlignmentHeads(head_cfg).eval()

    per_subject, global_stats = eeg_mod.build_normalizers(
        train_subjects, "train_subjects",
        cache_path=paths.DATA_ROOT / f"norm_train_subjects_{n_subjects}.json",
    )
    del per_subject
    subj = train_subjects[0]
    ds = eeg_mod.TrainEEGDataset([subj], {subj: global_stats}, {subj: subj})
    # A *grouped* batch, because that is what the retrieval terms actually see: 36
    # rows of one stimulus in a 512-row batch.  A uniform batch would leave almost
    # every row without a same-image peer, which changes both the soft target and the
    # constant-encoder baseline this script compares against.
    rng = np.random.default_rng(0)
    by_slot: dict[int, list[int]] = {}
    for i, (_, _, slot) in enumerate(ds.trials):
        by_slot.setdefault(slot, []).append(i)
    reps = max(len(v) for v in by_slot.values())
    group = max(1, args.batch // reps)
    chosen = rng.choice(len(by_slot), group, replace=False)
    keys = list(by_slot)
    idx = [i for c in chosen for i in by_slot[keys[c]][:reps]]
    idx = idx[:args.batch]
    samples = [ds[int(i)] for i in idx]
    x = torch.stack([s.x for s in samples])
    slots = torch.tensor([s.target_slot for s in samples])
    subject_id = torch.zeros(len(x), dtype=torch.long)
    print(f"[geom] batch={tuple(x.shape)} unique images={len(slots.unique())} "
          f"({reps} reps each)")

    targets = store.as_normalized(slots)["clip_image"]
    scale = heads.scaled_logit_scale()
    print(f"[geom] logit_scale={float(scale):.4f}")

    # 16,540 images would make a 16,540^2 similarity matrix, so the gallery statistics
    # are taken on a random subset; they are distributional, not per-row.
    sub = rng.choice(store.shapes.n_images, 2000, replace=False)
    gallery = store.as_normalized(torch.from_numpy(sub))["clip_image"]
    gu = torch.nn.functional.normalize(gallery.float(), dim=-1)
    gc = gu @ gu.t()
    n = len(gc)
    off = gc[~torch.eye(n, dtype=torch.bool)]
    print(f"[geom] clip_image gallery (2000 of {store.shapes.n_images}): "
          f"off-diag cos mean={off.mean():+.4f} p95={off.quantile(0.95):+.4f} "
          f"max={off.max():+.4f}")

    for tag, ckpt in (("init", None), ("trained", args.ckpt or None)):
        if tag == "trained" and not ckpt:
            continue
        if ckpt:
            blob = torch.load(ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(blob["encoder"])
            heads.load_state_dict(blob["heads"])
        with torch.no_grad():
            out = model(x, subject_id, return_tokens=True)
            z = out["z_inv"]
            p_img = heads.img(z)
            p_text = heads.text(z)
        print(f"\n[geom] === {tag} ===")
        describe("z_inv", z)
        describe("centred(z_inv)", z - z.mean(0, keepdim=True))
        describe("p_img", p_img)
        # Batch-internal margin against the batch's own targets: the diagonal is the
        # correct key, which is what the InfoNCE term rewards.
        pos = torch.arange(len(p_img))
        print(f"  margin(p_img)  = {margin(p_img, targets, pos):+.4f}   "
              f"(target space off-diag mean {off.mean():+.4f}, "
              f"noise floor for a constant is 0)")
        print(f"  margin(p_text) = {margin(p_text, targets, pos):+.4f}")
        # The plateau the loss sits on: with a near-constant prediction every logit
        # is equal and the loss is the uniform value, whatever the encoder contains.
        # Hard-diagonal CE is used here rather than the soft-label form so the number
        # is directly comparable to `constant_encoder_loss`.
        from loso import diagnostics as D
        ce = float(torch.nn.functional.cross_entropy(
            scale * p_img.float() @ targets.float().t(),
            torch.arange(args.batch)))
        const = D.constant_encoder_loss(p_img, targets, scale)
        print(f"img CE={ce:.4f}  constant_encoder={const:.4f}  "
              f"margin={ce - const:+.4f}")


if __name__ == "__main__":
    main()
