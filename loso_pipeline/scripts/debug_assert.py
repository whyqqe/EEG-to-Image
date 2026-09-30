"""Reproduce the preview job's CUDA assert on CPU with the real stores.

`RuntimeError: CUDA error: device-side assert triggered` is reported asynchronously,
so the traceback's last frame names whichever kernel happened to synchronise next --
here `vicreg_covariance`, which does no indexing at all.  The underlying message is
`nll_loss_forward_reduce_cuda_kernel_2d: Assertion t >= 0 && t < n_classes failed`,
i.e. a cross-entropy whose integer target is out of range, and the only integer-label
cross-entropy in the objective is `subject_adversarial_loss`.  So this walks the real
data path on CPU and reports the label ranges and class counts that the GPU asserted
on.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from loso import paths
from loso.data import eeg as eeg_mod
from loso.data import things
from loso.data.targets import TargetStore
from loso.losses import align as L
from loso.models.eeg_encoder import EEGEncoder, EncoderConfig
from loso.models.heads import AlignmentHeads, HeadConfig
from loso.train.align import AlignConfig, build_train_loader, compute_losses


def main() -> int:
    train_subjects, test_subject = things.loso_split("sub-08")
    n_subjects = len(train_subjects)
    print(f"train subjects: {n_subjects} {train_subjects}")

    per_subject, global_stats = eeg_mod.build_normalizers(
        train_subjects, "train_subjects",
        cache_path=paths.DATA_ROOT / f"norm_train_subjects_{n_subjects}.json")
    norm = {s: global_stats for s in list(train_subjects) + [test_subject]}
    norm_keys = {s: s for s in norm}

    ds = eeg_mod.TrainEEGDataset(train_subjects, norm, norm_keys,
                                 augment_cfg=eeg_mod.AugmentConfig())

    groups = [slot for _, _, slot in ds.trials]
    uniq, counts = np.unique(groups, return_counts=True)
    print(f"dataset rows={len(ds)}  groups={len(uniq)}  "
          f"group size min={counts.min()} max={counts.max()}")
    print(f"rows per group histogram (top 5): "
          f"{np.bincount(counts)[-5:] if counts.max() < 100 else 'n/a'}")

    cfg = AlignConfig(batch_size=512, num_workers=0)
    cfg.enc.pretrained_subjects = n_subjects
    cfg.head.n_subjects = n_subjects
    cfg.head.d_model = cfg.enc.d_model
    cfg.head.d_inv = cfg.enc.d_inv
    cfg.head.d_sub = cfg.enc.d_sub
    cfg.head.n_time_patches = cfg.enc.n_tokens

    loader = build_train_loader(ds, cfg)
    batch = next(iter(loader))

    sid = batch["subject_id"]
    print(f"\nbatch rows={sid.numel()}  subject_id range=[{int(sid.min())}, "
          f"{int(sid.max())}]  distinct={int(sid.unique().numel())}")
    print(f"target_slot range=[{int(batch['target_slot'].min())}, "
          f"{int(batch['target_slot'].max())}]")
    print(f"repeat_id range=[{int(batch['repeat_id'].min())}, "
          f"{int(batch['repeat_id'].max())}]")

    model = EEGEncoder(cfg.enc)
    heads = AlignmentHeads(cfg.head)
    n_classes = heads.subject.net[-1].out_features
    print(f"\ncfg.head.n_subjects={cfg.head.n_subjects}  "
          f"SubjectClassifier n_classes={n_classes}  "
          f"needs subject_id in [0, {n_classes - 1}]")

    if int(sid.max()) >= n_classes or int(sid.min()) < 0:
        print(f"\n*** subject_id out of range for cross_entropy: "
              f"max={int(sid.max())} >= n_classes={n_classes} ***")
    else:
        print("subject_id is in range; the assert must come from elsewhere")

    store = TargetStore("train", names=("clip_image", "clip_text_caption", "dino",
                                        "vae_latent"))
    print("\nrunning compute_losses on CPU ...")
    loss, terms = compute_losses(model, heads, store, batch, cfg, 1.0)
    print(f"  total={float(loss):.4f}")
    for k, v in sorted(terms.items()):
        print(f"  {k:6s} {float(v):+.6f}")
    print("\ncompute_losses is fine on CPU -> the CUDA assert is GPU-specific")
    return 0


if __name__ == "__main__":
    sys.exit(main())
