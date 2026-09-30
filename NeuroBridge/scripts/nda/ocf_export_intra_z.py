#!/usr/bin/env python3
"""Export `shared_r` for a SINGLE-SUBJECT encoder checkpoint.

WHY THIS EXISTS (and why ocf_export_ss_parts.py cannot be reused here)
--------------------------------------------------------------------
`ocf_export_ss_parts.py` verifies a re-export against the STORED `z_eeg_proj` and
hard-fails if the cosine drops below 0.99.  That check is exactly right for its
job (confirming that an 8-subject export is reproducible) and exactly wrong for
this one: an encoder trained on sub-08 ALONE must NOT reproduce the 9-subject
export, and the failure of that check is the whole point of the run.

So this script does the same dataset construction -- identical channels, identical
[0, 250] window, identical augmentation flags, so the rows line up one-to-one with
every other export in the project -- and writes `shared_r_{train,test}.npy` in the
layout `ocf_train.py --z-cache-root` expects:

    <out>/sub-08/shared_r_train.npy
    <out>/sub-08/shared_r_test.npy

`shared_r` is `r` from `model(eeg, sid, return_parts=True)`, i.e. the raw shared
backbone output (1024-d).  It was measured to beat the pipeline's own `z_eeg_proj`
export and `specific_s` on BOTH the semantic and the layout probe, which is why it
is the input the model is trained on.

The importances here are only two: HONESTY OF THE COMPARISON (same rows) and the
provenance record written to `intra_z_report.json`, which states which checkpoint
produced the file so an intra and a LOSO export can never be confused later.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT))
sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))

DEFAULT_CHANNELS = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]

from ss_modules import CHANNEL_SETS, resolve_channels  # noqa: E402


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--out", type=str, required=True,
                    help="dir that will contain sub-XX/shared_r_{train,test}.npy")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--channels", type=str, default="", choices=[""] + sorted(CHANNEL_SETS),
                    help=("electrode set. Leave EMPTY (the default) to inherit it from "
                          "the checkpoint, which is the only safe choice for an existing "
                          "encoder: the model's first layer is a weight over "
                          "channels*samples, so feeding it a different montage produces "
                          "valid-looking but meaningless features. Pass a value only to "
                          "assert a specific set, and the export hard-fails if the "
                          "checkpoint disagrees."))
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    from module.dataset import EEGPreImageDataset  # noqa: E402
    from module.projector import ProjectorLinear  # noqa: E402
    from ss_modules import SharedSpecificEncoder  # noqa: E402

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    subjects = [int(s) for s in ck["subjects"]]
    if subjects != [args.subject]:
        raise SystemExit(
            f"[FATAL] checkpoint subjects = {subjects} but this is meant to be the "
            f"PURE INTRA export for sub-{args.subject} alone. Refusing to export: "
            f"a multi-subject checkpoint here would silently turn the intra run into "
            f"another LOSO run while still being labelled intra.")
    img_dim, feature_dim = int(ck["img_dim"]), int(ck["feature_dim"])
    eeg_len, n_ch = int(ck["eeg_sample_points"]), int(ck["channels_num"])

    # ---- montage resolution -------------------------------------------------
    # The checkpoint is the authority. `channels_num` alone cannot say WHICH
    # electrodes, and the model cannot tell: its input layer is a flat weight of
    # shape (feature_dim, n_ch*eeg_len), so any permutation or subset of length
    # n_ch would run happily and produce nonsense. So we rebuild the name list the
    # checkpoint was trained under and refuse anything that disagrees.
    ck_set = ck.get("channel_set")
    ck_names = ck.get("channel_names")
    if ck_names is None:
        # Written before channel_names existed: the 17-channel montage is the only
        # thing that was ever trained, and n_ch confirms it (or we bail).
        if n_ch != len(DEFAULT_CHANNELS):
            raise SystemExit(
                f"[FATAL] {Path(args.checkpoint).name} records channels_num={n_ch} "
                f"with no channel_names, which no historical run produced (the only "
                f"pre-names montage was the {len(DEFAULT_CHANNELS)}-channel posterior "
                f"one). Refusing to guess the electrode order."
            )
        ck_names, ck_set = list(DEFAULT_CHANNELS), "posterior"
    ck_names = [str(c) for c in ck_names]
    if len(ck_names) != n_ch:
        raise SystemExit(
            f"[FATAL] checkpoint lists {len(ck_names)} channel names but "
            f"channels_num={n_ch}; the layout is ambiguous and a wrong order would "
            f"silently scramble every feature."
        )
    if args.channels and args.channels != ck_set:
        raise SystemExit(
            f"[FATAL] --channels {args.channels} was requested but the checkpoint was "
            f"trained with channel_set='{ck_set}' ({n_ch} electrodes). Re-exporting an "
            f"encoder onto a montage it never saw is not an experiment, it is a bug."
        )
    selected_channels = list(ck_names)
    print(f"[intra_z] ckpt={Path(args.checkpoint).name} subjects={subjects} "
          f"phase={ck.get('phase')} epoch={ck.get('epoch')} "
          f"img_dim={img_dim} feat_dim={feature_dim} "
          f"channels={ck_set} n_ch={n_ch}")

    model = SharedSpecificEncoder(
        subject_ids=subjects, feature_dim=img_dim, eeg_sample_points=eeg_len,
        channels_num=n_ch, n_extra_blocks=int(ck.get("n_extra_blocks", 1)),
        use_adapter=True,
    ).to(dev)
    proj = ProjectorLinear(img_dim, feature_dim).to(dev)
    model.load_state_dict(ck["model_state_dict"])
    proj.load_state_dict(ck["eeg_projector_state_dict"])
    model.eval()
    proj.eval()

    eeg_dir = f"{NB_ROOT}/data/things_eeg/preprocessed_eeg"
    rn50_dir = f"{NB_ROOT}/data/things_eeg/image_feature/RN50"
    sub_dir = Path(args.out) / f"sub-{args.subject:02d}"
    sub_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"checkpoint": str(args.checkpoint), "subject": args.subject,
                    "purpose": "pure intra-subject encoder export (sub-XX only)",
                    "ckpt_epoch": ck.get("epoch"), "ckpt_phase": ck.get("phase"),
                    "lambda_diff": ck.get("lambda_diff"),
                    "channel_set": ck_set, "channels_num": n_ch,
                    "channel_names": ck_names, "folds": {}}

    for train_flag, tag in ((True, "train"), (False, "test")):
        ds = EEGPreImageDataset(
            [args.subject], eeg_dir, selected_channels, [0, 250],
            rn50_dir, "", False, [], True, False, None, train_flag, False, False, False,
        )
        acc: dict[str, list[np.ndarray]] = {}
        with torch.no_grad():
            for batch in DataLoader(ds, batch_size=args.batch_size, shuffle=False):
                eeg, _img, _t, sid, *_ = batch
                eeg, sid = eeg.to(dev), sid.to(dev)
                out, s, r = model(eeg, sid, return_parts=True)
                z = proj(out)
                for k, v in (("shared_r", r), ("specific_s", s), ("fused", out),
                             ("z_eeg_proj", z)):
                    acc.setdefault(k, []).append(v.float().cpu().numpy())
        arrs = {k: np.concatenate(v, 0).astype(np.float32) for k, v in acc.items()}
        for k, a in arrs.items():
            np.save(sub_dir / f"{k}_{tag}.npy", a)
        # a cheap sanity number per fold: is the export self-consistent at all?
        c = float((l2n(arrs["shared_r"]) * l2n(arrs["shared_r"][:1])).sum(1).mean())
        report["folds"][tag] = {"n": int(len(arrs["shared_r"])),
                                "dim": int(arrs["shared_r"].shape[1]),
                                "cos_to_first": c,
                                "gap_shared_minus_specific": None}
        print(f"[intra_z] {tag}: shared_r {arrs['shared_r'].shape} "
              f"(cos to first row {c:.4f})")

    # The claim this export carries: shared_r is a better encoder space than
    # specific_s for the semantic AND the layout read-out.  Report the mean
    # self-similarity of both, since a space that collapses towards its own mean
    # shows up here as a high value with low downstream discriminability.
    tr = np.load(sub_dir / "shared_r_train.npy")
    sp = np.load(sub_dir / "specific_s_train.npy")
    report["train_cos_to_own_mean"] = {
        "shared_r": float((l2n(tr) * l2n(tr.mean(0, keepdims=True))).sum(1).mean()),
        "specific_s": float((l2n(sp) * l2n(sp.mean(0, keepdims=True))).sum(1).mean()),
    }
    (Path(args.out) / "intra_z_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"[intra_z] done -> {sub_dir}  "
          f"cos_to_own_mean {report['train_cos_to_own_mean']}")


if __name__ == "__main__":
    main()
