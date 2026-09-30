#!/usr/bin/env python3
"""Export SAMGA's EEG-side representations for one subject, with stimulus indices.

WHY REUSE THE OFFICIAL DATASET
------------------------------
The features this script writes are only meaningful if the EEG going in is identical to
the EEG SAMGA was trained and evaluated on: same channel set, same `[0, 250]` time window,
the same offline MVNN-normalised arrays, and the same 4-repetition averaging. Re-deriving
any of that here would be a second, silently diverging implementation of the preprocessing
that produced our 53.23%-class retrieval encoder. So we instantiate SAMGA's own
`EEGPreImageDataset` from `third_party/SAMGA/module/dataset.py` and let it do the work.

The returned tuple already carries `object_idx` and `image_idx`
(`dataset.py:212-220`), which is what lets every downstream script index the CLIP target
array by stimulus identity instead of by row position. That matters because the two
projects' arrays are written by different code paths, and indexing by identity makes a row
order disagreement a hard error rather than a wrong number.

WHAT IS EXPORTED
----------------
    hidden     (N, 1024)  TSConv output      -- input to the generation head
    retrieval  (N,  512)  share_encoder(...) -- SAMGA's own embedding, for retrieval
                                                cross-checks against the training log
    object_idx (N,)       concept index      -- indexes the CLIP target's axis 0
    image_idx  (N,)       image-within-concept
    subject_id (N,)

NOTE ON AUGMENTATION
--------------------
`eeg_transform=None` and `frozen_eeg_prior=True` together guarantee no augmentation is
applied: the dataset skips its transform block entirely when `frozen_eeg_prior` is set
(`dataset.py:91-99`), and there is no transform to apply regardless. Features meant to
train or drive a decoder must be the clean averaged responses, not an augmented draw.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "recon"))

from samga_recon import CHANNELS, SAMPLE_POINTS, SamgaReconModel  # noqa: E402

DEFAULT_EEG_DIR = REPO / "data" / "preprocessed_eeg"
DEFAULT_IMG_FEATURE_DIR = REPO / "data" / "image_feature" / "internvit_multilevel_20_24_28_32_36"
DEFAULT_LAYER_IDS = (20, 24, 28, 32, 36)
TIME_WINDOW = [0, 250]   # same as inter.sh / train.py's default
SUBJECT_ID = {"sub-01": 1, "sub-02": 2, "sub-03": 3, "sub-04": 4, "sub-05": 5,
              "sub-06": 6, "sub-07": 7, "sub-08": 8, "sub-09": 9, "sub-10": 10}


def prepare_stacked_image_dir(src_dir: Path, layer_ids, cache_root: Path) -> Path:
    """Per-layer dir -> dir holding `image_{split}.npy` of shape [Nobj, Nimg, K, D].

    A faithful copy of SAMGA's own `prepare_multilayer_feature_dir`
    (`third_party/SAMGA/train.py:103-150`), which we cannot import: that module runs
    module-level code before its `if __name__ == '__main__'` guard, so importing it has
    side effects. The logic is short and the format is the contract, so it is reproduced
    exactly -- `np.stack(layer_arrays, axis=2)`, same cache naming, same shape check.

    This is needed because `EEGPreImageDataset` unconditionally loads
    `image_{split}.npy` from the directory it is given (`dataset.py:133`). We only want the
    dataset for its EEG path -- the image features it returns are discarded -- but it must
    be able to load them, so the stacked form has to exist.
    """
    dst_dir = cache_root / ("stacked_" + "_".join(map(str, layer_ids)))
    dst_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        dst = dst_dir / f"image_{split}.npy"
        if dst.exists():
            print(f"[INFO] cached stacked {dst.name}: {np.load(dst, mmap_mode='r').shape}")
            continue
        arrays = []
        for lid in layer_ids:
            cand = src_dir / f"image_{split}_layer{lid}.npy"
            if not cand.exists():
                raise SystemExit(
                    f"[FATAL] missing {cand}. Point --img-feature-dir at a folder holding "
                    f"image_{{split}}_layer{{{'|'.join(map(str, layer_ids))}}}.npy, or at an "
                    f"already-stacked folder holding image_{{split}}.npy."
                )
            arrays.append(np.load(cand))
        base = arrays[0].shape
        for lid, arr in zip(layer_ids, arrays):
            if arr.shape != base:
                raise SystemExit(
                    f"[FATAL] layer {lid} has shape {arr.shape}, expected {base}")
        stacked = np.stack(arrays, axis=2)      # [Nobj, Nimg, K, D]
        np.save(dst, stacked)
        print(f"[INFO] wrote {dst} shape={stacked.shape}")
    return dst_dir


def build_dataset(subject_id: int, split: str, eeg_dir: Path, img_feature_dir: Path):
    samga = REPO / "third_party" / "SAMGA"
    if str(samga) not in sys.path:
        sys.path.insert(0, str(samga))
    from module.dataset import EEGPreImageDataset  # noqa: E402

    return EEGPreImageDataset(
        subject_ids=[subject_id],
        eeg_data_dir=str(eeg_dir),
        selected_channels=[],                 # [] -> all 63 channels, as in inter.sh
        time_window=TIME_WINDOW,
        image_feature_dir=str(img_feature_dir),
        text_feature_dir="",
        image_aug=False,
        aug_image_feature_dirs=[],
        average=True,
        _random=False,
        eeg_transform=None,                   # no augmentation: clean averaged responses
        train=(split == "train"),
        image_test_aug=False,
        eeg_test_aug=False,
        frozen_eeg_prior=True,                # skips the dataset's transform block
    )


@torch.inference_mode()
def export(model: SamgaReconModel, ds, device: torch.device, batch: int,
           limit: int = 0) -> dict:
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False,
                                         num_workers=0, drop_last=False)
    hid, ret, obj, img, sub = [], [], [], [], []
    seen = set()
    n = 0
    t0 = time.time()
    for eeg, _img_feat, _txt, subject_id, object_idx, image_idx, _rep_idx in loader:
        eeg = eeg.to(device, non_blocking=True)
        out = model(eeg)
        hid.append(out["hidden"].float().cpu().numpy())
        ret.append(out["retrieval"].float().cpu().numpy())
        oi = object_idx.numpy()
        ii = image_idx.numpy()
        obj.append(oi)
        img.append(ii)
        sub.append(subject_id.numpy())
        seen.update(zip(oi.tolist(), ii.tolist()))
        n += eeg.shape[0]
        if n % (batch * 20) < batch:
            print(f"    {n} rows  ({time.time() - t0:.0f}s)", flush=True)
        if limit and n >= limit:
            break

    hidden = np.concatenate(hid, 0).astype(np.float32)
    retrieval = np.concatenate(ret, 0).astype(np.float32)
    object_idx = np.concatenate(obj, 0)
    image_idx = np.concatenate(img, 0)
    subject_id = np.concatenate(sub, 0)

    # A dataset that did NOT average repetitions would return 4x the rows and, worse,
    # duplicate (object, image) keys. Both are caught here rather than propagating into a
    # head trained on four correlated copies of every sample.
    dup = len(seen) != len(object_idx)
    return {
        "hidden": hidden, "retrieval": retrieval,
        "object_idx": object_idx, "image_idx": image_idx, "subject_id": subject_id,
        "_diag": {
            "n_rows": int(len(object_idx)),
            "n_unique_stimuli": len(seen),
            "repetitions_averaged": not dup,
            "hidden_dim": int(hidden.shape[1]),
            "retrieval_dim": int(retrieval.shape[1]),
            "hidden_norm_mean": float(np.linalg.norm(hidden, axis=1).mean()),
            "retrieval_norm_mean": float(np.linalg.norm(retrieval, axis=1).mean()),
            "seconds": round(time.time() - t0, 1),
            "limited": bool(limit),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True, help="SAMGA checkpoint_*.pth")
    ap.add_argument("--subject", required=True, help="sub-08 or 8")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--out", type=Path, required=True, help="output .npz path")
    ap.add_argument("--eeg-dir", type=Path, default=DEFAULT_EEG_DIR)
    ap.add_argument("--img-feature-dir", type=Path, default=DEFAULT_IMG_FEATURE_DIR)
    ap.add_argument("--layer-ids", type=int, nargs="+", default=list(DEFAULT_LAYER_IDS))
    ap.add_argument("--stack-cache", type=Path, default=None,
                    help="where to put the stacked image features (default: under the "
                         "recon output tree, NOT inside the SAMGA run dirs)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    subject = args.subject if args.subject.startswith("sub-") else f"sub-{int(args.subject):02d}"
    if subject not in SUBJECT_ID:
        raise SystemExit(f"[FATAL] unknown subject {subject}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] {subject} split={args.split} device={device}")

    model = SamgaReconModel(args.ckpt, device=device, verbose=True)
    model.eval()

    # The dataset needs the stacked form; accept either form on input so the caller does
    # not have to know which one a given feature directory happens to hold.
    img_dir = args.img_feature_dir
    if not (img_dir / "image_test.npy").exists():
        cache = args.stack_cache or (REPO / "outputs" / "recon" / "stack_cache")
        print(f"[INFO] {img_dir.name} holds per-layer files; preparing stacked form")
        img_dir = prepare_stacked_image_dir(img_dir, args.layer_ids, cache)

    ds = build_dataset(SUBJECT_ID[subject], args.split, args.eeg_dir, img_dir)
    print(f"[INFO] dataset rows={len(ds)}")

    out = export(model, ds, device, args.batch, args.limit)
    diag = out.pop("_diag")
    print("[DIAG] " + json.dumps(diag, indent=2))
    if not diag["repetitions_averaged"]:
        raise SystemExit(
            f"[FATAL] {diag['n_rows']} rows but only {diag['n_unique_stimuli']} unique "
            f"(object, image) pairs -- repetitions are NOT averaged, so every stimulus "
            f"would appear several times with slightly different features."
        )

    dest = args.out if args.out.is_absolute() else REPO / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **out)
    # Sidecar JSON because npz keys are not self-describing and these fields are the ones
    # a later reader needs in order to trust the file without re-deriving it.
    (dest.with_suffix(".json")).write_text(json.dumps(
        {"subject": subject, "split": args.split, "ckpt": str(args.ckpt), **diag},
        indent=2), encoding="utf-8")
    print(f"[OK] wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
