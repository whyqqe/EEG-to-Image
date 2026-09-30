#!/usr/bin/env python3
"""Build multi-subject z_ret bank with SharedSpecific encoder for RGT-CFM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT))
sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))

from module.dataset import EEGPreImageDataset  # noqa: E402
from module.projector import ProjectorLinear  # noqa: E402
from ss_modules import SharedSpecificEncoder  # noqa: E402

DEFAULT_CHANNELS = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb-root", type=str, default=str(NB_ROOT))
    ap.add_argument("--ss-checkpoint", type=str, required=True)
    ap.add_argument("--subjects", type=str, default="1,2,4,5,6,7,8,9,10")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    root = Path(args.nb_root)
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.ss_checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    subjects = [int(x) for x in args.subjects.split(",") if x.strip()]
    img_dim = int(ckpt["img_dim"])
    feature_dim = int(ckpt["feature_dim"])
    eeg_len = int(ckpt["eeg_sample_points"])
    channels_num = int(ckpt["channels_num"])
    model_subjects = [int(s) for s in ckpt.get("subjects", subjects)]

    model = SharedSpecificEncoder(
        subject_ids=model_subjects,
        feature_dim=img_dim,
        eeg_sample_points=eeg_len,
        channels_num=channels_num,
        n_extra_blocks=int(ckpt.get("n_extra_blocks", 1)),
        use_adapter=True,
    ).to(device)
    eeg_projector = ProjectorLinear(img_dim, feature_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    eeg_projector.load_state_dict(ckpt["eeg_projector_state_dict"])
    model.eval()
    eeg_projector.eval()

    eeg_dir = str(root / "data/things_eeg/preprocessed_eeg")
    rn50_dir = str(root / "data/things_eeg/image_feature/RN50")

    train_zs, train_sids, test_zs, test_sids = [], [], [], []
    per_subj = {}

    for sid in subjects:
        for train_flag, tag, z_list, s_list in (
            (True, "train", train_zs, train_sids),
            (False, "test", test_zs, test_sids),
        ):
            ds = EEGPreImageDataset(
                [sid], eeg_dir, DEFAULT_CHANNELS, [0, 250],
                rn50_dir, "", False, [], True, False, None, train_flag, False, False, False,
            )
            chunks = []
            with torch.no_grad():
                for batch in tqdm(DataLoader(ds, batch_size=512, shuffle=False), desc=f"sub{sid}-{tag}"):
                    eeg, _img, _t, sid_b, *_ = batch
                    eeg = eeg.to(device)
                    sid_b = sid_b.to(device)
                    z = eeg_projector(model(eeg, sid_b))
                    chunks.append(z.float().cpu().numpy())
            arr = np.concatenate(chunks).astype(np.float32)
            z_list.append(arr)
            s_list.append(np.full((arr.shape[0],), sid, dtype=np.int64))
            per_subj.setdefault(str(sid), {})[tag] = {"n": int(arr.shape[0]), "dim": int(arr.shape[1])}
            # also save per-subject for quick access
            np.save(out / f"z_ret_sub{sid:02d}_{tag}.npy", arr)

    z_tr = np.concatenate(train_zs).astype(np.float32)
    s_tr = np.concatenate(train_sids).astype(np.int64)
    z_te = np.concatenate(test_zs).astype(np.float32)
    s_te = np.concatenate(test_sids).astype(np.int64)
    np.save(out / "z_ret_train_all.npy", z_tr)
    np.save(out / "sid_train_all.npy", s_tr)
    np.save(out / "z_ret_test_all.npy", z_te)
    np.save(out / "sid_test_all.npy", s_te)
    # convenience aliases for target subject 8
    if 8 in subjects:
        np.save(out / "z_ret_train.npy", np.load(out / "z_ret_sub08_train.npy"))
        np.save(out / "z_ret_test.npy", np.load(out / "z_ret_sub08_test.npy"))
        # memory router naming
        np.save(out / "z_eeg_proj_train.npy", np.load(out / "z_ret_sub08_train.npy"))
        np.save(out / "z_eeg_proj_test.npy", np.load(out / "z_ret_sub08_test.npy"))

    report = {
        "subjects": subjects,
        "ss_checkpoint": str(ckpt_path),
        "train_n": int(z_tr.shape[0]),
        "test_n": int(z_te.shape[0]),
        "dim": int(z_tr.shape[1]),
        "per_subject": per_subj,
    }
    (out / "bank_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
