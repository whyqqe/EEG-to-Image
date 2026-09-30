#!/usr/bin/env python3
"""Prepare NOD EEG↔region-fMRI pairs (EVC / Ventral classmean features + CLIP)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_brainit.data.nod_pairs import (  # noqa: E402
    load_fmri_beta_index,
    pool_stimvar_rois,
)


def _class_prototypes(fmri: dict[str, np.ndarray], trials: list[dict]) -> dict[str, np.ndarray]:
    buckets: dict[str, list[np.ndarray]] = {}
    for t in trials:
        iid = t["image_id"]
        cid = t.get("class_id")
        if cid is None or iid not in fmri:
            continue
        buckets.setdefault(str(cid), []).append(fmri[iid].astype(np.float32).reshape(-1))
    return {c: np.mean(np.stack(v, 0), 0) for c, v in buckets.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmri-root", default="data/nod/raw/nod_fmri")
    parser.add_argument("--roi-dir", default="data/nod/processed/visual_rois")
    parser.add_argument("--clip-dir", default="data/nod/processed/clip_vit_h14_all")
    parser.add_argument("--output-dir", default="data/nod/processed/region_pairs")
    parser.add_argument("--subjects", default="sub-01")
    parser.add_argument("--num-rois-per-region", type=int, default=64)
    parser.add_argument("--use-classmean", action="store_true", default=True)
    parser.add_argument("--no-classmean", action="store_false", dest="use_classmean")
    parser.add_argument("--max-trials", type=int, default=0)
    args = parser.parse_args()

    roi_dir = ROOT / args.roi_dir
    evc_mask = np.load(roi_dir / "evc_mask.npy").astype(bool)
    vent_mask = np.load(roi_dir / "ventral_mask.npy").astype(bool)
    clip_index = json.loads((ROOT / args.clip_dir / "index.json").read_text(encoding="utf-8"))
    clip_emb = np.load(ROOT / args.clip_dir / "embeddings.npy").astype(np.float32)
    clip_emb = clip_emb / (np.linalg.norm(clip_emb, axis=1, keepdims=True) + 1e-8)

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    out_root = ROOT / args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    for subject in subjects:
        print(f"[INFO] building region pairs {subject} classmean={args.use_classmean}")
        fmri, _ = load_fmri_beta_index(ROOT / args.fmri_root, subject)
        classmean_meta = ROOT / "data/nod/processed/classmean" / subject / "pairs_meta.json"
        classmean_npz = ROOT / "data/nod/processed/classmean" / subject / "pairs.npz"
        if not (classmean_meta.is_file() and classmean_npz.is_file()):
            raise FileNotFoundError(f"Need classmean pairs for aligned EEG: {classmean_npz}")

        z = np.load(classmean_npz)
        meta = json.loads(classmean_meta.read_text(encoding="utf-8"))
        trials = meta["trials"]
        eeg = z["eeg"].astype(np.float32)
        fmri_global = z["fmri_roi"].astype(np.float32)
        protos = _class_prototypes(fmri, trials) if args.use_classmean else {}

        eeg_list, evc_list, vent_list, global_list, clip_list, ids, meta_rows = [], [], [], [], [], [], []
        for i, t in enumerate(trials):
            iid = t["image_id"]
            if iid not in fmri or iid not in clip_index:
                continue
            if args.use_classmean:
                cid = str(t.get("class_id"))
                beta = protos.get(cid, fmri[iid])
            else:
                beta = fmri[iid]
            eeg_list.append(eeg[i])
            evc_list.append(pool_stimvar_rois(beta, evc_mask, args.num_rois_per_region))
            vent_list.append(pool_stimvar_rois(beta, vent_mask, args.num_rois_per_region))
            global_list.append(fmri_global[i])
            clip_list.append(clip_emb[clip_index[iid]])
            ids.append(iid)
            meta_rows.append({"image_id": iid, "class_id": t.get("class_id")})
            if args.max_trials and len(ids) >= args.max_trials:
                break

        out_dir = out_root / subject
        out_dir.mkdir(parents=True, exist_ok=True)
        eeg_arr = np.stack(eeg_list, 0)
        evc_arr = np.stack(evc_list, 0)
        vent_arr = np.stack(vent_list, 0)
        global_arr = np.stack(global_list, 0)
        clip_arr = np.stack(clip_list, 0)
        for arr_name, arr in [("evc", evc_arr), ("vent", vent_arr)]:
            mu, sd = arr.mean(0, keepdims=True), arr.std(0, keepdims=True).clip(min=1e-4)
            if arr_name == "evc":
                evc_arr = (arr - mu) / sd
            else:
                vent_arr = (arr - mu) / sd

        np.savez_compressed(
            out_dir / "pairs.npz",
            eeg=eeg_arr.astype(np.float32),
            fmri_evc=evc_arr.astype(np.float32),
            fmri_ventral=vent_arr.astype(np.float32),
            fmri_global=global_arr.astype(np.float32),
            clip=clip_arr.astype(np.float32),
        )
        out_meta = {
            "subject": subject,
            "n": len(ids),
            "ch_names": meta.get("ch_names"),
            "eeg_shape": list(eeg_arr.shape),
            "evc_dim": int(evc_arr.shape[1]),
            "ventral_dim": int(vent_arr.shape[1]),
            "global_dim": int(global_arr.shape[1]),
            "clip_dim": int(clip_arr.shape[1]),
            "sfreq": 250.0,
            "tmin": -0.1,
            "tmax": 0.8,
            "use_classmean": args.use_classmean,
            "num_rois_per_region": args.num_rois_per_region,
            "trials": meta_rows,
            "roi_dir": str(roi_dir),
        }
        (out_dir / "pairs_meta.json").write_text(json.dumps(out_meta, indent=2), encoding="utf-8")
        print(
            f"[OK] {subject} n={len(ids)} eeg={eeg_arr.shape} "
            f"evc={evc_arr.shape} vent={vent_arr.shape} classmean={args.use_classmean}"
        )


if __name__ == "__main__":
    main()
