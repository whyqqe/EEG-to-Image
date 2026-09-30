#!/usr/bin/env python3
"""Prepare NOD EEG–fMRI–stimulus pairs for Phase-1 EEG→fMRI training.

NOD (Scientific Data 2025): same subjects viewed the same naturalistic images with
fMRI / MEG / EEG. EEG OpenNeuro: ds005811; NOD-fMRI: ds004496.

This script:
  1) prints download commands (--print-download-commands)
  2) scans data/nod/raw (--scan-raw)
  3) builds subject skeleton manifest (--build)
  4) builds trial caches from epochs + ciftify betas (--build-pairs)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NOD = ROOT / "data" / "nod"
RAW = NOD / "raw"
PROC = NOD / "processed"
MAN = NOD / "manifests"
sys.path.insert(0, str(ROOT / "src"))


def print_download_commands() -> None:
    eeg = RAW / "ds005811"
    fmri = RAW / "nod_fmri"
    print(
        f"""
# ===== NOD-EEG (ds005811) =====
cd {ROOT}
source scripts/activate.sh
bash scripts/download_nod_eeg_s3.sh

# ===== NOD-fMRI betas only (ds004496) — selective, not full BOLD =====
# Default subject 01; override: NOD_FMRI_SUBJECTS=01,02,03
bash scripts/download_nod_fmri_s3.sh
# Monitor: tail -f outputs/slurm/download_nod_fmri_s3.log

# After both are present:
python scripts/prepare_nod_pairs.py --scan-raw
python scripts/prepare_nod_pairs.py --build-pairs --max-subjects 1 --num-rois 64
""".strip()
    )


def scan_raw() -> dict:
    report = {
        "raw_root": str(RAW),
        "exists": RAW.is_dir(),
        "children": [],
        "hints": [],
        "eeg_epochs": 0,
        "fmri_betas": 0,
    }
    if not RAW.is_dir():
        report["hints"].append(f"Create {RAW} and download NOD data.")
        return report
    for p in sorted(RAW.iterdir()):
        entry = {"name": p.name, "is_dir": p.is_dir()}
        if p.is_dir():
            try:
                n = sum(1 for _ in p.rglob("*") if _.is_file())
            except OSError:
                n = -1
            entry["n_files_approx"] = n
        report["children"].append(entry)
    eeg_ep = RAW / "ds005811" / "derivatives" / "preprocessed" / "epochs"
    if eeg_ep.is_dir():
        report["eeg_epochs"] = len(list(eeg_ep.glob("*_eeg_epo.fif")))
    fmri = RAW / "nod_fmri"
    if fmri.is_dir():
        report["fmri_betas"] = len(list(fmri.glob("derivatives/ciftify/sub-*/results/ses-imagenet*/**/*_beta.dscalar.nii")))
    if report["eeg_epochs"] == 0:
        report["hints"].append("Missing NOD-EEG epochs under data/nod/raw/ds005811/derivatives/preprocessed/epochs/.")
    if report["fmri_betas"] == 0:
        report["hints"].append("Missing NOD-fMRI betas; run bash scripts/download_nod_fmri_s3.sh")
    return report


def build_manifest(max_subjects: int = 0) -> Path:
    MAN.mkdir(parents=True, exist_ok=True)
    scan = scan_raw()
    subjects: list[str] = []
    eeg_root = RAW / "ds005811"
    if eeg_root.is_dir():
        for p in sorted(eeg_root.glob("sub-*")):
            if p.is_dir():
                subjects.append(p.name)
    if max_subjects > 0:
        subjects = subjects[:max_subjects]
    fmri_root = RAW / "nod_fmri"
    manifest = {
        "dataset": "NOD",
        "eeg_root": str(eeg_root) if eeg_root.exists() else None,
        "fmri_root": str(fmri_root) if fmri_root.exists() else None,
        "n_subjects": len(subjects),
        "subjects": subjects,
        "status": "subjects_listed" if subjects else "skeleton",
        "scan": scan,
    }
    out = MAN / "nod_pairs_skeleton.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out} (subjects={len(subjects)})")
    for h in scan.get("hints") or []:
        print(f"[HINT] {h}")
    return out


def build_pairs(
    max_subjects: int = 1,
    num_rois: int = 200,
    max_trials: int = 0,
    rois_per_hemi: int | None = None,
    target_mode: str = "pca",
    pca_dim: int = 128,
    posterior_eeg: bool = True,
    clip_dir: str | None = "data/nod/processed/clip_vit_h14",
    stimvar_top_frac: float = 0.12,
) -> Path:
    from eeg_brainit.data.nod_pairs import (
        apply_pca_inplace,
        attach_clip_to_pairs,
        build_pairs_for_subject,
        cache_pairs,
        fit_joint_stimvar_mask,
    )

    eeg_root = RAW / "ds005811"
    fmri_root = RAW / "nod_fmri"
    subjects = sorted([p.name for p in eeg_root.glob("sub-*") if p.is_dir()])
    # Prefer subjects that already have fMRI betas downloaded.
    fmri_subs = {p.name for p in (fmri_root / "derivatives" / "ciftify").glob("sub-*") if p.is_dir()}
    if fmri_subs:
        subjects = [s for s in subjects if s in fmri_subs]
    if max_subjects > 0:
        subjects = subjects[:max_subjects]
    rph = rois_per_hemi if rois_per_hemi is not None else max(1, num_rois // 2)
    joint_pca = target_mode == "pca" and len(subjects) > 1
    stim_modes = target_mode in {"stimvar", "classmean"}
    n_out = int(pca_dim) if target_mode == "pca" else (int(num_rois) if stim_modes else int(rph * 2))
    all_meta = {
        "dataset": "NOD",
        "target_mode": target_mode,
        "pca_dim": pca_dim if target_mode == "pca" else None,
        "num_rois": n_out,
        "rois_per_hemi": int(rph) if target_mode == "hemi" else None,
        "stimvar_top_frac": float(stimvar_top_frac) if stim_modes else None,
        "roi_scheme": target_mode,
        "posterior_eeg": posterior_eeg,
        "joint_pca": joint_pca,
        "clip_dir": clip_dir,
        "subjects": {},
        "status": "pairs",
    }
    stim_mask = None
    if stim_modes:
        print(f"[INFO] fitting joint stimvar mask top_frac={stimvar_top_frac} on {subjects} ...")
        stim_mask = fit_joint_stimvar_mask(fmri_root, subjects, top_frac=stimvar_top_frac)
        mask_path = PROC / f"stimvar_mask_top{stimvar_top_frac:.2f}.npy"
        np.save(mask_path, stim_mask)
        all_meta["stimvar_mask"] = str(mask_path)
        all_meta["stimvar_n_vertices"] = int(stim_mask.sum())
        print(f"[INFO] stimvar vertices={int(stim_mask.sum())} → {mask_path}")

    bags: list[tuple[str, list]] = []
    for sub in subjects:
        print(f"[INFO] pairing {sub} mode={target_mode} joint_pca={joint_pca} ...")
        pairs = build_pairs_for_subject(
            eeg_root,
            fmri_root,
            sub,
            num_rois=n_out,
            max_trials=max_trials,
            rois_per_hemi=rph,
            target_mode=target_mode,
            pca_dim=pca_dim,
            posterior_eeg=posterior_eeg,
            fit_pca=not joint_pca,
            stimvar_mask=stim_mask,
            stimvar_top_frac=stimvar_top_frac,
            stimvar_num_rois=n_out if stim_modes else None,
        )
        bags.append((sub, pairs))

    if joint_pca:
        flat = [p for _, ps in bags for p in ps]
        print(f"[INFO] fitting joint PCA dim={pca_dim} on {len(flat)} trials ...")
        apply_pca_inplace(flat, pca_dim=pca_dim)
        if flat:
            all_meta["pca_explained"] = flat[0]["pca_explained"]
            all_meta["num_rois"] = flat[0]["pca_dim"]

    total = 0
    for sub, pairs in bags:
        if clip_dir:
            cdir = ROOT / clip_dir if not Path(clip_dir).is_absolute() else Path(clip_dir)
            if (cdir / "index.json").is_file():
                pairs = attach_clip_to_pairs(pairs, cdir)
                print(f"[INFO] attached CLIP from {cdir} → {len(pairs)} pairs")
            else:
                print(f"[WARN] CLIP dir missing ({cdir}); pairs without clip")
        out_dir = PROC / target_mode / sub
        man = cache_pairs(pairs, out_dir)
        all_meta["subjects"][sub] = {
            "n_pairs": len(pairs),
            "cache": str(out_dir / "pairs.npz"),
            "meta": str(man),
        }
        if pairs and "pca_explained" in pairs[0] and "pca_explained" not in all_meta:
            all_meta["pca_explained"] = pairs[0]["pca_explained"]
            all_meta["num_rois"] = pairs[0]["pca_dim"]
        total += len(pairs)
        print(f"[OK] {sub}: {len(pairs)} pairs → {out_dir}")
    out = MAN / f"nod_pairs_{target_mode}.json"
    all_meta["n_pairs_total"] = total
    out.write_text(json.dumps(all_meta, indent=2), encoding="utf-8")
    # Keep nod_pairs.json as alias to the latest build for convenience.
    (MAN / "nod_pairs.json").write_text(json.dumps(all_meta, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out} and nod_pairs.json total_pairs={total}")
    if total == 0:
        print("[WARN] zero pairs — check image_id overlap and that fMRI betas finished downloading.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-download-commands", action="store_true")
    parser.add_argument("--scan-raw", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--build-pairs", action="store_true")
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--num-rois", type=int, default=200, help="Total ROIs (= 2 * rois_per_hemi).")
    parser.add_argument("--rois-per-hemi", type=int, default=0, help="If >0, overrides num-rois/2.")
    parser.add_argument("--target-mode", type=str, default="pca", choices=["pca", "hemi", "stimvar", "classmean"])
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--stimvar-top-frac", type=float, default=0.12, help="Fraction of high-variance vertices for stimvar/classmean.")
    parser.add_argument("--all-channels", action="store_true", help="Use all EEG channels (default: posterior).")
    parser.add_argument(
        "--clip-dir",
        type=str,
        default="data/nod/processed/clip_vit_h14",
        help="CLIP cache from scripts/precompute_nod_clip.py; empty to disable.",
    )
    parser.add_argument("--max-trials", type=int, default=0, help="Cap pairs per subject (debug).")
    args = parser.parse_args()

    for d in (NOD, RAW, PROC, MAN):
        d.mkdir(parents=True, exist_ok=True)

    if args.print_download_commands:
        print_download_commands()
        return
    if args.scan_raw:
        print(json.dumps(scan_raw(), indent=2))
        return
    if args.build_pairs:
        ms = args.max_subjects if args.max_subjects > 0 else 1
        rph = args.rois_per_hemi if args.rois_per_hemi > 0 else None
        clip_dir = args.clip_dir.strip() or None
        build_pairs(
            max_subjects=ms,
            num_rois=args.num_rois,
            max_trials=args.max_trials,
            rois_per_hemi=rph,
            target_mode=args.target_mode,
            pca_dim=args.pca_dim,
            posterior_eeg=not args.all_channels,
            clip_dir=clip_dir,
            stimvar_top_frac=args.stimvar_top_frac,
        )
        return
    if args.build:
        build_manifest(max_subjects=args.max_subjects)
        return
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
