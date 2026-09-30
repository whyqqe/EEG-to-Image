#!/usr/bin/env python3
"""Build NOD visual-cortex ROI masks (EVC / Ventral) on fsLR-32k CIFTI.

Default method partitions the joint stimvar mask (top-variance vertices) into
posterior (EVC) and postero-inferior (Ventral) subregions — same stimulus-driven
mask as the successful classmean pipeline, split by midthickness coordinates.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_fslr_xyz(neuromaps_data: Path):
    import nibabel as nb
    from neuromaps.datasets import fetch_fslr

    os.environ.setdefault("NEUROMAPS_DATA", str(neuromaps_data))
    os.environ.setdefault("HOME", str(neuromaps_data.parent / "xdg-home"))
    at = fetch_fslr(density="32k")
    med_l = np.asarray(nb.load(at["medial"].L).darrays[0].data) != 0
    med_r = np.asarray(nb.load(at["medial"].R).darrays[0].data) != 0
    xyz_l = np.asarray(nb.load(at["midthickness"].L).darrays[0].data, dtype=np.float32)
    xyz_r = np.asarray(nb.load(at["midthickness"].R).darrays[0].data, dtype=np.float32)
    xyz = np.concatenate([xyz_l[med_l], xyz_r[med_r]], axis=0)
    return xyz, int(med_l.sum()), int(med_r.sum())


def _hemi_from_sample_beta(beta_path: Path) -> tuple[np.ndarray, np.ndarray]:
    import nibabel as nb

    img = nb.load(str(beta_path))
    names = np.asarray(list(img.header.get_axis(1).name))
    return names == "CIFTI_STRUCTURE_CORTEX_LEFT", names == "CIFTI_STRUCTURE_CORTEX_RIGHT"


def build_stimvar_partition_masks(
    xyz: np.ndarray,
    stimvar_mask: np.ndarray,
    *,
    evc_post_frac: float = 0.40,
    vent_inf_frac: float = 0.50,
) -> dict[str, np.ndarray]:
    """Split stimvar vertices into exclusive EVC / Ventral by midthickness coords."""
    idx = np.where(stimvar_mask)[0]
    if len(idx) < 200:
        raise RuntimeError(f"stimvar mask too small: {len(idx)}")
    y = xyz[idx, 1]
    z = xyz[idx, 2]
    y_thr = float(np.quantile(y, evc_post_frac))
    evc = np.zeros_like(stimvar_mask, dtype=bool)
    evc[idx[y <= y_thr]] = True

    rem_idx = idx[y > y_thr]
    if len(rem_idx) < 100:
        raise RuntimeError(f"not enough non-EVC stimvar vertices: {len(rem_idx)}")
    z_rem = xyz[rem_idx, 2]
    z_thr = float(np.quantile(z_rem, vent_inf_frac))
    vent = np.zeros_like(stimvar_mask, dtype=bool)
    vent[rem_idx[z_rem <= z_thr]] = True

    if evc.sum() < 100 or vent.sum() < 100:
        raise RuntimeError(f"ROI too small evc={evc.sum()} vent={vent.sum()}")
    return {"evc": evc, "ventral": vent}


def build_anatomical_masks(
    xyz: np.ndarray,
    *,
    evc_post_frac: float = 0.18,
    vent_post_frac: float = 0.45,
    vent_inf_frac: float = 0.40,
) -> dict[str, np.ndarray]:
    y = xyz[:, 1]
    z = xyz[:, 2]
    y_thr_evc = float(np.quantile(y, evc_post_frac))
    evc = y <= y_thr_evc
    y_thr_vent = float(np.quantile(y, vent_post_frac))
    z_thr_vent = float(np.quantile(z, vent_inf_frac))
    vent = (y <= y_thr_vent) & (z <= z_thr_vent) & (~evc)
    if evc.sum() < 100 or vent.sum() < 100:
        raise RuntimeError(f"ROI too small evc={evc.sum()} vent={vent.sum()}")
    return {"evc": evc.astype(bool), "ventral": vent.astype(bool)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmri-root", default="data/nod/raw/nod_fmri")
    parser.add_argument("--subject", default="sub-01")
    parser.add_argument("--output-dir", default="data/nod/processed/visual_rois")
    parser.add_argument("--neuromaps-data", default="/project/peilab/why/cache/eeg-brainit/neuromaps-data")
    parser.add_argument(
        "--method",
        choices=["stimvar_partition", "anatomical"],
        default="stimvar_partition",
    )
    parser.add_argument("--stimvar-top-frac", type=float, default=0.12)
    parser.add_argument("--evc-post-frac", type=float, default=0.40)
    parser.add_argument("--vent-inf-frac", type=float, default=0.50)
    parser.add_argument("--max-images-for-var", type=int, default=0, help="0 = all images")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    xyz, n_l, n_r = _load_fslr_xyz(Path(args.neuromaps_data))
    print(f"[INFO] xyz={xyz.shape} L={n_l} R={n_r}")

    from eeg_brainit.data.nod_pairs import load_fmri_beta_index, stimvar_vertex_mask

    fmri, sample = load_fmri_beta_index(ROOT / args.fmri_root, args.subject)
    if sample is None or not fmri:
        raise FileNotFoundError(f"no betas for {args.subject}")
    left, right = _hemi_from_sample_beta(sample)
    v = int(left.sum() + right.sum())
    if xyz.shape[0] != v:
        raise RuntimeError(f"xyz/beta length mismatch {xyz.shape[0]} vs {v}")

    ids = sorted(fmri.keys())
    if args.max_images_for_var > 0:
        ids = ids[: args.max_images_for_var]
    stack = np.stack([fmri[i] for i in ids], 0)
    stimvar = stimvar_vertex_mask(stack, top_frac=args.stimvar_top_frac)
    print(f"[INFO] stimvar n={int(stimvar.sum())} top_frac={args.stimvar_top_frac}")

    if args.method == "stimvar_partition":
        masks = build_stimvar_partition_masks(
            xyz, stimvar, evc_post_frac=args.evc_post_frac, vent_inf_frac=args.vent_inf_frac
        )
        method_desc = (
            f"stimvar_top{args.stimvar_top_frac}_partition "
            f"evc_post={args.evc_post_frac} vent_inf={args.vent_inf_frac}"
        )
    else:
        masks = build_anatomical_masks(xyz)
        method_desc = "fsLR32k_midthickness_quantile"

    meta = {
        "subject_for_geometry_check": args.subject,
        "n_vertices": v,
        "n_left": int(left.sum()),
        "n_right": int(right.sum()),
        "sample_beta": str(sample),
        "method": method_desc,
        "stimvar_n": int(stimvar.sum()),
    }

    for name, m in masks.items():
        path = out_dir / f"{name}_mask.npy"
        np.save(path, m.astype(np.uint8))
        meta[name] = {"n": int(m.sum()), "path": str(path)}
        print(f"[OK] {name} n={int(m.sum())} -> {path}")

    ov = int((masks["evc"] & masks["ventral"]).sum())
    meta["overlap_evc_ventral"] = ov
    meta["evc_in_stimvar"] = int((masks["evc"] & stimvar).sum())
    meta["ventral_in_stimvar"] = int((masks["ventral"] & stimvar).sum())
    (out_dir / "roi_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] overlap={ov} meta -> {out_dir / 'roi_meta.json'}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    main()
