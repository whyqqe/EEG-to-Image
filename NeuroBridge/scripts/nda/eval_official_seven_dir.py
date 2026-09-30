#!/usr/bin/env python3
"""Official seven metrics for one generation dir (ATM / MindEye / CogCap protocol).

Seven:
  PixCorr↑, SSIM↑ (skimage), AlexNet(2/5)↑ 2-way, Inception↑ 2-way, CLIP↑ 2-way, SwAV↓

Sources:
  - ATM NeurIPS'24 Reconstruction_Metrics (MindEye-style 2-way identification)
  - PixCorr/SSIM: eval_standard7.lowlevel (gray@425 gaussian) — NOT eval_paper_metrics RGB@256
  - Local: erdc_twoway_metrics.py + SwAV-ResNet50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def twoway_per_sample(sim: np.ndarray) -> np.ndarray:
    """The 2-way accuracy decomposed per ground-truth index.

    `twoway(sim)` is the fraction of off-diagonal pairs the diagonal beats, which is
    the mean over `i` of

        q_i = (1 / (n-1)) * #{ j != i : sim[i, i] > sim[i, j] }

    and `q_i` is exactly the per-concept score. It is returned separately because the
    aggregate alone is not testable across arms: the 200 * n comparisons inside one
    matrix are not independent (they share the same 200 rows), so the effective
    sample size is n = 200, not n(n-1) = 39,800, and the standard error of the mean
    is ~sqrt(p(1-p)/n) ~ 0.03 at p = 0.78.

    Every generation arm is scored on the SAME 200 test concepts in the SAME order,
    which makes the arms paired. Keeping `q` makes the paired comparison possible
    after the fact (paired bootstrap / McNemar on the sign of q_a - q_b), which the
    aggregate JSON cannot support.
    """
    n = sim.shape[0]
    # The diagonal of the comparison matrix is set to +inf, not -inf: `-inf` would
    # make `sim[i, i] > off[i, i]` true and silently count the j == i pair as a win,
    # inflating the per-concept score to (1 + wins_i) / (n - 1) and the mean to
    # (n + correct) / (n(n-1)). `+inf` makes the self-pair always lose, which is what
    # "the other n-1 gallery entries" means.
    off = sim.copy()
    np.fill_diagonal(off, np.inf)
    return (sim.diagonal()[:, None] > off).sum(axis=1) / float(n - 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", type=str, required=True)
    ap.add_argument("--output-json", type=str, required=True)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--skip-if-exists", action="store_true")
    args = ap.parse_args()

    out_p = Path(args.output_json)
    if args.skip_if_exists and out_p.is_file():
        print(f"[SKIP] {out_p}")
        return

    nb = Path("/project/peilab/why/NeuroBridge")
    sys.path.insert(0, str(nb / "scripts" / "nda"))
    sys.path.insert(0, "/project/peilab/why/eeg-brainit/scripts")

    from eval_atm_pipeline import list_test_images  # type: ignore
    from eval_standard7 import lowlevel, pearson_sim  # type: ignore
    from eval_standard_seven_table import compute_swav_distance  # type: ignore
    from erdc_fid_metrics import compute_fid  # type: ignore
    from erdc_twoway_metrics import encode_bundle, list_gen, twoway  # type: ignore

    gen_dir = Path(args.gen_dir)
    tag = args.tag or gen_dir.parent.name
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    gt = list_test_images(Path(args.images_root))
    gens = list_gen(gen_dir, len(gt))

    # --- 2-way (ATM official high-level suite) ---
    cache = gen_dir.parent / "_twoway_cache"
    cache.mkdir(parents=True, exist_ok=True)
    t_cache, g_cache = cache / "gt_feats.npz", cache / f"gen_feats_{tag}.npz"
    if t_cache.is_file():
        t = {k: np.load(t_cache)[k] for k in ["clip", "alex2", "alex5", "inception"]}
    else:
        t = encode_bundle(gt, device)
        np.savez_compressed(t_cache, **t)
    if g_cache.is_file():
        g = {k: np.load(g_cache)[k] for k in ["clip", "alex2", "alex5", "inception"]}
    else:
        g = encode_bundle(gens, device)
        np.savez_compressed(g_cache, **g)

    # The similarity statistic is NOT cosmetic. `encode_bundle` L2-normalises every
    # feature, so `t @ g.T` is cosine, which keeps the component shared by all 200
    # images; Pearson removes it. The published bars this table is compared against
    # (ATM 0.734 inception / 0.786 clip, CogCap, and our own HCMA prior) all come
    # from the Pearson form -- the official ATM notebook uses np.corrcoef -- so the
    # Pearson number is the headline and the cosine one is kept only so the rows can
    # be matched against JSONs written before this file changed.
    # `scripts/nda/nw4_official_twoway.py` documents the same discrepancy and
    # measured up to 0.21 on Inception; on these SDXL generations it is ~0.03.
    tw = {k: {} for k in ["clip", "alex2", "alex5", "inception"]}
    per_sample: dict[str, list[float]] = {}
    for key in ["clip", "alex2", "alex5", "inception"]:
        sim_p = pearson_sim(t[key], g[key])
        sim_c = t[key] @ g[key].T
        q = twoway_per_sample(sim_p)
        tw[key] = {"official": twoway(sim_p), "cosine": twoway(sim_c)}
        per_sample[key] = [float(v) for v in q]
        # The decomposition must reproduce the aggregate exactly or one of the two
        # is wrong; asserted rather than assumed because every paired claim
        # downstream is built on `q`.
        assert abs(float(q.mean()) - tw[key]["official"]) < 1e-12, (key, q.mean())

    # --- PixCorr / SSIM: official Ozcelik/ATM/CogCap (gray@425, gaussian) ---
    # Do NOT use eval_paper_metrics for these two. That path is RGB@256 / data_range=255
    # and is not comparable to the published tables.
    ll = lowlevel(gens, gt)
    fid = compute_fid(gen_dir, gt, device, args.batch_size)
    clip_cosine = float((t["clip"] * g["clip"]).sum(1).mean())

    # --- SwAV ---
    swav = compute_swav_distance(gens, gt, device, batch_size=args.batch_size)

    report = {
        "tag": tag,
        "protocol": {
            "name": "ATM / MindEye / CogCap standard seven",
            "metrics": [
                "PixCorr↑ (Pearson RGB @425 BILINEAR)",
                "SSIM↑ (skimage gray@425 gaussian σ=1.5, use_sample_covariance=False, data_range=1.0)",
                "AlexNet(2)↑ two-way ID (features[5])",
                "AlexNet(5)↑ two-way ID (features[12])",
                "Inception↑ two-way ID",
                "CLIP↑ two-way ID (OpenCLIP ViT-H/14)",
                "SwAV↓ mean(1-pearson) SwAV-ResNet50",
            ],
            "twoway_impl": "eeg-brainit/scripts/erdc_twoway_metrics.py",
            "twoway_statistic": "Pearson (centred cosine). The `<key>_cos` values are the "
                                "uncentred cosine form written by earlier revisions of "
                                "this file; only the Pearson form is comparable to the "
                                "published ATM/CogCap bars.",
            "lowlevel_impl": "scripts/nda/eval_standard7.py:lowlevel",
            "note": "PixCorr/SSIM match ATM Reconstruction_Metrics / Brain-HIVE utils_eval",
        },
        "pixcorr": float(ll["pixcorr"]),
        "ssim": float(ll["ssim"]),
        "alex2": float(tw["alex2"]["official"]),
        "alex5": float(tw["alex5"]["official"]),
        "inception": float(tw["inception"]["official"]),
        "clip": float(tw["clip"]["official"]),
        "alex2_cos": float(tw["alex2"]["cosine"]),
        "alex5_cos": float(tw["alex5"]["cosine"]),
        "inception_cos": float(tw["inception"]["cosine"]),
        "clip_cos": float(tw["clip"]["cosine"]),
        "swav": float(swav),
        "fid": float(fid["fid"] if isinstance(fid, dict) else fid),
        "clip_cosine": clip_cosine,
        "gen_dir": str(gen_dir),
        "n": len(gens),
    }
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # The per-concept 2-way scores, which the aggregate above cannot be tested from.
    # Written next to the report and deliberately NOT inside `_twoway_cache`, because
    # that directory is deleted by the runner once the metrics are in.
    per_p = out_p.with_name(out_p.stem + "_persample.json")
    per_p.write_text(json.dumps({
        "tag": tag,
        "n": len(gens),
        "order": "index i is test concept i, the same order as list_test_images() "
                 "and as the EEG test rows",
        "definition": "q_i = fraction of the other n-1 gallery entries that ground "
                      "truth i beats under the official (Pearson) similarity; "
                      "mean(q) == the reported two-way accuracy",
        "statistic": "pearson",
        "q": per_sample,
        # The low-level metrics have no identification decomposition, so they are
        # written per IMAGE instead. Index i is the same test concept as `q`'s index
        # i, which is what makes them pairable against another arm -- without this,
        # PixCorr (the metric the low-level arms are decided on) would be the only
        # headline number in the project with no interval.
        "per_image_lowlevel": {
            "pixcorr": ll["pixcorr_per_image"],
            "ssim": ll["ssim_per_image"],
            "definition": "per test concept i, computed against ground truth i in the "
                          "official protocol (RGB @425 BILINEAR for PixCorr; skimage "
                          "gray @425 gaussian sigma=1.5 for SSIM)",
        },
    }, indent=2), encoding="utf-8")
    print(f"[OK] per-sample writes {per_p}")

    # also drop regenerable cache to save disk after write
    if g_cache.is_file():
        g_cache.unlink(missing_ok=True)
    print(json.dumps({k: report[k] for k in ["tag", "pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "swav", "fid"]}, indent=2))


if __name__ == "__main__":
    main()
