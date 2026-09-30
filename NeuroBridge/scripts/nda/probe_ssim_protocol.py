#!/usr/bin/env python3
"""One-off probe: recompute SSIM/PixCorr under the ATM/CogCap protocol vs ours."""
import sys
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim_ski
from pathlib import Path

sys.path.insert(0, "/project/peilab/why/eeg-brainit/scripts")
from eval_atm_pipeline import list_test_images  # noqa: E402

root = Path("/project/peilab/why/data/images_set")
gt = list_test_images(root)

dirs = [
    ("sdedit_ll_s082", "/project/peilab/why/NeuroBridge/outputs/atm_aligned_decode/sub-08/generation/sdedit_ll_s082/generated"),
    ("ref_hcma_full_a40", "/project/peilab/why/NeuroBridge/outputs/hcma_10subj/sub-08/generation/hcma_full_a40/generated"),
]
for name, gdir in dirs:
    gens = [Path(gdir) / f"{i:03d}.png" for i in range(200)]
    ssims_rgb256, ssims_atm, pixs_425, pixs_256 = [], [], [], []
    for i in range(200):
        g = Image.open(gens[i]).convert("RGB")
        t = Image.open(gt[i]).convert("RGB")
        ga = np.asarray(g.resize((256, 256), Image.Resampling.BICUBIC))
        ta = np.asarray(t.resize((256, 256), Image.Resampling.BICUBIC))
        ssims_rgb256.append(float(ssim_ski(ta, ga, channel_axis=-1, data_range=255)))
        ga425 = np.asarray(g.resize((425, 425), Image.Resampling.BILINEAR)).astype(np.float64) / 255.0
        ta425 = np.asarray(t.resize((425, 425), Image.Resampling.BILINEAR)).astype(np.float64) / 255.0
        pixs_425.append(float(np.corrcoef(ta425.reshape(1, -1), ga425.reshape(1, -1))[0, 1]))
        pixs_256.append(float(np.corrcoef(ta.astype(np.float64).reshape(1, -1), ga.astype(np.float64).reshape(1, -1))[0, 1]))
        gg = rgb2gray(ga425)
        tt = rgb2gray(ta425)
        ssims_atm.append(float(ssim_ski(tt, gg, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)))
    print(name)
    print(f"  SSIM RGB256 uniform   (ours):       {np.mean(ssims_rgb256):.4f}")
    print(f"  SSIM GRAY425 gaussian (ATM/CogCap): {np.mean(ssims_atm):.4f}")
    print(f"  PixCorr RGB425 (ATM/CogCap):        {np.mean(pixs_425):.4f}")
    print(f"  PixCorr RGB256 (ours):              {np.mean(pixs_256):.4f}")
