"""Decisive check on the FRLA rows: what do the low-level INIT itself score?

WHY
    `frla_uniform` scored PixCorr 0.341 / SSIM 0.376 -- the highest of every row in
    the run, and roughly double its own no-anchoring control (`frla_off`, 0.171) --
    while its FID was 347, the worst of the run.  That signature (pixel agreement
    way up, perceptual/identity way down) is what it looks like when generation
    STOPS HAPPENING and the model returns its low-frequency initialisation.  So the
    initialisation is scored here with the SAME metric code and the SAME ground
    truth as every other row.  If the two agree, the anchoring did not steer
    generation -- it replaced it.

WHY IT IS SAFE TO RUN NOW
    `pred_lowlevel_rgb_512/` was deleted by the pipeline's cleanup (it is 77 MB per
    subject and only needed at generation time), but `pred_vae_test.npy` survives
    (7 MB) and is the EEG-PREDICTED latent -- this decodes exactly what the
    generator was initialised from, not the ground truth.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

NB_ROOT = os.environ.get("NB_ROOT", "/project/peilab/why/NeuroBridge")
OUT = os.environ.get("OUT", f"{NB_ROOT}/outputs/tdm_all")
SID = int(os.environ.get("SUBJECT", "8"))
sd = f"{SID:02d}"
dest = Path(sys.argv[1] if len(sys.argv) > 1 else f"/tmp/init_check/sub-{sd}")
dest.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, f"{NB_ROOT}/scripts/nda")
from train_eeg_vae_head import decode_latents, resolve_vae          # noqa: E402

hub = Path(os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lat = np.load(f"{OUT}/vae_head/sub-{sd}/pred_vae_test.npy").astype(np.float32)
print(f"[init-check] sub-{sd} EEG-predicted VAE latent {lat.shape} on {dev}")

# `scaling_factor` is the SDXL constant the training script uses.  It is NOT in the
# report's top level, so fall back to the script's own default (0.13025, the SDXL
# constant) rather than hard-coding a guess that could drift from the run.
import json                                                          # noqa: E402
rep = json.load(open(f"{OUT}/vae_head/sub-{sd}/vae_head_report.json"))
sf = float(rep.get("scaling_factor", rep.get("final", {}).get("scaling_factor", 0.13025)))
print(f"[init-check] scaling_factor={sf} | vae head selected_on="
      f"{rep['final'].get('selected_on')} | its own test latent pearson="
      f"{rep['final'].get('pearson')}")

vae = resolve_vae(hub, dev)
bs = 8
for start in range(0, len(lat), bs):
    chunk = torch.from_numpy(lat[start:start + bs]).to(dev)
    for j, im in enumerate(decode_latents(vae, chunk, sf)):
        im.save(dest / f"{start + j:03d}.png")
print(f"[init-check] wrote {len(list(dest.glob('*.png')))} PNGs -> {dest}")
print(f"[init-check] now score it with:\n"
      f"  python scripts/nda/eval_official_seven_dir.py --gen-dir {dest} "
      f"--output-json /tmp/init_check/sub-{sd}.json --tag init_lowlevel "
      f"--images-root /project/peilab/why/data/images_set --device {dev}")
