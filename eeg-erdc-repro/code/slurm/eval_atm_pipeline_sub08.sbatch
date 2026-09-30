#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-atmeval
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Full ATM eval: retrieval + SDXL/IP-Adapter generation from atm/prior/bit_clip (sub-08).
# Uses distill S1/S3 checkpoints and project HF cache only.

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export HUGGINGFACE_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export TRANSFORMERS_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export DIFFUSERS_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/eeg-brainit/open_clip
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
export TMPDIR=/project/peilab/why/cache/eeg-brainit/tmp
mkdir -p "$TMPDIR" "$HF_HUB_CACHE/.locks"

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
test -f checkpoints/atm_diffusion_prior/sub-08/diffusion_prior.pt
test -f outputs/atm_bridge/sub-08_test_eeg_1024.npy
test -f outputs/eval/atm_baseline/test_ViT-H-14_laion2b_s32b_b79k_features.npy
test -d /project/peilab/why/data/images_set/test_images
# Require complete SDXL snapshot (model_index.json)
python - <<'PY'
from pathlib import Path
hub = Path("/project/peilab/why/cache/eeg-brainit/hf/hub")
snaps = list((hub / "models--stabilityai--stable-diffusion-xl-base-1.0" / "snapshots").glob("*/model_index.json"))
assert snaps, "SDXL incomplete; run slurm/download_sdxl.sbatch first"
print("SDXL OK", snaps[0].parent)
PY

python scripts/eval_atm_pipeline.py \
  --subject sub-08 \
  --prior-ckpt checkpoints/atm_diffusion_prior/sub-08/diffusion_prior.pt \
  --s1-ckpt outputs/atm_distill_s1_sub08/checkpoints/atm_stage1_best.pt \
  --s3-ckpt outputs/atm_distill_s3_sub08/checkpoints/atm_stage3_best.pt \
  --output-dir outputs/eval/atm_pipeline_sub08 \
  --gen-sources prior,atm,bit_clip \
  --prior-steps 50 \
  --prior-guidance 5.0 \
  --gen-steps 30 \
  --gen-guidance 5.0 \
  --gen-size 512
echo "[OK] ATM pipeline eval finished"
