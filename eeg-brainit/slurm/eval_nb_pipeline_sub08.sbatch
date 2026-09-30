#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-nbeval
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Retrieval + SDXL gen for NeuroBOLT→BIT (after S0/S1/S3 train).

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
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/eeg-brainit/open_clip
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
export TMPDIR=/project/peilab/why/cache/eeg-brainit/tmp
mkdir -p "$TMPDIR"

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
test -f outputs/nb_bit_s3_sub08/checkpoints/nb_stage3_best.pt

python scripts/eval_neurobolt_pipeline.py \
  --subject sub-08 \
  --s0-ckpt outputs/nb_direct_s0_sub08/checkpoints/nb_stage0_best.pt \
  --s1-ckpt outputs/nb_bit_s1_sub08/checkpoints/nb_stage1_best.pt \
  --s3-ckpt outputs/nb_bit_s3_sub08/checkpoints/nb_stage3_best.pt \
  --output-dir outputs/eval/nb_pipeline_sub08 \
  --gen-sources bit_clip,direct_clip,atm \
  --gen-steps 30 \
  --gen-size 512
echo "[OK] NeuroBOLT pipeline eval finished"
