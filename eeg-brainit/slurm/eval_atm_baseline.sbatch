#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-atmbase
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Safe job: read-only inputs under data/ + eeg-to-image/checkpoints/atm;
# writes only under eeg-brainit/outputs/.

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/eeg-brainit/open_clip
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
test -d /project/peilab/why/eeg-to-image/checkpoints/atm
test -d /project/peilab/why/data/images_set/test_images

python scripts/eval_atm_baseline.py \
  --atm-dir /project/peilab/why/eeg-to-image/checkpoints/atm \
  --images-root /project/peilab/why/data/images_set \
  --output-dir outputs/eval/atm_baseline \
  --clip-model ViT-H-14 \
  --clip-pretrained laion2b_s32b_b79k \
  --verify-teacher \
  --batch-size 32
echo "[OK] ATM baseline finished"
