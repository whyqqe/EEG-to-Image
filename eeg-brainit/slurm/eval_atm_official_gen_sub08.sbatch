#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-atmoffgen
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Official ATM generated images download + PixCorr/SSIM/CLIP metrics (sub-08).

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/eeg-brainit/open_clip
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
export TMPDIR=/project/peilab/why/cache/eeg-brainit/tmp
mkdir -p "$TMPDIR"

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
test -d /project/peilab/why/data/images_set/test_images

python scripts/eval_atm_official_gen.py \
  --subject sub-08 \
  --output-dir outputs/eval/atm_official_gen_sub08 \
  --cache-dir /project/peilab/why/cache/eeg-brainit/hf
echo "[OK] official ATM gen metrics finished"
