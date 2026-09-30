#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-nb-bit-smoke
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Smoke: original NeuroBOLT tokens → BiT → SDXL (with/without ATM residual).

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/eeg-brainit/open_clip
export XFORMERS_DISABLED=1

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"

python scripts/eval_nb_bit_direct_smoke.py \
  --subject sub-08 \
  --ckpt outputs/nb_bit_s3_sub08/checkpoints/nb_stage3_best.pt \
  --output-dir outputs/eval/nb_bit_direct_smoke \
  --max-images 8 \
  --gen-steps 20 \
  --gen-size 512

echo "[OK] NeuroBOLT→BiT direct smoke finished"
ls -la outputs/eval/nb_bit_direct_smoke/
ls -la outputs/eval/nb_bit_direct_smoke/generated/*/ 2>/dev/null | head -40
