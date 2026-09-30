#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-atmbrevl
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Eval ATM-bridge ClipRefiner on sub-08 vs raw ATM baseline.
# Writes only under outputs/eval/atm_bridge_sub08/.

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
test -f outputs/atm_bridge_sub08/checkpoints/atm_stage1_best_e30.pt
test -f outputs/atm_bridge/sub-08_test_eeg_1024.npy
test -f outputs/eval/atm_baseline/test_ViT-H-14_laion2b_s32b_b79k_features.npy

python scripts/eval_atm_bridge.py \
  --checkpoint outputs/atm_bridge_sub08/checkpoints/atm_stage1_best_e30.pt \
  --subject sub-08 \
  --bridge-dir outputs/atm_bridge \
  --config configs/atm_bridge_sub08.yaml \
  --output-dir outputs/eval/atm_bridge_sub08 \
  --batch-size 256
echo "[OK] ATM bridge eval finished"
