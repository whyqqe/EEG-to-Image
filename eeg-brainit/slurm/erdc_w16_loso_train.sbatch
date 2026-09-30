#!/usr/bin/env bash
#SBATCH --job-name=erdc-w16-tr
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --array=1-10%3
#SBATCH --exclude=dgx-10,dgx-11
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err

# W16 — per-subject Stage-1 + Stage-3 training (within-subject protocol)

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
export XFORMERS_DISABLED=1
mkdir -p "$TMPDIR" outputs/slurm

IDX=${SLURM_ARRAY_TASK_ID:-1}
SUB=$(printf "sub-%02d" "${IDX}")
echo "[INFO] LOSO train subject=${SUB}"

python scripts/erdc_make_loso_config.py --subject "${SUB}"

S1_CKPT="outputs/atm_distill_s1_${SUB}/checkpoints/atm_stage1_best.pt"
if [[ ! -f "${S1_CKPT}" ]]; then
  echo "----- Train S1 ${SUB} -----"
  python scripts/train_atm_bridge.py \
    --config configs/base.yaml \
    --override "configs/_loso_s1_${SUB}.yaml"
else
  echo "[SKIP] S1 ${SUB}"
fi

# Refresh S3 yaml so init_checkpoint points to this subject's S1
python scripts/erdc_make_loso_config.py --subject "${SUB}"

S3_CKPT="outputs/atm_distill_s3_${SUB}/checkpoints/atm_stage3_best.pt"
if [[ ! -f "${S3_CKPT}" ]]; then
  echo "----- Train S3 ${SUB} -----"
  python scripts/train_atm_bridge.py \
    --config configs/base.yaml \
    --override "configs/_loso_s3_${SUB}.yaml"
else
  echo "[SKIP] S3 ${SUB}"
fi

echo "[OK] LOSO train ${SUB} done"
date
