#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-phys-v2
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_debug_qos
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Note: normal_debug_qos usually starts faster than preempt Priority queue
# (MaxWall=2h). For longer runs use: --partition=normal --qos=normal_qos --time=08:00:00

# Physics residual adapter v2:
#   freeze ATM embeddings + pretrain physics adapter + K-shot personalize

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
CACHE=/project/peilab/why/cache/eeg-brainit
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh

export HOME="${CACHE}/xdg-home"
export XDG_CACHE_HOME="${CACHE}/xdg"
export TMPDIR="${CACHE}/tmp"
export HF_HOME="${CACHE}/hf"
export HF_HUB_CACHE="${CACHE}/hf/hub"
export XFORMERS_DISABLED=1
export PYTHONUNBUFFERED=1
mkdir -p outputs/slurm outputs/physics_prior_adapt

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
python scripts/ensure_physics_prior_assets.py

OUT=outputs/physics_prior_adapt/v2_sub-08
python scripts/train_physics_prior_v2.py \
  --output-dir "${OUT}" \
  --meta-train-subjects sub-01,sub-02,sub-03,sub-04,sub-05,sub-06,sub-07,sub-09 \
  --meta-test-subject sub-08 \
  --pretrain-epochs 20 \
  --max-per-sub 4000 \
  --batch-size 128 \
  --lr 3e-4 \
  --rank 8 \
  --k-shot 50 \
  --inner-steps 30 \
  --inner-lr 1e-2 \
  --lambda-nce 1.0 \
  --lambda-res 1e-3 \
  --lambda-gate 1e-3 \
  --seed 42

echo "[OK] v2 finished"
cat "${OUT}/JOB_COMPLETE.json"
