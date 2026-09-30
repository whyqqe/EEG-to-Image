#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-stgate
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --exclude=dgx-10,dgx-11
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# ST-GATE LOSO — prefer overnight_sci_pipeline.sbatch (SCI canonical)
# This file kept as ST-GATE-only entry; skips open_clip redownload.

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
mkdir -p outputs/slurm outputs/st_gate

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
print("GPU", torch.cuda.get_device_name(0))
PY

OUT=outputs/st_gate/loso_v1
python scripts/train_st_gate_loso.py \
  --output-dir "${OUT}" \
  --backbone atm_style \
  --nz 256 \
  --profile-dim 128 \
  --rank 16 \
  --epochs 25 \
  --max-per-sub 0 \
  --batch-size 128 \
  --lr 3e-4 \
  --lambda-nce 0.5 \
  --lambda-atm 0.2 \
  --lambda-adv 0.1 \
  --lambda-geo 0.25 \
  --grl-max 1.0 \
  --ctx-m 32 \
  --null-profile-p 0.25 \
  --m-list 0,50,100 \
  --k-list 0,1,5,10,20 \
  --adapt-lr 5e-4 \
  --adapt-steps 100 \
  --eval-geo \
  --seed 42

echo "[OK] ST-GATE LOSO finished"
