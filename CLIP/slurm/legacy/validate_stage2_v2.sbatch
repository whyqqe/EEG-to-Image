#!/bin/bash
# Decisive Stage-2 validation, with a MATCHING Stage-1 checkpoint.
#
# The previous validation reused the pre-`z_norm` Stage-1 checkpoint, so the FiLM heads
# were being fed an input scale 20x larger than they were trained on. That makes every
# statement about `z_s` collapse unfalsifiable. This job trains a fresh Stage 1 with the
# current config (so `conditioning.z_norm: unit` is recorded in the checkpoint), then
# runs Stage 2 and reports:
#   * whether the trunk's frozen BatchNorm buffers survive (the freeze fix)
#   * whether `z_s` stays sensitive across support sets (the collapse)
#   * whether the episode's OWN support set beats another subject's (is the
#     conditioning load-bearing at all?)
#SBATCH --job-name=samclip-validate2
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/validate2-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/validate2-%j.err

set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
mkdir -p "${ROOT}/outputs/slurm"

export PYTHONNOUSERSITE=1
export HF_HOME=/project/peilab/why/cache/huggingface
export HF_HUB_CACHE=/project/peilab/why/cache/huggingface/hub
export TORCH_HOME=/project/peilab/why/cache/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
export TMPDIR=/tmp
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT}"
echo "[v2] job=${SLURM_JOB_ID} host=$(hostname) date=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

EPOCHS="${EPOCHS:-8}"

echo "[v2] stage 1 ($EPOCHS epochs, current config incl. z_norm=unit)"
"${PY}" scripts/run_stage1.py \
  --config configs/loso_sub08.yaml \
  --epochs "${EPOCHS}" \
  --out-dir outputs/validate/stage1 \
  --tag validate

echo "[v2] stage 2 (600 episodes)"
"${PY}" scripts/run_stage2_meta.py \
  --config configs/meta_s2.yaml \
  --stage1-dir outputs/validate/stage1 \
  --target-subject 8 \
  --episodes 600 \
  --out-dir outputs/validate/stage2

echo "[v2] eval both checkpoints"
"${PY}" scripts/run_eval.py \
  --ckpts outputs/validate/stage1/last.pt outputs/validate/stage2/last.pt \
  --target-subject 8 --k-support 50 --mvnn test \
  --out outputs/validate/eval.json

echo "[v2] done at $(date -Is)"
