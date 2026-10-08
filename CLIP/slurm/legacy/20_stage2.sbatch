#!/bin/bash
# Pipeline stage 2 -- episodic meta-training of the hypernetwork (new-subject path).
# Depends on stage 1: it loads stage1/last.pt and freezes the shared trunk.
#SBATCH --job-name=samclip-s2
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/samclip-s2-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/samclip-s2-%j.err

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

echo "[slurm] stage2 job=${SLURM_JOB_ID} host=$(hostname) date=$(date -Is)"
"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

TARGET_SUBJECT="${TARGET_SUBJECT:?TARGET_SUBJECT is required}"
TAG="sub-$(printf '%02d' "${TARGET_SUBJECT}")"
STAGE1_DIR="${STAGE1_DIR:-${ROOT}/outputs/stage1/${TAG}}"
OUT_DIR="${OUT_DIR:-${ROOT}/outputs/stage2/${TAG}}"

ARGS=(--config "${CONFIG:-configs/meta_s2.yaml}" --stage1-dir "${STAGE1_DIR}"
      --target-subject "${TARGET_SUBJECT}" --out-dir "${OUT_DIR}")
if [[ -n "${EPISODES:-}" ]]; then ARGS+=(--episodes "${EPISODES}"); fi
if [[ -n "${K_SUPPORT:-}" ]]; then ARGS+=(--k-support "${K_SUPPORT}"); fi

echo "[slurm] stage2 stage1=${STAGE1_DIR} out=${OUT_DIR}"
"${PY}" scripts/run_stage2_meta.py "${ARGS[@]}"

echo "[slurm] stage2 finished rc=$? at $(date -Is)"
