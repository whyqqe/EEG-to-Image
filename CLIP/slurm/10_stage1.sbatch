#!/bin/bash
# Pipeline stage A -- train the shared cross-subject EEG encoder for one LOSO fold.
# Single stage: no hypernetwork, no mapping network, no conditioning.
# Submit through scripts/submit_pipeline.py so the dependency chain is set up; running
# it directly requires TARGET_SUBJECT and OUT_DIR to be exported.
#SBATCH --job-name=samclip-s1
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/samclip-s1-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/samclip-s1-%j.err

set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
mkdir -p "${ROOT}/outputs/slurm"

export PYTHONNOUSERSITE=1
export HF_HOME=/project/peilab/why/cache/huggingface
export HF_HUB_CACHE=/project/peilab/why/cache/huggingface/hub
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/open_clip
export TORCH_HOME=/project/peilab/why/cache/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
export PIP_CACHE_DIR=/project/peilab/why/cache/pip
# TMPDIR is NODE-LOCAL on purpose (see samclip.config): NFS + multiprocessing cannot
# rmtree its own per-worker temp dirs (unlink-while-open is silly-renamed to `.nfs*`,
# which stays visible -> EBUSY), which buried every job in false tracebacks. The other
# two are small persistent state, so they stay in the project.
export TMPDIR="/tmp/clip-${SLURM_JOB_ID:-local}"
export XDG_CONFIG_HOME="${ROOT}/scratch/config"
export MPLCONFIGDIR="${ROOT}/scratch/mpl"
mkdir -p "${TMPDIR}" "${XDG_CONFIG_HOME}" "${MPLCONFIGDIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT}"

echo "[slurm] stage1 job=${SLURM_JOB_ID} host=$(hostname) date=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
nvidia-smi --query-gpu=driver_version --format=csv,noheader || true

# hard CUDA gate: fail fast instead of silently training on CPU for hours
"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

TARGET_SUBJECT="${TARGET_SUBJECT:?TARGET_SUBJECT is required}"
CONFIG="${CONFIG:-configs/loso_s1.yaml}"
OUT_DIR="${OUT_DIR:-${ROOT}/outputs/stage1/sub-$(printf '%02d' "${TARGET_SUBJECT}")}"

ARGS=(--config "${CONFIG}" --target-subject "${TARGET_SUBJECT}" --out-dir "${OUT_DIR}")
if [[ -n "${EPOCHS:-}" ]]; then ARGS+=(--epochs "${EPOCHS}"); fi
if [[ -n "${SEED:-}" ]]; then ARGS+=(--seed "${SEED}"); fi
if [[ -n "${MVNN:-}" ]]; then ARGS+=(--mvnn "${MVNN}"); fi
if [[ -n "${CHANNEL_SET:-}" ]]; then ARGS+=(--channel-set "${CHANNEL_SET}"); fi
if [[ -n "${DEBUG_STEPS:-}" ]]; then ARGS+=(--debug-steps "${DEBUG_STEPS}"); fi
if [[ -n "${STAGE1_EPOCHS:-}" ]]; then ARGS+=(--stage1-epochs "${STAGE1_EPOCHS}"); fi
if [[ -n "${NO_SCHEDULE:-}" ]]; then ARGS+=(--no-schedule); fi

echo "[slurm] stage1 config=${CONFIG} out=${OUT_DIR} extra=${ARGS[*]}"
"${PY}" scripts/run_stage1.py "${ARGS[@]}"

echo "[slurm] stage1 finished rc=$? at $(date -Is)"
