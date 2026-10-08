#!/bin/bash
# Pipeline stage C -- retrieval evaluation (no generation).
# Frozen features + label-free geometric calibration, so this is CPU-only work and
# does not consume a GPU (AGENTS.md §3.2). Reports the full ladder:
# raw cosine -> SAW whitening -> CSLS -> coordinate recovery.
#SBATCH --job-name=samclip-eval
#SBATCH --partition=cpu
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/samclip-eval-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/samclip-eval-%j.err

set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
mkdir -p "${ROOT}/outputs/slurm"

export PYTHONNOUSERSITE=1
export HF_HOME=/project/peilab/why/cache/huggingface
export TORCH_HOME=/project/peilab/why/cache/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
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
echo "[slurm] eval job=${SLURM_JOB_ID} host=$(hostname) date=$(date -Is)"

TARGET_SUBJECT="${TARGET_SUBJECT:?TARGET_SUBJECT is required}"
TAG="sub-$(printf '%02d' "${TARGET_SUBJECT}")"
# CKPT may hold several checkpoints so that one eval table compares them. They are
# separated by ':' rather than whitespace: this value travels through
# `sbatch --export=ALL,...`, where a space survives only as long as every layer in
# between quotes it correctly -- and one that does not would silently hand the whole
# string to `--ckpts` as a single, non-existent path. Paths here never contain ':'.
IFS=':' read -r -a CKPTS <<< "${CKPT:?CKPT is required}"
OUT="${OUT:-${ROOT}/outputs/eval/${TAG}.json}"

ARGS=(--ckpts "${CKPTS[@]}" --target-subject "${TARGET_SUBJECT}" --out "${OUT}")
[[ -n "${MVNN:-}" ]] && ARGS+=(--mvnn "${MVNN}")
[[ -n "${CSLS_K:-}" ]] && ARGS+=(--csls-k "${CSLS_K}")
[[ -n "${RHO:-}" ]] && ARGS+=(--rho "${RHO}")
[[ -n "${MIN_LANDMARK_RATE:-}" ]] && ARGS+=(--min-landmark-rate "${MIN_LANDMARK_RATE}")
[[ -n "${RECOVERY:-}" ]] && ARGS+=(--recovery)
[[ -n "${DUMP_RANKS:-}" ]] && ARGS+=(--dump-ranks)

echo "[slurm] eval ckpts=${CKPTS[*]} out=${OUT} mvnn=${MVNN:-auto}"
"${PY}" scripts/run_eval.py "${ARGS[@]}"

echo "[slurm] eval finished rc=$? at $(date -Is)"
