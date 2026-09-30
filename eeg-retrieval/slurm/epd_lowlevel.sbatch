#!/usr/bin/env bash
#SBATCH --job-name=epd-lowlevel
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-lowlevel-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-lowlevel-%j.err
#
# The img2img strength frontier on sub-08. No training: the init and the IP condition
# are artifacts of the last dual-tower run, held fixed while the strength varies.
#
# Resource sizing: 6 generations x ~6.5 min (the recorded arms ran 200 images at
# 1.89 s/it over 28 steps) + 8 metric runs x ~1 min + ~2 min of re-scoring the two
# free endpoints. ~50 min of GPU; 2 h covers a slow node.

set -uo pipefail
cd /project/peilab/why/eeg-retrieval

export PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
export TOKENIZERS_PARALLELISM=false
export SLURM_CPU_BIND=none

mkdir -p outputs/slurm outputs/logs

echo "########## job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds) ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# `--dry-run` first: it verifies that every input artifact exists and prints the arm
# table, so a missing init or a renamed condition directory fails in seconds instead of
# after the generations.
bash scripts/run_epd_lowlevel.sh --dry-run || { echo "[FATAL] dry run failed"; exit 2; }

LOG="outputs/logs/run_epd_lowlevel_${SLURM_JOB_ID:-local}.log"
echo "########## full pipeline -> ${LOG} ##########"

set -o pipefail
bash scripts/run_epd_lowlevel.sh 2>&1 | tee "${LOG}"
RC=${PIPESTATUS[0]}

echo "########## job ${SLURM_JOB_ID:-local} rc=${RC} @ $(date -Iseconds) ##########"
exit "${RC}"
