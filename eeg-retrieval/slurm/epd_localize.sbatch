#!/usr/bin/env bash
#SBATCH --job-name=epd-loc
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-loc-%A_%a.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-loc-%A_%a.err
#
# STAGE 1 -- four arms that localise the shortfall inside our own framework, each one
# variable, all with the two instrument fixes (EMA selection, EMA-scored diagnostics).
# See scripts/run_epd_localize.sh for what each arm is and what is deliberately absent.
#
#   base_ema   current recipe + EMA alone          <- the instrument-fix control
#   ch17       17 occipito-parietal channels       <- the channel axis
#   mmd        SAMGA's two-stage MMD               <- the coarse-alignment axis
#   samgaish   17ch + MMD + softplus + bs512 + lr 1e-4/stage2 5e-5  <- "their recipe,
#                                                     our code", the hypothesis test
#
# Four arms in parallel rather than a sequence because they share only read-only inputs
# and must be compared against each other: same node type, same code revision, same
# hour. A sequence would let the code drift between arms, which is exactly how the
# `prev` run's export came to look for a checkpoint its training had named differently.
#
# Cost per arm: 60 epochs, batch 128 (512 for samgaish), semantic tower only -- the
# 304M-param structural decoder is disabled, which is where the 57 s/epoch of the
# diagnosed run went; the same shape of step measured 15 s/epoch in the sweep. Plus 12
# diagnostic test passes (negligible) and an EMA shadow copy (~0.36 GB, one extra
# no-grad eval pass per epoch). Well inside 3 hours; the margin is for the samgaish arm
# at batch 512.
set -uo pipefail

ROOT="/project/peilab/why/eeg-retrieval"
cd "${ROOT}"
mkdir -p outputs/slurm outputs/logs

export PY="${PY:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export SUBJ="${SUBJ:-8}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"

ARMS=(base_ema ch17 mmd samgaish)
i="${SLURM_ARRAY_TASK_ID:-0}"
export ARM="${ARMS[$i]}"

echo "########## loc ${SLURM_JOB_ID:-local}.${i} arm=${ARM} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds) ##########"

bash scripts/run_epd_localize.sh
RC=$?

echo "########## loc ${SLURM_JOB_ID:-local}.${i} arm=${ARM} rc=${RC} @ $(date -Iseconds) ##########"
exit "${RC}"
