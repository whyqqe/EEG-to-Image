#!/usr/bin/env bash
#SBATCH --job-name=epd-anchor
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-anchor-%A_%a.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-anchor-%A_%a.err
#
# STAGE 0 -- the anchor. Run SAMGA's own code, unmodified, on sub-08, here.
#
# The one question this answers: is 94.8 (or anything near it) reachable on this
# machine, this data and this protocol? We have compared ourselves to that number for
# weeks without ever reproducing it, and our local leaderboard has +-3.5 points of
# resolution, so it cannot tell us whether we are 43 points behind or whether the
# comparison is invalid. Two outcomes, both decisive:
#
#   ~90   -> the recipe is reachable here, so OUR recipe is the problem, and Stage 1's
#            per-axis arms become a directed search with a known-good endpoint.
#   ~55   -> the shortfall lives in the data or the protocol, and every "gap to SOTA"
#            figure this project has produced is unverifiable. That is more important
#            to know than any hyperparameter result.
#
# Arms: samga_rn50 (SAMGA's own default feature space), samga_clip5 (same recipe, our
# CLIP ViT-H/14 5-depth router -- the shape of the config SAMGA reports 94.8 with).
# See scripts/run_epd_anchor.sh for the full rationale, including the one deliberate
# deviation (--early_stop_patience 0, because SAMGA's early stopping reads the TEST
# set and disabling it only extends the trajectory without moving its maximum).
#
# Cost: SAMGA's EEGProject encoder is a small conv stack, not a ViT. 60 epochs at
# batch 512 over 16540 training rows plus a 200-sample test pass per epoch. This should
# be minutes, not hours -- which is why both arms can run at once.
set -uo pipefail

ROOT="/project/peilab/why/eeg-retrieval"
cd "${ROOT}"
mkdir -p outputs/slurm outputs/logs outputs/anchor

export PY="${PY:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export SUBJ="${SUBJ:-8}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"

ARMS=(samga_rn50 samga_clip5)
i="${SLURM_ARRAY_TASK_ID:-0}"
export ARM="${ARMS[$i]}"

echo "########## anchor ${SLURM_JOB_ID:-local}.${i} arm=${ARM} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds) ##########"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true

bash scripts/run_epd_anchor.sh
RC=$?

echo "########## anchor ${SLURM_JOB_ID:-local}.${i} arm=${ARM} rc=${RC} @ $(date -Iseconds) ##########"
exit "${RC}"
