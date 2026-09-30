#!/usr/bin/env bash
#SBATCH --job-name=epd-probe
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-probe-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-probe-%j.err
#
# The structural-branch reachability probe. See scripts/run_epd_struct_probe.sh for
# the full argument; this file only supplies the allocation.
#
# Why 96G and not 64G: the largest ridge input here is the 48x768 dense token grid,
# and the probe's thin SVD holds both the 6016 x 36864 fit matrix and its right
# singular vectors in memory at once (~1.8 GB together in float32), plus the raw-EEG
# fit matrix for the two reference rows. 64G would very likely fit; 96G means a miss
# on that estimate costs a requeue rather than a silent OOM three rows in.
#
# 2 hours: the extraction is ~2 min of GPU, and the nine probes are dominated by
# nine SVDs whose cost is O(n^2 * D) at n = 6016 -- a few minutes each, with the
# 36864-dim row the slowest. The wall clock is not the binding constraint; the
# estimate is deliberately loose because a probe that dies at row 7 of 9 is worth
# nothing.
set -uo pipefail

ROOT="/project/peilab/why/eeg-retrieval"
cd "${ROOT}"
mkdir -p outputs/slurm outputs/logs

export PY="${PY:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export SUBJ="${SUBJ:-8}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"

# Default to the arm whose checkpoint is already on disk. `dino3` is the arm the
# pipeline's own narrative treats as the primary one, and all three arms' trunks are
# the same architecture, so probing one is the controlled experiment; swapping CKPT
# for `epd_dual_mae_best.pt` or `epd_dual_in21k_best.pt` is a one-variable diff if
# the first result leaves the backbone family still in question.
export CKPT="${CKPT:-${ROOT}/outputs/sub08/epd_dual_dino3_best.pt}"

echo "########## epd-probe ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds) ##########"
echo "ckpt: ${CKPT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

bash scripts/run_epd_struct_probe.sh
RC=$?

echo "########## epd-probe ${SLURM_JOB_ID:-local} rc=${RC} @ $(date -Iseconds) ##########"
exit "${RC}"
