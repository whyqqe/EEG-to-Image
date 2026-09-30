#!/usr/bin/env bash
#SBATCH --job-name=epd-pipe
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-pipe-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-pipe-%j.err
#
# =============================================================================
# epd-pipe -- the baseline-to-our-arm pipeline, sub-08 held out
# =============================================================================
# WHAT each arm does and WHY every setting is what it is lives in
# scripts/run_epd_loso_pipeline.sh's header. This file is the resource envelope and the
# scheduling decision, nothing else.
#
# Time: 24h for TWO arms, so ~12h each
# ------------------------------------
# Each arm is a full 50-epoch LOSO training on 9 x 1654 x 10 = 148,860 rows plus the
# CPU recovery ladder (seconds at 200 queries, so not the constraint). The arms are
# run SEQUENTIALLY rather than as two jobs for one reason: they share the MVNN caches
# and the feature memmaps, and two jobs on one node would double the resident set for
# no throughput gain -- the run is compute-bound, not IO-bound. The cost is that the
# second arm inherits any overrun from the first, which is why the per-arm idempotency
# guard in the runner matters: if the wall clock cuts arm 2, resubmitting continues
# from arm 1's finished json instead of retraining it.
#
# If the smoke says an arm needs more than ~12h, cut ARMS to one arm per submission
# (`ARMS="a0-pairwise" sbatch ...`) rather than raising the limit: 24h is the
# partition's ceiling and a job killed at hour 24 loses the arm in flight.
#
# Memory: 192G, unchanged from epd_loso.sbatch
# --------------------------------------------
# `load_loso` materialises nine subjects' standardised arrays and concatenates them:
# 9 x 1654 x 10 x 63 x 250 float32 = 9.4 GiB, and `np.concatenate` holds both the
# blocks and the destination at the peak -- ~19 GiB before the model, the optimiser
# state or the CUDA context. The multi-layer target adds five 1654x10x1024 float32
# arrays: 340 MB, so it does not move this number. 192G is headroom, because the
# failure mode of getting it wrong is an OOM kill in a multi-hour job.
#
# But the recovery ladder does add something to watch: it materialises the full
# 200x200 similarity matrix and a 1024x1024 SVD, which is trivial, and it needs the
# FEATURES retained for the whole test set rather than streamed. That is 200x1024
# floats -- also trivial. The reason `evaluate` takes `return_features` instead of
# re-encoding is to avoid a second pass, not to save memory.
#
# CPUs: 8
# ------
# The MVNN whiteners are pre-built (asserted by the dry run), so the BLAS-bound
# preprocessing does not run here; but the EEG tokenizer's bilinear resample inside the
# DataLoader workers is thread-hungry, and thread counts set explicitly beat whatever
# the scheduler picks.
#
# PRECONDITIONS, all checked by `run_epd_loso_pipeline.sh --dry-run`
# ----------------------------------------------------------------
#   * 20 MVNN whiteners in outputs/cache (scripts/epd/build_mvnn_cache.py) -- 9 sources
#     fitted on `train`, the held-out subject fitted on `test`
#   * outputs/features/clip_h14_layers/{train,test}/block{22,24,26,28,30}.npy
#   * sub-01..sub-10 standardised caches (load_subject_std)
set -euo pipefail

ROOT="/project/peilab/why/eeg-retrieval"
cd "${ROOT}"

# A compute node has no outbound network, and a failed download reads as a confusing
# error rather than a clear one, so the model cache is asserted instead.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

mkdir -p outputs/slurm outputs/loso

echo "########## epd-pipe ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
free -g | head -2
echo "########## ARMS=${ARMS:-<default: a0-pairwise a1-multipos>}"

# The same script is run locally for --dry-run and --smoke, so those are evidence about
# what this job will do rather than a separate code path that can drift. A real
# submission deliberately passes neither.
bash "${ROOT}/scripts/run_epd_loso_pipeline.sh"
RC=$?

echo "########## epd-pipe ${SLURM_JOB_ID:-local} rc=${RC} @ $(date -Iseconds)"
exit "${RC}"
