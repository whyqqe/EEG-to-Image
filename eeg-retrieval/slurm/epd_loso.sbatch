#!/usr/bin/env bash
#SBATCH --job-name=epd-loso
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-loso-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-loso-%j.err
#
# =============================================================================
# epd-loso -- the inter-subject fold (sub-08 held out), full protocol
# =============================================================================
# Everything about WHAT this run does and WHY each setting is what it is lives in
# scripts/run_epd_loso.sh's header. This file is only the resource envelope, and the
# numbers in it are worth justifying because they are the difference between a run
# that finishes and one that dies at epoch 40 with nothing.
#
# Memory: 192G, not the pipeline's usual 96G
# ------------------------------------------
# `load_loso` materialises nine subjects' standardised training arrays and then
# concatenates them: 9 x 1654 x 10 x 63 x 250 float32 = 9.4 GiB, and `np.concatenate`
# holds both the nine blocks AND the destination at the peak -- so ~19 GiB before the
# model, the optimiser state or the CUDA context. The intra-subject runs never needed
# more than ~10 GiB here. 192G is headroom rather than a tight fit, because the
# failure mode of getting this wrong is an OOM kill inside a multi-hour job.
#
# CPUs: 8
# ------
# The MVNN whitener is fitted per subject from a 4 GiB raw file and the Gram-style
# matrices involved are BLAS-bound, so the thread count matters. It is set explicitly
# rather than left to the scheduler because a single-threaded fit takes ~10 minutes per
# subject and there are ten.
#
# Time: 24h
# -------------
# The fit set is 9 x 1654 x 10 = 148,860 rows against the intra-subject run's 16,540,
# so an epoch is ~9x longer in samples. At batch 256 that is 582 steps/epoch. 50 epochs
# is the protocol (SCORE: "train each model for 50 epochs, and report the final
# epoch"). The estimate is dominated by the EEG tokenizer's bilinear resample, which is
# what the smoke run is meant to measure -- if the smoke says 30 min/epoch, this limit
# is wrong and the run will be cut off, so the smoke is run first for exactly that.
#
# Preconditions, checked by run_epd_loso.sh --dry-run
# -------------------------------------------------
#   * 20 MVNN whiteners in outputs/cache (scripts/epd/build_mvnn_cache.py)
#   * outputs/features/clip_h14_layers/{train,test}/block26.npy
#   * sub-01..sub-10 standardised caches (load_subject_std)
set -euo pipefail

ROOT="/project/peilab/why/eeg-retrieval"
cd "${ROOT}"

# A compute node has no outbound network and a failed download is a confusing error
# rather than a clear one, so the model cache is asserted instead.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

mkdir -p outputs/slurm outputs/loso

echo "########## epd-loso ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
free -g | head -2

# The runner is the same script used locally, so a local dry run is evidence about
# what this job will do. `--dry-run` is deliberately NOT passed here.
bash "${ROOT}/scripts/run_epd_loso.sh"
RC=$?

echo "########## epd-loso ${SLURM_JOB_ID:-local} rc=${RC} @ $(date -Iseconds)"
exit "${RC}"
