#!/usr/bin/env bash
#SBATCH --job-name=samga-inter
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/samga-inter-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/samga-inter-%j.err
#
# =============================================================================
# samga-inter -- the OFFICIAL baseline, sub-08 held out
# =============================================================================
# `scripts/run_samga_official_baseline.sh` is the runner and its header carries every
# argument-level decision (which flags come verbatim from inter.sh, the two forced
# substitutions, the test-set model selection that inflates their reported number).
# This file is the resource envelope only.
#
# Memory: 64G, set from a measurement rather than from the array shapes
# ----------------------------------------------------------------------
# The obvious estimate is wrong and it errs high by a lot, so it is worth writing down.
# SAMGA's dataset does `np.load` of the raw train split -- (1654, 10, 4, 63, 250)
# float32, 4.2 GiB per subject -- and it LOOKS like nine of those must coexist at
# 37 GiB. They do not. `--data_average` puts `eeg_data = np.mean(eeg_data, axis=2)`
# immediately after the load, which rebinds the name and frees the raw block, so only
# the 1.04 GiB averaged array survives per subject:
#
#     held after all nine                          9 x 1.04  = 9.4 GiB
#     transient while the ninth is being averaged  + 4.2 + 1.04 = 5.2 GiB
#     image features + torch/CUDA context                   ~  1.5 GiB
#                                                           ----------
#                                                           ~16 GiB peak
#
# `num_workers=0` in their DataLoader means there is no per-worker copy to add.
#
# The smoke job measured 5.66 GiB MaxRSS with two subjects, which is the same picture
# from the other end: 2 x 1.04 held plus the 5.2 GiB averaging transient and overhead.
# 64G is ~4x that measurement. The earlier 192G request on this job was inherited from
# `epd_loso_pipeline.sbatch`, where it IS earned -- our own LOSO loader keeps the
# subject-major concatenation and does not have this early-free. Over-requesting memory
# is not free on a shared cluster: it is a scheduling constraint, and there is no reason
# to hold a queue slot for allocations the process provably never makes.
#
# CPU: 8
# -----
# The augmentation is a Python loop, not a batched op: `RandomSmooth` is invoked once
# per (object, image) pair per subject, so 9 x 1654 x 10 = 148,860 calls happen before
# the first optimiser step. That is the dominant preprocessing cost in this job and it
# is single-threaded in their code, so the thread count here buys less than it looks --
# but the array copies around it are BLAS-backed and do respond to it.
#
# Time: 12h
# -------
# 50 epochs at batch 1024 over 148,860 rows = 146 steps/epoch, plus a 200-query test
# evaluation every epoch. The encoder is TSConv, which is small next to our own arms'
# ViT front-end, so the training itself is under an hour by any reasonable estimate.
#
# The measured cost that is NOT the training: `--frozen_eeg_prior` runs the
# augmentation as a Python double loop over (object, image) -- 1654 x 10 = 16,540
# `RandomSmooth` calls PER SUBJECT -- and the smoke job measured it at 323 s/subject.
# Nine sources is therefore ~48 min of loading before the first optimiser step, and the
# test split is loaded a second time on top of that. Twelve hours is generous against
# ~1.5h of work, and deliberately so: the binding constraint here is not throughput, it
# is that a job cut off at the wall clock loses everything, and this fold has to be
# rerun from scratch if it is.
#
# It is also the limit that matters for a different reason: `--early_stop_patience 10`
# means a degenerating run stops itself long before the wall clock would, so a run that
# has NOT stopped by hour 12 is the interesting signal.
#
# PRECONDITIONS, checked by the runner's --dry-run and by its own assertions
# ------------------------------------------------------------------------
#   * data/preprocessed_eeg/{info.json,sub-XX/{train,test}.npy} for sub-01..sub-08,
#     sub-09, sub-10 -- the SAMGA-format arrays (with the 4-repetition axis)
#   * outputs/features/clip_h14_layers/{train,test}/block{22,24,26,28,30}.npy -- five
#     CLIP ViT-H-14 layers standing in for InternViT's 20/24/28/32/36 (see the runner
#     header; the absence of InternViT is why the absolute number here is not the
#     paper's)
set -euo pipefail

ROOT="/project/peilab/why/eeg-retrieval"
cd "${ROOT}"

# No network on a compute node, and a failed download reads as a confusing error rather
# than a clear one.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

mkdir -p outputs/slurm outputs/samga_official

echo "########## samga-inter ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
free -g | head -2

bash "${ROOT}/scripts/run_samga_official_baseline.sh"
RC=$?

echo "########## samga-inter ${SLURM_JOB_ID:-local} rc=${RC} @ $(date -Iseconds)"
exit "${RC}"
