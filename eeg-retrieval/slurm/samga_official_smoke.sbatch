#!/usr/bin/env bash
#SBATCH --job-name=samga-smoke
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/samga-smoke-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/samga-smoke-%j.err
#
# =============================================================================
# samga-smoke -- does the OFFICIAL code run, before spending a queue slot on it
# =============================================================================
# Two subjects, one epoch. It is thirty seconds of training and the reason it exists is
# the thirty minutes in front of it: `frozen_eeg_prior` materialises the smoothed EEG
# for every subject at dataset-construction time, and the run does not reach its first
# optimiser step until all nine subjects are loaded. Without this job, a typo in the
# router wiring would be discovered on the far side of that load.
#
# What it exercises that nothing else in this project does: SAMGA's own
# `SubjectAwareLayerMixer` and `build_image_teacher`. Our pipeline reimplements the
# routed target and has never executed theirs, so the layer mixing, the subject
# conditioning, the layer dropout renormalisation and the `image_mid_dim` projection
# chain are all first-contact code.
#
# It does NOT verify that the full fold fits in memory or that 50 epochs converge; the
# real job inherits only the parts of its setup that are subject-count-independent.
#
# Memory: 64G. Two subjects peak at 5.66 GiB measured, and the full nine are ~16 GiB;
# the derivation is in `samga_official_inter.sbatch`'s header, where the shapes are
# written out (the raw 4.2 GiB per subject is averaged and freed immediately by
# `--data_average`, which is what makes this much smaller than it looks). This job uses
# the real job's limit rather than a smaller one on purpose: a smoke that passes under
# a different memory limit than the real job is a smoke that can pass while the real
# job is OOM killed.
set -euo pipefail

ROOT="/project/peilab/why/eeg-retrieval"
cd "${ROOT}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

mkdir -p outputs/slurm outputs/samga_official

echo "########## samga-smoke ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

bash "${ROOT}/scripts/run_samga_official_baseline.sh" --smoke
RC=$?

echo "########## samga-smoke ${SLURM_JOB_ID:-local} rc=${RC} @ $(date -Iseconds)"
exit "${RC}"
