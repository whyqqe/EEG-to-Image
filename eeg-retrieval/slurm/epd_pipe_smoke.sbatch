#!/usr/bin/env bash
#SBATCH --job-name=epd-pipe-smoke
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-pipe-smoke-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-pipe-smoke-%j.err
#
# =============================================================================
# epd-pipe-smoke -- does the pipeline RUN, on real data, in minutes
# =============================================================================
# `--dry-run` only validates flags, and every bug this pipeline has had so far was found
# at run time rather than parse time. Two were caught by the previous smoke:
#
#   * `from .data import ...` in train.py, which dies with "attempted relative import
#     with no known parent package" because train.py is executed as a script.
#   * the LOSO `TrainDataset` concept mismatch, where `tr_eeg`'s first axis is the
#     subject-major (subject, concept) rows but the dataset was handed concept-space
#     indices -- which tripped the dataset's own length guard and killed every LOSO run
#     before the first step.
#
# The changes since then touch exactly the same class of thing, so they need the same
# treatment rather than a flag check:
#
#   * `TrainDataset.__getitem__` now appends a spectrum of metadata fields and the loop
#     slices the structural targets with `batch[3 + n_meta:]`. An arity error there puts
#     a VAE latent where the subject id is read -- no exception, wrong tensor. With no
#     struct tower in this arm the slice is `batch[4:]`, so the smoke exercises the
#     boundary that matters.
#   * `--multipos` reads its grouping out of the batch and the loss mask divides by the
#     group sizes. A mask bug produces NaN, not a crash.
#   * `--recovery` runs a 1024x1024 SVD and a 200x200 mutual-NN pass on features that
#     were retained through `evaluate(return_features=True)`. It only runs after the
#     whole training loop has completed, so nothing upstream of it proves it works.
#
# `--limit-samples 512` and 1 stage-1 epoch keep it to a few minutes. What it CANNOT
# tell you is whether the objective helps, or whether an arm fits in 12 hours -- the
# second is what the real submission's wall clock is exposed to.
#
# Resource envelope is deliberately small: this job's job is to fail fast, not to
# produce a number.
set -euo pipefail

ROOT="/project/peilab/why/eeg-retrieval"
cd "${ROOT}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

mkdir -p outputs/slurm outputs/loso

echo "########## smoke ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

bash "${ROOT}/scripts/run_epd_loso_pipeline.sh" --smoke
RC=$?

echo "########## smoke ${SLURM_JOB_ID:-local} rc=${RC} @ $(date -Iseconds)"
exit "${RC}"
