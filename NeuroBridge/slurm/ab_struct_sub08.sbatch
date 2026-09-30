#!/usr/bin/env bash
# SBATCH --job-name=ab-struct-sub08
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/ab-struct-sub08-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/ab-struct-sub08-%j.err

# Part 1 (primary): Spectral Assembly -- per-band latent anchoring to break the
#                   measured `strength` Pareto frontier (structure vs semantics/FID).
# Part 2 (secondary): structure-specialised EEG encoder, gated on cross-sample spread.
#                   Needs one extra GPU pass over 16540 GT images for the depth cache;
#                   if that fails the script continues and only Part 2 is skipped.
set -euo pipefail
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
export SUB="${SUB:-sub-08}"
bash "${NB_ROOT}/scripts/nda/run_ab_struct_sub08.sh"
