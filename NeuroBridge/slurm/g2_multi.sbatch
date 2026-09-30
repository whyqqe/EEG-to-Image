#!/usr/bin/env bash
#SBATCH --job-name=g2-multi
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/g2-multi-%A_%a.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/g2-multi-%A_%a.err

# One fold per array task: leave-one-subject-out training + generation.
# sub-08 is excluded from the array because the first G2 job (566625) already covers
# it; including it here would duplicate GPU-hours.
#
# Partition is `normal` rather than `preempt` on purpose: g2_train.py now resumes from
# <out>/last.pth, but a preemption still costs the current epoch, and `normal` is not
# evictable. The generation step is separately resumable (it skips finished rows).
set -euo pipefail
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
export OUT="${OUT:-${NB_ROOT}/outputs/g2}"
export SUBJECT="${SLURM_ARRAY_TASK_ID}"
export EPOCHS="${EPOCHS:-40}"
export GEN_VARIANTS="${GEN_VARIANTS:-direct,cfm0}"

bash "${NB_ROOT}/scripts/nda/run_g2_multi.sh"
