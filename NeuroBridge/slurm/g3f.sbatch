#!/usr/bin/env bash
#SBATCH --job-name=g3f
#SBATCH --account=peilab
# `normal`, not `preempt`: one job that must run to the end; the preempt
# partition would interrupt it repeatedly and the earlier
# QOSMaxSubmitJobPerUserLimit came from submitting several jobs at once.
#SBATCH --partition=normal
# NODE PIN, not a preference, and passed as `-w` on the sbatch command line
# because a `--nodelist` directive makes this cluster's parser reject `--gpus`.
# The scheduler placed the first attempt on dgx-09, whose driver (12.8) is older
# than this venv's torch build (cu130), so torch silently fell back to CPU.
# Every job that ever succeeded for this project ran on dgx-10 or dgx-38
# (`sacct -X`), so those are the only nodes whose driver is confirmed new
# enough. Waiting for one costs minutes; a stale driver costs ~87 h of CPU
# training or an immediate abort.
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/g3f-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/g3f-%j.err

# Single job, full chain:
#   leak audit -> train intra + 10 LOSO folds -> generate controlled rows
#   -> ONE evaluation pass (standard-7 + pooled FID) over our rows AND the
#      existing reference rows, so every number comes from the same code on the
#      same GT cache.
set -euo pipefail
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
export EPOCHS="${EPOCHS:-26}"
export BATCH="${BATCH:-512}"
export OUT="${OUT:-${NB_ROOT}/outputs/g3f}"

bash "${NB_ROOT}/scripts/nda/run_g3f.sh"
