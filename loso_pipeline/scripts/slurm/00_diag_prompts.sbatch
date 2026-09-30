#!/usr/bin/env bash
#SBATCH --job-name=loso-diag
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/project/peilab/why/third_party/loso_pipeline/logs/%x-%j.out
#SBATCH --error=/project/peilab/why/third_party/loso_pipeline/logs/%x-%j.err

# Measure the BLIP2 empty-completion rate per candidate prompt on a small sample.
# Cheap evidence before spending ~6 min of GPU per full captioning pass.
set -uo pipefail
module load slurm nvhpc-hpcx-cuda12/23.11
HERE=/project/peilab/why/third_party/loso_pipeline
# shellcheck disable=SC1091
source "${HERE}/env.sh"
exec "${LOSO_VENV}/bin/python" "${HERE}/scripts/diag_prompts.py" --n "${DIAG_N:-800}"
