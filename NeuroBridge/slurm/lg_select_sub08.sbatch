#!/usr/bin/env bash
#SBATCH --job-name=lg-select-sub08
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/lg-select-sub08-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/lg-select-sub08-%j.err

set -euo pipefail
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
bash "${NB_ROOT}/scripts/nda/run_lg_select_sub08.sh"
