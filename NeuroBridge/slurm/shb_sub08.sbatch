#!/usr/bin/env bash
#SBATCH --job-name=shb-sub08
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/shb-sub08-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/shb-sub08-%j.err

set -euo pipefail
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
# structural channel policy for the SHB rows
export CN=0.40
export STRENGTH=0.86
bash "${NB_ROOT}/scripts/nda/run_shb_sub08.sh"
