#!/usr/bin/env bash
#SBATCH --job-name=inter-ll-full10
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/inter-ll-full10-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/inter-ll-full10-%j.err

set -euo pipefail
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
bash "${NB_ROOT}/scripts/nda/run_inter_ll_full10.sh"
