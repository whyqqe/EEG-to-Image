#!/usr/bin/env bash
#SBATCH --job-name=ocf_intra
#SBATCH --account=peilab
# `normal`, not `preempt`: this is one chain that must run to the end.
#SBATCH --partition=normal
# NODE PIN as `-w` on the command line (see submit_ocf_intra.sh): a `--nodelist`
# directive makes this cluster's parser reject `--gpus`. dgx-09 has driver 12.8,
# older than this venv's torch, and torch then falls back to CPU silently --
# every successful job in this project ran on dgx-10 or dgx-38.
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/ocf_intra-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/ocf_intra-%j.err

# ONE job, sub-08 only:
#   leak audit -> pure intra encoder (sub-08 alone) -> export shared_r
#   -> OCF heads (spherical read-out, theta solved on train) -> VAE low-level head
#   -> 7 generation rows (theta ablation + FRLA arms) -> ONE evaluation pass
#      over ours AND the reference rows, same code, same GT cache.
set -euo pipefail
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
export EPOCHS="${EPOCHS:-26}"
export OUT="${OUT:-${NB_ROOT}/outputs/ocf_intra}"

bash "${NB_ROOT}/scripts/nda/run_ocf_intra.sh"
