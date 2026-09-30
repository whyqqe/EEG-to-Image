#!/usr/bin/env bash
#SBATCH --job-name=tdm_intra
#SBATCH --account=peilab
# `normal`, not `preempt`: this is one chain that must run to the end.
#SBATCH --partition=normal
# NO NODE PIN.  dgx-09's driver is older than this venv's torch, so the run script
# guards CUDA availability loudly instead of pinning; the previous pinned OCF
# submission sat in PENDING for hours because dgx-10/dgx-38 were fully occupied.
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/tdm_intra-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/tdm_intra-%j.err
#
# TDM-DT sub-08 overnight chain.  One job, because the stages are strictly
# sequential (encoder -> CSM export -> raw cache -> iREPA targets -> both training
# arms -> VAE head -> 8 generation rows -> 8 evaluations) and a failure in any
# stage must stop the ones after it rather than produce a half-built comparison.
set -euo pipefail
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE="${DEVICE:-cuda:0}"
export EPOCHS="${EPOCHS:-26}"
export OUT="${OUT:-${NB_ROOT}/outputs/tdm_intra}"
export PYTHONUNBUFFERED=1
bash "${NB_ROOT}/scripts/nda/run_tdm_intra.sh"