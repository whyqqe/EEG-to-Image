#!/usr/bin/env bash
#SBATCH --job-name=tdm_all
#SBATCH --account=peilab
# `normal`, not `preempt`: this is one chain that must run to the end.  A wall-time
# kill is recoverable anyway -- `run_tdm_all.sh` keeps a per-subject DONE sentinel
# and every stage has a `[SKIP]` guard, so a requeue resumes instead of restarting.
#SBATCH --partition=normal
# NO NODE PIN.  dgx-09's driver is older than this venv's torch, so the run script
# guards CUDA availability loudly instead; the previous pinned submission sat in
# PENDING for hours because dgx-10/dgx-38 were fully occupied.
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/tdm_all-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/tdm_all-%j.err

# ALL 10 SUBJECTS, intra-subject protocol.  Per subject:
#   intra encoder -> export shared_r -> raw EEG cache -> timing probes
#   -> TDM-DT full arm + ablation arm (same code, same rows)
#   -> VAE low-level head -> 3 generation rows (sub-08 carries 8)
#   -> evaluate -> delete that subject's scratch
set -euo pipefail
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE="${DEVICE:-cuda:0}"
export EPOCHS="${EPOCHS:-26}"
export OUT="${OUT:-${NB_ROOT}/outputs/tdm_all}"
export SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
export ABL_SUBJECT="${ABL_SUBJECT:-8}"
export KEEP_IMAGES="${KEEP_IMAGES:-0}"
export PYTHONUNBUFFERED=1
bash "${NB_ROOT}/scripts/nda/run_tdm_all.sh"
