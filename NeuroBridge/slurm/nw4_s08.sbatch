#!/bin/bash
#SBATCH --job-name=nw4_s08
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_s08_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_s08_%j.err
#
# TIME BUDGET (sub-08, single GPU)
#   S1 open-set trial-level, 30 ep, 3 heads    ~35-50 min
#   S2 closed-form projection                  ~5-8 min
#   cond audit                                 ~2 min
#   V1 gen 11 arms x 200 @ 28 steps            ~70-110 min
#   seven + cycle x 11 arms                    ~60-80 min
#   => ~3-4.5 h typical; 16 h wall for slow nodes / resume.

set -uo pipefail
cd /project/peilab/why/NeuroBridge
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE="${DEVICE:-cuda:0}"
export NW4_OUT="${NW4_OUT:-${NB_ROOT}/outputs/nw4/sub-08}"
export S1_EPOCHS="${S1_EPOCHS:-30}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/eeg-brainit/xdg}"

mkdir -p "${NB_ROOT}/outputs/slurm" "${NW4_OUT}/logs"
bash scripts/nda/run_nw4_sub08.sh 2>&1 | tee "${NW4_OUT}/logs/orchestrator_${SLURM_JOB_ID:-local}.log"
