#!/bin/bash
#SBATCH --job-name=nw5c_varest
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --time=03:00:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/nw5c_varest_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/nw5c_varest_%j.err
#
# NW5c sub-08 -- our structural condition is SHRUNK by regression-to-the-mean.
#
# nw5b showed the structure branch carries TRIAL information: a constant centroid (C1),
# misaligned GT rows (C3) and our leak-free ridge (C2) all land AT OR BELOW the
# semantic-only floor, while GT structure adds +0.128 inception.
#
# It also showed why our attempts fail: trial-to-trial variability, as mean pairwise
# cosine across the 200 trials (near 1.0 == the bank is one vector):
#     GT depth/edge 0.566/0.530   a_hi SEM 0.373 (healthy, target 0.391)
#     ridge 0.955/0.935          UCK->CLIP 0.874/0.938
# Our semantic head is correctly scaled; the structure banks vary 6x less than the thing
# they predict, so a shared vector ate the energy and the trial-specific component -- the
# entire value of the branch -- got crushed.  Least squares does exactly this: predicting
# the mean minimises squared error.
#
# Rescaling only the deviation from the bank mean, with identical ridge weights and
# identical test EEG, lifts retrievability from 3x/11x chance to 26x/35x.  This job feeds
# the restored banks to the generator.  See scripts/nda/run_nw5c_varest.sh for the arms.
set -uo pipefail
cd /project/peilab/why/NeuroBridge

export PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
export HOME="${XDG_CACHE_HOME}"

mkdir -p outputs/slurm
echo "########## NW5c variance restoration @ $(date -Iseconds) ##########"
"${PYTHON}" - <<'PY' || { echo "[FATAL] no usable CUDA device"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("[FATAL] CUDA unavailable"); sys.exit(1)
print(f"[gpu] {torch.cuda.get_device_name(0)} torch={torch.__version__} cuda={torch.version.cuda}")
PY

bash scripts/nda/run_nw5c_varest.sh
RC=$?
echo
echo "########## NW5c done @ $(date -Iseconds) rc=${RC} ##########"
exit "${RC}"
