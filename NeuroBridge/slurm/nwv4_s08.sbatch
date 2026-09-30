#!/bin/bash
#SBATCH --job-name=nwv4_s08
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --time=04:00:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/nwv4_s08_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/nwv4_s08_%j.err
#
# NW-v4 sub-08 -- the config the evidence points at, run end to end with the SOTA bars.
#
# Operator (nw5): CogCapPro's cc3 asymmetric IP-Adapter layout, empty prompt, SDXL-Turbo
# 15 steps CFG 0, NO ControlNet, plus our L3 low-frequency band anchor.  On sub-08 that
# combination (A5) is Pareto-dominant over everything we have shipped: pixcorr 0.1980
# ssim 0.3756 incep 0.8400 clip 0.9123 vs a_hi 0.1684/0.3573/0.7106/0.8115.  Dropping
# ControlNet while keeping the anchor is +0.095 SSIM at zero semantic cost.
#
# Its one non-deployable ingredient is GT depth/edge.  nw5b proved the structure branch
# needs TRIAL-ALIGNED information (a constant centroid, misaligned GT rows, and our
# linear EEG->structure map all score at or below the semantic-only floor).  nw5c found
# why our real paths still failed: they are SHRUNK.  Trial-to-trial variation, as mean
# pairwise cosine (near 1.0 == one vector): GT 0.566/0.530, our semantic head 0.373
# (healthy), our structure banks 0.955/0.874/0.938.  Least squares buys its error
# reduction by predicting the mean, so the trial-specific component -- the entire value
# of the branch -- was crushed.  Rescaling only the deviation from the bank mean, with
# identical weights and EEG, lifts retrieval from 3x/11x to 26x/35x chance.
#
# Arms: E1 floor, E2 varest ridge (the A5 point, deployable), E3 varest UCK (needs no new
# head), E4-E6 anchor sigma sweep (the only pixel lever), M6 multi-candidate selection
# over 4 seeds (the only lever that moves both axes at once).  See run_nwv4_s08.sh.
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
echo "########## NW-v4 sub-08 @ $(date -Iseconds) ##########"
"${PYTHON}" - <<'PY' || { echo "[FATAL] no usable CUDA device"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("[FATAL] CUDA unavailable"); sys.exit(1)
print(f"[gpu] {torch.cuda.get_device_name(0)} torch={torch.__version__} cuda={torch.version.cuda}")
PY

bash scripts/nda/run_nwv4_s08.sh
RC=$?
echo
echo "########## NW-v4 sub-08 done @ $(date -Iseconds) rc=${RC} ##########"
exit "${RC}"
