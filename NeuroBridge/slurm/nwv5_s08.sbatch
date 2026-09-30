#!/bin/bash
#SBATCH --job-name=nwv5_s08
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --time=05:00:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/nwv5_s08_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/nwv5_s08_%j.err
#
# NW-v5 sub-08 -- the optimal architecture assembled from every measurement so far.
#
#   cc3 asymmetric IP-Adapter layout (CogCapPro's, mass-matched uniform loses 0.11 incep)
#   NO ControlNet; the L3 low-frequency band anchor is kept (+0.095 SSIM at zero
#       semantic cost versus dropping the pixel path entirely)
#   our a_hi semantic bank, plus the best structure bank we can actually deploy
#   SDXL-Turbo, 15 steps, CFG 0.0, empty prompt
#   M6 multi-candidate render-verify selection over a HETEROGENEOUS pool
#
# A5 -- the same operator with GROUND-TRUTH depth/edge -- is Pareto-dominant over
# everything this project has shipped: pixcorr 0.1980 ssim 0.3756 incep 0.8400
# clip 0.9123 versus a_hi 0.1684/0.3573/0.7106/0.8115.  The whole job is about closing
# that gap with ingredients that exist at test time.
#
# The three things this job decides:
#
#  1. Is a NONLINEAR structure head worth anything?  nwv5_struct_head.py trains an MLP on
#     the DEVIATION (GT row minus the GT TRAIN centroid) with an InfoNCE objective, and
#     fits a ridge on the same rows with the same validation as the control.  The linear
#     map is provably exhausted -- re-targeting it at the deviation leaves dev_corr at
#     0.3599/0.3485 to four decimals -- so this is the last mechanism that could recover
#     what A5 gets from real structure.  The head is NOT pre-judged on embedding metrics,
#     which have already misled us once; arms G3_head_s30/s50 measure it in pixels.
#
#  2. Does the anchor like being softer?  sigma 3.0 -> 0.5 worsens SSIM monotonically
#     (0.3706/0.3651/0.3600/0.2998).  Our anchor is a PREDICTION, so its high frequencies
#     are wrong and blurring them protects SSIM.  sigma 4/5/6/8 has never been run.
#
#  3. Does M6 selection want a pixel term?  beta > 0 has never been run and it is the one
#     selection criterion that pushes SSIM directly.  The score matrices are cached, so
#     the whole (alpha,beta) sweep costs evaluations and no generation.
#
# See scripts/nda/run_nwv5_s08.sh for the per-arm rationale and the numbers behind each
# choice.
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
echo "########## NW-v5 sub-08 @ $(date -Iseconds) ##########"
"${PYTHON}" - <<'PY' || { echo "[FATAL] no usable CUDA device"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("[FATAL] CUDA unavailable"); sys.exit(1)
print(f"[gpu] {torch.cuda.get_device_name(0)} torch={torch.__version__} cuda={torch.version.cuda}")
PY

bash scripts/nda/run_nwv5_s08.sh
RC=$?
echo
echo "########## NW-v5 sub-08 done @ $(date -Iseconds) rc=${RC} ##########"
exit "${RC}"
