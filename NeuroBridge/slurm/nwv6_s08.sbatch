#!/bin/bash
#SBATCH --job-name=nwv6_s08
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --time=05:00:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/nwv6_s08_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/nwv6_s08_%j.err
#
# NW-v6 sub-08 -- the CogCapPro-style ENCODER-SIDE upgrade, training through evaluation.
#
# Diagnosis this responds to: our generator is not the bottleneck.  A5 (our operator, our
# semantic bank, GROUND-TRUTH depth/edge) reaches incep 0.8400 / clip 0.9123, at or above
# CogCapPro's published sub-08 (0.831 / 0.903), while the fully deployable pipeline reaches
# 0.7440 / 0.8525.  The whole interval is the EEG->structure alignment, which recovers only
# 16.5% of the oracle gain (5.5% with a linear ridge).
#
# CogCapPro's own upgrade over CogCap was encoder-side too -- same SDXL-Turbo + multi-branch
# IP-Adapter + empty prompt + CFG 0 generator, and its headline gains are retrieval numbers.
# We already have its two generator-side pieces (cc3 asymmetric injection, text dropped at
# inference) and none of its four encoder-side ones, so this job adds:
#
#   (1) STH-Align      one shared trunk + three projection heads, jointly trained
#   (2) SCM-Loss       multi-positive InfoNCE, since 1654 concepts x 10 images is
#                      one-to-many and hard-positive InfoNCE mislabels same-concept pairs
#   (3) Fusion         feed shared_r (+) specific_s (2048-d)
#   (4) Uncertainty    learned per-modality log-variance instead of a hand-tuned balance
#
# The arms include the two 2x2 cells we never had (GT image + our structure; our image + GT
# structure), which say definitively which branch owns the remaining gap.  See
# scripts/nda/run_nwv6_s08.sh for the per-arm rationale and the numbers behind each choice.
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
echo "########## NW-v6 sub-08 @ $(date -Iseconds) ##########"
"${PYTHON}" - <<'PY' || { echo "[FATAL] no usable CUDA device"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("[FATAL] CUDA unavailable"); sys.exit(1)
print(f"[gpu] {torch.cuda.get_device_name(0)} torch={torch.__version__} cuda={torch.version.cuda}")
PY

bash scripts/nda/run_nwv6_s08.sh
RC=$?
echo
echo "########## NW-v6 sub-08 done @ $(date -Iseconds) rc=${RC} ##########"
exit "${RC}"
