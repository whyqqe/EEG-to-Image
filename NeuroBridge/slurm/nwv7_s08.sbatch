#!/bin/bash
#SBATCH --job-name=nwv7_s08
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --time=04:30:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/nwv7_s08_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/nwv7_s08_%j.err
#
# NW-v7 sub-08 -- break the IMAGE/SEMANTIC bottleneck.
#
# NW-v6 2x2 (W4/W5) showed structure is nearly saturated when image is GT
# (W4 incep 0.9557 recovers 82% of A9-A5), while image is the remaining gap.
# Encoder-side STH/SCM/fusion did not move deployable generation; self-cond via
# CLIP(pass-1) was measured and closed (pass-1 CLIP farther from GT than SEM).
#
# This job only rebuilds the image condition bank (cfmsf/uge identity transplant
# onto GT-mean+varest, blends, dedicated image-only head) and pairs it with the
# already-verified G3 / JOINT structure banks under the locked operator.
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
echo "########## NW-v7 sub-08 @ $(date -Iseconds) ##########"
"${PYTHON}" - <<'PY' || { echo "[FATAL] no CUDA"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("[FATAL] CUDA unavailable"); sys.exit(1)
print(f"[gpu] {torch.cuda.get_device_name(0)} torch={torch.__version__}")
PY

bash scripts/nda/run_nwv7_s08.sh
RC=$?
echo
echo "########## NW-v7 sub-08 done @ $(date -Iseconds) rc=${RC} ##########"
exit "${RC}"
