#!/bin/bash
#SBATCH --job-name=nw5b_struct
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --time=03:00:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/nw5b_struct_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/nw5b_struct_%j.err
#
# NW5b sub-08 -- the ONE question nw5_s08 leaves open:
#
#   A2 (SEM + GT depth/edge) scored inception 0.8616 / clip 0.9091.
#   A8 (SEM + our EEG depth) scored inception 0.7490 / clip 0.8245, i.e. +0.000 over
#   A1 (SEM alone).  The structure branch is worth +0.113 incep when it is fed the
#   ground truth and exactly nothing when it is fed ours.
#
# The probe (nw5_struct_probe.py) explained A8: our shipped depth/edge banks score
# BELOW a constant baseline on the GT banks (depth 0.476 vs 0.748; edge 0.127 vs 0.722),
# and the concentration calibration had made them worse still.  But it also showed
# something the probe CANNOT resolve: the GT depth/edge banks are so concentrated that a
# single fixed vector already achieves 0.748 / 0.722 against them, while the best
# leak-free linear EEG->structure map reaches only 0.782 / 0.761.
#
# So: is the structural condition being read as PER-TRIAL LAYOUT, or as a PRIOR?
# Three arms, identical to A1/A2 except for what the two structural branches receive:
#   C1_const  the train-bank centroid (one vector, zero trial information)
#   C2_ridge  leak-free ridge EEG->structure (top1 3x/11x chance, raw uncalibrated)
#   C3_shuf   real GT rows, wrong trials (matched distribution, broken alignment)
# Cost: 3 arms x 200 images, turbo 15 steps.  See run_nw5b_struct.sh for the reading key.
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

echo "########## NW5b structure ablation @ $(date -Iseconds) ##########"
"${PYTHON}" - <<'PY' || { echo "[FATAL] no usable CUDA device"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("[FATAL] CUDA unavailable"); sys.exit(1)
print(f"[gpu] {torch.cuda.get_device_name(0)} torch={torch.__version__} cuda={torch.version.cuda}")
PY

bash scripts/nda/run_nw5b_struct.sh
RC=$?

echo
echo "########## NW5b done @ $(date -Iseconds) rc=${RC} ##########"
echo "full log:  outputs/nw5_s08/logs/"
for a in C1_const C2_ridge C3_shuf; do
  f="outputs/nw5_s08/arms/${a}/eval/sub-08.json"
  [[ -f "${f}" ]] && echo "  ${a}: $(cat "${f}" | tr -d '\n' | cut -c1-160)"
done
exit "${RC}"
