#!/bin/bash
#SBATCH --job-name=nw5_s08
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --time=05:00:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/nw5_s08_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/nw5_s08_%j.err
#
# NW5 sub-08 -- the optimized configuration, built from CogCapPro's public code.
#
# We have been generating with a pixel-constrained chain: ControlNet-depth plus an
# img2img init at strength 0.82-0.92.  Our own oracle experiment bounded that chain's
# ceiling from the inside at inception 0.7946 even with PERFECT semantic conditions.
#
# Reading XiaoZhangYES/CognitionCapturerPro/src/cogcappro/generate_image/generator.py
# shows the published SOTA does not constrain pixels at all: no ControlNet, no init, an
# empty prompt, 15 steps, guidance_scale=0.0, and three IP-Adapter branches -- image,
# depth and edge -- injected at exactly
#
#   image {"down":{"block_2":[1.0,1.0]}, "up":{"block_0":[1.0,1.0,1.0]}}
#   depth {"down":{"block_2":[0.0,0.5]}, "up":{"block_0":[0.0,0.0,0.0]}}
#   edge  {"down":{"block_2":[0.0,0.5]}, "up":{"block_0":[0.0,0.0,0.0]}}
#
# i.e. structure lives in FEATURE space, at half strength, on a single UNet level.
#
# So the ceiling we measured is a property of OUR operator, not of the data -- and the
# fix is not a decoder tweak but dropping the pixel constraint.  This job runs that.
# See scripts/nda/run_nw5_s08.sh for the per-arm hypotheses and the reading key.
set -uo pipefail
cd /project/peilab/why/NeuroBridge

export PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
# keeps transformers/open_clip from trying (and failing) to write to an unwritable $HOME
export HOME="${XDG_CACHE_HOME}"

mkdir -p outputs/slurm

echo "########## NW5 sub-08 @ $(date -Iseconds) ##########"
"${PYTHON}" - <<'PY' || { echo "[FATAL] no usable CUDA device"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("[FATAL] CUDA unavailable"); sys.exit(1)
print(f"[gpu] {torch.cuda.get_device_name(0)} torch={torch.__version__} cuda={torch.version.cuda}")
PY

bash scripts/nda/run_nw5_s08.sh
RC=$?

echo
echo "########## NW5 sub-08 done @ $(date -Iseconds) rc=${RC} ##########"
echo "full log:  outputs/nw5_s08/logs/"
echo "summary:   outputs/nw5_s08/NW5_S08_SUMMARY.json"
ls -la outputs/nw5_s08/ 2>/dev/null
exit "${RC}"
