#!/bin/bash
#SBATCH --job-name=chab_s08
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/chab_s08_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/chab_s08_%j.err
#
# CHAB sub-08 -- 17 vs 63 electrodes at the encoder input.
#
# TIME BUDGET (measured, not guessed)
#   * correctness audit                      ~1 min   (CPU, one batch)
#   * warm0 encoder + real exporter          ~5 min   (dataset load dominates)
#   * encoder per arm, 30+15 epochs          ~22 min  (prior intra runs; 10 subjects
#                                                      took ~3.7 h in cfmsf_all)
#   * route probe per arm, 13 routes x 80ep  ~34 min  (job 582254)
#   => 1 + 5 + 2*(22 + 3 + 34) = ~124 min.  The 8 h wall leaves room for a slower
#      node and for one arm to be retried after a kill without losing the rest,
#      because every stage skips on an existing artifact.

set -euo pipefail

export NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"

# Cache dirs live on the project filesystem: the home quota is small and an
# earlier job died with `OSError: [Errno 28] No space left on device` while
# writing `~/.cache/scikit-image` mid-evaluation.
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
mkdir -p "${NB_ROOT}/outputs/slurm"

bash scripts/nda/run_chab_sub08.sh
