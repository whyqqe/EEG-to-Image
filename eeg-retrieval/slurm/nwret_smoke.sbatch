#!/usr/bin/env bash
# One-shot smoke test: verifies the whole path (data -> tokenizer -> backbone ->
# fusion -> retrieval metric) on a GPU with 2 epochs and 20 val concepts.
# Meant to be run to catch shape/bug regressions before spending an allocation
# on a full arm.
#SBATCH --job-name=nwret-smoke
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/nwret-smoke-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/nwret-smoke-%j.err
set -uo pipefail

ROOT="/project/peilab/why/eeg-retrieval"
cd "${ROOT}"
export PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
export HOME="${XDG_CACHE_HOME}"
mkdir -p outputs/slurm

set -x
"${PY}" -u scripts/nwret/train.py \
  --subject 8 --channels occipito_parietal \
  --backbone timm:vit_b16_in21k --layers 12 \
  --epochs 2 --val-concepts 20 --batch-size 256 \
  --out-dir outputs/smoke --tag smoke --smoke
RC=$?
set +x
echo "[smoke] rc=${RC}"
exit "${RC}"
