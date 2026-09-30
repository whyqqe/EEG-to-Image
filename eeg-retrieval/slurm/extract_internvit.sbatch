#!/usr/bin/env bash
#SBATCH --job-name=internvit-feat
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/internvit-feat-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/internvit-feat-%j.err
#
# =============================================================================
# internvit-feat -- extract SAMGA's image features so the baseline stops being
#                   about a different visual backbone
# =============================================================================
# `run_samga_official_baseline.sh` runs SAMGA's unmodified train.py, so the only
# deviation left in the "are we reproducing it" question was the image features:
# InternViT-6B-448px-V2.5 at layers {20,24,28,32,36} versus the CLIP ViT-H-14 blocks
# every recorded run used. This job produces the former.
#
# SMOKE=1 runs 40 training concepts (400 images) and exercises both hard gates; the
# full run is 16,740 images. Run the smoke first: a hook on the wrong module or an
# off-by-one in the layer index produces correctly-shaped arrays that carry the wrong
# information, which is exactly the failure mode nobody notices until the numbers
# come back wrong weeks later.
#
# Environment: HF_HOME is pinned to the project cache, deliberately. The default
# `~/.cache/huggingface` on this cluster is full, so an unset HF_HOME fails with
# "No space left on device" -- which reads like a disk fault rather than a
# configuration one. The pipeline pins the same path in every other run script.
# HF_HUB_OFFLINE=1 because the 11.1 GB of weights must already be in that cache;
# login-node fetch, compute-node use.
#
# Memory: 64G. The model is 12 GB in bf16, of which the 45 encoder layers at 3200x12800
# dominate; with `--batch 32` the largest activation is the naive-attention score
# matrix (32 x 25 heads x 1025^2 x 2 B = 1.7 GiB) since flash_attn is absent from this
# environment and `modeling_intern_vit.py` falls back to `_naive_attn`. That fallback
# costs ~8% of FLOPs (attention is 13.4 GFLOP/layer/image against the MLP's 168), so
# it is not worth building flash_attn to avoid.
#
# Time: 02:00:00. ~8.2 TFLOP/image x 16,740 images = 137 PFLOP, i.e. roughly 15-25
# minutes on one H800 (80 GiB) once JPEG decode overlaps with the GPU. The limit is
# generous because a requeue after 100 minutes of extraction would cost more than the
# slot is worth.
#
# Usage:  sbatch slurm/extract_internvit.sbatch                         # full, both splits
#         SMOKE=1 sbatch slurm/extract_internvit.sbatch                 # 400 images
#         LAYER_INDEX_BASE=0 POOL=mean sbatch slurm/extract_internvit.sbatch   # convention probe
set -euo pipefail

ROOT="/project/peilab/why/eeg-retrieval"
cd "${ROOT}"

export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

SMOKE="${SMOKE:-0}"
LAYER_INDEX_BASE="${LAYER_INDEX_BASE:-1}"
POOL="${POOL:-cls}"
OUT="${OUT:-${ROOT}/data/image_feature/internvit_multilevel_20_24_28_32_36}"

mkdir -p outputs/slurm outputs/internvit_extract

echo "########## internvit-feat ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

args=(--split both --batch 32 --workers "${SLURM_CPUS_PER_TASK:-8}" --out "${OUT}"
      --layer-index-base "${LAYER_INDEX_BASE}" --pool "${POOL}")
if [[ "${SMOKE}" == "1" ]]; then
  # Write somewhere else. The smoke produces arrays named exactly like the real ones
  # but holding only 40 concepts; leaving them in the canonical feature directory
  # would let SAMGA's train.py load a truncated `image_train_layer20.npy` and train
  # happily on it. A smoke artifact that is indistinguishable from a real one by
  # filename is a trap, so the path itself carries the distinction.
  OUT="${ROOT}/outputs/internvit_extract/smoke${SMOKE_LABEL:-}/"
  args=(--split train --limit 40 --batch 16 --workers 4 --out "${OUT}"
        --layer-index-base "${LAYER_INDEX_BASE}" --pool "${POOL}")
  echo "### SMOKE: 40 concepts / 400 images -> ${OUT}"
fi
echo "### args: ${args[*]}"

python scripts/epd/extract_internvit_layers.py "${args[@]}" 2>&1 | tee "outputs/internvit_extract/extract-${SLURM_JOB_ID:-local}.log"
RC=${PIPESTATUS[0]}

echo "########## internvit-feat ${SLURM_JOB_ID:-local} rc=${RC} @ $(date -Iseconds)"
exit "${RC}"
