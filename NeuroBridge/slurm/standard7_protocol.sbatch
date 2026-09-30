#!/usr/bin/env bash
#SBATCH --job-name=std7-protocol
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.err

# Standard-protocol 7 metrics + FID for HCMA and best sub-08 models.

set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
CACHE=/project/peilab/why/cache/eeg-brainit

cd "${NB_ROOT}"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh"
export PYTHONUNBUFFERED=1
export TORCH_HOME="${CACHE}/torch"
export HF_HOME="${CACHE}/hf"
export HF_HUB_CACHE="${CACHE}/hf/hub"
export TRANSFORMERS_CACHE="${CACHE}/hf/hub"
export OPENCLIP_CACHE_DIR="${CACHE}/open_clip"
export TMPDIR="${SLURM_TMPDIR:-${CACHE}/tmp}"
export OMP_NUM_THREADS=8

OUT="${NB_ROOT}/outputs/standard7_protocol"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
python -c "import skimage, torchmetrics, open_clip; print('deps OK')"

echo "===== [1/2] standard-7 + FID per row ====="
python "${NB_ROOT}/scripts/nda/eval_standard7.py" \
  --manifest "${OUT}/manifest.json" \
  --images-root /project/peilab/why/data/images_set \
  --out-dir "${OUT}" \
  --device cuda:0 \
  --batch-size 16

echo "===== [2/2] build table ====="
python "${NB_ROOT}/scripts/nda/make_standard7_table.py" \
  --results-json "${OUT}/results.json" \
  --manifest "${OUT}/manifest.json" \
  --output-md "${OUT}/STANDARD7_PROTOCOL_TABLE.md"

echo "===== DONE @ $(date -Iseconds) ====="
echo "Table: ${OUT}/STANDARD7_PROTOCOL_TABLE.md"
