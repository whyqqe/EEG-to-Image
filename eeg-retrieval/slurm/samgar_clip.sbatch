#!/usr/bin/env bash
#SBATCH --job-name=samgar-clip
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/recon/slurm-clip-%j.log
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/recon/slurm-clip-%j.err
#
# =============================================================================
# Shared OpenCLIP ViT-H-14 conditioning targets for every LOSO fold.
#
# WHY THIS IS ITS OWN JOB RATHER THAN A STAGE OF EACH FOLD
# --------------------------------------------------------
# The CLIP target array depends only on the image corpus, so it is identical for all ten
# folds -- and the fold jobs need the *complete* array (all 1654 training concepts for the
# head's supervision, all 16540 images as the generation neighbour gallery). If each fold
# extracted it, ten concurrent producers would write the same two filenames into the same
# directory, and the failure mode is the worst kind: a fold that reads a half-written
# `.npy` or a 40-concept fragment gets no error, just wrong numbers. So exactly one job
# produces it, and every fold job depends on that job.
#
# The script itself also skips cleanly if the arrays are already complete, so this is
# idempotent and safe to resubmit.
# =============================================================================
set -euo pipefail

RECON_ROOT="/project/peilab/why/eeg-retrieval"
cd "${RECON_ROOT}"
mkdir -p "${RECON_ROOT}/outputs/recon"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="/project/peilab/why/cache/eeg-brainit/hf"
export HF_HUB_CACHE="/project/peilab/why/cache/eeg-brainit/hf/hub"
export OPENCLIP_CACHE_DIR="/project/peilab/why/cache/eeg-brainit/open_clip"
export TORCH_HOME="/project/peilab/why/cache/eeg-brainit/torch"
export TMPDIR="/project/peilab/why/cache/eeg-brainit/tmp"
export PYTHONUNBUFFERED=1
mkdir -p "${TMPDIR}"

PY="/project/peilab/why/eeg-brainit/.venv/bin/python"

OUT="${RECON_ROOT}/data/image_feature/clip_h14_ip_adapter"
# Refuse to start if the shared array is already complete, so a stray resubmit cannot
# overwrite it while folds are reading it.
if [[ -f "${OUT}/clip_h14_train.npy" && -f "${OUT}/clip_h14_test.npy" ]]; then
  echo "[SKIP] ${OUT} already complete"
  exit 0
fi

echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi -L 2>/dev/null | head -1 || true

"${PY}" "${RECON_ROOT}/scripts/recon/extract_clip_h14.py" \
  --out-dir "${OUT}" --splits train,test --batch 128 \
  --verify-against "/project/peilab/why/eeg-brainit/outputs/atm_bridge/clip_img_train_1024.npy"

echo "[OK] shared CLIP targets done"
