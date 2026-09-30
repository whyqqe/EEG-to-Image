#!/usr/bin/env bash
#SBATCH --job-name=loso-vae
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=160G
#SBATCH --time=0-06:00:00
#SBATCH --output=/project/peilab/why/third_party/loso_pipeline/logs/%x-%j.out
#SBATCH --error=/project/peilab/why/third_party/loso_pipeline/logs/%x-%j.err

# Re-extract SDXL VAE latents in fp32 (the previous fp16 run wrote all-NaN).
set -uo pipefail
module load slurm nvhpc-hpcx-cuda12/23.11

HERE=/project/peilab/why/third_party/loso_pipeline
# shellcheck disable=SC1091
source "${HERE}/env.sh"
PY="${LOSO_VENV}/bin/python"

echo "host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "=== wipe + rebuild vae_latent (fp32 encode) ==="

rm -f "${LOSO_DATA}/targets/vae_latent_train.npy" \
      "${LOSO_DATA}/targets/vae_latent_train.npy.progress.json" \
      "${LOSO_DATA}/targets/vae_latent_test.npy" \
      "${LOSO_DATA}/targets/vae_latent_test.npy.progress.json"

"${PY}" "${HERE}/scripts/prep_multimodal.py" \
    --targets vae_latent \
    --splits train test \
    --batch-size "${LOSO_VAE_BS:-16}" \
    --dtype fp16 \
    || exit 40

"${PY}" "${HERE}/scripts/validate_targets.py" --splits train test || exit 50
echo "=== [OK] VAE rebuild complete $(date -Iseconds) ==="
