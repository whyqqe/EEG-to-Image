#!/usr/bin/env bash
#SBATCH --job-name=loso-prep
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --output=/project/peilab/why/third_party/loso_pipeline/logs/%x-%j.out
#SBATCH --error=/project/peilab/why/third_party/loso_pipeline/logs/%x-%j.err

# Stage 0 -- reconstruct the multimodal targets from scratch.
#
# This deliberately re-derives every frozen teacher target rather than reusing the
# arrays already in NeuroBridge/data/things_eeg/image_feature, because those were
# produced by a pipeline whose text branch never existed: its caption files hold
# the VLM *prompt* rather than generated text, and only at concept granularity.
# A target set that is partly a prompt and partly real features cannot be audited
# after the fact, so it is cheaper to rebuild.
#
#   1/5 index      image/concept listing + EEG<->image mapping contract, verified
#                  against the .npy headers
#   2/5 blip2      fetch the captioning model early so a network problem fails
#                  here in minutes instead of an hour into feature extraction
#   3/5 captions   per-image BLIP2 captions (16,540 train + 200 test)
#   4/5 targets    CLIP ViT-H-14 image/text, DINOv2-L, SDXL VAE latent, pixels
#   5/5 validate   completeness (memmap rows are indistinguishable from zeros)
#                  + CLIP zero-shot retrieval as the end-to-end pairing check
set -uo pipefail

module load slurm nvhpc-hpcx-cuda12/23.11

HERE=/project/peilab/why/third_party/loso_pipeline
# shellcheck disable=SC1091
source "${HERE}/env.sh"

PY="${LOSO_VENV}/bin/python"
echo "host=$(hostname)  job=${SLURM_JOB_ID:-local}  python=${PY}"
echo "clip=${LOSO_CLIP_ID}"
echo "dino=${LOSO_DINO_ID}"
echo "vae =${LOSO_VAE_ID}"
echo "blip2=${LOSO_BLIP2_ID}"
echo "hf_hub=${HUGGINGFACE_HUB_CACHE}"
echo

# --- 1/5 index ---------------------------------------------------------------
echo "=== [1/5] build + verify THINGS index ==="
"${PY}" "${HERE}/scripts/build_index.py" --subject sub-08 || exit 10

# --- 2/5 captioning model ----------------------------------------------------
echo
echo "=== [2/5] fetch ${LOSO_BLIP2_ID} ==="
"${PY}" - <<'PY' || exit 20
import sys
from huggingface_hub import snapshot_download
from loso import paths
try:
    p = snapshot_download(paths.BLIP2_ID, cache_dir=paths.HF_HUB,
                          allow_patterns=["*.json", "*.txt", "*.model",
                                          "*.safetensors", "*.bin", "*.md"])
    print(f"[blip2] available at {p}")
except Exception as exc:
    print(f"[FATAL] could not fetch {paths.BLIP2_ID}: {exc}", file=sys.stderr)
    raise SystemExit(20)
PY

# --- 3/5 captions ------------------------------------------------------------
echo
echo "=== [3/5] BLIP2 captions ==="
for split in train test; do
  echo "--- captions: ${split}"
  "${PY}" "${HERE}/scripts/gen_captions.py" \
      --split "${split}" \
      --batch-size "${LOSO_CAPTION_BS:-24}" \
      --max-new-tokens 48 || exit 30
done

# --- 4/5 frozen targets ------------------------------------------------------
echo
echo "=== [4/5] extract multimodal targets ==="
# BLIP2 is released now; sampling from the OPT decoder is stochastic under
# do_sample, but this pipeline runs greedy (do_sample=False) so captions are
# reproducible and no seed is required here.
"${PY}" "${HERE}/scripts/prep_multimodal.py" \
    --targets clip_image clip_text_caption clip_text_concept dino vae_latent pixels \
    --splits train test \
    --batch-size "${LOSO_FEAT_BS:-64}" \
    || exit 40

# --- 5/5 validate ------------------------------------------------------------
echo
echo "=== [5/5] validate ==="
"${PY}" "${HERE}/scripts/validate_targets.py" --splits train test || exit 50

echo
echo "=== [OK] Stage 0 data preparation complete $(date -Iseconds) ==="
