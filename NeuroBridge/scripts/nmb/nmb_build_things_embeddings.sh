#!/usr/bin/env bash
# Build THINGS parquet embeddings for Fusion Prior finetune (vae + CLIP-B + CLIP-H).
set -euo pipefail

BRAIN_HIVE="${BRAIN_HIVE:-/project/peilab/why/Brain-HIVE}"
IMAGE_DIR="${IMAGE_DIR:-/project/peilab/why/data/images_set}"
OUT_EMB="${OUT_EMB:-/project/peilab/why/NeuroBridge/outputs/nb_nmb_sota/sub-08/things_embeddings}"
CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
CFG="${BRAIN_HIVE}/configs/build_embeddings.yaml"
ACC_CFG="${BRAIN_HIVE}/configs/gpu_cfg_1gpu.yaml"
PY="${BRAIN_HIVE}/build_embeddings.py"

mkdir -p "${OUT_EMB}"
cd "${BRAIN_HIVE}"
export PYTHONPATH="${BRAIN_HIVE}:${PYTHONPATH:-}"

resolve_turbo_vae() {
  local snap_root="${CACHE}/models--stabilityai--sdxl-turbo/snapshots"
  if [[ -d "${snap_root}" ]]; then
    for snap in $(ls -1d "${snap_root}"/* 2>/dev/null | sort -r); do
      if [[ -f "${snap}/vae/config.json" ]]; then
        echo "${snap}/vae"
        return 0
      fi
    done
  fi
  echo "stabilityai/sdxl-vae"
}

VAE_PATH="$(resolve_turbo_vae)"
CLIP_H="laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
CLIP_B="laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

run_one() {
  local model_path="$1"
  local batch="$2"
  local split="$3"
  local resolution="${4:-}"

  export DATASET_NAME="things"
  export IMAGE_DIR="${IMAGE_DIR}"
  export OUTPUT_DIR="${OUT_EMB}"
  export MODEL_PATH="${model_path}"
  export BATCH_SIZE="${batch}"
  export SPLIT="${split}"
  if [[ -n "${resolution}" ]]; then
    export RESOLUTION="${resolution}"
  else
    unset RESOLUTION
  fi

  echo "[embed] model=${model_path} split=${split} batch=${batch}"
  accelerate launch --config_file "${ACC_CFG}" "${PY}" --config_file "${CFG}"
}

for split in train test; do
  run_one "${VAE_PATH}" 64 "${split}" 128
  run_one "${CLIP_B}" 512 "${split}"
  run_one "${CLIP_H}" 256 "${split}"
done

echo "[OK] THINGS embeddings -> ${OUT_EMB}"
