#!/usr/bin/env bash
# Official Intra recipe with OpenCLIP ViT-H-14 teacher (same space as SDXL IP-Adapter).
# Resume-safe: train.py skips subjects that already have result.csv.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

IMAGE_ENCODER_TYPE="${IMAGE_ENCODER_TYPE:-ViT-H-14}"
IMAGE_FEATURE_DIR="./data/things_eeg/image_feature/${IMAGE_ENCODER_TYPE}"
AUG_DIR="${IMAGE_FEATURE_DIR}/GaussianBlur-GaussianNoise-LowResolution-Mosaic"
EEG_DATA_DIR="./data/things_eeg/preprocessed_eeg"
OUTPUT_DIR="${OUTPUT_DIR:-./results/things_eeg/intra-subjects-vit-h}"
DEVICE="${DEVICE:-cuda:0}"
EEG_ENCODER_TYPE="EEGProject"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LEARNING_RATE="1e-4"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
SELECTED_CHANNELS=(P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz PO4 PO8 O1 Oz O2)
PROJECTOR="linear"
FEATURE_DIM=512
SUB_IDS="${SUB_IDS:-1 2 3 4 5 6 7 8 9 10}"

test -f "${EEG_DATA_DIR}/info.json"
test -f "${IMAGE_FEATURE_DIR}/image_train.npy"
test -f "${AUG_DIR}/train.npy"
for sid in ${SUB_IDS}; do
  printf -v sub "sub-%02d" "${sid}"
  test -f "${EEG_DATA_DIR}/${sub}/train.npy"
done

mkdir -p "${OUTPUT_DIR}"
echo "[INFO] ViT-H Intra → ${OUTPUT_DIR} subjects=${SUB_IDS}"

for SUB_ID in ${SUB_IDS}; do
  OUTPUT_NAME=$(printf "sub-%02d" "${SUB_ID}")
  echo "===== Training ${OUTPUT_NAME} (ViT-H) @ $(date -Iseconds) ====="
  python train.py \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${LEARNING_RATE}" \
    --output_name "${OUTPUT_NAME}" \
    --eeg_encoder_type "${EEG_ENCODER_TYPE}" \
    --train_subject_ids "${SUB_ID}" \
    --test_subject_ids "${SUB_ID}" \
    --softplus \
    --num_epochs "${NUM_EPOCHS}" \
    --image_feature_dir "${IMAGE_FEATURE_DIR}" \
    --text_feature_dir "" \
    --eeg_data_dir "${EEG_DATA_DIR}" \
    --device "${DEVICE}" \
    --output_dir "${OUTPUT_DIR}" \
    --selected_channels "${SELECTED_CHANNELS[@]}" \
    --image_aug \
    --aug_image_feature_dirs "${AUG_DIR}" \
    --eeg_aug \
    --eeg_aug_type "smooth" \
    --frozen_eeg_prior \
    --image_test_aug \
    --img_l2norm \
    --projector "${PROJECTOR}" \
    --feature_dim "${FEATURE_DIM}" \
    --data_average \
    --save_weights \
    --seed 2025
done

python compute_avg_results.py --result_dir "${OUTPUT_DIR}"
echo "===== ViT-H Intra finished @ $(date -Iseconds) ====="
