#!/bin/bash
# Official Intra-subject recipe (Things-EEG / RN50 / EEGProject).
# Resume-safe: train.py exits 0 if a completed result.csv already exists for that subject.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

IMAGE_FEATURE_DIR="./data/things_eeg/image_feature/RN50"
AUG_DIR="./data/things_eeg/image_feature/RN50/GaussianBlur-GaussianNoise-LowResolution-Mosaic"
EEG_DATA_DIR="./data/things_eeg/preprocessed_eeg"
OUTPUT_DIR="./results/things_eeg/intra-subjects"
DEVICE="${DEVICE:-cuda:0}"
EEG_ENCODER_TYPE="EEGProject"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LEARNING_RATE="1e-4"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
SELECTED_CHANNELS=(P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz PO4 PO8 O1 Oz O2)
PROJECTOR="linear"
FEATURE_DIM=512

# Optional: SUB_IDS="8" or "1 2 3" to limit subjects
SUB_IDS="${SUB_IDS:-1 2 3 4 5 6 7 8 9 10}"

# Sanity checks
test -f "${EEG_DATA_DIR}/info.json"
test -f "${IMAGE_FEATURE_DIR}/image_train.npy"
test -f "${AUG_DIR}/train.npy"
for sid in ${SUB_IDS}; do
  printf -v sub "sub-%02d" "${sid}"
  test -f "${EEG_DATA_DIR}/${sub}/train.npy"
  test -f "${EEG_DATA_DIR}/${sub}/test.npy"
done

mkdir -p "${OUTPUT_DIR}"

for SUB_ID in ${SUB_IDS}; do
  OUTPUT_NAME=$(printf "sub-%02d" "${SUB_ID}")
  echo "===== Training subject ${SUB_ID} @ $(date -Iseconds) ====="
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
echo "===== Intra-subject run finished @ $(date -Iseconds) ====="
