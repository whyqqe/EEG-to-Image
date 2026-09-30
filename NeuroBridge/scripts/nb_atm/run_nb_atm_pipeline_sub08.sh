#!/usr/bin/env bash
# NB-ATM hybrid pipeline (sub-08): Stage A encoder → Prior → DecodeAligner → gen/eval.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
OUT="${OUT:-${NB_ROOT}/outputs/nb_atm_vith/sub-08}"
TRAIN_ROOT="${TRAIN_ROOT:-${NB_ROOT}/results/things_eeg/nb-atm-vith-intra-sub08}"
VITH_FEAT="${NB_ROOT}/data/things_eeg/image_feature/ViT-H-14"
VITH_AUG="${VITH_FEAT}/GaussianBlur-GaussianNoise-LowResolution-Mosaic"
CLIP_TRAIN="${BRAINIT}/outputs/atm_bridge/clip_img_train_1024.npy"
CLIP_TEST="${BRAINIT}/outputs/atm_bridge/clip_img_test_1024.npy"
GALLERY="${BRAINIT}/outputs/eval/atm_baseline/test_ViT-H-14_laion2b_s32b_b79k_features.npy"
ATM_PRIOR="${BRAINIT}/checkpoints/atm_diffusion_prior/sub-08/diffusion_prior.pt"
DINO_DIR="${NB_ROOT}/outputs/nb_nmb_sota_v2/sub-08/targets"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
RETRAIN="${RETRAIN:-0}"

EMBED_DIR="${OUT}/embeds"
P1="${OUT}/phase1_prior"
ALIGN="${OUT}/decode_aligner"
GEN="${OUT}/generation"
mkdir -p "${OUT}" "${EMBED_DIR}" "${P1}" "${ALIGN}" "${GEN}" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"

find_ckpt() {
  find "${TRAIN_ROOT}" -type f -path '*sub-08*/checkpoint_test_best.pth' 2>/dev/null | sort | tail -1
}

echo "===== [Stage A] NB-ATM ViT-H encoder (1024-d direct) @ $(date -Iseconds) ====="
RESULT_CSV=$(find "${TRAIN_ROOT}" -path '*sub-08*/result.csv' 2>/dev/null | sort | tail -1 || true)
if [[ "${RETRAIN}" == "1" ]] || [[ -z "${RESULT_CSV}" ]]; then
  "${PYTHON}" train.py \
    --device "${DEVICE}" \
    --num_epochs 50 \
    --batch_size 1024 \
    --learning_rate 1e-4 \
    --output_dir "${TRAIN_ROOT}" \
    --output_name sub-08 \
    --train_subject_ids 8 \
    --test_subject_ids 8 \
    --eeg_encoder_type ATM \
    --num_subjects 10 \
    --image_feature_dir "${VITH_FEAT}" \
    --aug_image_feature_dirs "${VITH_AUG}" \
    --text_feature_dir "" \
    --eeg_data_dir data/things_eeg/preprocessed_eeg \
    --selected_channels P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz PO4 PO8 O1 Oz O2 \
    --image_aug --eeg_aug --eeg_aug_type smooth \
    --frozen_eeg_prior --image_test_aug \
    --img_l2norm --eeg_l2norm \
    --projector direct \
    --feature_dim 1024 \
    --data_average \
    --save_weights --save_by_top1 \
    --softplus --seed 2025
else
  echo "[SKIP] Stage A (result exists: ${RESULT_CSV})"
fi

CKPT=$(find_ckpt)
if [[ -z "${CKPT}" ]]; then
  echo "[ERROR] no checkpoint under ${TRAIN_ROOT}" >&2
  exit 1
fi
echo "[INFO] encoder ckpt=${CKPT}"

echo "===== [Stage A-e] Extract embeds + gallery retrieval ====="
if [[ "${RETRAIN}" == "1" ]] || [[ ! -f "${EMBED_DIR}/z_eeg_raw_test.npy" ]]; then
  "${PYTHON}" scripts/nb_atm/extract_nb_atm_embeds.py \
    --checkpoint "${CKPT}" \
    --encoder-type atm \
    --output-dir "${EMBED_DIR}" \
    --image-feature-dir "${VITH_FEAT}" \
    --device "${DEVICE}"
else
  echo "[SKIP] embed extract"
fi

"${PYTHON}" scripts/nb_atm/eval_clip_gallery_retrieval.py \
  --embed-npy "${EMBED_DIR}/z_eeg_raw_test.npy" \
  --gallery "${GALLERY}" \
  --tag "stage_a_raw" \
  --output-json "${OUT}/retrieval_stage_a_raw.json"

echo "===== [Stage B] ATM Diffusion Prior (NB cond, warm-start) @ $(date -Iseconds) ====="
if [[ "${RETRAIN}" == "1" ]] || [[ ! -f "${P1}/prior_pretrained_test_clip_1024.npy" ]]; then
  "${PYTHON}" scripts/nb_adapter/train_nb_prior.py \
    --embed-dir "${EMBED_DIR}" \
    --clip-train "${CLIP_TRAIN}" \
    --clip-test "${CLIP_TEST}" \
    --gallery "${GALLERY}" \
    --output-dir "${P1}" \
    --tag prior_pretrained \
    --input-key raw \
    --init-ckpt "${ATM_PRIOR}" \
    --epochs 40 \
    --batch-size 512 \
    --patience 10 \
    --sample-steps 50 \
    --n-samples 3 \
    --device "${DEVICE}"
else
  echo "[SKIP] prior"
fi

"${PYTHON}" scripts/nb_atm/eval_clip_gallery_retrieval.py \
  --embed-npy "${P1}/prior_pretrained_test_clip_1024.npy" \
  --gallery "${GALLERY}" \
  --tag "stage_b_prior" \
  --output-json "${OUT}/retrieval_stage_b_prior.json"

echo "===== [Stage B2] Blend encoder + prior (α=0.5) ====="
BLEND_PRIOR="${OUT}/blend_encoder_prior_a50.npy"
if [[ ! -f "${BLEND_PRIOR}" ]]; then
  "${PYTHON}" scripts/nb_adapter/ensemble_embeds.py \
    --rag-npy "${EMBED_DIR}/z_eeg_raw_test.npy" \
    --prior-npy "${P1}/prior_pretrained_test_clip_1024.npy" \
    --output-npy "${BLEND_PRIOR}" \
    --alpha 0.5
else
  echo "[SKIP] blend prior"
fi

echo "===== [Stage C0] DINOv2 targets ====="
if [[ ! -f "${DINO_DIR}/dinov2_train.npy" ]]; then
  export HF_HUB_CACHE="/project/peilab/why/cache/eeg-brainit/hf/hub"
  "${PYTHON}" scripts/nmb/nmb_build_offline_targets.py --output-dir "${DINO_DIR}" --device "${DEVICE}"
else
  echo "[SKIP] DINO targets"
fi

echo "===== [Stage C] DecodeAligner (ATM backbone) @ $(date -Iseconds) ====="
if [[ "${RETRAIN}" == "1" ]] || [[ ! -f "${ALIGN}/decode_aligner_report.json" ]]; then
  mkdir -p "${ALIGN}/probe"
  if [[ ! -f "${ALIGN}/train_neighbor_idx.npy" ]]; then
    "${PYTHON}" - <<PY
import numpy as np
from pathlib import Path
out = Path("${ALIGN}")
q = np.load("${EMBED_DIR}/z_eeg_raw_train.npy").astype(np.float32)
g = np.load("${CLIP_TRAIN}").astype(np.float32)
q = q / np.linalg.norm(q, axis=1, keepdims=True).clip(1e-8)
g = g / np.linalg.norm(g, axis=1, keepdims=True).clip(1e-8)
idx = np.argsort(-(q @ g.T), axis=1)[:, :5]
np.save(out / "train_neighbor_idx.npy", idx)
print("neighbors", idx.shape)
PY
  fi
  if [[ ! -f "${ALIGN}/probe/probe_supervision.npz" ]]; then
    "${PYTHON}" scripts/nmb/nmb_build_probe_targets.py \
      --embed-npy "${EMBED_DIR}/z_eeg_raw_train.npy" \
      --neighbor-idx-npy "${ALIGN}/train_neighbor_idx.npy" \
      --clip-train-npy "${CLIP_TRAIN}" \
      --output-dir "${ALIGN}/probe" \
      --max-samples 512 \
      --strength 0.4 \
      --device "${DEVICE}"
  fi
  "${PYTHON}" scripts/nmb/nmb_decode_aligner_train.py \
    --checkpoint "${CKPT}" \
    --encoder-type atm \
    --dino-train-npy "${DINO_DIR}/dinov2_train.npy" \
    --dino-test-npy "${DINO_DIR}/dinov2_test.npy" \
    --clip-train-npy "${CLIP_TRAIN}" \
    --clip-test-npy "${CLIP_TEST}" \
    --probe-supervision "${ALIGN}/probe/probe_supervision.npz" \
    --output-dir "${ALIGN}" \
    --feature-dim 512 \
    --num-epochs 40 \
    --batch-size 512 \
    --device "${DEVICE}"
else
  echo "[SKIP] DecodeAligner"
fi

echo "===== [Stage C-mem] Memory router ====="
if [[ ! -f "${ALIGN}/memory/rag_soft5_test_clip_1024.npy" ]]; then
  mkdir -p "${ALIGN}/memory"
  "${PYTHON}" scripts/nmb/nmb_memory_router.py \
    --embed-dir "${ALIGN}" \
    --clip-train "${CLIP_TRAIN}" \
    --clip-test "${CLIP_TEST}" \
    --output-dir "${ALIGN}/memory" \
    --input-key proj \
    --soft-k 5 --soft-tau 0.07
else
  echo "[SKIP] memory"
fi

echo "===== [Stage D] Blends ====="
BLEND_DECODE="${ALIGN}/blend/mem_decode_a50.npy"
BLEND_ALL="${OUT}/blend/mem_prior_decode_a50.npy"
mkdir -p "${ALIGN}/blend" "${OUT}/blend"
if [[ ! -f "${BLEND_DECODE}" ]]; then
  "${PYTHON}" scripts/nb_adapter/ensemble_embeds.py \
    --rag-npy "${ALIGN}/memory/rag_soft5_test_clip_1024.npy" \
    --prior-npy "${ALIGN}/decode_vith1024_test_clip_1024.npy" \
    --output-npy "${BLEND_DECODE}" \
    --alpha 0.5
fi
if [[ ! -f "${BLEND_ALL}" ]]; then
  "${PYTHON}" scripts/nb_adapter/ensemble_embeds.py \
    --rag-npy "${BLEND_DECODE}" \
    --prior-npy "${P1}/prior_pretrained_test_clip_1024.npy" \
    --output-npy "${BLEND_ALL}" \
    --alpha 0.5
fi

echo "===== [Stage E] Generation (img2img s=0.4) ====="
NEIGH="${ALIGN}/memory/rag_soft5_neighbor_idx_test.npy"
run_gen() {
  local tag="$1" emb="$2"
  local gdir="${GEN}/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then
    echo "[SKIP] gen ${tag}"
    return 0
  fi
  "${PYTHON}" scripts/nb_adapter/generate_rag_lowlevel.py \
    --embed-npy "${emb}" \
    --neighbor-idx-npy "${NEIGH}" \
    --output-dir "${gdir}" \
    --strength 0.4 \
    --seed 42 \
    --tag "${tag}" \
    --skip-metrics
}

run_gen "atm_raw_s40" "${EMBED_DIR}/z_eeg_raw_test.npy"
run_gen "prior_s40" "${P1}/prior_pretrained_test_clip_1024.npy"
run_gen "blend_enc_prior_s40" "${BLEND_PRIOR}"
run_gen "decode_direct_s40" "${ALIGN}/decode_vith1024_test_clip_1024.npy"
run_gen "blend_mem_decode_s40" "${BLEND_DECODE}"
run_gen "blend_all_s40" "${BLEND_ALL}"

echo "===== [Stage F] CLIP/FID + summary ====="
METRICS="${OUT}/clip_fid_metrics.json"
TAGS="atm_raw_s40,prior_s40,blend_enc_prior_s40,decode_direct_s40,blend_mem_decode_s40,blend_all_s40"
"${PYTHON}" scripts/nb_adapter/eval_clip_fid.py \
  --gen-root "${GEN}" \
  --tags "${TAGS}" \
  --output-json "${METRICS}"

"${PYTHON}" - <<'PY'
import json
from pathlib import Path

out = Path("/project/peilab/why/NeuroBridge/outputs/nb_atm_vith/sub-08")
metrics = json.loads((out / "clip_fid_metrics.json").read_text()) if (out / "clip_fid_metrics.json").is_file() else {}
results = metrics.get("results", [])
best = max(results, key=lambda r: r.get("clip_cosine", 0)) if results else None
ret_a = json.loads((out / "retrieval_stage_a_raw.json").read_text()) if (out / "retrieval_stage_a_raw.json").is_file() else {}
ret_b = json.loads((out / "retrieval_stage_b_prior.json").read_text()) if (out / "retrieval_stage_b_prior.json").is_file() else {}
align_report = {}
rp = out / "decode_aligner/decode_aligner_report.json"
if rp.is_file():
    align_report = json.loads(rp.read_text())
summary = {
    "pipeline": "NB-ATM-vith-sub08",
    "baseline_v1": 0.412,
    "retrieval_stage_a_top1": ret_a.get("top1"),
    "retrieval_prior_top1": ret_b.get("top1"),
    "atm_baseline_top1": 0.345,
    "best_gen": best,
    "decode_aligner": align_report,
    "all_gen": results,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "===== DONE NB-ATM pipeline @ $(date -Iseconds) ====="
