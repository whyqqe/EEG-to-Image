#!/usr/bin/env bash
# Full Phase 0–3 pipeline for sub-08: RAG → Prior → Dual-teacher → Ensemble + low-level → CLIP/FID.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
OUT="${OUT:-${NB_ROOT}/outputs/nb_full_phases/sub-08}"
SUBJECT="${SUBJECT:-8}"
MAX_IMAGES="${MAX_IMAGES:-0}"   # 0 = all 200 test images
DEVICE="${DEVICE:-cuda:0}"
PYTHON="${PYTHON:-python}"

CKPT_RN50="${CKPT_RN50:-${NB_ROOT}/results/things_eeg/intra-subjects/20260901-074319-sub-08/checkpoint_test_best.pth}"
CLIP_TRAIN="${CLIP_TRAIN:-${BRAINIT}/outputs/atm_bridge/clip_img_train_1024.npy}"
CLIP_TEST="${CLIP_TEST:-${BRAINIT}/outputs/atm_bridge/clip_img_test_1024.npy}"
GALLERY="${GALLERY:-${BRAINIT}/outputs/eval/atm_baseline/test_ViT-H-14_laion2b_s32b_b79k_features.npy}"
ATM_PRIOR_CKPT="${ATM_PRIOR_CKPT:-${BRAINIT}/checkpoints/atm_diffusion_prior/sub-08/diffusion_prior.pt}"

EMBED_DIR="${OUT}/embeds"
P0="${OUT}/phase0"
P1="${OUT}/phase1"
P2="${OUT}/phase2/dual_teacher"
P3="${OUT}/phase3"
GEN_ROOT="${OUT}/generation_full200"
METRICS_JSON="${OUT}/clip_fid_metrics_full200.json"
SUMMARY_JSON="${OUT}/summary.json"

mkdir -p "${OUT}" "${EMBED_DIR}" "${P0}" "${P1}" "${P2}" "${P3}" "${GEN_ROOT}"
cd "${NB_ROOT}"

echo "===== [Prep] Extract NB RN50 embeds @ $(date -Iseconds) ====="
if [[ ! -f "${EMBED_DIR}/z_eeg_raw_test.npy" ]]; then
  "${PYTHON}" scripts/nb_adapter/extract_nb_embeds.py \
    --nb-root "${NB_ROOT}" \
    --checkpoint "${CKPT_RN50}" \
    --subject "${SUBJECT}" \
    --output-dir "${EMBED_DIR}" \
    --device "${DEVICE}"
else
  echo "[INFO] embeds exist, skip"
fi

echo "===== [Phase 0] RAG-Recon @ $(date -Iseconds) ====="
if [[ -f "${P0}/rag_soft5_test_clip_1024.npy" ]]; then
  echo "[INFO] Phase 0 done, skip"
else
  "${PYTHON}" scripts/nb_adapter/rag_recon.py \
    --embed-dir "${EMBED_DIR}" \
    --clip-train "${CLIP_TRAIN}" \
    --output-dir "${P0}" \
    --input-key proj \
    --soft-k 5 \
    --soft-tau 0.07
fi

echo "===== [Phase 1] Diffusion prior (scratch) @ $(date -Iseconds) ====="
if [[ -f "${P1}/prior_scratch_test_clip_1024.npy" ]]; then
  echo "[INFO] prior_scratch done, skip"
else
  "${PYTHON}" scripts/nb_adapter/train_nb_prior.py \
    --embed-dir "${EMBED_DIR}" \
    --clip-train "${CLIP_TRAIN}" \
    --clip-test "${CLIP_TEST}" \
    --gallery "${GALLERY}" \
    --output-dir "${P1}" \
    --tag prior_scratch \
    --input-key raw \
    --epochs 50 \
    --batch-size 512 \
    --patience 12 \
    --sample-steps 50 \
    --n-samples 3 \
    --device "${DEVICE}"
fi

echo "===== [Phase 1] Diffusion prior (ATM init) @ $(date -Iseconds) ====="
if [[ -f "${P1}/prior_pretrained_test_clip_1024.npy" ]]; then
  echo "[INFO] prior_pretrained done, skip"
else
  "${PYTHON}" scripts/nb_adapter/train_nb_prior.py \
    --embed-dir "${EMBED_DIR}" \
    --clip-train "${CLIP_TRAIN}" \
    --clip-test "${CLIP_TEST}" \
    --gallery "${GALLERY}" \
    --output-dir "${P1}" \
    --tag prior_pretrained \
    --input-key raw \
    --init-ckpt "${ATM_PRIOR_CKPT}" \
    --epochs 40 \
    --batch-size 512 \
    --patience 10 \
    --sample-steps 50 \
    --n-samples 3 \
    --device "${DEVICE}"
fi

echo "===== [Phase 2] Dual-teacher fine-tune @ $(date -Iseconds) ====="
if [[ -f "${P2}/dual_vith1024_test_clip_1024.npy" ]]; then
  echo "[INFO] Phase 2 done, skip"
else
  "${PYTHON}" scripts/nb_adapter/finetune_nb_dual.py \
    --nb-root "${NB_ROOT}" \
    --checkpoint "${CKPT_RN50}" \
    --subject "${SUBJECT}" \
    --output-dir "${P2}" \
    --num-epochs 25 \
    --batch-size 1024 \
    --learning-rate 5e-5 \
    --lambda-vith 0.5 \
    --lambda-direct 0.3 \
    --device "${DEVICE}"
fi

echo "===== [Phase 3] Ensemble RAG + prior @ $(date -Iseconds) ====="
if [[ -f "${P3}/ensemble_rag_prior_test_clip_1024.npy" ]]; then
  echo "[INFO] Phase 3 ensemble done, skip"
else
  "${PYTHON}" scripts/nb_adapter/ensemble_embeds.py \
    --rag-npy "${P0}/rag_soft5_test_clip_1024.npy" \
    --prior-npy "${P1}/prior_pretrained_test_clip_1024.npy" \
    --output-npy "${P3}/ensemble_rag_prior_test_clip_1024.npy" \
    --alpha 0.5 \
    --report-json "${P3}/ensemble_report.json"
fi

echo "===== [Gen] SDXL full pipeline @ $(date -Iseconds) ====="
declare -A EMBEDS=(
  [teacher]="${CLIP_TEST}"
  [baseline_mlp]="${NB_ROOT}/outputs/nb_adapter/sub-08/adapters/mlp_test_clip_1024.npy"
  [rag_top1]="${P0}/rag_top1_test_clip_1024.npy"
  [rag_soft5]="${P0}/rag_soft5_test_clip_1024.npy"
  [prior_scratch]="${P1}/prior_scratch_test_clip_1024.npy"
  [prior_pretrained]="${P1}/prior_pretrained_test_clip_1024.npy"
  [dual_vith1024]="${P2}/dual_vith1024_test_clip_1024.npy"
  [ensemble_rag_prior]="${P3}/ensemble_rag_prior_test_clip_1024.npy"
)

for tag in teacher baseline_mlp rag_top1 rag_soft5 prior_scratch prior_pretrained dual_vith1024 ensemble_rag_prior; do
  emb="${EMBEDS[${tag}]}"
  if [[ ! -f "${emb}" ]]; then
    echo "[WARN] skip ${tag}: missing ${emb}"
    continue
  fi
  echo "--- generate ${tag} ---"
  "${PYTHON}" scripts/nb_adapter/generate_from_embeds.py \
    --embed-npy "${emb}" \
    --output-dir "${GEN_ROOT}/${tag}" \
    --tag "full200_${tag}" \
    --max-images "${MAX_IMAGES}" \
    --seed 42 \
    --skip-metrics
done

echo "===== [Gen] RAG low-level img2img @ $(date -Iseconds) ====="
"${PYTHON}" scripts/nb_adapter/generate_rag_lowlevel.py \
  --embed-npy "${P0}/rag_soft5_test_clip_1024.npy" \
  --neighbor-idx-npy "${P0}/rag_soft5_neighbor_idx.npy" \
  --output-dir "${GEN_ROOT}/rag_soft5_lowlevel" \
  --strength 0.5 \
  --max-images "${MAX_IMAGES}" \
  --seed 42 \
  --tag "rag_soft5_lowlevel" \
  --skip-metrics

echo "===== [Eval] CLIP + FID (200) @ $(date -Iseconds) ====="
TAGS="teacher,baseline_mlp,rag_top1,rag_soft5,prior_scratch,prior_pretrained,dual_vith1024,ensemble_rag_prior,rag_soft5_lowlevel"
"${PYTHON}" scripts/nb_adapter/eval_clip_fid.py \
  --gen-root "${GEN_ROOT}" \
  --tags "${TAGS}" \
  --output-json "${METRICS_JSON}" \
  --max-images "${MAX_IMAGES}"

echo "===== [Summary] @ $(date -Iseconds) ====="
"${PYTHON}" scripts/nb_adapter/summarize_phases.py \
  --out-root "${OUT}" \
  --output-json "${SUMMARY_JSON}"

echo "===== DONE all phases @ $(date -Iseconds) ====="
