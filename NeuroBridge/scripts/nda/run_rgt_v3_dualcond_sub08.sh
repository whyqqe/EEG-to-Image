#!/usr/bin/env bash
# RGT-v3: bottleneck-first — dual-condition gen (image IP + text prompt) + clean fusion
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
OUT="${OUT:-${NB_ROOT}/outputs/rgt_v3_dualcond/sub-08}"
V2="${V2:-${NB_ROOT}/outputs/rgt_cfm_v2/sub-08}"
NDA_SS="${NDA_SS:-${NB_ROOT}/outputs/nda_ss/sub-08}"
SEMTXT="${SEMTXT:-${NB_ROOT}/outputs/nda_v2_semtxt/sub-08}"
SS_CKPT="${SS_CKPT:-${NDA_SS}/ss/checkpoint_ss_calib_best.pth}"
CLIP_TRAIN="${BRAINIT}/outputs/atm_bridge/clip_img_train_1024.npy"
CLIP_TEST="${BRAINIT}/outputs/atm_bridge/clip_img_test_1024.npy"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "${OUT}/eeg_text" "${OUT}/adapt" "${OUT}/memory" "${OUT}/generation" "${OUT}/embeds"
cd "${NB_ROOT}"

echo "===== [0] Prefetch @ $(date -Iseconds) ====="
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
"${PYTHON}" - <<'PY'
import os
os.environ.setdefault("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub")
os.environ.setdefault("OPENCLIP_CACHE_DIR", "/project/peilab/why/cache/eeg-brainit/open_clip")
os.environ.setdefault("TORCH_HOME", "/project/peilab/why/cache/eeg-brainit/torch")
import open_clip
m,_,_=open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k", device="cpu")
print("[OK] openclip", len(m.visual.transformer.resblocks))
PY

test -f "${V2}/cfm/embeds/z_rgt_cfm_test.npy"
test -f "${NDA_SS}/train/z_decode_vith_test.npy"
test -f "${SS_CKPT}"

# reuse v2 embeds
cp -f "${V2}/cfm/embeds/z_rgt_cfm_test.npy" "${OUT}/embeds/"
cp -f "${V2}/cfm/embeds/z_rgt_cfm_train.npy" "${OUT}/embeds/"
cp -f "${V2}/cfm/embeds/z_eeg_proj_test.npy" "${OUT}/embeds/"
cp -f "${V2}/cfm/embeds/z_eeg_proj_train.npy" "${OUT}/embeds/"

echo "===== [1] Memory ====="
if [[ ! -f "${OUT}/memory/rag_soft5_test_clip_1024.npy" ]]; then
  if [[ -f "${V2}/memory/rag_soft5_test_clip_1024.npy" ]]; then
    cp -a "${V2}/memory/." "${OUT}/memory/"
    echo "[OK] reused v2 memory"
  else
    "${PYTHON}" scripts/nmb/nmb_memory_router.py \
      --embed-dir "${OUT}/embeds" \
      --clip-train "${CLIP_TRAIN}" --clip-test "${CLIP_TEST}" \
      --output-dir "${OUT}/memory" --input-key proj --soft-k 5 --soft-tau 0.07
  fi
fi

echo "===== [2] CLIP text teachers ====="
TEXT_DIR="${OUT}/clip_text"
if [[ ! -f "${TEXT_DIR}/clip_text_report.json" ]]; then
  if [[ -f "${SEMTXT}/clip_text/clip_text_report.json" ]]; then
    mkdir -p "${TEXT_DIR}"
    cp -a "${SEMTXT}/clip_text/." "${TEXT_DIR}/"
  else
    "${PYTHON}" scripts/nda/extract_clip_text.py \
      --images-root "${IMAGES_ROOT}" --output-dir "${TEXT_DIR}" --device "${DEVICE}"
  fi
fi

echo "===== [3] Multi-granularity EEG↔text align + prompts ====="
if [[ ! -f "${OUT}/eeg_text/eeg_text_report.json" ]]; then
  "${PYTHON}" scripts/nda/rgt_eeg_text_align.py \
    --ss-checkpoint "${SS_CKPT}" \
    --z-ret-train "${OUT}/embeds/z_eeg_proj_train.npy" \
    --z-ret-test "${OUT}/embeds/z_eeg_proj_test.npy" \
    --text-train "${TEXT_DIR}/train/text_flat_clip.npy" \
    --text-test "${TEXT_DIR}/test/text_flat_clip.npy" \
    --concept-phrases-train "${TEXT_DIR}/train/concept_phrases.json" \
    --concept-phrases-test "${TEXT_DIR}/test/concept_phrases.json" \
    --output-dir "${OUT}/eeg_text" \
    --subject 8 \
    --epochs 25 \
    --top-k-prompt 2 \
    --device "${DEVICE}"
else
  echo "[SKIP] eeg-text"
fi

echo "===== [4] Clean fusion adapt (no mem leak) ====="
if [[ ! -f "${OUT}/adapt/gen_adapt_report.json" ]]; then
  "${PYTHON}" scripts/nda/rgt_gen_adapt_v3.py \
    --cfm-train "${OUT}/embeds/z_rgt_cfm_train.npy" \
    --cfm-test "${OUT}/embeds/z_rgt_cfm_test.npy" \
    --nda-train "${NDA_SS}/train/z_decode_vith_train.npy" \
    --nda-test "${NDA_SS}/train/z_decode_vith_test.npy" \
    --mem-train "${OUT}/memory/rag_soft5_train_clip_1024.npy" \
    --mem-test "${OUT}/memory/rag_soft5_test_clip_1024.npy" \
    --clip-train "${CLIP_TRAIN}" --clip-test "${CLIP_TEST}" \
    --z-ret-test "${OUT}/embeds/z_eeg_proj_test.npy" \
    --z-ret-train-gallery "${OUT}/embeds/z_eeg_proj_train.npy" \
    --neighbor-idx "${OUT}/memory/rag_soft5_neighbor_idx_test.npy" \
    --output-dir "${OUT}/adapt" \
    --min-cfm 0.15 --max-mem 0.45 \
    --base-strength 0.45 --base-ip-scale 0.90
else
  echo "[SKIP] adapt"
fi

echo "===== [5] Dual-condition generation (image IP + text prompt) ====="
NEIGH="${OUT}/memory/rag_soft5_neighbor_idx_test.npy"
STR="${OUT}/adapt/strength_adapt.npy"
IPS="${OUT}/adapt/ip_scale_adapt.npy"
PROMPT="${OUT}/eeg_text/prompts_pred.json"
ORACLE="${OUT}/eeg_text/prompts_oracle.json"

run_gen() {
  local tag="$1" emb="$2"; shift 2
  local gdir="${OUT}/generation/${tag}"
  [[ -f "${gdir}/generated/199.png" ]] && echo "[SKIP] ${tag}" && return 0
  "${PYTHON}" scripts/nb_adapter/generate_rag_lowlevel.py \
    --embed-npy "${emb}" --neighbor-idx-npy "${NEIGH}" \
    --output-dir "${gdir}" --seed 42 --tag "${tag}" --skip-metrics "$@"
}

# A) previous protocol baselines (no text)
run_gen "v3_nda_mem_s40" "${OUT}/adapt/blend_nda_mem.npy" --strength 0.4 --ip-scale 1.0
run_gen "v3_fuse_clean_s40" "${OUT}/adapt/z_gen_adapt_test.npy" --strength 0.4 --ip-scale 1.0

# B) dual-condition: image + EEG-predicted text prompt
run_gen "v3_nda_mem_txt" "${OUT}/adapt/blend_nda_mem.npy" \
  --strength-npy "${STR}" --ip-scale-npy "${IPS}" --prompts-json "${PROMPT}" --gen-guidance 1.5
run_gen "v3_fuse_txt" "${OUT}/adapt/z_gen_adapt_test.npy" \
  --strength-npy "${STR}" --ip-scale-npy "${IPS}" --prompts-json "${PROMPT}" --gen-guidance 1.5
run_gen "v3_nda_cfm_txt" "${OUT}/adapt/blend_nda_cfm.npy" \
  --strength-npy "${STR}" --ip-scale-npy "${IPS}" --prompts-json "${PROMPT}" --gen-guidance 1.5

# C) oracle text upper bound (same image embeds)
run_gen "v3_nda_mem_oracle" "${OUT}/adapt/blend_nda_mem.npy" \
  --strength 0.45 --ip-scale 0.9 --prompts-json "${ORACLE}" --gen-guidance 1.5

echo "===== [6] Metrics ====="
"${PYTHON}" scripts/nb_adapter/eval_clip_fid.py \
  --gen-root "${OUT}/generation" \
  --tags "v3_nda_mem_s40,v3_fuse_clean_s40,v3_nda_mem_txt,v3_fuse_txt,v3_nda_cfm_txt,v3_nda_mem_oracle" \
  --output-json "${OUT}/clip_fid_metrics.json"

"${PYTHON}" - <<PY
import json
from pathlib import Path
out=Path("${OUT}")
metrics=json.loads((out/"clip_fid_metrics.json").read_text())
results=metrics.get("results",[])
best=max(results, key=lambda r:r.get("clip_cosine",0)) if results else None
et=json.loads((out/"eeg_text/eeg_text_report.json").read_text()) if (out/"eeg_text/eeg_text_report.json").is_file() else {}
ad=json.loads((out/"adapt/gen_adapt_report.json").read_text()) if (out/"adapt/gen_adapt_report.json").is_file() else {}
summary={
  "pipeline":"RGT-v3 dual-condition bottleneck fix",
  "focus":"image IP + EEG-text prompts; clean fusion",
  "eeg_text":{k:et.get(k) for k in ("best_text_cos","concept_top1","concept_inhitk","top_k_prompt") if et},
  "adapt":ad,
  "baseline_rgt_v2":0.419,
  "baseline_nda_ss":0.439,
  "baseline_erdc":0.422,
  "best_gen":best,
  "all_gen":results,
}
(out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "===== DONE RGT-v3 @ $(date -Iseconds) ====="
