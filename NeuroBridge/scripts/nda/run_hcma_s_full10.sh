#!/usr/bin/env bash
# HCMA-S FULL-10-subject (dual-tower: Depth-CN x LL-SDEdit), SOTA-aligned.
# For EVERY subject 01..10:
#   depth expert   = EEG->Depth regression trained on that subject's HCMA z_ret (512)
#   LL-RGB init    = sdedit_ll_full10/<sub>/vae_head/pred_lowlevel_rgb_512 (HCMA z_ret VAE head)
#   dual decode    = generate_hcma_s_decode.py  cn=0.40 (depth CN) x strength=0.82 (LL SDEdit)
# Then standard-7 eval on manifest_full10_hcs.json (33 rows) + pooled FID + SOTA table.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/hcma_s_full10}"
HCMA10="${HCMA10:-${NB_ROOT}/outputs/hcma_10subj}"
LL10="${LL10:-${NB_ROOT}/outputs/sdedit_ll_full10}"   # per-subject z_ret VAE heads (LL RGB)
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
STD7_OUT="${NB_ROOT}/outputs/standard7_protocol"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
SUBJECTS="${SUBJECTS:-1,2,3,4,5,6,7,8,9,10}"
DEPTH_EPOCHS="${DEPTH_EPOCHS:-60}"
CN="${CN:-0.40}"
STRENGTH="${STRENGTH:-0.82}"

export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${TORCH_HOME}"

mkdir -p "${OUT}/shared/gt_depth" "${OUT}/logs" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" || true
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export DEVICE OUT NB_ROOT IMAGES_ROOT

echo "{\"pipeline\":\"HCMA-S_full10\",\"started\":\"$(date -Iseconds)\",\"goal\":\"dual Depth-CN x LL-SDEdit on all 10 subjects, PixCorr avg > 0.150\",\"cn\":\"${CN}\",\"strength\":\"${STRENGTH}\"}" \
  > "${OUT}/job_running.json"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }

PROMPT="${HCMA10}/prompts/prompts_full_hcma_test.json"
require "${PROMPT}"

# ------------------------------------------------------------------ [0] shared GT depth cache
echo "===== [0] Shared GT depth cache (image-side, subject-agnostic) @ $(date -Iseconds) ====="
DC="${OUT}/shared/gt_depth"
if [[ ! -f "${DC}/train_depth_64.npy" || ! -f "${DC}/test_depth_64.npy" ]]; then
  "${PYTHON}" scripts/nda/build_gt_depth_cache.py \
    --images-root "${IMAGES_ROOT}" \
    --output-dir "${DC}" \
    --device "${DEVICE}" \
    --splits "train,test" \
    --batch-size 8
else
  echo "[SKIP] shared depth cache exists"
fi
require "${DC}/train_depth_64.npy"
require "${DC}/test_depth_64.npy"

IFS=',' read -ra SUBJ_ARR <<< "${SUBJECTS}"

# ------------------------------------------------------------------ per-subject depth + dual decode
for SID in "${SUBJ_ARR[@]}"; do
  SID="$(echo "${SID}" | tr -d ' ')"
  STAG="$(printf "sub-%02d" "${SID}")"
  SOUT="${OUT}/${STAG}"
  mkdir -p "${SOUT}/depth" "${SOUT}/generation"
  echo "########## ${STAG} @ $(date -Iseconds) ##########"

  ZTR="${HCMA10}/${STAG}/zret/z_ret_train.npy"
  ZTE="${HCMA10}/${STAG}/zret/z_ret_test.npy"
  EMB="${HCMA10}/${STAG}/ft/embeds/blend_nda_cfm_f_a40_test.npy"
  LL_RGB="${LL10}/${STAG}/vae_head/pred_lowlevel_rgb_512"
  require "${ZTR}"; require "${ZTE}"; require "${EMB}"
  require "${LL_RGB}/199.png"

  # ---- [1] Depth expert (EEG->depth on HCMA z_ret 512)
  DEPTH_OUT="${SOUT}/depth"
  # Reuse the exact same protocol sub-08 depth expert already trained for HCMA-S (HCMA z_ret 512).
  if [[ "${STAG}" == "sub-08" && -d "${NB_ROOT}/outputs/hcma_s/sub-08/depth" && -f "${NB_ROOT}/outputs/hcma_s/sub-08/depth/pred_depth_rgb_512/199.png" && ! -e "${DEPTH_OUT}/depth_head_report.json" ]]; then
    ln -sfn "${NB_ROOT}/outputs/hcma_s/sub-08/depth" "${SOUT}/depth_link"
    mkdir -p "${DEPTH_OUT}"
    ln -sfn "${NB_ROOT}/outputs/hcma_s/sub-08/depth/pred_depth_rgb_512" "${DEPTH_OUT}/pred_depth_rgb_512"
    ln -sfn "${NB_ROOT}/outputs/hcma_s/sub-08/depth/depth_head_report.json" "${DEPTH_OUT}/depth_head_report.json"
    echo "[REUSE] sub-08 existing depth expert (HCMA z_ret protocol)"
  fi
  if [[ ! -f "${DEPTH_OUT}/depth_head_report.json" ]]; then
    "${PYTHON}" scripts/nda/train_eeg_depth_head.py \
      --eeg-train-npy "${ZTR}" \
      --eeg-test-npy "${ZTE}" \
      --depth-train-npy "${DC}/train_depth_64.npy" \
      --depth-test-npy "${DC}/test_depth_64.npy" \
      --output-dir "${DEPTH_OUT}" \
      --num-epochs "${DEPTH_EPOCHS}" \
      --batch-size 256 \
      --lr 1e-3 \
      --lambda-grad 0.5 \
      --cn-min 0.25 \
      --cn-max 0.45 \
      --device "${DEVICE}"
  else
    echo "[SKIP] depth expert ${STAG}"
  fi
  DEPTH_RGB="${DEPTH_OUT}/pred_depth_rgb_512"
  require "${DEPTH_RGB}/199.png"

  # ---- [2] HCMA-S dual decode (Depth-CN x LL-SDEdit) hs_c040_s082 recipe
  GDIR="${SOUT}/generation/hs_c040_s082"
  if [[ ! -f "${GDIR}/generated/199.png" ]]; then
    rm -rf "${GDIR}"
    "${PYTHON}" scripts/nda/generate_hcma_s_decode.py \
      --embed-npy "${EMB}" \
      --prompts-json "${PROMPT}" \
      --depth-rgb-dir "${DEPTH_RGB}" \
      --lowlevel-rgb-dir "${LL_RGB}" \
      --output-dir "${GDIR}" \
      --tag "hs_c040_s082" \
      --cn-scale "${CN}" \
      --ip-scale 1.0 \
      --strength "${STRENGTH}" \
      --gen-steps 28 \
      --gen-guidance 5.0 \
      --seed 42
  else
    echo "[SKIP] dual decode ${STAG}"
  fi
  require "${GDIR}/generated/199.png"
done

# drop bulky train depth cache
if [[ -f "${DC}/train_depth_64.npy" ]]; then
  rm -f "${DC}/train_depth_64.npy"
  echo "[DISK] removed shared train depth cache"
fi

echo "===== [3] Standard-7 eval (33 rows) @ $(date -Iseconds) ====="
# Backup previous 25-row results before overwrite; cache reused so old rows are cheap.
[[ -f "${STD7_OUT}/results.json" ]] && cp -f "${STD7_OUT}/results.json" "${STD7_OUT}/results_25row_backup.json" || true
"${PYTHON}" scripts/nda/eval_standard7.py \
  --manifest "${STD7_OUT}/manifest_full10_hcs.json" \
  --images-root "${IMAGES_ROOT}" \
  --out-dir "${STD7_OUT}" \
  --device "${DEVICE}" \
  --batch-size 16

echo "===== [4] Pooled FID (HCMA-S full10) @ $(date -Iseconds) ====="
"${PYTHON}" scripts/nda/eval_pooled_fid.py \
  --root "${OUT}" \
  --tag "hs_c040_s082" \
  --images-root "${IMAGES_ROOT}" \
  --output-json "${OUT}/metrics_pooled_fid_hcma_s.json" \
  --device "${DEVICE}" \
  --batch-size 32

echo "===== [5] Patch manifest pooled FID + SOTA table @ $(date -Iseconds) ====="
"${PYTHON}" - <<'PY'
import json
from pathlib import Path
std7 = Path("/project/peilab/why/NeuroBridge/outputs/standard7_protocol")
man_p = std7 / "manifest_full10_hcs.json"
man = json.loads(man_p.read_text())
pooled = json.loads((Path("/project/peilab/why/NeuroBridge/outputs/hcma_s_full10/metrics_pooled_fid_hcma_s.json")).read_text())
val = pooled["pooled_fid_unique_gt"]
for spec in man["avg_rows"]:
    if spec["group"] == "HCMA-S-10subj":
        spec["fid_pooled"] = float(val)
man_p.write_text(json.dumps(man, indent=2), encoding="utf-8")
print(f"[OK] patched HCMA-S pooled FID = {val:.2f}")
PY

"${PYTHON}" scripts/nda/make_standard7_table.py \
  --results-json "${STD7_OUT}/results.json" \
  --manifest "${STD7_OUT}/manifest_full10_hcs.json" \
  --output-md "${STD7_OUT}/STANDARD7_FULL10_HCMA_S_SOTA.md"

echo "{\"pipeline\":\"HCMA-S_full10\",\"finished\":\"$(date -Iseconds)\"}" > "${OUT}/job_done.json"
du -sh "${OUT}" 2>/dev/null || true
echo "===== DONE HCMA-S full10 @ $(date -Iseconds) ====="
