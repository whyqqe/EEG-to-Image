#!/usr/bin/env bash
# sdedit_ll FULL-10-subject (complete result, SOTA-aligned):
# For EVERY subject 01..10: train per-subject EEG->VAE lowlevel head on HCMA z_ret,
#   decode LL blur RGB, then sdedit_ll (strength 0.82, 28 steps, CFG 5.0, IP 1.0)
#   with that subject's HCMA embed + shared HCMA prompts.
# Then standard-7 eval on manifest_full10 (25 rows) + pooled FID + SOTA table.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/sdedit_ll_full10}"
HCMA10="${HCMA10:-${NB_ROOT}/outputs/hcma_10subj}"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
SUBJECTS="${SUBJECTS:-1,2,3,4,5,6,7,8,9,10}"
VAE_EPOCHS="${VAE_EPOCHS:-80}"

export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${TORCH_HOME}"

mkdir -p "${OUT}/shared/vae_cache" "${OUT}/logs" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" || true
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export DEVICE OUT NB_ROOT IMAGES_ROOT

echo "{\"pipeline\":\"sdedit_ll_full10\",\"started\":\"$(date -Iseconds)\",\"goal\":\"per-subject sdedit_ll on all 10 subjects, SOTA-aligned 10-subj avg\"}" \
  > "${OUT}/job_running.json"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }

PROMPT="${HCMA10}/prompts/prompts_full_hcma_test.json"
require "${PROMPT}"

# ------------------------------------------------------------------ shared VAE target cache
echo "===== [0] Shared SDXL VAE target latents (image-side, subject-agnostic) @ $(date -Iseconds) ====="
VC="${OUT}/shared/vae_cache"
if [[ ! -f "${VC}/train_vae_latents_f16.npy" || ! -f "${VC}/test_vae_latents_f16.npy" ]]; then
  "${PYTHON}" scripts/nda/build_gt_vae_latents.py \
    --images-root "${IMAGES_ROOT}" \
    --output-dir "${VC}" \
    --device "${DEVICE}" \
    --batch-size 8 \
    --splits "train,test"
else
  echo "[SKIP] shared VAE cache exists"
fi
require "${VC}/train_vae_latents_f16.npy"
require "${VC}/test_vae_latents_f16.npy"

IFS=',' read -ra SUBJ_ARR <<< "${SUBJECTS}"

# ------------------------------------------------------------------ per-subject train + gen
for SID in "${SUBJ_ARR[@]}"; do
  SID="$(echo "${SID}" | tr -d ' ')"
  STAG="$(printf "sub-%02d" "${SID}")"
  SOUT="${OUT}/${STAG}"
  mkdir -p "${SOUT}/vae_head" "${SOUT}/generation"
  echo "########## ${STAG} @ $(date -Iseconds) ##########"

  ZTR="${HCMA10}/${STAG}/zret/z_ret_train.npy"
  ZTE="${HCMA10}/${STAG}/zret/z_ret_test.npy"
  EMB="${HCMA10}/${STAG}/ft/embeds/blend_nda_cfm_f_a40_test.npy"
  require "${ZTR}"; require "${ZTE}"; require "${EMB}"

  # ---- [1] train EEG->VAE lowlevel head (HCMA z_ret 512 -> SDXL VAE latent)
  HEAD_OUT="${SOUT}/vae_head"
  if [[ ! -f "${HEAD_OUT}/vae_head_report.json" ]]; then
    "${PYTHON}" scripts/nda/train_eeg_vae_head.py \
      --eeg-train-npy "${ZTR}" \
      --eeg-test-npy "${ZTE}" \
      --vae-train-npy "${VC}/train_vae_latents_f16.npy" \
      --vae-test-npy "${VC}/test_vae_latents_f16.npy" \
      --output-dir "${HEAD_OUT}" \
      --num-epochs "${VAE_EPOCHS}" \
      --batch-size 64 \
      --lr 3e-4 \
      --device "${DEVICE}" \
      --decode-rgb
  else
    echo "[SKIP] vae head ${STAG}"
  fi
  LL_RGB="${HEAD_OUT}/pred_lowlevel_rgb_512"
  require "${LL_RGB}/199.png"

  # ---- [2] sdedit_ll decode (LL init + HCMA semantics)
  GDIR="${SOUT}/generation/sdedit_ll"
  if [[ ! -f "${GDIR}/generated/199.png" ]]; then
    rm -rf "${GDIR}"
    "${PYTHON}" scripts/nda/generate_atm_aligned_decode.py \
      --mode sdedit \
      --embed-npy "${EMB}" \
      --prompts-json "${PROMPT}" \
      --lowlevel-rgb-dir "${LL_RGB}" \
      --output-dir "${GDIR}" \
      --tag "sdedit_ll" \
      --strength 0.82 \
      --ip-scale 1.0 \
      --gen-steps 28 \
      --gen-guidance 5.0 \
      --seed 42
  else
    echo "[SKIP] sdedit_ll gen ${STAG}"
  fi
  require "${GDIR}/generated/199.png"
done

echo "===== [3] Standard-7 eval (25 rows) @ $(date -Iseconds) ====="
# Reuse standard7_protocol out-dir/cache so existing hcma rows are cached; sdedit_ll rows new.
STD7_OUT="${NB_ROOT}/outputs/standard7_protocol"
"${PYTHON}" scripts/nda/eval_standard7.py \
  --manifest "${STD7_OUT}/manifest_full10.json" \
  --images-root "${IMAGES_ROOT}" \
  --out-dir "${STD7_OUT}" \
  --device "${DEVICE}" \
  --batch-size 16

echo "===== [4] Pooled FID (sdedit_ll full10) @ $(date -Iseconds) ====="
"${PYTHON}" scripts/nda/eval_pooled_fid.py \
  --root "${OUT}" \
  --tag "sdedit_ll" \
  --images-root "${IMAGES_ROOT}" \
  --output-json "${OUT}/metrics_pooled_fid_sdedit_ll.json" \
  --device "${DEVICE}" \
  --batch-size 32

echo "===== [5] Patch manifest pooled FID + SOTA table @ $(date -Iseconds) ====="
"${PYTHON}" - <<'PY'
import json
from pathlib import Path
std7 = Path("/project/peilab/why/NeuroBridge/outputs/standard7_protocol")
man_p = std7 / "manifest_full10.json"
man = json.loads(man_p.read_text())
pooled = json.loads((Path("/project/peilab/why/NeuroBridge/outputs/sdedit_ll_full10/metrics_pooled_fid_sdedit_ll.json")).read_text())
val = pooled["pooled_fid_unique_gt"]
for spec in man["avg_rows"]:
    if spec["group"] == "sdedit_ll-10subj":
        spec["fid_pooled"] = float(val)
man_p.write_text(json.dumps(man, indent=2), encoding="utf-8")
print(f"[OK] patched sdedit pooled FID = {val:.2f}")
PY

"${PYTHON}" scripts/nda/make_standard7_table.py \
  --results-json "${STD7_OUT}/results.json" \
  --manifest "${STD7_OUT}/manifest_full10.json" \
  --output-md "${STD7_OUT}/STANDARD7_FULL10_SOTA.md"

echo "{\"pipeline\":\"sdedit_ll_full10\",\"finished\":\"$(date -Iseconds)\"}" > "${OUT}/job_done.json"
du -sh "${OUT}" 2>/dev/null || true
echo "===== DONE sdedit_ll full10 @ $(date -Iseconds) ====="
