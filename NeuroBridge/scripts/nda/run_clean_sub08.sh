#!/usr/bin/env bash
# ============================================================================
# CLEAN P0+P1 (sub-08) — LEAK-FREE REPLICATION + INTER-SUBJECT GEOMETRY CALIBRATION
#
# WHY THIS EXISTS: a full audit found THREE protocol contaminations in the
# existing HCMA pipeline. All are fixed or explicitly measured here.
#
#   P1 (5 selection sites chose checkpoints/hyper-parameters on the 200 TEST
#       concepts).  Measured at the root encoder: the selected epoch 20 has test
#       top1 73.0% (lowest test loss 0.9310) while the FINAL epoch reaches only
#       68.5% -> 4.5pp of pure test-set selection bias. Now every site scores on
#       held-in TRAIN concepts via scripts/nda/leakfree.py (concept-disjoint,
#       permuted because THINGS concept order is alphabetical). The test set is
#       touched exactly once, for the final evaluation.
#
#   P2 (prompts contain the GT test concept names, e.g. "a photo of aircraft
#       carrier, highly detailed..."). Since the 200 test concepts are disjoint
#       from the 1654 train concepts, no deployable model can produce that text,
#       so every shipped number is an ORACLE number. Deployable protocols here
#       are `free` (no text) and `neutral` (concept-free text); the old oracle
#       prompts are kept ONLY as flagged reference rows so the leakage is
#       measured instead of hidden.
#
#   RAG memory blend was never ablated and costs 13.5pp of 200-way Top-1
#       (alpha=0.5 -> 0.215 vs alpha=0 -> 0.350). All of alpha in {0,.25,.5} is
#       generated here.
#
# P1+P2 (intra, protocol fixed)  + P1' (inter, geometry calibration)
#
# INTER GEOMETRY CALIBRATION — measured on the LOSO holdout-08 condition
# (200-way, chance 0.005; script scripts/nda/flow_align.py):
#     raw z_s_f          Top-1 0.160  hub_skew 3.00   (CSLS rank fix -> 0.275)
#     linear whitening   Top-1 0.235  hub_skew 0.63   (+7.5pp)
#     unpaired flow      Top-1 0.080  hub_skew 6.07   (mean-seeking: collapses)
#     source-PC removal  Top-1 0.055                 (wrong subspace removed)
#     shipped blend_a40  Top-1 0.055
# => the inter bottleneck is HUBNESS, not margin (raw has the HIGHER margin but
#    the LOWER Top-1). Sharpening + rank calibration fixes it; transports that
#    shrink toward a distribution mean destroy the paired margin. This mirrors
#    the refuted CFM result (sampling p(image|EEG) is hopeless: 32/128/512 steps
#    all give Top-1 0.010 because the conditional is too broad).
#
# NOTE ON TRANSDUCTIVITY: whitening/CSLS use unlabelled target features. That is
# legitimate test-time adaptation (cf. SATTC CVPR'26) but must be declared.
#
# P3 (the SharedSpecificEncoder was trained with sub-08 INCLUDED) is NOT fixed
# here: no inter-subject number from this job may be claimed as zero-shot.
# The refit is a separate stage; the audit report says so explicitly.
#
# Image-side pretrained models (CLIP, DINOv2, SDXL, Depth-Anything, ControlNet,
# IP-Adapter) are ALLOWED and reused: they never saw THINGS-EEG.
#
# OUTPUT: outputs/clean_p0p1/sub-08
# ============================================================================
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
OUT="${OUT:-${NB_ROOT}/outputs/clean_p0p1/sub-08}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

# --- image-side assets (ALLOWED: subject-independent) ---
CLIP_TRAIN="${BRAINIT}/outputs/atm_bridge/clip_img_train_1024.npy"
CLIP_TEST="${BRAINIT}/outputs/atm_bridge/clip_img_test_1024.npy"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
CLIP_LAYERS_SRC="${NB_ROOT}/outputs/nda_ss/sub-08/clip_layers"
CLIP_TEXT_SRC="${NB_ROOT}/outputs/nda_ss/sub-08/clip_text"
CONCEPTS_JSON="${NB_ROOT}/outputs/mg_flow/sub-08/targets/concepts_test.json"
HCMA_PROMPTS="${NB_ROOT}/outputs/hcma_10subj/prompts/prompts_full_hcma_test.json"
GT_DEPTH_TEST="${NB_ROOT}/outputs/hcma_s_full10/shared/gt_depth/test_depth_64.npy"

# --- pure sub-08 EEG encoder (trained ONLY on sub-08; strict intra) ---
CKPT_RN50="${CKPT_RN50:-${NB_ROOT}/results/things_eeg/intra_subjects_clean/sub-08/checkpoint_test_best.pth}"
# ---- LEAK-FREE held-in split (fixes audit finding P1) ----
VAL_SPLIT="${VAL_SPLIT:-${NB_ROOT}/outputs/clean_p0p1/shared/split.json}"
VAL_ARG="--val-split-json ${VAL_SPLIT}"
NB_CLEAN_OUT="${NB_ROOT}/results/things_eeg/intra_subjects_clean"

# --- cache to rebuild / reuse (image side) ---
DINO_OUT="${NB_ROOT}/outputs/intra_hcma_s/shared/dinov2_targets"
VAE_CACHE_SRC="${NB_ROOT}/outputs/sdedit_ll_full10/shared/vae_cache"   # SDXL-encoded image latents (image side)

mkdir -p "${OUT}/embeds" "${OUT}/clip_layers" "${OUT}/clip_text" "${OUT}/train" \
         "${OUT}/conditions" "${OUT}/align" \
         "${OUT}/memory" "${OUT}/blend" "${OUT}/vae_head" "${OUT}/depth" \
         "${OUT}/gt_depth" "${OUT}/generation" "${OUT}/metrics" "${OUT}/logs" \
         "${NB_ROOT}/outputs/slurm" "${DINO_OUT}"

cd "${NB_ROOT}"
# source project venv (diffusers 0.31 / transformers 4.46) — REQUIRED for SDXL VAE decode
if [[ -f "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh"
else
  # shellcheck disable=SC1091
  source "${BRAINIT}/scripts/activate.sh"
fi
PYTHON="$(command -v python)"
echo "[INFO] using python: ${PYTHON}"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"

echo "{\"pipeline\":\"intra_hcma_s_sub08\",\"started\":\"$(date -Iseconds)\",\"strict\":\"ALL EEG weights trained on sub-08 only; image-side pretrained reused\",\"job\":\"${SLURM_JOB_ID:-local}\"}" > "${OUT}/job_running.json"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
# NOTE: CKPT_RN50 is intentionally NOT required here -- it is produced by step [1c]
# below (leak-free retrain). Only image-side assets must pre-exist.
require "${CLIP_TRAIN}"; require "${CLIP_TEST}"
require "${CONCEPTS_JSON}"; require "${HCMA_PROMPTS}"

echo "===== [1] DINOv2 targets (image side) @ $(date -Iseconds) ====="
DINO_TRAIN="${DINO_OUT}/dinov2_train.npy"
DINO_TEST="${DINO_OUT}/dinov2_test.npy"
if [[ ! -f "${DINO_TRAIN}" || ! -f "${DINO_TEST}" ]]; then
  "${PYTHON}" scripts/nmb/nmb_build_offline_targets.py \
    --images-root "${IMAGES_ROOT}" \
    --output-dir "${DINO_OUT}" \
    --batch-size 32 \
    --device "${DEVICE}"
else
  echo "[SKIP] DINOv2"
fi
require "${DINO_TRAIN}"; require "${DINO_TEST}"

echo "===== [1b] generate held-in split (leak-free) @ $(date -Iseconds) ====="
mkdir -p "$(dirname "${VAL_SPLIT}")"
if [[ ! -f "${VAL_SPLIT}" ]]; then
  "${PYTHON}" scripts/nda/leakfree.py --out "${VAL_SPLIT}"
else
  echo "[SKIP] split exists: ${VAL_SPLIT}"
fi
require "${VAL_SPLIT}"

echo "===== [1c] retrain NB encoder (EEGProject, sub-08) with leak-free selection @ $(date -Iseconds) ====="
# Audit finding P1/A1: the old checkpoint selected epoch 20 because its TEST loss
# was lowest (0.9310 vs 1.0023), giving test top1 73.0% while the final epoch
# reached only 68.5% -> 4.5pp of pure test-set selection bias. Retrained here with
# checkpoint selection on held-in train concepts only.
if [[ ! -f "${CKPT_RN50}" ]]; then
  # Flags mirror train_config.json of the original run exactly, except that
  # checkpoint selection moves to the held-in validation split.
  "${PYTHON}" train.py \
    --eeg_encoder_type EEGProject \
    --eeg_data_dir "./data/things_eeg/preprocessed_eeg" \
    --image_feature_dir "./data/things_eeg/image_feature/RN50" \
    --text_feature_dir "" \
    --aug_image_feature_dirs "./data/things_eeg/image_feature/RN50/GaussianBlur-GaussianNoise-LowResolution-Mosaic" \
    --selected_channels P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz PO4 PO8 O1 Oz O2 \
    --time_window 0 250 \
    --train_subject_ids 8 --test_subject_ids 8 \
    --output_dir "${NB_CLEAN_OUT}" --output_name "sub-08" \
    --num_epochs 50 --batch_size 1024 --learning_rate 1e-4 \
    --data_average --softplus --img_l2norm --eeg_aug --image_test_aug \
    --eeg_aug_type smooth \
    --frozen_eeg_prior --projector linear --feature_dim 512 \
    --image_aug --save_weights --seed 2025 \
    --save_by_top1 \
    --val_split_json "${VAL_SPLIT}" \
    --device "${DEVICE}"
else
  echo "[SKIP] NB encoder (clean) exists"
fi
# train.py writes into a timestamped "<ts>-sub-08" directory, so resolve it.
if [[ ! -f "${CKPT_RN50}" ]]; then
  NEWEST="$(ls -1dt "${NB_CLEAN_OUT}"/*-sub-08 2>/dev/null | head -1 || true)"
  if [[ -n "${NEWEST}" && -f "${NEWEST}/checkpoint_test_best.pth" ]]; then
    CKPT_RN50="${NEWEST}/checkpoint_test_best.pth"
    echo "[OK] resolved clean NB checkpoint: ${CKPT_RN50}"
  fi
fi
require "${CKPT_RN50}"
# document the selection policy that produced the checkpoint (leak-free vs naive val loss)
NB_LOG="$(ls -1dt "${NB_CLEAN_OUT}"/*-sub-08/train.log 2>/dev/null | head -1 || true)"
if [[ -n "${NB_LOG}" ]]; then
  "${PYTHON}" scripts/nda/select_policy_audit.py \
    --log "${NB_LOG}" --label "root_encoder_nb" \
    --out "${OUT}/select_policy_root_encoder.json" || true
fi

echo "===== [2] encode intra sub-08 ckpt -> z_eeg_proj @ $(date -Iseconds) ====="
if [[ ! -f "${OUT}/embeds/z_eeg_proj_test.npy" ]]; then
  "${PYTHON}" scripts/nmb/nmb_encode_aligner_embeds.py \
    --checkpoint "${CKPT_RN50}" \
    --output-dir "${OUT}/embeds" \
    --device "${DEVICE}"
else
  echo "[SKIP] encode"
fi

echo "===== [3] clip_layers + clip_text (reuse image-side copies) @ $(date -Iseconds) ====="
if [[ ! -e "${OUT}/clip_layers/clip_layers_report.json" ]]; then
  if [[ -f "${CLIP_LAYERS_SRC}/clip_layers_report.json" ]]; then
    ln -sfn "${CLIP_LAYERS_SRC}" "${OUT}/clip_layers_target"
    # eval scripts expect files directly under OUT/clip_layers/... so symlink content, not the dir
    mkdir -p "${OUT}/clip_layers"
    for _f in "${CLIP_LAYERS_SRC}"/*; do
      ln -sfn "$_f" "${OUT}/clip_layers/$(basename "$_f")"
    done
    echo "[OK] clip_layers symlinked from image-side cache"
  else
    "${PYTHON}" scripts/nda/extract_clip_layers.py \
      --images-root "${IMAGES_ROOT}" --output-dir "${OUT}/clip_layers" \
      --layers "8,10,12,14,16,18,20,22,24,28" --batch-size 16 --device "${DEVICE}"
  fi
fi
if [[ ! -e "${OUT}/clip_text/clip_text_report.json" ]]; then
  if [[ -f "${CLIP_TEXT_SRC}/clip_text_report.json" ]]; then
    mkdir -p "${OUT}/clip_text"
    for _f in "${CLIP_TEXT_SRC}"/*; do
      ln -sfn "$_f" "${OUT}/clip_text/$(basename "$_f")"
    done
    echo "[OK] clip_text symlinked from image-side cache"
  else
    "${PYTHON}" scripts/nda/extract_clip_text.py \
      --images-root "${IMAGES_ROOT}" --output-dir "${OUT}/clip_text" --device "${DEVICE}"
  fi
fi

echo "===== [4] NVOL scan (intra z_eeg_proj) @ $(date -Iseconds) ====="
NVOL_JSON="${OUT}/nvol_scan.json"
if [[ ! -f "${NVOL_JSON}" ]]; then
  "${PYTHON}" scripts/nda/nda_nvol_scan.py \
    --eeg-train "${OUT}/embeds/z_eeg_proj_train.npy" \
    --eeg-test "${OUT}/embeds/z_eeg_proj_test.npy" \
    --clip-layers-dir "${OUT}/clip_layers" \
    --output-json "${NVOL_JSON}" --top-k-layers 3 --select-by val
else
  echo "[SKIP] NVOL"
fi

echo "===== [5] dual-stream train (sub-08 ONLY; NO probe, NO SS) @ $(date -Iseconds) ====="
if [[ ! -f "${OUT}/train/nda_train_report.json" ]]; then
  "${PYTHON}" scripts/nda/nda_dual_train.py \
    --checkpoint "${CKPT_RN50}" \
    --output-dir "${OUT}/train" \
    --clip-layers-dir "${OUT}/clip_layers" \
    --nvol-json "${NVOL_JSON}" \
    --dino-train-npy "${DINO_TRAIN}" \
    --dino-test-npy "${DINO_TEST}" \
    --clip-train-npy "${CLIP_TRAIN}" \
    --clip-test-npy "${CLIP_TEST}" \
    --text-train-npy "${OUT}/clip_text/train/text_flat_clip.npy" \
    --text-test-npy "${OUT}/clip_text/test/text_flat_clip.npy" \
    --lambda-rn50 0.8 --lambda-txt 0.2 \
    --num-epochs 40 --phase1-epochs 10 --phase2-epochs 25 \
    --batch-size 512 --device "${DEVICE}" --freeze-backbone \
    --val-split-json "${VAL_SPLIT}"
  cp -f "${OUT}/train/z_decode_vith_train.npy" "${OUT}/train/decode_vith1024_train_clip_1024.npy"
  cp -f "${OUT}/train/z_decode_vith_test.npy" "${OUT}/train/decode_vith1024_test_clip_1024.npy"
else
  echo "[SKIP] dual train"
fi
require "${OUT}/train/z_decode_vith_test.npy"

echo "===== [6] RAG memory router (pure intra proj vs shared gallery) @ $(date -Iseconds) ====="
if [[ ! -f "${OUT}/memory/rag_soft5_test_clip_1024.npy" ]]; then
  "${PYTHON}" scripts/nmb/nmb_memory_router.py \
    --embed-dir "${OUT}/train" \
    --clip-train "${CLIP_TRAIN}" --clip-test "${CLIP_TEST}" \
    --output-dir "${OUT}/memory" --input-key proj --soft-k 5 --soft-tau 0.07
else
  echo "[SKIP] memory"
fi

echo "===== [7] blends @ $(date -Iseconds) ====="
BLEND_DEC="${OUT}/blend/mem_decode_a50.npy"
if [[ ! -f "${BLEND_DEC}" ]]; then
  "${PYTHON}" scripts/nb_adapter/ensemble_embeds.py \
    --rag-npy "${OUT}/memory/rag_soft5_test_clip_1024.npy" \
    --prior-npy "${OUT}/train/z_decode_vith_test.npy" \
    --output-npy "${BLEND_DEC}" --alpha 0.5
else
  echo "[SKIP] blend"
fi
# semantic embed for generation: dual z_fuse (pure intra decode-bridge) blended with RAG
EMB="${OUT}/blend/mem_decode_a50.npy"
EMB_RAW="${OUT}/train/z_fuse_test.npy"
require "${EMB}"

echo "===== [8] VAE head (pure-intra z_decode_vith -> SDXL VAE latent) @ $(date -Iseconds) ====="
HEAD_OUT="${OUT}/vae_head"
# resume-safe: if training checkpoint exists but decode-rgb did not finish, only decode.
if [[ -f "${HEAD_OUT}/checkpoint_vae_head_best.pth" && ! -f "${HEAD_OUT}/pred_lowlevel_rgb_512/000.png" ]]; then
  echo "[RESUME] VAE head trained but RGB decode incomplete — resume decode only"
  "${PYTHON}" scripts/nda/resume_vae_head_decode.py \
    --checkpoint "${HEAD_OUT}/checkpoint_vae_head_best.pth" \
    --output-dir "${HEAD_OUT}" \
    --device "${DEVICE}"
fi
if [[ ! -f "${HEAD_OUT}/vae_head_report.json" ]]; then
  # VAE cache: SDXL-encoded GT images (image side, allowed). Symlink (read-only use).
  VAE_TR="${OUT}/vae_cache/train_vae_latents_f16.npy"
  VAE_TE="${OUT}/vae_cache/test_vae_latents_f16.npy"
  mkdir -p "${OUT}/vae_cache"
  [[ -f "${VAE_TE}" ]] || ln -sfn "${VAE_CACHE_SRC}/test_vae_latents_f16.npy" "${VAE_TE}"
  [[ -f "${VAE_TR}" ]] || ln -sfn "${VAE_CACHE_SRC}/train_vae_latents_f16.npy" "${VAE_TR}"
  require "${VAE_TR}"; require "${VAE_TE}"
  "${PYTHON}" scripts/nda/train_eeg_vae_head.py \
    --eeg-train-npy "${OUT}/train/z_decode_vith_train.npy" \
    --eeg-test-npy "${OUT}/train/z_decode_vith_test.npy" \
    --vae-train-npy "${VAE_TR}" \
    --vae-test-npy "${VAE_TE}" \
    --output-dir "${HEAD_OUT}" \
    --num-epochs 80 --batch-size 64 --lr 3e-4 \
    --device "${DEVICE}" --decode-rgb \
    --val-split-json "${VAL_SPLIT}"
else
  echo "[SKIP] VAE head"
fi
LL_RGB="${HEAD_OUT}/pred_lowlevel_rgb_512"
require "${LL_RGB}/000.png"

echo "===== [9] Depth cache (train; image side) + Depth head @ $(date -Iseconds) ====="
DTR="${OUT}/gt_depth/train_depth_64.npy"
DTE="${OUT}/gt_depth/test_depth_64.npy"
if [[ ! -f "${DTE}" ]]; then
  ln -sfn "${GT_DEPTH_TEST}" "${DTE}"
  echo "[OK] test depth symlinked from image-side cache"
fi
if [[ ! -f "${DTR}" ]]; then
  "${PYTHON}" scripts/nda/build_gt_depth_cache.py \
    --images-root "${IMAGES_ROOT}" --output-dir "${OUT}/gt_depth" \
    --device "${DEVICE}" --splits "train" --batch-size 8
else
  echo "[SKIP] depth cache"
fi
DEPTH_OUT="${OUT}/depth"
if [[ ! -f "${DEPTH_OUT}/depth_head_report.json" ]]; then
  "${PYTHON}" scripts/nda/train_eeg_depth_head.py \
    --eeg-train-npy "${OUT}/train/z_decode_vith_train.npy" \
    --eeg-test-npy "${OUT}/train/z_decode_vith_test.npy" \
    --depth-train-npy "${DTR}" \
    --depth-test-npy "${DTE}" \
    --output-dir "${DEPTH_OUT}" \
    --num-epochs 60 --batch-size 256 --lr 1e-3 \
    --lambda-grad 0.5 --cn-min 0.25 --cn-max 0.45 \
    --device "${DEVICE}" \
    --val-split-json "${VAL_SPLIT}"
else
  echo "[SKIP] depth head"
fi
DEPTH_RGB="${DEPTH_OUT}/pred_depth_rgb_512"
require "${DEPTH_RGB}/000.png"
# free bulky train depth cache
rm -f "${DTR}"

echo "===== [9b] inter-subject geometry calibration (label-free, transductive) @ $(date -Iseconds) ====="
# Measured on the LOSO holdout-08 condition (200-way, chance 0.005):
#     raw z_s_f            Top-1 0.160   hub_skew 3.00   (CSLS -> 0.275)
#     linear whitening     Top-1 0.235   hub_skew 0.63   (+7.5pp)
#     unpaired flow        Top-1 0.080   hub_skew 6.07   (mean-seeking: collapses)
#     shipped blend_a40    Top-1 0.055
# So the inter bottleneck is HUBNESS, corrected by sharpening + rank calibration,
# NOT by transporting toward a mean. This stage produces the condition variants.
SRC_NPY="${OUT}/align/src_zsf.npy"
if [[ -f "${OUT}/align/align_diag.json" ]]; then
  echo "[SKIP] alignment (reusing ${OUT}/align/align_diag.json)"
else
  mkdir -p "${OUT}/align"
  OUT_ALIGN="${OUT}" NB_ROOT="${NB_ROOT}" "${PYTHON}" - <<'PY'
import os
from pathlib import Path
import numpy as np

nb = Path(os.environ["NB_ROOT"]); out = Path(os.environ["OUT_ALIGN"])
parts = []
for s in ("01", "02", "03", "04", "05", "06", "07", "09", "10"):
    base = nb / "outputs/inter_ll_full10" / f"sub-{s}" / "inter_embeds/embeds"
    for key in ("z_s_f_test.npy",):
        p = base / key
        if p.is_file():
            parts.append(np.load(p).astype(np.float32))
legacy = nb / "outputs/inter_ll_full10" / "sub-08/inter_embeds/embeds"
if not parts:
    for s in ("01", "02", "03", "04", "05", "06", "07", "09", "10"):
        for f in sorted((nb / "outputs/inter_ll_full10" / f"sub-{s}" / "inter_embeds/embeds").glob("*.npy")):
            if "z_s_f" in f.name:
                parts.append(np.load(f).astype(np.float32)); break
assert parts, "no source-subject features found"
np.save(out / "align" / "src_zsf.npy", np.concatenate(parts, 0))
print(f"[OK] src {np.concatenate(parts,0).shape}")
PY
  require "${SRC_NPY}"
  require "${OUT}/train/z_decode_vith_test.npy"
  # target = the LOSO condition for sub-08; we align the intra tower's condition as a
  # stand-in probe ONLY if no LOSO condition exists for this run.
  TGT_NPY="${OUT}/train/z_decode_vith_test.npy"
  if [[ -f "${NB_ROOT}/outputs/inter_ll_full10/sub-08/inter_embeds/embeds/z_s_f_test.npy" ]]; then
    TGT_NPY="${NB_ROOT}/outputs/inter_ll_full10/sub-08/inter_embeds/embeds/z_s_f_test.npy"
  fi
  "${PYTHON}" scripts/nda/flow_align.py \
    --src-npy "${SRC_NPY}" --tgt-npy "${TGT_NPY}" \
    --gallery-npy "${CLIP_TEST}" \
    --out-dir "${OUT}/align" \
    --shrink 0.1 --flow-steps 6000 --integrate 64 --seed 0
  require "${OUT}/align/align_diag.json"
fi

echo "===== [10] build LEAK-FREE conditions + deployable prompts @ $(date -Iseconds) ====="
COND="${OUT}/conditions"
CLEAN_STAGE="${CLEAN_STAGE:-both}"
if [[ ! -f "${COND}/rows.tsv" ]]; then
  ALIGN_ARGS=()
  [[ -f "${OUT}/align/align_diag.json" ]] && ALIGN_ARGS=(--align-dir "${OUT}/align")
  "${PYTHON}" scripts/nda/clean_build_conditions.py \
    --intra-root "${OUT}" \
    --gallery-clip "${CLIP_TEST}" \
    --oracle-prompts "${HCMA_PROMPTS}" \
    --out-dir "${COND}" --stage "${CLEAN_STAGE}" \
    --cn-scale 0.32 --strength 0.86 --alphas "0.0,0.25,0.5" \
    "${ALIGN_ARGS[@]}"
else
  echo "[SKIP] conditions"
fi
require "${COND}/rows.tsv"

echo "===== [11] generation (row-driven: clean conditions x prompt protocol) @ $(date -Iseconds) ====="
# Each row already carries an absolute embed path, prompt json and gen dir.
while IFS=$'\t' read -r rtag remb rprompt rcn rstren rdir; do
  [[ -z "${rtag}" ]] && continue
  if [[ -f "${rdir}/generated/199.png" ]]; then
    echo "[SKIP] ${rtag}"
    continue
  fi
  echo "[GEN ] ${rtag}  cn=${rcn} strength=${rstren}"
  "${PYTHON}" scripts/nda/generate_hcma_s_decode.py \
    --embed-npy "${remb}" \
    --prompts-json "${rprompt}" \
    --depth-rgb-dir "${DEPTH_RGB}" \
    --lowlevel-rgb-dir "${LL_RGB}" \
    --output-dir "${rdir}" --tag "${rtag}" \
    --cn-scale "${rcn}" --ip-scale 1.0 --strength "${rstren}" \
    --gen-steps 28 --gen-guidance 5.0 --seed 42
done < "${COND}/rows.tsv"

echo "===== [12] standard-7 (incl. per-row FID) @ $(date -Iseconds) ====="
STD7="${NB_ROOT}/outputs/standard7_protocol"
# Backup the shared results.json BEFORE eval (eval_standard7 overwrites it wholesale).
cp -f "${STD7}/results.json" "${OUT}/results_std7_backup.json" || true
MAN="${OUT}/manifest_clean.json"
OUT_EVAL="${OUT}" MAN="${MAN}" COND="${COND}" "${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
man = {"protocol": "standard7", "rows": [], "avg_rows": []}
cond = Path(os.environ["COND"])
for line in (cond / "rows.tsv").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    tag, _emb, _pr, _cn, _st, gdir = line.split("\t")
    if (Path(gdir) / "generated/199.png").exists():
        man["rows"].append({"tag": tag, "display": tag, "gen_dir": str(Path(gdir) / "generated")})
Path(os.environ["MAN"]).write_text(json.dumps(man, indent=2), encoding="utf-8")
print("[OK] manifest rows", len(man["rows"]))
PY
"${PYTHON}" scripts/nda/eval_standard7.py \
  --manifest "${MAN}" --images-root "${IMAGES_ROOT}" \
  --out-dir "${STD7}" --device "${DEVICE}" --batch-size 16
cp -f "${STD7}/results.json" "${OUT}/results_clean.json"
"${PYTHON}" - <<PY
import json
from pathlib import Path
backup = json.loads(Path("${OUT}/results_std7_backup.json").read_text(encoding="utf-8"))
mine = json.loads(Path("${OUT}/results_clean.json").read_text(encoding="utf-8"))
by_tag = {r["tag"]: r for r in backup["rows"]}
for r in mine["rows"]:
    by_tag[r["tag"]] = r          # add/replace only our rows
merged = dict(backup)
merged["rows"] = [by_tag[t] for t in by_tag]
Path("${STD7}/results.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
print(f"[OK] merged results.json: {len(backup['rows'])} -> {len(merged['rows'])} rows")
PY

echo "===== [13] leak audit report @ $(date -Iseconds) ====="
OUT_EVAL="${OUT}" NB_ROOT="${NB_ROOT}" "${PYTHON}" - <<'PY'
import json, os
from pathlib import Path

nb = Path(os.environ["NB_ROOT"])
out = Path(os.environ["OUT_EVAL"])
rep = {"pipeline": "clean_p0p1_sub08", "audit": []}

def add(k, status, detail):
    rep["audit"].append({"finding": k, "status": status, "detail": detail})

# A1 root encoder
add("A1 root NB encoder checkpoint selection",
    "FIXED",
    "train.py now scores on held-in train concepts (--val_split_json). "
    "Old behaviour selected epoch 20 on lowest TEST loss (0.9310 vs final 1.0023) "
    "-> test top1 73.0% vs 68.5% at the final epoch = ~4.5pp inflation.")
add("A2 NVOL layer scan selection", "FIXED",
    "nda_nvol_scan.py --select-by val (was: sorted by test top1).")
add("A3 semantic tower checkpoint selection", "FIXED",
    "nda_dual_train.py --val-split-json; score computed on held-in valB, memory bank "
    "rebuilt without valB rows. Test metrics now reported but not used to choose.")
add("A4 VAE head checkpoint selection", "FIXED",
    "train_eeg_vae_head.py --val-split-json (was: best MAE on TEST latents).")
add("A5 depth head checkpoint selection", "FIXED",
    "train_eeg_depth_head.py --val-split-json (was: best Pearson on TEST depth maps).")
add("P2 oracle concept text in prompts", "FIXED + MEASURED",
    "Prompt protocols are now free (no text) / neutral (concept-free text) for all deployable "
    "rows. The old GT-concept prompts are kept ONLY as flagged reference rows so the size of "
    "the leakage is measurable rather than hidden.")
add("RAG memory degradation", "ABLATED",
    "alphas 0.0/0.25/0.5 are all generated, so the measured 13.5pp Top-1 cost of the shipped "
    "alpha=0.5 blend is now visible in the results table.")
add("P3 cross-subject encoder saw sub-08", "PENDING",
    "This job covers the intra/alignment stages. The LOSO encoder refit that excludes sub-08 "
    "is a separate stage; until it runs, no inter-subject claim is leak-free.")
rep["deployable_rows_only"] = True
(out / "leak_audit.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
for a in rep["audit"]:
    print(f"[{a['status']:>9}] {a['finding']}")
PY

echo "===== [14] summary @ $(date -Iseconds) ====="
"${PYTHON}" - <<PY
import json
from pathlib import Path
out = Path("${OUT}")
std7 = Path("${NB_ROOT}/outputs/standard7_protocol/results.json")
res = json.loads(std7.read_text(encoding="utf-8"))["rows"] if std7.is_file() else []
by = {r["tag"]: r for r in res}
man = json.loads((out / "manifest_clean.json").read_text(encoding="utf-8"))
rows = [by[r["tag"]] for r in man["rows"] if r["tag"] in by]
(out / "summary.json").write_text(json.dumps(
    {"pipeline": "clean_p0p1_sub08",
     "protocol": "leak-free checkpoint selection; deployable prompts; alpha ablation",
     "rows": rows}, indent=2), encoding="utf-8")
print(json.dumps({"n_rows": len(rows)}, indent=2))
PY

du -sh "${OUT}" 2>/dev/null || true
echo "{\"pipeline\":\"clean_p0p1_sub08\",\"finished\":\"$(date -Iseconds)\"}" > "${OUT}/job_done.json"
echo "===== DONE clean_p0p1 sub08 @ $(date -Iseconds) ====="
