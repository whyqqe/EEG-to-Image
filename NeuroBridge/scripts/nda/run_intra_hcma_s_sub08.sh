#!/usr/bin/env bash
# ============================================================================
# INTRA-HCMA-S (sub-08) — STRICT pure-intra, EVERY project weight trained on
# sub-08 ONLY (16540 train samples). Image-side pretrained models/features
# (OpenCLIP ViT-H-14, CLIP-Image/Text, DINOv2, SDXL VAE, Depth-Anything,
#  RN50 gallery, ControlNet/IP-Adapter/SDXL diffusion) are ALLOWED and reused.
#
# NO reuse of any EEG-side weight/feature that touched subjects 1..7/9..10:
#   - SharedSpecificEncoder 9-subj pretrain/calib          -> EXCLUDED
#   - RGT bank / z_ret 9-subj SS encodings                 -> EXCLUDED
#   - hcma_10subj MG-Flow cross-subject / per-subj FT      -> EXCLUDED
#   - nda_ss decode_vith / mem blends (SS-produced)        -> EXCLUDED
#
# Pipeline (each step resume-safe via guard file):
#   [0] env / assets
#   [1] DINOv2 targets            (image side, rebuild; ~10-20 min)
#   [2] encode intra-subject ckpt -> z_eeg_proj (pure sub-08, RN50 SSP-512)
#   [3] clip_layers + clip_text   (image side; reuse NDA-SS copies via symlink)
#   [4] NVOL scan                 (eeg = intra z_eeg_proj)
#   [5] dual-stream train         (nda_dual_train.py, sub-08 only, text reg)
#         -> z_decode_vith / z_sem_rn50 / z_fuse / z_eeg_proj (pure intra)
#   [6] RAG memory router         (query = intra proj vs shared image gallery)
#   [7] blend mem_decode_a50
#   [8] VAE head (EEG->SDXL latent, trained on pure-intra z_decode_vith)
#   [9] Depth cache GT (train; image side Depth-Anything) + Depth head
#   [10] HCMA-S dual decode grid (Depth-CN x LL-SDEdit) @ sdedit semantics
#   [11] standard-7 eval + FID + compare vs cross-subject SOTA (sdedit_ll_s082 /
#        hs_c040_s082 / hcma_10subj refs)
#
# OUTPUT: outputs/intra_hcma_s/sub-08
# ============================================================================
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
OUT="${OUT:-${NB_ROOT}/outputs/intra_hcma_s/sub-08}"
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
CKPT_RN50="${CKPT_RN50:-${NB_ROOT}/results/things_eeg/intra-subjects/20260901-074319-sub-08/checkpoint_test_best.pth}"

# --- cache to rebuild / reuse (image side) ---
DINO_OUT="${NB_ROOT}/outputs/intra_hcma_s/shared/dinov2_targets"
VAE_CACHE_SRC="${NB_ROOT}/outputs/sdedit_ll_full10/shared/vae_cache"   # SDXL-encoded image latents (image side)

mkdir -p "${OUT}/embeds" "${OUT}/clip_layers" "${OUT}/clip_text" "${OUT}/train" \
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
require "${CKPT_RN50}"; require "${CLIP_TRAIN}"; require "${CLIP_TEST}"
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
    --output-json "${NVOL_JSON}" --top-k-layers 3
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
    --batch-size 512 --device "${DEVICE}" --freeze-backbone
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
    --device "${DEVICE}" --decode-rgb
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
    --device "${DEVICE}"
else
  echo "[SKIP] depth head"
fi
DEPTH_RGB="${DEPTH_OUT}/pred_depth_rgb_512"
require "${DEPTH_RGB}/000.png"
# free bulky train depth cache
rm -f "${DTR}"

echo "===== [10] generation: sdedit-LL baseline + HCMA-S dual grid @ $(date -Iseconds) ====="
# baseline: pure-intra semantic + LL SDEdit (mirror sdedit_ll_s082 recipe)
SDIR="${OUT}/generation/sdedit_ll_intra"
if [[ ! -f "${SDIR}/generated/199.png" ]]; then
  "${PYTHON}" scripts/nda/generate_atm_aligned_decode.py \
    --mode sdedit --embed-npy "${EMB}" \
    --prompts-json "${HCMA_PROMPTS}" \
    --output-dir "${SDIR}" --tag "sdedit_ll_intra" \
    --lowlevel-rgb-dir "${LL_RGB}" \
    --strength 0.82 --ip-scale 1.0 --gen-steps 28 --gen-guidance 5.0 --seed 42
else
  echo "[SKIP] sdedit_ll_intra"
fi

# HCMA-S dual decode grid (Depth-CN x LL-SDEdit) over pure-intra towers
run_dual() {
  local tag="$1" cn="$2" strength="$3"
  local gdir="${OUT}/generation/${tag}"
  [[ -f "${gdir}/generated/199.png" ]] && echo "[SKIP] ${tag}" && return 0
  "${PYTHON}" scripts/nda/generate_hcma_s_decode.py \
    --embed-npy "${EMB}" \
    --prompts-json "${HCMA_PROMPTS}" \
    --depth-rgb-dir "${DEPTH_RGB}" \
    --lowlevel-rgb-dir "${LL_RGB}" \
    --output-dir "${gdir}" --tag "${tag}" \
    --cn-scale "${cn}" --ip-scale 1.0 --strength "${strength}" \
    --gen-steps 28 --gen-guidance 5.0 --seed 42
}
for cn in 0.25 0.32 0.40; do
  for s in 0.82 0.86 0.88; do
    ctag=$(echo "$cn" | tr -d .); stag=$(echo "$s" | tr -d .)
    run_dual "intra_hs_c${ctag}_s${stag}" "$cn" "$s"
  done
done

echo "===== [11] standard-7 (incl. per-row FID) @ $(date -Iseconds) ====="
STD7="${NB_ROOT}/outputs/standard7_protocol"
# Backup the shared results.json BEFORE eval (eval_standard7 overwrites it wholesale).
cp -f "${STD7}/results.json" "${OUT}/results_std7_backup.json" || true
# Build manifest for this experiment (eval only intra rows; refs cached in standard7)
MAN="${OUT}/manifest_intra_hcma_s.json"
OUT_EVAL="${OUT}" MAN="${MAN}" "${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT_EVAL"])
man = {"protocol": "standard7", "rows": [], "avg_rows": []}
def add(tag, display, gdir):
    man["rows"].append({"tag": tag, "display": display, "gen_dir": gdir})
add("intra_sdedit_ll_s082", "intra sdedit_ll sub-08 (pure)", str(out / "generation/sdedit_ll_intra/generated"))
for cn in ("025","032","040"):
    for s in ("082","086","088"):
        t = f"intra_hs_c{cn}_s{s}"
        d = out / "generation" / t / "generated"
        if (d / "199.png").exists():
            add(t, t + " (pure)", str(d))
Path(os.environ["MAN"]).write_text(json.dumps(man, indent=2), encoding="utf-8")
print("[OK] manifest rows", len(man["rows"]))
PY
"${PYTHON}" scripts/nda/eval_standard7.py \
  --manifest "${MAN}" --images-root "${IMAGES_ROOT}" \
  --out-dir "${STD7}" --device "${DEVICE}" --batch-size 16
# stash intra-only results, then merge into the pre-existing full results.json
cp -f "${STD7}/results.json" "${OUT}/results_intra.json"
"${PYTHON}" - <<PY
import json
from pathlib import Path
backup = json.loads(Path("${OUT}/results_std7_backup.json").read_text(encoding="utf-8"))
intra = json.loads(Path("${OUT}/results_intra.json").read_text(encoding="utf-8"))
by_tag = {r["tag"]: r for r in backup["rows"]}
for r in intra["rows"]:
    by_tag[r["tag"]] = r          # add/replace intra rows only
merged = dict(backup)
merged["rows"] = [by_tag[t] for t in by_tag]   # stable dedup (insertion order kept)
Path("${STD7}/results.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
print(f"[OK] merged results.json: {len(backup['rows'])} -> {len(merged['rows'])} rows")
PY

echo "===== [12] summary @ $(date -Iseconds) ====="
"${PYTHON}" - <<PY
import json
from pathlib import Path
out = Path("/project/peilab/why/NeuroBridge/outputs/intra_hcma_s/sub-08")
std7 = Path("/project/peilab/why/NeuroBridge/outputs/standard7_protocol/results.json")
res = json.loads(std7.read_text(encoding="utf-8"))["rows"] if std7.is_file() else []
by = {r["tag"]: r for r in res}
want = [r["tag"] for r in json.loads((out / "manifest_intra_hcma_s.json").read_text(encoding="utf-8"))["rows"]]
rows = [by[t] for t in want if t in by]
summary = {
  "pipeline": "intra_hcma_s_sub08",
  "strict": "EEG weights trained on sub-08 only (16540); image-side pretrained reused",
  "excluded": ["SharedSpecificEncoder 9subj","RGT bank","hcma_10subj MG-Flow","nda_ss SS features"],
  "semantic_embed": "NDA-v2 dual (sub-08) z_decode_vith/RAG blend -> IP",
  "structure": "VAE-LL + Depth-CN heads trained on pure-intra z_decode_vith",
  "rows": rows,
}
(out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

du -sh "${OUT}" 2>/dev/null || true
echo "{\"pipeline\":\"intra_hcma_s_sub08\",\"finished\":\"$(date -Iseconds)\"}" > "${OUT}/job_done.json"
echo "===== DONE intra_hcma_s sub08 @ $(date -Iseconds) ====="
