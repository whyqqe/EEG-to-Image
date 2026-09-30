#!/usr/bin/env bash
# MG-Flow a40 on ALL THINGS-EEG subjects (1-10).
# Recipe: dual-granularity CFM fine ⊕ NDA/memory blend α=0.40 + dual prompt + Depth-CN(neighbor).
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
ROOT_OUT="${ROOT_OUT:-${NB_ROOT}/outputs/mg_flow_a40_all}"
SHARED="${SHARED:-${NB_ROOT}/outputs/mg_flow/sub-08}"
NDA_SS="${NDA_SS:-${NB_ROOT}/outputs/nda_ss/sub-08}"
RGT_BANK="${RGT_BANK:-${NB_ROOT}/outputs/rgt_cfm/sub-08/bank}"
DEPTH_CACHE="${DEPTH_CACHE:-${NB_ROOT}/outputs/coca_depth/sub-08/depth_cache}"
SS_CKPT="${SS_CKPT:-${NDA_SS}/ss/checkpoint_ss_calib_best.pth}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
SUBJECTS="${SUBJECTS:-1,2,3,4,5,6,7,8,9,10}"
ALPHA="${ALPHA:-0.40}"

CLIP_TRAIN="${CLIP_TRAIN:-${BRAINIT}/outputs/atm_bridge/clip_img_train_1024.npy}"
CLIP_TEST="${CLIP_TEST:-${BRAINIT}/outputs/atm_bridge/clip_img_test_1024.npy}"
if [[ ! -f "${CLIP_TRAIN}" ]]; then
  CLIP_TRAIN="${NDA_SS}/train/decode_vith1024_train_clip_1024.npy"
  CLIP_TEST="${NDA_SS}/train/decode_vith1024_test_clip_1024.npy"
fi

mkdir -p "${ROOT_OUT}" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"

echo "===== [0] Prefetch + shared targets @ $(date -Iseconds) ====="
"${PYTHON}" - <<'PY'
import os
os.environ.setdefault("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub")
os.environ.setdefault("OPENCLIP_CACHE_DIR", "/project/peilab/why/cache/eeg-brainit/open_clip")
import open_clip
open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k", device="cpu")
open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device="cpu")
print("[OK] OpenCLIP ready")
PY

TARGETS="${ROOT_OUT}/shared_targets"
if [[ ! -f "${TARGETS}/t_fine_train.npy" ]]; then
  if [[ -f "${SHARED}/targets/t_fine_train.npy" ]]; then
    mkdir -p "${TARGETS}"
    cp -a "${SHARED}/targets/." "${TARGETS}/"
    echo "[OK] reused shared targets from mg_flow/sub-08"
  else
    "${PYTHON}" scripts/nda/build_mg_flow_targets.py \
      --clip-text-root "${NDA_SS}/clip_text" \
      --output-dir "${TARGETS}"
  fi
fi
PROMPT_DUAL="${TARGETS}/prompts_dual_test.json"
test -f "${PROMPT_DUAL}"
test -d "${DEPTH_CACHE}"

IFS=',' read -ra SUBJ_ARR <<< "${SUBJECTS}"
ALL_ROWS=()

for SID in "${SUBJ_ARR[@]}"; do
  SID=$(echo "${SID}" | tr -d ' ')
  STAG=$(printf "sub-%02d" "${SID}")
  OUT="${ROOT_OUT}/${STAG}"
  mkdir -p "${OUT}/zret" "${OUT}/memory" "${OUT}/train" "${OUT}/generation" "${OUT}/compare"
  echo "########## ${STAG} @ $(date -Iseconds) ##########"

  # ---- z_ret ----
  ZTR="${RGT_BANK}/z_ret_sub$(printf '%02d' "${SID}")_train.npy"
  ZTE="${RGT_BANK}/z_ret_sub$(printf '%02d' "${SID}")_test.npy"
  if [[ ! -f "${ZTR}" || ! -f "${ZTE}" ]]; then
    echo "[INFO] missing bank z_ret for ${STAG}; encode via SS ckpt"
    test -f "${SS_CKPT}"
    "${PYTHON}" scripts/nda/nda_ss_encode.py \
      --checkpoint "${SS_CKPT}" \
      --subject "${SID}" \
      --output-dir "${OUT}/zret" \
      --device "${DEVICE}"
    # SS encode writes z_eeg_proj_*; treat as z_ret (same 512-d SSP space)
    cp -f "${OUT}/zret/z_eeg_proj_train.npy" "${OUT}/zret/z_ret_train.npy"
    cp -f "${OUT}/zret/z_eeg_proj_test.npy" "${OUT}/zret/z_ret_test.npy"
    ZTR="${OUT}/zret/z_ret_train.npy"
    ZTE="${OUT}/zret/z_ret_test.npy"
  else
    mkdir -p "${OUT}/zret"
    cp -f "${ZTR}" "${OUT}/zret/z_ret_train.npy"
    cp -f "${ZTE}" "${OUT}/zret/z_ret_test.npy"
    # memory router expects z_eeg_proj_*
    cp -f "${OUT}/zret/z_ret_train.npy" "${OUT}/zret/z_eeg_proj_train.npy"
    cp -f "${OUT}/zret/z_ret_test.npy" "${OUT}/zret/z_eeg_proj_test.npy"
    ZTR="${OUT}/zret/z_ret_train.npy"
    ZTE="${OUT}/zret/z_ret_test.npy"
  fi
  # ensure proj names for memory
  [[ -f "${OUT}/zret/z_eeg_proj_train.npy" ]] || cp -f "${ZTR}" "${OUT}/zret/z_eeg_proj_train.npy"
  [[ -f "${OUT}/zret/z_eeg_proj_test.npy" ]] || cp -f "${ZTE}" "${OUT}/zret/z_eeg_proj_test.npy"

  # ---- memory (NDA proxy + neighbors) ----
  if [[ ! -f "${OUT}/memory/rag_soft5_test_clip_1024.npy" ]]; then
    "${PYTHON}" scripts/nmb/nmb_memory_router.py \
      --embed-dir "${OUT}/zret" \
      --clip-train "${CLIP_TRAIN}" \
      --clip-test "${CLIP_TEST}" \
      --output-dir "${OUT}/memory" \
      --input-key proj --soft-k 5 --soft-tau 0.07
  fi
  NEIGH="${OUT}/memory/rag_soft5_neighbor_idx_test.npy"
  MEM_TR="${OUT}/memory/rag_soft5_train_clip_1024.npy"
  MEM_TE="${OUT}/memory/rag_soft5_test_clip_1024.npy"

  # NDA decode base: prefer true decode for sub-08, else memory
  if [[ "${SID}" == "8" && -f "${NDA_SS}/train/z_decode_vith_train.npy" ]]; then
    NDA_TR="${NDA_SS}/train/z_decode_vith_train.npy"
    NDA_TE="${NDA_SS}/train/z_decode_vith_test.npy"
    NDA_SRC="z_decode_vith"
  else
    NDA_TR="${MEM_TR}"
    NDA_TE="${MEM_TE}"
    NDA_SRC="memory_soft5"
  fi

  # ---- train MG-Flow (skip if a40 embed exists) ----
  EMB_A40="${OUT}/train/embeds/blend_nda_cfm_f_a40_test.npy"
  # reuse sub-08 previous MG-Flow a40 embeds if available
  if [[ "${SID}" == "8" && ! -f "${EMB_A40}" && -f "${SHARED}/train/embeds/blend_nda_cfm_f_a40_test.npy" ]]; then
    mkdir -p "${OUT}/train/embeds"
    cp -a "${SHARED}/train/embeds/." "${OUT}/train/embeds/"
    [[ -f "${SHARED}/train/mg_flow_train_report.json" ]] && cp -f "${SHARED}/train/mg_flow_train_report.json" "${OUT}/train/"
    echo "[OK] reused sub-08 MG-Flow embeds from prior run"
  fi
  if [[ ! -f "${EMB_A40}" ]]; then
    "${PYTHON}" scripts/nda/mg_flow_train.py \
      --z-ret-train "${ZTR}" \
      --z-ret-test "${ZTE}" \
      --clip-img-train "${CLIP_TRAIN}" \
      --clip-img-test "${CLIP_TEST}" \
      --t-coarse-train "${TARGETS}/t_coarse_train.npy" \
      --t-fine-train "${TARGETS}/t_fine_train.npy" \
      --t-coarse-test "${TARGETS}/t_coarse_test.npy" \
      --t-fine-test "${TARGETS}/t_fine_test.npy" \
      --nda-decode-train "${NDA_TR}" \
      --nda-decode-test "${NDA_TE}" \
      --text-concept-test "${NDA_SS}/clip_text/test/text_concept_clip.npy" \
      --output-dir "${OUT}/train" \
      --epochs 40 \
      --batch-size 512 \
      --ode-steps 16 \
      --device "${DEVICE}"
  else
    echo "[SKIP] train ${STAG}"
  fi
  test -f "${EMB_A40}"

  # ---- generate a40 only ----
  TAG="mg_blend_a40_dual"
  GDIR="${OUT}/generation/${TAG}"
  if [[ ! -f "${GDIR}/generated/199.png" ]]; then
    "${PYTHON}" scripts/nda/generate_cn_ip_decode.py \
      --embed-npy "${EMB_A40}" \
      --neighbor-idx-npy "${NEIGH}" \
      --output-dir "${GDIR}" \
      --tag "${TAG}" \
      --seed 42 \
      --control-type depth \
      --depth-cache-dir "${DEPTH_CACHE}" \
      --cn-scale 0.5 \
      --ip-scale 1.0 \
      --gen-steps 30 \
      --gen-guidance 5.0 \
      --prompts-json "${PROMPT_DUAL}" \
      --skip-metrics
  else
    echo "[SKIP] gen ${STAG}"
  fi

  # ---- metrics ----
  "${PYTHON}" scripts/nda/eval_paper_metrics.py \
    --gen-root "${OUT}/generation" \
    --tags "${TAG}" \
    --output-json "${OUT}/paper_metrics.json"

  "${PYTHON}" scripts/nda/eval_clip_2way.py \
    --gen-dirs "${TAG}=${OUT}/generation/${TAG}/generated" \
    --output-json "${OUT}/clip_2way_report.json" \
    --device "${DEVICE}" --batch-size 16

  "${PYTHON}" scripts/nda/eval_class_consistency.py \
    --gen-dirs "${TAG}=${OUT}/generation/${TAG}/generated" \
    --text-concept-npy "${NDA_SS}/clip_text/test/text_concept_clip.npy" \
    --concepts-json "${TARGETS}/concepts_test.json" \
    --output-json "${OUT}/class_consistency.json" \
    --device "${DEVICE}"

  # ---- compare: auto-select semantically closest rows ----
  "${PYTHON}" scripts/nda/make_compare_grid.py \
    --output-dir "${OUT}/compare" \
    --cell 168 \
    --auto-select-gen-dir "${OUT}/generation/${TAG}/generated" \
    --auto-select-k 12 \
    --device "${DEVICE}" \
    --metrics-json "${OUT}/clip_2way_report.json" \
    --cols "a40=${OUT}/generation/${TAG}/generated"

  # per-subject summary
  "${PYTHON}" - <<PY
import json
from pathlib import Path
out = Path("${OUT}")
paper = json.loads((out/"paper_metrics.json").read_text())["results"][0]
tw = json.loads((out/"clip_2way_report.json").read_text())["generation_2way"][0]
cls = json.loads((out/"class_consistency.json").read_text())["results"][0]
cmp = json.loads((out/"compare/compare_report.json").read_text())
summary = {
  "subject": ${SID},
  "tag": "mg_blend_a40_dual",
  "nda_src": "${NDA_SRC}",
  "clip_2way": tw.get("clip_2way"),
  "class_top1": cls.get("class_top1"),
  "class_top5": cls.get("class_top5"),
  "fid": paper.get("fid"),
  "ssim": paper.get("ssim"),
  "pixcorr": paper.get("pixcorr"),
  "clip_cosine": paper.get("clip_cosine"),
  "compare_indices": cmp.get("indices"),
  "compare_scores": cmp.get("scores"),
  "compare_grid": cmp.get("grid"),
}
(out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
done

echo "===== Aggregate all subjects @ $(date -Iseconds) ====="
"${PYTHON}" - <<'PY'
import json
from pathlib import Path
import numpy as np
root = Path("/project/peilab/why/NeuroBridge/outputs/mg_flow_a40_all")
rows = []
for p in sorted(root.glob("sub-*/summary.json")):
    rows.append(json.loads(p.read_text()))
rows.sort(key=lambda r: r["subject"])
def mean(key):
    xs = [float(r[key]) for r in rows if r.get(key) is not None]
    return float(np.mean(xs)) if xs else None
agg = {
  "pipeline": "MG-Flow a40 all-subjects",
  "tag": "mg_blend_a40_dual",
  "n_subjects": len(rows),
  "subjects": [r["subject"] for r in rows],
  "mean": {
    "clip_2way": mean("clip_2way"),
    "class_top1": mean("class_top1"),
    "class_top5": mean("class_top5"),
    "fid": mean("fid"),
    "ssim": mean("ssim"),
    "pixcorr": mean("pixcorr"),
    "clip_cosine": mean("clip_cosine"),
  },
  "per_subject": rows,
}
(root/"summary_all.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
print(json.dumps(agg, indent=2))
PY

# global compare board: pick top semantic from sub-08 a40 if present, else first subject
REF_SUB=$(printf "sub-%02d" "${SUBJ_ARR[0]}")
if [[ -f "${ROOT_OUT}/sub-08/generation/mg_blend_a40_dual/generated/199.png" ]]; then
  REF_SUB=sub-08
fi
echo "===== Global semantic-best compare (@${REF_SUB}) ====="
COLS=""
for SID in "${SUBJ_ARR[@]}"; do
  SID=$(echo "${SID}" | tr -d ' ')
  STAG=$(printf "sub-%02d" "${SID}")
  G="${ROOT_OUT}/${STAG}/generation/mg_blend_a40_dual/generated"
  if [[ -f "${G}/000.png" ]]; then
    COLS="${COLS:+$COLS,}s$(printf '%02d' "${SID}")=${G}"
  fi
done
# too many cols if 10 subjects — make two grids: s01-s05 and s06-s10 plus a compact GT|best-per-subj for shared indices from REF_SUB
mkdir -p "${ROOT_OUT}/compare"
IDX_JSON="${ROOT_OUT}/${REF_SUB}/compare/compare_report.json"
IDXS=$(${PYTHON} -c "import json; print(','.join(map(str,json.load(open('${IDX_JSON}'))['indices'])))")

# Grid A: subjects 1-5
COLS_A=""
for SID in 1 2 3 4 5; do
  G="${ROOT_OUT}/sub-$(printf '%02d' ${SID})/generation/mg_blend_a40_dual/generated"
  [[ -f "${G}/000.png" ]] && COLS_A="${COLS_A:+$COLS_A,}s$(printf '%02d' ${SID})=${G}"
done
if [[ -n "${COLS_A}" ]]; then
  "${PYTHON}" scripts/nda/make_compare_grid.py \
    --output-dir "${ROOT_OUT}/compare/group_s01_s05" \
    --cell 140 --indices "${IDXS}" \
    --cols "${COLS_A}"
fi
# Grid B: subjects 6-10
COLS_B=""
for SID in 6 7 8 9 10; do
  G="${ROOT_OUT}/sub-$(printf '%02d' ${SID})/generation/mg_blend_a40_dual/generated"
  [[ -f "${G}/000.png" ]] && COLS_B="${COLS_B:+$COLS_B,}s$(printf '%02d' ${SID})=${G}"
done
if [[ -n "${COLS_B}" ]]; then
  "${PYTHON}" scripts/nda/make_compare_grid.py \
    --output-dir "${ROOT_OUT}/compare/group_s06_s10" \
    --cell 140 --indices "${IDXS}" \
    --cols "${COLS_B}"
fi

du -sh "${ROOT_OUT}" 2>/dev/null || true
echo "===== DONE MG-Flow a40 ALL @ $(date -Iseconds) ====="
