#!/usr/bin/env bash
# MG-Flow: multi-granular semantic alignment + gated hierarchical CFM → generation
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
OUT="${OUT:-${NB_ROOT}/outputs/mg_flow/sub-08}"
NDA_SS="${NDA_SS:-${NB_ROOT}/outputs/nda_ss/sub-08}"
RGT_BANK="${RGT_BANK:-${NB_ROOT}/outputs/rgt_cfm/sub-08/bank}"
T1="${T1:-${NB_ROOT}/outputs/top1_structure/sub-08}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

CLIP_TRAIN="${CLIP_TRAIN:-${BRAINIT}/outputs/atm_bridge/clip_img_train_1024.npy}"
CLIP_TEST="${CLIP_TEST:-${BRAINIT}/outputs/atm_bridge/clip_img_test_1024.npy}"
# fallback to nda_ss galleries if atm_bridge missing
if [[ ! -f "${CLIP_TRAIN}" ]]; then
  CLIP_TRAIN="${NDA_SS}/train/decode_vith1024_train_clip_1024.npy"
  CLIP_TEST="${NDA_SS}/train/decode_vith1024_test_clip_1024.npy"
fi

mkdir -p "${OUT}/targets" "${OUT}/train" "${OUT}/generation" "${OUT}/compare" "${OUT}/logs"
cd "${NB_ROOT}"

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"

echo "===== [0] Prefetch OpenCLIP (local cache) @ $(date -Iseconds) ====="
"${PYTHON}" - <<'PY'
import os
os.environ.setdefault("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub")
os.environ.setdefault("OPENCLIP_CACHE_DIR", "/project/peilab/why/cache/eeg-brainit/open_clip")
import open_clip
m, _, _ = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k", device="cpu")
print("[OK] ViT-H ready")
PY

echo "===== [1] Build coarse/fine targets @ $(date -Iseconds) ====="
if [[ ! -f "${OUT}/targets/t_fine_train.npy" ]]; then
  "${PYTHON}" scripts/nda/build_mg_flow_targets.py \
    --clip-text-root "${NDA_SS}/clip_text" \
    --output-dir "${OUT}/targets"
else
  echo "[SKIP] targets"
fi

echo "===== [2] Train MG-Flow @ $(date -Iseconds) ====="
if [[ ! -f "${OUT}/train/mg_flow_train_report.json" ]]; then
  "${PYTHON}" scripts/nda/mg_flow_train.py \
    --z-ret-train "${RGT_BANK}/z_ret_sub08_train.npy" \
    --z-ret-test "${RGT_BANK}/z_ret_sub08_test.npy" \
    --clip-img-train "${CLIP_TRAIN}" \
    --clip-img-test "${CLIP_TEST}" \
    --t-coarse-train "${OUT}/targets/t_coarse_train.npy" \
    --t-fine-train "${OUT}/targets/t_fine_train.npy" \
    --t-coarse-test "${OUT}/targets/t_coarse_test.npy" \
    --t-fine-test "${OUT}/targets/t_fine_test.npy" \
    --nda-decode-train "${NDA_SS}/train/z_decode_vith_train.npy" \
    --nda-decode-test "${NDA_SS}/train/z_decode_vith_test.npy" \
    --text-concept-test "${NDA_SS}/clip_text/test/text_concept_clip.npy" \
    --output-dir "${OUT}/train" \
    --epochs 40 \
    --batch-size 512 \
    --ode-steps 16 \
    --device "${DEVICE}"
else
  echo "[SKIP] train"
fi

# assets for generation
EMB_NDA="${NDA_SS}/blend/mem_decode_a50.npy"
EMB_GATED="${OUT}/train/embeds/z_mg_gated_test.npy"
EMB_A40="${OUT}/train/embeds/blend_nda_cfm_f_a40_test.npy"
NEIGH="${NDA_SS}/memory/rag_soft5_neighbor_idx_test.npy"
PRED_DEPTH="${T1}/track_s/depth_head/pred_depth_rgb_512"
CN_COCA="${T1}/track_s/depth_head/cn_scale_coca.npy"
if [[ ! -d "${PRED_DEPTH}" ]]; then
  PRED_DEPTH="${NB_ROOT}/outputs/overnight_ablation/sub-08/depth_vith/pred_depth_rgb_512"
fi
if [[ ! -f "${CN_COCA}" ]]; then
  CN_COCA="${NB_ROOT}/outputs/overnight_ablation/sub-08/depth_vith/cn_scale_coca.npy"
fi
PROMPT_DUAL="${OUT}/targets/prompts_dual_test.json"
PROMPT_CPA="${T1}/prompts/prompts_cpa.json"

test -f "${EMB_NDA}" && test -f "${EMB_GATED}"
test -d "${PRED_DEPTH}" || echo "[WARN] missing pred depth; generation may fail"
test -f "${NEIGH}"

echo "===== [3] Generation @ $(date -Iseconds) ====="
run_gen() {
  local tag="$1" emb="$2" prompt="$3"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; return 0; fi
  local extra=()
  if [[ -f "${CN_COCA}" ]]; then extra+=(--cn-scale-npy "${CN_COCA}"); fi
  "${PYTHON}" scripts/nda/generate_cn_ip_decode.py \
    --embed-npy "${emb}" \
    --neighbor-idx-npy "${NEIGH}" \
    --output-dir "${gdir}" \
    --tag "${tag}" \
    --seed 42 \
    --control-type depth \
    --depth-sample-dir "${PRED_DEPTH}" \
    --cn-scale 0.5 \
    --ip-scale 1.0 \
    --gen-steps 30 \
    --gen-guidance 5.0 \
    --prompts-json "${prompt}" \
    --skip-metrics \
    "${extra[@]}"
}

# baselines / ablations
run_gen "ref_sem_coca_cpa" "${EMB_NDA}" "${PROMPT_CPA}"
run_gen "mg_gated_dual" "${EMB_GATED}" "${PROMPT_DUAL}"
run_gen "mg_blend_a40_dual" "${EMB_A40}" "${PROMPT_DUAL}"
run_gen "mg_gated_cpa" "${EMB_GATED}" "${PROMPT_CPA}"

echo "===== [4] Paper metrics @ $(date -Iseconds) ====="
TAGS="ref_sem_coca_cpa,mg_gated_dual,mg_blend_a40_dual,mg_gated_cpa"
VALID=""
IFS=',' read -ra ARR <<< "${TAGS}"
for t in "${ARR[@]}"; do
  [[ -f "${OUT}/generation/${t}/generated/199.png" ]] && VALID="${VALID:+$VALID,}${t}"
done
"${PYTHON}" scripts/nda/eval_paper_metrics.py \
  --gen-root "${OUT}/generation" \
  --tags "${VALID}" \
  --output-json "${OUT}/paper_metrics.json"

echo "===== [5] CLIP 2-way @ $(date -Iseconds) ====="
GEN_ARGS=""
IFS=',' read -ra ARR <<< "${VALID}"
for t in "${ARR[@]}"; do
  GEN_ARGS="${GEN_ARGS:+$GEN_ARGS,}${t}=${OUT}/generation/${t}/generated"
done
"${PYTHON}" scripts/nda/eval_clip_2way.py \
  --gen-dirs "${GEN_ARGS}" \
  --output-json "${OUT}/clip_2way_report.json" \
  --device "${DEVICE}" \
  --batch-size 16

echo "===== [6] Class consistency @ $(date -Iseconds) ====="
"${PYTHON}" scripts/nda/eval_class_consistency.py \
  --gen-dirs "${GEN_ARGS}" \
  --text-concept-npy "${NDA_SS}/clip_text/test/text_concept_clip.npy" \
  --concepts-json "${OUT}/targets/concepts_test.json" \
  --output-json "${OUT}/class_consistency.json" \
  --device "${DEVICE}"

echo "===== [7] Compare grid @ $(date -Iseconds) ====="
SEM_REF="${T1}/generation/pred_coca_ip1.0_cpa/generated"
COLS="sem=${SEM_REF}"
for pair in "ref=${OUT}/generation/ref_sem_coca_cpa/generated" \
            "gated=${OUT}/generation/mg_gated_dual/generated" \
            "a40=${OUT}/generation/mg_blend_a40_dual/generated" \
            "gated_cpa=${OUT}/generation/mg_gated_cpa/generated"; do
  name="${pair%%=*}"; path="${pair#*=}"
  [[ -f "${path}/000.png" ]] && COLS="${COLS},${name}=${path}"
done
"${PYTHON}" scripts/nda/make_compare_grid.py \
  --output-dir "${OUT}/compare" \
  --cell 160 \
  --indices "3,12,28,45,67,88,110,133,156,178,190,199" \
  --metrics-json "${OUT}/clip_2way_report.json" \
  --cols "${COLS}"

echo "===== [8] Summary @ $(date -Iseconds) ====="
"${PYTHON}" - <<'PY'
import json
from pathlib import Path
out = Path("/project/peilab/why/NeuroBridge/outputs/mg_flow/sub-08")
paper = json.loads((out/"paper_metrics.json").read_text()) if (out/"paper_metrics.json").is_file() else {"results":[]}
tw = json.loads((out/"clip_2way_report.json").read_text()) if (out/"clip_2way_report.json").is_file() else {"generation_2way":[]}
cls = json.loads((out/"class_consistency.json").read_text()) if (out/"class_consistency.json").is_file() else {"results":[]}
train = json.loads((out/"train/mg_flow_train_report.json").read_text()) if (out/"train/mg_flow_train_report.json").is_file() else {}
by_p = {r["tag"]: r for r in paper.get("results", [])}
by_2 = {r["tag"]: r for r in tw.get("generation_2way", [])}
by_c = {r["tag"]: r for r in cls.get("results", [])}
rows = []
for tag in sorted(set(by_p)|set(by_2)|set(by_c)):
    p, w, c = by_p.get(tag, {}), by_2.get(tag, {}), by_c.get(tag, {})
    twoway = float(w.get("clip_2way", 0))
    clstop = float(c.get("class_top1", 0))
    fid = float(p.get("fid", 999))
    ssim = float(p.get("ssim", 0))
    # semantic-first score
    score = 0.40*twoway + 0.40*clstop + 0.15*max(0,(320-fid)/170) + 0.05*ssim
    rows.append({"tag": tag, "clip_2way": twoway, "class_top1": clstop, "fid": fid, "ssim": ssim,
                 "pixcorr": p.get("pixcorr"), "clip_cosine": p.get("clip_cosine"), "score": score})
rows.sort(key=lambda x: -x["score"])
summary = {
  "pipeline": "MG-Flow",
  "claim": "Multi-granular semantic alignment + gated hierarchical CFM (no forced RGT transport)",
  "train_best": train.get("best"),
  "best_gen": rows[0] if rows else None,
  "all_ranked": rows,
  "compare_grid": str(out/"compare/compare_grid.png"),
  "targets": {"clip_2way": 0.90, "class_top1": 0.35, "fid": 160},
}
(out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

du -sh "${OUT}" 2>/dev/null || true
echo "===== DONE MG-Flow @ $(date -Iseconds) ====="
