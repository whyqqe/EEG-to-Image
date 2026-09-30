#!/usr/bin/env bash
# Top-1 (clean dual) × Structure (EEG→Depth + COCA) full training + decode experiment
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/top1_structure/sub-08}"
NDA_SS="${NDA_SS:-${NB_ROOT}/outputs/nda_ss/sub-08}"
COCA_PREV="${COCA_PREV:-${NB_ROOT}/outputs/coca_depth/sub-08}"
V2="${V2:-${NB_ROOT}/outputs/oracle_chase_v2/sub-08}"
NB_CKPT="${NB_CKPT:-${NB_ROOT}/results/things_eeg/intra-subjects/20260901-074319-sub-08/checkpoint_test_best.pth}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

T_OUT="${OUT}/track_t"
S_OUT="${OUT}/track_s"
GEN_OUT="${OUT}/generation"
RET_OUT="${OUT}/retrieval"

mkdir -p "${T_OUT}" "${S_OUT}" "${GEN_OUT}" "${RET_OUT}/clean" "${RET_OUT}/cpa" "${OUT}/prompts"
cd "${NB_ROOT}"

echo "===== [0] Prefetch models @ $(date -Iseconds) ====="
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
"${PYTHON}" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
hub=Path(os.environ["HF_HUB_CACHE"])
for repo in [
  "stabilityai/stable-diffusion-xl-base-1.0",
  "diffusers/controlnet-depth-sdxl-1.0",
  "depth-anything/Depth-Anything-V2-Small-hf",
]:
    print("[prefetch]", repo)
    snapshot_download(repo_id=repo, cache_dir=str(hub))
print("[OK] prefetch")
PY

PHRASES="${NB_ROOT}/outputs/nda_v2_semtxt/sub-08/clip_text/test/concept_phrases.json"
test -f "${NB_CKPT}"
test -f "${PHRASES}"
test -f "${NDA_SS}/blend/mem_decode_a50.npy"
test -f "${NDA_SS}/memory/rag_soft5_neighbor_idx_test.npy"
cp -f "${PHRASES}" "${OUT}/prompts/concept_phrases.json"
if [[ -f "${V2}/prompts/prompts_pred.json" ]]; then
  cp -f "${V2}/prompts/prompts_pred.json" "${OUT}/prompts/prompts_cpa.json"
elif [[ -f "${COCA_PREV}/prompts/prompts_pred.json" ]]; then
  cp -f "${COCA_PREV}/prompts/prompts_pred.json" "${OUT}/prompts/prompts_cpa.json"
else
  echo "[WARN] no CPA prompts; will build later"
fi

EMB_IP="${NDA_SS}/blend/mem_decode_a50.npy"
NEIGH="${NDA_SS}/memory/rag_soft5_neighbor_idx_test.npy"
HCF_TRAIN="${NDA_SS}/train/hcf_train.npy"
HCF_TEST="${NDA_SS}/train/hcf_test.npy"

# ---------- Track T: dual clean+CPA fine-tune ----------
echo "===== [T1] Dual clean+CPA fine-tune @ $(date -Iseconds) ====="
T_CKPT="${T_OUT}/checkpoint_clean_dual_best.pth"
if [[ ! -f "${T_OUT}/clean_dual_report.json" ]]; then
  MID_ARGS=()
  if [[ -f "${HCF_TRAIN}" && -f "${HCF_TEST}" ]]; then
    MID_ARGS=(--lambda-mid 0.25 --hcf-train "${HCF_TRAIN}" --hcf-test "${HCF_TEST}")
  else
    MID_ARGS=(--lambda-mid 0.0)
  fi
  "${PYTHON}" scripts/nda/nda_clean_dual_finetune.py \
    --init-checkpoint "${NB_CKPT}" \
    --output-dir "${T_OUT}" \
    --subject 8 \
    --num-epochs 40 \
    --batch-size 512 \
    --lr 3e-5 \
    --lambda-clean 1.0 \
    --lambda-cpa 0.5 \
    --device "${DEVICE}" \
    "${MID_ARGS[@]}"
else
  echo "[SKIP] Track T finetune"
fi
test -f "${T_CKPT}"

echo "===== [T0] Dual gallery retrieval (baseline NB + finetuned) ====="
"${PYTHON}" scripts/nda/chase_oracle_prompts.py \
  --nb-ckpt "${NB_CKPT}" --gallery clean \
  --concept-phrases-test "${OUT}/prompts/concept_phrases.json" \
  --output-dir "${RET_OUT}/baseline_clean" --device "${DEVICE}" --top-k 1
"${PYTHON}" scripts/nda/chase_oracle_prompts.py \
  --nb-ckpt "${NB_CKPT}" --gallery cpa \
  --concept-phrases-test "${OUT}/prompts/concept_phrases.json" \
  --output-dir "${RET_OUT}/baseline_cpa" --device "${DEVICE}" --top-k 1
"${PYTHON}" scripts/nda/chase_oracle_prompts.py \
  --nb-ckpt "${T_CKPT}" --gallery clean \
  --concept-phrases-test "${OUT}/prompts/concept_phrases.json" \
  --output-dir "${RET_OUT}/clean" --device "${DEVICE}" --top-k 1
"${PYTHON}" scripts/nda/chase_oracle_prompts.py \
  --nb-ckpt "${T_CKPT}" --gallery cpa \
  --concept-phrases-test "${OUT}/prompts/concept_phrases.json" \
  --output-dir "${RET_OUT}/cpa" --device "${DEVICE}" --top-k 1

# clean prompts for T2 ablation
cp -f "${RET_OUT}/clean/prompts_pred.json" "${OUT}/prompts/prompts_clean.json"
if [[ ! -f "${OUT}/prompts/prompts_cpa.json" ]]; then
  cp -f "${RET_OUT}/cpa/prompts_pred.json" "${OUT}/prompts/prompts_cpa.json"
fi
PROMPT_CPA="${OUT}/prompts/prompts_cpa.json"
PROMPT_CLEAN="${OUT}/prompts/prompts_clean.json"

# ---------- Track S: GT depth + EEG→Depth ----------
echo "===== [S1] GT DepthAnything cache @ $(date -Iseconds) ====="
DEPTH_CACHE="${S_OUT}/gt_depth"
if [[ ! -f "${DEPTH_CACHE}/train_depth_64.npy" || ! -f "${DEPTH_CACHE}/test_depth_64.npy" ]]; then
  "${PYTHON}" scripts/nda/build_gt_depth_cache.py \
    --output-dir "${DEPTH_CACHE}" \
    --device "${DEVICE}" \
    --low-res 64 \
    --rgb-size 512 \
    --batch-size 8 \
    --splits "train,test"
else
  echo "[SKIP] GT depth cache"
fi

echo "===== [S2] Train EEG→Depth head @ $(date -Iseconds) ====="
# Prefer dual-finetune embeds if present; else NDA-SS embeds
EEG_TR="${T_OUT}/embeds/z_eeg_proj_train.npy"
EEG_TE="${T_OUT}/embeds/z_eeg_proj_test.npy"
if [[ ! -f "${EEG_TR}" ]]; then
  EEG_TR="${NDA_SS}/embeds/z_eeg_proj_train.npy"
  EEG_TE="${NDA_SS}/embeds/z_eeg_proj_test.npy"
fi
DEPTH_HEAD_OUT="${S_OUT}/depth_head"
if [[ ! -f "${DEPTH_HEAD_OUT}/depth_head_report.json" ]]; then
  "${PYTHON}" scripts/nda/train_eeg_depth_head.py \
    --eeg-train-npy "${EEG_TR}" \
    --eeg-test-npy "${EEG_TE}" \
    --depth-train-npy "${DEPTH_CACHE}/train_depth_64.npy" \
    --depth-test-npy "${DEPTH_CACHE}/test_depth_64.npy" \
    --output-dir "${DEPTH_HEAD_OUT}" \
    --num-epochs 60 \
    --batch-size 256 \
    --lr 1e-3 \
    --device "${DEVICE}"
else
  echo "[SKIP] DepthHead train"
fi
PRED_DEPTH="${DEPTH_HEAD_OUT}/pred_depth_rgb_512"
CN_COCA="${DEPTH_HEAD_OUT}/cn_scale_coca.npy"
test -d "${PRED_DEPTH}"
test -f "${CN_COCA}"

# ---------- Merge: generate with predicted depth + COCA ----------
echo "===== [MERGE] Generation ablations @ $(date -Iseconds) ====="
run_gen() {
  local tag="$1"; shift
  local gdir="${GEN_OUT}/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then
    echo "[SKIP] ${tag}"
    return 0
  fi
  "${PYTHON}" scripts/nda/generate_cn_ip_decode.py \
    --embed-npy "${EMB_IP}" \
    --neighbor-idx-npy "${NEIGH}" \
    --output-dir "${gdir}" \
    --tag "${tag}" \
    --seed 42 \
    --control-type depth \
    --gen-steps 30 \
    --gen-guidance 5.0 \
    --skip-metrics \
    "$@"
}

# ref: previous neighbor-depth champion (copy if available)
REF=ref_neighbor_depth_cn0.5_ip1.0_cpa
if [[ ! -f "${GEN_OUT}/${REF}/generated/199.png" ]]; then
  if [[ -f "${COCA_PREV}/generation/depth_cn0.5_ip1.0_cpa/generated/199.png" ]]; then
    mkdir -p "${GEN_OUT}/${REF}"
    cp -a "${COCA_PREV}/generation/depth_cn0.5_ip1.0_cpa/generated" "${GEN_OUT}/${REF}/"
    echo "[OK] copied neighbor-depth champion as reference"
  fi
fi

# A: EEG-pred depth + fixed cn + CPA text
run_gen "pred_cn0.5_ip1.0_cpa" \
  --depth-sample-dir "${PRED_DEPTH}" --cn-scale 0.5 --ip-scale 1.0 \
  --prompts-json "${PROMPT_CPA}"

# B: EEG-pred depth + COCA cn + CPA text
run_gen "pred_coca_ip1.0_cpa" \
  --depth-sample-dir "${PRED_DEPTH}" --cn-scale 0.5 --ip-scale 1.0 \
  --cn-scale-npy "${CN_COCA}" \
  --prompts-json "${PROMPT_CPA}"

# C: EEG-pred depth + COCA cn + CLEAN text (honest main-table prompt)
run_gen "pred_coca_ip1.0_clean" \
  --depth-sample-dir "${PRED_DEPTH}" --cn-scale 0.5 --ip-scale 1.0 \
  --cn-scale-npy "${CN_COCA}" \
  --prompts-json "${PROMPT_CLEAN}"

# D: EEG-pred depth + COCA, no text
run_gen "pred_coca_ip1.0" \
  --depth-sample-dir "${PRED_DEPTH}" --cn-scale 0.5 --ip-scale 1.0 \
  --cn-scale-npy "${CN_COCA}"

# E: oracle GT depth upper bound (CPA text)
if [[ -d "${DEPTH_CACHE}/test_rgb_512" ]]; then
  run_gen "oracle_gt_cn0.5_ip1.0_cpa" \
    --depth-sample-dir "${DEPTH_CACHE}/test_rgb_512" --cn-scale 0.5 --ip-scale 1.0 \
    --prompts-json "${PROMPT_CPA}"
fi

echo "===== [EVAL] Paper-grade metrics @ $(date -Iseconds) ====="
TAGS="${REF},pred_cn0.5_ip1.0_cpa,pred_coca_ip1.0_cpa,pred_coca_ip1.0_clean,pred_coca_ip1.0"
if [[ -f "${GEN_OUT}/oracle_gt_cn0.5_ip1.0_cpa/generated/199.png" ]]; then
  TAGS="${TAGS},oracle_gt_cn0.5_ip1.0_cpa"
fi
# drop missing ref from tags
VALID_TAGS=""
IFS=',' read -ra ARR <<< "${TAGS}"
for t in "${ARR[@]}"; do
  if [[ -f "${GEN_OUT}/${t}/generated/199.png" ]]; then
    VALID_TAGS="${VALID_TAGS:+$VALID_TAGS,}${t}"
  fi
done

"${PYTHON}" scripts/nda/eval_paper_metrics.py \
  --gen-root "${GEN_OUT}" \
  --tags "${VALID_TAGS}" \
  --output-json "${OUT}/paper_metrics.json"

"${PYTHON}" - <<PY
import json
from pathlib import Path
out = Path("${OUT}")

def load_prompt_report(p):
    f = Path(p) / "prompt_report.json"
    return json.loads(f.read_text()) if f.is_file() else {}

metrics = json.loads((out / "paper_metrics.json").read_text()) if (out / "paper_metrics.json").is_file() else {}
results = metrics.get("results", [])
by = {r["tag"]: r for r in results}
t_rep = json.loads((out / "track_t/clean_dual_report.json").read_text()) if (out / "track_t/clean_dual_report.json").is_file() else {}
s_rep = json.loads((out / "track_s/depth_head/depth_head_report.json").read_text()) if (out / "track_s/depth_head/depth_head_report.json").is_file() else {}
base_c = load_prompt_report(out / "retrieval/baseline_clean")
base_a = load_prompt_report(out / "retrieval/baseline_cpa")
ft_c = load_prompt_report(out / "retrieval/clean")
ft_a = load_prompt_report(out / "retrieval/cpa")
best_ssim = max(results, key=lambda r: r.get("ssim", -1), default=None)
best_clip = max(results, key=lambda r: r.get("clip_cosine", -1), default=None)
summary = {
  "pipeline": "TOP1_STRUCTURE_dual_track",
  "plan": "docs/TOP1_STRUCTURE_PLAN.md",
  "track_t": {
    "baseline_clean_top1": base_c.get("concept_top1"),
    "baseline_cpa_top1": base_a.get("concept_top1"),
    "finetune_clean_top1": ft_c.get("concept_top1"),
    "finetune_cpa_top1": ft_a.get("concept_top1"),
    "train_report": {
      "best_epoch": t_rep.get("best_epoch"),
      "best_top1_clean": t_rep.get("best_top1_clean"),
      "best_top1_cpa": t_rep.get("best_top1_cpa"),
      "baseline": t_rep.get("baseline"),
    },
  },
  "track_s": s_rep,
  "best_ssim": best_ssim,
  "best_clip": best_clip,
  "all_gen": results,
  "success_lines": {
    "clean_top1": ">=0.50",
    "ssim_skimage": ">=0.28",
    "clip_floor": 0.46,
  },
}
(out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "===== DONE TOP1×STRUCTURE @ $(date -Iseconds) ====="
