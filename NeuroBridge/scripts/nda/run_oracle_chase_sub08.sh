#!/usr/bin/env bash
# Chase oracle: restore NDA-SS backbone + RN50 retrieval prompts (no CFM/RGT damage)
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/oracle_chase/sub-08}"
NDA_SS="${NDA_SS:-${NB_ROOT}/outputs/nda_ss/sub-08}"
SEMTXT="${SEMTXT:-${NB_ROOT}/outputs/nda_v2_semtxt/sub-08}"
SS_CKPT="${SS_CKPT:-${NDA_SS}/ss/checkpoint_ss_calib_best.pth}"
NB_CKPT="${NB_CKPT:-${NB_ROOT}/results/things_eeg/intra-subjects/20260901-074319-sub-08/checkpoint_test_best.pth}"
NB_EMB="${NB_EMB:-${NB_ROOT}/outputs/nda_v2_semtxt/sub-08/embeds/z_eeg_proj_test.npy}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "${OUT}/prompts" "${OUT}/generation"
cd "${NB_ROOT}"

echo "===== [0] Prefetch @ $(date -Iseconds) ====="
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
"${PYTHON}" - <<'PY'
import os
os.environ.setdefault("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub")
os.environ.setdefault("OPENCLIP_CACHE_DIR", "/project/peilab/why/cache/eeg-brainit/open_clip")
import open_clip
open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k", device="cpu")
print("[OK]")
PY

test -f "${NDA_SS}/blend/mem_decode_a50.npy"
test -f "${NDA_SS}/memory/rag_soft5_neighbor_idx_test.npy"
test -f "${NB_EMB}"
test -f "${NB_CKPT}"

TEXT_PHRASES="${SEMTXT}/clip_text/test/concept_phrases.json"
if [[ ! -f "${TEXT_PHRASES}" ]]; then
  TEXT_PHRASES="${NB_ROOT}/outputs/rgt_v4_txtfix/sub-08/clip_text/test/concept_phrases.json"
fi
test -f "${TEXT_PHRASES}"

echo "===== [1] Strongest RN50 prompts (official NB embed + matched img projector) ====="
# Empirically best pair: semtxt/NB z_eeg_proj + official img_projector → ~35% Top-1
# (nda_ss z_sem + ss projector only ~24.5%; mismatched pairs worse)
"${PYTHON}" scripts/nda/chase_oracle_prompts.py \
  --eeg-embed-npy "${NB_EMB}" \
  --img-projector-ckpt "${NB_CKPT}" \
  --rn50-image-test "${NB_ROOT}/data/things_eeg/image_feature/RN50/image_test.npy" \
  --concept-phrases-test "${TEXT_PHRASES}" \
  --output-dir "${OUT}/prompts" \
  --top-k 1

PROMPT="${OUT}/prompts/prompts_pred.json"
ORACLE="${OUT}/prompts/prompts_oracle.json"
EMB="${NDA_SS}/blend/mem_decode_a50.npy"
NEIGH="${NDA_SS}/memory/rag_soft5_neighbor_idx_test.npy"
DECODE="${NDA_SS}/train/z_decode_vith_test.npy"

echo "===== [2] Generation on RESTORED NDA-SS backbone ====="
run_gen() {
  local tag="$1" emb="$2"; shift 2
  local gdir="${OUT}/generation/${tag}"
  [[ -f "${gdir}/generated/199.png" ]] && echo "[SKIP] ${tag}" && return 0
  "${PYTHON}" scripts/nb_adapter/generate_rag_lowlevel.py \
    --embed-npy "${emb}" --neighbor-idx-npy "${NEIGH}" \
    --output-dir "${gdir}" --seed 42 --tag "${tag}" --skip-metrics "$@"
}

# A) restore baseline (should ~0.439)
run_gen "chase_nda_ss_mem_decode" "${EMB}" --strength 0.4 --ip-scale 1.0

# B) chase oracle with RN50-retrieved prompts
run_gen "chase_nda_ss_rn50prompt" "${EMB}" \
  --strength 0.45 --ip-scale 0.9 --prompts-json "${PROMPT}" --gen-guidance 1.5

# C) oracle ceiling on same backbone
run_gen "chase_nda_ss_oracle" "${EMB}" \
  --strength 0.45 --ip-scale 0.9 --prompts-json "${ORACLE}" --gen-guidance 1.5

# D) decode-only + rn50 prompt (ablate mem)
run_gen "chase_decode_rn50prompt" "${DECODE}" \
  --strength 0.45 --ip-scale 0.9 --prompts-json "${PROMPT}" --gen-guidance 1.5

echo "===== [3] Metrics ====="
"${PYTHON}" scripts/nb_adapter/eval_clip_fid.py \
  --gen-root "${OUT}/generation" \
  --tags "chase_nda_ss_mem_decode,chase_nda_ss_rn50prompt,chase_nda_ss_oracle,chase_decode_rn50prompt" \
  --output-json "${OUT}/clip_fid_metrics.json"

"${PYTHON}" - <<PY
import json
from pathlib import Path
out=Path("${OUT}")
metrics=json.loads((out/"clip_fid_metrics.json").read_text())
results=metrics.get("results",[])
pr=json.loads((out/"prompts/prompt_report.json").read_text())
by={r["tag"]:r for r in results}
summary={
  "pipeline":"oracle_chase_restore_nda_ss",
  "architecture_fix":"strip RGT/CFM from gen path; NDA-SS backbone + RN50 prompts",
  "concept_top1_rn50": pr.get("concept_top1"),
  "concept_top5_rn50": pr.get("concept_top5"),
  "targets":{"nda_ss_original":0.439, "oracle_ceiling":0.454},
  "restore_baseline": by.get("chase_nda_ss_mem_decode"),
  "rn50_prompt": by.get("chase_nda_ss_rn50prompt"),
  "oracle": by.get("chase_nda_ss_oracle"),
  "gap_to_oracle": None,
  "all_gen": results,
}
if by.get("chase_nda_ss_rn50prompt") and by.get("chase_nda_ss_oracle"):
  summary["gap_to_oracle"]=by["chase_nda_ss_oracle"]["clip_cosine"]-by["chase_nda_ss_rn50prompt"]["clip_cosine"]
(out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "===== DONE oracle-chase @ $(date -Iseconds) ====="
