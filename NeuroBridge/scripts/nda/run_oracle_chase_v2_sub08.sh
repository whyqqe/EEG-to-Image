#!/usr/bin/env bash
# Oracle-chase v2: CPA-aug concept retrieval (~73% Top-1) + margin-gated prompts on NDA-SS backbone
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/oracle_chase_v2/sub-08}"
NDA_SS="${NDA_SS:-${NB_ROOT}/outputs/nda_ss/sub-08}"
SEMTXT="${SEMTXT:-${NB_ROOT}/outputs/nda_v2_semtxt/sub-08}"
NB_CKPT="${NB_CKPT:-${NB_ROOT}/results/things_eeg/intra-subjects/20260901-074319-sub-08/checkpoint_test_best.pth}"
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
test -f "${NB_CKPT}"

TEXT_PHRASES="${SEMTXT}/clip_text/test/concept_phrases.json"
if [[ ! -f "${TEXT_PHRASES}" ]]; then
  TEXT_PHRASES="${NB_ROOT}/outputs/rgt_v4_txtfix/sub-08/clip_text/test/concept_phrases.json"
fi
test -f "${TEXT_PHRASES}"

echo "===== [1] CPA-aug concept prompts (expect ~73% Top-1) + margin gate ====="
"${PYTHON}" scripts/nda/chase_oracle_prompts.py \
  --nb-ckpt "${NB_CKPT}" \
  --gallery cpa \
  --concept-phrases-test "${TEXT_PHRASES}" \
  --output-dir "${OUT}/prompts" \
  --margin-gate 0.02 \
  --soft-gate 0.05 \
  --device "${DEVICE}" \
  --top-k 1

PROMPT_ALWAYS="${OUT}/prompts/prompts_pred.json"
PROMPT_GATED="${OUT}/prompts/prompts_gated.json"
ORACLE="${OUT}/prompts/prompts_oracle.json"
STRENGTH="${OUT}/prompts/strength_gated.npy"
IPSCALE="${OUT}/prompts/ip_scale_gated.npy"
EMB="${NDA_SS}/blend/mem_decode_a50.npy"
NEIGH="${NDA_SS}/memory/rag_soft5_neighbor_idx_test.npy"

echo "===== [2] Generation on RESTORED NDA-SS backbone ====="
run_gen() {
  local tag="$1" emb="$2"; shift 2
  local gdir="${OUT}/generation/${tag}"
  [[ -f "${gdir}/generated/199.png" ]] && echo "[SKIP] ${tag}" && return 0
  "${PYTHON}" scripts/nb_adapter/generate_rag_lowlevel.py \
    --embed-npy "${emb}" --neighbor-idx-npy "${NEIGH}" \
    --output-dir "${gdir}" --seed 42 --tag "${tag}" --skip-metrics "$@"
}

# A) restore baseline (no text)
run_gen "v2_nda_ss_mem_decode" "${EMB}" --strength 0.4 --ip-scale 1.0

# B) always CPA prompt (~73% Top-1) — highest expected CLIP
run_gen "v2_cpa_prompt_always" "${EMB}" \
  --strength 0.45 --ip-scale 0.9 --prompts-json "${PROMPT_ALWAYS}" --gen-guidance 1.5

# C) margin-gated prompts + per-sample strength/ip
run_gen "v2_cpa_prompt_gated" "${EMB}" \
  --strength 0.4 --ip-scale 1.0 \
  --strength-npy "${STRENGTH}" --ip-scale-npy "${IPSCALE}" \
  --prompts-json "${PROMPT_GATED}" --gen-guidance 1.5

# D) oracle ceiling
run_gen "v2_nda_ss_oracle" "${EMB}" \
  --strength 0.45 --ip-scale 0.9 --prompts-json "${ORACLE}" --gen-guidance 1.5

echo "===== [3] CLIP + FID metrics ====="
"${PYTHON}" scripts/nb_adapter/eval_clip_fid.py \
  --gen-root "${OUT}/generation" \
  --tags "v2_nda_ss_mem_decode,v2_cpa_prompt_always,v2_cpa_prompt_gated,v2_nda_ss_oracle" \
  --output-json "${OUT}/clip_fid_metrics.json"

"${PYTHON}" - <<PY
import json
from pathlib import Path
out=Path("${OUT}")
metrics=json.loads((out/"clip_fid_metrics.json").read_text())
results=metrics.get("results",[])
pr=json.loads((out/"prompts/prompt_report.json").read_text())
by={r["tag"]:r for r in results}
best=max(
  [r for r in results if r["tag"] in ("v2_cpa_prompt_always","v2_cpa_prompt_gated")],
  key=lambda r: r.get("clip_cosine", -1),
  default=None,
)
summary={
  "pipeline":"oracle_chase_v2_cpa_gallery",
  "fix":"CPA-aug gallery concept retrieval (~73% Top-1) + optional margin gate; NDA-SS backbone",
  "concept_top1": pr.get("concept_top1"),
  "concept_top5": pr.get("concept_top5"),
  "concept_top1_clean_gallery": pr.get("concept_top1_clean_gallery"),
  "concept_top1_cpa_gallery": pr.get("concept_top1_cpa_gallery"),
  "gated_coverage": pr.get("gated_prompt_coverage"),
  "gated_precision": pr.get("gated_prompt_precision"),
  "restore_baseline": by.get("v2_nda_ss_mem_decode"),
  "cpa_prompt_always": by.get("v2_cpa_prompt_always"),
  "cpa_prompt_gated": by.get("v2_cpa_prompt_gated"),
  "oracle": by.get("v2_nda_ss_oracle"),
  "best_deployable": best,
  "gap_to_oracle": None,
  "all_gen": results,
}
if best and by.get("v2_nda_ss_oracle"):
  summary["gap_to_oracle"]=by["v2_nda_ss_oracle"]["clip_cosine"]-best["clip_cosine"]
(out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "===== DONE oracle-chase-v2 @ $(date -Iseconds) ====="
