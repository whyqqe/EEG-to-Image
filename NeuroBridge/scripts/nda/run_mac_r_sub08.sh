#!/usr/bin/env bash
# MAC-R P1 on restored NDA-SS + CPA prompts (no CFM; stage-wise + router + optional LL fuse)
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/mac_r/sub-08}"
NDA_SS="${NDA_SS:-${NB_ROOT}/outputs/nda_ss/sub-08}"
V2="${V2:-${NB_ROOT}/outputs/oracle_chase_v2/sub-08}"
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
[[ -f "${TEXT_PHRASES}" ]] || TEXT_PHRASES="${NB_ROOT}/outputs/rgt_v4_txtfix/sub-08/clip_text/test/concept_phrases.json"

echo "===== [1] CPA prompts + margins (reuse v2 if present) ====="
if [[ -f "${V2}/prompts/prompts_pred.json" && -f "${V2}/prompts/margins.npy" ]]; then
  cp -a "${V2}/prompts/." "${OUT}/prompts/"
  echo "[OK] reused CPA prompts from oracle_chase_v2"
else
  "${PYTHON}" scripts/nda/chase_oracle_prompts.py \
    --nb-ckpt "${NB_CKPT}" \
    --gallery cpa \
    --concept-phrases-test "${TEXT_PHRASES}" \
    --output-dir "${OUT}/prompts" \
    --margin-gate 0.02 \
    --soft-gate 0.05 \
    --device "${DEVICE}" \
    --top-k 1
fi

EMB="${NDA_SS}/blend/mem_decode_a50.npy"
NEIGH="${NDA_SS}/memory/rag_soft5_neighbor_idx_test.npy"
PROMPT="${OUT}/prompts/prompts_pred.json"
MARGINS="${OUT}/prompts/margins.npy"

echo "===== [2] MAC-R generation ====="
run_mac() {
  local tag="$1"; shift
  local gdir="${OUT}/generation/${tag}"
  [[ -f "${gdir}/generated/199.png" ]] && echo "[SKIP] ${tag}" && return 0
  "${PYTHON}" scripts/nda/generate_mac_r.py \
    --embed-npy "${EMB}" \
    --neighbor-idx-npy "${NEIGH}" \
    --prompts-json "${PROMPT}" \
    --margins-npy "${MARGINS}" \
    --output-dir "${gdir}" \
    --seed 42 \
    --tag "${tag}" \
    --gen-guidance 1.5 \
    --skip-metrics \
    "$@"
}

# A) reference: single-pass CPA always (same as oracle_chase_v2 best)
REF_DIR="${OUT}/generation/mac_r_ref_cpa_always"
if [[ ! -f "${REF_DIR}/generated/199.png" ]]; then
  if [[ -f "${V2}/generation/v2_cpa_prompt_always/generated/199.png" ]]; then
    mkdir -p "${REF_DIR}"
    cp -a "${V2}/generation/v2_cpa_prompt_always/generated" "${REF_DIR}/"
    echo '{"tag":"mac_r_ref_cpa_always","note":"copied from oracle_chase_v2"}' > "${REF_DIR}/metrics.json"
    echo "[OK] copied v2 CPA always as reference"
  else
    "${PYTHON}" scripts/nb_adapter/generate_rag_lowlevel.py \
      --embed-npy "${EMB}" --neighbor-idx-npy "${NEIGH}" \
      --output-dir "${REF_DIR}" --seed 42 --tag mac_r_ref_cpa_always --skip-metrics \
      --strength 0.45 --ip-scale 0.9 --prompts-json "${PROMPT}" --gen-guidance 1.5
  fi
fi

# B) MAC-R staged + router (no LL fuse)
run_mac "mac_r_staged_routed" --edge-boost 0.25

# C) MAC-R staged + router + low-level fuse
run_mac "mac_r_staged_routed_fuse" --edge-boost 0.25 --enable-fuse --fuse-beta 0.85

# D) MAC-R light: always-strong CPA text + mild structure (no abstain)
run_mac "mac_r_light_always" --edge-boost 0.12 --margin-gate -1.0 --soft-gate -1.0

echo "===== [3] CLIP + FID ====="
"${PYTHON}" scripts/nb_adapter/eval_clip_fid.py \
  --gen-root "${OUT}/generation" \
  --tags "mac_r_ref_cpa_always,mac_r_staged_routed,mac_r_staged_routed_fuse,mac_r_light_always" \
  --output-json "${OUT}/clip_fid_metrics.json"

"${PYTHON}" - <<PY
import json
from pathlib import Path
out=Path("${OUT}")
metrics=json.loads((out/"clip_fid_metrics.json").read_text())
results=metrics.get("results",[])
pr=json.loads((out/"prompts/prompt_report.json").read_text()) if (out/"prompts/prompt_report.json").is_file() else {}
by={r["tag"]:r for r in results}
best=max(results, key=lambda r: r.get("clip_cosine", -1), default=None)
summary={
  "pipeline":"MAC-R_P1",
  "claim":"manifold-aware condition routing: keep NDA-SS/CPA; stage-wise assembly + confidence router; no CFM transport",
  "concept_top1": pr.get("concept_top1"),
  "concept_top5": pr.get("concept_top5"),
  "ref_cpa_always": by.get("mac_r_ref_cpa_always"),
  "staged_routed": by.get("mac_r_staged_routed"),
  "staged_routed_fuse": by.get("mac_r_staged_routed_fuse"),
  "light_always": by.get("mac_r_light_always"),
  "best": best,
  "all_gen": results,
}
(out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "===== DONE MAC-R @ $(date -Iseconds) ====="
