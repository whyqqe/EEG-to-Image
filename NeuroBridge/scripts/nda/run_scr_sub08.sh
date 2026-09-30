#!/usr/bin/env bash
# SCR: Structure-Credible Routing on ControlNet decoder + paper-grade SSIM
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/scr_decode/sub-08}"
NDA_SS="${NDA_SS:-${NB_ROOT}/outputs/nda_ss/sub-08}"
CN_PREV="${CN_PREV:-${NB_ROOT}/outputs/cn_ip_decode/sub-08}"
V2="${V2:-${NB_ROOT}/outputs/oracle_chase_v2/sub-08}"
PYTHON="${PYTHON:-python}"

mkdir -p "${OUT}/prompts" "${OUT}/routing" "${OUT}/generation"
cd "${NB_ROOT}"

echo "===== [0] Prefetch @ $(date -Iseconds) ====="
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
"${PYTHON}" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
hub=Path(os.environ["HF_HUB_CACHE"])
for repo in ["stabilityai/stable-diffusion-xl-base-1.0","diffusers/controlnet-canny-sdxl-1.0"]:
    print("[prefetch]", repo)
    snapshot_download(repo_id=repo, cache_dir=str(hub))
print("[OK]")
PY

EMB="${NDA_SS}/blend/mem_decode_a50.npy"
NEIGH="${NDA_SS}/memory/rag_soft5_neighbor_idx_test.npy"
NEIGH_CLIP="${NDA_SS}/memory/rag_soft5_train_clip_1024.npy"
test -f "${EMB}" && test -f "${NEIGH}" && test -f "${NEIGH_CLIP}"

if [[ -f "${V2}/prompts/prompts_pred.json" ]]; then
  cp -a "${V2}/prompts/." "${OUT}/prompts/"
else
  echo "[ERR] need CPA prompts"; exit 1
fi
PROMPT="${OUT}/prompts/prompts_pred.json"
MARGINS="${OUT}/prompts/margins.npy"

echo "===== [1] Compute SCR routing ====="
"${PYTHON}" scripts/nda/compute_scr_routing.py \
  --eeg-embed-npy "${EMB}" \
  --neighbor-idx-npy "${NEIGH}" \
  --neighbor-clip-train-npy "${NEIGH_CLIP}" \
  --margins-npy "${MARGINS}" \
  --output-dir "${OUT}/routing" \
  --cn-min 0.35 --cn-max 1.0 \
  --ip-min 0.85 --ip-max 1.0 \
  --fuse-beta-min 0.82 --fuse-beta-max 1.0

CN_NPY="${OUT}/routing/cn_scale.npy"
IP_NPY="${OUT}/routing/ip_scale.npy"
FUSE_NPY="${OUT}/routing/fuse_beta.npy"

echo "===== [2] Generation ====="
run_cn() {
  local tag="$1"; shift
  local gdir="${OUT}/generation/${tag}"
  [[ -f "${gdir}/generated/199.png" ]] && echo "[SKIP] ${tag}" && return 0
  "${PYTHON}" scripts/nda/generate_cn_ip_decode.py \
    --embed-npy "${EMB}" --neighbor-idx-npy "${NEIGH}" \
    --output-dir "${gdir}" --tag "${tag}" --seed 42 \
    --gen-steps 30 --gen-guidance 5.0 --skip-metrics "$@"
}

# reuse previous best fixed decode as reference (copy)
REF_TAG="ref_cn0.5_ip1.0_cpa"
if [[ ! -f "${OUT}/generation/${REF_TAG}/generated/199.png" ]]; then
  if [[ -f "${CN_PREV}/generation/cn0.5_ip1.0_cpa/generated/199.png" ]]; then
    mkdir -p "${OUT}/generation/${REF_TAG}"
    cp -a "${CN_PREV}/generation/cn0.5_ip1.0_cpa/generated" "${OUT}/generation/${REF_TAG}/"
    echo "[OK] copied previous best fixed CN decode"
  else
    run_cn "${REF_TAG}" --cn-scale 0.5 --ip-scale 1.0 --prompts-json "${PROMPT}"
  fi
fi

# SCR routed + CPA
run_cn "scr_cpa" \
  --cn-scale 0.5 --ip-scale 1.0 \
  --cn-scale-npy "${CN_NPY}" --ip-scale-npy "${IP_NPY}" \
  --prompts-json "${PROMPT}"

# SCR + CPA + low-level fuse when structure credible
run_cn "scr_cpa_fuse" \
  --cn-scale 0.5 --ip-scale 1.0 \
  --cn-scale-npy "${CN_NPY}" --ip-scale-npy "${IP_NPY}" \
  --fuse-beta-npy "${FUSE_NPY}" --enable-fuse \
  --prompts-json "${PROMPT}"

echo "===== [3] PAPER-GRADE metrics (skimage SSIM) ====="
"${PYTHON}" scripts/nda/eval_paper_metrics.py \
  --gen-root "${OUT}/generation" \
  --tags "${REF_TAG},scr_cpa,scr_cpa_fuse" \
  --output-json "${OUT}/paper_metrics.json"

"${PYTHON}" - <<PY
import json
from pathlib import Path
out=Path("${OUT}")
metrics=json.loads((out/"paper_metrics.json").read_text())
scr=json.loads((out/"routing/scr_report.json").read_text())
results=metrics.get("results",[])
by={r["tag"]:r for r in results}
best=max(results, key=lambda r: (r.get("ssim",-1), r.get("clip_cosine",-1)), default=None)
summary={
  "pipeline":"SCR_structure_credible_routing",
  "innovation":"route ControlNet/IP by EEG↔neighbor alignment + CPA margin; no CFM transport",
  "ssim_protocol":"skimage RGB 256 paper-grade (NOT ssim_simple proxy)",
  "scr": scr,
  "ref_fixed_cn": by.get("${REF_TAG}"),
  "scr_cpa": by.get("scr_cpa"),
  "scr_cpa_fuse": by.get("scr_cpa_fuse"),
  "best_by_ssim_then_clip": best,
  "targets":{"cogcap_pixcorr":0.15,"cogcap_ssim":0.347},
  "all": results,
}
(out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "===== DONE SCR @ $(date -Iseconds) ====="
