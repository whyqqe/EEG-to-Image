#!/usr/bin/env bash
# COCA Phase A + B1: paper metrics on champion + Depth-ControlNet decoder
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/coca_depth/sub-08}"
NDA_SS="${NDA_SS:-${NB_ROOT}/outputs/nda_ss/sub-08}"
CN_PREV="${CN_PREV:-${NB_ROOT}/outputs/cn_ip_decode/sub-08}"
V2="${V2:-${NB_ROOT}/outputs/oracle_chase_v2/sub-08}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "${OUT}/prompts" "${OUT}/depth_cache" "${OUT}/generation" "${OUT}/retrieval"
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
for repo in [
  "stabilityai/stable-diffusion-xl-base-1.0",
  "diffusers/controlnet-depth-sdxl-1.0",
  "depth-anything/Depth-Anything-V2-Small-hf",
]:
  print("[prefetch]", repo)
  snapshot_download(repo_id=repo, cache_dir=str(hub))
print("[OK] prefetch")
PY

EMB="${NDA_SS}/blend/mem_decode_a50.npy"
NEIGH="${NDA_SS}/memory/rag_soft5_neighbor_idx_test.npy"
test -f "${EMB}" && test -f "${NEIGH}"
test -f "${V2}/prompts/prompts_pred.json"
cp -a "${V2}/prompts/." "${OUT}/prompts/"
PROMPT="${OUT}/prompts/prompts_pred.json"

echo "===== [A1] Clean vs CPA retrieval Top-1 report ====="
PHRASES="${NB_ROOT}/outputs/nda_v2_semtxt/sub-08/clip_text/test/concept_phrases.json"
test -f "${PHRASES}"
cp -f "${PHRASES}" "${OUT}/prompts/concept_phrases.json"
NB_CKPT="${NB_ROOT}/results/things_eeg/intra-subjects/20260901-074319-sub-08/checkpoint_test_best.pth"
"${PYTHON}" scripts/nda/chase_oracle_prompts.py \
  --nb-ckpt "${NB_CKPT}" --gallery clean \
  --concept-phrases-test "${OUT}/prompts/concept_phrases.json" \
  --output-dir "${OUT}/retrieval/clean" --device "${DEVICE}" --top-k 1
"${PYTHON}" scripts/nda/chase_oracle_prompts.py \
  --nb-ckpt "${NB_CKPT}" --gallery cpa \
  --concept-phrases-test "${OUT}/prompts/concept_phrases.json" \
  --output-dir "${OUT}/retrieval/cpa" --device "${DEVICE}" --top-k 1

echo "===== [B1] Depth cache for neighbors ====="
"${PYTHON}" scripts/nda/build_neighbor_depth_cache.py \
  --neighbor-idx-npy "${NEIGH}" \
  --output-dir "${OUT}/depth_cache" \
  --device "${DEVICE}" \
  --size 512

echo "===== [B1] Depth-ControlNet generation ====="
run_depth() {
  local tag="$1"; shift
  local gdir="${OUT}/generation/${tag}"
  [[ -f "${gdir}/generated/199.png" ]] && echo "[SKIP] ${tag}" && return 0
  "${PYTHON}" scripts/nda/generate_cn_ip_decode.py \
    --embed-npy "${EMB}" --neighbor-idx-npy "${NEIGH}" \
    --output-dir "${gdir}" --tag "${tag}" --seed 42 \
    --control-type depth --depth-cache-dir "${OUT}/depth_cache" \
    --gen-steps 30 --gen-guidance 5.0 --skip-metrics "$@"
}

# reference: copy previous canny champion
REF=ref_canny_cn0.5_ip1.0_cpa
if [[ ! -f "${OUT}/generation/${REF}/generated/199.png" ]]; then
  mkdir -p "${OUT}/generation/${REF}"
  cp -a "${CN_PREV}/generation/cn0.5_ip1.0_cpa/generated" "${OUT}/generation/${REF}/"
  echo "[OK] copied canny champion as reference"
fi

run_depth "depth_cn0.5_ip1.0_cpa" --cn-scale 0.5 --ip-scale 1.0 --prompts-json "${PROMPT}"
run_depth "depth_cn0.8_ip0.9_cpa" --cn-scale 0.8 --ip-scale 0.9 --prompts-json "${PROMPT}"
run_depth "depth_cn0.6_ip1.0" --cn-scale 0.6 --ip-scale 1.0

echo "===== [A2] PAPER-GRADE metrics ====="
"${PYTHON}" scripts/nda/eval_paper_metrics.py \
  --gen-root "${OUT}/generation" \
  --tags "${REF},depth_cn0.5_ip1.0_cpa,depth_cn0.8_ip0.9_cpa,depth_cn0.6_ip1.0" \
  --output-json "${OUT}/paper_metrics.json"

"${PYTHON}" - <<PY
import json
from pathlib import Path
out=Path("${OUT}")
metrics=json.loads((out/"paper_metrics.json").read_text())
clean=json.loads((out/"retrieval/clean/prompt_report.json").read_text()) if (out/"retrieval/clean/prompt_report.json").is_file() else {}
cpa=json.loads((out/"retrieval/cpa/prompt_report.json").read_text()) if (out/"retrieval/cpa/prompt_report.json").is_file() else {}
results=metrics.get("results",[])
by={r["tag"]:r for r in results}
best_ssim=max(results, key=lambda r: r.get("ssim",-1), default=None)
best_clip=max(results, key=lambda r: r.get("clip_cosine",-1), default=None)
summary={
  "pipeline":"COCA_Phase_A_B1",
  "plan":"docs/COCA_MASTER_PLAN.md",
  "retrieval":{
    "clean_top1": clean.get("concept_top1"),
    "clean_top5": clean.get("concept_top5"),
    "cpa_top1": cpa.get("concept_top1"),
    "cpa_top5": cpa.get("concept_top5"),
    "note":"main-table must use clean gallery; CPA is appendix/appendix only",
  },
  "ssim_protocol":"skimage RGB 256 paper-grade",
  "ref_canny": by.get("${REF}"),
  "best_ssim": best_ssim,
  "best_clip": best_clip,
  "all": results,
}
(out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "===== DONE COCA A+B1 @ $(date -Iseconds) ====="
