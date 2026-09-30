#!/usr/bin/env bash
# Decoder upgrade: NDA-SS semantics + CPA text + ControlNet-Canny structure
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/cn_ip_decode/sub-08}"
NDA_SS="${NDA_SS:-${NB_ROOT}/outputs/nda_ss/sub-08}"
V2="${V2:-${NB_ROOT}/outputs/oracle_chase_v2/sub-08}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "${OUT}/prompts" "${OUT}/generation" "${OUT}/prefetch"
cd "${NB_ROOT}"

echo "===== [0] Prefetch SDXL + ControlNet @ $(date -Iseconds) ====="
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"

"${PYTHON}" - <<'PY'
import os
from pathlib import Path
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
from huggingface_hub import snapshot_download
hub = Path(os.environ["HF_HUB_CACHE"])
for repo in [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "diffusers/controlnet-canny-sdxl-1.0",
]:
    print(f"[prefetch] {repo}")
    snapshot_download(repo_id=repo, cache_dir=str(hub))
print("[OK] models ready")
import open_clip
open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k", device="cpu")
print("[OK] openclip")
PY

test -f "${NDA_SS}/blend/mem_decode_a50.npy"
test -f "${NDA_SS}/memory/rag_soft5_neighbor_idx_test.npy"

if [[ -f "${V2}/prompts/prompts_pred.json" ]]; then
  cp -a "${V2}/prompts/." "${OUT}/prompts/"
  echo "[OK] reused CPA prompts"
else
  echo "[ERR] missing CPA prompts at ${V2}/prompts"; exit 1
fi

EMB="${NDA_SS}/blend/mem_decode_a50.npy"
NEIGH="${NDA_SS}/memory/rag_soft5_neighbor_idx_test.npy"
PROMPT="${OUT}/prompts/prompts_pred.json"

echo "===== [1] ControlNet decoder variants ====="
run_cn() {
  local tag="$1"; shift
  local gdir="${OUT}/generation/${tag}"
  [[ -f "${gdir}/generated/199.png" ]] && echo "[SKIP] ${tag}" && return 0
  "${PYTHON}" scripts/nda/generate_cn_ip_decode.py \
    --embed-npy "${EMB}" \
    --neighbor-idx-npy "${NEIGH}" \
    --output-dir "${gdir}" \
    --tag "${tag}" \
    --seed 42 \
    --gen-steps 30 \
    --gen-guidance 5.0 \
    --skip-metrics \
    "$@"
}

# A) structure+semantic (no text) — high success structural baseline
run_cn "cn0.8_ip0.9" --cn-scale 0.8 --ip-scale 0.9

# B) stronger structure
run_cn "cn1.0_ip0.85" --cn-scale 1.0 --ip-scale 0.85

# C) structure + CPA text (full MAC-R decoder intent)
run_cn "cn0.8_ip0.9_cpa" --cn-scale 0.8 --ip-scale 0.9 --prompts-json "${PROMPT}"

# D) milder CN + CPA (protect semantics)
run_cn "cn0.5_ip1.0_cpa" --cn-scale 0.5 --ip-scale 1.0 --prompts-json "${PROMPT}"

echo "===== [2] CLIP + FID ====="
"${PYTHON}" scripts/nb_adapter/eval_clip_fid.py \
  --gen-root "${OUT}/generation" \
  --tags "cn0.8_ip0.9,cn1.0_ip0.85,cn0.8_ip0.9_cpa,cn0.5_ip1.0_cpa" \
  --output-json "${OUT}/clip_fid_metrics.json"

"${PYTHON}" - <<PY
import json
from pathlib import Path
out=Path("${OUT}")
metrics=json.loads((out/"clip_fid_metrics.json").read_text())
results=metrics.get("results",[])
by={r["tag"]:r for r in results}
best_clip=max(results, key=lambda r: r.get("clip_cosine", -1), default=None)
best_pix=max(results, key=lambda r: r.get("pixcorr", -1), default=None)
best_ssim=max(results, key=lambda r: r.get("ssim", -1), default=None)
summary={
  "pipeline":"cn_ip_decode_P1",
  "diagnosis":"main bottleneck was decoder (turbo img2img); switched to SDXL ControlNet-Canny + IP",
  "backbone":"NDA-SS mem_decode_a50 (frozen)",
  "structure":"neighbor Canny ControlNet (no GT)",
  "targets":{"cogcap_pixcorr":0.15, "cogcap_ssim":0.347},
  "prev_best_cpa_img2img":{"clip":0.4637, "pixcorr":0.1346, "ssim":0.0504},
  "best_clip": best_clip,
  "best_pixcorr": best_pix,
  "best_ssim": best_ssim,
  "all_gen": results,
}
(out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "===== DONE cn_ip_decode @ $(date -Iseconds) ====="
