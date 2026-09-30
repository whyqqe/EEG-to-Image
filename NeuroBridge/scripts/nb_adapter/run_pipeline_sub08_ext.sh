#!/usr/bin/env bash
# Extended adapter architectures for sub-08 (reuses existing NB embeds).
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
OUT="${OUT:-${NB_ROOT}/outputs/nb_adapter/sub-08}"
EMBED_DIR="${OUT}/embeds"
EXT_DIR="${OUT}/adapters_ext"
GEN_ROOT="${OUT}/generation_ext"
MAX_IMAGES="${MAX_IMAGES:-50}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON="${PYTHON:-python}"
ARCHS="${ARCHS:-ridge,linear_nobias,mlp_deep,mlp_res,cfm,diffprior}"

CLIP_TRAIN="${CLIP_TRAIN:-${BRAINIT}/outputs/atm_bridge/clip_img_train_1024.npy}"
CLIP_TEST="${CLIP_TEST:-${BRAINIT}/outputs/atm_bridge/clip_img_test_1024.npy}"
GALLERY="${GALLERY:-${BRAINIT}/outputs/eval/atm_baseline/test_ViT-H-14_laion2b_s32b_b79k_features.npy}"

mkdir -p "${EXT_DIR}" "${GEN_ROOT}"
cd "${NB_ROOT}"

test -f "${EMBED_DIR}/z_eeg_proj_train.npy"
test -f "${EMBED_DIR}/z_eeg_proj_test.npy"

echo "===== [1/3] Train extended adapters (${ARCHS}) @ $(date -Iseconds) ====="
"${PYTHON}" scripts/nb_adapter/train_nb_adapter_ext.py \
  --embed-dir "${EMBED_DIR}" \
  --clip-train "${CLIP_TRAIN}" \
  --clip-test "${CLIP_TEST}" \
  --gallery "${GALLERY}" \
  --output-dir "${EXT_DIR}" \
  --archs "${ARCHS}" \
  --device "${DEVICE}"

echo "===== [2/3] SDXL generation for extended adapters @ $(date -Iseconds) ====="
IFS=',' read -r -a ARCH_ARR <<< "${ARCHS}"
for tag in "${ARCH_ARR[@]}"; do
  npy="${EXT_DIR}/${tag}_test_clip_1024.npy"
  if [[ ! -f "${npy}" ]]; then
    echo "[WARN] missing ${npy}, skip gen"
    continue
  fi
  echo "--- generate ${tag} ---"
  "${PYTHON}" scripts/nb_adapter/generate_from_embeds.py \
    --embed-npy "${npy}" \
    --output-dir "${GEN_ROOT}/${tag}" \
    --tag "${tag}" \
    --max-images "${MAX_IMAGES}" \
    --seed 42
done

echo "===== [3/3] Summarize @ $(date -Iseconds) ====="
"${PYTHON}" - <<PY
import json
from pathlib import Path
out = Path("${OUT}")
ext = json.loads((out / "adapters_ext" / "adapter_eval_ext.json").read_text())
base = {}
bp = out / "adapters" / "adapter_eval.json"
if bp.is_file():
    base = json.loads(bp.read_text()).get("adapters", {})
summary = {
    "subject": "sub-08",
    "max_images": int("${MAX_IMAGES}"),
    "base_adapters": {k: {"gt_cos": v.get("test_gt_cos"), "top1": v["test_retrieval"]["top1"], "top5": v["test_retrieval"]["top5"]} for k,v in base.items()},
    "ext_adapters": {k: {"gt_cos": v.get("test_gt_cos"), "top1": v["test_retrieval"]["top1"], "top5": v["test_retrieval"]["top5"]} for k,v in ext["adapters"].items()},
    "generation_ext": {},
}
gen_root = out / "generation_ext"
for tag, _ in ext["adapters"].items():
    p = gen_root / tag / "metrics.json"
    if p.is_file():
        summary["generation_ext"][tag] = json.loads(p.read_text()).get("metrics")
(out / "summary_ext.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print("[OK]", out / "summary_ext.json")
PY

echo "===== DONE EXT @ $(date -Iseconds) ====="
