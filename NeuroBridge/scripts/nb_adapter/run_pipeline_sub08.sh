#!/usr/bin/env bash
# NeuroBridge → ViT-H adapter pipeline (sub-08): extract → Linear/MLP → SDXL gen.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
OUT="${OUT:-${NB_ROOT}/outputs/nb_adapter/sub-08}"
SUBJECT="${SUBJECT:-8}"
MAX_IMAGES="${MAX_IMAGES:-50}"   # 0 = all 200; default 50 for faster first pass
DEVICE="${DEVICE:-cuda:0}"
PYTHON="${PYTHON:-python}"

CKPT="${CKPT:-${NB_ROOT}/results/things_eeg/intra-subjects/20260901-074319-sub-08/checkpoint_test_best.pth}"
CLIP_TRAIN="${CLIP_TRAIN:-${BRAINIT}/outputs/atm_bridge/clip_img_train_1024.npy}"
CLIP_TEST="${CLIP_TEST:-${BRAINIT}/outputs/atm_bridge/clip_img_test_1024.npy}"
GALLERY="${GALLERY:-${BRAINIT}/outputs/eval/atm_baseline/test_ViT-H-14_laion2b_s32b_b79k_features.npy}"
PRIOR_NPY="${PRIOR_NPY:-${BRAINIT}/outputs/eval/atm_pipeline_sub08/sub-08_prior_clip_1024.npy}"

EMBED_DIR="${OUT}/embeds"
ADAPT_DIR="${OUT}/adapters"
GEN_ROOT="${OUT}/generation"

mkdir -p "${OUT}" "${EMBED_DIR}" "${ADAPT_DIR}" "${GEN_ROOT}"
cd "${NB_ROOT}"

echo "===== [1/4] Extract NB embeds @ $(date -Iseconds) ====="
if [[ ! -f "${EMBED_DIR}/z_eeg_proj_test.npy" ]]; then
  "${PYTHON}" scripts/nb_adapter/extract_nb_embeds.py \
    --nb-root "${NB_ROOT}" \
    --checkpoint "${CKPT}" \
    --subject "${SUBJECT}" \
    --output-dir "${EMBED_DIR}" \
    --device "${DEVICE}"
else
  echo "[INFO] embeds exist, skip extract"
fi

echo "===== [2/4] Train Linear + MLP adapters @ $(date -Iseconds) ====="
"${PYTHON}" scripts/nb_adapter/train_nb_adapter.py \
  --embed-dir "${EMBED_DIR}" \
  --clip-train "${CLIP_TRAIN}" \
  --clip-test "${CLIP_TEST}" \
  --gallery "${GALLERY}" \
  --output-dir "${ADAPT_DIR}" \
  --input-key proj \
  --device "${DEVICE}"

echo "===== [3/4] SDXL generation (linear / mlp / ATM prior / teacher) @ $(date -Iseconds) ====="
declare -A EMBEDS=(
  [linear]="${ADAPT_DIR}/linear_test_clip_1024.npy"
  [mlp]="${ADAPT_DIR}/mlp_test_clip_1024.npy"
  [atm_prior]="${PRIOR_NPY}"
  [teacher]="${CLIP_TEST}"
)

for tag in linear mlp atm_prior teacher; do
  echo "--- generate ${tag} ---"
  "${PYTHON}" scripts/nb_adapter/generate_from_embeds.py \
    --embed-npy "${EMBEDS[${tag}]}" \
    --output-dir "${GEN_ROOT}/${tag}" \
    --tag "${tag}" \
    --max-images "${MAX_IMAGES}" \
    --seed 42
done

echo "===== [4/4] Summarize @ $(date -Iseconds) ====="
"${PYTHON}" - <<PY
import json
from pathlib import Path
out = Path("${OUT}")
adapt = json.loads((out / "adapters" / "adapter_eval.json").read_text())
summary = {
    "subject": "sub-08",
    "max_images": int("${MAX_IMAGES}"),
    "adapter_eval": adapt,
    "generation": {},
}
gen_root = out / "generation"
for tag in ["linear", "mlp", "atm_prior", "teacher"]:
    p = gen_root / tag / "metrics.json"
    if p.is_file():
        summary["generation"][tag] = json.loads(p.read_text())
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({
    "adapters": {k: {"gt_cos": v.get("test_gt_cos"), "top1": v["test_retrieval"]["top1"], "top5": v["test_retrieval"]["top5"]} for k,v in adapt["adapters"].items()},
    "generation": {k: (v.get("metrics") or {}) for k,v in summary["generation"].items()},
}, indent=2))
print("[OK]", out / "summary.json")
PY

echo "===== DONE @ $(date -Iseconds) ====="
