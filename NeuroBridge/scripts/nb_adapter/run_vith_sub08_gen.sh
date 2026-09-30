#!/usr/bin/env bash
# After ViT-H NB training: extract sub-08 embeds → Linear/MLP → SDXL gen
# (same ViT-H space as IP-Adapter; compare to RN50-adapter run).
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
RESULT_ROOT="${RESULT_ROOT:-${NB_ROOT}/results/things_eeg/intra-subjects-vit-h}"
OUT="${OUT:-${NB_ROOT}/outputs/nb_adapter/sub-08-vit-h}"
MAX_IMAGES="${MAX_IMAGES:-50}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON="${PYTHON:-python}"

CLIP_TRAIN="${CLIP_TRAIN:-${BRAINIT}/outputs/atm_bridge/clip_img_train_1024.npy}"
CLIP_TEST="${CLIP_TEST:-${BRAINIT}/outputs/atm_bridge/clip_img_test_1024.npy}"
GALLERY="${GALLERY:-${BRAINIT}/outputs/eval/atm_baseline/test_ViT-H-14_laion2b_s32b_b79k_features.npy}"
PRIOR_NPY="${PRIOR_NPY:-${BRAINIT}/outputs/eval/atm_pipeline_sub08/sub-08_prior_clip_1024.npy}"

# Find sub-08 checkpoint (best)
CKPT=$(find "${RESULT_ROOT}" -type f -path '*sub-08*/checkpoint_test_best.pth' | sort | tail -1)
if [[ -z "${CKPT}" ]]; then
  echo "[ERROR] no ViT-H sub-08 checkpoint under ${RESULT_ROOT}" >&2
  exit 1
fi
echo "[INFO] using ckpt=${CKPT}"

EMBED_DIR="${OUT}/embeds"
ADAPT_DIR="${OUT}/adapters"
GEN_ROOT="${OUT}/generation"
mkdir -p "${EMBED_DIR}" "${ADAPT_DIR}" "${GEN_ROOT}"
cd "${NB_ROOT}"

echo "===== Extract ViT-H NB embeds ====="
"${PYTHON}" scripts/nb_adapter/extract_nb_embeds.py \
  --nb-root "${NB_ROOT}" \
  --checkpoint "${CKPT}" \
  --subject 8 \
  --image-feature-dir "data/things_eeg/image_feature/ViT-H-14" \
  --output-dir "${EMBED_DIR}" \
  --device "${DEVICE}"

echo "===== Train Linear+MLP (NB ViT-H → ViT-H 1024) ====="
"${PYTHON}" scripts/nb_adapter/train_nb_adapter.py \
  --embed-dir "${EMBED_DIR}" \
  --clip-train "${CLIP_TRAIN}" \
  --clip-test "${CLIP_TEST}" \
  --gallery "${GALLERY}" \
  --output-dir "${ADAPT_DIR}" \
  --input-key proj \
  --device "${DEVICE}"

echo "===== Also try raw 1024 backbone as adapter input ====="
"${PYTHON}" scripts/nb_adapter/train_nb_adapter.py \
  --embed-dir "${EMBED_DIR}" \
  --clip-train "${CLIP_TRAIN}" \
  --clip-test "${CLIP_TEST}" \
  --gallery "${GALLERY}" \
  --output-dir "${ADAPT_DIR}/raw_in" \
  --input-key raw \
  --device "${DEVICE}"

echo "===== Generate ====="
declare -A EMBEDS=(
  [linear]="${ADAPT_DIR}/linear_test_clip_1024.npy"
  [mlp]="${ADAPT_DIR}/mlp_test_clip_1024.npy"
  [linear_raw]="${ADAPT_DIR}/raw_in/linear_test_clip_1024.npy"
  [mlp_raw]="${ADAPT_DIR}/raw_in/mlp_test_clip_1024.npy"
  [atm_prior]="${PRIOR_NPY}"
  [teacher]="${CLIP_TEST}"
)
for tag in linear mlp linear_raw mlp_raw atm_prior teacher; do
  echo "--- ${tag} ---"
  "${PYTHON}" scripts/nb_adapter/generate_from_embeds.py \
    --embed-npy "${EMBEDS[${tag}]}" \
    --output-dir "${GEN_ROOT}/${tag}" \
    --tag "vith_${tag}" \
    --max-images "${MAX_IMAGES}" \
    --seed 42
done

"${PYTHON}" - <<PY
import json
from pathlib import Path
out = Path("${OUT}")
summary = {"subject": "sub-08", "teacher": "ViT-H-14", "ckpt": "${CKPT}", "adapters": {}, "adapters_raw": {}, "generation": {}}
for name, path in [("proj", out/"adapters"/"adapter_eval.json"), ("raw", out/"adapters"/"raw_in"/"adapter_eval.json")]:
    if path.is_file():
        d = json.loads(path.read_text())
        key = "adapters" if name == "proj" else "adapters_raw"
        summary[key] = {
            k: {"gt_cos": v["test_gt_cos"], "top1": v["test_retrieval"]["top1"], "top5": v["test_retrieval"]["top5"]}
            for k, v in d["adapters"].items()
        }
for tag in ["linear", "mlp", "linear_raw", "mlp_raw", "atm_prior", "teacher"]:
    p = out / "generation" / tag / "metrics.json"
    if p.is_file():
        summary["generation"][tag] = json.loads(p.read_text()).get("metrics")
# retrieval from NB train result.csv if present
import csv
from glob import glob
rcs = sorted(glob("${RESULT_ROOT}/*sub-08*/result.csv"))
if rcs:
    with open(rcs[-1]) as f:
        summary["nb_retrieval_result_csv"] = list(csv.DictReader(f))
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
echo "[OK] ${OUT}/summary.json"
