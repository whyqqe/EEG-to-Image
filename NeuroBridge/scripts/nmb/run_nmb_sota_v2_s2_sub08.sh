#!/usr/bin/env bash
# Track S: S1 DINOv2 targets + S2 DecodeAligner (dual-teacher ViT-H finetune).
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/nb_nmb_sota_v2/sub-08}"
TARGETS="${OUT}/targets"
S2_DIR="${OUT}/s2_dual"
CKPT_RN50="${CKPT_RN50:-${NB_ROOT}/results/things_eeg/intra-subjects/20260901-074319-sub-08/checkpoint_test_best.pth}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "${TARGETS}" "${S2_DIR}"
cd "${NB_ROOT}"

echo "===== [S1] DINOv2 offline targets (optional; uses HF cache) ====="
if [[ ! -f "${TARGETS}/dinov2_train.npy" ]]; then
  # DINOv2 weights live in shared huggingface hub cache (not eeg-brainit hf)
  export HF_HUB_CACHE="/project/peilab/why/cache/huggingface/hub"
  export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
  if "${PYTHON}" scripts/nmb/nmb_build_offline_targets.py \
    --output-dir "${TARGETS}" \
    --device "${DEVICE}"; then
    echo "[OK] S1 DINOv2 targets"
  else
    echo "[WARN] S1 DINOv2 failed — continuing S2 (dual finetune does not require dinov2 npy)"
  fi
else
  echo "[SKIP] DINOv2 targets exist"
fi

echo "===== [S2] Dual-teacher decode alignment finetune ====="
if [[ ! -f "${S2_DIR}/dual_vith1024_test_clip_1024.npy" ]]; then
  "${PYTHON}" scripts/nb_adapter/finetune_nb_dual.py \
    --nb-root "${NB_ROOT}" \
    --checkpoint "${CKPT_RN50}" \
    --subject 8 \
    --output-dir "${S2_DIR}" \
    --lambda-vith 0.7 \
    --lambda-direct 0.5 \
    --num-epochs 30 \
    --batch-size 1024 \
    --learning-rate 5e-5 \
    --device "${DEVICE}"
else
  echo "[SKIP] S2 dual finetune outputs exist"
fi

echo "===== [S2] Summary ====="
"${PYTHON}" - <<PY
import json
from pathlib import Path
out = Path("${OUT}")
s2 = out / "s2_dual"
report = json.loads((s2 / "dual_report.json").read_text()) if (s2 / "dual_report.json").is_file() else {}
summary = {"pipeline": "NMB-SOTA-v2-S2", "s2_dual": report, "targets": str(out / "targets")}
(out / "summary_s2.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "===== DONE S2 track @ $(date -Iseconds) ====="
