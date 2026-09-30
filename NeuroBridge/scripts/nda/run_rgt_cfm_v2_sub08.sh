#!/usr/bin/env bash
# RGT-CFM v2: cos-first transport + encoder adapter + gen-side conditioning adapt
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
OUT="${OUT:-${NB_ROOT}/outputs/rgt_cfm_v2/sub-08}"
BANK="${BANK:-${NB_ROOT}/outputs/rgt_cfm/sub-08/bank}"
NDA_SS="${NDA_SS:-${NB_ROOT}/outputs/nda_ss/sub-08}"
CLIP_TRAIN="${BRAINIT}/outputs/atm_bridge/clip_img_train_1024.npy"
CLIP_TEST="${BRAINIT}/outputs/atm_bridge/clip_img_test_1024.npy"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "${OUT}/cfm" "${OUT}/memory" "${OUT}/adapt" "${OUT}/generation"
cd "${NB_ROOT}"

echo "===== [0] Prefetch @ $(date -Iseconds) ====="
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
"${PYTHON}" - <<'PY'
import os
os.environ.setdefault("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub")
os.environ.setdefault("OPENCLIP_CACHE_DIR", "/project/peilab/why/cache/eeg-brainit/open_clip")
os.environ.setdefault("TORCH_HOME", "/project/peilab/why/cache/eeg-brainit/torch")
import open_clip
m, _, _ = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k", device="cpu")
print("[OK] OpenCLIP ready, blocks=", len(m.visual.transformer.resblocks))
PY

test -f "${BANK}/z_ret_train_all.npy"
test -f "${NDA_SS}/train/z_decode_vith_train.npy"
test -f "${CLIP_TRAIN}"

echo "===== [1] RGT-CFM v2 train (cos-first + adapter + NDA cond) ====="
if [[ ! -f "${OUT}/cfm/rgt_train_report.json" ]]; then
  "${PYTHON}" scripts/nda/rgt_cfm_train_v2.py \
    --bank-dir "${BANK}" \
    --clip-train "${CLIP_TRAIN}" \
    --clip-test "${CLIP_TEST}" \
    --nda-decode-train "${NDA_SS}/train/z_decode_vith_train.npy" \
    --nda-decode-test "${NDA_SS}/train/z_decode_vith_test.npy" \
    --output-dir "${OUT}/cfm" \
    --target-subject 8 \
    --epochs 60 \
    --batch-size 512 \
    --lr 8e-5 \
    --adapter-lr 2e-4 \
    --hidden 2560 \
    --lambda-fm 1.0 \
    --lambda-cos 2.5 \
    --lambda-nce 0.35 \
    --lambda-nb 0.15 \
    --lift-pretrain-epochs 8 \
    --ode-steps 24 \
    --early-stop 15 \
    --device "${DEVICE}"
else
  echo "[SKIP] cfm v2 train"
fi

echo "===== [2] Memory from z_ret ====="
if [[ ! -f "${OUT}/memory/rag_soft5_test_clip_1024.npy" ]]; then
  "${PYTHON}" scripts/nmb/nmb_memory_router.py \
    --embed-dir "${OUT}/cfm/embeds" \
    --clip-train "${CLIP_TRAIN}" \
    --clip-test "${CLIP_TEST}" \
    --output-dir "${OUT}/memory" \
    --input-key proj --soft-k 5 --soft-tau 0.07
else
  echo "[SKIP] memory"
fi

echo "===== [3] Generation-side adaptation (fuse + strength/ip) ====="
if [[ ! -f "${OUT}/adapt/gen_adapt_report.json" ]]; then
  "${PYTHON}" scripts/nda/rgt_gen_adapt.py \
    --cfm-test "${OUT}/cfm/embeds/z_rgt_cfm_test.npy" \
    --cfm-train "${OUT}/cfm/embeds/z_rgt_cfm_train.npy" \
    --nda-decode-test "${NDA_SS}/train/z_decode_vith_test.npy" \
    --nda-decode-train "${NDA_SS}/train/z_decode_vith_train.npy" \
    --mem-test "${OUT}/memory/rag_soft5_test_clip_1024.npy" \
    --mem-train "${OUT}/memory/rag_soft5_train_clip_1024.npy" \
    --clip-test "${CLIP_TEST}" \
    --clip-train "${CLIP_TRAIN}" \
    --neighbor-idx "${OUT}/memory/rag_soft5_neighbor_idx_test.npy" \
    --z-ret-test "${OUT}/cfm/embeds/z_eeg_proj_test.npy" \
    --z-ret-train-gallery "${OUT}/cfm/embeds/z_eeg_proj_train.npy" \
    --output-dir "${OUT}/adapt" \
    --base-strength 0.40 \
    --base-ip-scale 1.0
else
  echo "[SKIP] gen adapt"
fi

echo "===== [4] Generation ====="
NEIGH="${OUT}/memory/rag_soft5_neighbor_idx_test.npy"
STR="${OUT}/adapt/strength_adapt.npy"
IPS="${OUT}/adapt/ip_scale_adapt.npy"
run_gen() {
  local tag="$1" emb="$2" extra=("${@:3}")
  local gdir="${OUT}/generation/${tag}"
  [[ -f "${gdir}/generated/199.png" ]] && echo "[SKIP] gen ${tag}" && return 0
  "${PYTHON}" scripts/nb_adapter/generate_rag_lowlevel.py \
    --embed-npy "${emb}" --neighbor-idx-npy "${NEIGH}" \
    --output-dir "${gdir}" --strength 0.4 --seed 42 --tag "${tag}" --skip-metrics \
    "${extra[@]}"
}
# baselines
run_gen "v2_cfm_s40" "${OUT}/cfm/embeds/z_rgt_cfm_test.npy"
run_gen "v2_nda_s40" "${NDA_SS}/train/z_decode_vith_test.npy"
run_gen "v2_fuse_s40" "${OUT}/adapt/z_gen_adapt_test.npy"
# adapted scheduling
run_gen "v2_fuse_adapt" "${OUT}/adapt/z_gen_adapt_test.npy" \
  --strength-npy "${STR}" --ip-scale-npy "${IPS}"
run_gen "v2_nda_cfm_a50_adapt" "${OUT}/adapt/blend_nda_cfm_a50.npy" \
  --strength-npy "${STR}" --ip-scale-npy "${IPS}"

echo "===== [5] Metrics ====="
"${PYTHON}" scripts/nb_adapter/eval_clip_fid.py \
  --gen-root "${OUT}/generation" \
  --tags "v2_cfm_s40,v2_nda_s40,v2_fuse_s40,v2_fuse_adapt,v2_nda_cfm_a50_adapt" \
  --output-json "${OUT}/clip_fid_metrics.json"

"${PYTHON}" - <<PY
import json
from pathlib import Path
out = Path("${OUT}")
metrics = json.loads((out/"clip_fid_metrics.json").read_text())
results = metrics.get("results", [])
best = max(results, key=lambda r: r.get("clip_cosine", 0)) if results else None
cfm = json.loads((out/"cfm/rgt_train_report.json").read_text())
adapt = json.loads((out/"adapt/gen_adapt_report.json").read_text())
summary = {
  "pipeline": "RGT-CFM-v2 cos-first + gen adapt",
  "cfm": {k: cfm.get(k) for k in ("best_epoch","cfm_final","lift_pretrain","delta_cos_vs_lift","lambdas","use_nda_cond")},
  "gen_adapt": adapt,
  "baseline_rgt_v1": 0.416,
  "baseline_nda_ss": 0.439,
  "baseline_erdc": 0.422,
  "best_gen": best,
  "all_gen": results,
}
(out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "===== DONE RGT-CFM-v2 @ $(date -Iseconds) ====="
