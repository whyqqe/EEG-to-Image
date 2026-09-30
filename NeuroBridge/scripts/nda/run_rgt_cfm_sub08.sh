#!/usr/bin/env bash
# RGT-CFM: multi-subject SharedSpecific z_ret → CFM transport → ViT-H gen
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
OUT="${OUT:-${NB_ROOT}/outputs/rgt_cfm/sub-08}"
SS_CKPT="${SS_CKPT:-${NB_ROOT}/outputs/nda_ss/sub-08/ss/checkpoint_ss_calib_best.pth}"
NDA_SS_OUT="${NDA_SS_OUT:-${NB_ROOT}/outputs/nda_ss/sub-08}"
CLIP_TRAIN="${BRAINIT}/outputs/atm_bridge/clip_img_train_1024.npy"
CLIP_TEST="${BRAINIT}/outputs/atm_bridge/clip_img_test_1024.npy"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
SUBJECTS="${SUBJECTS:-1,2,4,5,6,7,8,9,10}"

mkdir -p "${OUT}/bank" "${OUT}/cfm" "${OUT}/memory" "${OUT}/blend" "${OUT}/generation"
cd "${NB_ROOT}"

echo "===== [0] Prefetch @ $(date -Iseconds) ====="
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
"${PYTHON}" - <<'PY'
import os
os.environ.setdefault("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub")
os.environ.setdefault("OPENCLIP_CACHE_DIR", "/project/peilab/why/cache/eeg-brainit/open_clip")
os.environ.setdefault("TORCH_HOME", "/project/peilab/why/cache/eeg-brainit/torch")
import open_clip
print("[INFO] OpenCLIP ViT-H-14 ...")
m, _, _ = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k", device="cpu")
print("[OK] blocks=", len(m.visual.transformer.resblocks))
PY

test -f "${SS_CKPT}"
test -f "${CLIP_TRAIN}"
test -f "${CLIP_TEST}"

echo "===== [1] Multi-subject z_ret bank (SharedSpecific) ====="
if [[ ! -f "${OUT}/bank/z_ret_train_all.npy" ]]; then
  "${PYTHON}" scripts/nda/rgt_build_bank.py \
    --ss-checkpoint "${SS_CKPT}" \
    --subjects "${SUBJECTS}" \
    --output-dir "${OUT}/bank" \
    --device "${DEVICE}"
else
  echo "[SKIP] bank"
fi

echo "===== [2] RGT-CFM transport train ====="
if [[ ! -f "${OUT}/cfm/rgt_train_report.json" ]]; then
  "${PYTHON}" scripts/nda/rgt_cfm_train.py \
    --bank-dir "${OUT}/bank" \
    --clip-train "${CLIP_TRAIN}" \
    --clip-test "${CLIP_TEST}" \
    --output-dir "${OUT}/cfm" \
    --target-subject 8 \
    --epochs 50 \
    --batch-size 512 \
    --lr 1e-4 \
    --hidden 2048 \
    --lambda-fm 1.0 \
    --lambda-nce 1.5 \
    --lambda-cos 1.0 \
    --lambda-nb 0.5 \
    --ode-steps 20 \
    --early-stop 12 \
    --device "${DEVICE}"
else
  echo "[SKIP] cfm train"
fi

echo "===== [3] Memory from z_ret (preserve retrieval) ====="
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

echo "===== [4] Blends ====="
BLEND_A50="${OUT}/blend/mem_cfm_a50.npy"
BLEND_A30="${OUT}/blend/mem_cfm_a30.npy"
[[ -f "${BLEND_A50}" ]] || "${PYTHON}" scripts/nb_adapter/ensemble_embeds.py \
  --rag-npy "${OUT}/memory/rag_soft5_test_clip_1024.npy" \
  --prior-npy "${OUT}/cfm/embeds/z_rgt_cfm_test.npy" \
  --output-npy "${BLEND_A50}" --alpha 0.5
[[ -f "${BLEND_A30}" ]] || "${PYTHON}" scripts/nb_adapter/ensemble_embeds.py \
  --rag-npy "${OUT}/memory/rag_soft5_test_clip_1024.npy" \
  --prior-npy "${OUT}/cfm/embeds/z_rgt_cfm_test.npy" \
  --output-npy "${BLEND_A30}" --alpha 0.3

echo "===== [5] Generation ====="
NEIGH="${OUT}/memory/rag_soft5_neighbor_idx_test.npy"
run_gen() {
  local tag="$1" emb="$2"
  local gdir="${OUT}/generation/${tag}"
  [[ -f "${gdir}/generated/199.png" ]] && echo "[SKIP] gen ${tag}" && return 0
  "${PYTHON}" scripts/nb_adapter/generate_rag_lowlevel.py \
    --embed-npy "${emb}" --neighbor-idx-npy "${NEIGH}" \
    --output-dir "${gdir}" --strength 0.4 --seed 42 --tag "${tag}" --skip-metrics
}
run_gen "rgt_cfm_s40" "${OUT}/cfm/embeds/z_rgt_cfm_test.npy"
run_gen "rgt_linear_s40" "${OUT}/cfm/embeds/z_linear_test.npy"
run_gen "rgt_mem_cfm_a50_s40" "${BLEND_A50}"
run_gen "rgt_mem_cfm_a30_s40" "${BLEND_A30}"

echo "===== [6] Metrics ====="
METRICS="${OUT}/clip_fid_metrics.json"
"${PYTHON}" scripts/nb_adapter/eval_clip_fid.py \
  --gen-root "${OUT}/generation" \
  --tags "rgt_cfm_s40,rgt_linear_s40,rgt_mem_cfm_a50_s40,rgt_mem_cfm_a30_s40" \
  --output-json "${METRICS}"

"${PYTHON}" - <<PY
import json
from pathlib import Path
out = Path("${OUT}")
metrics = json.loads((out/"clip_fid_metrics.json").read_text()) if (out/"clip_fid_metrics.json").is_file() else {}
results = metrics.get("results", [])
best = max(results, key=lambda r: r.get("clip_cosine", 0)) if results else None
cfm = json.loads((out/"cfm/rgt_train_report.json").read_text()) if (out/"cfm/rgt_train_report.json").is_file() else {}
bank = json.loads((out/"bank/bank_report.json").read_text()) if (out/"bank/bank_report.json").is_file() else {}
nda = None
nda_p = Path("${NDA_SS_OUT}")/"summary.json"
if nda_p.is_file():
  nda = json.loads(nda_p.read_text())
summary = {
  "pipeline": "RGT-CFM multi-subject transport",
  "claim": "Retrieval-optimal z_ret → generation manifold via subject-conditioned CFM",
  "subjects": bank.get("subjects"),
  "train_n": bank.get("train_n"),
  "gap_analysis": cfm.get("gap_analysis"),
  "cfm_final": cfm.get("cfm_final"),
  "linear_diag": cfm.get("linear_diag"),
  "baseline_nda_ss": (nda or {}).get("best_gen", {}).get("clip_cosine"),
  "baseline_erdc": 0.422,
  "baseline_semtxt": 0.414,
  "best_gen": best,
  "all_gen": results,
}
(out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "===== DONE RGT-CFM @ $(date -Iseconds) ====="
