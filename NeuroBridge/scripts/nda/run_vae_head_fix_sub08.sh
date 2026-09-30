#!/usr/bin/env bash
# FIX adjudication (sub-08): VAE head input upgrade 512 z_ret -> concat(z_ret512, rag_soft5_clip1024)=1536
# Compare new LL-init decodes vs anchors:
#   old single-point  sdedit_ll_s082 Pix .1774 / hs_c040_s082 Pix .1871 (input 1024 z_decode_vith)
#   full10 uniform    sdedit_ll_sub08 Pix .1651 / hcs_s08         Pix .1735 (input 512 z_ret)
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
STAG="sub-08"
SOUT="${NB_ROOT}/outputs/vae_head_fix/${STAG}"
HCMA="${NB_ROOT}/outputs/hcma_10subj/${STAG}"
VC="${NB_ROOT}/outputs/sdedit_ll_full10/shared/vae_cache"
STD7="${NB_ROOT}/outputs/standard7_protocol"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
mkdir -p "${SOUT}/head_concat" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" || true
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
require "${HCMA}/zret/z_ret_train.npy"
require "${HCMA}/memory/rag_soft5_train_clip_1024.npy"
require "${VC}/train_vae_latents_f16.npy"

echo "===== [0] Build concat train/test (512+1024=1536) @ $(date -Iseconds) ====="
FC="${SOUT}/feats"
mkdir -p "${FC}"
if [[ ! -f "${FC}/concat_train.npy" ]]; then
  "${PYTHON}" - <<PY
import numpy as np
z=np.load("${HCMA}/zret/z_ret_train.npy").astype(np.float32)
r=np.load("${HCMA}/memory/rag_soft5_train_clip_1024.npy").astype(np.float32)
zt=np.load("${HCMA}/zret/z_ret_test.npy").astype(np.float32)
rt=np.load("${HCMA}/memory/rag_soft5_test_clip_1024.npy").astype(np.float32)
assert len(z)==len(r) and len(zt)==len(rt)
np.save("${FC}/concat_train.npy", np.concatenate([z,r],axis=1).astype(np.float32))
np.save("${FC}/concat_test.npy",  np.concatenate([zt,rt],axis=1).astype(np.float32))
print("[OK] concat", np.concatenate([z,r],axis=1).shape, np.concatenate([zt,rt],axis=1).shape)
PY
fi
require "${FC}/concat_train.npy"

echo "===== [1] Train VAE head C: 1536-dim concat @ $(date -Iseconds) ====="
HEAD="${SOUT}/head_concat"
if [[ ! -f "${HEAD}/vae_head_report.json" ]]; then
  "${PYTHON}" scripts/nda/train_eeg_vae_head.py \
    --eeg-train-npy "${FC}/concat_train.npy" \
    --eeg-test-npy "${FC}/concat_test.npy" \
    --vae-train-npy "${VC}/train_vae_latents_f16.npy" \
    --vae-test-npy "${VC}/test_vae_latents_f16.npy" \
    --output-dir "${HEAD}" \
    --num-epochs 80 \
    --batch-size 64 \
    --lr 3e-4 \
    --device "${DEVICE}" \
    --decode-rgb
else
  echo "[SKIP] head C already trained"
fi
LL="${HEAD}/pred_lowlevel_rgb_512"
require "${LL}/199.png"

EMB="${HCMA}/ft/embeds/blend_nda_cfm_f_a40_test.npy"
PROMPT="${NB_ROOT}/outputs/hcma_10subj/prompts/prompts_full_hcma_test.json"
DEPTH_RGB="${NB_ROOT}/outputs/hcma_s/sub-08/depth/pred_depth_rgb_512"
require "${EMB}"; require "${PROMPT}"; require "${DEPTH_RGB}/199.png"

echo "===== [2] Decode A: sdedit_ll (concat init, s=0.82) @ $(date -Iseconds) ====="
GA="${SOUT}/generation/sdedit_ll_concat"
if [[ ! -f "${GA}/generated/199.png" ]]; then
  rm -rf "${GA}"
  "${PYTHON}" scripts/nda/generate_atm_aligned_decode.py \
    --mode sdedit \
    --embed-npy "${EMB}" \
    --prompts-json "${PROMPT}" \
    --lowlevel-rgb-dir "${LL}" \
    --output-dir "${GA}" \
    --tag "sdedit_ll_concat" \
    --strength 0.82 \
    --ip-scale 1.0 \
    --gen-steps 28 \
    --gen-guidance 5.0 \
    --seed 42
else
  echo "[SKIP] sdedit_ll_concat"
fi
require "${GA}/generated/199.png"

echo "===== [3] Decode B: dual hs_c040_s082 (concat init + depth CN) @ $(date -Iseconds) ====="
GB="${SOUT}/generation/hs_c040_s082_concat"
if [[ ! -f "${GB}/generated/199.png" ]]; then
  rm -rf "${GB}"
  "${PYTHON}" scripts/nda/generate_hcma_s_decode.py \
    --embed-npy "${EMB}" \
    --prompts-json "${PROMPT}" \
    --depth-rgb-dir "${DEPTH_RGB}" \
    --lowlevel-rgb-dir "${LL}" \
    --output-dir "${GB}" \
    --tag "hs_c040_s082_concat" \
    --cn-scale 0.40 \
    --ip-scale 1.0 \
    --strength 0.82 \
    --gen-steps 28 \
    --gen-guidance 5.0 \
    --seed 42
else
  echo "[SKIP] hs_c040_s082_concat"
fi
require "${GB}/generated/199.png"

echo "===== [4] Standard-7 eval (2 new rows, cache warm) @ $(date -Iseconds) ====="
"${PYTHON}" - <<PY
import json
from pathlib import Path
std7 = Path("${STD7}")
m = json.loads((std7 / "manifest_full10_hcs.json").read_text(encoding="utf-8"))
# drop previous fix rows if any
m["rows"] = [r for r in m["rows"] if not r["tag"].startswith("concat_")]
m["rows"].extend([
 {"tag": "concat_sdedit_sub08", "group": "vaehead-fix", "display": "sdedit_ll concat-1536 (sub-08)", "gen_dir": "${GA}/generated"},
 {"tag": "concat_dual_sub08",  "group": "vaehead-fix", "display": "HCMA-S hs_c040_s082 concat-1536 (sub-08)", "gen_dir": "${GB}/generated"},
])
(std7 / "manifest_vaehead_fix.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
print("[OK] manifest rows =", len(m["rows"]))
PY
cp -f "${STD7}/results.json" "${STD7}/results_33row_backup.json"
"${PYTHON}" scripts/nda/eval_standard7.py \
  --manifest "${STD7}/manifest_vaehead_fix.json" \
  --images-root "${IMAGES_ROOT:-/project/peilab/why/data/images_set}" \
  --out-dir "${STD7}" \
  --device "${DEVICE}" \
  --batch-size 16

echo "===== [5] Compare table @ $(date -Iseconds) ====="
"${PYTHON}" - <<PY
import json
r = json.loads(Path("${STD7}/results.json").read_text(encoding="utf-8"))
by = {x["tag"]: x for x in r["rows"]}
keys = ["pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "swav"]
picks = ["official_atm_sub08", "sdedit_ll_s082", "hs_c040_s082", "sdedit_ll_sub08", "hcs_s08", "concat_sdedit_sub08", "concat_dual_sub08"]
print(f"{'tag':24s} " + " ".join(f"{k:>8s}" for k in keys))
for t in picks:
    x = by.get(t)
    if not x: print(t, "MISSING"); continue
    print(f"{t:24s} " + " ".join(f"{x[k]:8.4f}" for k in keys))
PY
echo "===== DONE vae_head_fix ${STAG} @ $(date -Iseconds) ====="
