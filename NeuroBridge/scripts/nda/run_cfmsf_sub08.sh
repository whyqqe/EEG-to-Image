#!/usr/bin/env bash
# CF-MSF on sub-08: concept-factorized multi-route score fusion for 200-way retrieval.
#
# Stages
#   1) train 4 heads (img / text / depth / edge) with gallery NCE on leak-free fit
#   2) fuse scores + CSLS + Sinkhorn ablations; leave-one-route-out; neural-address control
#
# Protocol notes (frozen for this run)
#   - per-subject (sub-08 only)
#   - 200-way zero-shot (test concepts disjoint from train gallery)
#   - checkpoint on val_b concepts; test never used for selection
#   - primary = uniform-weight fusion + CSLS; Sinkhorn reported separately
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" >/dev/null 2>&1 || true

export CFMSF_OUT="${CFMSF_OUT:-${NB_ROOT}/outputs/cfmsf_s08}"
OUT="${CFMSF_OUT}"
PYTHON="$(command -v python)"
if ! "${PYTHON}" -c "import torch" >/dev/null 2>&1; then
  PYTHON=/project/peilab/why/eeg-brainit/.venv/bin/python
fi
[[ -x "${PYTHON}" ]] || { echo "[FATAL] no python with torch"; exit 1; }
echo "[env] PYTHON=${PYTHON}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-80}"
INST_W="${INST_W:-0.2}"
ROUTES="${ROUTES:-img,text,depth,edge}"

mkdir -p "${OUT}"/{logs,train,fuse} "${NB_ROOT}/outputs/slurm"
log() { echo "[$(date -Iseconds)] $*"; }
require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1"; exit 1; }; }

require outputs/ocf/intra_z/sub-08/shared_r_train.npy
require outputs/ocf/intra_z/sub-08/shared_r_test.npy
require outputs/uck/shared/g_img_concept.npy
require outputs/gem/cond_cache/clip_img1024_train.npy
require outputs/gem/cond_cache/clip_depth1024_train.npy
require outputs/gem/cond_cache/clip_edge1024_train.npy
require outputs/gem/cond_cache/clip_img1024_test.npy
require outputs/gem/cond_cache/clip_depth1024_test.npy
require outputs/gem/cond_cache/clip_edge1024_test.npy
require outputs/nda_ss/sub-08/clip_text/train/text_concept_clip.npy
require outputs/nda_ss/sub-08/clip_text/test/text_concept_clip.npy
require outputs/leakfree/split.json
require outputs/g2/captions/captions_train.jsonl

# ---------------- 1. train ----------------
if [[ ! -f "${OUT}/train/train_report.json" ]]; then
  log "===== CF-MSF train (routes=${ROUTES}, epochs=${EPOCHS}) ====="
  "${PYTHON}" scripts/nda/cfmsf_train.py \
      --out "${OUT}/train" --test-subject 8 --routes "${ROUTES}" \
      --epochs "${EPOCHS}" --inst-weight "${INST_W}" --device "${DEVICE}" \
      2>&1 | tee "${OUT}/logs/train.log"
fi
require "${OUT}/train/train_report.json"
for r in ${ROUTES//,/ }; do
  require "${OUT}/train/conds/q_${r}_test.npy"
done

# ---------------- 2. fuse + eval ----------------
log "===== CF-MSF fuse + ablations ====="
"${PYTHON}" scripts/nda/cfmsf_fuse_eval.py \
    --train-out "${OUT}/train" --out "${OUT}/fuse" --routes "${ROUTES}" \
    2>&1 | tee "${OUT}/logs/fuse.log"

# ---------------- 3. summary ----------------
log "===== summary ====="
"${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
OUT = Path(os.environ["CFMSF_OUT"])
tr = json.loads((OUT/"train/train_report.json").read_text())
fu = json.loads((OUT/"fuse/fuse_report.json").read_text())
print("=== train (val_b concept-gallery top1 / neural-address) ===")
for r, d in tr["routes"].items():
    na = d["neural_address_val"]
    te = d.get("test200_raw", {})
    print(f"  {r:<6} val_top1={na['paired_top1']:.4f}  shuf={na['shuffled_top1']:.4f}  "
          f"gain={na['gain']:+.4f}  | test200_raw top1={te.get('top1', float('nan')):.4f}")
print("\n=== fuse primary ===")
p = fu["primary_metrics"]
ps = fu["primary_plus_sinkhorn"]
print(f"  primary={fu['primary']}")
print(f"  top1={p['top1']:.4f}  top5={p['top5']:.4f}  mean_rank={p['mean_rank']:.2f}  hub={p['hub_skew']:.2f}")
print(f"  +sinkhorn top1={ps['top1']:.4f}  top5={ps['top5']:.4f}  (transductive, reported separately)")
print("\n=== key arms ===")
keys = [k for k in fu["arms"] if not k.endswith("+sinkhorn")]
print(f"{'arm':<28}{'top1':>8}{'top5':>8}{'rank':>8}")
for k in keys:
    v = fu["arms"][k]
    print(f"{k:<28}{v['top1']:>8.4f}{v['top5']:>8.4f}{v['mean_rank']:>8.2f}")
summ = {"pipeline": "cfmsf_s08", "train": tr, "fuse": {
    "primary": fu["primary"], "primary_metrics": p,
    "primary_plus_sinkhorn": ps, "arms": fu["arms"],
    "temperature_grid_csls": fu.get("temperature_grid_csls"),
}}
(OUT/"summary.json").write_text(json.dumps(summ, indent=2))
print(f"\n[summary] -> {OUT/'summary.json'}")
PY
log "===== done ====="
du -sh "${OUT}" | sed 's/^/[disk] /'
