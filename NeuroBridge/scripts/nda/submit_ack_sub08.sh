#!/usr/bin/env bash
# Submit ACK-DT sub-08. Preflight + CPU/GPU-free syntax checks, then sbatch.
set -euo pipefail
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
mkdir -p outputs/slurm outputs/ack_s08/logs

echo "===== preflight ====="
fail=0
chk() { if [[ -e "$1" ]]; then echo "  [ok]   $1"; else echo "  [MISS] $1"; fail=1; fi; }
chk scripts/nda/ack_heads_train.py
chk scripts/nda/run_ack_sub08.sh
chk scripts/nda/build_hcma_prompts.py
chk scripts/nda/gem_calib.py
chk scripts/nda/generate_hcma_s_decode.py
chk scripts/nda/eval_official_seven_dir.py
chk slurm/ack_sub08.sbatch
chk outputs/ocf/intra_z/sub-08/shared_r_train.npy
chk outputs/ocf/intra_z/sub-08/shared_r_test.npy
chk outputs/uck/sub-08/full/conds/ip_mem_test.npy
chk outputs/uck/sub-08/full/spatial/pred_depth_rgb_512/199.png
chk outputs/uck/shared/g_img_concept.npy
chk outputs/nda_ss/sub-08/clip_text/train/text_concept_clip.npy
chk outputs/gem/cond_cache/clip_img1024_train.npy
chk outputs/gem/cond_cache/clip_img1024_test.npy
chk outputs/g2/captions/captions_train.jsonl
chk outputs/g2/captions/captions_test.jsonl
chk outputs/g2f/prompts/prompts_oracle.json
chk outputs/leakfree/split.json
# LL path: either UCK own or shared
if [[ -f outputs/uck/sub-08/full/spatial/pred_lowlevel_rgb_512/199.png ]]; then
  chk outputs/uck/sub-08/full/spatial/pred_lowlevel_rgb_512/199.png
else
  chk outputs/sdedit_ll_full10/sub-08/vae_head/pred_lowlevel_rgb_512/199.png
fi
bash -n scripts/nda/run_ack_sub08.sh || fail=1
bash -n scripts/nda/submit_ack_sub08.sh || fail=1
python3 -m py_compile scripts/nda/ack_heads_train.py || fail=1
if (( fail )); then echo "[FATAL] preflight failed"; exit 1; fi
echo "  [ok]   syntax"

echo "===== import smoke (CPU) ====="
python3 - <<'PY'
import sys
sys.path.insert(0, "scripts/nda")
from build_hcma_prompts import scene_for
from ack_heads_train import ACKHeads, sinkhorn, hcma_prompt
import torch, numpy as np
m = ACKHeads()
o = m(torch.randn(4, 1024))
assert o["q"].shape == (4, 1024) and o["logits"].shape == (4, 1654)
idx = sinkhorn(np.eye(5).astype(np.float32))
assert list(idx) == [0, 1, 2, 3, 4]
print("  [ok]   import/forward", hcma_prompt("banana")[:60])
PY

echo "===== submit ====="
EXCL="${ACK_EXCLUDE:-dgx-09}"
JID="$(sbatch --parsable --exclude="${EXCL}" slurm/ack_sub08.sbatch)"
echo "JOBID=${JID}  (excluded: ${EXCL})"
echo "  log: outputs/slurm/ack_s08_${JID}.out"
echo "  err: outputs/slurm/ack_s08_${JID}.err"
echo "  out: outputs/ack_s08"
echo "  rows: empty/deploy/pred/pred_gate/oracle/neural_nn/sinkhorn + ack_pred"
