#!/usr/bin/env bash
# Shape/logic validation of the whole GEM path on REAL data at a tiny batch size.
#
# WHY THIS EXISTS.  `gem_train.py` had not executed once before its first Slurm
# submission, and two defects got through: a gain broadcast, and a back-projection
# that was missing from the global-direction removal.  Each one cost a ~10 minute
# submit-crash-resubmit cycle.  This runs the same code, on the same inputs, at
# `--limit-train 8 --batch-size 8`, so the ENTIRE path -- front end, trunk, all
# three towers, the teacher-forced decode, the InfoNCE terms, the backward pass,
# the ridge fits, the greedy prompt decode and every export -- executes in a couple
# of minutes.  The spherical-interpolation angle and the norm-balancing loss that
# this comment used to list are both GONE; nothing here depends on them.
#
# The numbers it produces are meaningless (8 rows).  It is a SHAPE AND LOGIC test,
# not a result: it writes to outputs/gem/_selftest and must never be reported.
set -euo pipefail
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
source scripts/nmb/nmb_sota_v2_env.sh >/dev/null 2>&1 || true

OUT="${NB_ROOT}/outputs/gem/_selftest"
mkdir -p "${OUT}"
echo "===== selftest @ $(date -Iseconds) ====="
echo "HF_HUB_CACHE=${HF_HUB_CACHE:-unset}"

python scripts/nda/gem_train.py \
  --train-subjects 8 --test-subject 8 \
  --raw-cache "${NB_ROOT}/outputs/tdm/cache" \
  --z-root "${NB_ROOT}/outputs/ocf/intra_z" \
  --targets-dir "${NB_ROOT}/outputs/g2/targets" \
  --clip-img-dir "${NB_ROOT}/outputs/gem/cond_cache" \
  --captions-dir "${NB_ROOT}/outputs/g2/captions" \
  --vae-cache "${NB_ROOT}/outputs/sdedit_ll_full10/shared/vae_cache" \
  --clip-text-dir "${NB_ROOT}/outputs/nda_ss/sub-08/clip_text" \
  --clip-patch-npy "${NB_ROOT}/outputs/tdm/clip_patch/train_patch_f16.npy" \
  --out "${OUT}" --epochs 1 --limit-train 8 --batch-size 8 --text-limit 64 \
  --d-model 96 --tf-layers 2 --attr-draws 1 \
  --device "${DEVICE:-cpu}" "$@"
echo "===== selftest OK @ $(date -Iseconds) ====="
