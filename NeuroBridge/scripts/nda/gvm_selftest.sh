#!/usr/bin/env bash
# Shape/logic validation of the GVM path on REAL data at a small row count.
#
# WHY THIS IS NOT THE OLD `gem_selftest.sh` WITH NEW FLAGS.  The old test ran 8 rows
# for 1 epoch.  That cannot exercise the three things GVM adds, because all three
# need rows and epochs to exist at all:
#
#   * M2's anchor vocabulary has a frequency floor, so 8 descriptions produce almost
#     no vocabulary and `anchor_multihot` returns an empty target -- the head would
#     be "tested" against nothing.
#   * innovation 4's schedule runs at `--sched-warmup`, so a 1-epoch run never
#     reaches the code that measures the ridge R^2 and re-weights the levels.
#   * M4's arbitration is FITTED against held-in reliability spread; with 8 rows the
#     spread is noise and the gain is meaningless.
#
# So this runs 1 024 rows for 3 epochs with the schedule armed at epoch 1.  It still
# finishes in minutes, and the numbers are still meaningless -- it is a SHAPE AND
# LOGIC test that writes to outputs/gvm/_selftest and must never be reported.
set -euo pipefail
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
source scripts/nmb/nmb_sota_v2_env.sh >/dev/null 2>&1 || true

OUT="${NB_ROOT}/outputs/gvm/_selftest"
rm -rf "${OUT}"
mkdir -p "${OUT}"
echo "===== GVM selftest @ $(date -Iseconds) ====="
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
  --out "${OUT}" --epochs 3 --text-limit 1024 --batch-size 32 \
  --d-model 96 --tf-layers 2 --attr-draws 1 --attr-swap-draws 1 \
  --n-anchor 192 --anchor-min-count 4 --anchor-topk 2 --anchor-total 5 \
  --w-anchor 1.0 --w-nvol 1.0 --w-arb 0.5 \
  --w-sched 1 --sched-warmup 1 --sched-rows 4096 \
  --device "${DEVICE:-cpu}" "$@"
echo "===== GVM selftest OK @ $(date -Iseconds) ====="
