#!/usr/bin/env bash
# The EEGiT-consistency gate: pure 200-way retrieval on sub-08, no generation.
#
# Why this exists
# ---------------
# The semantic side was supposed to BE EEGiT. It reads 46.50 test Top-1 against
# EEGiT's reported 70.4, and "it is a faithful reproduction" plus "it is 24 points
# behind" cannot both be true. So the released code was read line by line and the
# differences were enumerated; they are listed in `--compare` below. This script
# runs the cheap readout (retrieval only -- no VAE, no diffusion, no seven
# metrics) so the gap can be attributed BEFORE an hour of generation is spent.
#
# Retrieval is the right readout for this question because it is the only metric
# EEGiT reports, so it is the only one where a like-for-like number exists. The
# seven-metric generation evaluation cannot answer "did we reproduce EEGiT".
#
# What was found in the official code, and what each arm turns on
# --------------------------------------------------------------
# Four discrepancies, in rough order of expected size:
#
# 1. The EEG patch image layout. Official `use_kinematic` + `spatial_interpolate`
#    build H=time, W=regions, anterior->posterior, in the dataset's own channel
#    order, with ONE 2D bilinear `F.interpolate` over each region's (time,
#    electrode) plane. Every arm run before `--patch-style eegit_official` existed
#    built the TRANSPOSE: H=regions, W=time, posterior->anterior, electrodes
#    sorted by montage x, 1D interpolation along electrodes. Same 70 tokens, same
#    tensor shape, different tensor -- so the pretrained conv saw a different image
#    and nothing raised.
#
# 2. `pos_embed` resampling. When a model is built at an `img_size` other than its
#    pretrained cfg, timm resizes the checkpoint's `pos_embed` through
#    `resample_abs_pos_embed`, which defaults to `antialias=True`. Our
#    reimplementation of that function omitted the antialias filter: max absolute
#    deviation 9.38 in `pos_embed`, 2.61 in the pooled feature. This is in the
#    shared encoder, so it affected every `eegit`-tokenizer arm, including the
#    46.50 one. Fixed in `encoders.resample_pos_embed`, verifiable by
#    `scripts/test_eegit_official_interface.py`.
#
# 3. The loss. Official `ClipLoss` + `PLModel.forward`:
#      * `logit_scale = softplus(log(1/0.07))` = 2.727 -> the effective temperature
#        is 0.367, not 0.07. The paper's prose says tau=0.07; the code softpluses
#        it, which softens the objective ~5x. (`--softplus`)
#      * `self.logit_scale` is a Parameter but is NOT in the optimizer's parameter
#        list (`configure_optimizers` passes three explicit groups), so tau is
#        FIXED, not learned. (`--fixed-temp`)
#      * Only the image side is L2-normalised before the loss; the raw EEG
#        embedding goes in, so its norm is a free per-sample logit scale.
#        (`--no-eeg-l2norm`)
#    The 46.50 run had none of the three.
#
# 4. The optimizer. Official `torch.optim.Adam(..., weight_decay=1e-4)` over three
#    groups: EEG encoder at `lr*10`, image encoder at `lr`, both ProjectionHeads at
#    `lr*10`; launched with `--lr 5e-6`, i.e. 5e-5 for the EEG encoder and the
#    heads. Flat, no warmup, no decay. Our 46.50 run used AdamW, 5e-4 for the new
#    parts and 5e-5 for the blocks, with 5-epoch warmup then cosine to 1%.
#    `--optimizer adam --wd-all-params --lr 5e-5 --backbone-lr-mult 1.0` is the
#    official schedule: with every block at the same LR and `wd` on 1-D params too
#    (Adam's weight_decay is an L2 penalty inside the gradient, not decoupled).
#
# Deliberately NOT changed, and why
# --------------------------------
# * The alignment target stays the frozen CLIP ViT-H/14 `block26` (1280-d, 1024-d
#   out). Official aligns to its own TRAINABLE ViT-B/16 + ProjectionHead, i.e. to
#   a space that adapts to whatever the EEG can predict. Adopting that would make
#   the retrieval number incomparable to ours AND useless downstream, because the
#   generation stack consumes 1024-d CLIP joint embeddings, not EEGiT's learned
#   space. It is also the largest single reason to expect their 70.4 to be
#   unattainable here, and it is stated as such rather than tuned away.
# * 10 images per training concept. Official averages the 4 reps AND keeps only
#   image slot 0 (`train_avg: True`), so it trains on 1654 pairs; we train on
#   ~14.8k (concept, image) pairs. Ours has strictly more supervision and a
#   harder positive (any of 10 images), and changing it would confound the
#   interface comparison with a data change.
# * The 150-concept held-out selection split. Official does not select at all --
#   intra-subject it trains 100 epochs and tests `last`. Keeping our split makes
#   the number directly comparable to the 46.50 run and, if anything, favours us.
#
# The arms
# --------
#   layout  the 46.50 config verbatim, with `--patch-style eegit_official` added.
#           Isolates items 1+2: the interface repair, nothing else.
#   full    full EEGiT consistency: official layout + official head + single final
#           layer (no fusion) + all four items. The reproduction attempt.
#   fuse    `full` but keeping our 3-layer uniform fusion and the `nw` MLP head.
#           Separates "the strategy/interface" from "single-layer + official head",
#           because the 46.50 run got its win partly from fusing 8/10/12.
#
# Usage:
#   ARM=layout bash scripts/run_eegit_gate.sh --dry-run
#   ARM=full   bash scripts/run_eegit_gate.sh
set -uo pipefail

ROOT="${ROOT:-/project/peilab/why/eeg-retrieval}"
cd "${ROOT}"
PY="${PY:-/project/peilab/why/eeg-brainit/.venv/bin/python}"

ARM="${ARM:-full}"
SUBJ="${SUBJ:-8}"
OUT="${ROOT}/outputs/sub$(printf '%02d' "${SUBJ}")"
FEAT="${FEAT:-${ROOT}/outputs/features/clip_h14_layers}"
EXTRA="${EXTRA:-}"

# `TRAIN_SLOTS=0` drops nine of the ten images per concept, leaving 1654 training
# pairs instead of 16540. It is a one-variable probe of the TASK, not a reproduction
# of the official one: with 10 images per concept the objective also asks the model
# to separate the ten images of one concept from each other, an
# instance-discrimination term the one-image-per-concept retrieval protocol never
# evaluates, and this arm removes it. It is deliberately NOT motivated by false
# negatives -- measured on this split, same-concept pairs are 0.06% of all negative
# pairs, at every batch size.
#
# CORRECTION: this comment used to claim the official code "averages the repetitions
# AND keeps only image slot 0, so it trains on 1654 pairs", i.e. that removing slots
# 0..9 reproduces EEGiT. It does not. Reading the released preprocessing settles it:
# `EEGiT/preprocess/process_eeg_whiten.py` allocates `sorted_session_list =
# np.zeros((16540, 4))` and builds `merged_train` with one row per IMAGE; its
# `avg: True` path averages `loaded_data['eeg'].mean(axis=1)`, which is the
# repetition axis, and leaves all ten slots in place. So EEGiT trains on the same
# 16540 pairs we do, `avg` is about repetitions rather than slots, and this arm is
# NOT the official setting. The claim had been load-bearing for treating
# `slot0` as a fidelity arm; it is a coverage arm.
TRAIN_SLOTS="${TRAIN_SLOTS:-}"

case "${ARM}" in
  layout|full|fuse) ;;
  *) echo "[gate] ARM must be layout|full|fuse, got '${ARM}'" >&2; exit 2 ;;
esac

if [ -n "${TRAIN_SLOTS}" ]; then TAG="gate_eegit_${ARM}_slot${TRAIN_SLOTS}"
else TAG="gate_eegit_${ARM}"; fi

# ---- the blocks every arm shares -------------------------------------------
# The interface repair (1+2). Non-negotiable: without it the arm is not testing
# EEGiT, it is testing a transposed EEG image.
common=(
  --subject "${SUBJ}"
  --tag "${TAG}"
  --out-dir "${OUT}"
  --tokenizer eegit
  --patch-style eegit_official
  --patch-size 16
  --n-patches-w 14
  --channels all
  --backbone timm:vit_b16_in21k_orig   # the tag the official code names; resolves
                                       # to the same weights as vit_b16_in21k
  --target-features "${FEAT}"
  --target-layer block26
  --target-fusion single
  --epochs 100
  --patience 0                         # official runs the full 100 epochs
  --aug full
  --seed 2025
  --stage1-epochs 0                    # MMD: absent from the official code
  --fit-diagnostic                     # keeps the fit-vs-val diagnostic
)

# ---- items 3+4: the official loss and optimizer -----------------------------
# Shared by `full` and `fuse`; `layout` keeps the previous run's loss/optimizer on
# purpose, so that arm is a one-variable diff against gate_eegit_layout's baseline.
official_opt=(
  --optimizer adam
  --wd-all-params
  --lr 5e-5                            # official: 5e-6 * 10 for the EEG encoder
  --backbone-lr-mult 1.0               # official has ONE LR for the whole encoder
  --fixed-temp                         # official's logit_scale is never optimised
  --softplus                           # scale softplus(log(1/0.07)) = 2.727
  --no-eeg-l2norm                      # official normalises only the image side
  --pool mean                          # official's `global_pool="avg"`
  --batch-size 256                     # official train_batch_size
  # no --warmup-epochs, no --cosine: official has neither
)

case "${ARM}" in
  layout)
    cfg=("${common[@]}"
      # the 46.50 run's own settings, so this arm is a clean one-variable diff
      --layers 8 10 12
      --fusion-mode uniform
      --batch-size 512
      --lr 5e-4
      --backbone-lr-mult 0.1
      --warmup-epochs 5
      --cosine
      --min-lr-ratio 0.01
    )
    ;;
  fuse)
    cfg=("${common[@]}" "${official_opt[@]}"
      # keep our fusion + nw head: separates strategy from architecture
      --layers 8 10 12
      --fusion-mode uniform
    )
    ;;
  full)
    cfg=("${common[@]}" "${official_opt[@]}"
      # official reads the final block only, pools with global_pool='avg', and
      # feeds one vector to the ProjectionHead -- so no fusion module may sit
      # between the pretrained encoder and the head.
      --layers 12
      --fusion-mode none
      --head-kind eegit
      --img-head-kind eegit
      --head-drop 0.5
      --d-embed 1024                     # the paper's 768 -> 1024 FC
    )
    ;;
esac

if [ "${1:-}" = "--dry-run" ]; then
  echo "[gate:${ARM}] validating the exact flag list (no GPU, no dataset)"
  # shellcheck disable=SC2086
  "${PY}" -u scripts/nwret/train.py "${cfg[@]}" ${EXTRA} ${TRAIN_SLOTS:+--train-slots ${TRAIN_SLOTS}} \
    --validate-only
  exit $?
fi

# ---- gate: the official-interface equivalence test must pass ---------------
# This is the test that would have caught the antialias bug, and it is the only
# thing standing between "we call it EEGiT" and "it is EEGiT". Cheap (20 s, CPU).
echo "[gate:${ARM}] official-interface equivalence test"
if ! "${PY}" -u scripts/test_eegit_official_interface.py >/tmp/gate_eegit_iface.log 2>&1; then
  echo "[gate:${ARM}] ABORT: test_eegit_official_interface.py failed" >&2
  tail -30 /tmp/gate_eegit_iface.log >&2
  exit 1
fi
tail -1 /tmp/gate_eegit_iface.log

echo "[gate:${ARM}] regression tests"
if ! "${PY}" -u scripts/test_fixes.py >/tmp/gate_fixes.log 2>&1; then
  echo "[gate:${ARM}] ABORT: scripts/test_fixes.py failed" >&2
  tail -30 /tmp/gate_fixes.log >&2
  exit 1
fi
tail -2 /tmp/gate_fixes.log

# ---- gate: smoke the real config end to end --------------------------------
# The smoke shortens the run to 3 epochs, which collides with any warmup longer
# than that (`--warmup-epochs 5 >= --epochs 3` is a hard error in train.py, and it
# fired here on the `layout` arm). The smoke therefore mirrors the arm's own
# schedule rather than forcing one: an arm with warmup gets a 1-epoch warmup, an
# arm without keeps none -- so the smoke still exercises the same branch the real
# run will take.
smoke_sched=()
for a in "${cfg[@]}"; do
  if [ "${a}" = "--warmup-epochs" ]; then smoke_sched=(--warmup-epochs 1); break; fi
done
echo "[gate:${ARM}] smoke: the real config, 3 epochs, 1024 fit samples (sched: ${smoke_sched[*]:-no warmup})"
# shellcheck disable=SC2086
"${PY}" -u scripts/nwret/train.py "${cfg[@]}" "${smoke_sched[@]}" \
  ${TRAIN_SLOTS:+--train-slots ${TRAIN_SLOTS}} \
  --tag "_smoke_${TAG}" --epochs 3 --limit-samples 1024 --fit-diag-concepts 20 \
  > /tmp/gate_smoke_${ARM}.log 2>&1
SMOKE_RC=$?
if [ "${SMOKE_RC}" -ne 0 ]; then
  echo "[gate:${ARM}] ABORT: smoke run failed (rc=${SMOKE_RC})" >&2
  tail -30 /tmp/gate_smoke_${ARM}.log >&2
  exit "${SMOKE_RC}"
fi
rm -f "${OUT}/_smoke_${TAG}_result.json" "${OUT}/_smoke_${TAG}_best.pt"
echo "[gate:${ARM}] smoke passed"

# ---- the run ---------------------------------------------------------------
echo "=================================================================="
echo "[gate:${ARM}] tag=${TAG}  $(date -Iseconds)"
for a in "${cfg[@]}"; do printf '%s ' "$a"; done; echo
echo "  EXTRA: ${EXTRA:-<none>}"
echo "=================================================================="
# shellcheck disable=SC2086
"${PY}" -u scripts/nwret/train.py "${cfg[@]}" ${EXTRA} \
  ${TRAIN_SLOTS:+--train-slots ${TRAIN_SLOTS}}
RC=$?
echo "[gate:${ARM}] rc=${RC} @ $(date -Iseconds)"

# ---- the comparison this gate exists to make -------------------------------
if [ "${RC}" -eq 0 ]; then
  "${PY}" -u - "${OUT}/${TAG}_result.json" <<'PYEOF' || true
import json, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text())
t, b = r["test"], r["best_val"]
print()
print("=" * 78)
print(f"{'arm':<12} {'test@1':>8} {'test@5':>8} {'val@1':>8} {'rank':>7} {'epoch':>6}")
print("-" * 78)
print(f"{r['tag']:<12} {t['top1']:>8.2f} {t['top5']:>8.2f} {b['top1']:>8.2f} "
      f"{t['mean_rank']:>7.2f} {b['epoch']:>6}")
print("=" * 78)
ref = {"nw8_eegit70_rgn5 (previous, layout=nw, no antialias)": (46.50, 72.50),
       "EEGiT paper, intra-subject, 10-subject mean":       (70.40, 95.10),
       "linear ridge, same split and features":             (25.00, 52.50)}
print("for reference:")
for k, (a, c) in ref.items():
    print(f"  {k:<54} {a:>6.2f} {c:>8.2f}")
print()
print("reading this: the gap to 70.4 is only partly an interface question. The")
print("larger structural difference is that official EEGiT aligns EEG to its OWN")
print("TRAINABLE ViT-B/16 + ProjectionHead, so the target space bends toward what")
print("EEG can predict, while this gate aligns to frozen CLIP ViT-H/14 block26.")
PYEOF
fi
exit "${RC}"
