#!/usr/bin/env bash
# NW-v8: ONE configuration for sub-08. No sweep.
#
# Why a single run instead of the arm sweep
# -----------------------------------------
# The sweep ranked arms that were all confined to a narrow band (16-21.5 test
# Top-1, then 33.5-38.5 once the alignment layer was corrected) while a
# closed-form linear ridge reached 25.0 on the same features and split. Each arm
# was a hyperparameter change to a *broken interface*: the EEG tokenizer learned
# a randomly-initialised projection on per-cell waveforms and never called the
# pretrained `patch_embed` at all. Tuning above a broken interface buys noise.
# So this run fixes the interface and spends the one shot there.
#
# What is fixed here (A1-A5, B1-B2, C3 from the analysis)
# ------------------------------------------------------
# A1  EEGiT patch representation. EEG signals become an 80x224 "EEG image"
#     (5 anatomically defined regions linearly interpolated to 16 channels each
#     x 14 time patches of 16 samples), replicated to 3 planes and fed through
#     the PRETRAINED Conv2d(3,768,16,16) patch_embed. 70 tokens, exactly EEGiT's
#     "14 x 5 = 70 spatiotemporal patches". Their ablation prices this
#     representation at +16.4 intra-subject Top-1 (54.0 -> 70.4) and the
#     pretrained weights at +6.8, so it is the largest single item on the list.
# A2  The degenerate grid is gone with it. The old path placed 17
#     occipito-parietal electrodes on a fixed 7x7 grid spanning lim=1.05, which
#     is a full-head extent: ~57% of the tokens were extrapolations and largely
#     redundant, spending model capacity on noise.
# A3  The positional-embedding distortion shrinks with it. pos_embed is resampled
#     14x14 -> 5x14; along one axis that is 14->5, whereas the old grid needed
#     14x14 -> 7x28, which upsampled time 2x and destroyed the temporal ordering.
# A4  The ViT's final LayerNorm is applied before pooling. The per-layer loop
#     calls the blocks directly, bypassing timm's forward_features, so the
#     alignment target was an unnormalised residual stream -- plainly wrong for
#     block 12, the depth every winning arm used, and inconsistent across depths,
#     which LayerFusion depends on.
# A5  MMD is fixed (unit-normalised inputs + median-heuristic bandwidths). With
#     sigmas pinned at (1,2,4,8) on 512-d embeddings its value was the constant
#     2/N and its gradient was exactly zero, so the earlier reading that
#     "SAMGA's MMD warm-up hurts here" measured its ABSENCE, not MMD. It stays
#     off in this run, because that judgement now has to be re-earned.
# B1  Step budget: 100 epochs (EEGiT's) x 29 steps = ~2900 optimisation steps
#     for 86M parameters, against ~800 before. Early stopping is disabled so the
#     cosine schedule actually completes.
# B2  Layer-wise LR with linear warmup then cosine. One LR for everything was a
#     defect when the model mixes randomly-initialised modules (patch_embed is
#     now the interface, plus the heads) with a pretrained 86M backbone. Blocks
#     run at 5e-5 -- EEGiT's encoder LR -- and the new parts at 5e-4.
# C3  A fit diagnostic: after checkpoint selection, 150 FIT concepts are scored
#     with the identical protocol as validation. That is what separates "cannot
#     fit the training set" (an optimisation/capacity problem) from "fits but
#     does not generalise" (a regularisation problem) -- currently the two are
#     indistinguishable from the val curve alone.
#
# Also fixed: the cls slot now carries `cls_token + pos_embed[0]` as timm
# assembles it. It previously held pos_embed[0] alone, dropping a pretrained
# parameter from the input entirely.
#
# What is deliberately NOT changed, and why
# -----------------------------------------
# * Target layer stays block26. It is the one axis already confirmed by a
#   closed-form probe on validation (+13.0 test Top-1 over the final layer), and
#   its per-layer profile is an inverted U with a band around the peak, so this
#   is a real choice rather than a val-noise artefact.
# * EEG-side layer fusion stays uniform over 8/10/12 (the best measured arm,
#   38.50). Uniform means fixed 1/k weights, so it cannot overfit the selection.
# * MMD stays off, softplus stays off, learnable temperature stays ON, and
#   augmentation stays `full`: all four are previously measured here and the
#   temperature default in particular was left alone on purpose, since flipping
#   it would confound the tokenisation change with a loss change.
# * No block freezing. EEGiT fine-tunes all of ViT-B/16 with this interface and
#   reaches 70.4, so capacity is not obviously the binding constraint now that
#   the interface works. `--freeze-blocks` is the next lever if the fit
#   diagnostic says the model cannot fit.
# * `--channels all` (63). EEGiT's region grouping needs the 5 canonical
#   regions; the 17-channel occipito-parietal subset is exactly their parietal +
#   occipital pair, which makes only 2 regions -> 28 tokens, and it makes the
#   pos_embed resample 14->2 instead of 14->5. HANDOFF section 7 measured the
#   17-vs-63 difference as not significant, and EEGiT's Fig.7 ablation reports
#   that retaining the occipital region matches the full montage, so widening
#   costs little -- and reproducing EEGiT's exact 70-patch geometry is what
#   makes their +16.4 transferable evidence for this run.
#
# Honest limits of a single run
# -----------------------------
# n=1 subject, n=1 seed: the difference from 38.50 is not a significance claim,
# and this run CANNOT by itself separate "the patch representation helped" from
# "the LR schedule helped" -- every arm of the design is turned on at once. The
# control that licenses attribution is a `--no-pretrained` run at identical
# settings: EEGiT's own table puts the weights at +6.8, and without that control a
# good number here could come from the interface, the schedule, or both.
#
# Usage:
#   bash scripts/run_nw8.sh --dry-run    # validate flags, touch nothing
#   bash scripts/run_nw8.sh              # the real run
set -uo pipefail

ROOT="${ROOT:-/project/peilab/why/eeg-retrieval}"
cd "${ROOT}"
PY="${PY:-/project/peilab/why/eeg-brainit/.venv/bin/python}"

SUBJ="${SUBJ:-8}"
OUT="${ROOT}/outputs/sub$(printf '%02d' "${SUBJ}")"
FEAT="${FEAT:-${ROOT}/outputs/features/clip_h14_layers}"
TAG="${TAG:-nw8_eegit70_rgn5}"
EXTRA="${EXTRA:-}"

# One config. Every line below is a deliberate decision documented above.
cfg=(
  --subject "${SUBJ}"
  --tag "${TAG}"
  --out-dir "${OUT}"

  # ---- A1/A2/A3: EEGiT's patch representation through the pretrained conv ----
  --tokenizer eegit
  --patch-size 16
  --n-patches-w 14
  --channels all                 # 5 EEGiT regions -> 5x14 = 70 tokens
  --backbone timm:vit_b16_in21k

  # ---- A4 + cls token: the input interface must match what timm expects -------
  # (both are the default; stated explicitly so the record shows they were on)

  # ---- alignment target: the one axis already confirmed by a val-selected probe
  --target-features "${FEAT}"
  --target-layer block26
  --target-fusion single

  # ---- EEG-side fusion: the best measured arm, fixed 1/k weights -------------
  --layers 8 10 12
  --fusion-mode uniform

  # ---- B1: step budget -------------------------------------------------------
  --epochs 100
  --batch-size 512
  --patience 0                   # 0 = disabled, so the cosine schedule completes

  # ---- B2: layer-wise LR, warmup, cosine ------------------------------------
  --lr 5e-4                      # new parts (patch_embed interface, heads)
  --backbone-lr-mult 0.1         # blocks -> 5e-5, EEGiT's encoder LR
  --warmup-epochs 5
  --cosine
  --min-lr-ratio 0.01

  # ---- unchanged, previously measured components -----------------------------
  --aug full
  --stage1-epochs 0              # MMD off (A5 -- fix is in, decision re-earned)
  --seed 2025

  # ---- C3: diagnostics -------------------------------------------------------
  --fit-diagnostic
)

if [ "${1:-}" = "--dry-run" ]; then
  echo "[nw8] validating the exact flag list (no GPU, no dataset)"
  # shellcheck disable=SC2086
  "${PY}" -u scripts/nwret/train.py "${cfg[@]}" ${EXTRA} --validate-only
  exit $?
fi

# ---- gate: the unit tests must pass before a GPU is spent -------------------
echo "[nw8] unit tests"
"${PY}" -u scripts/test_fixes.py 2>&1 | tail -3
if ! "${PY}" -u scripts/test_fixes.py >/dev/null 2>&1; then
  echo "[nw8] ABORT: scripts/test_fixes.py failed" >&2
  exit 1
fi

# ---- gate: smoke the real config end to end on the GPU ---------------------
# A tiny run through the SAME code path: the EEGiT tokenizer, the z-score
# statistics, patch_embed, per-layer pooling, the LR groups, warmup+cosine, the
# fit diagnostic. If any of these is broken the full run would waste an hour
# before showing it. Uses --limit-samples so it cannot be mistaken for a result.
echo "[nw8] smoke: the real config, 3 epochs, 1024 fit samples"
"${PY}" -u scripts/nwret/train.py "${cfg[@]}" \
  --tag "_smoke_${TAG}" --epochs 3 --warmup-epochs 1 --limit-samples 1024 \
  --fit-diag-concepts 20 2>&1 | tail -25
SMOKE_RC=${PIPESTATUS[0]}
if [ "${SMOKE_RC}" -ne 0 ]; then
  echo "[nw8] ABORT: smoke run failed (rc=${SMOKE_RC}); not starting the real run" >&2
  exit "${SMOKE_RC}"
fi
# The smoke writes a real result JSON under a reserved tag; remove it so it can
# never be picked up by summarize.py or mistaken for the experiment.
rm -f "${OUT}/_smoke_${TAG}_result.json" "${OUT}/_smoke_${TAG}_best.pt"
echo "[nw8] smoke passed; starting the real run"

# ---- the run ---------------------------------------------------------------
echo "=================================================================="
echo "[nw8] tag=${TAG}"
echo "[nw8] $(date -Iseconds)"
for a in "${cfg[@]}"; do printf '%s ' "$a"; done; echo; echo "  EXTRA: ${EXTRA:-<none>}"
echo "=================================================================="
# shellcheck disable=SC2086
"${PY}" -u scripts/nwret/train.py "${cfg[@]}" ${EXTRA}
RC=$?
echo "[nw8] rc=${RC} @ $(date -Iseconds)"

if [ "${RC}" -eq 0 ]; then
  echo "[nw8] leaderboard"
  "${PY}" -u scripts/nwret/summarize.py --subject "${SUBJ}" 2>&1 | tail -40 || true
fi
exit "${RC}"
