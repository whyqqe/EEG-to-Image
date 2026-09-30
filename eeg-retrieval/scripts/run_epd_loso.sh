#!/usr/bin/env bash
# =============================================================================
# run_epd_loso -- the inter-subject (LOSO) arm: nine subjects in, one held out
# =============================================================================
# This is the first run in this project that answers the question the SOTA table
# actually asks. Every previous arm trained on ONE subject and was scored against
# intra-subject numbers; the inter-subject column is four times harder (SAMGA 26.22
# vs its own 91.3 intra-subject), and no amount of tuning an intra-subject pipeline
# reaches it. See PROTOCOL_INTER.md for the transcription of the protocol, which is
# not inferred -- it is copied from SCORE's, SATTC's and SAMGA's setup sections.
#
# WHAT IS FIXED BY THE PROTOCOL (not tunable here)
# ------------------------------------------------
#   * all 63 channels. Not 17: the inter-subject literature's most reproducible
#     finding is that anterior channels HELP across subjects (+6.1 Top-1 for SIMON,
#     +4.1 for NeuroBridge) while hurting within a subject (-11.7 for Shallow
#     Alignment). Carrying 17 over from the intra arms would silently discard it.
#   * --val-concepts 0. The holdout is 150 of the 1654 training concepts and the
#     published runs train on all 1654. The flag and --select-last are one decision.
#   * --select-last. SCORE: "train each model for 50 epochs, and report the final
#     epoch". Shallow Alignment: "selecting the checkpoint by test accuracy would
#     constitute test-set over-selection". Both leak-free, and the only policy left
#     once the holdout is gone.
#   * 200-way retrieval, repetitions averaged, correct pairing on the diagonal.
#
# THE ONE CHOICE THIS FILE MAKES: MVNN
# ------------------------------------
# --mvnn train on the nine sources, --mvnn test on the held-out subject. That
# asymmetry is `load_loso`'s and it is the protocol rather than a shortcut: a source
# subject is whitened from its own labelled training residuals, while the held-out
# subject's training split is exactly what the fold excludes, so only its test-trial
# within-condition residuals remain. That grouping is already required to average
# repetitions, so it needs no labels. Measured on sub-08: lam 0.012 (train) and
# 0.127 (test), cond 12.0 -- well-conditioned both ways, no degenerate fit.
#
# WHAT IS DELIBERATELY INHERITED FROM THE EEGiT ARM
# -------------------------------------------------
# The EEG-side representation is EEGiT's, unchanged and already validated in this
# codebase: `--tokenizer eegit --patch-style time-region --patch-size 16
# --n-patches-w 14`, the ViT-B/16 backbone, one layer with no fusion, mean pooling,
# the `eegit` head at d_embed 1024, and EEGiT's own objective transcribed from the
# released code -- `--fixed-temp --softplus --no-eeg-l2norm`. See run_epd_sem_eegit.sh
# for why `--softplus` is not optional: the paper's prose says tau=0.07, the code
# softpluses it to an effective 0.367, and running the prose costs 5.2x in logit
# scale. That mistake was already made once and cost a 16-epoch stall at ln(128).
#
# WHAT IS NOT HERE
# ----------------
#   * no structure tower. `_validate` refuses LOSO + `--struct-backbone` because
#     AuxTargetDataset indexes its caches by row = concept*10+slot, a single-subject
#     layout, so the task targets would be read from the wrong rows. Solving that is
#     a separate piece of work and it is not on the critical path: the inter-subject
#     SOTA table is a retrieval table.
#   * no generation metrics. The bar in PROTOCOL_INTER.md section 7 is Top-1/Top-5.
#   * no test-time coordinate recovery. SCORE shows that is worth 27 points on top of
#     SAMGA, which makes it the single largest known lever -- but it is a deployment
#     step on top of a trained encoder, and there has to be an encoder first.
#
# COST
# ----
# The fit set is 9 subjects x 1654 concepts x 10 images = 148,860 rows, roughly 4.5x
# the intra-subject run's 16,540, so an epoch is ~4.5x longer. 50 epochs at a batch of
# 256 on one A100-class GPU is the estimate the sbatch time limit is set from; if it
# is short, the run is resumable from the result-json guard below.
#
# USAGE
#   ./scripts/run_epd_loso.sh --dry-run          # validate flags, no GPU
#   ./scripts/run_epd_loso.sh                    # train + evaluate the fold
#   sbatch slurm/epd_loso.sbatch
set -uo pipefail

ROOT="/project/peilab/why/eeg-retrieval"
cd "${ROOT}"

export PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"

# /home is at 100% capacity and a model download there fails with ENOSPC mid-run.
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"     # no downloads from a compute node
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# ----------------------------------------------------------------------------- config
TARGET="${TARGET_SUBJECT:-8}"
# Every other subject, in ascending order. The ORDER IS PART OF THE CONFIG: the
# per-subject embedding's row i means "the i-th subject listed here", not "sub-(i+1)",
# so a checkpoint is only meaningful next to the source list that produced it.
SOURCES=(${SOURCE_SUBJECTS:-1 2 3 4 5 6 7 9 10})
SEED="${SEED:-2025}"
EPOCHS="${EPOCHS:-50}"          # SCORE reports the final epoch of a 50-epoch run
BATCH="${BATCH:-256}"

TAG="${TAG:-loso_tgt$(printf '%02d' "${TARGET}")_mvnn}"
OUT="${ROOT}/outputs/loso/sub$(printf '%02d' "${TARGET}")"
CKPT="${OUT}/${TAG}_best.pt"
FEAT="${ROOT}/outputs/features/clip_h14_layers"

log() { printf '[loso ] %s\n' "$*" >&2; }
die() { printf '[loso ] FATAL: %s\n' "$*" >&2; exit 1; }

# ----------------------------------------------------------------------------- flags
# The EEG representation, EEGiT's, identical to the validated semantic arm.
sem=(
  --tag "${TAG}"
  --out-dir "${OUT}"
  --seed "${SEED}"
  --split-seed 2025
  # ---- inter-subject: the nine sources, one holdout --------------------------
  --source-subjects "${SOURCES[@]}"
  --target-subject "${TARGET}"
  --channels all                      # 63, not 17 -- see the header
  --val-concepts 0                    # the published runs train on all 1654
  --select-last                       # SCORE: report the final epoch
  # ---- EEGiT's EEG patch representation --------------------------------------
  --tokenizer eegit
  --patch-style time-region
  --patch-size 16
  --n-patches-w 14
  --backbone timm:vit_b16_in21k_orig
  --freeze-blocks 0
  --layers 12
  --fusion-mode none
  --pool mean
  --timm-global-pool avg
  --head-kind eegit
  --img-head-kind eegit
  --head-drop 0.5
  --d-embed 1024
  # ---- the alignment target ---------------------------------------------------
  --target-features "${FEAT}"
  --target-layer block26
  --target-fusion single
  # ---- MVNN: the inter-subject preprocessing this project was missing --------
  --mvnn train                        # resolves per role inside load_loso
  --mvnn-shrinkage lw
)

# EEGiT's objective, all three items, transcribed from the released code.
obj=(
  --fixed-temp
  --softplus
  --no-eeg-l2norm
)

# EEGiT's schedule. The first LOSO run deliberately does NOT add SAMGA's
# coarse-to-fine MMD stage: `--stage1-epochs 0` keeps the objective a single InfoNCE,
# so if the fold underperforms there is one thing to look at rather than three.
sched=(
  --optimizer adamw
  --lr 5e-4
  --backbone-lr-mult 0.1
  --warmup-epochs 5
  --cosine
  --min-lr-ratio 0.01
  --ema-decay 0.999
  --ema-warmup-steps 200
  --epochs "${EPOCHS}"
  --batch-size "${BATCH}"
  --patience 0
  --aug full
  --stage1-epochs 0
)

train_cfg=("${sem[@]}" "${obj[@]}" "${sched[@]}")

# ----------------------------------------------------------------------------- [1] dry run
if [[ "${1:-}" == "--dry-run" ]]; then
  log "validating TARGET=${TARGET} SOURCES=${SOURCES[*]} (no GPU, no dataset)"
  "${PYTHON}" -u "${ROOT}/scripts/epd/train.py" "${train_cfg[@]}" --validate-only \
    || die "flag validation failed"
  for f in "${ROOT}/scripts/epd/train.py" "${ROOT}/scripts/epd/data.py" \
           "${ROOT}/scripts/epd/mvnn.py"; do
    [[ -f "$f" ]] || die "missing ${f}"
  done
  [[ -d "${FEAT}/train" && -d "${FEAT}/test" ]] || die "missing target features under ${FEAT}"
  # The whitener must already exist, or the first training job pays for it. Fitting
  # one is minutes of single-threaded preprocessing on a GPU allocation, and it is
  # the same answer every time -- `build_mvnn_cache.py` is what pays that cost once.
  for s in "${SOURCES[@]}"; do
    [[ -f "${ROOT}/outputs/cache/mvnn_W_sub$(printf '%02d' "$s")_all63_train_lw.npy" ]] \
      || die "no MVNN whitener for source sub-$(printf '%02d' "$s"); run scripts/epd/build_mvnn_cache.py first"
  done
  [[ -f "${ROOT}/outputs/cache/mvnn_W_sub$(printf '%02d' "${TARGET}")_all63_test_lw.npy" ]] \
    || die "no MVNN whitener for held-out sub-$(printf '%02d' "${TARGET}"); run scripts/epd/build_mvnn_cache.py first"
  log "dry run ok"
  exit 0
fi

mkdir -p "${OUT}" "${ROOT}/outputs/slurm"

# ----------------------------------------------------------------------------- [2] train
# The result JSON is the completion marker, not the checkpoint. `train.py` writes
# `{tag}_best.pt` inside the epoch loop -- the first epoch always beats the +/-inf
# baseline -- and `{tag}_result.json` only at the end of `main`. A run killed at epoch
# 3 leaves a checkpoint that looks finished and is not. Guarding on both is what makes
# this step idempotent for the right reason; guarding on the checkpoint alone is how
# an earlier job (608183) printed "training skipped" and died on the missing json.
if [[ -f "${OUT}/${TAG}_result.json" && -f "${CKPT}" ]]; then
  log "[2] training already complete (result json + checkpoint), skipped"
else
  if [[ -f "${CKPT}" ]]; then
    log "[2] checkpoint present but no result json: a previous run was killed"
    log "    mid-training. Retraining from scratch and overwriting it."
    rm -f "${CKPT}"
  fi
  log "[2] training ${TAG}: sources ${SOURCES[*]} -> hold out sub-$(printf '%02d' "${TARGET}")"
  log "    ${EPOCHS} epochs, batch ${BATCH}, MVNN train/test per role, EEGiT objective"
  "${PYTHON}" -u "${ROOT}/scripts/epd/train.py" "${train_cfg[@]}" || die "training failed"
  [[ -f "${CKPT}" ]] || die "no checkpoint at ${CKPT} after training"
fi
[[ -f "${OUT}/${TAG}_result.json" ]] || die "no result json at ${OUT}/${TAG}_result.json"

# ----------------------------------------------------------------------------- [3] report
"${PYTHON}" - "${OUT}/${TAG}_result.json" "${TARGET}" <<'PY'
import json, sys

res = json.load(open(sys.argv[1]))
tgt = int(sys.argv[2])
test = res["test"]
proto = res.get("protocol", {})
print(f"\n{'=' * 74}")
print(f"INTER-SUBJECT (LOSO) RESULT -- held out sub-{tgt:02d}")
print(f"{'=' * 74}")
print(f"  trained on      : {proto.get('source_subjects')} "
      f"({proto.get('n_subjects')} subjects, per-subject z-score "
      f"{proto.get('per_subject_zscore')})")
print(f"  selection       : {proto.get('selection')}")
print(f"  retrieval       : {test['n']}-way, mean rank {test['mean_rank']:.1f}")
print()
print(f"  Top-1  {test['top1']:6.2f}   [{test['ci95'][0]:.2f}, {test['ci95'][1]:.2f}]")
print(f"  Top-5  {test['top5']:6.2f}")
print()
print("  --- the published bar, same protocol (PROTOCOL_INTER.md s7) ---")
for name, v in (res.get("reference_sota") or {}).items():
    if isinstance(v, dict) and "top1" in v:
        t5 = v.get("top5")
        print(f"  {name:<22s} Top-1 {v['top1']:>6.2f}"
              + (f"  Top-5 {t5:>6.2f}" if t5 is not None else ""))
print()
print(f"  this run resolves differences of {test['min_detectable_diff']:.2f} points "
      f"or more; anything smaller is not evidence.")
print(f"{'=' * 74}")
PY

log "[3] done -> ${OUT}/${TAG}_result.json"
