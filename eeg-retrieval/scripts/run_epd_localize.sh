#!/usr/bin/env bash
# =============================================================================
# STAGE 1 -- localise the shortfall in OUR framework, four arms, one question each.
#
# Read this together with `slurm/epd_anchor.sbatch` (Stage 0). Stage 0 tells us whether
# anything near the published numbers is reachable here at all. Stage 1 asks which of
# our own choices is responsible if it is, and runs at the same time because it does
# not depend on Stage 0's answer.
#
# What changed since the 6-arm sweep, and why
# -------------------------------------------
# That sweep tested five candidate causes of overfitting (lr, temperature, capacity,
# augmentation, batch size) and all five came back negative: no arm beat the control,
# two were harmful, and every arm's test score was inside +-3.5 points of every other.
# So it did not localise anything -- it established that the local leaderboard has no
# resolution. Two instrument defects came out of it and are fixed in every arm here:
#
#   1. `--ema-decay 0.999` on. Weight averaging smooths the validation curve that the
#      checkpoint choice is taken from. The sweep measured the validation peak moving
#      by up to 28 epochs between settings whose test scores were within noise, i.e. we
#      were taking an argmax over a curve whose noise was comparable to its shape.
#      EMA is the standard fix, and it doubles as a regulariser against the 53-point
#      fit-vs-holdout gap the same run measured.
#   2. Selection reads the EMA curve, and the test trajectory is scored on the EMA
#      weights too, so every arrow in the log points at the model that would ship.
#
# Both are recorded but never selected on: `--test-every 5` writes `test_top1_diag`,
# which nothing consumes. The reported number stays val-selected.
#
# The arms
# --------
#   base_ema   our current recipe + EMA. The instrument fix ALONE. If this jumps, the
#              last weeks of tuning were being read through a broken instrument and
#              nothing else needs to change.
#   ch17       base + SAMGA/EEGiT's 17 occipito-parietal channels instead of all 63.
#              Both references use the posterior subset; our own target probe found 63
#              roughly equal to 17 on semantic targets, so this is an open question
#              rather than an expected win -- and it removes ~2/3 of the input.
#   mmd        base + SAMGA's two-stage MMD (0.9 -> 0.5 over 20 epochs). The coarse
#              alignment stage is a documented part of the reference recipe and is
#              absent from our recent runs (`--stage1-epochs 0`). Note MMD and
#              --cosine are mutually exclusive in this codebase, so this arm runs on a
#              constant LR then drops to `--stage2-lr`, which is what SAMGA itself does.
#   samgaish   all of the SAMGA recipe axes transplantable into our code at once: 17
#              channels, two-stage MMD, softplus temperature, batch 512, lr 1e-4 with a
#              stage-2 drop to 5e-5, no cosine. This is the arm that should approach
#              Stage 0's number IF our implementation is faithful and the recipe is
#              what matters. It is deliberately a bundle: it is a hypothesis test
#              ("their recipe, our code"), not a per-axis measurement. The per-axis
#              arms above are what decompose it if it lands short.
#
# What is NOT here, and why
# -------------------------
# * The structural tower. Every arm is `--struct-backbone ""`. The structural head
#   collapsed (vae_top1 1.2% against a 0.67% floor) and it was never the bottleneck:
#   the reference run's validation curve is BELOW the control's at every matched epoch
#   (ep20 33.13 vs 34.87, ep40 29.47 vs 31.27, ep60 25.47 vs 29.73), so whatever that
#   tower contributed to its 51.50 test score was inside noise. Keeping it would also
#   put a near-chance term into `sel`, which is the one place noise does the most
#   damage. It returns in Stage 2 only if the semantic side is fixed first.
# * `--train-slots 0`. It removes a real task mismatch (training asks which image,
#   evaluation asks which concept) but at the cost of 90% of the data, so it is two
#   variables and belongs in Stage 2.
# * The 100-epoch schedule. The control arm showed 60 epochs beats it on validation
#   (36.27 vs 33.67) with an identical memorisation gap, so the extra 40 epochs were
#   pure cost. All arms run 60.
#
# Reading the result
# ------------------
# `python scripts/analyze_diag_sweep.py` does not read these (different naming); the
# per-arm JSON in `outputs/sub<NN>/loc_<arm>/loc_<arm>_result.json` carries `history`
# (val + val_ema + the sampled test curve), `best_val` with `is_ema`, `test.ci95` and
# `test.min_detectable_diff`. Compare arms only where the gap exceeds the printed
# `min_detectable_diff` (~7 points at this n); anything smaller is not evidence.
# =============================================================================
set -uo pipefail

ROOT="${ROOT:-/project/peilab/why/eeg-retrieval}"
PY="${PY:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
FEAT="${FEAT:-${ROOT}/outputs/features/clip_h14_layers}"
SUBJ="${SUBJ:-8}"
OUT="${OUT:-${ROOT}/outputs/sub$(printf '%02d' "${SUBJ}")}"
ARM="${ARM:-base_ema}"

cd "${ROOT}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
mkdir -p "${XDG_CACHE_HOME}" "${ROOT}/outputs/logs"

# The blocks every arm shares: the `prev` semantic configuration, semantic-only.
common=(
  --subject "${SUBJ}"
  --tag "loc_${ARM}"
  --out-dir "${OUT}"
  --tokenizer eegit
  --patch-style region-time
  --patch-size 16
  --n-patches-w 14
  --backbone timm:vit_b16_in21k_orig
  --layers 8 10 12
  --fusion-mode uniform
  --target-features "${FEAT}"
  --target-layer block26
  --target-fusion single
  --struct-backbone ""
  --epochs 60
  --patience 0                        # keep the full curve; the decay is a measurement
  --ema-decay 0.999                   # instrument fix 1
  --test-every 5                      # diagnostic trajectory, never selected on
  --fit-diagnostic
  --d-embed 512
  --seed 2025
)

case "${ARM}" in
  base_ema)
    common+=(--channels all --batch-size 128 --lr 5e-4 --backbone-lr-mult 0.1
             --warmup-epochs 5 --cosine --min-lr-ratio 0.01 --aug full)
    ;;
  ch17)
    common+=(--channels occipito_parietal --batch-size 128 --lr 5e-4
             --backbone-lr-mult 0.1 --warmup-epochs 5 --cosine --min-lr-ratio 0.01
             --aug full)
    ;;
  mmd)
    common+=(--channels all --batch-size 128 --lr 5e-4 --backbone-lr-mult 0.1
             --aug full
             --stage1-epochs 20 --mmd-start 0.9 --mmd-end 0.5
             --stage2-lr 5e-5)
    ;;
  samgaish)
    common+=(--channels occipito_parietal --batch-size 512 --lr 1e-4
             --backbone-lr-mult 1.0 --aug full
             --softplus --fixed-temp
             --stage1-epochs 20 --mmd-start 0.9 --mmd-end 0.5
             --stage2-lr 5e-5)
    ;;
  *)
    echo "[loc] ARM must be base_ema|ch17|mmd|samgaish, got '${ARM}'" >&2
    exit 2
    ;;
esac

# Fail before burning GPU time on a config the code will reject.
"${PY}" -u "${ROOT}/scripts/epd/train.py" "${common[@]}" --validate-only || exit 1

D="${OUT}/loc_${ARM}"
rm -f "${D}/loc_${ARM}_best.pt"
mkdir -p "${D}"

echo "########## loc arm ${ARM} sub-${SUBJ} @ $(date -Iseconds) ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"${PY}" -u "${ROOT}/scripts/epd/train.py" "${common[@]}" 2>&1 \
  | tee "${ROOT}/outputs/logs/loc_${ARM}.log"
RC=${PIPESTATUS[0]}
echo "########## loc arm ${ARM} rc=${RC} @ $(date -Iseconds) ##########"
exit "${RC}"
