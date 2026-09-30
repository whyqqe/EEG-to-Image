#!/usr/bin/env bash
# =============================================================================
# Stage A: localise the overfitting, one variable per arm.
#
# What this answers
# -----------------
# The `prev` run (sub-08, 63ch) peaked on the 150-concept holdout at epoch 16 and
# then decayed 9.93 points by epoch 100, while in-batch training Top-1 climbed to
# 99.98%. The fit diagnostic measured 86.73% on concepts it had seen against 33.67%
# on concepts it had not, at the SAME epoch, under the SAME protocol -- so 53 points
# of the model's capacity were already going into memorisation at the peak, and the
# remaining 84 epochs spent themselves deepening that. The question this sweep asks
# is not "is it overfitting" (measured) but WHICH SETTING makes it overfit.
#
# That question has competing answers and they are cheap to separate, because the
# failure is fast: every arm can run a short schedule (60 epochs, against the
# original's 100) and a fix still has room to show itself as a higher and later
# peak. One variable per arm, same holdout, same seed, same everything else.
#
# The arms
# --------
#   A0 ctl60    cosine over 60 epochs instead of 100       <- the schedule itself
#   A1 lr1e4    head LR 5e-4 -> 1e-4                       <- LR 5-10x the literature
#   A2 soft     softplus temperature, scale 2.73 not 14.29 <- objective sharpness
#   A3 frz8     freeze the first 8 ViT blocks              <- 151.7M trainable params
#   A4 noaug    augmentation off                           <- what is aug buying?
#   A5 b512     batch 128 -> 512                           <- negative-pool size
#
# Why each one is a candidate, in the order the evidence supports it
# -----------------------------------------------------------------
# A1: head LR 5e-4 is 5-10x every reference recipe (EEGiT 5e-5, SAMGA/Shallow
#     Alignment 1e-4, NeuroBridge 1e-4, eeg-brainit 3e-4). Our backbone/head 10x
#     RATIO matches EEGiT's 5e-5/5e-6, but the absolute value does not.
# A0: the original met its peak with the LR still at 96.8% of maximum (epoch 16 of a
#     100-epoch cosine is step 1872, lr_lambda 0.968). It then annealed for 84 more
#     epochs while the holdout fell. A schedule that spends its useful window at full
#     LR and its overfitting window at low LR is mis-set even if nothing else is
#     wrong. This arm is also the one that can turn the finding into an immediate
#     saving: if its peak matches the original's 33.67, then 100 epochs were simply
#     wasted and nothing needs fixing.
# A2: our effective logit scale is 14.29 (temperature 0.07, via `exp`). EEGiT's is
#     2.73 (softplus of the same initialisation) -- 5.24x softer, so its gradients
#     concentrate far less on the hardest negative pairs. Sharp temperatures reward
#     exactly the fine-grained distinctions that do not transfer to new concepts.
# A3: 151.7M trainable parameters against 15040 training samples is ~10k per sample
#     and ~100k per training CONCEPT, which is ample to store a per-concept template.
#     The counter-evidence is real and is why this is an arm rather than a fix:
#     EEGiT fine-tunes a full ViT-B/16 and EEG-FM-Bench reports full fine-tuning
#     beating frozen encoders. Nobody has published the frozen-vs-full ablation on
#     THINGS-EEG2, so it has to be measured here.
# A4: attribution, not a proposed fix. If removing augmentation changes nothing, then
#     the four transforms are inert at these magnitudes and one suspected cause is
#     eliminated. If it collapses the peak, augmentation is doing real work and the
#     remaining arms are competing against a working regulariser.
# A5: every reference recipe uses batch 1024. Our 128 gives a 7x smaller negative
#     pool AND 7x more gradient steps per epoch (117 vs 16), which is why "100
#     epochs" means something different here than in EEGiT. 512 is a compromise: 4x
#     the negatives while keeping a usable step count inside 60 epochs. If it helps,
#     Stage B pushes to 1024.
#
# What is deliberately NOT in this sweep
# --------------------------------------
# * The structure tower. These arms are `--struct-backbone ""`, i.e. semantic only.
#   Three reasons: the collapse was in the STRUCTURAL head, not this one; the trunk
#   is the most expensive module and removing it roughly halves the cost; and the
#   selection score was `val_top1 + 0.5*val_vae_top1`, so leaving in a head whose
#   `vae_top1` sits at 1.2% against a 0.67% chance would have injected a near-chance
#   term into every arm's checkpoint choice. With no tower, `sel` is exactly
#   `val_top1`.
# * MMD. `--stage1-epochs 0`, as in the run being diagnosed.
# * `--train-slots 0`. It removes a real instance-discrimination term the eval does
#   not ask for, but it also removes 90% of the training data, so it is a two-variable
#   change and belongs in Stage B with the confound named.
#
# The instrument that makes this a diagnosis rather than a leaderboard
# -------------------------------------------------------------------
# Every arm runs `--test-every 5`, which records the 200-way TEST score as
# `test_top1_diag` in `history`. This is the measurement the original run could not
# make: it kept only the best checkpoint, so "validation peaked at epoch 16" and "the
# test set peaked at epoch 16" are indistinguishable in its artefacts, and they are
# NOT the same claim -- the holdout averages 4 EEG repetitions per image and the test
# set averages 80, so the test query is roughly 4.5x cleaner and a materially easier
# task (which is why test 51.50% on 200-way exceeds val 33.67% on 150-way).
# If the two curves peak together, the holdout is a faithful proxy and the recipe is
# the problem. If test keeps climbing while the holdout decays, the selection rule is
# the problem and that has to be fixed first.
# The curve is read for diagnosis ONLY: it never touches `sel`, never writes a
# checkpoint, and the headline number stays val-selected.
#
# Reading the result
# ------------------
# Each arm writes `outputs/sub08/diag_<arm>/diag_<arm>_result.json` with `history`
# (per-epoch val + the sampled test curve), `best_val`, `test`, `test_diag` and
# `fit_diagnostic`. Compare arms on: the val peak and its epoch, the test peak and
# its epoch, and the fit-vs-holdout gap (the direct measure of how much capacity is
# going into memorisation). A change that raises the peak is a fix; one that only
# moves it later is a delay; one that leaves the gap untouched is neither.
# =============================================================================
set -uo pipefail

ROOT="${ROOT:-/project/peilab/why/eeg-retrieval}"
PY="${PY:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
FEAT="${FEAT:-${ROOT}/outputs/features/clip_h14_layers}"
SUBJ="${SUBJ:-8}"
OUT="${OUT:-${ROOT}/outputs/sub$(printf '%02d' "${SUBJ}")}"
ARM="${ARM:-ctl60}"

cd "${ROOT}"

export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
mkdir -p "${XDG_CACHE_HOME}" "${ROOT}/outputs/logs"

# ---- the blocks every arm shares -------------------------------------------
# Identical to `run_patch_dual.sh`'s `SEM=prev` semantic configuration, minus the
# structure tower. If these drift, the arms stop being comparable to the run they
# are diagnosing, so they are copied rather than re-derived.
common=(
  --subject "${SUBJ}"
  --tag "diag_${ARM}"
  --out-dir "${OUT}"
  --tokenizer eegit
  --patch-style nw                 # the layout `prev` used and the gate measured
  --patch-size 16
  --n-patches-w 14
  --channels all
  --backbone timm:vit_b16_in21k_orig
  --layers 8 10 12
  --fusion-mode uniform
  --target-features "${FEAT}"
  --target-layer block26
  --target-fusion single
  --struct-backbone ""             # semantic only -- see the header
  --epochs 60
  --patience 0                     # we WANT to see the post-peak decay
  --batch-size 128
  --lr 5e-4
  --backbone-lr-mult 0.1
  --warmup-epochs 5
  --cosine
  --min-lr-ratio 0.01
  --aug full
  --seed 2025
  --stage1-epochs 0
  --fit-diagnostic
  --test-every 5                   # the diagnostic test trajectory
  --d-embed 512
)

# ---- the one variable each arm changes -------------------------------------
case "${ARM}" in
  ctl60)
    # The control. Note the only difference from the diagnosed run is the schedule
    # length: cosine over 60 epochs rather than 100, which changes the LR at every
    # step, so this arm is the schedule hypothesis AND the baseline the others are
    # read against. That conflation is deliberate -- a control that reproduced 100
    # epochs at 100 epochs would cost 1.7h per arm and could not be run six ways.
    ;;
  lr1e4)
    common+=(--lr 1e-4)
    ;;
  soft)
    common+=(--softplus --fixed-temp)
    ;;
  frz8)
    common+=(--freeze-blocks 8)
    ;;
  noaug)
    common+=(--aug none)
    ;;
  b512)
    common+=(--batch-size 512)
    ;;
  *)
    echo "[diag] ARM must be ctl60|lr1e4|soft|frz8|noaug|b512, got '${ARM}'" >&2
    exit 2
    ;;
esac

D="${OUT}/diag_${ARM}"
rm -f "${D}/diag_${ARM}_best.pt"
mkdir -p "${D}"

echo "########## diag arm ${ARM} sub-${SUBJ} @ $(date -Iseconds) ##########"
echo "[diag] args: ${common[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"${PY}" -u "${ROOT}/scripts/nwret/train.py" "${common[@]}" 2>&1 | tee "${ROOT}/outputs/logs/diag_${ARM}.log"
RC=${PIPESTATUS[0]}
echo "########## diag arm ${ARM} rc=${RC} @ $(date -Iseconds) ##########"
exit "${RC}"
