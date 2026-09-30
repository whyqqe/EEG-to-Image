#!/usr/bin/env bash
# =============================================================================
# run_samga_official_baseline -- SAMGA's OWN code, sub-08 held out
# =============================================================================
# This is not a reimplementation. It is `third_party/SAMGA/train.py` executed with
# `third_party/SAMGA/inter.sh`'s argument list, minus the subject loop. Nothing in
# third_party/SAMGA is patched: if a number comes out of here it came out of the
# published implementation, which is the entire point of having a baseline at all.
# A reimplementation that reproduces a number still leaves open "did you implement it
# the way they did"; this leaves nothing open.
#
# WHY THE SUBJECT LOOP IS NOT REPRODUCED
# --------------------------------------
# inter.sh trains all ten folds. The protocol decision here is one fold (sub-08), so
# only `--train_subject_ids 1 2 3 4 5 6 7 9 10 --test_subject_ids 8` is issued. The
# loop's per-fold arguments are otherwise copied verbatim, including the ones that
# only matter as defaults (`--stage1_mmd_start 0.9 --stage1_mmd_end 0.5`, which is NOT
# the argparse default of 0.2 -- reading the default instead of the launcher would
# reproduce a different method).
#
# WAS THE IMAGE FEATURE DEVIATION FORCED? IT WAS NOT
# ------------------------------------------------
# This header used to assert that `inter.sh` points at
#     internvit_multilevel_20_24_28_32_36
# and that we could not match it, "not on disk, and a compute node has no network".
# Both halves of that were wrong, and the cost of believing them was the headline
# deviation in every number this project has produced:
#
#   * all 16,540 training images and all 200 test images are on disk under
#     `data/images_set/`, and their listing reproduces `image_metadata.npy`'s canonical
#     sequence exactly (verified, both splits);
#   * the login node has outbound network, and `OpenGVLab/InternViT-6B-448px-V2_5` is
#     11.1 GB.
#
# So `FEATURE_SET=internvit` is now the default as soon as the arrays exist, produced by
# `scripts/epd/extract_internvit_layers.py` (via `slurm/extract_internvit.sbatch`), and
# `FEATURE_SET=clip` keeps the historical runs reproducible. `auto` picks whichever is
# complete and logs which, so a half-finished extraction cannot silently change what a
# run means.
#
# The CLIP fallback, when selected, maps the five InternViT ids onto
# `block22 24 26 28 30` of open_clip ViT-H-14 (CLS token per block, 1280-d), renamed to
# `image_{split}_layer{20,24,28,32,36}.npy`. The renaming is deliberate and it is not a
# claim that CLIP block22 IS InternViT layer20: SAMGA uses `layer_ids` in exactly two
# ways, as the STACK ORDER of the five features and as float positions for the
# subject-aware prior, and what has to be preserved is that the five are evenly spaced
# with the prior centred on the middle one. Mapping k-th-of-five to k-th-of-five keeps
# both, so `--layer_prior_center 28` still means "the middle layer" and the prior is
# identical in shape to the published one. The band differs (CLIP's 69-94% depth against
# InternViT's 44-80% of 45 layers), and that difference is real -- which is precisely
# why it had to be removed rather than described, since it is shared by no published
# result and by no competitor.
#
# REMOVED EARLIER: `--image_mid_dim 1280`
# ---------------------------------------
# An earlier version of this script passed it, on the reasoning that a 1024-wide linear
# could not sit in front of 1280-wide CLIP features. That was wrong. The image path is
#     ProjectorLinear(image_feature_dim, image_mid_dim) -> ProjectorLinear(image_mid_dim, 512)
# so `image_mid_dim` is a free intermediate width, NOT required to equal the feature
# width: 1280 -> 1024 -> 512 is well-formed. The official default (1024) was therefore
# usable as-is with either backbone, and overriding it was an unnecessary deviation from
# `inter.sh` introduced by my own misreading. It is now left alone, which also makes the
# script correct for the InternViT features the extraction produces -- 3200 -> 1024 -> 512.
#
# THE ONE DELIBERATE ADDITION: `--seed 2025`
# ------------------------------------------
# inter.sh passes no seed, so `seed_everything(None)` seeds from entropy and the
# published number is not reproducible by rerunning. A baseline you cannot rerun is a
# baseline you cannot debug, and our own arms use 2025. Passing it changes nothing
# about the method and makes a rerun an actual rerun.
#
# WHAT THIS RUN REPORTS, AND THE LEAK IN IT
# -----------------------------------------
# SAMGA's loop calls `--early_stop_patience 10` (their default; inter.sh does not
# override it) and evaluates on the TEST set once per epoch, keeping the best test
# Top-1 and stopping when it has not improved for ten epochs. So the published protocol
# selects a model on the test set -- which inflates the reported number -- and their
# `result.csv` therefore carries BOTH `top1 acc` (final epoch) and `best top1 acc`
# (selected). Both are printed below and the summary step reads both, because our arms
# use `--select-last` and comparing their best-epoch number to our last-epoch number
# would credit us a margin we did not earn.
#
# USAGE
#   ./scripts/run_samga_official_baseline.sh --dry-run   # assets + flags, no training
#   ./scripts/run_samga_official_baseline.sh --smoke     # 2 subjects, 1 epoch
#   ./scripts/run_samga_official_baseline.sh             # the real fold
#   sbatch slurm/samga_official_inter.sbatch
set -uo pipefail

ROOT="/project/peilab/why/eeg-retrieval"
SAMGA="${ROOT}/third_party/SAMGA"
PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"

TARGET="${TARGET_SUBJECT:-8}"
SEED="${SEED:-2025}"
# Seed-scoped output directory, and it is not cosmetic. `train.py` names its run
# directory from the wall clock at second granularity (`20260923-145403-sub-08`), so two
# runs launched in the same second land in the SAME directory and overwrite each other's
# logs and result.csv. A seed sweep launched as a job array does exactly that, because
# the array tasks start within milliseconds of one another. Giving each seed its own
# parent directory removes the race rather than hoping the timestamps differ, and it
# makes the seed sweep aggregatable by a glob.
OUTDIR="${OUTDIR:-${ROOT}/outputs/samga_official/inter/seed${SEED}}"
EEGDIR="${ROOT}/data/preprocessed_eeg"

# The five InternViT-6B-448px-V2_5 layers inter.sh names, sorted ascending. Both
# feature sets are relabelled onto these ids, so `--layer_prior_center 28` keeps
# meaning "the middle of the five" and the subject-aware prior is unchanged in shape.
LAYER_IDS=(20 24 28 32 36)

log() { printf '[samga] %s\n' "$*" >&2; }
die() { printf '[samga] FATAL: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- feature set
# `inter.sh` points at `internvit_multilevel_20_24_28_32_36`, and for a long time that
# was the one deviation this script could not close, because InternViT was believed
# absent from the cluster and a compute node has no network. Both halves of that belief
# turned out to be false (all 16,740 images are on disk; `HF_HOME` has the 11.1 GB of
# weights), so `internvit` is now the default as soon as the arrays exist, and `clip`
# remains reachable for the runs already recorded against it.
#
# `auto` exists so that an InternViT extraction still in flight cannot make this script
# silently change meaning: it picks whatever is actually complete and says which.
FEATURE_SET="${FEATURE_SET:-auto}"
CLIP_FEATSRC="${ROOT}/data/image_feature/clip_h14_multilevel"
INTERNVIT_FEATSRC="${ROOT}/data/image_feature/internvit_multilevel_20_24_28_32_36"
CLIP_BLOCKS=(block22 block24 block26 block28 block30)

features_complete() {
  local dir="$1" split lid
  for split in train test; do
    for lid in "${LAYER_IDS[@]}"; do
      [[ -f "${dir}/image_${split}_layer${lid}.npy" ]] || return 1
    done
  done
  return 0
}

case "${FEATURE_SET}" in
  internvit)
    features_complete "${INTERNVIT_FEATSRC}" || die "FEATURE_SET=internvit but ${INTERNVIT_FEATSRC} is incomplete"
    FEATSRC="${INTERNVIT_FEATSRC}"; FEATURE_DIM=3200 ;;
  clip)
    FEATSRC="${CLIP_FEATSRC}"; FEATURE_DIM=1280 ;;
  auto)
    if features_complete "${INTERNVIT_FEATSRC}"; then
      FEATURE_SET="internvit"; FEATSRC="${INTERNVIT_FEATSRC}"; FEATURE_DIM=3200
    else
      FEATURE_SET="clip"; FEATSRC="${CLIP_FEATSRC}"; FEATURE_DIM=1280
    fi ;;
  *) die "FEATURE_SET must be auto|internvit|clip, got '${FEATURE_SET}'" ;;
esac

MODE="run"
case "${1:-}" in
  --dry-run) MODE="dry" ;;
  --smoke)   MODE="smoke" ;;
  "")        ;;
  *) die "unknown argument '$1' (expected --dry-run, --smoke or nothing)" ;;
esac

SOURCES=()
for s in 1 2 3 4 5 6 7 8 9 10; do
  [[ "${s}" == "${TARGET}" ]] && continue
  SOURCES+=("${s}")
done
[[ ${#SOURCES[@]} -eq 9 ]] || die "expected 9 source subjects, got ${#SOURCES[@]}"

EPOCHS=50
if [[ "${MODE}" == "smoke" ]]; then
  # The subject-aware router is the one code path our own implementation never runs,
  # so it is the one most likely to fail on first contact: `build_image_teacher`
  # mixes five per-layer projections through a subject-conditioned softmax, and
  # `apply_layer_dropout` renormalises the result. A crash inside that is visible in
  # seconds here and would otherwise surface after the full data load. Two sources and
  # one epoch keep the load small; the router's shapes do not depend on subject count.
  EPOCHS=1
  SOURCES=(1 2)
  OUTDIR="${ROOT}/outputs/samga_official/inter/smoke"
  log "SMOKE: 2 source subjects, 1 epoch, no early stop -> ${OUTDIR}"
fi

# ------------------------------------------------------------------ assets
# For the InternViT set the files are already named the way the loader wants, in the
# canonical directory, so there is nothing to build. The CLIP set needs the symlinks
# below because those arrays live under `outputs/features/...` with block names.
#
# Rejecting a partial feature directory here rather than letting the loader fail later:
# `image_{split}_layer{lid}.npy` is exactly the name `prepare_multilayer_feature_dir`
# looks for, so a directory holding four of the five layers for one split is a
# configuration error that can only be diagnosed by counting files.
build_feature_dir() {
  if [[ "${FEATURE_SET}" == "internvit" ]]; then
    features_complete "${FEATSRC}" || die "incomplete InternViT feature dir: ${FEATSRC}"
    local n; n=$(ls -1 "${FEATSRC}"/image_*_layer*.npy | wc -l)
    log "features: InternViT-6B-448px-V2_5 layers ${LAYER_IDS[*]} (${FEATURE_DIM}-d), ${n} arrays"
    return 0
  fi
  mkdir -p "${FEATSRC}"
  for i in "${!CLIP_BLOCKS[@]}"; do
    local blk="${CLIP_BLOCKS[$i]}" lid="${LAYER_IDS[$i]}"
    for split in train test; do
      local src="${ROOT}/outputs/features/clip_h14_layers/${split}/${blk}.npy"
      [[ -f "${src}" ]] || die "missing ${src}"
      ln -sfn "$(readlink -f "${src}")" "${FEATSRC}/image_${split}_layer${lid}.npy"
    done
  done
  log "features: CLIP ViT-H-14 blocks ${CLIP_BLOCKS[*]} relabelled onto layers ${LAYER_IDS[*]} (${FEATURE_DIM}-d)"
}

check_assets() {
  [[ -d "${SAMGA}/module" ]] || die "SAMGA code not found at ${SAMGA}"
  [[ -x "${PYTHON}" ]] || die "python not executable: ${PYTHON}"
  [[ -f "${EEGDIR}/info.json" ]] || die "missing ${EEGDIR}/info.json"
  for s in "${SOURCES[@]}" "${TARGET}"; do
    [[ -f "${EEGDIR}/sub-$(printf '%02d' "${s}")/train.npy" ]] \
      || die "missing train.npy for sub-$(printf '%02d' "${s}")"
    [[ -f "${EEGDIR}/sub-$(printf '%02d' "${s}")/test.npy" ]] \
      || die "missing test.npy for sub-$(printf '%02d' "${s}")"
  done
}

# ------------------------------------------------------------------ the run
# inter.sh's argument list, with the two documented substitutions and the seed. The
# order follows the launcher so a diff against it is readable.
samga_argv() {
  printf '%s\n' \
    --batch_size 1024 \
    --learning_rate 1e-4 \
    --output_name "sub-$(printf '%02d' "${TARGET}")" \
    --eeg_encoder_type TSConv \
    --train_subject_ids "${SOURCES[@]}" \
    --test_subject_ids "${TARGET}" \
    --softplus \
    --num_epochs "${EPOCHS}" \
    --image_feature_dir "${FEATSRC}" \
    --text_feature_dir "" \
    --eeg_data_dir "${EEGDIR}" \
    --device cuda:0 \
    --output_dir "${OUTDIR}" \
    --eeg_aug \
    --eeg_aug_type smooth \
    --frozen_eeg_prior \
    --img_l2norm \
    --eeg_feature_dim 1024 \
    --projector linear \
    --feature_dim 512 \
    --data_average \
    --save_weights \
    --stage1_mmd_start 0.9 \
    --stage1_mmd_end 0.5 \
    --use_multilayer_router \
    --layer_ids "${LAYER_IDS[@]}" \
    --layer_prior_center 28 \
    --layer_prior_strength 1.0 \
    --router_eval_mode global \
    --seed "${SEED}"
}

# ------------------------------------------------------------------ main
log "official SAMGA baseline; hold out sub-$(printf '%02d' "${TARGET}")"
[[ "${MODE}" == "smoke" ]] || log "sources: ${SOURCES[*]}"
[[ "${MODE}" == "smoke" ]] || log "NOTE: absolute numbers are CLIP-fed, not the paper's InternViT-fed ones"

check_assets
build_feature_dir

if [[ "${MODE}" == "dry" ]]; then
  # Assert the official loader accepts the features we just built, WITHOUT training.
  # This is the failure that costs a queue slot: `prepare_multilayer_feature_dir`
  # raises FileNotFoundError on a naming mismatch and the router raises on K != len(
  # layer_ids), both of which are cheapest to catch here.
  cd "${SAMGA}"
  SAMGA_DRY_FEATSRC="${FEATSRC}" SAMGA_DRY_EEG="${EEGDIR}" \
  SAMGA_DRY_DIM="${FEATURE_DIM}" SAMGA_DRY_SET="${FEATURE_SET}" \
  "${PYTHON}" - <<'PY' || die "official loader rejected the feature dir"
import os, sys
sys.path.insert(0, '.')
import numpy as np
from train import prepare_multilayer_feature_dir
from module.dataset import EEGPreImageDataset

SRC = os.environ['SAMGA_DRY_FEATSRC']
EEG = os.environ['SAMGA_DRY_EEG']
DIM = int(os.environ['SAMGA_DRY_DIM'])
cache = prepare_multilayer_feature_dir(SRC, [20, 24, 28, 32, 36], '/tmp/samga_dry_cache',
                                       log_fn=lambda *a, **k: None)
tr = EEGPreImageDataset(subject_ids=[1], eeg_data_dir=EEG, selected_channels=[],
                        time_window=[0, 250], image_feature_dir=cache, text_feature_dir='',
                        image_aug=False, aug_image_feature_dirs=[], average=True, train=True)
te = EEGPreImageDataset(subject_ids=[8], eeg_data_dir=EEG, selected_channels=[],
                        time_window=[0, 250], image_feature_dir=cache, text_feature_dir='',
                        image_aug=False, aug_image_feature_dirs=[], average=True, train=False)
assert tr.image_features.ndim == 4, tr.image_features.shape
assert tr.image_features.shape[2] == 5, 'K != len(layer_ids)'
# Assert against the feature set actually selected, not a hardcoded width: the check
# exists to catch exactly the handoff between backbones, so hardcoding 1280 would have
# let the InternViT path pass on stale assumptions.
assert tr.feature_dim == DIM, f'feature_dim {tr.feature_dim} != {DIM} for {os.environ["SAMGA_DRY_SET"]}'
assert te.image_features.shape == (200, 1, 5, DIM), te.image_features.shape
x, f, t, s, o, i, r = tr[0]
assert f.shape == (5, DIM), f.shape
print(f'[samga] loader ok ({os.environ["SAMGA_DRY_SET"]}): train {tr.image_features.shape}, '
      f'test {te.image_features.shape}, x {tuple(x.shape)}, f {tuple(f.shape)}', file=sys.stderr)
PY
  log "dry run ok"
  exit 0
fi

mkdir -p "${OUTDIR}"
cd "${SAMGA}"
# train.py resolves `module.*` relative to cwd, so the cwd has to be the repo; every
# path handed to it above is absolute for exactly that reason.
mapfile -t ARGV < <(samga_argv)
log "argv: ${ARGV[*]}"
"${PYTHON}" train.py "${ARGV[@]}"
RC=$?
[[ "${RC}" -eq 0 ]] || die "SAMGA train.py exited ${RC}"

# ------------------------------------------------------------- the numbers
NEWEST=$(ls -1dt "${OUTDIR}"/*sub-$(printf '%02d' "${TARGET}") 2>/dev/null | head -1)
[[ -n "${NEWEST}" ]] || die "no result dir matching *sub-$(printf '%02d' "${TARGET}") under ${OUTDIR}"
log "result dir: ${NEWEST}"

# ------------------------------------------------- we used the right features, or we stop
# This is the check that decides whether the run counts, so it is a hard failure and it
# reads the ground truth rather than the intent. `train.py` infers the visual width from
# the loaded array and logs it (`train.py:317`: `image_feature_dim =
# train_dataset.image_features.shape[-1]`), so `train.log` records which backbone's
# features actually reached the model. Intent is recorded elsewhere and can be stale:
# `FEATURE_SET` says what we asked for, the stacked cache is keyed only on layer ids
# (`stacked_20_24_28_32_36`) and so is blind to the backbone, and a feature directory
# missing one `image_{split}_layer{lid}.npy` is indistinguishable by name. This line
# cannot be fooled by any of that.
#
# It is the reason the first baseline could be mistaken for a reproduction attempt: it
# reported 1280 and nothing was asserting that it should have said 3200.
if [[ "${MODE}" != "smoke" ]]; then
  ACTUAL_DIM=$(grep -o 'image raw feature dimension: [0-9]*' "${NEWEST}/train.log" 2>/dev/null \
               | head -1 | grep -o '[0-9]*$')
  if [[ -z "${ACTUAL_DIM}" ]]; then
    die "no 'image raw feature dimension' line in ${NEWEST}/train.log; cannot verify \
which image features were used, and an unverifiable run is not a reproduction"
  fi
  if [[ "${ACTUAL_DIM}" != "${FEATURE_DIM}" ]]; then
    die "image features WRONG: the model saw ${ACTUAL_DIM}-d, expected ${FEATURE_DIM}-d for \
FEATURE_SET=${FEATURE_SET}. The run is not comparable to SAMGA (InternViT-6B is 3200-d). \
A 1280 here means the CLIP substitutes reached the model."
  fi
  log "feature gate OK: model saw ${ACTUAL_DIM}-d features (FEATURE_SET=${FEATURE_SET})"
fi

if [[ -f "${NEWEST}/result.csv" ]]; then
  log "BOTH selection protocols are in this file; ours (--select-last) is 'top1 acc':"
  cat "${NEWEST}/result.csv" >&2
  log "the final-epoch test line, if you want it without the csv:"
  grep -h "top5 acc" "${NEWEST}/train.log" | tail -1 >&2 || true
else
  die "no result.csv in ${NEWEST}; the run did not reach its summary"
fi
