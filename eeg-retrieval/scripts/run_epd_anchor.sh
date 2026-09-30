#!/usr/bin/env bash
# =============================================================================
# Stage 0 (anchor): run SAMGA's OWN code, on sub-08, in this environment.
#
# Why this exists
# ---------------
# This project has been comparing itself to two published numbers it has never
# reproduced on its own data: SAMGA sub-08 94.8 and EEGiT sub-08 84.8, against a local
# best of 51.50. Every "gap to SOTA" figure, every architectural argument and every
# hyperparameter choice made in the last weeks is measured against those two numbers,
# and neither has been verified here. Worse, the 6-arm sweep showed the local
# leaderboard's resolution is +-3.5 points on the 200-way test, so even the local
# ordering is not resolvable, and there is no known-good endpoint to calibrate against.
#
# Stage 0 produces that endpoint. Two arms, both running SAMGA's code path unmodified:
#
#   samga_rn50   SAMGA with its own default feature space (RN50, the argparse default
#                in its train.py) and no router. This is the recipe in the space the
#                code ships with: 2-stage MMD (0.9 -> 0.5), 17 occipito-parietal
#                channels, EEGProject encoder, linear projector to 512, softplus
#                temperature, `smooth` EEG augmentation, frozen EEG prior, batch 512,
#                lr 1e-4 / stage-2 5e-5, 60 epochs. The four augmentation feature
#                directories it looks for are all on disk and correctly shaped.
#
#   samga_clip5  the same recipe, with SAMGA's multilayer router fed by FIVE depths of
#                CLIP ViT-H/14 -- the feature space this project actually targets.
#                This is the shape of the configuration SAMGA reports 94.8 with, since
#                its intra.sh uses `--use_multilayer_router` over five depths of
#                another ViT. It also previews the first axis Stage 2 would tune.
#
# Both arms are cheap: SAMGA's EEGProject encoder is a small conv stack, not a ViT.
#
# Why the feature space differs from intra.sh for arm 2
# ----------------------------------------------------
# intra.sh uses `internvit_multilevel_20_24_28_32_36`. Those features are NOT on disk
# and extracting InternViT-6B is a sizeable side quest. CLIP ViT-H/14 is on disk, is
# the space our own semantic tower targets, and is the same family as the RN50 arm --
# so the pair brackets the recipe and the space separately. The layer choice is
# `--layer_ids 22 24 26 28 30` centred on block26 because our own closed-form layer
# probe put block26 at or near the optimum for this data; SAMGA's own layer count (5)
# and prior strength (1.0) are kept.
#
# The ONE deviation from intra.sh, and why it is information-neutral
# ----------------------------------------------------------------
# `--early_stop_patience 0` instead of the default 10. SAMGA's early stopping is driven
# by the TEST Top-1 (see train.py: `is_better` is computed from `top1_acc` on the test
# loader), so the default simply halts the run ~10 epochs after the best test score.
# Disabling it runs the same trajectory for all 60 epochs and therefore sees a
# superset of the epochs the faithful run would see: the maximum is unchanged, and we
# additionally get the full curve, which is what lets us measure the optimism below.
# Nothing else in the recipe is touched.
#
# The measurement this enables, and it is the real reason for Stage 0
# -----------------------------------------------------------------
# SAMGA has NO validation split. Its checkpoint is literally named
# `checkpoint_test_best.pth` and is saved when `top1_acc` on the 200-way TEST set sets
# a new record, so its published sub-08 number is the MAXIMUM of ~60 evaluations of the
# test set. That is a different quantity from a held-out score: a max over 60 noisy
# draws sits above the draw-mean by roughly the noise, which is ~3.5 points here. This
# script reports both, from SAMGA's own per-epoch log:
#
#   (a) best-test-epoch score   <- what the published protocol would quote
#   (b) final-epoch score       <- what a fixed schedule with no test access would give
#   (a) - (b) is the selection optimism, and it is a floor on how much of SAMGA's 94.8
#   is protocol rather than capability. If that gap is large, then our "43-point
#   shortfall" is partly a comparison between a max and a mean, and the honest target
#   is lower than 94.8 -- which changes what Stage 2 is optimising toward.
#
# Usage
#   bash scripts/run_epd_anchor.sh              # both arms
#   ARM=samga_rn50 bash scripts/run_epd_anchor.sh
# =============================================================================
set -uo pipefail

ROOT="${ROOT:-/project/peilab/why/eeg-retrieval}"
PY="${PY:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
SAMGA="${SAMGA:-${ROOT}/third_party/SAMGA}"
SUBJ="${SUBJ:-8}"
ARM="${ARM:-all}"
OUT="${OUT:-${ROOT}/outputs/anchor}"
FEATDIR="${FEATDIR:-${ROOT}/outputs/features/samga_clip5}"

cd "${ROOT}"
mkdir -p "${OUT}" "${ROOT}/outputs/logs"

# SAMGA reads EEG from a dir holding info.json + sub-NN/{train,test}.npy, which is
# exactly our data/preprocessed_eeg (a symlink to the NeuroBridge copy). Pointed at
# explicitly rather than relying on its default, so the run cannot silently train on
# some other checkout's data.
EEG_DIR="${ROOT}/data/preprocessed_eeg"

# ---- build the SAMGA-format multilayer feature dir for arm 2 ----------------
# SAMGA wants, in ONE directory, `image_{split}_layer{N}.npy`, each [Nobj, Nimg, D]; it
# stacks them along a new axis to [Nobj, Nimg, K, D] and writes that as
# `image_{split}.npy`. Our cache is `outputs/features/clip_h14_layers/{split}/blockNN.npy`
# with exactly the [Nobj, Nimg, D] shape, so symlinks suffice. Symlinks rather than
# copies because the five layers are ~1.4 GB and the conversion is pure renaming.
build_featdir() {
  local layers=(22 24 26 28 30)
  mkdir -p "${FEATDIR}"
  local missing=0
  for s in train test; do
    for l in "${layers[@]}"; do
      local src="${ROOT}/outputs/features/clip_h14_layers/${s}/block$(printf '%02d' "$l").npy"
      local dst="${FEATDIR}/image_${s}_layer${l}.npy"
      if [ ! -e "${src}" ]; then
        echo "[anchor] MISSING source layer: ${src}" >&2
        missing=1
        continue
      fi
      [ -L "${dst}" ] || ln -sf "${src}" "${dst}"
    done
  done
  if [ "${missing}" -ne 0 ]; then
    echo "[anchor] cannot build ${FEATDIR}: a source layer is absent." >&2
    echo "[anchor] Regenerate it with scripts/epd/extract_layers.py, or drop the" >&2
    echo "[anchor] samga_clip5 arm and run samga_rn50 only." >&2
    return 1
  fi
  echo "[anchor] feature dir ${FEATDIR} ready (layers ${layers[*]})"
}

# intra.sh's shared recipe, minus the encoder-specific bits that move per arm.
common=(
  --device cuda:0
  --batch_size 512
  --learning_rate 1e-4
  --stage2_learning_rate 5e-5
  --num_epochs 60
  --eeg_encoder_type "${ENC:-EEGProject}"
  --projector linear
  --feature_dim 512
  --eeg_feature_dim 1024
  --eeg_data_dir "${EEG_DIR}"
  --selected_channels P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz PO4 PO8 O1 Oz O2
  --softplus
  --eeg_aug
  --eeg_aug_type smooth
  --frozen_eeg_prior
  --img_l2norm
  --data_average
  --stage1_mmd_start 0.9
  --stage1_mmd_end 0.5
  --early_stop_patience 0
  --seed 2025
)

run_arm() {
  local arm="$1"; shift
  local d="${OUT}/${arm}"
  mkdir -p "${d}"
  # SAMGA names its log dir `%Y%m%d-%H%M%S-<output_name>` and, on a name collision,
  # DELETES the existing directory (train.py:219-226). Passing a stable output_name
  # per arm would therefore make reruns destroy their own history, so each run gets a
  # private output_dir instead and the name is left to the timestamp.
  echo "########## anchor arm ${arm} sub-${SUBJ} @ $(date -Iseconds) ##########"
  (
    cd "${SAMGA}"
    "${PY}" -u train.py --output_dir "${d}" --output_name "sub$(printf '%02d' "${SUBJ}")" "$@"
  ) 2>&1 | tee "${ROOT}/outputs/logs/anchor_${arm}.log"
  local rc=${PIPESTATUS[0]}
  echo "########## anchor arm ${arm} rc=${rc} @ $(date -Iseconds) ##########"
  return "${rc}"
}

case "${ARM}" in
  samga_rn50)
    # No --image_feature_dir: SAMGA's own default is ./data/things_eeg/image_feature/RN50
    # relative to its cwd, which does not exist here, so the RN50 directory is passed
    # explicitly. Same feature space as the default, just correctly located.
    run_arm samga_rn50 "${common[@]}" \
      --image_feature_dir "${ROOT}/data/image_feature/RN50" \
      --text_feature_dir ""
    ;;
  samga_clip5)
    build_featdir || exit 1
    run_arm samga_clip5 "${common[@]}" \
      --image_feature_dir "${FEATDIR}" \
      --text_feature_dir "" \
      --use_multilayer_router \
      --layer_ids 22 24 26 28 30 \
      --layer_prior_center 26 \
      --layer_prior_strength 1.0 \
      --router_eval_mode global
    ;;
  all)
    rc=0
    bash "$0" ARM=samga_rn50   || rc=1
    bash "$0" ARM=samga_clip5  || rc=1
    echo "[anchor] all arms done, worst rc=${rc}"
    exit "${rc}"
    ;;
  # ---- ENCODER AXIS -------------------------------------------------------
  # Same recipe, same feature space (RN50), same everything -- only the EEG encoder
  # changes. This is the axis Stage 0 + Stage 1 jointly point at, and it is cheap to
  # test here because SAMGA ships six encoders behind one flag.
  #
  # Why this is the right next measurement: `samga_clip5` runs THEIR encoder against
  # OUR target space (5 layers of CLIP ViT-H/14, router) and scores 65.0, while our own
  # frame reaches 50.5 peak on the same target space. So the target space is exonerated
  # -- the same target yields 65 in their frame and 50 in ours. The remaining suspect is
  # the encoder/interface, and `EEGTransformer` is the control that makes it testable:
  # a transformer EEG encoder inside a frame already known to produce 59-65.
  #   EEGProject  (done, samga_rn50)   59.50 peak / 56.00 final -- their default, conv
  #   EEGTransformer                   ~? -- their transformer, our architecture family
  #   EEGNet                           ~? -- a third, independent conv
  #   TSConv                           ~? -- their other conv
  # If EEGProject ~= EEGTransformer, the encoder family is NOT the lever and the gap is
  # elsewhere. If EEGProject is far ahead, the answer to "what should our EEG Patch
  # architecture be" is a convolutional or hybrid front end, and that is a concrete
  # architectural decision rather than another sweep.
  samga_enc_*)
    enc="${ARM#samga_enc_}"
    echo "[anchor] encoder axis: ${enc}"
    run_arm "samga_enc_${enc}" "${common[@]}" \
      --eeg_encoder_type "${enc}" \
      --image_feature_dir "${ROOT}/data/image_feature/RN50" \
      --text_feature_dir ""
    ;;
  *)
    echo "[anchor] ARM must be samga_rn50|samga_clip5|all, got '${ARM}'" >&2
    exit 2
    ;;
esac
