#!/usr/bin/env bash
# G2 OVERNIGHT EXTENSION: multi-subject LOSO.
#
# The first G2 job (566625) answers "does the granularity dual tower work?" on a
# single held-out subject (sub-08). One subject is an anecdote; the cross-subject
# claim needs all ten. This script adds the missing folds.
#
# Two things it deliberately does NOT do:
#   * rebuild captions or targets -- those are image-side and subject-independent,
#     already produced once by the first job and shared by every fold;
#   * evaluate -- evaluation must not run concurrently across folds, because
#     eval_standard7.py writes a shared GT feature cache and a single results.json.
#     Evaluation is a separate dependent job (submit_g2_multi.sh).
#
# Subject isolation: every artifact is namespaced by the subject directory
# (${OUT}/gen_multi/${STAG}/...). The first job's tags had no subject in them, which
# would have made two folds overwrite each other's images.
#
# Cross-subject module and EEG encoder are untouched: all folds read the same frozen
# per-subject latent `z_eeg_proj` (512-d) used by the existing HCMA pipeline.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/g2}"
TARGETS="${TARGETS:-${OUT}/targets}"
SUBJECT="${SUBJECT:?SUBJECT must be set (1..10)}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON="${PYTHON:-python}"
Z_ROOT="${Z_ROOT:-${NB_ROOT}/outputs/hcma_10subj}"
IP_TRAIN="${IP_TRAIN:-/project/peilab/why/eeg-brainit/outputs/atm_bridge/clip_img_train_1024.npy}"
IP_TEST="${IP_TEST:-/project/peilab/why/eeg-brainit/outputs/atm_bridge/clip_img_test_1024.npy}"
HCMA_PROMPTS="${HCMA_PROMPTS:-${NB_ROOT}/outputs/hcma_10subj/prompts/prompts_full_hcma_test.json}"
CUT="${CUT:-0.0625}"
EPOCHS="${EPOCHS:-40}"
BATCH="${BATCH:-512}"
# only the two rows that answer the cross-subject question; the extra ablations were
# already run for sub-08 and repeating them for nine folds would not fit the night.
GEN_VARIANTS="${GEN_VARIANTS:-direct,cfm0}"
TRAIN_TIMEOUT_MIN="${TRAIN_TIMEOUT_MIN:-420}"
WAIT_TARGETS_MIN="${WAIT_TARGETS_MIN:-240}"
EMPTY_PROMPTS_SHARED="${EMPTY_PROMPTS_SHARED:-${OUT}/prompts_empty.json}"

# ------------------------------------------------------------------- environment
# MANDATORY, and the single cause of the 2026-09-11 failure (see run_g2_pipeline.sh).
# Without it `python` is the system interpreter plus ~/.local site-packages
# (transformers 4.36 + diffusers 0.30), and the generation stage dies at import with
# "cannot import name 'EncoderDecoderCache'". Every fold trained and then produced
# zero images. Sources the venv, PYTHONNOUSERSITE=1 and the /project caches.
# shellcheck disable=SC1091
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh"

SID="$(printf "sub-%02d" "${SUBJECT}")"
if [[ "${SUBJECT}" -ge 10 ]]; then STAG="sub-${SUBJECT}"; else STAG="sub-0${SUBJECT}"; fi
# Mirror the canonical SOTA layout (root/sub-XX/generation/<tag>/generated) so
# eval_pooled_fid.py can compute the pooled number across folds without any
# reshuffling. That protocol is: fake = all folds' generations (10 x 200 = 2000),
# real = the 200 unique test images.
MROOT="${MROOT:-${OUT}/multi}"
FDIR="${MROOT}/${STAG}/generation"

mkdir -p "${OUT}/logs" "${FDIR}" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"
{
  echo "{\"pipeline\":\"g2_multi\",\"subject\":${SUBJECT},\"fold\":\"${STAG}\",\"variants\":\"${GEN_VARIANTS}\",\"started\":\"$(date -Iseconds)\"}"
} > "${FDIR}/../job_running.json"

echo "===== g2_multi ${STAG} (LOSO) @ $(date -Iseconds) ====="

# ---------------------------------------------------------------- wait for targets
# The first job produces the shared caption/target stage. This fold only needs the
# targets, which land well before that job's training does, so poll instead of taking
# a full afterok dependency on the whole first job (that would serialise ~8 hours).
WAITED=0
while [[ ! -f "${TARGETS}/g2_targets_report.json" ]]; do
  if (( WAITED >= WAIT_TARGETS_MIN )); then
    echo "[FATAL] targets not ready after ${WAIT_TARGETS_MIN} min: ${TARGETS}/g2_targets_report.json" >&2
    exit 1
  fi
  sleep 60
  WAITED=$((WAITED + 1))
  if (( WAITED % 10 == 0 )); then
    echo "[wait] ${WAITED} min for shared targets ..."
  fi
done
echo "[OK] shared targets present (waited ${WAITED} min)"

for f in perc_struct_train.npy perc_texture_train.npy sem_image_train.npy sem_overall_train.npy; do
  [[ -e "${TARGETS}/${f}" ]] || { echo "[FATAL] missing target ${f}" >&2; exit 1; }
done
for f in "${Z_ROOT}/${SID}/zret/z_eeg_proj_train.npy" "${Z_ROOT}/${SID}/zret/z_eeg_proj_test.npy" \
         "${IP_TRAIN}" "${IP_TEST}" "${HCMA_PROMPTS}"; do
  [[ -e "${f}" ]] || { echo "[FATAL] missing ${f}" >&2; exit 1; }
done

# ---------------------------------------------------------------- train (inter / LOSO)
TDIR="${OUT}/inter_loso_${STAG}"
if [[ -f "${TDIR}/g2_report.json" ]]; then
  echo "[SKIP] train inter ${STAG}"
else
  echo "===== train INTER LOSO ${STAG} (9 subjects, ${STAG} unseen) @ $(date -Iseconds) ====="
  # --resume=1 lets a preempted fold continue from its last epoch instead of
  # restarting; g2_train.py writes <out>/last.pth every epoch.
  timeout "$((TRAIN_TIMEOUT_MIN * 60))" "${PYTHON}" scripts/nda/g2_train.py \
    --protocol inter --subject "${SUBJECT}" \
    --z-root "${Z_ROOT}" --targets-dir "${TARGETS}" \
    --ip-train-npy "${IP_TRAIN}" --ip-test-npy "${IP_TEST}" \
    --out "${TDIR}" --epochs "${EPOCHS}" --batch-size "${BATCH}" \
    --device "${DEVICE}" --cut "${CUT}" --resume 1 \
    2>&1 | tee "${OUT}/logs/train_inter_loso_${STAG}.log" || {
      rc=$?
      # 124 = timeout. Keep whatever the resume file holds so a requeue can continue.
      echo "[WARN] inter training for ${STAG} exited rc=${rc}"
      [[ -f "${TDIR}/g2_report.json" ]] || { echo "[FATAL] no trained model for ${STAG}" >&2; exit 1; }
    }
fi
[[ -f "${TDIR}/g2_report.json" ]] || { echo "[FATAL] ${TDIR}/g2_report.json missing" >&2; exit 1; }

# ---------------------------------------------------------------- generate
gen_one() {
  local variant="$1"; local emb="$2"; local anchor="$3"; local cut="$4"; local prompts="$5"
  local gdir="${FDIR}/${variant}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${variant}"; return 0; fi
  local a=()
  if [[ -n "${anchor}" && "${cut}" != "0" ]]; then a=(--anchor-latent-npy "${anchor}"); fi
  "${PYTHON}" scripts/nda/generate_spectral_decode.py \
    --embed-npy "${emb}" --prompts-json "${prompts}" \
    --output-dir "${gdir}" --tag "g2_${variant}" \
    --cut "${cut}" --gamma 1.0 --start-step 0 \
    --strength 1.0 --ip-scale 1.0 \
    --gen-steps 28 --gen-guidance 5.0 --seed 42 \
    "${a[@]}" 2>&1 | tee "${OUT}/logs/gen_inter_${STAG}_${variant}.log"
}

echo "===== generate ${STAG} @ $(date -Iseconds) ====="
IFS=',' read -ra VARS <<< "${GEN_VARIANTS}"
for v in "${VARS[@]}"; do
  v="$(echo "${v}" | tr -d ' ')"
  case "${v}" in
    direct) gen_one direct      "${TDIR}/ip_direct_test.npy" "${TDIR}/lf_latent_test.npy" "${CUT}" "${HCMA_PROMPTS}" ;;
    cfm0)   gen_one cfm0        "${TDIR}/ip_cfm0_test.npy"   "${TDIR}/lf_latent_test.npy" "${CUT}" "${HCMA_PROMPTS}" ;;
    nc0)    gen_one direct_nc0  "${TDIR}/ip_direct_test.npy" ""                          "0"   "${HCMA_PROMPTS}" ;;
    np)     gen_one direct_np   "${TDIR}/ip_direct_test.npy" "${TDIR}/lf_latent_test.npy" "${CUT}" "${EMPTY_PROMPTS_SHARED}" ;;
    *)      echo "[WARN] unknown variant ${v}"; ;;
  esac
done

{
  echo "{\"pipeline\":\"g2_multi\",\"subject\":${SUBJECT},\"fold\":\"${STAG}\",\"finished\":\"$(date -Iseconds)\"}"
} > "${FDIR}/../job_done.json"
echo "===== g2_multi ${STAG} complete @ $(date -Iseconds) ====="
