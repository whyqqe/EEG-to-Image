#!/usr/bin/env bash
# =============================================================================
# SAMGA-R: turn SAMGA (inter-subject retrieval SOTA) into a reconstruction model.
#
# THE PIPELINE, AND WHY EACH STAGE IS WHERE IT IS
# -----------------------------------------------
#   clip     Extract OpenCLIP ViT-H-14 1024-d `image_embeds` for the image corpus.
#            This is the conditioning space IP-Adapter's SDXL ViT-H adapter accepts
#            (see extract_clip_h14.py for the source-level proof). Gated against
#            image_metadata.npy so the row order cannot silently permute.
#
#   feats    Run the LOSO-trained SAMGA encoder over every trial, exporting its 1024-d
#            output plus SAMGA's own 512-d retrieval embedding. Source subjects AND the
#            held-out subject, but the held-out subject's file is used for nothing until
#            `cond`.
#
#   head     Train the generation head (1024 -> 1024) on SOURCE SUBJECTS ONLY. This is
#            the LOSO boundary: nothing about sub-08 informs this fit.
#
#   cond     Apply the frozen head to the held-out subject's features, emitting the
#            (200,1024) array the reconstruction stack consumes.
#
#   gen      SDXL-Turbo (4 steps) + ip-adapter_sdxl_vit-h, reusing the sibling
#            eeg-brainit pipeline verbatim so the outputs are comparable to the
#            `bit_clip` / `prior_atm` rows already in its paper tables.
#
#   metrics  PixCorr, SSIM, AlexNet(2/5), Inception, EfficientNet-B1, CLIP cosine,
#            CLIP 2-way identification (full 200-way: 199 distractors), FID, bootstrap CI.
#
# THE ONE DEVIATION THAT MATTERS, AND WHY IT IS THE DEFAULT
# --------------------------------------------------------
# SAMGA's own protocol early-stops on the *test* set, so its headline checkpoint
# `checkpoint_test_best.pth` is chosen with test-set information. Reusing that checkpoint
# would import the leak straight into every reconstruction number. So the default here is
# `checkpoint_last.pth` (the end of the 50-epoch schedule, no test-set selection), and
# CKPT_TAG=checkpoint_test_best.pth is available for a deliberately leaky upper bound.
# Say which one a number came from; they are not interchangeable.
#
# WHAT IS COMPARABLE TO WHAT
# --------------------------
# eeg-brainit's existing tables are PER-SUBJECT (each subject has its own ATM encoder and
# its own diffusion prior). This pipeline is INTER-SUBJECT: one encoder and one head,
# trained on nine subjects, deployed on the tenth with no target data. So the honest
# framing of any comparison is "inter-subject reconstruction, measured on the same
# stimulus set with the same decoder and the same metrics", not "we beat per-subject ATM".
# Its `prior_atm` row is still worth generating side by side, because the decoder is
# identical and the only moving part is where the condition came from.
# =============================================================================
set -euo pipefail

RECON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BRAINIT="/project/peilab/why/eeg-brainit"

STAGE="${STAGE:-all}"
TARGET="${TARGET:-8}"
CKPT="${CKPT:-}"
CKPT_TAG="${CKPT_TAG:-checkpoint_last.pth}"
SOURCES="${SOURCES:-}"
EPOCHS="${EPOCHS:-60}"
SEED="${SEED:-0}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"

OUT="${RECON_ROOT}/outputs/recon"
FEAT_DIR="${OUT}/feats"
COND_DIR="${OUT}/conditions"
CLIP_DIR="${RECON_ROOT}/data/image_feature/clip_h14_ip_adapter"
GEN_DIR="${BRAINIT}/outputs/erdc/samgar_sub$(printf '%02d' "${TARGET}")"
MET_DIR="${BRAINIT}/outputs/erdc/recon_metrics"

# Smoke runs are isolated under their own roots. Without this, a `--limit` feature export
# or a 3-epoch head would land on the same paths the real run checks for, and the real
# run's idempotency guards would then skip every stage and report success while all the
# numbers came from the smoke. Same reasoning as the InternViT extractor's smoke dir.
if [[ "${SMOKE}" == "1" ]]; then
  OUT="${OUT}/smoke"
  FEAT_DIR="${OUT}/feats"
  COND_DIR="${OUT}/conditions"
  GEN_DIR="${GEN_DIR}_smoke"
  MET_DIR="${MET_DIR}/smoke"
fi

TGT="sub-$(printf '%02d' "${TARGET}")"

log()  { printf '\n=== %s ===\n' "$*"; }
die()  { printf '[FATAL] %s\n' "$*" >&2; exit 1; }
run()  { if [[ "${DRY_RUN}" == "1" ]]; then printf '[dry-run] %s\n' "$*"; else "$@"; fi; }

# The LOSO source set is derived from the target rather than hard-coded. It was hard-coded
# to sub-08's complement at first, which would have silently trained every other fold's
# generation head on data INCLUDING that fold's held-out subject -- the exact leak the LOSO
# protocol exists to prevent, and one that would have looked like unusually good results.
if [[ -z "${SOURCES}" ]]; then
  for s in $(seq 1 10); do [[ "${s}" == "${TARGET}" ]] || SOURCES="${SOURCES} ${s}"; done
  SOURCES="${SOURCES# }"
fi
[[ " ${SOURCES} " == *" ${TARGET} "* ]] \
  && die "SOURCES contains the held-out subject ${TARGET}; that is test-set leakage"
echo "[INFO] fold: hold out ${TGT}; sources = ${SOURCES}"

# The reconstruction stack (diffusers, open_clip, torchvision metrics) lives in the
# sibling project's venv; SAMGA's own modules only need torch, which is there too. Using
# one interpreter for every stage avoids the classic failure where the export runs under
# a different torch build than the head training.
PY="${PY:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
[[ -x "${PY}" ]] || PY="$(command -v python3 || true)"
[[ -n "${PY}" && -x "${PY}" ]] || die "no usable python interpreter (tried eeg-brainit venv and python3)"
echo "[INFO] interpreter: ${PY}"

mkdir -p "${OUT}" "${FEAT_DIR}" "${COND_DIR}" "${MET_DIR}"

# ----------------------------------------------------------------- discovery / checks
resolve_ckpt() {
  if [[ -n "${CKPT}" ]]; then
    [[ -f "${CKPT}" ]] || die "--CKPT ${CKPT} does not exist"
    printf '%s' "${CKPT}"; return
  fi
  local found
  found=$(ls -1dt "${RECON_ROOT}/outputs/samga_official/inter/seed"*/*"${TGT}"/"${CKPT_TAG}" 2>/dev/null | head -1 || true)
  [[ -n "${found}" ]] || die "no ${CKPT_TAG} for ${TGT} under outputs/samga_official/. Pass CKPT=... or wait for the SAMGA LOSO job to finish."
  printf '%s' "${found}"
}

need_clip() {
  [[ -f "${CLIP_DIR}/clip_h14_train.npy" && -f "${CLIP_DIR}/clip_h14_test.npy" ]]
}

# --------------------------------------------------------------------------- preflight
# Cheap, and it exists because its absence cost a job. A single stray nested triple quote
# in a module docstring closed the docstring early and turned the whole file into a
# SyntaxError, which nothing caught until a GPU allocation had already been spent: the
# shell was syntax-checked but the Python never was. Compiling every script here moves
# that class of error from "costs an allocation" to "costs a second on the login node".
stage_preflight() {
  log "preflight: syntax + import check on every pipeline script"
  local files=("${RECON_ROOT}"/scripts/recon/*.py
               "${BRAINIT}/scripts/erdc_official_atm_pipeline.py")
  local rc=0
  for f in "${files[@]}"; do
    [[ -f "${f}" ]] || { echo "[WARN] missing ${f}"; rc=1; continue; }
    if out=$("${PY}" -m py_compile "${f}" 2>&1); then
      printf '  OK   %s\n' "$(basename "${f}")"
    else
      printf '  FAIL %s\n%s\n' "$(basename "${f}")" "${out}"
      rc=1
    fi
  done
  [[ "${rc}" == "0" ]] || die "preflight failed: fix the reported syntax errors above"
  # Import the two modules the stages depend on, so a bad import (missing dependency,
  # wrong SAMGA path) also fails here rather than inside a stage.
  if [[ "${DRY_RUN}" != "1" ]]; then
    "${PY}" - <<'PY' || die "preflight import check failed"
import sys, pathlib
recon = pathlib.Path("/project/peilab/why/eeg-retrieval/scripts/recon")
sys.path.insert(0, str(recon))
import samga_recon                    # pulls in third_party/SAMGA modules
sys.path.insert(0, "/project/peilab/why/eeg-brainit/scripts")
import numpy, torch, torchvision      # generation + metrics stack
print(f"  OK   imports (torch {torch.__version__}, cuda {torch.cuda.is_available()})")
PY
  fi
}

# --------------------------------------------------------------------------- stages
stage_clip() {
  log "clip: OpenCLIP ViT-H-14 1024-d conditioning targets"
  if need_clip; then echo "[SKIP] ${CLIP_DIR} already complete"; return 0; fi
  mkdir -p "${CLIP_DIR}"
  # Deliberately NOT limited in smoke mode, even though that would make the smoke faster.
  # A partial CLIP array is not a smaller version of the real one, it is an unusable one:
  # the head's supervision indexes all 1654 concepts, and generation retrieves neighbours
  # from all 16540 training images, so a 40-concept array makes both stages fail with
  # index errors rather than exercising them. Full extraction is also cheap next to the
  # rest of the pipeline (~1-2 min of ViT-H-14 forwards on an H800). The `_smoke` filename
  # suffix inside extract_clip_h14.py stays as a guard for anyone who does limit it by
  # hand: a partial array then cannot masquerade as this stage's output.
  run "${PY}" "${RECON_ROOT}/scripts/recon/extract_clip_h14.py" \
      --out-dir "${CLIP_DIR}" --splits train,test \
      --verify-against "${BRAINIT}/outputs/atm_bridge/clip_img_train_1024.npy"
  [[ "${DRY_RUN}" == "1" ]] || need_clip || die "clip extraction produced incomplete output"
}

stage_feats() {
  local ckpt="$1"
  log "feats: SAMGA features for ${TGT} + sources [${SOURCES}]"
  local extra=()
  [[ "${SMOKE}" == "1" ]] && extra+=(--limit 512)
  for s in ${SOURCES} ${TARGET}; do
    local tag="sub-$(printf '%02d' "${s}")"
    local tr="${FEAT_DIR}/${tag}_train.npz"
    local te="${FEAT_DIR}/${tag}_test.npz"
    if [[ -f "${tr}" && ( "${s}" != "${TARGET}" || -f "${te}" ) ]]; then
      echo "[SKIP] ${tag}"; continue
    fi
    # Source subjects need their train split (that is the head's training data). The
    # target subject needs only its test split -- its train split is not part of LOSO.
    if [[ "${s}" != "${TARGET}" ]]; then
      run "${PY}" "${RECON_ROOT}/scripts/recon/export_eeg_feats.py" \
        --ckpt "${ckpt}" --subject "${tag}" --split train --out "${tr}" "${extra[@]}"
    fi
    run "${PY}" "${RECON_ROOT}/scripts/recon/export_eeg_feats.py" \
      --ckpt "${ckpt}" --subject "${tag}" --split test --out "${te}" "${extra[@]}"
  done
}

stage_head() {
  log "head: train generation head on sources only"
  local dest="${OUT}/gen_head_${TGT}_seed${SEED}.pt"
  if [[ -f "${dest}" ]]; then echo "[SKIP] ${dest}"; return 0; fi
  local srcs=()
  for s in ${SOURCES}; do srcs+=("${FEAT_DIR}/sub-$(printf '%02d' "${s}")_train.npz"); done
  if [[ "${DRY_RUN}" != "1" ]]; then
    for f in "${srcs[@]}"; do [[ -f "${f}" ]] || die "missing source features ${f}"; done
  fi
  local extra=()
  [[ "${SMOKE}" == "1" ]] && extra+=(--epochs 3 --limit-rows 4096)
  run "${PY}" "${RECON_ROOT}/scripts/recon/train_gen_head.py" \
      --src-feats "${srcs[@]}" \
      --clip-train "${CLIP_DIR}/clip_h14_train.npy" \
      --out "${dest}" --epochs "${EPOCHS}" --seed "${SEED}" "${extra[@]}"
}

stage_cond() {
  local head="$1"
  log "cond: (200,1024) conditioning arrays"
  if [[ "${DRY_RUN}" != "1" && ! -f "${head}" ]]; then die "missing head ${head}"; fi
  local extra=()
  [[ "${SMOKE}" == "1" ]] && extra+=(--allow-cpu)
  for mode in head identity; do
    local out="${COND_DIR}/samga_${mode}_${TGT}_seed${SEED}.npy"
    [[ -f "${out}" ]] && { echo "[SKIP] ${out}"; continue; }
    local args=(--test-feats "${FEAT_DIR}/${TGT}_test.npz" --mode "${mode}"
                --out "${out}" --subject "${TGT}"
                --clip-test "${CLIP_DIR}/clip_h14_test.npy")
    [[ "${mode}" == "head" ]] && args+=(--gen-head "${head}")
    run "${PY}" "${RECON_ROOT}/scripts/recon/export_conditions.py" "${args[@]}" "${extra[@]}"
  done
}

# The reconstruction stack lives in the sibling project and is driven from its root, so
# that its relative defaults (outputs/atm_bridge, checkpoints/, the image root) resolve
# exactly as they do for the numbers already in its tables.
stage_gen() {
  log "gen: SDXL-Turbo 4 steps + ip-adapter_sdxl_vit-h (reusing eeg-brainit)"
  cd "${BRAINIT}"
  local extra=(); [[ "${SMOKE}" == "1" ]] && extra+=(--max-images 8)
  for mode in head identity; do
    local cond="${COND_DIR}/samga_${mode}_${TGT}_seed${SEED}.npy"
    [[ -f "${cond}" ]] || { echo "[WARN] skip ${mode}: no conditions"; continue; }
    local out="${GEN_DIR}_${mode}"
    [[ -f "${out}/metrics.json" ]] && { echo "[SKIP] ${out}"; continue; }
    run "${PY}" scripts/erdc_official_atm_pipeline.py \
      --subject "${TGT}" \
      --embed-source npy \
      --bit-npy "${cond}" \
      --low-level-mode neighbor_image \
      --top-m 2 --strengths 0.35,0.50,0.65,0.85 \
      --use-turbo --gen-steps 4 --gen-guidance 0.0 \
      --output-dir "${out}" "${extra[@]}"
  done
}

stage_metrics() {
  log "metrics: full suite + FID + 2WC + bootstrap CI"
  cd "${BRAINIT}"
  # Four separate scripts, matching erdc_paper_finalize.sh exactly, so these numbers land
  # in the same namespace and format as the rows already in the paper tables:
  #   erdc_full_metrics  -> pixcorr/ssim/alexnet2/alexnet5/inception/clip_cosine/effnet_b1
  #   erdc_fid_metrics   -> FID
  #   erdc_twoway_metrics-> CLIP 2-way identification, full 200-way (199 distractors)
  #   erdc_bootstrap_ci  -> 95% CI over 2000 resamples
  local extra=(); [[ "${SMOKE}" == "1" ]] && extra+=(--max-images 8)
  for mode in head identity; do
    local gen="${GEN_DIR}_${mode}/selected_brain"
    [[ -d "${gen}" ]] || { echo "[WARN] skip ${mode}: no selected_brain"; continue; }
    local tag="samgar_${TGT}_${mode}"
    run "${PY}" scripts/erdc_full_metrics.py --gen-dir "${gen}" \
        --output-json "${MET_DIR}/${tag}.json" --tag "${tag}" "${extra[@]}"
    run "${PY}" scripts/erdc_fid_metrics.py --gen-dir "${gen}" \
        --output-json "${MET_DIR}/${tag}_fid.json" --tag "${tag}" "${extra[@]}"
    run "${PY}" scripts/erdc_twoway_metrics.py --gen-dir "${gen}" \
        --output-json "${MET_DIR}/${tag}_2wc.json" --tag "${tag}" "${extra[@]}"
    run "${PY}" scripts/erdc_bootstrap_ci.py --gen-dir "${gen}" \
        --output-json "${MET_DIR}/${tag}_bootstrap.json" --tag "${tag}" "${extra[@]}"
  done

  echo
  echo "--- summary ($(printf '%s' "${TGT}")) ---"
  if [[ "${DRY_RUN}" != "1" ]]; then
    # The merge logic here is subtle enough (four files, overlapping key names, one of
    # them written in a type-incompatible form) that it lives in its own module so it can
    # be run and diffed on its own. See the docstring there for the failure it prevents.
    "${PY}" "${RECON_ROOT}/scripts/recon/summarize_metrics.py" "${MET_DIR}" "${TGT}"
  fi
}

# ----------------------------------------------------------------------------- main
CKPT_PATH=""
case "${STAGE}" in
  all|feats|head|cond) CKPT_PATH="$(resolve_ckpt)" ;;
esac

# Every stage that touches Python runs preflight first, so a syntax or import error is
# always reported by preflight rather than surfacing as a stage failure.
case "${STAGE}" in
  all|preflight|clip|feats|head|cond|gen|metrics) stage_preflight ;;
esac

case "${STAGE}" in
  preflight) : ;;
  all)
    stage_clip
    stage_feats "${CKPT_PATH}"
    HEAD="${OUT}/gen_head_${TGT}_seed${SEED}.pt"
    stage_head; stage_cond "${HEAD}"; stage_gen; stage_metrics
    ;;
  clip)    stage_clip ;;
  feats)   stage_feats "${CKPT_PATH}" ;;
  head)    stage_head ;;
  cond)    stage_cond "${OUT}/gen_head_${TGT}_seed${SEED}.pt" ;;
  gen)     stage_gen ;;
  metrics) stage_metrics ;;
  *) die "STAGE must be one of all|preflight|clip|feats|head|cond|gen|metrics, got '${STAGE}'" ;;
esac

echo
echo "[OK] SAMGA-R stage '${STAGE}' done."
echo "     encoder ckpt : ${CKPT_PATH:-<n/a>}   (tag ${CKPT_TAG})"
echo "     conditions   : ${COND_DIR}/samga_{head,identity}_${TGT}_seed${SEED}.npy"
echo "     generations  : ${GEN_DIR}_{head,identity}/"
echo "     metrics      : ${MET_DIR}/samgar_${TGT}_{head,identity}*.json"
