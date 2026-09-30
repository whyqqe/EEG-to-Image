#!/bin/bash
# =============================================================================
# NW4 — open-set trial-level semantics (A0+A1) + closed-form manifold projection
#       (A2) + multi-branch instead of dense fusion (A3) + layered injection (A4)
# =============================================================================
# Protocol lock, same as nw3 so the numbers stay comparable:
#   prompts      = prompts_deploy.json (GENERIC, no class names)
#   pixcorr/ssim = official gray@425 gaussian
#   branches     = each condition projected onto the REAL clip_img trial manifold
# Resumable: every stage is skipped when its artifacts exist.
set -uo pipefail

cd "${NB_ROOT}"
# The venv is NOT optional: the system python has transformers 4.36 / diffusers 0.30,
# whose `from diffusers import AutoencoderKL` dies with
#   "cannot import name 'EncoderDecoderCache' from 'transformers'"
# which silently killed the S1 VAE decode (no init RGB -> generation could not start).
PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# Same cache layout nw3 used, so cached weights resolve on the compute node.
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
export HOME="${XDG_CACHE_HOME}"

OUT="${NW4_OUT:-${NB_ROOT}/outputs/nw4/sub-08}"
DEVICE="${DEVICE:-cuda:0}"
S1_EPOCHS="${S1_EPOCHS:-30}"
ARM_NAMES="${NW4_ARMS:-w4_img_only w4_fused w4_flat w4_layered w4_mirror w4_allearly w4_alllate w4_sem_only w4_struct_only w4_layered_s20 w4_layered_s40}"
EVAL_ONLY="${EVAL_ONLY:-0}"
SKIP_LAYER_SANITY="${SKIP_LAYER_SANITY:-0}"

PROMPTS="${NB_ROOT}/outputs/g2f/prompts/prompts_deploy.json"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
COND="${OUT}/s2/conds"

log() { echo "[$(date -Iseconds)] $*"; }
require() { [[ -e "$1" ]] || { log "FATAL missing $1"; exit 1; }; }
mkdir -p "${OUT}"/{logs,s1,s2,gen,eval,cycle,v0}

log "===== NW4 sub-08 OUT=${OUT} device=${DEVICE} ====="
require "${PROMPTS}"
# generic prompts must NOT contain class names (this is the whole protocol lock)
PROMPTS_CHECK=$("${PYTHON}" - "${PROMPTS}" <<'PY'
import json, sys
prompts = json.load(open(sys.argv[1]))
if len(prompts) < 200:
    print(f"FATAL prompts {len(prompts)} < 200"); sys.exit(1)
uniq = sorted(set(prompts))
print(f"{len(prompts)} prompts, {len(uniq)} unique")
PY
) || exit 1
log "[protocol] prompts: ${PROMPTS_CHECK}"

# =============================================================================
# V0 — the fidelity ceiling (blurry EEG->VAE decode), reused from an earlier run
# =============================================================================
V0_SRC="${NB_ROOT}/outputs/nw3/sub-08/v0/init_seven.json"
if [[ ! -f "${OUT}/v0/init_seven.json" ]]; then
  if [[ -f "${V0_SRC}" ]]; then
    cp "${V0_SRC}" "${OUT}/v0/init_seven.json"
    log "V0 init seven reused from nw3 (same init images, same protocol)"
  else
    log "WARN no V0 init_seven.json found; summary will run without the ceiling row"
  fi
fi

# =============================================================================
# S1 — open-set, trial-level multi-space alignment (A0 + A1)
# =============================================================================
if [[ "${EVAL_ONLY}" != "1" ]]; then
  if [[ ! -f "${OUT}/s1/report.json" ]]; then
    log "===== S1: open-set trial-level alignment (A0+A1) ====="
    "${PYTHON}" scripts/nda/nw4_s1_train.py \
      --out "${OUT}/s1" \
      --test-subject 8 \
      --epochs "${S1_EPOCHS}" \
      --device "${DEVICE}" \
      --w-anchor "${S1_W_ANCHOR:-400.0}" \
      --select-metric "${S1_SELECT:-img_vs_cl}" \
      2>&1 | tee "${OUT}/logs/s1.log"
    require "${OUT}/s1/conds/z_img_test.npy"
  else
    log "S1 present — skip"
  fi

  # ===========================================================================
  # S2 — closed-form manifold projection (A2), replacing the 20-step sampler
  # ===========================================================================
  if [[ ! -f "${OUT}/s2/report.json" ]]; then
    log "===== S2: closed-form manifold projection (A2) ====="
    "${PYTHON}" scripts/nda/nw4_s2_project.py \
      --s1 "${OUT}/s1" \
      --out "${OUT}/s2" \
      --test-subject 8 \
      --device "${DEVICE}" \
      2>&1 | tee "${OUT}/logs/s2.log"
  else
    log "S2 present — skip"
  fi
fi
require "${COND}/img_test.npy"

# =============================================================================
# CONCENTRATION CALIBRATION — the step whose omission was the whole gap.
# =============================================================================
# `nw4_diag_rsa.py` replaced the old gate metric with the one that actually predicts
# generation, over every condition we have generation scores for:
#
#     corr(RSA, inception)  = +0.967     RSA = corr( sim(cond_i,cond_j), sim(T_i,T_j) )
#     corr(vs_cl, inception) = +0.687    (the metric used as the gate before)
#
# On that axis our condition was mis-CONCENTRATED, not mis-directed:
#
#     condition              RSA     rowcos    top1
#     real CLIP  bank       1.0000   0.6275   --
#     nw4 img raw           0.1362   0.9380   0.035   <- all rows point the same way,
#     nw4 img + gem_calib   0.2085   0.6123   0.215      so trials cannot differ
#     hybrid ip_uck calib   0.2120   0.6119   0.160   <- generated 0.7280
#
# `rowcos` 0.938 vs the bank's own 0.6275 is the tell: the IP-Adapter reads the
# direction of a condition but its response is also governed by how tightly packed
# the bank is, so an over-concentrated bank collapses to one appearance.  hybrid_s08
# ran `gem_calib.py`; this pipeline did not, and that one omitted step accounts for
# 0.6228 vs 0.7280.  NOTE this also retires A2: the closed-form manifold projection
# acted on each row's DIRECTION (top-K neighbour mixing) when the defect was the
# bank's CONCENTRATION, so it could never have fixed this.
RAW_COND="${COND}"
COND="${OUT}/s2/conds_cal"
if [[ "${EVAL_ONLY}" != "1" ]] && [[ ! -f "${COND}/img_test.npy" ]]; then
  log "===== concentration calibration onto the TRAIN bank (label- and test-free) ====="
  mkdir -p "${COND}"
  for f in "${RAW_COND}"/*_test.npy; do
    b=$(basename "$f" _test.npy)
    if ! "${PYTHON}" scripts/nda/gem_calib.py \
          --in "$f" --out "${COND}/${b}_test.npy" \
          --ref "${NB_ROOT}/outputs/gem/cond_cache/clip_img1024_train.npy" \
          --tag "nw4_${b}" >> "${OUT}/logs/calib.log" 2>&1; then
      log "WARN calib ${b} failed; falling back to the raw bank"
      cp "$f" "${COND}/${b}_test.npy"
    fi
  done
  log "calibrated condition bank -> ${COND}"
fi
if [[ -f "${COND}/img_test.npy" ]]; then require "${COND}/img_test.npy"; else COND="${RAW_COND}"; fi

# ---- condition-health audit: the generator is only as good as this bank ----
if [[ ! -f "${OUT}/cond_audit.json" ]]; then
  log "===== condition health audit (nw3's S2/S3 offdiag was 0.5668 / 0.6994) ====="
  BANK_ARGS=()
  for f in "${COND}"/*_test.npy; do
    b=$(basename "$f" _test.npy)
    # an independent CLIP concept bank so top1 is NOT self-referential
    BANK_ARGS+=(--bank "${b}=${f}")
  done
  "${PYTHON}" scripts/nda/nw3_cond_audit.py \
    --gallery "${NB_ROOT}/outputs/gem/cond_cache/clip_img1024_test.npy" \
    "${BANK_ARGS[@]}" \
    --json-out "${OUT}/cond_audit.json" 2>&1 | tee "${OUT}/logs/cond_audit.log" || \
    log "WARN cond audit failed (non-blocking)"
fi

# =============================================================================
# CONDITION GATE — refuse to generate from a condition that cannot carry trials.
# =============================================================================
# Gated on RSA, not vs_cl.  `nw4_diag_rsa.py` scored every condition with a known
# generation score and found
#     corr(RSA,  inception) = +0.967
#     corr(vs_cl, inception) = +0.687
# and the vs_cl fit had in any case been measured on un-calibrated banks while the
# generator consumes calibrated ones.  RSA asks whether the trial-to-trial similarity
# structure survives, which is what the adapter must actually render.  nw3 emitted a
# near-constant bank (RSA 0.048 -> 0.5236); the honest reference sits at 0.212 -> 0.7280.
if [[ "${EVAL_ONLY}" != "1" ]] && [[ "${SKIP_COND_GATE:-0}" != "1" ]]; then
  log "===== condition gate (RSA axis) ====="
  "${PYTHON}" scripts/nda/nw4_gate_cond.py \
    --conds "${COND}" --out "${OUT}/cond_gate.json" \
    > "${OUT}/logs/cond_gate.log" 2>&1
  GATE_RC=$?
  cat "${OUT}/logs/cond_gate.log"
  # NB read the JSON rather than the exit code: the caller pipes through tee, which
  # would mask the script's status and silently pass the gate.
  if [[ ! -f "${OUT}/cond_gate.json" ]]; then
    log "FATAL condition gate could not be evaluated (see logs/cond_gate.log)"
    exit 1
  fi
  if ! "${PYTHON}" -c "
import json, sys
d = json.load(open('${OUT}/cond_gate.json'))
sys.exit(0 if d.get('any_pass') else 1)"; then
    log "FATAL condition gate FAILED (rc=${GATE_RC}): no condition clears RSA >= 0.190."
    log "      A3/A4 (branch wiring, layer placement) cannot be assessed on a bank that"
    log "      carries no trial structure, so generation is skipped rather than burning"
    log "      GPU hours on a result that says nothing about the architecture."
    log "      Usual cause: rowcos far above the real bank's -- run gem_calib.py."
    log "      Set SKIP_COND_GATE=1 to override for a diagnostic run."
    exit 1
  fi
  log "condition gate PASSED (RSA axis)"
fi

# =============================================================================
# V1 — generation grid (A3 multi-branch + A4 layered injection)
# =============================================================================
DEPTH_DIR="${OUT}/s1/spatial/pred_depth_rgb_512"
INIT_DIR="${OUT}/s1/spatial/pred_lowlevel_rgb_512"
PRED_VAE="${OUT}/s1/spatial/pred_vae_test_scaled.npy"
PRED_DEPTH="${OUT}/s1/spatial/pred_depth_test_64.npy"
require "${DEPTH_DIR}/000.png"
require "${INIT_DIR}/000.png"

if [[ "${EVAL_ONLY}" != "1" ]]; then
  for arm in ${ARM_NAMES}; do
    GEN="${OUT}/gen/${arm}"
    if [[ -f "${GEN}/generated/199.png" ]]; then
      log "gen ${arm} exists — skip"
      continue
    fi
    read -r BRANCHES SPEC STRENGTH CN_SCALE CN_END SANITY < <("${PYTHON}" -c "
import sys; sys.path.insert(0,'scripts/nda')
from nw4_arms import ARMS
a=ARMS['${arm}']
print(','.join(a['branches']), a['spec'], a['strength'], a['cn_scale'], a['cn_end'],
      1 if '${SKIP_LAYER_SANITY}' != '1' else 0)
")
    # map branch names to condition files; 'fused' is the A3 control bank
    NPYS=""
    IFS=',' read -ra BLIST <<< "${BRANCHES}"
    for b in "${BLIST[@]}"; do
      f="${COND}/${b}_test.npy"
      [[ -f "$f" ]] || { log "FATAL branch ${b} -> missing ${f}"; exit 1; }
      NPYS="${NPYS:+${NPYS},}${f}"
    done
    log "===== V1 gen ${arm}: branches=[${BRANCHES}] spec=${SPEC} strength=${STRENGTH} cn=${CN_SCALE}@${CN_END} ====="
    mkdir -p "${GEN}"
    SANITY_ARG=()
    if [[ "${SANITY}" == "1" ]]; then
      REV_SPEC=$("${PYTHON}" -c "
import sys; sys.path.insert(0,'scripts/nda')
from nw4_arms import ARMS
b=ARMS['${arm}']['branches']
# a no-op-looking reference: every branch 'all:0.0' (must render differently if the
# spec is really active).  Uses the same branch count so parse_spec accepts it.
print(','.join('all:0.0' for _ in b))
")
      SANITY_ARG=(--layer-sanity 2 --sanity-ref-spec "${REV_SPEC}"
                  --sanity-out "${OUT}/gen/${arm}/layer_sanity.json")
    fi
    "${PYTHON}" scripts/nda/generate_layered_decode.py \
      --cond-npys "${NPYS}" \
      --branch-spec "${SPEC}" \
      --prompts-json "${PROMPTS}" \
      --output-dir "${GEN}" \
      --tag "${arm}" \
      --pipeline base \
      --depth-rgb-dir "${DEPTH_DIR}" \
      --lowlevel-rgb-dir "${INIT_DIR}" \
      --cn-scale "${CN_SCALE}" \
      --strength "${STRENGTH}" \
      --gen-steps 28 \
      --gen-guidance 5.0 \
      --gen-size 512 \
      --seed 42 \
      --device "${DEVICE}" \
      --layer-report "${OUT}/gen/${arm}/layer_report.json" \
      "${SANITY_ARG[@]}" \
      2>&1 | tee "${OUT}/logs/gen_${arm}.log"
    if [[ ! -f "${GEN}/generated/199.png" ]]; then
      log "WARN gen ${arm} produced no 199.png — continuing with the other arms"
    fi
  done
fi

# =============================================================================
# Eval — official seven + spatial cycle
# =============================================================================
for arm in ${ARM_NAMES}; do
  GEN="${OUT}/gen/${arm}/generated"
  [[ -f "${GEN}/199.png" ]] || { log "WARN missing gen ${arm}"; continue; }
  if [[ ! -f "${OUT}/eval/${arm}.json" ]]; then
    log "===== seven ${arm} ====="
    "${PYTHON}" scripts/nda/eval_official_seven_dir.py \
      --gen-dir "${GEN}" \
      --output-json "${OUT}/eval/${arm}.json" \
      --tag "${arm}" \
      --images-root "${IMAGES_ROOT}" \
      --device "${DEVICE}" \
      2>&1 | tee "${OUT}/logs/eval_${arm}.log"
  fi
  if [[ ! -f "${OUT}/cycle/${arm}.json" && -f "${PRED_VAE}" ]]; then
    log "===== spatial cycle ${arm} ====="
    "${PYTHON}" scripts/nda/nw3_spatial_cycle.py \
      --gen-dir "${GEN}" \
      --pred-vae "${PRED_VAE}" \
      --pred-depth "${PRED_DEPTH}" \
      --output-json "${OUT}/cycle/${arm}.json" \
      --tag "${arm}" \
      --device "${DEVICE}" \
      2>&1 | tee "${OUT}/logs/cycle_${arm}.log" || log "WARN cycle failed for ${arm}"
  fi
done

# =============================================================================
# Summary — the corrected 2-D bar
# =============================================================================
log "===== summary ====="
"${PYTHON}" scripts/nda/nw4_summary.py --out "${OUT}" 2>&1 | tee "${OUT}/logs/summary.log"

log "DONE OUT=${OUT}"
