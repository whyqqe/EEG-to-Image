#!/usr/bin/env bash
# ============================================================================
# Layer-wise condition injection -- the decisive NeuroWeave test, on sub-08.
#
# HYPOTHESIS (pre-registered, see scripts/nda/layered_arms.py)
#   Our condition stack injects every modality (semantic CLIP, depth, edge) at
#   every UNet attention level, so they compete for one conditioning channel.
#   Measured consequence: a hard frontier -- more structural conditioning buys
#   SSIM and destroys semantic identifiability.
#     atm_aligned  SSIM 0.2300 / cycle_disc_top1 0.565
#     mb_p1_i1     SSIM 0.3500 / cycle_disc_top1 0.130
#     single       SSIM 0.3696 / cycle_disc_top1 0.080
#   Layered injection claims this is PLACEMENT, not capacity: semantic late
#   (mid + up blocks), structure early (down blocks).
#
# ARMS (specs read from layered_arms.py so they cannot drift)
#   single    all:1.0 x3            REUSED -- exactly outputs/mb_s08/gen/mb_p3_i3_cn
#   layered   late,early,early      the candidate
#   rev       early,late,late       direction control
#   lowall    all:5/11 x3           matched-mass strength control (mass == layered)
#
# WHY lowall MATTERS: layered carries less total conditioning than single
# (mass 15 vs 33).  If lowall matches layered, the effect is strength, not
# placement, and the placement story must be dropped even if the numbers look
# good.  If lowall matches single, placement is the active ingredient.
#
# BARS: SSIM >= 0.3696 AND cycle_disc_top1 >= 0.25 (strong: >= 0.40)
# ============================================================================
set -uo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
OUT="${LAYERED_OUT:-${NB_ROOT}/outputs/layered/sub-08}"
SD="08"

export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"

IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
IP="${NB_ROOT}/outputs/uck/sub-08/full/conds/ip_mem_test.npy"
D="${NB_ROOT}/outputs/mb_s08/heads/conds/depth_pred_test_cal.npy"
E="${NB_ROOT}/outputs/mb_s08/heads/conds/edge_pred_test_cal.npy"
CONDS="${IP},${D},${E}"
PROMPTS="${NB_ROOT}/outputs/g2f/prompts/prompts_deploy.json"
DEPTH_RGB="${NB_ROOT}/outputs/uck/sub-08/full/spatial/pred_depth_rgb_512"
LL_RGB="${NB_ROOT}/outputs/sdedit_ll_full10/sub-08/vae_head/pred_lowlevel_rgb_512"
[[ -f "${LL_RGB}/199.png" ]] || LL_RGB="${NB_ROOT}/outputs/uck/sub-08/full/spatial/pred_lowlevel_rgb_512"
ATM_GEN="${NB_ROOT}/outputs/atm_aligned_decode/sub-08/generation/combo_d40_luma_pc_a060/generated"
SINGLE_GEN="${NB_ROOT}/outputs/mb_s08/gen/mb_p3_i3_cn/generated"

mkdir -p "${OUT}"/{logs,gen,eval,cycle} "${NB_ROOT}/outputs/slurm"

log() { echo "[$(date -Iseconds)] $*"; }
require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1"; exit 1; }; }

# ---- specs, from the single source of truth --------------------------------
spec_of() {
  "${PYTHON}" -c "import sys; sys.path.insert(0, 'scripts/nda'); \
from layered_arms import LAYERED_ARMS; print(LAYERED_ARMS['$1']['spec'])"
}
SPEC_LAYERED="$(spec_of layered)"
SPEC_REV="$(spec_of rev)"
SPEC_LOWALL="$(spec_of lowall)"
log "specs: layered=${SPEC_LAYERED} rev=${SPEC_REV} lowall=${SPEC_LOWALL}"

for f in scripts/nda/generate_layered_decode.py scripts/nda/layered_arms.py \
         scripts/nda/layer_plan_audit.py scripts/nda/layered_summary.py \
         scripts/nda/neuroweave_cycle_score.py scripts/nda/eval_official_seven_dir.py; do
  require "$f"
done
for f in "${IP}" "${D}" "${E}" "${PROMPTS}" "${DEPTH_RGB}/199.png" "${LL_RGB}/199.png"; do
  require "$f"
done
require "${ATM_GEN}/199.png"
require "${SINGLE_GEN}/199.png"

# ---- 0. device audit + pure-logic layer audit (both hard gates) ------------
log "===== device audit ====="
"${PYTHON}" scripts/nda/device_audit.py scripts/nda/generate_layered_decode.py \
  || { echo "[FATAL] device audit failed"; exit 1; }

log "===== layer plan audit (CPU, negative controls must raise) ====="
"${PYTHON}" scripts/nda/layer_plan_audit.py \
  || { echo "[FATAL] layer plan audit failed"; exit 1; }

# ---- 1. resolve each spec against the REAL unet config --------------------
log "===== plan resolution against the real UNet config ====="
for a in layered rev lowall; do
  case "$a" in
    layered) S="${SPEC_LAYERED}";; rev) S="${SPEC_REV}";; lowall) S="${SPEC_LOWALL}";;
  esac
  "${PYTHON}" scripts/nda/generate_layered_decode.py \
      --cond-npys "${CONDS}" --branch-spec "${S}" --prompts-json "${PROMPTS}" \
      --output-dir "${OUT}/gen/${a}" --tag "${a}" --plan-only \
    2>&1 | tee "${OUT}/logs/plan_${a}.log"
done

# ---- 2. layer sanity: prove each spec actually changes the render ---------
# A spec that silently does not land renders plausible images and would be read
# as evidence AGAINST layered injection, so this is a hard gate per arm.
for a in layered rev lowall; do
  case "$a" in
    layered) S="${SPEC_LAYERED}";; rev) S="${SPEC_REV}";; lowall) S="${SPEC_LOWALL}";;
  esac
  SAN="${OUT}/logs/layer_sanity_${a}.json"
  if [[ -f "${SAN}" ]]; then log "[skip] layer sanity ${a}"; continue; fi
  log "===== layer sanity ${a} ====="
  rm -f "${SAN}"
  "${PYTHON}" scripts/nda/generate_layered_decode.py \
      --cond-npys "${CONDS}" --branch-spec "${S}" \
      --sanity-ref-spec "all:1.0,all:1.0,all:1.0" \
      --prompts-json "${PROMPTS}" --output-dir "${OUT}/sanity_${a}" --tag "sanity_${a}" \
      --pipeline base --layer-sanity 3 --sanity-out "${SAN}" \
      --gen-steps 28 --gen-guidance 5.0 --device "${DEVICE}" \
      > "${OUT}/logs/sanity_${a}.log" 2>&1
  rc=$?
  tail -n 20 "${OUT}/logs/sanity_${a}.log"
  if [[ ${rc} -ne 0 || ! -f "${SAN}" ]]; then
    echo "[FATAL] layer sanity ${a} produced no verdict (rc=${rc}, ${SAN} missing)"
    exit 1
  fi
  V="$("${PYTHON}" -c "import json;print(json.load(open('${SAN}')).get('verdict',''))" 2>/dev/null)"
  if [[ "${V}" != "LAYER_SPEC_ACTIVE" ]]; then
    echo "[FATAL] ${a}: the layer spec is not proven to change the render"
    echo "        verdict='${V}' -- a silent no-op here would be reported as"
    echo "        evidence AGAINST layered injection, so this is fatal."
    grep -nE "pixel_diff|verdict|verify|mass|branch" "${OUT}/logs/sanity_${a}.log" | tail -n 15
    exit 1
  fi
  log "layer sanity ${a}: ${V} (spec proven to change the render)"
done

# ---- 3. generate the three new arms ---------------------------------------
gen_arm() { # arm spec
  local a="$1" S="$2"
  if [[ -f "${OUT}/gen/${a}/generated/199.png" ]]; then log "[skip] gen ${a}"; return 0; fi
  log "===== gen ${a} (${S}) ====="
  "${PYTHON}" scripts/nda/generate_layered_decode.py \
      --cond-npys "${CONDS}" --branch-spec "${S}" \
      --prompts-json "${PROMPTS}" --output-dir "${OUT}/gen/${a}" --tag "${a}" \
      --pipeline base --depth-rgb-dir "${DEPTH_RGB}" --lowlevel-rgb-dir "${LL_RGB}" \
      --cn-scale 0.40 --strength 0.82 --gen-steps 28 --gen-guidance 5.0 \
      --seed 42 --device "${DEVICE}" \
      --layer-report "${OUT}/logs/layer_assignment_${a}.json" \
      > "${OUT}/logs/gen_${a}.log" 2>&1
  local rc=$?
  [[ ${rc} -eq 0 && -f "${OUT}/gen/${a}/generated/199.png" ]] \
    || { echo "[FATAL] gen ${a} failed (rc=${rc}); tail:"; tail -n 30 "${OUT}/logs/gen_${a}.log"; return 1; }
  return 0
}

gen_arm layered "${SPEC_LAYERED}" || exit 1
gen_arm rev     "${SPEC_REV}"     || exit 1
gen_arm lowall  "${SPEC_LOWALL}"  || exit 1

# ---- 4. the seven official metrics, same scorer as the frontier -----------
eval_arm() { # arm gendir
  local a="$1" g="$2"
  local ev="${OUT}/eval/s${SD}_${a}.json"
  if [[ -f "${ev}" ]]; then log "[skip] eval ${a}"; return 0; fi
  log "===== eval seven ${a} ====="
  "${PYTHON}" scripts/nda/eval_official_seven_dir.py \
      --gen-dir "${g}" --output-json "${ev}" --tag "${a}" \
      --images-root "${IMAGES_ROOT}" --device "${DEVICE}" \
      > "${OUT}/logs/eval_${a}.log" 2>&1 \
    || { echo "[FATAL] eval ${a} failed; tail:"; tail -n 20 "${OUT}/logs/eval_${a}.log"; return 1; }
  return 0
}

eval_arm layered "${OUT}/gen/layered/generated" || exit 1
eval_arm rev     "${OUT}/gen/rev/generated"     || exit 1
eval_arm lowall  "${OUT}/gen/lowall/generated"  || exit 1

# ---- 5. cycle consistency in three spaces (+ GT ordering hard gate) -------
if [[ -f "${OUT}/cycle/cycle_report.json" ]]; then
  log "[skip] cycle report present"
else
  log "===== cycle: 3 visual spaces, 5 sets ====="
  SETS="atm_aligned=${ATM_GEN},single=${SINGLE_GEN}"
  SETS="${SETS},layered=${OUT}/gen/layered/generated"
  SETS="${SETS},rev=${OUT}/gen/rev/generated"
  SETS="${SETS},lowall=${OUT}/gen/lowall/generated"
  "${PYTHON}" scripts/nda/neuroweave_cycle_score.py \
      --out "${OUT}/cycle" --sets "${SETS}" --device "${DEVICE}" \
      > "${OUT}/logs/cycle.log" 2>&1 \
    || { echo "[FATAL] cycle scoring failed; tail:"; tail -n 30 "${OUT}/logs/cycle.log"; exit 1; }
fi

# ---- 6. pre-registered verdicts (gates inside; exits 2 if unproven) -------
log "===== pre-registered verdicts ====="
"${PYTHON}" scripts/nda/layered_summary.py --root "${OUT}" \
  2>&1 | tee "${OUT}/logs/layered_summary.log"
RC="${PIPESTATUS[0]}"
if [[ "${RC}" -ne 0 ]]; then
  echo "[FATAL] summary exited ${RC} -- gates not satisfied, verdicts withheld"
  exit "${RC}"
fi

log "===== done ====="
du -sh "${OUT}" | sed 's/^/[disk] /'
