#!/usr/bin/env bash
# ============================================================================
# NeuroWeave v3 — full sub-08 pipeline (train → generate → eval).
#
# Stages (see docs/NEUROWEAVE_V3_ARCHITECTURE.md):
#   V0  score spatial-init RGB (ceiling / retention baseline)
#   S1  task-factorized encoder (M2)
#   S2  per-modality prior refinement (M3)
#   S3  cross-modal fusion (M4)
#   V1  fidelity-preserving dual-pathway generation grid (M5)
#   EV  official seven + spatial cycle (M8) + summary / kill criteria
#   M7  optional retrieval probe on z_sem (score-level fusion asset)
#
# Protocol lock: prompts_deploy.json (GENERIC, no class names).
# ============================================================================
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
OUT="${NW3_OUT:-${NB_ROOT}/outputs/nw3/sub-08}"
SD="08"
SUB="sub-08"

export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
export HOME="${XDG_CACHE_HOME}"

Z_ROOT="${Z_ROOT:-${NB_ROOT}/outputs/ocf/intra_z}"
PROMPTS="${NB_ROOT}/outputs/g2f/prompts/prompts_deploy.json"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"

S1_EPOCHS="${S1_EPOCHS:-30}"
S2_EPOCHS="${S2_EPOCHS:-40}"
S3_EPOCHS="${S3_EPOCHS:-30}"
# If 1: skip S1–S3 training and reuse existing UCK/mb artifacts (V0+V1 only).
BOOTSTRAP="${NW3_BOOTSTRAP:-0}"
# If 1: skip generation (eval/summary only on existing gens).
EVAL_ONLY="${NW3_EVAL_ONLY:-0}"

mkdir -p "${OUT}"/{logs,s1,s2,s3,gen,eval,cycle,v0,m7} "${NB_ROOT}/outputs/slurm"

log() { echo "[$(date -Iseconds)] $*"; }
require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1"; exit 1; }; }

for f in scripts/nda/nw3_arms.py scripts/nda/nw3_s1_train.py scripts/nda/nw3_s2_prior.py \
         scripts/nda/nw3_s3_fuse.py scripts/nda/nw3_spatial_cycle.py scripts/nda/nw3_summary.py \
         scripts/nda/generate_struct_inject_decode.py scripts/nda/eval_official_seven_dir.py; do
  require "$f"
done
require "${PROMPTS}"
require "${Z_ROOT}/${SUB}/shared_r_train.npy"
require "${Z_ROOT}/${SUB}/shared_r_test.npy"

# ---- resolve V1 arms from nw3_arms.py --------------------------------------
mapfile -t ARM_NAMES < <("${PYTHON}" -c "import sys; sys.path.insert(0,'scripts/nda'); from nw3_arms import V1_ARMS; print('\n'.join(V1_ARMS))")
log "V1 arms: ${ARM_NAMES[*]}"
log "OUT=${OUT} BOOTSTRAP=${BOOTSTRAP} DEVICE=${DEVICE}"

# ============================================================================
# V0 — score existing spatial init (UCK lowlevel) as ceiling reference
# ============================================================================
V0_INIT="${NB_ROOT}/outputs/uck/sub-08/full/spatial/pred_lowlevel_rgb_512"
if [[ -f "${V0_INIT}/199.png" && ! -f "${OUT}/v0/init_seven.json" ]]; then
  log "===== V0: seven-metric on spatial init ====="
  "${PYTHON}" scripts/nda/eval_official_seven_dir.py \
    --gen-dir "${V0_INIT}" \
    --output-json "${OUT}/v0/init_seven.json" \
    --tag init_lowlevel \
    --images-root "${IMAGES_ROOT}" \
    --device "${DEVICE}" \
    2>&1 | tee "${OUT}/logs/v0_init.log"
else
  log "V0 skip (exists or missing init)"
fi

# ============================================================================
# S1 — task-factorized encoder
# ============================================================================
if [[ "${BOOTSTRAP}" == "1" ]]; then
  log "===== BOOTSTRAP: reuse UCK spatial + IP ====="
  mkdir -p "${OUT}/s1/conds" "${OUT}/s1/spatial"
  # semantic: UCK IP mem; spatial: UCK preds
  cp -n "${NB_ROOT}/outputs/uck/sub-08/full/conds/ip_mem_test.npy" "${OUT}/s1/conds/z_sem_test.npy" || true
  cp -n "${NB_ROOT}/outputs/uck/sub-08/full/spatial/pred_vae_test_scaled.npy" "${OUT}/s1/spatial/" || true
  cp -n "${NB_ROOT}/outputs/uck/sub-08/full/spatial/pred_depth_test_64.npy" "${OUT}/s1/spatial/" || true
  if [[ ! -d "${OUT}/s1/spatial/pred_lowlevel_rgb_512" ]]; then
    ln -sfn "${NB_ROOT}/outputs/uck/sub-08/full/spatial/pred_lowlevel_rgb_512" "${OUT}/s1/spatial/pred_lowlevel_rgb_512"
  fi
  if [[ ! -d "${OUT}/s1/spatial/pred_depth_rgb_512" ]]; then
    ln -sfn "${NB_ROOT}/outputs/uck/sub-08/full/spatial/pred_depth_rgb_512" "${OUT}/s1/spatial/pred_depth_rgb_512"
  fi
  # fake checkpoint marker for downstream guards
  echo '{"bootstrap": true}' > "${OUT}/s1/s1_report.json"
else
  if [[ ! -f "${OUT}/s1/conds/z_sem_test.npy" || ! -f "${OUT}/s1/spatial/pred_vae_test_scaled.npy" ]]; then
    log "===== S1: task-factorized encoder (${S1_EPOCHS} ep) ====="
    "${PYTHON}" scripts/nda/nw3_s1_train.py \
      --out "${OUT}/s1" \
      --test-subject 8 \
      --z-root "${Z_ROOT}" \
      --epochs "${S1_EPOCHS}" \
      --device "${DEVICE}" \
      --decode-rgb 1 \
      2>&1 | tee "${OUT}/logs/s1.log"
  else
    log "S1 exports present — skip"
  fi
fi
require "${OUT}/s1/spatial/pred_lowlevel_rgb_512/199.png"
require "${OUT}/s1/spatial/pred_depth_rgb_512/199.png"

# ============================================================================
# S2 / S3 — prior + fusion (skipped in bootstrap: use UCK IP as primary)
# ============================================================================
IP_PRIMARY=""
if [[ "${BOOTSTRAP}" == "1" ]]; then
  IP_PRIMARY="${OUT}/s1/conds/z_sem_test.npy"
  mkdir -p "${OUT}/s3/conds"
  cp -n "${IP_PRIMARY}" "${OUT}/s3/conds/ip_primary_test.npy" || true
else
  if [[ ! -f "${OUT}/s2/conds/z_img_test.npy" ]]; then
    log "===== S2: prior refinement (${S2_EPOCHS} ep × 4 mods) ====="
    require "${OUT}/s1/best.pth"
    "${PYTHON}" scripts/nda/nw3_s2_prior.py \
      --out "${OUT}/s2" \
      --s1-dir "${OUT}/s1" \
      --z-root "${Z_ROOT}" \
      --test-subject 8 \
      --epochs "${S2_EPOCHS}" \
      --device "${DEVICE}" \
      2>&1 | tee "${OUT}/logs/s2.log"
  else
    log "S2 present — skip"
  fi
  if [[ ! -f "${OUT}/s3/conds/ip_primary_test.npy" ]]; then
    log "===== S3: cross-modal fusion (${S3_EPOCHS} ep) ====="
    "${PYTHON}" scripts/nda/nw3_s3_fuse.py \
      --out "${OUT}/s3" \
      --s2-dir "${OUT}/s2" \
      --s1-dir "${OUT}/s1" \
      --z-root "${Z_ROOT}" \
      --test-subject 8 \
      --epochs "${S3_EPOCHS}" \
      --device "${DEVICE}" \
      2>&1 | tee "${OUT}/logs/s3.log"
  else
    log "S3 present — skip"
  fi
  IP_PRIMARY="${OUT}/s3/conds/ip_primary_test.npy"
fi
require "${IP_PRIMARY}"

INIT_DIR="${OUT}/s1/spatial/pred_lowlevel_rgb_512"
DEPTH_DIR="${OUT}/s1/spatial/pred_depth_rgb_512"
PRED_VAE="${OUT}/s1/spatial/pred_vae_test_scaled.npy"
PRED_DEPTH="${OUT}/s1/spatial/pred_depth_test_64.npy"

# ============================================================================
# V1 — fidelity-preserving generation grid (M5)
# ============================================================================
if [[ "${EVAL_ONLY}" != "1" ]]; then
  for arm in "${ARM_NAMES[@]}"; do
    GEN="${OUT}/gen/${arm}"
    if [[ -f "${GEN}/generated/199.png" ]]; then
      log "gen ${arm} exists — skip"
      continue
    fi
    # pull knobs
    read -r STRENGTH CN_SCALE CN_END IP_SCALE < <("${PYTHON}" -c "
import sys; sys.path.insert(0,'scripts/nda')
from nw3_arms import V1_ARMS
a=V1_ARMS['${arm}']
print(a['strength'], a['cn_scale'], a['cn_end'], a['ip_scale'])
")
    log "===== V1 gen ${arm}: strength=${STRENGTH} cn=${CN_SCALE}@${CN_END} ip=${IP_SCALE} ====="
    mkdir -p "${GEN}"
    # cn_scale==0 → still pass a tiny scale but end=0 effectively disables via guidance end
    CN_ARG=(--cn-scale "${CN_SCALE}" --control-guidance-start 0.0 --control-guidance-end "${CN_END}")
    if awk "BEGIN{exit !(${CN_SCALE}<=0)}"; then
      # CN ablation: conditioning_scale=0 makes ControlNet a true no-op, but
      # diffusers requires start < end, so use a minimal positive band.
      CN_ARG=(--cn-scale 0.0 --control-guidance-start 0.0 --control-guidance-end 0.01)
    fi
    "${PYTHON}" scripts/nda/generate_struct_inject_decode.py \
      --mode img2img \
      --embed-npy "${IP_PRIMARY}" \
      --output-dir "${GEN}" \
      --tag "${arm}" \
      --prompts-json "${PROMPTS}" \
      --control-type depth \
      --cond-dir "${DEPTH_DIR}" \
      --init-dir "${INIT_DIR}" \
      --strength "${STRENGTH}" \
      --ip-scale "${IP_SCALE}" \
      --gen-steps 28 \
      --gen-guidance 5.0 \
      --seed 42 \
      --skip-metrics \
      "${CN_ARG[@]}" \
      2>&1 | tee "${OUT}/logs/gen_${arm}.log"
    require "${GEN}/generated/199.png"
  done
fi

# ============================================================================
# Eval: official seven + spatial cycle
# ============================================================================
for arm in "${ARM_NAMES[@]}"; do
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

# ============================================================================
# M7 — retrieval probe on z_sem (optional, non-blocking)
# ============================================================================
if [[ -f scripts/nda/cfmsf_route_probe.py && -f "${OUT}/s1/conds/z_sem_test.npy" && "${BOOTSTRAP}" != "1" ]]; then
  if [[ ! -f "${OUT}/m7/probe_report.json" ]]; then
    log "===== M7: route probe on z_sem (best-effort) ====="
    # export a fake z-root layout for probe if needed — skip if wiring heavy
    log "M7 deferred (probe wiring is subject-specific); z_sem saved for offline probe"
  fi
fi

# ============================================================================
# Summary
# ============================================================================
log "===== summary ====="
INIT_JSON=""
[[ -f "${OUT}/v0/init_seven.json" ]] && INIT_JSON="--init-seven ${OUT}/v0/init_seven.json"
"${PYTHON}" scripts/nda/nw3_summary.py --out "${OUT}" ${INIT_JSON} \
  2>&1 | tee "${OUT}/logs/summary.log"

log "DONE OUT=${OUT}"
ls -la "${OUT}/nw3_summary.json" "${OUT}/eval/" 2>/dev/null | head
