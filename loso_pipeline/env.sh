#!/usr/bin/env bash
# Shared environment for the sub-08 LOSO EEG-to-image pipeline.
#
# This pipeline is deliberately independent of `inter_pipeline` (the earlier
# "train on all 10 subjects / evaluate on all 10" experiment) but it *reuses* that
# project's venv and HF cache, because:
#   * /project is at 98% capacity, so a second torch/diffusers install is wasteful;
#   * the venv already carries torch 2.6.0+cu124, diffusers 0.36.0, transformers
#     4.57.6, open_clip 3.3.0 and timm 1.0.27 -- everything this pipeline needs;
#   * the HF cache there already holds the laion ViT-H-14 CLIP, DINOv2-L, SDXL-VAE,
#     SDXL-base and IP-Adapter weights that Stage 0 requires.
# The environment hardening below (PATH order, CC=gcc, relocated HOME) is copied
# from `inter_pipeline/env.sh` on purpose: each of those fixes was bought with a
# real failure and re-deriving them would be a waste.  See that file for the
# long-form rationale behind each block.
set -euo pipefail

# The real home must be captured before $HOME is relocated: several packages were
# installed with `pip --user` and only exist under ~/.local.
export LOSO_REAL_HOME="${LOSO_REAL_HOME:-${HOME}}"

export PROJECT_ROOT=/project/peilab/why
export LOSO_ROOT="${PROJECT_ROOT}/third_party/loso_pipeline"

# --- reused environment (venv + model caches owned by inter_pipeline) ---------
export INTER_ROOT="${PROJECT_ROOT}/third_party/inter_pipeline"
export LOSO_VENV="${LOSO_VENV:-${INTER_ROOT}/venv}"
# Point straight at the big cache: it already contains every frozen teacher this
# pipeline needs, so nothing has to be re-downloaded except BLIP2.
export LOSO_HF_HUB="${LOSO_HF_HUB:-${PROJECT_ROOT}/cache/eeg-brainit/hf/hub}"

# --- sources -----------------------------------------------------------------
export EEG_SRC="${EEG_SRC:-${PROJECT_ROOT}/NeuroBridge/data/things_eeg/preprocessed_eeg}"
export IMG_SRC="${IMG_SRC:-${PROJECT_ROOT}/data/images_set}"

# --- this pipeline's own trees -----------------------------------------------
export LOSO_DATA="${LOSO_ROOT}/data"
export LOSO_ASSETS="${LOSO_ROOT}/assets"
export LOSO_OUT="${LOSO_ROOT}/outputs"
export LOSO_LOG="${LOSO_ROOT}/logs"

# --- caches (all on /project; /home is at 100%) ------------------------------
export HOME="${LOSO_ROOT}/home"
export XDG_CACHE_HOME="${LOSO_ROOT}/cache/xdg"
export XDG_CONFIG_HOME="${LOSO_ROOT}/cache/xdg-config"
export TMPDIR="${LOSO_ROOT}/cache/tmp"
export PIP_CACHE_DIR="${LOSO_ROOT}/cache/pip"
export HF_HOME="${LOSO_ROOT}/cache/hf"
export HUGGINGFACE_HUB_CACHE="${LOSO_HF_HUB}"
export HF_HUB_CACHE="${LOSO_HF_HUB}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_OFFLINE=0
export TORCH_HOME="${LOSO_ROOT}/cache/torch"
export OPENCLIP_CACHE_DIR="${LOSO_ROOT}/cache/open_clip"
export WANDB_MODE=offline
export WANDB_DIR="${LOSO_ROOT}/cache/wandb"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${TMPDIR}" \
         "${PIP_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${OPENCLIP_CACHE_DIR}" \
         "${WANDB_DIR}" "${LOSO_DATA}" "${LOSO_ASSETS}" "${LOSO_OUT}" "${LOSO_LOG}"

# Pinned model ids, written once so the extraction scripts and the training
# scripts cannot drift apart.  Override via the environment if a different
# checkpoint is wanted.
export LOSO_CLIP_ID="${LOSO_CLIP_ID:-laion/CLIP-ViT-H-14-laion2B-s32B-b79K}"
export LOSO_DINO_ID="${LOSO_DINO_ID:-vit_large_patch14_dinov2.lvd142m}"
export LOSO_VAE_ID="${LOSO_VAE_ID:-stabilityai/sdxl-vae}"
export LOSO_BLIP2_ID="${LOSO_BLIP2_ID:-Salesforce/blip2-opt-2.7b}"
export LOSO_SD_ID="${LOSO_SD_ID:-stabilityai/stable-diffusion-xl-base-1.0}"
export LOSO_IP_ADAPTER_ID="${LOSO_IP_ADAPTER_ID:-h94/IP-Adapter}"

# --- python import resolution ------------------------------------------------
# Order is load-bearing and mirrors inter_pipeline/env.sh: venv first (its newer
# pins win), conda base second (torch lives there), relocated user-site last.
INTER_PY_VER="$(ls "${LOSO_VENV}/lib" 2>/dev/null | grep -E '^python3\.[0-9]+$' | head -1)"
INTER_PY_VER="${INTER_PY_VER:-python3.10}"
INTER_BASE_HOME="$(sed -n 's/^home *= *//p' "${LOSO_VENV}/pyvenv.cfg" 2>/dev/null | head -1)"
INTER_BASE_PREFIX="${INTER_BASE_HOME%/bin}"
export INTER_BASE_SITE="${INTER_BASE_PREFIX}/lib/${INTER_PY_VER}/site-packages"
export INTER_REAL_USER_SITE="${LOSO_REAL_HOME}/.local/lib/${INTER_PY_VER}/site-packages"

if [[ -x "${LOSO_VENV}/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "${LOSO_VENV}/bin/activate"
  export PYTHONPATH="${LOSO_ROOT}/src:${LOSO_VENV}/lib/${INTER_PY_VER}/site-packages:${INTER_BASE_SITE}:${INTER_REAL_USER_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
else
  echo "[FATAL] venv missing at ${LOSO_VENV}" >&2
  echo "        This pipeline reuses inter_pipeline's venv; build it first:" >&2
  echo "          bash ${INTER_ROOT}/scripts/shared_prep.sh" >&2
  exit 3
fi

# --- PATH: keep the venv authoritative for console scripts -------------------
# A console script runs under whatever interpreter its shebang names, so PATH
# order -- not PYTHONPATH -- decides correctness.  The inherited PATH puts
# ~/.local/bin (system-Anaconda-shebanged `accelerate`/`torchrun`) first, which
# boots a foreign interpreter and dies with the misleading
# "Failed to load PyTorch C extensions".
LOSO_PATH_KEEP=""
IFS=':' read -r -a _loso_path_entries <<< "${PATH}"
for _entry in "${_loso_path_entries[@]}"; do
  [[ -z "${_entry}" ]] && continue
  case "${_entry}" in
    "${LOSO_REAL_HOME}/.local/bin")  continue ;;
    /cm/shared/apps/Anaconda3/*/bin)  continue ;;
    "${LOSO_VENV}/bin")               continue ;;
    "${INTER_BASE_PREFIX}/bin")       continue ;;
  esac
  LOSO_PATH_KEEP="${LOSO_PATH_KEEP:+${LOSO_PATH_KEEP}:}${_entry}"
done
unset _loso_path_entries _entry
export PATH="${LOSO_VENV}/bin:${INTER_BASE_PREFIX}/bin${LOSO_PATH_KEEP:+:${LOSO_PATH_KEEP}}"

# --- C compiler --------------------------------------------------------------
# `module load nvhpc` exports CC=nvc, and triton passes the GNU-only flag
# -Wno-psabi unconditionally ("nvc-Error-Unknown switch: -Wno-psabi").  Only the
# generic C/C++ compiler is normalised; NVCC and CUDA_HOME are left untouched.
if command -v gcc >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then
  export CC="$(command -v gcc)"
  export CXX="$(command -v g++)"
fi

loso_assert_env() {
  local -a bad=()
  local tool path
  for tool in python; do
    path="$(command -v "${tool}" 2>/dev/null || true)"
    if [[ -z "${path}" ]]; then
      bad+=("${tool}: not found on PATH")
    elif [[ "${path}" != "${LOSO_VENV}/bin/"* ]]; then
      bad+=("${tool}: resolves to ${path}, expected ${LOSO_VENV}/bin/${tool}")
    fi
  done
  if (( ${#bad[@]} > 0 )); then
    { echo "[FATAL] environment misconfigured:"; printf '  - %s\n' "${bad[@]}"; } >&2
    return 1
  fi
  return 0
}

# --- generated path file (written by scripts/prep_assets.py) -----------------
if [[ -f "${LOSO_ASSETS}/paths.env" ]]; then
  # shellcheck disable=SC1091
  source "${LOSO_ASSETS}/paths.env"
fi

if [[ "${LOSO_SKIP_TOOLCHECK:-0}" != "1" ]]; then
  loso_assert_env
fi
