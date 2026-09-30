#!/usr/bin/env bash
#SBATCH --job-name=samga-recon
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/recon/slurm-%A_%a.log
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/recon/slurm-%A_%a.err
#
# =============================================================================
# SAMGA-R: inter-subject (LOSO) EEG-to-image reconstruction.
#
# This is the only job in this project that needs a GPU for all of its stages and a
# network-free environment afterwards: the OpenCLIP ViT-H-14 weights, SDXL-Turbo and
# ip-adapter_sdxl_vit-h.bin are all already under /project/peilab/why/cache, and the
# sibling project's scripts set HF_HUB_OFFLINE=1 themselves. Nothing is downloaded here.
#
# MEMORY
# ------
# The binding constraint is not a single tensor but the sum of three things that overlap:
#   * the generation head's pooled training set: 9 sources x 16540 rows x 1024 x 4 B
#     = 1.2 GB for `hidden` plus 1.2 GB for the CLIP targets, doubled once by the
#     in-flight torch copies (~5 GB);
#   * `export_eeg_feats.py` holds one subject's train array (1.0 GB, [1654,10,4,63,250])
#     and the InternViT image features the dataset insists on loading (0.5 GB) -- but only
#     one subject at a time, since subjects are exported sequentially;
#   * SDXL-Turbo plus the IP-Adapter image projection (~7 GB resident during generation).
# 64 GB leaves comfortable headroom. The earlier official-SAMGA job needed only 16 GB
# because it never pooled nine subjects at once; this pipeline does, by construction.
#
# TIME
# ----
# ~20 min CLIP extraction, ~20 min feature export, ~15 min head training, ~30 min
# generation (200 stimuli x 8 candidates, 4-step Turbo), ~15 min metrics. 4 h is roughly
# 2x the expected wall time, which is the right ratio for a job whose stages are
# individually idempotent -- a timeout costs a resubmit, not a rerun from scratch.
# =============================================================================
set -euo pipefail

RECON_ROOT="/project/peilab/why/eeg-retrieval"
cd "${RECON_ROOT}"
mkdir -p "${RECON_ROOT}/outputs/recon"

# A GPU node has no route to the model hubs. Fail loudly here rather than halfway through
# a decoder load with a confusing DNS error.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="/project/peilab/why/cache/eeg-brainit/hf"
export HF_HUB_CACHE="/project/peilab/why/cache/eeg-brainit/hf/hub"
export OPENCLIP_CACHE_DIR="/project/peilab/why/cache/eeg-brainit/open_clip"
export TORCH_HOME="/project/peilab/why/cache/eeg-brainit/torch"
export TMPDIR="/project/peilab/why/cache/eeg-brainit/tmp"
export XFORMERS_DISABLED=1
export PYTHONUNBUFFERED=1
mkdir -p "${TMPDIR}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} gpu=$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# Array support: one element per LOSO fold. `--array=0-9` maps to sub-01..sub-10, so a
# single submission covers the whole LOSO protocol and each fold gets its own encoder
# checkpoint, its own head, and its own generations. TARGET is overridden per element
# here rather than by ten separate `--export`s, which keeps the fold definition in one
# place and impossible to get out of step between submissions.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  export TARGET="$(( SLURM_ARRAY_TASK_ID + 1 ))"
  echo "[INFO] array task ${SLURM_ARRAY_TASK_ID} -> TARGET=${TARGET}"
fi
# Per-fold logging, so concurrent folds do not interleave into one file. Emitted
# unconditionally rather than only for arrays: a single-fold submission has no
# SLURM_ARRAY_TASK_ID, and Slurm renders `%a` in the --output name as the sentinel
# 4294967294 in that case, so without this the pipeline's log would live at an
# unpredictable filename exactly when someone is debugging a one-fold run.
exec > >(tee -a "${RECON_ROOT}/outputs/recon/recon-sub$(printf '%02d' "${TARGET}").log") 2>&1

# CKPT is discovered from the SAMGA LOSO run's output tree unless overridden. CKPT_TAG
# defaults to `checkpoint_last.pth`: SAMGA's published protocol selects its checkpoint on
# the test set, and importing that selection into a reconstruction result would put the
# leak in every number downstream. See the header of run_samga_recon.sh.
bash "${RECON_ROOT}/scripts/recon/run_samga_recon.sh"

echo "[OK] samga-recon done"
