#!/usr/bin/env bash
#SBATCH --job-name=epd-dual
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-dual-%A_%a.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-dual-%A_%a.err
#
# EEG Patch dual tower, sub-08, from training to the seven metrics.
#
# The array axis is the structural trunk. It is the run's only independent
# variable: the semantic tower, the geometry of BOTH towers, the decoder, the
# losses, the schedule, the export, the generator and the metrics are byte-identical
# across the three tasks, and the tasks share read-only inputs.
#
#   [0] dino3   DINOv3 ViT-B/16. Self-supervised with Gram anchoring, whose stated
#               purpose is to stop the dense/patch metrics degrading as the global
#               ones improve -- i.e. it regularises exactly the quantity the
#               structural tower means by "structure".
#   [1] mae     MAE ViT-B/16. Trained by reconstructing masked pixels, so it keeps
#               the most recoverable appearance. The control for "spatial
#               coherence beats pixel fidelity".
#   [2] in21k   EEGiT's own backbone. Deliberately NOT a structure-focused prior,
#               and that is the point: it holds the mechanism fixed and changes only
#               the pretrained weights, which prices "a structural prior exists"
#               against "any prior exists".
#
# Without [2] the architecture change and the prior change would be confounded: a
# win for dino3 would be equally explainable by "the backbone got smaller" (DINOv2-L
# 304M -> ViT-B 86M) or "the geometry changed" as by the prior itself.
#
# Three tasks and not one, because no single arm answers the question. dino3 alone
# says "it worked" with nothing to attribute it to; dino3 against in21k says whether
# the prior matters; dino3 against mae says which property of the prior matters.
#
# Cost, per task (estimated on an 80 GB card, batch 128):
#   [0] VAE target cache   present and content-validated; the tasks contend for it
#                          through a flock + atomic-rename publish, so whoever gets
#                          there first builds it once and the other two reuse it
#   [1] train 100 epochs   ~1.5-2 h   (two towers, 12 blocks each, fp32, no autocast)
#   [2] export             ~10 min
#   [3] generate 4 arms    ~40 min    (4 x 200 images x 28 SDXL steps)
#   [4] seven metrics      ~25 min
# so ~3 h each; 6 h requested so a slow node still finishes rather than being killed
# with the generations on disk and the metrics half-written.
#
# The three tasks write to disjoint output dirs but SHARE the VAE target cache and
# the SDXL/CLIP weights, so their peak file-system traffic overlaps. That is intended:
# it is one read of 517 MB each plus the model loads, against 3 h of GPU.
set -uo pipefail

ROOT="/project/peilab/why/eeg-retrieval"
cd "${ROOT}"
mkdir -p outputs/slurm outputs/logs

export PY="${PY:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export SUBJ="${SUBJ:-8}"

# /home is at 100% and a download there fails with ENOSPC partway through a run, so
# every cache is pinned to /project before torch or huggingface_hub is imported.
# `scripts/epd/config.py` sets the same defaults when the scripts are run directly;
# they are set here too because this is the process the scheduler starts.
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"

# The array axis. A task index outside the list is a submission mistake, not
# something to paper over with a default, so it aborts rather than silently running
# dino3 three times and producing one arm's worth of evidence at three times the cost.
ARCHS=(dino3 mae in21k)
i="${SLURM_ARRAY_TASK_ID:-0}"
if (( i < 0 || i >= ${#ARCHS[@]} )); then
  echo "[FATAL] SLURM_ARRAY_TASK_ID=${i} is outside 0..$((${#ARCHS[@]} - 1))" >&2
  exit 2
fi
export ARCH="${ARCHS[$i]}"
export TAG="epd_dual_${ARCH}"

echo "########## epd-dual ${SLURM_JOB_ID:-local}.${i} arch=${ARCH} sub-${SUBJ} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds) ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

bash scripts/run_epd_dual.sh
RC=$?

echo "########## epd-dual ${SLURM_JOB_ID:-local}.${i} arch=${ARCH} rc=${RC} @ $(date -Iseconds) ##########"
exit "${RC}"
