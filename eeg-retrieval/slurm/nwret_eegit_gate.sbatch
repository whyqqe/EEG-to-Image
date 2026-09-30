#!/usr/bin/env bash
#SBATCH --job-name=nwret-eegitgate
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/nwret-eegitgate-%A_%a.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/nwret-eegitgate-%A_%a.err
#
# The EEGiT-consistency gate: three retrieval-only arms on sub-08.
#
# Why an array of three
# ---------------------
# The semantic side claims to BE EEGiT and reads 46.50 Top-1 against their 70.4.
# Reading the released code turned up four concrete discrepancies (patch-image
# layout, `pos_embed` antialias, loss/temperature, optimizer), and the question is
# how much of the 24-point gap each one accounts for. One arm cannot answer that;
# it can only produce a different number. Three arms, each changing a documented
# set of the four items, can.
#
# This is retrieval only -- no VAE targets, no diffusion decode, no seven metrics.
# Retrieval is what EEGiT reports, so it is the only readout where a like-for-like
# reference number exists, and it costs ~30 min per arm instead of ~4 h.
#
# Cost: 3 arms x ~30 min = ~1.5 h of one GPU, well inside the 3 h wall clock.
# Memory: memory is 96G, not the 128G the dual-tower job asks for, because the
# structure tower (DINOv2-L, ~300M parameters, batch 256 activations) is absent
# from these runs -- the semantic tower alone fits comfortably.
set -uo pipefail

ROOT="/project/peilab/why/eeg-retrieval"
cd "${ROOT}"

export PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export SUBJ="${SUBJ:-8}"

# Weights live on /project, NOT /home: /home is at 100% capacity and a download
# there fails with ENOSPC mid-run.
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"

mkdir -p outputs/slurm

# arm 0 = layout : the 46.50 config + `--patch-style eegit_official`. Isolates the
#                 interface repair (layout + region/electrode order + 2D
#                 interpolation + the `pos_embed` antialias fix).
# arm 1 = full   : full EEGiT consistency -- official layout, official head, single
#                 final layer, official loss and optimizer. The reproduction try.
# arm 2 = fuse   : `full`'s strategy, our 3-layer uniform fusion and `nw` head.
#                 Separates the strategy/interface from the architecture, because
#                 the 46.50 run got part of its win from fusing layers 8/10/12.
ARMS=(layout full fuse)
ARM="${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}"
export ARM

echo "########## job ${SLURM_JOB_ID:-local}.${SLURM_ARRAY_TASK_ID:-0} arm=${ARM} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds) ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

bash scripts/run_eegit_gate.sh
RC=$?

echo "########## job ${SLURM_JOB_ID:-local}.${SLURM_ARRAY_TASK_ID:-0} arm=${ARM} rc=${RC} @ $(date -Iseconds) ##########"
exit "${RC}"
