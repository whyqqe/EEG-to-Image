#!/usr/bin/env bash
#SBATCH --job-name=nwret-slotgate
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/nwret-slotgate-%A_%a.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/nwret-slotgate-%A_%a.err
#
# The second half of the EEGiT-consistency gate: the data protocol.
#
# Why this is a separate submission
# ---------------------------------
# The first three gate arms (layout / full / fuse) all peak at val Top-1 31-32 around
# epoch 20 and then overfit for 80 more epochs, against 34.4 for the run they were
# trying to improve on. That is a clean negative on the interface and the optimizer:
# the EEG patch layout, the `pos_embed` antialias fix, the softplus temperature, the
# fixed temperature, the unnormalised EEG embedding and the flat Adam schedule are
# worth nothing measurable on this split, and layer fusion vs a single final block is
# a wash. So the 24-point gap to 70.4 is NOT in the semantic interface.
#
# What is left, and what these two arms test: the released code trains on ONE image
# per concept (`train_avg: True` averages the repetitions and then keeps `img[:, 0]`),
# i.e. 1654 pairs. We train on all 10 images per concept, i.e. ~15k pairs. That is a
# difference in the TASK, not the data volume: with 10 images per concept the
# objective also requires separating the ten images of one concept from each other,
# an instance-discrimination term the val/test protocol -- one image per concept,
# 200-way -- never asks for. The trained model may therefore be spending capacity on
# within-concept variation that the holdout does not reward. (This is deliberately
# not the false-negative story: same-concept pairs are 0.06% of all negative pairs on
# this split, at every batch size.)
#
#   arm 0 = full  + --train-slots 0   the official interface AND the official task
#   arm 1 = fuse  + --train-slots 0   our 3-layer fusion, official task
#
# If that reading is right these should move materially; if they do not, the remaining
# explanation is the one structural difference left, which is that official EEGiT
# aligns EEG to its OWN TRAINABLE ViT-B/16 + ProjectionHead and we align to frozen
# CLIP ViT-H/14 `block26` -- a space nobody optimised for EEG predictability.
#
# Cost: 2 arms x ~30 min = ~1 h of one GPU.
set -uo pipefail

ROOT="/project/peilab/why/eeg-retrieval"
cd "${ROOT}"

export PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export SUBJ="${SUBJ:-8}"
export TRAIN_SLOTS=0

export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"

mkdir -p outputs/slurm

ARMS=(full fuse)
export ARM="${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}"

echo "########## job ${SLURM_JOB_ID:-local}.${SLURM_ARRAY_TASK_ID:-0} arm=${ARM} TRAIN_SLOTS=${TRAIN_SLOTS} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds) ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

bash scripts/run_eegit_gate.sh
RC=$?

echo "########## job ${SLURM_JOB_ID:-local}.${SLURM_ARRAY_TASK_ID:-0} arm=${ARM} rc=${RC} @ $(date -Iseconds) ##########"
exit "${RC}"
