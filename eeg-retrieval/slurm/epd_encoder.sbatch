#!/usr/bin/env bash
#SBATCH --job-name=epd-enc
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-enc-%A_%a.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-enc-%A_%a.err
#
# STAGE 2 -- the encoder axis. Everything else is held fixed.
#
# What Stage 0 + Stage 1 already established, which is what puts the encoder in the dock
# ------------------------------------------------------------------------------------
#   samga_clip5   THEIR encoder + OUR target space (5 CLIP ViT-H/14 layers) = 65.00 peak
#   base_ema      OUR encoder   + OUR target space (CLIP block26)           = 50.50 peak
# Two runs, same target space, 14.5 points apart. So the target space is exonerated: it
# is capable of 65 in a frame that works. What differs is the encoder/interface.
#
#   samgaish      OUR encoder + THEIR recipe (17ch, MMD, softplus, bs512, lr1e-4) = 23.50
#   samga_rn50    THEIR encoder + that same recipe family                       = 59.50
# Same recipe, 36 points apart. The recipe axes we have spent weeks tuning (channels,
# MMD, temperature, lr, batch) are therefore not where the deficit lives -- ch17 and mmd
# were both measured TIES against the control in Stage 1, and the recipe bundle that
# reproduces the reference's own settings is the WORST arm we have ever run.
#
# So the encoder/interface is the remaining suspect, and this stage tests it directly
# rather than by argument: the reference's own framework, its own feature space, its own
# recipe, four encoders behind one flag. EEGProject is already measured (59.50 peak).
#
#   EEGTransformer   the control that matters: a transformer EEG encoder inside a frame
#                    known to reach 59-65. Our architecture is a transformer (ViT-B/16
#                    over an EEG-as-image tokenisation) with a pretrained natural-image
#                    patch_embed conv as the interface.
#   EEGNet           an independent convolutional encoder.
#   TSConv           the reference's other convolutional encoder.
#
# Three outcomes, each of which settles the architecture question
# --------------------------------------------------------------
#   all four within noise (~8 points)   -> the encoder family is NOT the lever. The
#                                          deficit is in the interface (how EEG becomes
#                                          tokens / what the pretrained conv sees) or in
#                                          something not yet isolated, and the next probe
#                                          is the tokenisation rather than the backbone.
#   EEGProject far ahead of EEGTransformer -> transformers underperform here, and the
#                                          EEG Patch architecture should move to a
#                                          convolutional or hybrid front end. Concrete
#                                          and immediate.
#   EEGTransformer competitive          -> our backbone family is fine and the problem is
#                                          strictly the interface. Then the ViT is kept
#                                          and the tokenisation is replaced.
#
# Cost: identical to epd_anchor.sbatch (the same script, one flag different), so minutes
# per arm. Three in parallel because they share only read-only inputs and must be
# compared against each other.
set -uo pipefail

ROOT="/project/peilab/why/eeg-retrieval"
cd "${ROOT}"
mkdir -p outputs/slurm outputs/logs outputs/anchor

export PY="${PY:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export SUBJ="${SUBJ:-8}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"

# EEGProject is already measured in epd_anchor; no need to spend a slot on it.
ENCS=(EEGTransformer EEGNet TSConv)
i="${SLURM_ARRAY_TASK_ID:-0}"
export ARM="samga_enc_${ENCS[$i]}"

echo "########## enc ${SLURM_JOB_ID:-local}.${i} arm=${ARM} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds) ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

bash scripts/run_epd_anchor.sh
RC=$?

echo "########## enc ${SLURM_JOB_ID:-local}.${i} arm=${ARM} rc=${RC} @ $(date -Iseconds) ##########"
exit "${RC}"
