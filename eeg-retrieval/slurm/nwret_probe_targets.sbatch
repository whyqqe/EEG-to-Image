#!/usr/bin/env bash
#SBATCH --job-name=nwret-probetgt
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/nwret-probetgt-%A_%a.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/nwret-probetgt-%A_%a.err
#
# STAGE 0 of the EEG-to-image rebuild: which target space can this EEG actually carry?
#
# Why this runs before anything else
# ----------------------------------
# The previous structural tower was trained against SDXL VAE latents and a monocular
# depth map, and it collapsed to the conditional mean. The numbers, measured on the
# exported predictions of the deployed run:
#
#   depth   across-sample var / total var  0.0068   (the target itself sits at 0.63)
#   depth   mean pairwise cosine           0.9991
#   depth   r(pred_i, GT_i)               +0.6485   vs +0.6540 for the constant predictor
#   VAE     instance Top-1                 1.00%    against 0.50% chance, mean rank 85/200
#
# A collapsed head is not evidence that the target is unreachable -- it is evidence
# that under L1 and limited capacity the conditional mean was the best answer
# available. So before spending another GPU run on a structural target, this asks
# the question one level up: how much of each candidate space is *linearly* decodable
# from the EEG, and how much of that is above what a constant predictor already gets
# for free?
#
# The probe is closed form (ridge, one SVD shared across targets), so it costs minutes
# rather than hours and there is no training to confound the answer. It is the same
# machinery as `probe_layers.py`, which produced the val 17.33 / test 27.50 reference
# for CLIP `block26`; this scans target SPACES, which is the axis the collapse lives on.
#
# What the four arms are for
# --------------------------
#   arm 0  sub-08, all 63 channels        the setting the shipped pipeline trains on
#   arm 1  sub-08, occipito-parietal 17   the setting probe_layers.py's 17.33 used,
#                                          so the new numbers are read against a
#                                          reference that exists rather than a new one
#   arm 2  sub-10, all 63 channels        the second subject the experiment needs
#   arm 3  sub-10, occipito-parietal 17   same comparison on sub-10
#
# Both channel settings matter and neither is redundant: the pipeline trains on all 63,
# while every historical probe number is on 17, and the difference between the two is
# large enough (17ch/40-fit smoke put clip_block26 at 2.20 against 17.33 at full fit)
# that mixing them up would silently invalidate the comparison.
#
# How to read the result
# ----------------------
# `probe_targets.py` prints a verdict per target. The decisions it drives:
#   * dino_l vs clip_block26  -> whether the structural tower should be moved onto
#     DINOv2 at all. If DINOv2 is not more decodable, the swap trades one collapse
#     for another target that is also absent.
#   * vae / depth margin and variance ratio -> whether a spatial head can exist.
#     A margin <= 0 means the linear ceiling IS the mean, and no head design fixes it.
#
# Targets are stimulus properties, not recording properties, so the feature banks are
# shared across all ten subjects -- `--target-root` and `--dino-dir` point at single
# copies. The DINOv2/SigLIP bank was built by `eeg-thor-a`'s teacher builder, whose
# `list_images()` ordering is the same sorted(categories)-then-sorted(images) walk under
# the same `images_root`; that equality is asserted against our own index files
# (16540 and 200 rows, zero mismatches) before the arrays are used.
#
# Cost: 4 arms x ~10 min of one GPU. Almost all of it is the single SVD of the
# 15040 x 15750 fit matrix; the per-target work after that is one matmul each.
set -uo pipefail

ROOT="/project/peilab/why/eeg-retrieval"
cd "${ROOT}"

export PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# /home is at 100% capacity and a download there fails with ENOSPC mid-run.
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"

mkdir -p outputs/slurm

# array index -> (subject, channels)
SUBJECTS=(8 8 10 10)
CHANNELS=(all occipito_parietal all occipito_parietal)
i="${SLURM_ARRAY_TASK_ID:-0}"
SUBJ="${SUBJECTS[$i]}"
CH="${CHANNELS[$i]}"

OUT="outputs/sub$(printf '%02d' "${SUBJ}")/probe_targets_${CH}.json"

echo "########## job ${SLURM_JOB_ID:-local}.${SLURM_ARRAY_TASK_ID:-0} sub-${SUBJ} channels=${CH} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds) ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# `siglip` is included next to `dino_l` because both are frozen self-supervised image
# banks of the same kind, and if the structural tower is going to be moved off the
# CLIP space it should be moved onto whichever of them the EEG actually carries --
# not onto the first one that was convenient.
"${PYTHON}" -u scripts/nwret/probe_targets.py \
  --subject "${SUBJ}" \
  --channels "${CH}" \
  --targets clip_block26 clip_pooled dino_l siglip vae depth \
  --out "${OUT}"
RC=$?

echo "########## job ${SLURM_JOB_ID:-local}.${SLURM_ARRAY_TASK_ID:-0} rc=${RC} @ $(date -Iseconds) ##########"
exit "${RC}"
