#!/usr/bin/env bash
#SBATCH --job-name=epd-coarse-gate
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-coarse-gate-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-coarse-gate-%j.err
#
# THE GATE for the structural branch's direction: at which spatial scale, if any,
# does this EEG linearly carry the target at all?
#
# Why this is the deciding measurement
# ------------------------------------
# Two closed-form results already exist and they point opposite ways for the shipped
# 64x64 targets:
#
#   vae   @ 64x64   margin +0.0617   (top1 1.00% vs 0.50% chance, mean rank 85/200)
#   depth @ 64x64   margin -0.3735   (worse than a constant predictor)
#
# A negative margin is NOT a statement that EEG has no spatial information. It is a
# statement about the estimator: a 64x64 spatial target has 4096 dimensions per sample,
# so the ridge has to estimate 15750 x 4096 coefficients from 15040 rows. The variance
# of that fit can exceed the signal even when the signal is there. Coarsening to
# 4x4 reduces the target to 16 dimensions -- a 256x reduction in the number of
# coefficients to estimate -- while leaving the EEG side untouched.
#
# So this isolates SCALE from everything else. Each coarse target is a deterministic
# linear functional (area average) of the fine target already cached and already
# probed, on the same images, with the same fit/val/test split, the same SVD and the
# same lambdas. Any difference in margin is attributable to scale alone. No GPU
# training, no new encoding, no re-derived targets.
#
# What the answer decides
# -----------------------
#   * depth becomes decodable at some coarse scale -> the structural tower gets a
#     real target at that scale, and Depth Anything V2's output is a genuine
#     ControlNet-depth condition this EEG can drive. The tower's job becomes
#     "recover the coarse depth layout", which is a well-posed regression.
#   * depth stays negative at every scale -> depth is not carried by this EEG at any
#     reachable resolution, and a depth-ControlNet structural branch is not
#     defensible. The branch would then have to be dropped rather than retuned,
#     which is a different (and cheaper) experiment.
#   * the vae ladder matters the same way: it tells us whether the shipped VAE
#     latent target was simply too fine, which is the hypothesis the previous
#     collapse is most consistent with.
#
# The 64x64 `depth` and `vae` rows are included deliberately as anchors: they are the
# numbers the rest of the project is indexed against, and running them here means the
# ladder is read within one job rather than across two with different conditions.
#
# Cost: the single SVD of the 15040 x 15750 fit matrix, which is why this asks for a
# GPU -- on CPU it is ~10+ min and is the entire cost of the job; on GPU it is seconds.
# Everything after the factorisation is one matmul per target per lambda.
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

mkdir -p outputs/slurm outputs/probe/layer_sweep

echo "########## job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds) ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"${PYTHON}" -u scripts/epd/probe_targets.py \
  --subject 8 \
  --channels all \
  --device cuda \
  --targets depth depth4 depth8 depth16 depth32 vae vae4 vae8 vae16 vae32 \
  --out outputs/probe/layer_sweep/coarse.json
RC=$?

echo "########## job ${SLURM_JOB_ID:-local} rc=${RC} @ $(date -Iseconds) ##########"
exit "${RC}"
