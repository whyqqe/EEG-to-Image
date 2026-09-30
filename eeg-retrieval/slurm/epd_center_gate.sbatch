#!/usr/bin/env bash
#SBATCH --job-name=epd-center-gate
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-center-gate-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-center-gate-%j.err
#
# The second half of the gate: is the target's INSTANCE content in the EEG, once the
# shared mean layout is taken out of the metric?
#
# Why job 605981's verdict is not the whole story
# -----------------------------------------------
# 605981 measured, against the uncentered target, on sub-08 / all 63 channels:
#
#   depth     margin -0.3735   floor r +0.5333
#   depth4    margin -0.4505   floor r +0.6076
#   depth8    margin -0.3982   floor r +0.5731
#   depth16   margin -0.3847   floor r +0.5533
#   depth32   margin -0.3781   floor r +0.5418
#   vae       margin +0.0617   floor r +0.1030
#   vae4      margin +0.1962   floor r +0.1685   <- best spatial margin measured
#
# and the obvious reading -- "depth is not decodable" -- is too strong. The floor
# there is `pearson_rows(mu_t, gt)`, and `pearson_rows` centres each ROW, so it is the
# correlation between the SHAPE of the fit-set mean depth map and the SHAPE of each
# individual map. Depth maps of COCO scenes nearly all have the same layout (far at
# the top, near at the bottom), so that floor is high for a reason that has nothing to
# do with this EEG. Centring also shows up on the other side: the ridge design matrix
# is z-scored per feature, so it has mean zero and NO intercept, which means it cannot
# express the mean map at all -- it has to reconstruct it out of EEG coefficients, and
# `lam*` was selected at 1e5, the heaviest shrink in the grid, which is what a fit
# that is finding nothing at every lambda looks like.
#
# So the uncentered margin is not a clean measurement of whether the EEG carries the
# target. This job re-asks with the fit-set per-pixel mean subtracted from the target,
# which is exactly "is the per-concept deviation from the shared mean map in the EEG".
# With that centring the constant predictor is the zero vector and its row Pearson r is
# identically 0, so the margin collapses to `r_pred` -- one interpretable number.
#
# What the answer decides, and why both branches are actionable
# ------------------------------------------------------------
#   * centered depth becomes decodable (r_pred meaningfully > 0 and rising as the grid
#     coarsens) -> depth IS a viable structural interface, the branch should predict
#     the deviation from the mean depth map rather than the map, and a
#     Depth-Anything-V2 trunk feeding ControlNet-depth is back on the table.
#   * centered depth stays ~0 while centered vae4 is clearly positive -> depth is
#     genuinely absent from this EEG and the depth/ControlNet route is dead; the
#     structural target is the VAE latent, and the coarsening result from 605981
#     (+0.1962 at 4x4 against +0.0617 at 64x64) says which scale to supervise at.
#
# The `vae`/`vae4` rows are included centred as well, and they are the control that
# makes the depth rows readable: if centring moves vae4 too, the effect being measured
# is the centring operation rather than depth specifically.
#
# Cost: the same single 43 s SVD on GPU, then one matmul per target.
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
  --center-spatial \
  --targets depth depth4 depth8 depth16 depth32 vae vae4 vae8 vae16 vae32 \
  --out outputs/probe/layer_sweep/centered.json
RC=$?

echo "########## job ${SLURM_JOB_ID:-local} rc=${RC} @ $(date -Iseconds) ##########"
exit "${RC}"
