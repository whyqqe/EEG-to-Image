#!/usr/bin/env bash
#SBATCH --job-name=h2g-sub08
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/h2g-sub08-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/h2g-sub08-%j.err

# HCMA-2G: introduce a GRANULARITY axis orthogonal to the modality axis.
#
# Part 1 (new): DINOv2 patch-grid local semantic targets -- every HCMA alignment
#               target to date is a global pooled vector (timm num_classes=0 returns
#               only CLS), so I(e; y_loc | y_glob) was never supervised.
# Part 2 (new): 2x2 heads = modality x granularity, with heteroscedastic abstention on
#               the high band, coarse-to-fine residual consistency, and VICReg geometry.
# Part 3 (new): band-anchored generation using the FROZEN calibrated semantics (so the
#               saturated high-level metrics are protected) at strength=1.0 with no
#               ControlNet and no img2img init, isolating the anchor's effect.
# Part 4:       first-ever standard-7 + FID scoring of the clean_p0p1 arms.
#
# Both branches run here: the a00 conditions are intra-subject, the al_* conditions are
# LOSO/inter-subject (fitted on the 9 source subjects, n_src=1800).
set -euo pipefail
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
export SUB="${SUB:-sub-08}"
bash "${NB_ROOT}/scripts/nda/run_h2g_sub08.sh"
