#!/usr/bin/env bash
#SBATCH --job-name=epd-opt
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-opt-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-opt-%j.err
#
# EEG Patch dual tower, sub-08, one arm, training through the seven metrics.
#
# ---------------------------------------------------------------------------
# Why this is ONE task where the previous submission was three
# ---------------------------------------------------------------------------
# The last run's array axis was the structural TRUNK (dino3 / mae / in21k), on the
# argument that a single arm gives no way to attribute its result. That argument is
# still correct in general and it is not what changed. What changed is that the
# three-arm run finished, and all three tasks failed the same way -- the export gate
# rejected the structural branch for a per-sample correlation margin at or below the
# constant predictor. Three arms, three identical failures, one conclusion about the
# variable they held fixed and varied nothing about: the trunk was not the lever.
#
# `run_epd_struct_probe.sh` then measured where the lever is, by fitting a closed-form
# ridge from every candidate tensor in the structural tower to the VAE latent and
# scoring it against the constant predictor (val-selected lambda, 200-way, sub-08):
#
#   row (all fitted to the SAME VAE latent)      val@1  test@1    rank  margin
#   raw EEG, 10 view slots (the floor to beat)     5.07    6.00    24.8  +0.0617
#   pretrained struct_grid (topography geom)       1.40    0.00    90.4  -0.0745
#   pretrained struct_gridmean                     0.87    1.00   100.1  -0.0992
#   pretrained struct_pooled                       1.00    0.00   100.3  -0.0928
#   TRAINED    struct_grid                         1.20    0.00   101.5  -0.0923
#   TRAINED    struct_gridmean                     0.93    1.00   102.1  -0.0920
#   TRAINED    struct_pooled                       1.27    0.50    98.9  -0.0917
#   TRAINED    semantic tower's 1024-d embed       4.47    6.50    38.9  +0.0502
#   (chance: val@1 0.50, rank 100.5)
#
# Every structural tensor is at or BELOW the constant map, at initialisation as well
# as after training (best pretrained 1.40 vs best trained 1.27), so the geometry the
# trunk was fed carried nothing for no loss function to extract. Meanwhile the
# semantic tower's embedding -- EEGiT's OWN region-band geometry into a ViT-B --
# reaches the raw-EEG ceiling. The target is reachable through a deep ViT EEG encoder;
# the interface was wrong, and a working one was already in the codebase.
#
# So this run changes the interface rather than the trunk, which means the trunk does
# not need to be swept: it was never the variable that decided the outcome, and the
# probe says so directly rather than by elimination. DINOv3 is kept as the trunk
# because the separate trunk is what makes this a second tower -- and because its
# structural prior is still the thing worth testing, once it is fed an interface that
# carries the target.
#
# ---------------------------------------------------------------------------
# The three deltas from `epd_dual.sbatch`, and how strong the evidence is for each
# ---------------------------------------------------------------------------
# 1. `--struct-tokenizer eegit` (was `topography`).   MEASURED, and the reason this
#    submission exists. See the table above: the topography geometry's features are
#    at the constant predictor on all six tensors, EEGiT's geometry's features are at
#    the raw-EEG ceiling. This is the one change with a direct before/after number
#    attached to it.
#
# 2. `--vae-loss mse --w-var 0` (was L1 + a variance-floor hinge).   ARGUED, not
#    measured, and worth stating plainly. The floor constrained only the prediction's
#    spread, so "conditional mean field + zero-mean noise" satisfied it while costing
#    almost nothing in L1 -- which is exactly what the last run produced (variance
#    ratio 0.6577, healthy-looking; per-sample r +0.063 against the constant's
#    +0.157). MSE charges for that noise by construction, and it is the loss every
#    successful row of the probe was fitted with (all of them are closed-form L2).
#    There is no corresponding evidence anywhere in this project that L1 suffices on
#    these features.
#
#    Note this change is NOT a fix for the last run's failure and cannot be: the probe
#    shows the features carried nothing, so no loss could have helped. It is a fix for
#    a defect that would have bitten the moment the features became usable, and if the
#    VAE branch fails again while the trunk probe reads positive, this is the item to
#    reopen -- not the geometry, and not the backbone.
#
# 3. Nothing else. The semantic tower, the schedule, the decoder, the export, the
#    generator and the metrics are the previous run's, unmodified.
#
# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------
#   VAE target cache   already built and content-validated on the shared path; the
#                      build is skipped, and if the cache is missing this rebuilds it
#                      (~13 min of GPU for 16540 images) rather than failing
#   train 100 epochs   ~1.5-2 h   (two ViT-B towers, fp32, no autocast)
#   export             ~10 min
#   generate 4 arms    ~15 min    (4 x 200 images x 28 SDXL steps)
#   seven metrics      ~10 min
# so ~2.5-3 h expected; 5 h requested so a slow node finishes rather than being
# killed with the generations on disk and the metrics half-written.
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

# The two knobs this run is about, exported rather than left to the defaults so that
# the log's first line names the configuration that produced the numbers below it. A
# single-arm submission whose geometry is implicit would be unreadable a week later.
export ARCH="${ARCH:-dino3}"
export STRUCT_TOKENIZER="${STRUCT_TOKENIZER:-eegit}"
export TAG="${TAG:-epd_opt_${ARCH}_${STRUCT_TOKENIZER}}"

echo "########## epd-opt ${SLURM_JOB_ID:-local} arch=${ARCH} struct-tokenizer=${STRUCT_TOKENIZER} sub-${SUBJ} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds) ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

bash scripts/run_epd_dual.sh
RC=$?

echo "########## epd-opt ${SLURM_JOB_ID:-local} arch=${ARCH} rc=${RC} @ $(date -Iseconds) ##########"
exit "${RC}"
