#!/usr/bin/env bash
#SBATCH --job-name=nwret-sub08
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/nwret-sub08-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/nwret-sub08-%j.err
#
# NW-Retrieval: the full axis sweep for sub-08, in one job, probe first.
#
# Everything runs sequentially inside a single allocation (see
# scripts/run_pipeline.sh for the ordering and the reasoning behind it).
# The pipeline is resumable: an arm whose result JSON is already complete is
# skipped, so `--requeue` -- and a plain re-submission -- cost at most the arm
# that was in flight.
#
# Ordering
# --------
# P0  ridge baseline (closed form)                  -- the floor every arm must clear
# P1  extract per-layer CLIP features               -- one pass, all 33 targets
# P2  probe: which layer is most alignable?         -- closed form, val-selected
# P3  alignment-target scan                         -- extend the confirmed lever
# P4  image-side multi-layer alignment              -- doc phase 2, target axis
# P5  EEG-encoder depth                             -- doc phase 1, encoder axis
# P6  EEG-side multi-layer fusion (uniform, routed) -- doc phase 2, encoder axis
# P7  tokenisation variants                         -- EEGiT prices this as large
# P8  leaderboard + the five questions, answered explicitly
#
# There are TWO independent layer axes and the first sweep only moved one of them.
# `--target-layer` is which layer of the FROZEN IMAGE tower the EEG must match;
# `--layers` is which depth of the EEG ENCODER emits the aligned feature. The doc's
# phase 1 is written about the second; the first turned out to be worth +13.00 test
# Top-1 (34.00 at block26 vs 21.00 at the final layer, everything else fixed). Both
# are swept here and the leaderboard labels them separately.
#
# Why ordering was the fix, and why the expensive scan is not first any more
# -------------------------------------------------------------------------
# The first version of this job queued 19 arms and ran 6-8 hours. Every arm landed
# between 16% and 21.5% test Top-1 while a closed-form linear ridge reached 25.0% on
# the same features and split, so the sweep was ranking overfitting and none of it
# was interpretable. The deeper error was the axis: the doc prices the alignment
# layer as the single largest lever and phase 1 says to sweep all 32 CLIP blocks, but
# the implementation swept the *EEG pathway's* depth instead, and the target was
# hard-wired to CLIP's final layer (our cached features are 1024-d, which is exactly
# visual.proj's output width). The lever was never actually tested.
#
# The expensive scan does not have to be first: ridge is closed form, so it needs no
# training, and its factorisation depends only on the EEG, so one decomposition
# prices every candidate layer -- tens of seconds per layer instead of hours.
#
# Not in this run, deliberately (see the header of scripts/run_pipeline.sh):
# the CLIP ViT-H-14 / DINOv2-L backbones (7.4x the activation memory forces batch
# 512 -> ~64, confounding backbone with batch size) and the DINOv2 structural branch
# (its target per docs 5.2 is SDXL VAE latents, a separate axis).
#
# Numbers are only comparable to SAMGA because the retrieval protocol is
# identical (200-way, train reps 4 / test reps 80 averaged, metric copied
# verbatim from third_party/SAMGA/module/util.py). Two deliberate differences:
# we select checkpoints on a concept-level holdout of training concepts rather
# than on the test split, and we do not fine-tune the image tower (it is the
# frozen target).
#
# Two further deliberate deviations from SAMGA's training recipe, both measured
# rather than assumed (sub-08, F1 vs F3 and F2 vs F4): the stage-1 MMD warm-up is
# off, and the loss temperature uses exp rather than softplus. SAMGA's EEG encoder
# is a from-scratch projector that needs a geometry warm-up before instance
# discrimination can work; ours is a pretrained ViT that already has one, and the
# warm-up cost 3.5-4.0 test Top-1 here. Both remain behind flags for the ablation.
set -uo pipefail

ROOT="/project/peilab/why/eeg-retrieval"
cd "${ROOT}"

export PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export SUBJ="${SUBJ:-8}"
export EPOCHS="${EPOCHS:-50}"
# 11h budget under a 12h wall clock. The budget must stay BELOW the wall time or the
# internal guard never fires and the job is killed mid-arm instead of stopping after
# one. Measured cost: one arm is ~17-25 min (PA: 1098s wall, 1007s train + 70s val,
# 28 epochs, 1240 ms/step; F2: 1008s), so the 10 new arms are ~4h, plus minutes for
# the probe and the layer extraction. The guard defers -- never aborts -- an arm it
# cannot fit, and a deferral is reported in the summary and on the next submission.
export PIPELINE_HOURS="${PIPELINE_HOURS:-11}"

mkdir -p outputs/slurm outputs/logs

echo "########## job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds) ##########"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

bash scripts/run_pipeline.sh
RC=$?

echo "########## job ${SLURM_JOB_ID:-local} rc=${RC} @ $(date -Iseconds) ##########"
exit "${RC}"
