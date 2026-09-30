#!/usr/bin/env bash
#SBATCH --job-name=epd-struct
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-struct-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-struct-%j.err
#
# EEG Patch (epd) STRUCTURE TOWER, sub-08, ONE task, TRAINING ONLY.
#
# ---------------------------------------------------------------------------
# The one axis never varied, and what it is worth
# ---------------------------------------------------------------------------
# THREE structural runs have failed the same way (trunk DINOv3-B / DINOv2-L, interface
# topography -> eegit, objective L1+floor -> MSE) while all three regressed the SAME
# target: the fine 64x64 VAE latent, 16384 coordinates, normalised only by a
# per-channel scalar. `probe_targets.py` prices that target against its alternatives as
# a closed-form ridge on RAW EEG (sub-08, 63ch, val 150-way, chance 0.67%):
#
#     vae @ 64x64, uncentred  (ALL THREE RUNS)   dim 16384   val@1 5.07   margin +0.0617
#     vae @ 64x64, centred                       dim 16384   val@1 5.13   margin +0.1639
#     vae @ 16x16, centred                       dim  1024   val@1 6.07   margin +0.2731
#     vae @ 8x8,   centred       <- THIS RUN     dim   256   val@1 6.67   margin +0.3171
#     vae @ 4x4,   centred                       dim    64   val@1 6.53   margin +0.3623  (unreachable)
#
# 5.1x on the criterion this codebase scores structural tensors by. So this submission
# moves the target -- `--struct-scale 8 --struct-out-hw 8 --struct-base-hw 1
# --struct-center` -- and changes nothing else about the structural tower.
#
# ---------------------------------------------------------------------------
# Why 8x8, why it needed a code change, and why 4x4 is not the answer
# ---------------------------------------------------------------------------
# 4x4 tops the ladder by margin and is UNREACHABLE. On the pooled decoder branch the
# head is `Linear(d, base_ch*base_hw**2) -> (B,base_ch,base_hw,base_hw)` followed by
# three x2 `_up_block`s, so `out_hw == base_hw * 8` is the architecture's own
# parameterisation; `losses.py` raises "latent shape mismatch" when the head and the
# target disagree and there is no resample anywhere in the training path, so
# `--struct-scale` and `--struct-out-hw` must agree. With `base_hw` pinned at its
# default 8 the ONLY legal `out_hw` is 64 -- which is exactly the fine target every
# run so far has regressed. 4x4 would need `base_hw = 0.5`.
#
# `StructureTower.dense = (tokenizer_kind == "topography")`, so the measured-good
# `eegit` geometry takes the pooled branch, and the pooled branch ignores the 14x5
# token grid entirely. This was verified rather than reasoned about:
#
#     tokenizer     out_hw   result
#     topography    4        OK      dense=True   (the geometry the probe killed)
#     eegit         8        RAISE   "out_hw 8 is not base_hw 8 upsampled x2 three times"
#     eegit         64       OK      dense=False  (the fine target only)
#
# So the coarse ladder was unreachable with the geometry the probe says to ship. This
# submission adds `--struct-base-hw` (model.py + train.py, additive, default 8, every
# existing config constructs byte-identically) instead of settling for
# `--struct-scale 64 --struct-center`, which IS reachable and worth +0.1639 against
# +0.3171. 8x8 rather than 16x16 for the same reason, and it also carries the highest
# `val@1` of the entire centred ladder -- which matters because `val_vae_top1` is half
# of what selects the checkpoint.
#
# ---------------------------------------------------------------------------
# Two other changes, and one that is a judgment
# ---------------------------------------------------------------------------
#  * `--struct-center`: subtracts the fit-set per-pixel mean field. This is the flag
#    DA2 uses -- for depth. No VAE-target run has ever used it, and at the fine scale
#    it is worth +0.0617 -> +0.1639 on its own.
#  * `--struct-sel-w 5.0` (inherited 0.5): at 0.5 the structural term cannot move the
#    decision (semantic top1 spans ~15 points across the schedule, 0.5*vae_top1 spans
#    ~3), so "best" was in practice the semantic optimum and the saved checkpoint was
#    not the structural tower's best epoch -- the wrong artifact to export next. This
#    one is a judgment rather than a measurement; the per-epoch history is the primary
#    read if it turns out to have been wrong.
#  * `--w-var 1.0 --var-margin 0.15` is KEPT, and that is measured. DA2's header
#    records that turning it off under centring+MSE collapsed within 248 steps of a
#    2-epoch smoke (`vae_cos +0.000`, `vae_var 0.000`, only consistent with a spatially
#    flat prediction).
#
# Everything else -- trunk, eegit geometry, MSE, the semantic tower and its schedule --
# is `epd_opt_dino3_eegit`'s, the best structural run so far. Reasoning per flag is in
# `scripts/run_epd_struct.sh`.
#
# ---------------------------------------------------------------------------
# Training only, and why
# ---------------------------------------------------------------------------
# No export, no generation, no seven metrics. An 8x8x4 latent decodes to 64x64, so it
# cannot be a 512px `img2img` init either; it is injectable only through a learned
# channel (a zero-init residual on the UNet's `conv_in`), which does not exist yet.
# Scoring generation before that is settled would price a route this target cannot
# take. The output is `_best.pt`, `_result.json` and the curve.
#
# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------
#   target caches   present and content-verified in-run against the fine cache
#                   (sampled 8x8-block area-average; DA2's validation covered depth only)
#   train 100 epochs ~1.5-2 h, two ViT-B towers, fp32, batch 128
#   no export / no generate / no metrics -- the saving is the whole 5-arm gen matrix
# 4 h requested so a slow node finishes rather than being killed with the checkpoint
# unwritten; ~2 h expected.
set -uo pipefail

ROOT="/project/peilab/why/eeg-retrieval"
cd "${ROOT}"
mkdir -p outputs/slurm outputs/logs

export PY="${PY:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export SUBJ="${SUBJ:-8}"

# The cache pins must be set HERE, not only inside `run_epd_struct.sh`, because the
# wiring test below runs before that script. /home is at 100% and a model download
# there fails with ENOSPC, which would fail the test gate and abort the job before
# training ever starts -- measured on the login node while validating this change.
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${TORCH_HOME}"

# Named and exported rather than inlined: a submission whose decisive flag is a
# literal inside the runner is unreadable a week later.
export ARCH="${ARCH:-dino3}"
export STRUCT_SCALE="${STRUCT_SCALE:-8}"
export SEED="${SEED:-2025}"
# `c` for centred, in the tag, so this run is never confused on disk with the three
# uncentred ones.
export TAG="${TAG:-epd_struct_${ARCH}_eegit_vae${STRUCT_SCALE}c}"

# The wiring test is the first gate: `--validate-only` returns BEFORE the model is
# built, so it cannot prove that a coarse `--struct-out-hw` constructs. The test does,
# for this exact base_hw/out_hw pair, in seconds on the login node.
echo "########## epd-struct ${SLURM_JOB_ID:-local} arch=${ARCH} scale=${STRUCT_SCALE} tag=${TAG} sub-${SUBJ} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds) ##########"
"${PY}" -u scripts/test_epd_arch.py > outputs/logs/epd_struct_archtest_${SLURM_JOB_ID:-local}.log 2>&1
RC=$?
if (( RC != 0 )); then
  echo "########## epd-struct wiring tests FAILED rc=${RC}; not training ##########"
  tail -20 outputs/logs/epd_struct_archtest_${SLURM_JOB_ID:-local}.log
  exit "${RC}"
fi
echo "########## wiring tests passed ##########"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

bash scripts/run_epd_struct.sh
RC=$?

echo "########## epd-struct ${SLURM_JOB_ID:-local} tag=${TAG} rc=${RC} @ $(date -Iseconds) ##########"
exit "${RC}"
