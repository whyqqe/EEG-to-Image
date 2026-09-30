#!/usr/bin/env bash
#SBATCH --job-name=epd-loso-smoke
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=01:30:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-loso-smoke-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-loso-smoke-%j.err
#
# =============================================================================
# epd-loso-smoke -- does the LOSO path run, and how long is an epoch?
# =============================================================================
# Not a result, a measurement. `run_epd_loso.sh --dry-run` proves the FLAGS are
# consistent; it touches no data and so cannot catch the things that actually break a
# wide run:
#
#   * `load_loso` materialising nine subjects at once. The arithmetic says ~19 GiB at
#     the concatenate peak; the arithmetic has been wrong before, and an OOM kill at
#     hour six with no output is the worst possible way to find out.
#   * the MVNN path resolving the right whitener per ROLE (train for the nine
#     sources, test for the holdout). A silent mix-up here changes what every subject
#     is measured in and produces a plausible number.
#   * `subject_ids` actually reaching `LayerFusion`. The plumbing exists end to end
#     and is unit-tested, but no real run has ever carried a `subject_of_row` tensor
#     through a training step -- PROTOCOL_INTER.md gap 4, which this closes or does not.
#   * the EEG tokenizer's bilinear resample, which is the throughput bottleneck. The
#     whole point of measuring is that the 24h limit on the real job was estimated
#     from the intra-subject run's step time scaled by the row-count ratio, and that
#     scaling ignores the tokenizer entirely -- it is per-token work, not per-example
#     work, so the estimate could be wrong in either direction.
#
# What is reported at the end is ms/step and the seconds of each phase, which is what
# the real job's time limit gets set from.
#
# 2 epochs is not a training run: it is enough to show the loop, the EMA, the
# evaluation pass on the held-out subject, and the result JSON all execute end to end.
# A smoke that stops before the test pass would leave the least-exercised code path --
# `evaluate` on a fold whose val set is EMPTY, which is the new `--val-concepts 0` case
# -- untested, and that is the path most likely to have a bug.
set -euo pipefail

ROOT="/project/peilab/why/eeg-retrieval"
cd "${ROOT}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"

mkdir -p outputs/slurm outputs/loso/smoke

echo "########## epd-loso-smoke ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
free -g | head -2

SOURCES=(1 2 3 4 5 6 7 9 10)
TARGET=8
TAG="loso_tgt08_smoke"
OUT="${ROOT}/outputs/loso/smoke"
FEAT="${ROOT}/outputs/features/clip_h14_layers"

# The REAL flag set, with three deliberate deviations, each of which is only safe
# because the number it produces is thrown away:
#   --smoke           2 epochs and a capped val set (there is no val set here anyway)
#   --batch-size 64   4x smaller, so the step time is measured away from the memory
#                     ceiling and one OOM does not end the measurement
#   --epochs 2        by --smoke
# Everything else -- 63 channels, val-concepts 0, select-last, MVNN, the EEGiT
# representation, the full nine-subject fit set, block26 -- is exactly what the real
# run uses. A smoke that also shrinks the DATA proves nothing about the data path.
"${PYTHON}" -u scripts/epd/train.py \
  --tag "${TAG}" \
  --out-dir "${OUT}" \
  --seed 2025 \
  --split-seed 2025 \
  --source-subjects "${SOURCES[@]}" \
  --target-subject "${TARGET}" \
  --channels all \
  --val-concepts 0 \
  --select-last \
  --smoke \
  --tokenizer eegit \
  --patch-style time-region \
  --patch-size 16 \
  --n-patches-w 14 \
  --backbone timm:vit_b16_in21k_orig \
  --freeze-blocks 0 \
  --layers 12 \
  --fusion-mode none \
  --pool mean \
  --timm-global-pool avg \
  --head-kind eegit \
  --img-head-kind eegit \
  --head-drop 0.5 \
  --d-embed 1024 \
  --target-features "${FEAT}" \
  --target-layer block26 \
  --target-fusion single \
  --mvnn train \
  --mvnn-shrinkage lw \
  --fixed-temp \
  --softplus \
  --no-eeg-l2norm \
  --optimizer adamw \
  --lr 5e-4 \
  --backbone-lr-mult 0.1 \
  --warmup-epochs 1 \
  --cosine \
  --min-lr-ratio 0.01 \
  --ema-decay 0.999 \
  --ema-warmup-steps 20 \
  --epochs 2 \
  --batch-size 64 \
  --patience 0 \
  --aug full \
  --stage1-epochs 0
RC=$?

echo "########## epd-loso-smoke rc=${RC} @ $(date -Iseconds)"
echo "--- the smoke's result json, which is what proves the eval path ran ---"
[[ -f "${OUT}/${TAG}_result.json" ]] \
  && "${PYTHON}" -c "
import json,sys
r=json.load(open('${OUT}/${TAG}_result.json'))
p=r['protocol']; t=r['test']
print(f'  n_subjects       {p[\"n_subjects\"]}')
print(f'  source_subjects  {p[\"source_subjects\"]}')
print(f'  target_subject   {p[\"target_subject\"]}')
print(f'  selection        {p[\"selection\"]}')
print(f'  test top1/top5   {t[\"top1\"]:.2f} / {t[\"top5\"]:.2f}  (n={t[\"n\"]})')
print(f'  ms/step          {r[\"timing\"][\"ms_per_step\"]}')
print(f'  train seconds    {r[\"timing\"][\"train_seconds\"]}')
print(f'  steps            {r[\"timing\"][\"steps\"]}')
# Extrapolate to the real job so the 24h limit is a measurement, not a guess.
ms=float(r['timing']['ms_per_step']); steps=int(r['timing']['steps'])
real = ms*steps/1000*(582/max(steps,1))
print(f'  -> real job estimate {real/3600:.1f} h for 50 epochs at 582 steps/epoch')
" \
  || echo "  MISSING: ${OUT}/${TAG}_result.json"

exit "${RC}"
