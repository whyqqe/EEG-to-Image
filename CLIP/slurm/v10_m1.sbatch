#!/bin/bash
#SBATCH --job-name=v10-m1
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/v10-m1-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/v10-m1-%j.err
#
# v10 STAGE 2.5 -- M1, THE SOURCE-METRIC TEMPLATE (docs/eeg2image_v10_pipeline.md).
#
# THE HYPOTHESIS, and where it comes from.
#
# Stage 1 established that the structural term's gain is GATED by how well the EEG-side
# reference metric is estimated: +3.08pp at R=80, +1.12pp at R=20, -0.43pp at R=1 -- monotone.
# Stage 2 then tried to fix that with shrinkage and FAILED in an informative way: increasing
# `--rep-shrink` reduced the gain at EVERY R (3.08 -> 1.13 at R=80). Shrinkage buys variance
# with ISOTROPIC BIAS -- it pulls the covariance estimator toward the identity, so it degrades
# the very metric the structural term exists to use. Variance reduction that carries no such
# bias is therefore the ONLY remaining route, and one is licensed by measurement rather than
# assumption: the concept metric is shared across subjects at corr(D_eeg_s, D_eeg_t) = +0.565
# over 45 pairs (chance -0.003). A source-averaged template `de_src` is a lower-variance,
# unbiased-for-the-shared-component estimate of the same object `de_target`.
#
# THE PREDICTION (pre-registered, so a failure is a result and not a re-specification).
#
# `de_used = (1-m)*de_target + m*de_src`. With independent estimates of a shared metric the
# variance is (1-m)^2*s_t^2 + m^2*s_s^2/S, whose minimiser is m* = S/(S+1) = 0.90 for S=9. The
# source average is NOT unbiased for the target, however -- the +0.565 correlation leaves ~43%
# target-specific structure -- so the predicted peak is pulled INWARD, to roughly m in [0.5, 0.9].
#
#   H1  the mix curve has an INTERIOR PEAK. A peak at m=0 means the template carries nothing;
#       a peak at m=1 means the target's own metric is pure noise, which the +0.565 measurement
#       already refutes. Either endpoint is a falsification, not a tuning result.
#   H2  the gain from mixing is LARGER at R=20 than at R=80. That is the whole point: R=20 is
#       where the target's metric is noisiest, so a variance-reduced reference should help most.
#       If the gain is flat in R, the mechanism is not variance reduction.
#   H3  the diagnostic corr(de_src, de_target) is ~+0.4..0.6. It is recorded PER RUN before any
#       accuracy is read, because it is the one number that predicts pay-off in advance: it is
#       the shared component relative to the total, i.e. exactly the signal-to-noise the blend
#       can exploit. Near 0 would mean the template is a differently-shaped metric and mixing
#       must hurt -- so the diagnostic can kill the whole direction on a single fold.
#
# WHY THE DUMP AND THE LOAD ARE THE SAME FILE. The template must be the subjects THIS
# checkpoint was trained on, embedded by THIS checkpoint, on THIS normalisation. `--dump-src-metric`
# runs inside the fold that owns the encoder and writes the file before the eval block reads it,
# so channels/mvnn/feature-set cannot drift between producer and consumer. A separate builder
# script would re-derive all three from config and would silently diverge the first time a
# default changed.
#
# Set SUBJECTS/SEEDS to run a subset (the smoke test before the full grid).
set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
SEEDS="${SEEDS:-2025 2026 2027}"
# The mix grid. Includes 0 and 1 on purpose: both endpoints are FALSIFICATIONS under H1, so
# they have to be measured rather than assumed to be bad.
MIX="${MIX:-0,0.35,0.6,0.85,1.0}"

cd "${ROOT}"
mkdir -p outputs/slurm outputs/eval/v10_m1 outputs/scores/v10_m1 outputs/src_metric/v8

export PYTHONNOUSERSITE=1
export HF_HOME=/project/peilab/why/cache/huggingface
export HF_HUB_CACHE=/project/peilab/why/cache/huggingface/hub
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/open_clip
export TORCH_HOME=/project/peilab/why/cache/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
export TMPDIR="/tmp/clip-${SLURM_JOB_ID:-local}"      # node-local, NOT NFS (AGENTS.md 2.4)
export XDG_CONFIG_HOME="${ROOT}/scratch/config"
export MPLCONFIGDIR="${ROOT}/scratch/mpl"
mkdir -p "${TMPDIR}" "${XDG_CONFIG_HOME}" "${MPLCONFIGDIR}"
export PYTHONPATH="${ROOT}/src"

echo "[slurm] job=${SLURM_JOB_ID} host=$(hostname) date=$(date -Is)"
echo "[m1] subjects='${SUBJECTS}' seeds='${SEEDS}' mix='${MIX}'"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

for s in ${SUBJECTS}; do
  tag=$(printf 'sub%02d' "${s}")
  # sub-08 is the fold the SAMGA-style reference was measured on; every other fold gets
  # `unknown` so run_eval refuses to print a delta it cannot justify.
  if [ "${s}" -eq 8 ]; then REF="--ref-top1 22.00 --ref-top5 50.00"; else REF="--ref-top1 unknown"; fi
  for seed in ${SEEDS}; do
    ck="${ROOT}/outputs/stage1/v8/${tag}_k20_seed${seed}/last.pt"
    ev="${ROOT}/outputs/eval/v10_m1/${tag}_seed${seed}.json"
    sc="${ROOT}/outputs/scores/v10_m1/${tag}_seed${seed}.npz"
    sm="${ROOT}/outputs/src_metric/v8/${tag}_seed${seed}.npz"
    [ -s "${ck}" ] || { echo "  [skip] no ckpt ${tag}/seed${seed}"; continue; }
    if [ -s "${ev}" ] && [ -s "${sc}" ]; then echo "  [skip] done ${tag}/seed${seed}"; continue; fi
    echo "[m1] === ${tag}/seed${seed} ==="
    # `--reps` is what embeds the repetition cloud; `--fgw-sweep` is the pre-registered grid.
    # The mix enters as a LIST so the whole curve comes out of ONE embedding and is therefore
    # paired cell-by-cell at every R.
    "${PY}" scripts/run_eval.py --ckpts "${ck}" \
      --target-subject "${s}" --out "${ev}" \
      --dump-src-metric "${sm}" \
      --recovery --reps --fgw-sweep --csls-k 10 --rho 0.1 \
      --recovery-operator fgw --recovery-tau 0.01 --recovery-alpha 0.75 \
      --fgw-src-metric "${sm}" --fgw-src-mix "${MIX}" \
      --rep-subsample 80 40 20 10 5 1 \
      --save-scores "${sc}" ${REF} \
      && echo "  [ok] ${tag}/seed${seed}" || echo "  [FAIL] ${tag}/seed${seed}"
  done
done

echo "[m1] === summary ==="
"${PY}" scripts/summarize_v10.py \
  --sweep-dir outputs/eval/v10_m1 \
  --fidelity "R=80,a=0.75,t=0.03@outputs/eval/v8scr/a0.75_t0.03" \
             "R=80,a=0.75,t=0.01@outputs/eval/v8_fgw075" \
  --curve-alpha 0.75 --curve-tau 0.03 \
  --out outputs/v10_m1_summary.json || echo "[m1] summary reported an issue (see above)"

echo "[m1] DONE $(date -Is)"
