#!/bin/bash
#SBATCH --job-name=v10-shrink
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/v10-shrink-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/v10-shrink-%j.err
#
# v10 STAGE 2 -- try to REMOVE the variance gate that stage 1 measured.
#
# Stage 1 (job 645433, outputs/v10_summary.json) found the structural gain is gated by the
# repetition count: +3.08pp at R=80 falling monotonically to -0.43pp at R=1, both temperatures,
# fidelity exact. Since both sides of every gain share the SAME whitening, the drop is a
# property of the target-side estimate the transport is built on -- not of the level, which
# collapses for a different reason (the rep-cloud whitening is fitted on C*R samples and is
# degenerate at R=1).
#
# The one free regulariser in that estimator is `shrink` in `_whiten_from_cloud`, left at the
# probes' 0.1 and never tuned for this. This job sweeps it.
#
# PRE-REGISTERED SUCCESS CONDITION (fixed before the run, so a lucky cell cannot be promoted
# after the fact): a single shrink value with
#       gain(R=20) >= gain(R=80) - 0.5pp        AND   gain(R=80) >= 3.08 - 0.5pp
# means the gate is an estimator artefact and the term works in a near-averaged-query regime --
# the version that can be compared to SCORE without a protocol caveat. No such value means the
# gate is a real data requirement and the paper quotes R with the headline.
set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
cd "${ROOT}"
mkdir -p outputs/slurm outputs/eval/v10_shrink outputs/scores/v10_shrink

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
nvidia-smi --query-gpu=name --format=csv,noheader || true
"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

# shrink=0.1 is the deployed value and is re-run first, into its own directory, so the sweep
# has its own reproduction of the stage-1 numbers rather than trusting the earlier job.
for S in 0.1 0.2 0.35 0.5 0.7; do
  EV="${ROOT}/outputs/eval/v10_shrink/s${S}"
  SC="${ROOT}/outputs/scores/v10_shrink/s${S}"
  mkdir -p "${EV}" "${SC}"
  echo "[shrink] ================= shrink=${S} ================="
  for s in 1 2 3 4 5 6 7 8 9 10; do
    tag=$(printf 'sub%02d' "${s}")
    if [ "${s}" -eq 8 ]; then REF="--ref-top1 22.00 --ref-top5 50.00"; else REF="--ref-top1 unknown"; fi
    for seed in 2025 2026 2027; do
      ck="${ROOT}/outputs/stage1/v8/${tag}_k20_seed${seed}/last.pt"
      ev="${EV}/${tag}_seed${seed}.json"
      sc="${SC}/${tag}_seed${seed}.npz"
      [ -s "${ck}" ] || { echo "  [skip] no ckpt ${tag}/seed${seed}"; continue; }
      if [ -s "${ev}" ] && [ -s "${sc}" ]; then echo "  [skip] done ${tag}/seed${seed}"; continue; fi
      "${PY}" scripts/run_eval.py --ckpts "${ck}" \
        --target-subject "${s}" --out "${ev}" \
        --recovery --reps --fgw-sweep --csls-k 10 --rho 0.1 --rep-shrink "${S}" \
        --recovery-operator fgw --recovery-tau 0.03 --recovery-alpha 0.75 \
        --save-scores "${sc}" ${REF} >/dev/null 2>&1 \
        && echo "  [ok] ${tag}/seed${seed}" || echo "  [FAIL] ${tag}/seed${seed}"
    done
  done
done

# --- summarise each shrink. Fidelity is only meaningful at shrink=0.1: every other value is
# --- SUPPOSED to differ from the banked (shrink=0.1) reference, so gating them on it would
# --- reject exactly the runs the experiment is about.
echo "[shrink] === per-shrink summaries ==="
for S in 0.1 0.2 0.35 0.5 0.7; do
  EV="${ROOT}/outputs/eval/v10_shrink/s${S}"
  OUT="${ROOT}/outputs/v10_shrink_summary_s${S}.json"
  echo "----- shrink=${S} -----"
  if [ "${S}" = "0.1" ]; then
    FID=(--fidelity "R=80,a=0.75,t=0.03@outputs/eval/v8scr/a0.75_t0.03"
                   "R=80,a=0.75,t=0.01@outputs/eval/v8_fgw075")
  else
    FID=(--fidelity)          # nargs="*" with no values -> no gate, reported as not evaluated
  fi
  "${PY}" scripts/summarize_v10.py --sweep-dir "${EV}" "${FID[@]}" \
    --curve-alpha 0.75 --curve-tau 0.03 --out "${OUT}" 2>&1 | \
    grep -E "FIDELITY|gain|^ +[0-9]+ +[0-9]|GATED|FLAT|INTERMEDIATE|NOT SWEPT|wrote" || true
done

echo "[shrink] DONE $(date -Is)"
