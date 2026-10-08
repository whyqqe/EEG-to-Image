#!/bin/bash
#SBATCH --job-name=v10-sweep
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/v10-sweep-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/v10-sweep-%j.err
#
# v10 STAGE 1: THE VALIDATION SWEEP -- the checks that must pass BEFORE any v10 claim.
#
# Everything here runs on the BANKED v8 encoder (10 subjects x 3 seeds) with NO training, so
# it is a measurement of the deployed operator and not of a new model. Three questions, in the
# order the pipeline has to answer them (see scripts/summarize_v10.py for the reading rules):
#
#   1. FIDELITY   The sweep recomputes the deployed `+ T2 reps` row from scratch. Its
#                 (R=full, alpha=0.75, tau=0.01) cell must reproduce the banked Top-1, or the
#                 sweep is a different operator and nothing else here counts. Aborts on
#                 mismatch, per checkpoint.
#   2. GATING     gain(R) = Top-1(structural ON) - Top-1(structural OFF) on the IDENTICAL
#                 subset, R = 80 -> 1. Decides whether the +3.58pp headline is the structural
#                 term or the extra test information, i.e. whether the comparison to a
#                 repetition-free method is FAIR. This is the load-bearing check.
#   3. HONESTY    alpha/tau were picked on the same folds that get reported. The nested-LOSO
#                 loop re-picks them per held-out subject and reports that number instead.
#
# Why GPU at all: the only expensive step is embedding the R=80 repetition cloud (200 x 80
# trials) once per checkpoint. The whole grid is then matrix products on that one embedding,
# which is why 10 cells x 30 checkpoints is minutes and not hours.
set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
cd "${ROOT}"
mkdir -p outputs/slurm outputs/eval/v10_sweep outputs/scores/v10_sweep

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

# --- the sweep itself, resumable: a checkpoint with its JSON already on disk is skipped ---
for s in 1 2 3 4 5 6 7 8 9 10; do
  tag=$(printf 'sub%02d' "${s}")
  # sub-08 is the fold the SAMGA-style reference was measured on; every other fold gets
  # `unknown` so no fake delta is printed (run_eval refuses to invent a reference).
  if [ "${s}" -eq 8 ]; then REF="--ref-top1 22.00 --ref-top5 50.00"; else REF="--ref-top1 unknown"; fi
  for seed in 2025 2026 2027; do
    ck="${ROOT}/outputs/stage1/v8/${tag}_k20_seed${seed}/last.pt"
    ev="${ROOT}/outputs/eval/v10_sweep/${tag}_seed${seed}.json"
    sc="${ROOT}/outputs/scores/v10_sweep/${tag}_seed${seed}.npz"
    [ -s "${ck}" ] || { echo "  [skip] no ckpt ${tag}/seed${seed}"; continue; }
    if [ -s "${ev}" ] && [ -s "${sc}" ]; then echo "  [skip] done ${tag}/seed${seed}"; continue; fi
    echo "[sweep] === ${tag}/seed${seed} ==="
    "${PY}" scripts/run_eval.py --ckpts "${ck}" \
      --target-subject "${s}" --out "${ev}" \
      --recovery --reps --fgw-sweep --csls-k 10 --rho 0.1 \
      --recovery-operator fgw --recovery-tau 0.01 --recovery-alpha 0.75 \
      --save-scores "${sc}" ${REF} \
      && echo "  [ok] ${tag}/seed${seed}" || echo "  [FAIL] ${tag}/seed${seed}"
  done
done

# --- summarise: fidelity, gating curve, nested-LOSO. Fails loudly, never silently. ---
echo "[sweep] === summary ==="
# Two fidelity references, because the v10 headline is at tau=0.03 while `v8_fgw075` is the
# tau=0.01 duplicate. A single reference could not catch a tau mix-up, and a tau mix-up here
# would change the headline by ~1pp -- the same order as the effect being claimed.
"${PY}" scripts/summarize_v10.py \
  --sweep-dir outputs/eval/v10_sweep \
  --fidelity "R=80,a=0.75,t=0.03@outputs/eval/v8scr/a0.75_t0.03" \
             "R=80,a=0.75,t=0.01@outputs/eval/v8_fgw075" \
  --curve-alpha 0.75 --curve-tau 0.03 \
  --out outputs/v10_summary.json

echo "[sweep] DONE $(date -Is)"
