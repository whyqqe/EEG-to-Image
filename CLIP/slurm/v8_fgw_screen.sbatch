#!/bin/bash
#SBATCH --job-name=v8-fgw-screen
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/v8-fgw-screen-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/v8-fgw-screen-%j.err
#
# GPU screening of the FGW operator on the BANKED v8 encoder, saving SCORE MATRICES.
#
# Why GPU: 100+ evals at ~14s each on the login node is both slow and against this cluster's
# rule that batch work does not belong there.
#
# Why --save-scores: the headline lever turned out to be SEED ENSEMBLING (+4.28pp measured on
# a=0.5, 10/10 folds). Ensembling means averaging the three seeds' SCORE MATRICES; scalar
# reports cannot be re-pooled, so the matrices have to be written at eval time. A plain rung
# comparison (T1 vs T1+T2) is NOT the same experiment and understates the effect, because the
# seeds disagree on different concepts and only the matrix knows where.
set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
cd "${ROOT}"
mkdir -p outputs/slurm outputs/scores

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

# "alpha tau" -- 0.75/0.01 is the must-have (best single-seed so far); the rest probe whether the
# single-seed number can clear SCORE on its own, which is a cleaner claim than the ensemble.
for cfg in "0.75 0.01" "0.875 0.01" "0.625 0.01" "0.75 0.03"; do
  set -- ${cfg}
  A="$1"; T="$2"
  EV="${ROOT}/outputs/eval/v8scr/a${A}_t${T}"
  SC="${ROOT}/outputs/scores/v8scr_a${A}_t${T}"
  mkdir -p "${EV}" "${SC}"
  echo "[screen] === alpha=${A} tau=${T} ==="
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
        --recovery --reps --csls-k 10 --rho 0.1 \
        --recovery-operator fgw --recovery-alpha "${A}" --recovery-tau "${T}" \
        --save-scores "${sc}" ${REF} >/dev/null 2>&1 \
        && echo "  [ok] ${tag}/seed${seed}" || echo "  [FAIL] ${tag}/seed${seed}"
    done
  done
done

echo "[screen] === per-config summary: single-seed vs seed-ensembled ==="
for cfg in "0.75 0.01" "0.875 0.01" "0.625 0.01" "0.75 0.03" "0.5 0.01"; do
  set -- ${cfg}
  D="${ROOT}/outputs/scores/v8scr_a${1}_t${2}"
  [ -d "${D}" ] || continue
  echo "--- alpha=$1 tau=$2 ($(ls ${D}/*.npz 2>/dev/null | wc -l) score files) ---"
  "${PY}" "${ROOT}/scratch/ensemble.py" "${D}" "fused::+ T1(CSLS + recovery) + T2 reps" || true
  "${PY}" "${ROOT}/scratch/ensemble.py" "${D}" "row::+ T2 reps" || true
done

echo "[screen] SCREENING DONE $(date -Is)"
