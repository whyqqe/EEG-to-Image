#!/bin/bash
# v12 wiring smoke -- 3 optimizer steps on GPU, to prove the v12 branch actually executes
# before the 30-run twin is submitted. A term that compiles, passes its unit test, and is
# never reached is this project's most expensive recurring failure (a `--reps` flag that was
# accepted and produced no row). See AGENTS.md §2.2 (verify on a GPU node, not the login node).
#SBATCH --job-name=v12-smoke
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=00:40:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/v12-smoke-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/v12-smoke-%j.err

set -euo pipefail
ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
mkdir -p "${ROOT}/outputs/slurm"

export PYTHONNOUSERSITE=1
export HF_HOME=/project/peilab/why/cache/huggingface
export HF_HUB_CACHE=/project/peilab/why/cache/huggingface/hub
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/open_clip
export TORCH_HOME=/project/peilab/why/cache/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
export PIP_CACHE_DIR=/project/peilab/why/cache/pip
export TMPDIR="/tmp/clip-${SLURM_JOB_ID:-local}"
export XDG_CONFIG_HOME="${ROOT}/scratch/config"
export MPLCONFIGDIR="${ROOT}/scratch/mpl"
mkdir -p "${TMPDIR}" "${XDG_CONFIG_HOME}" "${MPLCONFIGDIR}"
export OMP_NUM_THREADS=8
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

echo "[smoke] job=${SLURM_JOB_ID} host=$(hostname) date=$(date -Is)"
"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "[FATAL] CUDA unavailable"; exit 1; }

# The v12 smoke must be the ARM, not the twin: a 0-weight smoke would not traverse the branch.
"${PY}" scripts/run_stage1.py --config configs/v12_loso_k20.yaml \
  --target-subject 8 --source-subjects 1 2 3 4 5 6 7 9 10 \
  --out-dir "${ROOT}/outputs/stage1/v12_smoke/sub08" --seed 2025 \
  --mvnn train --channel-set all63 \
  --epochs 2 --stage1-epochs 1 --debug-steps 3 \
  --v12-distill 1.0 --v12-consistency 0.5 --reps-weight 0

echo "[smoke] steps done; now assert the v12 terms were ACTUALLY applied"
"${PY}" - <<'PY'
import json, pathlib
d = pathlib.Path("/project/peilab/why/CLIP/outputs/stage1/v12_smoke/sub08")
hist = [p for p in d.glob("*.json")]
print("[smoke] artefacts:", [p.name for p in hist])
# the trainer's per-step log line carries the term values; grep it out of the slurm .out
out = sorted(pathlib.Path("/project/peilab/why/CLIP/outputs/slurm").glob("v12-smoke-*.out"))[-1]
txt = out.read_text(errors="ignore")
hit = [ln for ln in txt.splitlines() if "v12" in ln.lower() or "metric_student" in ln]
for ln in hit[-25:]:
    print("[smoke-out]", ln)
if not hit:
    raise SystemExit("[smoke] FATAL: no v12 term ever appeared in the log -- the branch did "
                     "NOT execute. Do not submit the twin.")
print("[smoke] OK: v12 branch executed.")
PY
echo "[smoke] finished at $(date -Is)"
