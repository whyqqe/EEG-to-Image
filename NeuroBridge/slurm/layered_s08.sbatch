#!/bin/bash
#SBATCH --job-name=layered_s08
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/layered_s08_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/layered_s08_%j.err
#
# TIME BUDGET (measured on mb_p3_i3_cn, same pipeline/steps/seed)
#   * plan resolution, 3 arms                 <1 min
#   * layer sanity, 3 arms x 1 load x 6 renders ~10 min  (pipeline load dominates)
#   * generation, 3 arms x 200 img @ 28 steps  ~21 min   (mb_p3_i3_cn: 6.5 min/200)
#   * seven-metric eval, 3 arms                ~15 min
#   * cycle, 3 spaces x 5 sets x 200 img        ~8 min   (round 2: 3 sets ~6 min)
#   => ~55 min.  4 h wall leaves room for a slow node, a pipeline reload and resume.

set -euo pipefail
cd /project/peilab/why/NeuroBridge
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
export LAYERED_OUT=/project/peilab/why/NeuroBridge/outputs/layered/sub-08
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
mkdir -p outputs/slurm "${LAYERED_OUT}/logs"

echo "[slurm] layer-wise injection ${SLURM_JOB_ID:-local} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

set +e
bash scripts/nda/run_layered_sub08.sh
rc=$?
set -e
echo "[slurm] layered rc=${rc} @ $(date -Iseconds)"
echo "[slurm] Re-submit resumes: existing gen/eval/cycle artifacts are skipped."
exit "${rc}"
