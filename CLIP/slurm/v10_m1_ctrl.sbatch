#!/bin/bash
#SBATCH --job-name=v10-m1-ctrl
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/m1-ctrl-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/m1-ctrl-%j.err
set -euo pipefail
ROOT="/project/peilab/why/CLIP"; PY="${ROOT}/.venv/bin/python"; cd "${ROOT}"
export PYTHONNOUSERSITE=1 HF_HOME=/project/peilab/why/cache/huggingface
export HF_HUB_CACHE=/project/peilab/why/cache/huggingface/hub OPENCLIP_CACHE_DIR=/project/peilab/why/cache/open_clip
export TORCH_HOME=/project/peilab/why/cache/torch XDG_CACHE_HOME=/project/peilab/why/cache/xdg
export TMPDIR="/tmp/clip-${SLURM_JOB_ID:-local}"; mkdir -p "${TMPDIR}"
export XDG_CONFIG_HOME="${ROOT}/scratch/config" MPLCONFIGDIR="${ROOT}/scratch/mpl"
export PYTHONPATH="${ROOT}/src"; mkdir -p outputs/eval/v10_m1ctrl
"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" || exit 1
for s in 1 4 8; do
  tag=$(printf 'sub%02d' "$s")
  for seed in 2025; do
    ck="${ROOT}/outputs/stage1/v8/${tag}_k20_seed${seed}/last.pt"
    sm="${ROOT}/outputs/src_metric/v8/${tag}_seed${seed}.npz"
    ev="${ROOT}/outputs/eval/v10_m1ctrl/${tag}_seed${seed}.json"
    [ -s "${ck}" ] && [ -s "${sm}" ] || { echo "  [skip] $tag/$seed"; continue; }
    [ -s "${ev}" ] && { echo "  [skip] done $tag/$seed"; continue; }
    echo "[ctrl] === ${tag}/seed${seed} ==="
    "${PY}" scripts/run_eval.py --ckpts "${ck}" --target-subject "$s" --out "${ev}" \
      --recovery --reps --fgw-sweep --csls-k 10 --rho 0.1 \
      --recovery-operator fgw --recovery-tau 0.01 --recovery-alpha 0.75 \
      --fgw-src-metric "${sm}" --fgw-src-mix 0,1.0 --fgw-src-permute 7 \
      --ref-top1 unknown && echo "  [ok] $tag/$seed" || echo "  [FAIL] $tag/$seed"
  done
done
echo "[ctrl] DONE $(date -Is)"
