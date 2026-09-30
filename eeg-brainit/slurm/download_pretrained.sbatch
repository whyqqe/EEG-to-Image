#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-pretrain-dl
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Safe download: only writes under eeg-brainit/checkpoints and cache/

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
module load slurm
cd "${ROOT}"
source scripts/activate.sh
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export PYTHONUNBUFFERED=1
python scripts/download_pretrained.py
echo "[OK] pretrained download finished"
