#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-smoke
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=28
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
python scripts/smoke_test.py --config configs/base.yaml --stage 1
python -m pytest -q tests/test_shapes.py || python tests/test_shapes.py
