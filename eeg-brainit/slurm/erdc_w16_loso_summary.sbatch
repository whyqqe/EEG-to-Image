#!/usr/bin/env bash
#SBATCH --job-name=erdc-w16-sum
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Aggregate W16 per-subject metrics into paper appendix table

set -euo pipefail
cd /project/peilab/why/eeg-brainit
source scripts/activate.sh

python scripts/erdc_ten_subject_table.py \
  --metrics-dir outputs/erdc/w16_loso_metrics \
  --ours-prefix w16_loso_fuse \
  --official-prefix w16_official_prior \
  --output-md outputs/erdc/paper_ten_subject_table.md \
  --output-tex outputs/erdc/paper_ten_subject_table.tex

echo "[OK] ten-subject table done"
date
