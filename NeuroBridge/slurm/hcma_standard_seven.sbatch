#!/usr/bin/env bash
#SBATCH --job-name=hcma-std7
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.err

set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh"
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
OUT="${NB_ROOT}/outputs/hcma_10subj"

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"

python "${NB_ROOT}/scripts/nda/eval_standard_seven_table.py" \
  --root "${OUT}" \
  --tag hcma_full_a40 \
  --device cuda:0 \
  --batch-size 16

echo "===== DONE standard-seven @ $(date -Iseconds) ====="
echo "Table: ${OUT}/STANDARD_SEVEN_TABLE.md"
