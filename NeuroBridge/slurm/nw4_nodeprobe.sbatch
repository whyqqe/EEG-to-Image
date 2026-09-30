#!/bin/bash
#SBATCH --job-name=nw4_np
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/nw4/nodes/%x_%j.out
# One-shot probe: record whether this node's driver can actually run
# torch 2.13.0+cu130.  The cluster mixes driver versions, so a job that lands on an
# old one silently ran on the CPU until dev_guard.py made it raise instead.
set -uo pipefail
cd /project/peilab/why/NeuroBridge
export HOME=/project/peilab/why/cache/xdg
D=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
OK=$(/project/peilab/why/eeg-brainit/.venv/bin/python -c "
import torch; print('1' if torch.cuda.is_available() else '0')" 2>/dev/null | tail -1)
echo "node=${SLURMD_NODENAME:-?} driver=${D:-none} cuda_usable=${OK:-0}" \
  >> outputs/nw4/nodes/probe.txt
