#!/bin/bash
#SBATCH --job-name=nw4_guard
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_guard_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_guard_%j.err
# Prove the fail-fast works: on a node with a usable driver it should report the GPU;
# on a node whose driver is too old it must RAISE, not silently continue on CPU.
set -uo pipefail
cd /project/peilab/why/NeuroBridge
export HOME=/project/peilab/why/cache/xdg
P=/project/peilab/why/eeg-brainit/.venv/bin/python
echo "node: ${SLURMD_NODENAME:-?}"
$P -c "
import sys; sys.path.insert(0,'scripts/nda')
import torch, dev_guard
print('cuda_available =', torch.cuda.is_available())
try:
    d = dev_guard.pick_device('cuda:0')
    print('pick_device ->', d, '(GPU is usable here)')
except dev_guard.DeviceError as e:
    print('DeviceError raised (correct behaviour on a bad node):'); print(' ', str(e).splitlines()[0])
"
echo "--- and the real learner must refuse too ---"
$P scripts/nda/nw4_s1_train.py --out /tmp/guardtest --test-subject 8 --epochs 1 \
   --device cuda:0 --decode-rgb 0 2>&1 | tail -4
