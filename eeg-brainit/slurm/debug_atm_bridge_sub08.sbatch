#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-atmbdbg
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1
python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
python - <<'PY'
import torch, sys
from pathlib import Path
sys.path.insert(0, 'src')
from eeg_brainit.utils.config import load_config
from eeg_brainit.models.atm_bridge import AtmBrainITPipeline
import copy

def deep_update(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out

cfg = deep_update(load_config('configs/base.yaml'), load_config('configs/debug_atm_bridge_sub08.yaml'))
m = AtmBrainITPipeline.from_config(cfg, project_root='.').cuda()
m.apply_stage(1)
x = torch.randn(4, 1024, device='cuda')
o = m(x)
assert 'clip_emb' in o and 'clip_tokens' in o
print('smoke ok', tuple(o['clip_emb'].shape), tuple(o['clip_tokens'].shape))
PY
python scripts/train_atm_bridge.py \
  --config configs/base.yaml \
  --override configs/debug_atm_bridge_sub08.yaml
echo "[OK] ATM bridge debug finished"
