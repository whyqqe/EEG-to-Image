#!/usr/bin/env bash
# Preflight + CPU smoke for NeuroWeave Stage-1, then submit.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"

PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"

SMOKE="$(mktemp -d /tmp/nweave_smoke.XXXXXX)"
trap 'rm -rf "${SMOKE}"' EXIT

chk() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
say() { echo; echo "===== $* ====="; }

say "0/5 sources + assets"
chk scripts/nda/neuroweave_s1_train.py
chk scripts/nda/neuroweave_anytime_eval.py
chk scripts/nda/neuroweave_cycle_score.py
chk scripts/nda/neuroweave_summary.py
chk scripts/nda/run_neuroweave_s08.sh
chk slurm/nweave_s08.sbatch
chk scripts/nda/cfmsf_route_probe.py
chk scripts/nda/device_audit.py
chk outputs/ocf/intra_enc/sub-08/checkpoint_ss_calib_best.pth
chk outputs/leakfree/split.json
chk data/things_eeg/image_feature/ViT-H-14/image_train.npy
chk data/things_eeg/image_feature/ViT-H-14/GaussianBlur/train.npy
chk data/things_eeg/image_feature/ViT-H-14/LowResolution/train.npy

say "1/5 device audit"
"${PYTHON}" scripts/nda/device_audit.py \
  scripts/nda/neuroweave_s1_train.py \
  scripts/nda/neuroweave_anytime_eval.py

say "2/5 wiring: ARM_CFG + WIN + mask_time"
# Redirect to a file: interactive streaming of torch imports has hung the
# submit shell under the agent harness before (import itself is fine).
"${PYTHON}" - <<'PY' >"${SMOKE}/wiring.log" 2>&1
import sys
sys.path.insert(0, "scripts/nda")
from neuroweave_s1_train import ARM_CFG, WIN, mask_time, MultiHead
import torch
assert set(ARM_CFG) >= {"frozen","lora","direct","multi_head","anytime_train","causal_stage"}
assert WIN == {"early": 38, "mid": 88, "late": 175, "full": 250}
x = torch.randn(2, 17, 250)
y = mask_time(x, 38)
assert (y[..., 38:] == 0).all() and (y[..., :38] == x[..., :38]).all()
m = MultiHead()
o = m(torch.randn(4, 1024))
assert set(o) == {"early","mid","late"}
print("[ok] ARM_CFG / WIN / mask_time / MultiHead")
PY
cat "${SMOKE}/wiring.log"

say "3/5 LoRA device inheritance (meta)"
"${PYTHON}" - <<'PY'
import sys, torch, torch.nn as nn
sys.path.insert(0, "scripts/nda")
from cfmsf_fix_train import LoRALinear, inject_lora
lin = nn.Linear(8, 4)
# stay on CPU for smoke; check dtype/device inheritance
mod = LoRALinear(lin, rank=2, alpha=4)
assert mod.A.device == lin.weight.device and mod.B.device == lin.weight.device
y = mod(torch.randn(3, 8))
assert y.shape == (3, 4)
# zero-init B => identical to base at step 0
with torch.no_grad():
    assert torch.allclose(mod(torch.ones(2, 8)), lin(torch.ones(2, 8)))
print("[ok] LoRA zero-init + device inheritance")
PY

say "4/5 target bank shapes + concept disjointness (real names)"
"${PYTHON}" - <<'PY'
import sys, os, re
from pathlib import Path
sys.path.insert(0, "scripts/nda")
from neuroweave_s1_train import build_level_bank
from cfmsf_joint_train import build_target
banks = build_level_bank()
for k,(tr,te) in banks.items():
    assert tr.shape == (16540, 1024), (k, tr.shape)
    assert te.shape == (200, 1024), (k, te.shape)
tr, te, d = build_target("levels_mean")
assert d == 1024 and tr.shape[0] == 16540
# real concept-name disjointness (the vacuous .npy-string audit is NOT used)
root = Path("data/images_set") if Path("data/images_set").is_dir() else Path("/project/peilab/why/data/images_set")
def names(d):
    out=set()
    for x in os.listdir(d):
        m=re.match(r"^\d+_(.+)$", x)
        if m: out.add(m.group(1).lower())
    return out
a,b = names(root/"training_images"), names(root/"test_images")
assert len(a)==1654 and len(b)==200 and len(a&b)==0, (len(a),len(b),len(a&b))
print(f"[ok] banks + concept disjoint {len(a)}/{len(b)} intersection=0")
PY

say "5/5 orchestration parses + arm list present"
bash -n scripts/nda/run_neuroweave_s08.sh
bash -n slurm/nweave_s08.sbatch
grep -q causal_stage scripts/nda/run_neuroweave_s08.sh
grep -q anytime_train scripts/nda/run_neuroweave_s08.sh
grep -q neuroweave_summary scripts/nda/run_neuroweave_s08.sh
echo "[ok] orchestration"

say "SUBMIT"
mkdir -p outputs/slurm outputs/neuroweave/sub-08/logs
JOB=$(sbatch --parsable slurm/nweave_s08.sbatch)
echo "[submitted] job ${JOB}"
echo "${JOB}" > outputs/neuroweave/sub-08/job_id.txt
squeue -j "${JOB}" || true
