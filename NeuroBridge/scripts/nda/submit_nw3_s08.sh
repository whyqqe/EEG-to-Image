#!/usr/bin/env bash
# Preflight + GPU submission for NeuroWeave v3 full sub-08 pipeline.
set -uo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"

OUT="${NW3_OUT:-${NB_ROOT}/outputs/nw3/sub-08}"
mkdir -p "${OUT}/logs" "${NB_ROOT}/outputs/slurm" "${OUT}/"{s1,s2,s3,gen,eval,cycle,v0,m7}

echo "=== [1] sources ==="
for f in scripts/nda/nw3_arms.py scripts/nda/nw3_s1_train.py scripts/nda/nw3_s2_prior.py \
         scripts/nda/nw3_s3_fuse.py scripts/nda/nw3_spatial_cycle.py scripts/nda/nw3_summary.py \
         scripts/nda/generate_struct_inject_decode.py scripts/nda/eval_official_seven_dir.py \
         scripts/nda/run_nw3_sub08.sh slurm/nw3_s08.sbatch \
         docs/NEUROWEAVE_V3_ARCHITECTURE.md docs/INVALID_ARTIFACTS.md; do
  [[ -f "$f" ]] || { echo "[FATAL] missing $f"; exit 1; }
  echo "  ok $f"
done
bash -n scripts/nda/run_nw3_sub08.sh || { echo "[FATAL] run syntax"; exit 1; }
bash -n slurm/nw3_s08.sbatch || { echo "[FATAL] sbatch syntax"; exit 1; }

echo "=== [2] assets ==="
for f in outputs/ocf/intra_z/sub-08/shared_r_train.npy \
         outputs/ocf/intra_z/sub-08/shared_r_test.npy \
         outputs/g2f/prompts/prompts_deploy.json \
         outputs/leakfree/split.json \
         outputs/sdedit_ll_full10/shared/vae_cache/train_vae_latents_f16.npy \
         outputs/sdedit_ll_full10/shared/vae_cache/test_vae_latents_f16.npy \
         outputs/uck/shared/gt_depth/train_depth_64.npy \
         outputs/uck/shared/g_img_concept.npy \
         outputs/nda_ss/sub-08/clip_text/train/text_concept_clip.npy \
         outputs/nda_ss/sub-08/clip_text/train/concept_phrases.json \
         outputs/g2/captions/captions_train.jsonl \
         outputs/g2/captions/captions_test.jsonl \
         outputs/gem/cond_cache/clip_img1024_train.npy \
         outputs/gem/cond_cache/clip_depth1024_train.npy \
         outputs/gem/cond_cache/clip_edge1024_train.npy \
         outputs/uck/sub-08/full/spatial/pred_lowlevel_rgb_512/199.png; do
  [[ -e "$f" ]] || { echo "[FATAL] missing $f"; exit 1; }
done
# generic prompt audit: zero own-class hits
"${PYTHON}" - <<'PY'
import json, numpy as np
from pathlib import Path
p=json.loads(Path('outputs/g2f/prompts/prompts_deploy.json').read_text())
meta=np.load('/project/peilab/why/data/images_set/image_metadata.npy',allow_pickle=True).item()
cs=list(meta['test_img_concepts'])
bare=[c.split('_',1)[1].replace('_',' ').lower() for c in cs]
hits=sum(1 for i,x in enumerate(p) if isinstance(x,str) and bare[i] in x.lower())
assert hits==0, hits
print(f"  prompts_deploy: {hits}/200 own-class (OK)")
PY

echo "=== [3] arms / kill criteria wiring ==="
"${PYTHON}" - <<'PY'
import sys
sys.path.insert(0,'scripts/nda')
from nw3_arms import V1_ARMS, V1_BAR, DOMINATE, passes_v1, dominates, INIT_CEILING
assert len(V1_ARMS)==6, len(V1_ARMS)
assert passes_v1({'pixcorr':0.22,'ssim':0.44})
assert not passes_v1({'pixcorr':0.10,'ssim':0.44})
d=dominates({'pixcorr':0.22,'ssim':0.44,'alex2':0.82,'alex5':0.92,'inception':0.84,'clip':0.91,'swav':0.48})
assert 'pixcorr' in d['wins']
print(f"  arms={list(V1_ARMS)} bar={V1_BAR} ceiling={INIT_CEILING}")
PY

echo "=== [4] CPU smoke: FactorizedEncoder forward + prior sample + fusion ==="
SMOKE_LOG="${OUT}/logs/smoke_cpu.log"
"${PYTHON}" - <<'PY' >"${SMOKE_LOG}" 2>&1
import sys, torch
sys.path.insert(0,'scripts/nda')
from nw3_s1_train import FactorizedEncoder, gallery_nce, l2t
from nw3_s2_prior import PriorDenoiser, sample, cosine_schedule
from nw3_s3_fuse import CrossModalFusion
import torch.nn.functional as F

m=FactorizedEncoder(z_dim=1024)
z=torch.randn(4,1024)
o=m(z)
assert o['z_sem'].shape==(4,1024)
assert o['vae'].shape==(4,4,64,64)
assert o['depth'].shape==(4,64,64)
assert o['z_hat'].shape==(4,1024)
g=torch.randn(10,1024); g=F.normalize(g,dim=-1)
cid=torch.randint(0,10,(4,))
loss=gallery_nce(o['z_sem'], g, cid, 0.07)
assert loss.ndim==0

p=PriorDenoiser()
zt=torch.randn(4,1024); t=torch.rand(4); cond=F.normalize(torch.randn(4,1024),dim=-1)
eps=p(zt,t,cond)
assert eps.shape==(4,1024)
a=cosine_schedule(t)
out=sample(p, cond, steps=3)
assert out.shape==(4,1024)

f=CrossModalFusion()
x=torch.randn(4,4,1024)
y=f(x)
assert y.shape==(4,1024)
print('SMOKE_OK')
PY
tail -3 "${SMOKE_LOG}"
grep -q SMOKE_OK "${SMOKE_LOG}" || { echo "[FATAL] smoke failed"; cat "${SMOKE_LOG}"; exit 1; }

echo "=== [5] generate_struct_inject flags used by run script exist ==="
"${PYTHON}" - <<'PY'
import ast, re
from pathlib import Path
src=Path('scripts/nda/generate_struct_inject_decode.py').read_text()
tree=ast.parse(src)
flags=set()
for n in ast.walk(tree):
    if isinstance(n, ast.Call) and getattr(n.func,'attr',None)=='add_argument':
        if n.args and isinstance(n.args[0], ast.Constant):
            flags.add(n.args[0].value)
need=['--mode','--embed-npy','--output-dir','--prompts-json','--control-type',
      '--cond-dir','--init-dir','--strength','--ip-scale','--cn-scale',
      '--control-guidance-start','--control-guidance-end','--skip-metrics']
miss=[f for f in need if f not in flags]
assert not miss, miss
print('  struct_inject flags OK')
# run script must reference prompts_deploy not hcma
run=Path('scripts/nda/run_nw3_sub08.sh').read_text()
assert 'prompts_deploy.json' in run
assert 'prompts_full_hcma' not in run
print('  run script uses generic prompts')
PY

echo "=== [6] duplicate job guard ==="
if squeue -u "$USER" -n nw3_s08 -h 2>/dev/null | grep -q .; then
  echo "[FATAL] nw3_s08 already queued/running:"
  squeue -u "$USER" -n nw3_s08
  exit 1
fi

echo "=== [7] submit ==="
JOB=$(sbatch --parsable slurm/nw3_s08.sbatch)
echo "SUBMITTED job_id=${JOB}"
echo "${JOB}" > "${OUT}/logs/job_id.txt"
echo "OUT=${OUT}"
echo "Monitor:  squeue -j ${JOB};  tail -f outputs/slurm/nw3_s08_${JOB}.out"
