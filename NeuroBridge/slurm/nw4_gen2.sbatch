#!/bin/bash
#SBATCH --job-name=nw4_gen2
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --time=03:00:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_gen2_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_gen2_%j.err
#
# Does the RSA we bought with SPA actually become images?
#
# The SPA sweep raised RSA monotonically and now clears the reference by a wide margin:
#
#     head     w_spa   raw RSA   calibrated RSA     (hybrid ip_uck = 0.2120)
#     spa0       0      0.1810      0.2414
#     spa1       1      0.2008      0.2476
#     spa5       5      0.2719      0.2884
#     spa20     20      0.3153      0.3037
#
# Two things are worth noting before spending GPU time.  First, spa0 -- the control that
# differs from the shipped head ONLY in the checkpoint selector (img_rsa instead of
# img_vs_cl) -- already sits at 0.2414 against the shipped head's 0.2085, so switching
# the selector is a free gain independent of the new loss.  Second, the oracle says the
# chain itself can reach inception 0.7946 from real CLIP embeddings, so there is genuine
# headroom above the hybrid reference's 0.6952-on-our-chain.
#
# What is NOT settled is the shape of the RSA -> inception map.  Fitting it on the
# hybrid family gives ~0.126 inception per unit RSA and predicts only ~0.71 here;
# fitting it on our own two points gives ~0.168 and predicts ~0.68.  Both are linear
# extrapolations across a range where the relation is almost certainly not linear, and
# the oracle adds a third constraint: RSA 1.0 -> 0.7946.  So the only way to know is to
# render.
#
# Arms (all single-branch all:0.9, hybrid init+depth, identical to x_hyb_1br so the
# comparison against its 0.6952 is exact):
#   s_spa0_cal   control: new selector, no SPA
#   s_spa5_cal   mid SPA
#   s_spa20_cal  max SPA
#   s_spa20_raw  is calibration still helping now that RSA is high?  the oracle says
#                calibration costs it 0.005, so the answer may have flipped
set -uo pipefail
cd /project/peilab/why/NeuroBridge

PYTHON=/project/peilab/why/eeg-brainit/.venv/bin/python
export OMP_NUM_THREADS=8
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
export HOME="${XDG_CACHE_HOME}"

OUT=outputs/nw4/gen2
PROMPTS=outputs/g2f/prompts/prompts_deploy.json
IMAGES_ROOT=/project/peilab/why/data/images_set
REF=outputs/gem/cond_cache/clip_img1024_train.npy
HYB_INIT=outputs/sdedit_ll_full10/sub-08/vae_head/pred_lowlevel_rgb_512
HYB_DEPTH=outputs/uck/sub-08/full/spatial/pred_depth_rgb_512

mkdir -p "${OUT}"/{conds,gen,eval,logs} outputs/slurm
for f in "${REF}" "${PROMPTS}" "${HYB_INIT}/199.png" "${HYB_DEPTH}/199.png"; do
  [[ -e "$f" ]] || { echo "[FATAL] missing $f"; exit 1; }
done

"${PYTHON}" - <<'PY'
import sys, numpy as np
from pathlib import Path
sys.path.insert(0,'scripts/nda')
from ocf_train import calibrate_quantile
R=np.load("outputs/gem/cond_cache/clip_img1024_train.npy").astype(np.float32)
out=Path("outputs/nw4/gen2/conds"); out.mkdir(parents=True,exist_ok=True)
for t in ("spa0","spa5","spa20"):
    p=Path(f"outputs/nw4/spa/{t}/s1/conds/z_img_test.npy")
    if not p.is_file():
        print(f"[FATAL] missing {p}"); raise SystemExit(1)
    z=np.load(p).astype(np.float32)
    y,meta=calibrate_quantile(z,R,two_sided=True)
    np.save(out/f"{t}_cal_test.npy", np.asarray(y,dtype=np.float32))
    if t=="spa20":
        # also keep the uncalibrated one
        np.save(out/"spa20_raw_test.npy", z)
    print(f"[cond] {t}: calibrated c_self {meta.get('c_self_before',float('nan')):.4f} -> "
          f"{meta.get('c_self_after',float('nan')):.4f}")
PY

echo
echo "===== RSA of the banks about to be rendered (must be >= 0.2120 to beat ref) ====="
"${PYTHON}" - <<'PY'
import sys, numpy as np
from pathlib import Path
sys.path.insert(0,'scripts/nda')
from nw4_s2_project import l2n, row2mean, retrieval
B_te=l2n(np.load('outputs/gem/cond_cache/clip_img1024_test.npy').astype(np.float32))
def rsa(z,t):
    z,t=l2n(z),l2n(t); cz,ct=z@z.T,t@t.T
    iu=np.triu_indices(len(cz),1); return float(np.corrcoef(cz[iu],ct[iu])[0,1])
h=l2n(np.load('outputs/hybrid_s08/conds/uck.npy').astype(np.float32))
print(f"{'bank':<20}{'RSA':>9}{'top1':>7}{'rowcos':>9}")
print(f"{'hybrid ip_uck (0.6952)':<20}{rsa(h,B_te):>9.4f}{retrieval(h,B_te)['top1']:>7.3f}{row2mean(h):>9.4f}")
for f in sorted(Path("outputs/nw4/gen2/conds").glob("*_test.npy")):
    z=l2n(np.load(f).astype(np.float32))
    print(f"{f.stem:<20}{rsa(z,B_te):>9.4f}{retrieval(z,B_te)['top1']:>7.3f}{row2mean(z):>9.4f}")
PY

gen_one(){
  local tag="$1" cond="$2"
  local g="${OUT}/gen/${tag}"
  [[ -f "${g}/generated/199.png" ]] && { echo "[SKIP] gen ${tag}"; return 0; }
  echo "===== gen ${tag} @ $(date -Iseconds) ====="
  "${PYTHON}" scripts/nda/generate_layered_decode.py \
    --cond-npys "${cond}" --branch-spec "all:0.9" \
    --prompts-json "${PROMPTS}" --output-dir "${g}" --tag "${tag}" \
    --pipeline base --depth-rgb-dir "${HYB_DEPTH}" --lowlevel-rgb-dir "${HYB_INIT}" \
    --cn-scale 0.40 --strength 0.82 --gen-steps 28 --gen-guidance 5.0 \
    --gen-size 512 --seed 42 --device cuda:0 \
    > "${OUT}/logs/gen_${tag}.log" 2>&1 || { echo "[WARN] gen ${tag} failed"; tail -n 15 "${OUT}/logs/gen_${tag}.log"; return 1; }
  return 0
}
ev(){
  local tag="$1"; local g="${OUT}/gen/${tag}/generated"; local j="${OUT}/eval/${tag}.json"
  [[ -f "${j}" ]] && { echo "[SKIP] eval ${tag}"; return 0; }
  [[ -f "${g}/199.png" ]] || { echo "[WARN] no images ${tag}"; return 1; }
  "${PYTHON}" scripts/nda/eval_official_seven_dir.py \
    --gen-dir "${g}" --output-json "${j}" --tag "${tag}" \
    --images-root "${IMAGES_ROOT}" --device cuda:0 \
    > "${OUT}/logs/eval_${tag}.log" 2>&1 || { echo "[WARN] eval ${tag} failed"; return 1; }
  return 0
}

C="${OUT}/conds"
gen_one s_spa0_cal  "${C}/spa0_cal_test.npy"  && ev s_spa0_cal
gen_one s_spa5_cal  "${C}/spa5_cal_test.npy"  && ev s_spa5_cal
gen_one s_spa20_cal "${C}/spa20_cal_test.npy" && ev s_spa20_cal
gen_one s_spa20_raw "${C}/spa20_raw_test.npy" && ev s_spa20_raw

echo
echo "==================== SPA -> image ===================="
"${PYTHON}" - <<'PY'
import json
from pathlib import Path
hdr=["inception","clip","pixcorr","ssim","fid"]
rows=[("hybrid published","outputs/hybrid_s08/eval/s08_hs_uck_uckF.json"),
      ("hybrid cond, our chain","outputs/nw4/cross/eval/x_hyb_1br.json"),
      ("nw4 img_cal (shipped)","outputs/nw4/calib/eval/g_img_cal_hybspace.json"),
      ("ORACLE ceiling","outputs/nw4/oracle/eval/o_clip_oracle.json"),
      ("spa0 + calib (selector only)","outputs/nw4/gen2/eval/s_spa0_cal.json"),
      ("spa5 + calib","outputs/nw4/gen2/eval/s_spa5_cal.json"),
      ("spa20 + calib","outputs/nw4/gen2/eval/s_spa20_cal.json"),
      ("spa20 raw (no calib)","outputs/nw4/gen2/eval/s_spa20_raw.json")]
print(f"{'run':<30}"+"".join(f"{h:>10}" for h in hdr))
for nm,p in rows:
    if not Path(p).is_file(): print(f"{nm:<30}  (pending)"); continue
    d=json.loads(Path(p).read_text())
    print(f"{nm:<30}"+"".join(f"{d.get(h,float('nan')):>10.4f}" for h in hdr))
bar=Path("outputs/hybrid_s08/eval/s08_hs_uck_uckF.json")
own=Path("outputs/nw4/cross/eval/x_hyb_1br.json")
print()
for nm,p in rows[4:]:
    f=Path(p)
    if not f.is_file(): continue
    d=json.loads(f.read_text())
    msg=[]
    if bar.is_file():
        b=json.loads(bar.read_text()); msg.append(f"dIncep vs published {d['inception']-b['inception']:+.4f}")
    if own.is_file():
        o=json.loads(own.read_text()); msg.append(f"vs our chain on hybrid cond {d['inception']-o['inception']:+.4f}")
    print(f"[{nm}] "+" | ".join(msg))
o=Path("outputs/nw4/oracle/eval/o_clip_oracle.json")
if o.is_file():
    od=json.loads(o.read_text())
    for nm,p in rows[4:]:
        f=Path(p)
        if not f.is_file(): continue
        d=json.loads(f.read_text())
        print(f"[{nm}] reaches {d['inception']/od['inception']*100:.1f}% of the oracle ceiling ({od['inception']:.4f})")
PY
echo "[nw4-gen2] done @ $(date -Iseconds)"
