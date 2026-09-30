#!/bin/bash
#SBATCH --job-name=nw4_calib
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --time=02:00:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_calib_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_calib_%j.err
#
# THE FIX.  `nw4_diag_rsa.py` found the axis that actually predicts generation:
#
#     corr(RSA, inception) = +0.967      RSA = corr(sim(cond_i,cond_j), sim(T_i,T_j))
#     corr(vs_cl, inception) = +0.687    (the gate used before -- too weak)
#
# Against that axis, nw4's condition was mis-concentrated rather than mis-directed:
#
#     variant              RSA     rowcos    top1
#     real CLIP bank       1.0000   0.6275   --
#     nw4 img raw          0.1362   0.9380   0.035   <- rows too similar to differ
#     nw4 img + gem_calib  0.2085   0.6123   0.215   <- matches the real bank
#     hybrid ip_uck calib  0.2120   0.6119   0.160   <- generated 0.7280
#
# `rowcos` 0.938 means every row points the same way, so the trial-to-trial structure
# is squashed and the adapter cannot tell trials apart.  `gem_calib.py` (quantile
# match each row's concentration onto the TRAIN bank's own distribution) restores it,
# and hybrid_s08 applied it while our pipeline did not -- that single omitted step is
# the whole 0.6228 vs 0.7280 gap we were trying to explain.
#
# A2 (the closed-form manifold projection) was aimed at the wrong axis: it changed the
# DIRECTION of each row (top-K neighbour combination), when what was wrong was the
# CONCENTRATION of the bank as a whole.
#
# Arms (all at hybrid's settings: strength 0.82, cn 0.40, steps 28, guidance 5.0):
#   g_img_cal_hybspace : calibrated img, hybrid init+depth -> direct test of 0.7280
#   g_img_cal_ourspace : calibrated img, OUR init+depth   -> isolated from 0.6228
#   g_4br_cal          : 4 branches each calibrated, our space -> multi-branch + calib
#   g_fused_cal        : calibrated dense mean, our space  -> the A3 control, fixed

set -uo pipefail
cd /project/peilab/why/NeuroBridge

PYTHON=/project/peilab/why/eeg-brainit/.venv/bin/python
export OMP_NUM_THREADS=8
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
export HOME="${XDG_CACHE_HOME}"

OUT=outputs/nw4/calib
PROMPTS=outputs/g2f/prompts/prompts_deploy.json
IMAGES_ROOT=/project/peilab/why/data/images_set
REF=outputs/gem/cond_cache/clip_img1024_train.npy
HYB_INIT=outputs/sdedit_ll_full10/sub-08/vae_head/pred_lowlevel_rgb_512
HYB_DEPTH=outputs/uck/sub-08/full/spatial/pred_depth_rgb_512
OUR_INIT=outputs/nw4/sub-08/s1/spatial/pred_lowlevel_rgb_512
OUR_DEPTH=outputs/nw4/sub-08/s1/spatial/pred_depth_rgb_512
S2=outputs/nw4/sub-08/s2/conds

mkdir -p "${OUT}"/{conds,gen,eval,logs} outputs/slurm
require(){ [[ -e "$1" ]] || { echo "[FATAL] missing $1"; exit 1; }; }
require "${REF}"; require "${OUR_INIT}/199.png"; require "${OUR_DEPTH}/199.png"
require "${S2}/img_test.npy"

# ------------------------------------------------------------------ calibrate
# Every S2 condition is already projected into the clip_img anchor space, so the same
# TRAIN clip_img bank is the correct reference for all of them.
echo "===== calibrate @ $(date -Iseconds) ====="
for b in img attr depth edge; do
  d="${OUT}/conds/${b}_cal.npy"
  [[ -f "${d}" ]] && { echo "[SKIP] calib ${b}"; continue; }
  "${PYTHON}" scripts/nda/gem_calib.py \
    --in "${S2}/${b}_test.npy" --out "${d}" --ref "${REF}" --tag "nw4_${b}" \
    >> "${OUT}/logs/calib.log" 2>&1 \
    || { echo "[WARN] calib ${b} failed; using raw"; cp "${S2}/${b}_test.npy" "${d}"; }
done
# the dense mean, calibrated (A3 control, now on equal footing)
"${PYTHON}" - <<'PY' || true
import sys, numpy as np
sys.path.insert(0,'scripts/nda')
from nw4_s2_project import l2n
from ocf_train import calibrate_quantile
S2="outputs/nw4/sub-08/s2/conds"; OUT="outputs/nw4/calib/conds"
z=[l2n(np.load(f"{S2}/{b}_test.npy").astype(np.float32)) for b in ("img","attr","depth","edge")]
fused=l2n(np.stack(z).mean(0).astype(np.float32))
ref=np.load("outputs/gem/cond_cache/clip_img1024_train.npy").astype(np.float32)
y,_=calibrate_quantile(fused, ref, two_sided=True)
np.save(f"{OUT}/fused_cal.npy", y.astype(np.float32))
print("[calib] fused_cal.npy written")
PY

# ------------------------------------------------------------------ gate report
echo
echo "----- RSA / vs_cl of every candidate before generating -----"
"${PYTHON}" - <<'PY'
import sys, numpy as np
from pathlib import Path
sys.path.insert(0,'scripts/nda')
from nw4_s2_project import l2n, row2mean, retrieval
B_tr=l2n(np.load('outputs/gem/cond_cache/clip_img1024_train.npy').astype(np.float32))
B_te=l2n(np.load('outputs/gem/cond_cache/clip_img1024_test.npy').astype(np.float32))
c=l2n(B_tr.mean(0,keepdims=True))[0]
floor=float((np.tile(c,(len(B_te),1))*B_te).sum(1).mean())
def rsa(z,t):
    cz=l2n(z)@l2n(z).T; ct=l2n(t)@l2n(t).T
    iu=np.triu_indices(len(cz),1)
    return float(np.corrcoef(cz[iu],ct[iu])[0,1])
print(f"{'candidate':<26}{'RSA':>9}{'vs_cl':>9}{'rowcos':>9}{'top1':>7}")
print(f"{'real CLIP bank (target)':<26}{1.0:>9.4f}{'--':>9}{row2mean(B_te):>9.4f}{'--':>7}")
print(f"{'hybrid ip_uck (0.7280)':<26}{rsa(l2n(np.load('outputs/hybrid_s08/conds/uck.npy').astype(np.float32)),B_te):>9.4f}"
      f"{float((l2n(np.load('outputs/hybrid_s08/conds/uck.npy').astype(np.float32))*B_te).sum(1).mean())-floor:>+9.4f}"
      f"{row2mean(l2n(np.load('outputs/hybrid_s08/conds/uck.npy').astype(np.float32))):>9.4f}"
      f"{retrieval(l2n(np.load('outputs/hybrid_s08/conds/uck.npy').astype(np.float32)),B_te)['top1']:>7.3f}")
for f in sorted(Path("outputs/nw4/calib/conds").glob("*_cal.npy")):
    z=l2n(np.load(f).astype(np.float32))
    print(f"{f.stem:<26}{rsa(z,B_te):>9.4f}{float((z*B_te).sum(1).mean())-floor:>+9.4f}"
          f"{row2mean(z):>9.4f}{retrieval(z,B_te)['top1']:>7.3f}")
PY

gen_one(){
  local tag="$1" conds="$2" spec="$3" init="$4" depth="$5"
  local g="${OUT}/gen/${tag}"
  if [[ -f "${g}/generated/199.png" ]]; then echo "[SKIP] gen ${tag}"; return 0; fi
  echo "===== gen ${tag} @ $(date -Iseconds) ====="
  "${PYTHON}" scripts/nda/generate_layered_decode.py \
    --cond-npys "${conds}" --branch-spec "${spec}" \
    --prompts-json "${PROMPTS}" --output-dir "${g}" --tag "${tag}" \
    --pipeline base --depth-rgb-dir "${depth}" --lowlevel-rgb-dir "${init}" \
    --cn-scale 0.40 --strength 0.82 --gen-steps 28 --gen-guidance 5.0 \
    --gen-size 512 --seed 42 --device cuda:0 \
    > "${OUT}/logs/gen_${tag}.log" 2>&1 || { echo "[WARN] gen ${tag} failed"; return 1; }
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
gen_one g_img_cal_hybspace "${C}/img_cal.npy" "all:0.9" "${HYB_INIT}" "${HYB_DEPTH}" && ev g_img_cal_hybspace
gen_one g_img_cal_ourspace "${C}/img_cal.npy" "all:0.9" "${OUR_INIT}" "${OUR_DEPTH}" && ev g_img_cal_ourspace
gen_one g_4br_cal "${C}/img_cal.npy,${C}/attr_cal.npy,${C}/depth_cal.npy,${C}/edge_cal.npy" \
        "all:0.9,all:0.9,all:0.9,all:0.9" "${OUR_INIT}" "${OUR_DEPTH}" && ev g_4br_cal
gen_one g_fused_cal "${C}/fused_cal.npy" "all:0.9" "${OUR_INIT}" "${OUR_DEPTH}" && ev g_fused_cal

echo
echo "==================== calib experiment result ===================="
"${PYTHON}" - <<'PY'
import json
from pathlib import Path
REF=[("hybrid ip_uck (the bar)","outputs/hybrid_s08/eval/s08_hs_uck_uckF.json"),
     ("nw4 w4_layered (no calib, our init)","outputs/nw4/sub-08/eval/w4_layered.json"),
     ("nw4 w4_layered_s40 (no calib)","outputs/nw4/sub-08/eval/w4_layered_s40.json"),
     ("V0 init blur (ceiling)","outputs/nw3/sub-08/v0/init_seven.json")]
hdr=["inception","clip","pixcorr","ssim","fid"]
print(f"{'run':<38}"+"".join(f"{h:>10}" for h in hdr))
for nm,p in REF:
    if Path(p).is_file():
        d=json.loads(Path(p).read_text())
        print(f"{nm:<38}"+"".join(f"{d.get(h,float('nan')):>10.4f}" for h in hdr))
print("-"*88)
for f in sorted(Path("outputs/nw4/calib/eval").glob("*.json")):
    d=json.loads(f.read_text())
    print(f"{f.stem:<38}"+"".join(f"{d.get(h,float('nan')):>10.4f}" for h in hdr))
print()
bar=json.loads(Path("outputs/hybrid_s08/eval/s08_hs_uck_uckF.json").read_text())
for f in sorted(Path("outputs/nw4/calib/eval").glob("*.json")):
    d=json.loads(f.read_text())
    di=d["inception"]-bar["inception"]
    print(f"[{f.stem}] vs hybrid bar: dIncep={di:+.4f} dClip={d['clip']-bar['clip']:+.4f} "
          f"dSSIM={d['ssim']-bar['ssim']:+.4f} -> {'BEATS' if di>0.005 else ('ties' if abs(di)<=0.005 else 'below')}")
PY
echo "[nw4-calib] done @ $(date -Iseconds)"
