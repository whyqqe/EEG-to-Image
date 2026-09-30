#!/bin/bash
#SBATCH --job-name=nw4_oracle
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --time=02:00:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_oracle_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_oracle_%j.err
#
# THE CEILING TEST.  Every structural axis we have now says our condition is at least
# as good as the ones that generate 0.70, yet it generates 0.662:
#
#     condition       RSA     kNN@5   kNN@50   inception
#     hybrid blend   0.1918  0.1220  0.4153    0.7053
#     hybrid uck     0.2120  0.1290  0.4194    0.6952
#     nw4 img_cal    0.2085  0.1390  0.4201    0.6620   <- best structure, worst image
#     nw4 raw img    0.1362  0.1320  0.4032    0.6228
#
# and nw4 raw img even beats hybrid uck on kNN@5 (0.132 vs 0.129) while sitting 0.07
# below it in inception.  A monotone function of these structure metrics cannot produce
# that ordering, so the residual is NOT in the condition's semantic content.  Two
# possibilities remain and this job separates them:
#
#   (a) the CONDITION is the limit  -> a perfect condition would still not do better,
#   (b) the CHAIN is the limit      -> init / depth / ControlNet / prompt cap what any
#                                      condition can reach, and our 0.662 is already
#                                      close to that cap.
#
# The way to tell them apart is to remove the encoder entirely and feed the GENERATOR
# the real thing.  `clip_img1024_test.npy` IS the ground-truth trial image embedding, so
# running it as a condition gives the ceiling of this whole pipeline: no condition can
# beat it, and whatever gap it leaves to 1.0 is chargeable to the chain.
#
#   o_clip_oracle      real CLIP test embeddings as the condition (no EEG at all)
#   o_clip_oracle_cal  the same, concentration-calibrated
#
# Reading the result:
#   o_clip_oracle ~0.90  -> the chain is fine; our 0.662 means RSA 0.2085 is a long way
#                           from RSA 1.0 and the encoder is where the loss lives
#   o_clip_oracle ~0.72  -> the chain caps everything near 0.72; our 0.662 is already
#                           ~92% of the reachable range and the encoder is NOT the
#                           bottleneck people (including the RSA fit) assumed
#   o_clip_oracle ~0.66  -> the chain itself is broken and no condition tuning matters
#
# The calibrated twin matters because calibration rescales every row's concentration,
# so if the oracle also drops when calibrated, calibration is destroying signal rather
# than repairing a defect -- which would mean the +0.039 we measured from it was a
# coincidence of the specific banks involved.

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

OUT=outputs/nw4/oracle
PROMPTS=outputs/g2f/prompts/prompts_deploy.json
IMAGES_ROOT=/project/peilab/why/data/images_set
BANK=outputs/gem/cond_cache/clip_img1024_test.npy
REF=outputs/gem/cond_cache/clip_img1024_train.npy
HYB_INIT=outputs/sdedit_ll_full10/sub-08/vae_head/pred_lowlevel_rgb_512
HYB_DEPTH=outputs/uck/sub-08/full/spatial/pred_depth_rgb_512

mkdir -p "${OUT}"/{conds,gen,eval,logs} outputs/slurm
for f in "${BANK}" "${REF}" "${PROMPTS}" "${HYB_INIT}/199.png" "${HYB_DEPTH}/199.png"; do
  [[ -e "$f" ]] || { echo "[FATAL] missing $f"; exit 1; }
done

"${PYTHON}" - <<'PY'
import sys, numpy as np
sys.path.insert(0,'scripts/nda')
from ocf_train import calibrate_quantile
B=np.load("outputs/gem/cond_cache/clip_img1024_test.npy").astype(np.float32)
R=np.load("outputs/gem/cond_cache/clip_img1024_train.npy").astype(np.float32)
np.save("outputs/nw4/oracle/conds/oracle_test.npy", B)
y,_=calibrate_quantile(B, R, two_sided=True)
np.save("outputs/nw4/oracle/conds/oracle_cal_test.npy", np.asarray(y,dtype=np.float32))
print("[oracle] wrote oracle_test.npy and oracle_cal_test.npy")
PY

echo "===== structural axes of the oracle itself ====="
"${PYTHON}" - <<'PY'
import sys, numpy as np
from pathlib import Path
sys.path.insert(0,'scripts/nda')
from nw4_s2_project import l2n, row2mean
from ocf_train import calibrate_quantile
B=l2n(np.load("outputs/gem/cond_cache/clip_img1024_test.npy").astype(np.float32))
R=np.load("outputs/gem/cond_cache/clip_img1024_train.npy").astype(np.float32)
def eff_rank(z):
    z=z-z.mean(0); s=np.linalg.svd(z,compute_uv=False)**2
    p=np.maximum(s/s.sum(),1e-12); return float(np.exp(-(p*np.log(p)).sum()))
def rsa(z,t):
    z,t=l2n(z),l2n(t); cz,ct=z@z.T,t@t.T
    iu=np.triu_indices(len(cz),1); return float(np.corrcoef(cz[iu],ct[iu])[0,1])
def knn(z,t,k):
    zn,tn=l2n(z),l2n(t)
    a=np.argsort(-(zn@zn.T),1)[:,1:k+1]; b=np.argsort(-(tn@tn.T),1)[:,1:k+1]
    return float(np.mean([len(set(x)&set(y))/k for x,y in zip(a,b)]))
for nm,z in [("oracle (raw CLIP)",B),("oracle + gem_calib",calibrate_quantile(B,R,two_sided=True)[0])]:
    z=l2n(np.asarray(z,dtype=np.float32))
    print(f"  {nm:<22} RSA={rsa(z,B):.4f} kNN@5={knn(z,B,5):.4f} eff_rank={eff_rank(z):6.1f} rowcos={row2mean(z):.4f}")
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
gen_one o_clip_oracle     "${C}/oracle_test.npy"     && ev o_clip_oracle
gen_one o_clip_oracle_cal "${C}/oracle_cal_test.npy" && ev o_clip_oracle_cal

echo
echo "==================== ceiling ===================="
"${PYTHON}" - <<'PY'
import json
from pathlib import Path
hdr=["inception","clip","pixcorr","ssim","fid"]
rows=[("hybrid published","outputs/hybrid_s08/eval/s08_hs_uck_uckF.json"),
      ("hybrid cond on our chain","outputs/nw4/cross/eval/x_hyb_1br.json"),
      ("nw4 img_cal (ours)","outputs/nw4/calib/eval/g_img_cal_hybspace.json"),
      ("ORACLE: real CLIP embeds","outputs/nw4/oracle/eval/o_clip_oracle.json"),
      ("ORACLE + calib","outputs/nw4/oracle/eval/o_clip_oracle_cal.json")]
print(f"{'run':<28}"+"".join(f"{h:>10}" for h in hdr))
for nm,p in rows:
    if not Path(p).is_file(): print(f"{nm:<28}  (pending)"); continue
    d=json.loads(Path(p).read_text())
    print(f"{nm:<28}"+"".join(f"{d.get(h,float('nan')):>10.4f}" for h in hdr))
o=Path("outputs/nw4/oracle/eval/o_clip_oracle.json")
m=Path("outputs/nw4/calib/eval/g_img_cal_hybspace.json")
if o.is_file() and m.is_file():
    od,md=json.loads(o.read_text()),json.loads(m.read_text())
    print()
    print(f"  ceiling (oracle)      inception = {od['inception']:.4f}")
    print(f"  ours                  inception = {md['inception']:.4f}")
    print(f"  unreached headroom    = {od['inception']-md['inception']:+.4f}")
    if od["inception"]>0.85:
        print("  -> the CHAIN is healthy. Our shortfall is the encoder: RSA 0.2085 vs 1.0")
        print("     means the condition carries a third of the structure it could, and")
        print("     that, not the wiring, is what caps us.")
    elif od["inception"]>0.70:
        print("  -> the CHAIN CAPS everything near 0.70. Our 0.662 is already most of the")
        print("     reachable range, so improving the condition cannot reach the published")
        print("     0.728 -- the init/depth/ControlNet path itself must be revisited.")
    else:
        print("  -> the CHAIN is broken: even a perfect condition fails, so no condition")
        print("     work (calibration, SPA, branch wiring) can matter until this is fixed.")
PY
echo "[nw4-oracle] done @ $(date -Iseconds)"
