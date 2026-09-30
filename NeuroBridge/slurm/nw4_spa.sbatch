#!/bin/bash
#SBATCH --job-name=nw4_spa
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --time=04:00:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_spa_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_spa_%j.err
#
# Optimise the axis that predicts generation, instead of a proxy for it.
#
# `nw4_diag_rsa.py` established, over every condition with a known generation score,
#
#     corr(RSA,  inception) = +0.967      RSA = corr( sim(z_i,z_j), sim(T_i,T_j) )
#     corr(vs_cl, inception) = +0.687
#
# and calibration showed how much of RSA comes free of the encoder:
#
#     nw4 S1 head, raw         RSA 0.1362   rowcos 0.9380   (the old pipeline ran on this)
#     nw4 S1 head, calibrated  RSA 0.2085   rowcos 0.6123   -> predicted 0.7247
#     hybrid ip_uck, calib     RSA 0.2120   rowcos 0.6119   -> measured 0.7280
#
# So calibration is exhausted -- it can only undo a concentration defect, and it lands
# us level with the honest reference, not above it.  The remaining RSA is the encoder's:
# 0.2085 against a real bank's 1.000.  Nothing in the previous loss aimed at it.  InfoNCE
# is satisfied once the top-1 neighbour is right and is blind to whether i is *more*
# similar to j than to k; the anchor pins each row to its own target and says nothing
# about the geometry between rows; w_cpt collapses to concept means, the degenerate
# structure that sank the closed-set design.  `batch_rsa` makes the pairwise ordering
# the objective directly.
#
# Verified before running (CPU, in submit_nw4_sub08.sh style):
#   batch_rsa(t,t)          = -1.0000  (perfect structure)
#   batch_rsa(t, t[perm])   = -0.0266  (structure destroyed -> no reward)
#   batch_rsa(constant, t)  = -0.0000  (a constant cannot farm the term)
#   monotone in structure kept: 0.00 -> +0.0232, 0.50 -> -0.4932, 1.00 -> -1.0000
#   |grad| at realistic noise 0.1..1.0 = 0.14 .. 0.88   (not vanishing)
#   insensitive to batch size: B=16 -> 0.8916, B=1024 -> 0.9174 on the same construction
#
# Checkpoint selection moves to img_rsa for the same reason the objective did: the old
# selector maximised vs_cl, which is the weaker axis (and was measured on un-calibrated
# banks while the generator consumes calibrated ones).
#
# Weights: `batch_rsa` returns -corr in [-1, 1] and the anchor term w_anchor=400 sits at
# ~40 for a typical (1-cos)=0.1, so the sweep spans two orders of magnitude around the
# NCE terms (which are O(1)) to find where structure starts to bind.

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

SW=outputs/nw4/spa
mkdir -p "${SW}" outputs/slurm

run_s1(){
  local tag="$1" w_spa="$2"
  local d="${SW}/${tag}"
  if [[ -f "${d}/s1/best.pth" ]]; then echo "[SKIP] ${tag}"; return 0; fi
  echo "===== S1 ${tag}: w_spa=${w_spa} (select=img_rsa, w_anchor=400) @ $(date -Iseconds) ====="
  "${PYTHON}" scripts/nda/nw4_s1_train.py \
    --out "${d}/s1" \
    --device cuda:0 \
    --w-inst-img 1.0 --w-inst-vith 1.0 --w-inst-attr 1.0 \
    --w-anchor 400.0 --w-spa "${w_spa}" \
    --w-depth 1.0 --w-vae 1.0 --w-recon 0.2 \
    --select-metric img_rsa \
    > "${SW}/logs_${tag}.log" 2>&1 || { echo "[WARN] ${tag} failed"; tail -n 15 "${SW}/logs_${tag}.log"; return 1; }
  return 0
}

# w_spa=0 is the control: it must reproduce the previous head (RSA ~0.136 raw) so that
# any movement in the sweep is attributable to the new term and not to re-running.
for w in 0 1 5 20; do
  run_s1 "spa$w" "$w"
done

echo
echo "===== raw RSA of each S1 head (before calibration) ====="
"${PYTHON}" - <<'PY'
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0,'scripts/nda')
from nw4_s2_project import l2n, row2mean, retrieval
src=open("scripts/nda/nw4_s1_train.py").read()
exec(compile(src.split("def main(")[0], "m", "exec"), globals())
B_tr=l2n(np.load('outputs/gem/cond_cache/clip_img1024_train.npy').astype(np.float32))
B_te=l2n(np.load('outputs/gem/cond_cache/clip_img1024_test.npy').astype(np.float32))
c=l2n(B_tr.mean(0,keepdims=True))[0]
floor=float((np.tile(c,(len(B_te),1))*B_te).sum(1).mean())
rows=[]
print(f"{'head':<10}{'raw RSA':>10}{'raw vs_cl':>11}{'rowcos':>9}{'top1':>7}   {'after gem_calib':>16}")
for d in sorted(Path("outputs/nw4/spa").glob("spa*")):
    p=d/"s1"/"conds"/"z_img_test.npy"
    if not p.is_file():
        cands=list(d.rglob("z_img_test.npy"))
        if not cands: print(f"{d.name:<10}  (no z_img_test.npy)"); continue
        p=cands[0]
    z=l2n(np.load(p).astype(np.float32))
    r=np_rsa(z,B_te)
    cal,_=calibrate_quantile(z,B_tr,two_sided=True)
    cal=l2n(cal)
    rows.append({"head":d.name,"raw_rsa":round(r,4),"raw_vs_cl":round(float((z*B_te).sum(1).mean())-floor,4),
                 "rowcos":round(row2mean(z),4),"top1":retrieval(z,B_te)["top1"],
                 "cal_rsa":round(np_rsa(cal,B_te),4),
                 "pred_inception":round(0.464+1.25*max(r,np_rsa(cal,B_te)),4)})
    print(f"{d.name:<10}{r:>10.4f}{rows[-1]['raw_vs_cl']:>+11.4f}{row2mean(z):>9.4f}"
          f"{rows[-1]['top1']:>7.3f}   {rows[-1]['cal_rsa']:>16.4f}")
print()
print(f"reference: hybrid ip_uck calibrated RSA 0.2120 -> measured inception 0.7280")
if rows:
    b=max(rows,key=lambda r:max(r['raw_rsa'],r['cal_rsa']))
    best=max(b['raw_rsa'],b['cal_rsa'])
    print(f"best head: {b['head']} best RSA {best:.4f} -> predicted {0.464+1.25*best:.4f}")
    print(f"  -> {'ABOVE' if best>0.2120 else 'at or below'} the hybrid reference on RSA")
    ctrl=next((r for r in rows if r['head']=='spa0'),None)
    if ctrl:
        print(f"  control spa0 raw RSA {ctrl['raw_rsa']:.4f} (previous head measured 0.1362)")
Path("outputs/nw4/spa/spa_sweep.json").write_text(json.dumps({"rows":rows},indent=2))
PY
echo "[nw4-spa] done @ $(date -Iseconds)"
