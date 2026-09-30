#!/bin/bash
#SBATCH --job-name=nw4_cross
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --time=02:00:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_cross_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_cross_%j.err
#
# WHY THIS EXISTS
# --------------
# The full nw4 pipeline finished with a contradiction that decides the next move.
# At IDENTICAL generation settings (strength 0.82, cn 0.40, steps 28, guidance 5.0)
# our condition scores HIGHER on the metric that was supposed to predict everything:
#
#     condition      vs_cl     generation inception
#     hybrid ip_uck  +0.0367   0.7280
#     nw4 img        +0.0451   0.6228     <- better condition, worse image
#
# So either (a) vs_cl is not sufficient, (b) the 4-branch injection dilutes the good
# branch with the three bad ones (attr/depth/edge sit at vs_cl -0.026/-0.151/-0.198),
# or (c) something outside the condition differs -- the init images, the depth maps,
# the generator, or hybrid's `calib` step.
#
# All three are separated here by holding the SPATIAL PATHWAY FIXED to exactly what
# hybrid_s08 used and varying only the condition and the branch count:
#
#   init  = outputs/sdedit_ll_full10/sub-08/vae_head/pred_lowlevel_rgb_512  (hybrid's)
#   depth = outputs/uck/sub-08/full/spatial/pred_depth_rgb_512              (hybrid's)
#
# Arms:
#   x_hyb_1br     hybrid's calibrated ip_uck, single branch, all levels
#                 -> MUST reproduce ~0.728.  If it does not, the generator or the
#                    init/depth chain is the problem, not our condition.
#   x_nw4_1br     our img condition, single branch, all levels
#                 -> the head-to-head: our condition vs hybrid's, nothing else varied.
#   x_nw4_2br     img + attr (semantic pair) -> does adding a weak branch help or hurt?
#   x_nw4_4br     all four branches (== w4_flat but on hybrid's spatial pathway)
#                 -> isolates the multi-branch dilution factor.
#
# Reading the result:
#   x_hyb_1br ~0.728 and x_nw4_1br > 0.728  -> vs_cl holds, multi-branch was the drag
#   x_hyb_1br ~0.728 and x_nw4_1br < 0.728  -> vs_cl is NOT sufficient; the condition
#                                              needs a better target, not better wiring
#   x_hyb_1br << 0.728                       -> the discrepancy is NOT the condition

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

OUT=outputs/nw4/cross
PROMPTS=outputs/g2f/prompts/prompts_deploy.json
IMAGES_ROOT=/project/peilab/why/data/images_set
INIT=outputs/sdedit_ll_full10/sub-08/vae_head/pred_lowlevel_rgb_512
DEPTH=outputs/uck/sub-08/full/spatial/pred_depth_rgb_512

NW4_IMG=outputs/nw4/sub-08/s2/conds/img_test.npy
NW4_ATTR=outputs/nw4/sub-08/s2/conds/attr_test.npy
NW4_DEPTH=outputs/nw4/sub-08/s2/conds/depth_test.npy
NW4_EDGE=outputs/nw4/sub-08/s2/conds/edge_test.npy
HYB_UCK=outputs/hybrid_s08/conds/uck.npy

mkdir -p "${OUT}"/{gen,eval,logs} outputs/slurm
for f in "${PROMPTS}" "${INIT}/199.png" "${DEPTH}/199.png" "${NW4_IMG}" "${HYB_UCK}"; do
  [[ -e "$f" ]] || { echo "[FATAL] missing $f"; exit 1; }
done

gen_one() {
  local tag="$1" conds="$2" spec="$3"
  local g="${OUT}/gen/${tag}"
  if [[ -f "${g}/generated/199.png" ]]; then echo "[SKIP] gen ${tag}"; return 0; fi
  echo "===== gen ${tag}: conds=[${conds}] spec=[${spec}] @ $(date -Iseconds) ====="
  if ! "${PYTHON}" scripts/nda/generate_layered_decode.py \
      --cond-npys "${conds}" --branch-spec "${spec}" \
      --prompts-json "${PROMPTS}" --output-dir "${g}" --tag "${tag}" \
      --pipeline base \
      --depth-rgb-dir "${DEPTH}" --lowlevel-rgb-dir "${INIT}" \
      --cn-scale 0.40 --strength 0.82 --gen-steps 28 --gen-guidance 5.0 \
      --gen-size 512 --seed 42 --device cuda:0 \
      > "${OUT}/logs/gen_${tag}.log" 2>&1; then
    echo "[WARN] gen ${tag} failed"; tail -n 20 "${OUT}/logs/gen_${tag}.log"; return 1
  fi
  return 0
}

ev() {
  local tag="$1"
  local g="${OUT}/gen/${tag}/generated"
  local j="${OUT}/eval/${tag}.json"
  [[ -f "${j}" ]] && { echo "[SKIP] eval ${tag}"; return 0; }
  [[ -f "${g}/199.png" ]] || { echo "[WARN] no images for ${tag}"; return 1; }
  "${PYTHON}" scripts/nda/eval_official_seven_dir.py \
    --gen-dir "${g}" --output-json "${j}" --tag "${tag}" \
    --images-root "${IMAGES_ROOT}" --device cuda:0 \
    > "${OUT}/logs/eval_${tag}.log" 2>&1 || { echo "[WARN] eval ${tag} failed"; return 1; }
  return 0
}

gen_one x_hyb_1br  "${HYB_UCK}"  "all:0.9"                          && ev x_hyb_1br
gen_one x_nw4_1br  "${NW4_IMG}"  "all:0.9"                          && ev x_nw4_1br
gen_one x_nw4_2br  "${NW4_IMG},${NW4_ATTR}"                "all:0.9,all:0.9"           && ev x_nw4_2br
gen_one x_nw4_4br  "${NW4_IMG},${NW4_ATTR},${NW4_DEPTH},${NW4_EDGE}" "all:0.9,all:0.9,all:0.9,all:0.9" && ev x_nw4_4br

echo
echo "==================== cross-experiment result ===================="
"${PYTHON}" - <<'PY'
import json
from pathlib import Path
ROOT = Path("outputs/nw4/cross/eval")
REF = {"hybrid ip_uck (published bar)": "outputs/hybrid_s08/eval/s08_hs_uck_uckF.json",
       "nw4 w4_layered (4br, layered, our init)": "outputs/nw4/sub-08/eval/w4_layered.json",
       "nw4 w4_flat (4br, all levels, our init)": "outputs/nw4/sub-08/eval/w4_flat.json"}
hdr = ["inception", "clip", "pixcorr", "ssim", "fid"]
print(f"{'run':<42}" + "".join(f"{h:>10}" for h in hdr))
for nm, p in REF.items():
    d = json.loads(Path(p).read_text())
    print(f"{nm:<42}" + "".join(f"{d.get(h, float('nan')):>10.4f}" for h in hdr))
print("-" * 92)
for f in sorted(ROOT.glob("*.json")):
    d = json.loads(f.read_text())
    print(f"{f.stem:<42}" + "".join(f"{d.get(h, float('nan')):>10.4f}" for h in hdr))
print()
res = {f.stem: json.loads(f.read_text()) for f in ROOT.glob("*.json")}
hyp = json.loads(Path(REF["hybrid ip_uck (published bar)"]).read_text())
if "x_hyb_1br" in res:
    d = res["x_hyb_1br"]["inception"] - hyp["inception"]
    print(f"[generator check] x_hyb_1br vs the published hybrid row: dIncep={d:+.4f}")
    if abs(d) <= 0.02:
        print("   -> our generator+init+depth chain REPRODUCES the published bar, so a")
        print("      shortfall in the nw4 arms is attributable to the CONDITION")
    else:
        print("   -> the chain does NOT reproduce it; the discrepancy is outside the")
        print("      condition (generator, init, depth, or hybrid's calib step)")
if "x_nw4_1br" in res and "x_hyb_1br" in res:
    d = res["x_nw4_1br"]["inception"] - res["x_hyb_1br"]["inception"]
    print(f"[head-to-head] x_nw4_1br - x_hyb_1br: dIncep={d:+.4f}")
    print(f"   -> our condition {'WINS' if d > 0.01 else ('ties' if abs(d) <= 0.01 else 'LOSES')}"
          f" against hybrid's at identical settings")
if "x_nw4_4br" in res and "x_nw4_1br" in res:
    d = res["x_nw4_4br"]["inception"] - res["x_nw4_1br"]["inception"]
    print(f"[multi-branch dilution] x_nw4_4br - x_nw4_1br: dIncep={d:+.4f}")
    print(f"   -> adding the 3 weak branches {'HURTS' if d < -0.01 else ('helps' if d > 0.01 else 'is neutral')}")
PY
echo "[nw4-cross] done @ $(date -Iseconds)"
