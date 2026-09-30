#!/usr/bin/env bash
# IP-mass diagnostic: WHY do PixCorr/SSIM collapse when we add depth/edge branches?
#
# The collapse is isolated by p1_i1 vs p2_i3: identical turbo / 5 steps /
# guidance 0 / no CN / no init / same prompts / seed 42. ONLY the number of
# IP-Adapter branches differs (1 vs 3), and PixCorr goes 0.132 -> 0.050.
#
# diffusers SUMS the attention output of each registered adapter, so 3 adapters
# at scale 1.0 inject ~3x the conditioning mass of one. These arms separate
# "total conditioning mass" from "the extra depth/edge modalities":
#
#   d_img1_b0.33   [1, 1, 1]        reproduce p2_i3 (control)
#   d_eqmass       [.333,.333,.333] SAME total mass as 1 branch at 1.0
#   d_halfmass     [.5, .5, .5]     total 1.5
#   d_imgonly      [1, 0, 0]        image branch only, 3 adapters registered
#   d_img1_tail    [1, .5, .5]      total 2.0
#   d_cn_eqmass    base + CN + 3 branches at equal-mass (structure path on)
set -euo pipefail
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" >/dev/null 2>&1 || true
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"

OUT="${DIAG_OUT:-${NB_ROOT}/outputs/mb_diag}"
PYTHON="$(command -v python)"
DEVICE="${DEVICE:-cuda:0}"
IP="${UCK_OUT:-${NB_ROOT}/outputs/uck}/sub-08/full/conds/ip_mem_test.npy"
D="${NB_ROOT}/outputs/mb_s08/heads/conds/depth_pred_test_cal.npy"
E="${NB_ROOT}/outputs/mb_s08/heads/conds/edge_pred_test_cal.npy"
PF="${NB_ROOT}/outputs/g2f/prompts/prompts_deploy.json"
DEPTH_RGB="${NB_ROOT}/outputs/uck/sub-08/full/spatial/pred_depth_rgb_512"
LL_RGB="${NB_ROOT}/outputs/sdedit_ll_full10/sub-08/vae_head/pred_lowlevel_rgb_512"
[[ -f "${LL_RGB}/199.png" ]] || LL_RGB="${NB_ROOT}/outputs/uck/sub-08/full/spatial/pred_lowlevel_rgb_512"

mkdir -p "${OUT}"/{logs,gen,eval}
log() { echo "[$(date -Iseconds)] $*"; }
require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1"; exit 1; }; }
for f in "${IP}" "${D}" "${E}" "${PF}"; do require "${f}"; done

run() { # tag pipeline scales use_cn
  local tag="$1" pipe="$2" scales="$3" cn_flag="${4:-0}"
  local gdir="${OUT}/gen/${tag}"
  if [[ ! -f "${gdir}/generated/199.png" ]]; then
    local cn=()
    if [[ "${cn_flag}" == "1" ]]; then
      cn=(--depth-rgb-dir "${DEPTH_RGB}" --lowlevel-rgb-dir "${LL_RGB}" \
          --cn-scale 0.40 --strength 0.82)
    fi
    log "===== gen ${tag} (${pipe}, scales=${scales}) ====="
    "${PYTHON}" scripts/nda/generate_multibranch_decode.py \
      --cond-npys "${IP},${D},${E}" --branch-scales "${scales}" \
      --prompts-json "${PF}" --output-dir "${gdir}" --tag "${tag}" \
      --pipeline "${pipe}" "${cn[@]}" \
      --gen-steps "$([[ "${pipe}" == turbo ]] && echo 5 || echo 28)" \
      --gen-guidance "$([[ "${pipe}" == turbo ]] && echo 0.0 || echo 5.0)" \
      --seed 42 >> "${OUT}/logs/gen_${tag}.log" 2>&1 \
      || { echo "[WARN] gen ${tag} failed"; return 1; }
  fi
  local ev="${OUT}/eval/${tag}.json"
  [[ -f "${ev}" ]] && { echo "[SKIP] score ${tag}"; return 0; }
  "${PYTHON}" scripts/nda/diag_quick_score.py \
    --gen-dir "${gdir}/generated" --tag "${tag}" --output-json "${ev}" \
    --device "${DEVICE}" >> "${OUT}/logs/score.log" 2>&1 || echo "[WARN] score ${tag} failed"
}

# turbo arms (no CN) — isolate conditioning mass
run d_t_1_1_1      turbo "1.0,1.0,1.0"
run d_t_eqmass     turbo "0.333,0.333,0.333"
run d_t_halfmass   turbo "0.5,0.5,0.5"
run d_t_imgonly    turbo "1.0,0.0,0.0"
run d_t_img1_tail  turbo "1.0,0.5,0.5"
run d_t_1br        turbo "1.0"

# base arm with CN at equal mass (structure path on)
run d_c_eqmass     base  "0.333,0.333,0.333" 1
run d_c_1_1_1      base  "1.0,1.0,1.0" 1

log "===== summary ====="
"${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
OUT = Path(os.environ.get("DIAG_OUT", "outputs/mb_diag"))
print(f"{'arm':<16}{'pix':>8}{'ssim':>8}{'clip2w':>8}{'layout':>9}{'hf':>10}{'div':>8}")
print("-"*68)
for p in sorted((OUT/"eval").glob("*.json")):
    d = json.loads(p.read_text())
    print(f"{d['tag']:<16}{d['pixcorr']:>8.4f}{d['ssim']:>8.4f}{d['clip_2way']:>8.3f}"
          f"{d['layout_corr']:>9.4f}{d['hf_energy']:>10.5f}{d['inter_div']:>8.4f}")
PY
log "done"
