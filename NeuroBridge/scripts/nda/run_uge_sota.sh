#!/usr/bin/env bash
# ============================================================================
# UGE-SOTA / sub-08 -- no retraining.  Swap the already-working leak-free IP
# into the two decoders this project already uses, and score them with the
# SAME official-seven script as UGE.
#
# Why this, and not another UGE train: 571745 measured that regressing
# encode_image() from intra EEG lands at clip-cos 0.39, BELOW the image-mean
# centreline 0.63.  The generation CLIP of that head (0.670) is beaten by the
# language-ridge arm (0.703) and by existing g2f/g3f rows (0.779).  The SOTA
# gap is therefore an ASSEMBLY gap:
#   semantic  = g2f intra IP (mem / fused), already in CLIP-image space
#   structure = ATM SDEdit (UGE VAE) or HCMA-S (Depth-CN x LL RGB)
#   prompt    = deploy generic / UGE 5-word / empty   (NEVER test class names)
# ============================================================================
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/uge_sota/sub-08}"
COND="${NB_ROOT}/outputs/gem/cond_cache"
G2F_INTRA="${NB_ROOT}/outputs/g2f/intra_sub-08/conds"
UGE="${NB_ROOT}/outputs/uge/sub-08"
HS="${NB_ROOT}/outputs/hcma_s/sub-08"
LL_RGB="${NB_ROOT}/outputs/lowlevel_decoder/sub-08/vae_head/pred_lowlevel_rgb_512"
DEPTH_RGB="${HS}/depth/pred_depth_rgb_512"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
SID="${SID:-8}"
SD="$(printf '%02d' "${SID}")"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"

SD_STRENGTH="${SD_STRENGTH:-0.82}"
GEN_STEPS="${GEN_STEPS:-28}"
GEN_GUIDANCE="${GEN_GUIDANCE:-5.0}"
IP_SCALE="${IP_SCALE:-1.0}"
HS_CN="${HS_CN:-0.40}"
HS_STRENGTH="${HS_STRENGTH:-0.82}"

export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
mkdir -p "${OUT}/logs" "${OUT}/eval" "${OUT}/conds" "${OUT}/prompts" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" >/dev/null 2>&1 || true
PYTHON="$(command -v python)"
export OUT
echo "[env] PYTHON=${PYTHON} out=${OUT}"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }

require "${G2F_INTRA}/ip_mem_test.npy"
require "${G2F_INTRA}/ip_fused_test.npy"
require "${UGE}/full/pred_vae_test_scaled.npy"
require "${UGE}/full/prompts/prompts_self.json"
require "${UGE}/full/prompts/prompts_generic.json"
require "${NB_ROOT}/outputs/g2f/prompts/prompts_deploy.json"
require "${COND}/clip_img1024_train.npy"
require "${DEPTH_RGB}/199.png"
require "${LL_RGB}/199.png"
require scripts/nda/generate_atm_aligned_decode.py
require scripts/nda/generate_hcma_s_decode.py
require scripts/nda/gem_calib.py
require scripts/nda/eval_official_seven_dir.py

V="${UGE}/full/pred_vae_test_scaled.npy"
P5="${UGE}/full/prompts/prompts_self.json"
PGEN="${UGE}/full/prompts/prompts_generic.json"
PDEP="${NB_ROOT}/outputs/g2f/prompts/prompts_deploy.json"
PEMPTY="${OUT}/prompts/prompts_empty.json"
if [[ ! -f "${PEMPTY}" ]]; then
  "${PYTHON}" - <<PY
import json
from pathlib import Path
Path("${PEMPTY}").write_text(json.dumps([""] * 200, indent=1), encoding="utf-8")
PY
fi

echo "===== [1] calibrate g2f IPs onto the TRAIN CLIP-image bank ====="
cal_one() {
  local src="$1" tag="$2"
  local o="${OUT}/conds/${tag}.npy"
  if [[ ! -f "${o}" ]]; then
    "${PYTHON}" scripts/nda/gem_calib.py \
      --in "${src}" --out "${o}" \
      --ref "${COND}/clip_img1024_train.npy" --tag "${tag}" \
      >> "${OUT}/logs/calib.log" 2>&1 \
      || { echo "[WARN] calib ${tag} failed; using raw" >&2; cp -f "${src}" "${o}"; }
  fi
  printf '%s\n' "${o}"
}

# de-mean in the condition's own space.  UGE measure.json: stripping the
# row-constant raised clip row-identity 0.04 -> 0.20.  Applied AFTER
# calibration so the quantile match is not undone.
center_one() {
  local src="$1" tag="$2"
  local o="${OUT}/conds/${tag}.npy"
  if [[ ! -f "${o}" ]]; then
    "${PYTHON}" - <<PY
import numpy as np
x = np.load("${src}").astype(np.float32)
n = np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)
z = x / n
mu = z.mean(0, keepdims=True)
mu = mu / np.clip(np.linalg.norm(mu), 1e-8, None)
r = z - (z * mu).sum(1, keepdims=True) * mu
r = r / np.clip(np.linalg.norm(r, axis=-1, keepdims=True), 1e-8, None)
np.save("${o}", r.astype(np.float32))
print(f"[center] ${tag}: wrote {r.shape}")
PY
  fi
  printf '%s\n' "${o}"
}

MEM="$(cal_one "${G2F_INTRA}/ip_mem_test.npy" mem)"
FUS="$(cal_one "${G2F_INTRA}/ip_fused_test.npy" fused)"
MEM_C="$(center_one "${MEM}" mem_center)"

eval_row() {
  local tag="$1" gdir="$2"
  local ev="${OUT}/eval/s${SD}_${tag}.json"
  [[ -f "${ev}" ]] && { echo "[SKIP] eval ${tag}"; return 0; }
  [[ -f "${gdir}/199.png" ]] || { echo "[WARN] eval ${tag}: no images"; return 0; }
  echo "--- eval ${tag}"
  "${PYTHON}" scripts/nda/eval_official_seven_dir.py \
    --gen-dir "${gdir}" --output-json "${ev}" --tag "${tag}" \
    --images-root "${IMAGES_ROOT}" --device "${DEVICE}" --skip-if-exists \
    2>&1 | tee -a "${OUT}/logs/eval.log" \
    || { echo "[WARN] eval ${tag} failed"; return 0; }
  if [[ -s "${ev}" && "${KEEP_IMAGES:-0}" != "1" ]]; then
    rm -rf "${gdir}"
  fi
}

gen_atm() {   # tag cond prompts|none
  local tag="$1" cond="$2" pf="$3"
  local gdir="${OUT}/gen/${tag}"
  local ev="${OUT}/eval/s${SD}_${tag}.json"
  [[ -f "${ev}" ]] && { echo "[SKIP] ${tag}"; return 0; }
  require "${cond}"
  local pflag=(--prompts-json "")
  if [[ "${pf}" != "none" && -n "${pf}" ]]; then
    require "${pf}"
    pflag=(--prompts-json "${pf}")
  fi
  echo "===== [2] ATM ${tag} @ $(date -Iseconds) ====="
  "${PYTHON}" scripts/nda/generate_atm_aligned_decode.py \
    --mode sdedit --embed-npy "${cond}" "${pflag[@]}" \
    --vae-latent-npy "${V}" \
    --output-dir "${gdir}" --tag "${tag}" \
    --strength "${SD_STRENGTH}" --ip-scale "${IP_SCALE}" \
    --gen-steps "${GEN_STEPS}" --gen-guidance "${GEN_GUIDANCE}" \
    --device "${DEVICE}" 2>&1 | tee "${OUT}/logs/gen_${tag}.log" \
    || { echo "[WARN] gen ${tag} failed"; return 0; }
  eval_row "${tag}" "${gdir}/generated"
}

gen_hs() {    # tag cond prompts
  local tag="$1" cond="$2" pf="$3"
  local gdir="${OUT}/gen/${tag}"
  local ev="${OUT}/eval/s${SD}_${tag}.json"
  [[ -f "${ev}" ]] && { echo "[SKIP] ${tag}"; return 0; }
  require "${cond}"; require "${pf}"
  echo "===== [3] HCMA-S ${tag} cn=${HS_CN} s=${HS_STRENGTH} @ $(date -Iseconds) ====="
  "${PYTHON}" scripts/nda/generate_hcma_s_decode.py \
    --embed-npy "${cond}" --prompts-json "${pf}" \
    --depth-rgb-dir "${DEPTH_RGB}" --lowlevel-rgb-dir "${LL_RGB}" \
    --output-dir "${gdir}" --tag "${tag}" \
    --cn-scale "${HS_CN}" --strength "${HS_STRENGTH}" --ip-scale "${IP_SCALE}" \
    --gen-steps "${GEN_STEPS}" --gen-guidance "${GEN_GUIDANCE}" \
    2>&1 | tee "${OUT}/logs/gen_${tag}.log" \
    || { echo "[WARN] gen ${tag} failed"; return 0; }
  eval_row "${tag}" "${gdir}/generated"
}

# ---- ATM: same decoder as UGE, so the only variable is the IP / prompt.
gen_atm atm_mem_deploy   "${MEM}"   "${PDEP}"
gen_atm atm_mem_none     "${MEM}"   none
gen_atm atm_mem_uge5     "${MEM}"   "${P5}"
gen_atm atm_fused_deploy "${FUS}"   "${PDEP}"
gen_atm atm_fused_none   "${FUS}"   none
gen_atm atm_memc_deploy  "${MEM_C}" "${PDEP}"

# ---- HCMA-S: the structure recipe that already posted PixCorr 0.19 on this
# subject, now driven by leak-free g2f IP + deploy/UGE prompts (not GT class names).
gen_hs hs_mem_deploy   "${MEM}"   "${PDEP}"
gen_hs hs_fused_deploy "${FUS}"   "${PDEP}"
gen_hs hs_mem_uge5     "${MEM}"   "${P5}"
gen_hs hs_mem_none     "${MEM}"   "${PEMPTY}"

echo "===== [4] summary vs published bars @ $(date -Iseconds) ====="
"${PYTHON}" - <<'PY'
import json
from pathlib import Path
OUT = Path(__import__("os").environ.get("OUT", "outputs/uge_sota/sub-08"))
bars = [
    ("CogCap sub-08", 0.175, 0.366, 0.610, 0.721, 0.744, 0.577),
    ("ATM sub-08",    0.160, 0.345, 0.866, 0.734, 0.786, 0.582),
    ("UGE gem_ll_self", 0.050, 0.053, 0.604, 0.547, 0.670, 0.647),
    ("UGE sem_noprompt", 0.033, 0.061, 0.609, 0.551, 0.703, 0.648),
    ("ref g3f_ll_selfgate", 0.183, 0.251, 0.882, 0.696, 0.779, 0.570),
]
print(f"{'row':<22} {'Pix':>7} {'SSIM':>7} {'A5':>6} {'Inc':>6} {'CLIP':>6} {'SwAV':>6}")
print("-" * 64)
for name, pix, ss, a5, inc, cl, sw in bars:
    print(f"{name:<22} {pix:7.3f} {ss:7.3f} {a5:6.3f} {inc:6.3f} {cl:6.3f} {sw:6.3f}")
print("-- this run --")
for p in sorted((OUT / "eval").glob("s08_*.json")):
    d = json.loads(p.read_text())
    n = p.stem.replace("s08_", "")
    print(f"{n:<22} {d.get('pixcorr', float('nan')):7.3f} "
          f"{d.get('ssim', float('nan')):7.3f} {d.get('alex5', float('nan')):6.3f} "
          f"{d.get('inception', float('nan')):6.3f} {d.get('clip', float('nan')):6.3f} "
          f"{d.get('swav', float('nan')):6.3f}")
best_clip = best_pix = None
for p in (OUT / "eval").glob("s08_*.json"):
    d = json.loads(p.read_text())
    if best_clip is None or d.get("clip", 0) > best_clip[1]:
        best_clip = (p.stem, d.get("clip", 0))
    if best_pix is None or d.get("pixcorr", 0) > best_pix[1]:
        best_pix = (p.stem, d.get("pixcorr", 0))
if best_clip:
    print(f"best CLIP {best_clip[0]} {best_clip[1]:.3f}  "
          f"(CogCap 0.744 / ATM 0.786)")
    print(f"best Pix  {best_pix[0]} {best_pix[1]:.3f}  "
          f"(CogCap 0.175 / ATM 0.160)")
PY

echo "===== done @ $(date -Iseconds) ====="
du -sh "${OUT}" | sed 's/^/[disk] /'
