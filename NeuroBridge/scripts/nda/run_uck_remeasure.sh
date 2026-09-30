#!/usr/bin/env bash
# Re-generate UCK rows and score PixCorr/SSIM with the official gray@425 protocol.
# No retraining. Old RGB@256 eval JSONs are moved to eval_rgb256/.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" >/dev/null 2>&1 || true
UCK_OUT="${UCK_OUT:-${NB_ROOT}/outputs/uck}"
PYTHON="$(command -v python)"
DEVICE="${DEVICE:-cuda:0}"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
KEEP_IMAGES="${KEEP_IMAGES:-0}"
PDEP="${NB_ROOT}/outputs/g2f/prompts/prompts_deploy.json"

HS_CN="${HS_CN:-0.40}"
HS_STRENGTH="${HS_STRENGTH:-0.82}"
SD_STRENGTH="${SD_STRENGTH:-0.82}"
GEN_STEPS="${GEN_STEPS:-28}"
GEN_GUIDANCE="${GEN_GUIDANCE:-5.0}"
IP_SCALE="${IP_SCALE:-1.0}"

mkdir -p "${UCK_OUT}/logs"
export UCK_OUT
echo "[env] PYTHON=${PYTHON} out=${UCK_OUT}"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
warn() { echo "[WARN] $*" >&2; }

require scripts/nda/eval_official_seven_dir.py
require scripts/nda/eval_standard7.py
require "${PDEP}"

# archive the RGB@256 scores so the delta stays auditable
for evdir in "${UCK_OUT}"/sub-*/eval; do
  [[ -d "${evdir}" ]] || continue
  bak="$(dirname "${evdir}")/eval_rgb256"
  mkdir -p "${bak}"
  find "${evdir}" -maxdepth 1 -name 's*.json' -exec mv -n {} "${bak}/" \;
done

eval_row() {
  local tag="$1" gdir="$2" ev="$3"
  [[ -f "${gdir}/199.png" ]] || { warn "eval ${tag}: no images"; return 0; }
  echo "--- eval ${tag} (official gray@425)"
  if ! "${PYTHON}" scripts/nda/eval_official_seven_dir.py \
      --gen-dir "${gdir}" --output-json "${ev}" --tag "${tag}" \
      --images-root "${IMAGES_ROOT}" --device "${DEVICE}" \
      >> "${UCK_OUT}/logs/remeasure_eval.log" 2>&1; then
    echo "[FATAL] eval ${tag} failed" >&2
    tail -n 30 "${UCK_OUT}/logs/remeasure_eval.log" >&2 || true
    exit 1
  fi
  if [[ -s "${ev}" && "${KEEP_IMAGES}" != "1" ]]; then
    rm -rf "$(dirname "${gdir}")"
  fi
}

pick_depth() {
  local sub="$1"
  local own="${UCK_OUT}/${sub}/full/spatial/pred_depth_rgb_512/199.png"
  if [[ -f "${own}" ]]; then
    printf '%s\n' "${UCK_OUT}/${sub}/full/spatial/pred_depth_rgb_512"
  else
    printf '%s\n' "${NB_ROOT}/outputs/hcma_s_full10/${sub}/depth/pred_depth_rgb_512"
  fi
}

pick_ll() {
  local sub="$1"
  local own="${UCK_OUT}/${sub}/full/spatial/pred_lowlevel_rgb_512/199.png"
  if [[ -f "${own}" ]]; then
    printf '%s\n' "${UCK_OUT}/${sub}/full/spatial/pred_lowlevel_rgb_512"
  else
    printf '%s\n' "${NB_ROOT}/outputs/sdedit_ll_full10/${sub}/vae_head/pred_lowlevel_rgb_512"
  fi
}

pick_vae() {
  local sub="$1"
  local own="${UCK_OUT}/${sub}/full/spatial/pred_vae_test_scaled.npy"
  if [[ -f "${own}" ]]; then
    printf '%s\n' "${own}"
  else
    printf '%s\n' ""
  fi
}

gen_hs() {
  local sub="$1" tag="$2" cond="$3" pf="$4"
  local sd="${sub#sub-}"
  local ev="${UCK_OUT}/${sub}/eval/s${sd}_${tag}.json"
  local gdir="${UCK_OUT}/${sub}/gen/${tag}"
  [[ -f "${ev}" ]] && { echo "[SKIP] ${tag}"; return 0; }
  require "${cond}"; require "${pf}"
  local depth ll
  depth="$(pick_depth "${sub}")"
  ll="$(pick_ll "${sub}")"
  require "${depth}/199.png"
  require "${ll}/199.png"
  echo "===== HS ${sub} ${tag} @ $(date -Iseconds) ====="
  if ! "${PYTHON}" scripts/nda/generate_hcma_s_decode.py \
      --embed-npy "${cond}" --prompts-json "${pf}" \
      --depth-rgb-dir "${depth}" --lowlevel-rgb-dir "${ll}" \
      --output-dir "${gdir}" --tag "${tag}" \
      --cn-scale "${HS_CN}" --strength "${HS_STRENGTH}" --ip-scale "${IP_SCALE}" \
      --gen-steps "${GEN_STEPS}" --gen-guidance "${GEN_GUIDANCE}" \
      >> "${UCK_OUT}/logs/remeasure_gen.log" 2>&1; then
    echo "[FATAL] gen ${tag} failed" >&2
    tail -n 40 "${UCK_OUT}/logs/remeasure_gen.log" >&2 || true
    exit 1
  fi
  eval_row "${tag}" "${gdir}/generated" "${ev}"
}

gen_atm() {
  local sub="$1" tag="$2" cond="$3" pf="$4"
  local sd="${sub#sub-}"
  local ev="${UCK_OUT}/${sub}/eval/s${sd}_${tag}.json"
  local gdir="${UCK_OUT}/${sub}/gen/${tag}"
  [[ -f "${ev}" ]] && { echo "[SKIP] ${tag}"; return 0; }
  require "${cond}"
  local vae extra=()
  vae="$(pick_vae "${sub}")"
  [[ -n "${vae}" ]] && extra+=(--vae-latent-npy "${vae}")
  [[ -z "${vae}" ]] && extra+=(--lowlevel-rgb-dir "$(pick_ll "${sub}")")
  echo "===== ATM ${sub} ${tag} @ $(date -Iseconds) ====="
  if ! "${PYTHON}" scripts/nda/generate_atm_aligned_decode.py \
      --mode sdedit --embed-npy "${cond}" --prompts-json "${pf}" \
      "${extra[@]}" \
      --output-dir "${gdir}" --tag "${tag}" \
      --strength "${SD_STRENGTH}" --ip-scale "${IP_SCALE}" \
      --gen-steps "${GEN_STEPS}" --gen-guidance "${GEN_GUIDANCE}" \
      --device "${DEVICE}" >> "${UCK_OUT}/logs/remeasure_gen.log" 2>&1; then
    echo "[FATAL] gen ${tag} failed" >&2
    tail -n 40 "${UCK_OUT}/logs/remeasure_gen.log" >&2 || true
    exit 1
  fi
  eval_row "${tag}" "${gdir}/generated" "${ev}"
}

for SID in 8 1 2 3 4 5 6 7 9 10; do
  SUB="$(printf 'sub-%02d' "${SID}")"
  echo "########## remasure ${SUB} @ $(date -Iseconds) ##########"
  require "${UCK_OUT}/${SUB}/conds/mem.npy"
  mkdir -p "${UCK_OUT}/${SUB}/eval"
  MEM="${UCK_OUT}/${SUB}/conds/mem.npy"
  gen_hs "${SUB}" hs_mem_deploy "${MEM}" "${PDEP}"
  gen_atm "${SUB}" atm_mem_deploy "${MEM}" "${PDEP}"
  if [[ "${SID}" == "8" ]]; then
    FIVE="${UCK_OUT}/sub-08/full/prompts/prompts_five.json"
    EMPTY="${UCK_OUT}/sub-08/full/prompts/prompts_empty.json"
    gen_hs "${SUB}" hs_mem_five "${MEM}" "${FIVE}"
    gen_hs "${SUB}" hs_mem_none "${MEM}" "${EMPTY}"
    gen_hs "${SUB}" hs_q_deploy "${UCK_OUT}/sub-08/conds/q.npy" "${PDEP}"
    gen_hs "${SUB}" hs_imgbank_deploy "${UCK_OUT}/sub-08/conds/imgbank.npy" "${PDEP}"
    gen_hs "${SUB}" hs_noise_deploy "${UCK_OUT}/sub-08/conds/noise.npy" "${PDEP}"
    gen_hs "${SUB}" hs_text_deploy "${UCK_OUT}/sub-08/conds/text.npy" "${PDEP}"
  fi
  echo "[disk] ${SUB}: $(du -sh "${UCK_OUT}/${SUB}" | cut -f1)"
done

echo "===== remasure summary @ $(date -Iseconds) ====="
"${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
from statistics import mean
ROOT = Path(os.environ["UCK_OUT"])
print(f"{'row':<28} {'Pix':>7} {'SSIM':>6} {'CLIP':>6}  (old RGB256 Pix/SSIM)")
print("-" * 78)
hs = []
for p in sorted(ROOT.glob("sub-*/eval/s*.json")):
    d = json.loads(p.read_text())
    oldp = p.parent.parent / "eval_rgb256" / p.name
    old = json.loads(oldp.read_text()) if oldp.is_file() else {}
    tag = f"{p.parent.parent.name}/{p.stem}"
    print(f"{tag:<28} {d.get('pixcorr', float('nan')):7.3f} {d.get('ssim', float('nan')):6.3f} "
          f"{d.get('clip', float('nan')):6.3f}    "
          f"{old.get('pixcorr', float('nan')):6.3f}/{old.get('ssim', float('nan')):5.3f}")
    if p.stem.endswith("hs_mem_deploy"):
        hs.append(d)
if hs:
    print(f"HS 10-subj mean  Pix={mean(x['pixcorr'] for x in hs):.3f}  "
          f"SSIM={mean(x['ssim'] for x in hs):.3f}  CLIP={mean(x['clip'] for x in hs):.3f}")
print("bars: CogCap08 Pix 0.175 SSIM 0.366 CLIP 0.744 | ATM08 0.160/0.345/0.786 | CogCap10 0.150/0.347/0.715")
PY
echo "===== done @ $(date -Iseconds) ====="
du -sh "${UCK_OUT}" | sed 's/^/[disk] /'
