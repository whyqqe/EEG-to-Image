#!/usr/bin/env bash
# ============================================================================
# Hybrid sub-08 feasibility check.
#
# Full architecture under test:
#   neural address (μ) → hybrid IP variants → HCMA-S decode
#   structure = UCK F (depth) + existing LL   (reuse proven structure path)
#
# Rows (all deploy prompt, official gray@425):
#   hs_uck_uckF     UCK mem IP  + UCK F     (current best control)
#   hs_nat_uckF     pure NAT IP + UCK F     (IP bottleneck?)
#   hs_snap_uckF    snap hybrid + UCK F     (translate→gallery mem)
#   hs_short_uckF   shortlist+q + UCK F     (neural shortlist, CLIP refine)
#   hs_gate_uckF    α⊙q gate   + UCK F     (neural prior × CLIP affinity)
#   hs_blend_uckF   0.5 nat+uck + UCK F
#   hs_hard_uckF    top1 translate + UCK F
#   hs_snap_natF    snap + NAT F            (structure ablation)
# ============================================================================
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" >/dev/null 2>&1 || true

HYB_OUT="${HYB_OUT:-${NB_ROOT}/outputs/hybrid_s08}"
UCK_OUT="${UCK_OUT:-${NB_ROOT}/outputs/uck}"
NAT_OUT="${NAT_OUT:-${NB_ROOT}/outputs/nat}"
COND="${NB_ROOT}/outputs/gem/cond_cache"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
PYTHON="$(command -v python)"
DEVICE="${DEVICE:-cuda:0}"
KEEP_IMAGES="${KEEP_IMAGES:-0}"
PDEP="${NB_ROOT}/outputs/g2f/prompts/prompts_deploy.json"

HS_CN="${HS_CN:-0.40}"
HS_STRENGTH="${HS_STRENGTH:-0.82}"
GEN_STEPS="${GEN_STEPS:-28}"
GEN_GUIDANCE="${GEN_GUIDANCE:-5.0}"
IP_SCALE="${IP_SCALE:-1.0}"

mkdir -p "${HYB_OUT}/logs" "${HYB_OUT}/conds" "${HYB_OUT}/eval" "${NB_ROOT}/outputs/slurm"
export HYB_OUT
echo "[env] PYTHON=${PYTHON} out=${HYB_OUT}"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
warn() { echo "[WARN] $*" >&2; }

require scripts/nda/hybrid_export.py
require scripts/nda/gem_calib.py
require scripts/nda/generate_hcma_s_decode.py
require scripts/nda/eval_official_seven_dir.py
require "${PDEP}"
require "${UCK_OUT}/sub-08/full/conds/ip_mem_test.npy"
require "${UCK_OUT}/sub-08/full/conds/ip_q_test.npy"
require "${UCK_OUT}/sub-08/full/spatial/pred_depth_rgb_512/199.png"
require "${NAT_OUT}/sub-08/full/conds/ip_nat_test.npy"
require "${NAT_OUT}/sub-08/full/spatial/pred_depth_rgb_512/199.png"
require "${NAT_OUT}/sub-08/full/proto/mu_all.npy"
require "${NB_ROOT}/outputs/sdedit_ll_full10/sub-08/vae_head/pred_lowlevel_rgb_512/199.png"
require "${UCK_OUT}/shared/g_img_concept.npy"

UCK_F="${UCK_OUT}/sub-08/full/spatial/pred_depth_rgb_512"
NAT_F="${NAT_OUT}/sub-08/full/spatial/pred_depth_rgb_512"
LL="${NB_ROOT}/outputs/sdedit_ll_full10/sub-08/vae_head/pred_lowlevel_rgb_512"

echo "===== [0] export hybrid IPs @ $(date -Iseconds) ====="
if [[ ! -f "${HYB_OUT}/full/report.json" ]]; then
  "${PYTHON}" scripts/nda/hybrid_export.py \
      --out "${HYB_OUT}/full" --test-subject 8 \
      --gallery-cache "${UCK_OUT}/shared" \
      --uck-conds "${UCK_OUT}/sub-08/full/conds" \
      --nat-proto "${NAT_OUT}/sub-08/full/proto" \
      --device cpu \
      >> "${HYB_OUT}/logs/export.log" 2>&1
else
  echo "[SKIP] hybrid export"
fi
require "${HYB_OUT}/full/conds/ip_snap_test.npy"
cp -f "${HYB_OUT}/full/report.json" "${HYB_OUT}/export_report.json"

calib() {
  local src="$1" dst="$2" tag="$3"
  if [[ -f "${dst}" ]]; then
    printf '%s\n' "${dst}"
    return 0
  fi
  if ! "${PYTHON}" scripts/nda/gem_calib.py \
      --in "${src}" --out "${dst}" \
      --ref "${COND}/clip_img1024_train.npy" --tag "${tag}" \
      >> "${HYB_OUT}/logs/calib.log" 2>&1; then
    warn "calib ${tag} failed; using raw"
    cp -f "${src}" "${dst}"
  fi
  printf '%s\n' "${dst}"
}

eval_row() {
  local tag="$1" gdir="$2" ev="$3"
  [[ -f "${ev}" ]] && { echo "[SKIP] eval ${tag}"; return 0; }
  [[ -f "${gdir}/199.png" ]] || { warn "eval ${tag}: no images"; return 0; }
  echo "--- eval ${tag}"
  if ! "${PYTHON}" scripts/nda/eval_official_seven_dir.py \
      --gen-dir "${gdir}" --output-json "${ev}" --tag "${tag}" \
      --images-root "${IMAGES_ROOT}" --device "${DEVICE}" --skip-if-exists \
      >> "${HYB_OUT}/logs/eval.log" 2>&1; then
    warn "eval ${tag} failed"
    return 0
  fi
  if [[ -s "${ev}" && "${KEEP_IMAGES}" != "1" ]]; then
    rm -rf "$(dirname "${gdir}")"
  fi
}

gen_hs() {
  local tag="$1" cond="$2" depth="$3"
  local ev="${HYB_OUT}/eval/s08_${tag}.json"
  local gdir="${HYB_OUT}/gen/${tag}"
  [[ -f "${ev}" ]] && { echo "[SKIP] ${tag}"; return 0; }
  require "${cond}"; require "${depth}/199.png"; require "${LL}/199.png"
  echo "===== HS ${tag} @ $(date -Iseconds) ====="
  if ! "${PYTHON}" scripts/nda/generate_hcma_s_decode.py \
      --embed-npy "${cond}" --prompts-json "${PDEP}" \
      --depth-rgb-dir "${depth}" --lowlevel-rgb-dir "${LL}" \
      --output-dir "${gdir}" --tag "${tag}" \
      --cn-scale "${HS_CN}" --strength "${HS_STRENGTH}" --ip-scale "${IP_SCALE}" \
      --gen-steps "${GEN_STEPS}" --gen-guidance "${GEN_GUIDANCE}" \
      >> "${HYB_OUT}/logs/gen_${tag}.log" 2>&1; then
    echo "[FATAL] gen ${tag} failed" >&2
    tail -n 40 "${HYB_OUT}/logs/gen_${tag}.log" >&2 || true
    exit 1
  fi
  eval_row "${tag}" "${gdir}/generated" "${ev}"
}

echo "===== [1] calibrate @ $(date -Iseconds) ====="
UCKIP="$(calib "${HYB_OUT}/full/conds/ip_uck_test.npy" "${HYB_OUT}/conds/uck.npy" "s08_hyb_uck")"
NATIP="$(calib "${HYB_OUT}/full/conds/ip_nat_test.npy" "${HYB_OUT}/conds/nat.npy" "s08_hyb_nat")"
SNAP="$(calib "${HYB_OUT}/full/conds/ip_snap_test.npy" "${HYB_OUT}/conds/snap.npy" "s08_hyb_snap")"
SHORT="$(calib "${HYB_OUT}/full/conds/ip_short_test.npy" "${HYB_OUT}/conds/short.npy" "s08_hyb_short")"
GATE="$(calib "${HYB_OUT}/full/conds/ip_gate_test.npy" "${HYB_OUT}/conds/gate.npy" "s08_hyb_gate")"
BLEND="$(calib "${HYB_OUT}/full/conds/ip_blend_test.npy" "${HYB_OUT}/conds/blend.npy" "s08_hyb_blend")"
HARD="$(calib "${HYB_OUT}/full/conds/ip_hard_test.npy" "${HYB_OUT}/conds/hard.npy" "s08_hyb_hard")"

echo "===== [2] generate+eval @ $(date -Iseconds) ====="
gen_hs hs_uck_uckF   "${UCKIP}" "${UCK_F}"
gen_hs hs_nat_uckF   "${NATIP}" "${UCK_F}"
gen_hs hs_snap_uckF  "${SNAP}"  "${UCK_F}"
gen_hs hs_short_uckF "${SHORT}" "${UCK_F}"
gen_hs hs_gate_uckF  "${GATE}"  "${UCK_F}"
gen_hs hs_blend_uckF "${BLEND}" "${UCK_F}"
gen_hs hs_hard_uckF  "${HARD}"  "${UCK_F}"
gen_hs hs_snap_natF  "${SNAP}"  "${NAT_F}"

echo "===== summary @ $(date -Iseconds) ====="
"${PYTHON}" - <<'PY'
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2] if False else Path("/project/peilab/why/NeuroBridge")
HYB = Path("/project/peilab/why/NeuroBridge/outputs/hybrid_s08")
bars = [
    ("CogCap sub-08", 0.175, 0.366, 0.744),
    ("ATM sub-08", 0.160, 0.345, 0.786),
    ("UCK hs_mem_deploy", 0.165, 0.376, 0.803),
    ("NAT hs_nat_deploy", 0.164, 0.366, 0.770),
]
print(f"{'row':<28} {'Pix':>7} {'SSIM':>6} {'CLIP':>6}")
print("-" * 52)
for n, pix, ss, cl in bars:
    print(f"{n:<28} {pix:7.3f} {ss:6.3f} {cl:6.3f}")
print("-- hybrid sub-08 --")
for p in sorted(HYB.glob("eval/s08_*.json")):
    d = json.loads(p.read_text())
    print(f"{p.stem:<28} {d.get('pixcorr', float('nan')):7.3f} "
          f"{d.get('ssim', float('nan')):6.3f} {d.get('clip', float('nan')):6.3f}")
er = HYB / "export_report.json"
if er.is_file():
    d = json.loads(er.read_text())
    print("-- condition vs true CLIP-image --")
    for k, v in d.get("rows", {}).items():
        print(f"  {k:<22} vs_true={v['vs_true']:.3f} vs_cl={v['vs_cl']:+.3f} rowcos={v['rowcos']:.3f}")
PY

rm -rf "${HYB_OUT}/gen"
echo "===== done @ $(date -Iseconds) ====="
du -sh "${HYB_OUT}" | sed 's/^/[disk] /'
