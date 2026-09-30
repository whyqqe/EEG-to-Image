#!/usr/bin/env bash
# ============================================================================
# ACK-DT sub-08 validation (conservative).
#
# Semantic frame : G_img + G_text (trained in ack_heads) + predicted class name
# Instance frame : reuse UCK depth + VAE/LL carrier (not retrained)
# System IP      : UCK mem (proven) ; ACK IP kept as ablation
#
# Prompt matrix (same IP + same F, only prompt changes):
#   hs_uck_empty / hs_uck_deploy / hs_uck_pred / hs_uck_pred_gate
#   hs_uck_oracle (LEAK upper bound) / hs_uck_neural_nn / hs_uck_sinkhorn
#   hs_ack_pred   (ACK IP + pred prompt)  -- readout ablation
# ============================================================================
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" >/dev/null 2>&1 || true

ACK_OUT="${ACK_OUT:-${NB_ROOT}/outputs/ack_s08}"
UCK_OUT="${UCK_OUT:-${NB_ROOT}/outputs/uck}"
COND="${NB_ROOT}/outputs/gem/cond_cache"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
PYTHON="$(command -v python)"
DEVICE="${DEVICE:-cuda:0}"
KEEP_IMAGES="${KEEP_IMAGES:-0}"
EPOCHS="${EPOCHS:-20}"

HS_CN="${HS_CN:-0.40}"
HS_STRENGTH="${HS_STRENGTH:-0.82}"
GEN_STEPS="${GEN_STEPS:-28}"
GEN_GUIDANCE="${GEN_GUIDANCE:-5.0}"
IP_SCALE="${IP_SCALE:-1.0}"

mkdir -p "${ACK_OUT}/logs" "${ACK_OUT}/conds" "${ACK_OUT}/eval" "${NB_ROOT}/outputs/slurm"
export ACK_OUT
echo "[env] PYTHON=${PYTHON} out=${ACK_OUT}"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
warn() { echo "[WARN] $*" >&2; }

require scripts/nda/ack_heads_train.py
require scripts/nda/gem_calib.py
require scripts/nda/generate_hcma_s_decode.py
require scripts/nda/eval_official_seven_dir.py
require "${UCK_OUT}/sub-08/full/conds/ip_mem_test.npy"
require "${UCK_OUT}/sub-08/full/spatial/pred_depth_rgb_512/199.png"
require "${UCK_OUT}/shared/g_img_concept.npy"
require "${COND}/clip_img1024_test.npy"
require "${COND}/clip_img1024_train.npy"
require "${NB_ROOT}/outputs/g2f/prompts/prompts_oracle.json"

# Prefer UCK's own LL; fall back to shared sdedit LL
UCK_F="${UCK_OUT}/sub-08/full/spatial/pred_depth_rgb_512"
if [[ -f "${UCK_OUT}/sub-08/full/spatial/pred_lowlevel_rgb_512/199.png" ]]; then
  LL="${UCK_OUT}/sub-08/full/spatial/pred_lowlevel_rgb_512"
else
  LL="${NB_ROOT}/outputs/sdedit_ll_full10/sub-08/vae_head/pred_lowlevel_rgb_512"
fi
require "${LL}/199.png"

echo "===== [0] train ACK heads @ $(date -Iseconds) ====="
HEAD="${ACK_OUT}/heads"
mkdir -p "${HEAD}"
if [[ ! -f "${HEAD}/report.json" || ! -f "${HEAD}/prompts/prompts_pred.json" ]]; then
  if ! "${PYTHON}" scripts/nda/ack_heads_train.py \
      --out "${HEAD}" --test-subject 8 \
      --gallery-cache "${UCK_OUT}/shared" \
      --uck-conds "${UCK_OUT}/sub-08/full/conds" \
      --epochs "${EPOCHS}" --device "${DEVICE}" \
      >> "${ACK_OUT}/logs/heads.log" 2>&1; then
    echo "[FATAL] ack_heads_train failed; last 50 lines:" >&2
    tail -n 50 "${ACK_OUT}/logs/heads.log" >&2 || true
    exit 1
  fi
else
  echo "[SKIP] ack heads"
fi
require "${HEAD}/prompts/prompts_pred.json"
require "${HEAD}/report.json"
cp -f "${HEAD}/report.json" "${ACK_OUT}/heads_report.json"

calib() {
  local src="$1" dst="$2" tag="$3"
  if [[ -f "${dst}" ]]; then
    printf '%s\n' "${dst}"
    return 0
  fi
  if ! "${PYTHON}" scripts/nda/gem_calib.py \
      --in "${src}" --out "${dst}" \
      --ref "${COND}/clip_img1024_train.npy" --tag "${tag}" \
      >> "${ACK_OUT}/logs/calib.log" 2>&1; then
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
      >> "${ACK_OUT}/logs/eval.log" 2>&1; then
    warn "eval ${tag} failed"
    return 0
  fi
  if [[ -s "${ev}" && "${KEEP_IMAGES}" != "1" ]]; then
    rm -rf "$(dirname "${gdir}")"
  fi
}

gen_hs() {
  local tag="$1" cond="$2" pf="$3"
  local ev="${ACK_OUT}/eval/s08_${tag}.json"
  local gdir="${ACK_OUT}/gen/${tag}"
  [[ -f "${ev}" ]] && { echo "[SKIP] ${tag}"; return 0; }
  require "${cond}"; require "${pf}"
  echo "===== HS ${tag} @ $(date -Iseconds) ====="
  if ! "${PYTHON}" scripts/nda/generate_hcma_s_decode.py \
      --embed-npy "${cond}" --prompts-json "${pf}" \
      --depth-rgb-dir "${UCK_F}" --lowlevel-rgb-dir "${LL}" \
      --output-dir "${gdir}" --tag "${tag}" \
      --cn-scale "${HS_CN}" --strength "${HS_STRENGTH}" --ip-scale "${IP_SCALE}" \
      --gen-steps "${GEN_STEPS}" --gen-guidance "${GEN_GUIDANCE}" \
      >> "${ACK_OUT}/logs/gen_${tag}.log" 2>&1; then
    echo "[FATAL] gen ${tag} failed" >&2
    tail -n 40 "${ACK_OUT}/logs/gen_${tag}.log" >&2 || true
    exit 1
  fi
  eval_row "${tag}" "${gdir}/generated" "${ev}"
}

echo "===== [1] calibrate IPs @ $(date -Iseconds) ====="
UCKIP="$(calib "${UCK_OUT}/sub-08/full/conds/ip_mem_test.npy" "${ACK_OUT}/conds/uck.npy" "s08_ack_uck")"
ACKIP="$(calib "${HEAD}/conds/ip_ack_test.npy" "${ACK_OUT}/conds/ack.npy" "s08_ack_ip")"

PDIR="${HEAD}/prompts"
require "${PDIR}/prompts_pred.json"
require "${PDIR}/prompts_pred_gate.json"
require "${PDIR}/prompts_empty.json"
require "${PDIR}/prompts_deploy.json"
require "${PDIR}/prompts_oracle.json"
require "${PDIR}/prompts_neural_nn.json"
require "${PDIR}/prompts_sinkhorn.json"

echo "===== [2] generate+eval @ $(date -Iseconds) ====="
# Same UCK IP + UCK F; only prompt changes (isolates class-name contribution)
gen_hs hs_uck_empty      "${UCKIP}" "${PDIR}/prompts_empty.json"
gen_hs hs_uck_deploy     "${UCKIP}" "${PDIR}/prompts_deploy.json"
gen_hs hs_uck_pred       "${UCKIP}" "${PDIR}/prompts_pred.json"
gen_hs hs_uck_pred_gate  "${UCKIP}" "${PDIR}/prompts_pred_gate.json"
gen_hs hs_uck_oracle     "${UCKIP}" "${PDIR}/prompts_oracle.json"
gen_hs hs_uck_neural_nn  "${UCKIP}" "${PDIR}/prompts_neural_nn.json"
gen_hs hs_uck_sinkhorn   "${UCKIP}" "${PDIR}/prompts_sinkhorn.json"
# ACK IP ablation with best legal prompt
gen_hs hs_ack_pred       "${ACKIP}" "${PDIR}/prompts_pred.json"

echo "===== summary @ $(date -Iseconds) ====="
"${PYTHON}" - <<'PY'
import json
from pathlib import Path
ACK = Path("/project/peilab/why/NeuroBridge/outputs/ack_s08")
bars = [
    ("CogCap sub-08", 0.175, 0.366, 0.744),
    ("ATM sub-08", 0.160, 0.345, 0.786),
    ("UCK remasure deploy", 0.165, 0.376, 0.803),
    ("oracle leak (UGE gem)", float("nan"), float("nan"), 0.935),
]
print(f"{'row':<28} {'Pix':>7} {'SSIM':>6} {'CLIP':>6}")
print("-" * 52)
for n, pix, ss, cl in bars:
    print(f"{n:<28} {pix:7.3f} {ss:6.3f} {cl:6.3f}")
print("-- ACK-DT sub-08 --")
for p in sorted(ACK.glob("eval/s08_*.json")):
    d = json.loads(p.read_text())
    print(f"{p.stem:<28} {d.get('pixcorr', float('nan')):7.3f} "
          f"{d.get('ssim', float('nan')):6.3f} {d.get('clip', float('nan')):6.3f}")
hr = ACK / "heads_report.json"
if hr.is_file():
    d = json.loads(hr.read_text())
    print("-- heads / naming --")
    print(f"  LOO1654={d.get('loo_train_1654',{}).get('top1', float('nan')):.4f} "
          f"valLOO={d.get('val_b_loo_1654',{}).get('top1', float('nan')):.4f}")
    print(f"  test200 top1 z/q/ip/sinkhorn = "
          f"{d.get('test200_top1_z', float('nan')):.3f}/"
          f"{d.get('test200_top1_q', float('nan')):.3f}/"
          f"{d.get('test200_top1_ip', float('nan')):.3f}/"
          f"{d.get('test200_top1_sinkhorn', float('nan')):.3f}")
    print(f"  gated_to_object={d.get('n_gated_to_object')} "
          f"margin_mean={d.get('mean_margin_z', float('nan')):.4f}")
PY

rm -rf "${ACK_OUT}/gen"
find "${ACK_OUT}" -name 'last.pth' -delete || true
echo "===== done @ $(date -Iseconds) ====="
du -sh "${ACK_OUT}" | sed 's/^/[disk] /'
