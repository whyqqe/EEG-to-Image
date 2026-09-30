#!/usr/bin/env bash
# ============================================================================
# ACK Phase-0 fix: rebuild prompts with q·G_test naming, prompt-only remeasure.
# No head retrain. No structure retrain. Reuses UCK IP + UCK F.
# Writes to ACK_OUT (default outputs/ack_s08r) so v1 results stay auditable.
# ============================================================================
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" >/dev/null 2>&1 || true

ACK_OUT="${ACK_OUT:-${NB_ROOT}/outputs/ack_s08r}"
ACK_HEADS="${ACK_HEADS:-${NB_ROOT}/outputs/ack_s08/heads}"
UCK_OUT="${UCK_OUT:-${NB_ROOT}/outputs/uck}"
COND="${NB_ROOT}/outputs/gem/cond_cache"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
PYTHON="$(command -v python)"
DEVICE="${DEVICE:-cuda:0}"
KEEP_IMAGES="${KEEP_IMAGES:-0}"
GATE_MARGIN="${GATE_MARGIN:-0.02}"

HS_CN="${HS_CN:-0.40}"
HS_STRENGTH="${HS_STRENGTH:-0.82}"
GEN_STEPS="${GEN_STEPS:-28}"
GEN_GUIDANCE="${GEN_GUIDANCE:-5.0}"
IP_SCALE="${IP_SCALE:-1.0}"

mkdir -p "${ACK_OUT}/logs" "${ACK_OUT}/conds" "${ACK_OUT}/eval" "${ACK_OUT}/prompts" \
         "${NB_ROOT}/outputs/slurm"
export ACK_OUT
echo "[env] PYTHON=${PYTHON} out=${ACK_OUT} heads=${ACK_HEADS}"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
warn() { echo "[WARN] $*" >&2; }

require scripts/nda/ack_rebuild_prompts.py
require scripts/nda/gem_calib.py
require scripts/nda/generate_hcma_s_decode.py
require scripts/nda/eval_official_seven_dir.py
require "${ACK_HEADS}/conds/ip_q_test.npy"
require "${UCK_OUT}/sub-08/full/conds/ip_mem_test.npy"
require "${UCK_OUT}/sub-08/full/conds/ip_q_test.npy"
require "${UCK_OUT}/sub-08/full/spatial/pred_depth_rgb_512/199.png"
require "${COND}/clip_img1024_test.npy"
require "${COND}/clip_img1024_train.npy"
require "${NB_ROOT}/outputs/g2f/prompts/prompts_oracle.json"

UCK_F="${UCK_OUT}/sub-08/full/spatial/pred_depth_rgb_512"
if [[ -f "${UCK_OUT}/sub-08/full/spatial/pred_lowlevel_rgb_512/199.png" ]]; then
  LL="${UCK_OUT}/sub-08/full/spatial/pred_lowlevel_rgb_512"
else
  LL="${NB_ROOT}/outputs/sdedit_ll_full10/sub-08/vae_head/pred_lowlevel_rgb_512"
fi
require "${LL}/199.png"

echo "===== [0] rebuild prompts with q @ $(date -Iseconds) ====="
if [[ ! -f "${ACK_OUT}/prompts/rebuild_report.json" ]]; then
  if ! "${PYTHON}" scripts/nda/ack_rebuild_prompts.py \
      --heads-dir "${ACK_HEADS}" \
      --out-prompts "${ACK_OUT}/prompts" \
      --uck-q "${UCK_OUT}/sub-08/full/conds/ip_q_test.npy" \
      --mu-npy "${ACK_HEADS}/proto/mu_all.npy" \
      --gate-margin "${GATE_MARGIN}" \
      --gate-margin-hi 0.03 \
      --report "${ACK_OUT}/prompts/rebuild_report.json" \
      >> "${ACK_OUT}/logs/rebuild.log" 2>&1; then
    echo "[FATAL] rebuild prompts failed; last 40 lines:" >&2
    tail -n 40 "${ACK_OUT}/logs/rebuild.log" >&2 || true
    exit 1
  fi
else
  echo "[SKIP] rebuild prompts"
fi
require "${ACK_OUT}/prompts/prompts_pred.json"
require "${ACK_OUT}/prompts/rebuild_report.json"
cp -f "${ACK_OUT}/prompts/rebuild_report.json" "${ACK_OUT}/rebuild_report.json"
echo "----- rebuild report -----"
"${PYTHON}" -c "import json;print(json.dumps(json.load(open('${ACK_OUT}/rebuild_report.json')),indent=2))"

# Abort early if q naming is still near chance (would repeat the v1 failure mode)
"${PYTHON}" - <<PY
import json, sys
d=json.load(open("${ACK_OUT}/rebuild_report.json"))
t1=float(d.get("test200_top1_q", 0.0))
print(f"[gate] test200_top1_q={t1:.4f}")
if t1 < 0.05:
    print("[FATAL] q naming still ~chance; refusing to spend GPU on bad prompts", file=sys.stderr)
    sys.exit(1)
PY

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
    warn "calib ${tag} failed; raw copy"
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

echo "===== [1] calibrate @ $(date -Iseconds) ====="
UCKIP="$(calib "${UCK_OUT}/sub-08/full/conds/ip_mem_test.npy" "${ACK_OUT}/conds/uck.npy" "s08r_uck")"
ACKIP="$(calib "${ACK_HEADS}/conds/ip_ack_test.npy" "${ACK_OUT}/conds/ack.npy" "s08r_ack")"

P="${ACK_OUT}/prompts"
echo "===== [2] generate+eval @ $(date -Iseconds) ====="
gen_hs hs_uck_empty         "${UCKIP}" "${P}/prompts_empty.json"
gen_hs hs_uck_deploy        "${UCKIP}" "${P}/prompts_deploy.json"
gen_hs hs_uck_pred_q        "${UCKIP}" "${P}/prompts_pred.json"
gen_hs hs_uck_pred_q_gate   "${UCKIP}" "${P}/prompts_pred_gate.json"
gen_hs hs_uck_pred_q_gatehi "${UCKIP}" "${P}/prompts_pred_gate_hi.json"
gen_hs hs_uck_sinkhorn_q    "${UCKIP}" "${P}/prompts_sinkhorn.json"
gen_hs hs_uck_oracle        "${UCKIP}" "${P}/prompts_oracle.json"
gen_hs hs_uck_neural_nn     "${UCKIP}" "${P}/prompts_neural_nn.json"
if [[ -f "${P}/prompts_pred_uckq.json" ]]; then
  gen_hs hs_uck_pred_uckq      "${UCKIP}" "${P}/prompts_pred_uckq.json"
  gen_hs hs_uck_pred_uckq_gate "${UCKIP}" "${P}/prompts_pred_uckq_gate.json"
fi
gen_hs hs_ack_pred_q        "${ACKIP}" "${P}/prompts_pred.json"

echo "===== summary @ $(date -Iseconds) ====="
"${PYTHON}" - <<'PY'
import json
from pathlib import Path
ACK = Path("/project/peilab/why/NeuroBridge/outputs/ack_s08r")
OLD = Path("/project/peilab/why/NeuroBridge/outputs/ack_s08")
print(f"{'row':<28} {'Pix':>7} {'SSIM':>6} {'CLIP':>6}")
print("-" * 52)
print(f"{'v1 pred (z, bad)':<28} {0.172:7.3f} {0.387:6.3f} {0.776:6.3f}")
print(f"{'v1 empty':<28} {0.168:7.3f} {0.376:6.3f} {0.817:6.3f}")
print(f"{'v1 oracle':<28} {0.174:7.3f} {0.387:6.3f} {0.910:6.3f}")
print("-- reprompt (q naming) --")
for p in sorted(ACK.glob("eval/s08_*.json")):
    d = json.loads(p.read_text())
    print(f"{p.stem:<28} {d.get('pixcorr', float('nan')):7.3f} "
          f"{d.get('ssim', float('nan')):6.3f} {d.get('clip', float('nan')):6.3f}")
rr = ACK / "rebuild_report.json"
if rr.is_file():
    d = json.loads(rr.read_text())
    print("-- naming --")
    print(f"  top1 z/q/sinkhorn = {d.get('test200_top1_z_diagnostic'):.3f}/"
          f"{d.get('test200_top1_q'):.3f}/{d.get('test200_top1_sinkhorn_q'):.3f}")
    print(f"  gated={d.get('n_gated_to_object')}/200 "
          f"frac_ge_gate={d.get('frac_margin_ge_gate'):.3f} "
          f"correct_pred={d.get('n_correct_pred')}/200")
    if "test200_top1_uck_q" in d:
        print(f"  uck_q top1={d['test200_top1_uck_q']:.3f} gated={d.get('n_gated_uck_q')}")
PY

rm -rf "${ACK_OUT}/gen"
echo "===== done @ $(date -Iseconds) ====="
du -sh "${ACK_OUT}" | sed 's/^/[disk] /'
