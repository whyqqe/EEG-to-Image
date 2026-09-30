#!/usr/bin/env bash
# sub-08 paper finalize: metrics, FID, 2WC, bootstrap, panels, main table, freeze md
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MAX_IMAGES="${ERDC_MAX_IMAGES:-0}"
EXTRA=()
[[ "${MAX_IMAGES}" != "0" ]] && EXTRA+=(--max-images "${MAX_IMAGES}")

MET_DIR="outputs/erdc/w15_metrics"

full_metrics () {
  local gen="$1"; local tag="$2"
  local out="${MET_DIR}/${tag}.json"
  [[ -f "${out}" ]] && echo "[SKIP] ${tag}" && return 0
  [[ -d "${gen}" ]] || { echo "[WARN] miss ${gen}"; return 0; }
  python scripts/erdc_full_metrics.py --gen-dir "${gen}" --output-json "${out}" --tag "${tag}" "${EXTRA[@]}"
}

fid_metrics () {
  local gen="$1"; local tag="$2"
  local out="${MET_DIR}/${tag}_fid.json"
  [[ -f "${out}" ]] && echo "[SKIP] fid ${tag}" && return 0
  [[ -d "${gen}" ]] || return 0
  python scripts/erdc_fid_metrics.py --gen-dir "${gen}" --output-json "${out}" --tag "${tag}" "${EXTRA[@]}"
}

twoway () {
  local gen="$1"; local tag="$2"
  local out="${MET_DIR}/${tag}_2wc.json"
  [[ -f "${out}" ]] && echo "[SKIP] 2wc ${tag}" && return 0
  [[ -d "${gen}" ]] || return 0
  python scripts/erdc_twoway_metrics.py --gen-dir "${gen}" --output-json "${out}" --tag "${tag}" "${EXTRA[@]}"
}

bootstrap_ci () {
  local gen="$1"; local tag="$2"
  local out="${MET_DIR}/${tag}_bootstrap.json"
  [[ -f "${out}" ]] && echo "[SKIP] boot ${tag}" && return 0
  [[ -d "${gen}" ]] || return 0
  python scripts/erdc_bootstrap_ci.py --gen-dir "${gen}" --output-json "${out}" --tag "${tag}" "${EXTRA[@]}"
}

echo "################ [1] Reference metrics + FID ################"
for tag_gen in \
  "official_atm_gen:outputs/erdc/w7_official_flat" \
  "w12_atm_brain:outputs/erdc/w12_official_atm_img_sub08/selected_brain" \
  "w13_merged_fuse_l0p15:outputs/erdc/w13_merged_fuse_l0p15/selected_fused" \
  "w13_merged_brain:outputs/erdc/w13_merged_fuse_l0p15/selected_brain" \
  "w10_sdxl_fuse:outputs/erdc/w10_fuse_lowstr_l0p35_sub08/selected_fused"; do
  tag="${tag_gen%%:*}"
  gen="${tag_gen##*:}"
  full_metrics "${gen}" "${tag}"
  fid_metrics "${gen}" "${tag}"
done

full_metrics outputs/erdc/w15_mega_fuse_l0p10/selected_brain w15_mega_brain
fid_metrics outputs/erdc/w15_mega_fuse_l0p10/selected_brain w15_mega_brain
for lam in 0.10 0.15 0.20; do
  tag="w15_mega_fuse_l${lam/./p}"
  full_metrics "outputs/erdc/${tag}/selected_fused" "${tag}"
  fid_metrics "outputs/erdc/${tag}/selected_fused" "${tag}"
done
for tk in 2 3; do
  tag="w15_mega_topk${tk}_l0p15"
  full_metrics "outputs/erdc/${tag}/selected_topk_struct" "${tag}"
  fid_metrics "outputs/erdc/${tag}/selected_topk_struct" "${tag}"
done
if [[ -d outputs/erdc/w15_dual_atm_bit_pair/fuse_run/selected_fused ]]; then
  full_metrics outputs/erdc/w15_dual_atm_bit_pair/fuse_run/selected_fused w15_dual_atm_bit_fuse
  fid_metrics outputs/erdc/w15_dual_atm_bit_pair/fuse_run/selected_fused w15_dual_atm_bit_fuse
fi

echo "################ [2] 2WC ################"
twoway outputs/erdc/w7_official_flat official_atm_gen
twoway outputs/erdc/w12_official_atm_img_sub08/selected_brain w12_atm_brain
twoway outputs/erdc/w13_merged_fuse_l0p15/selected_fused w13_merged_fuse_l0p15
twoway outputs/erdc/w13_merged_fuse_l0p15/selected_brain w13_merged_brain
twoway outputs/erdc/w15_mega_fuse_l0p15/selected_fused w15_mega_fuse_l0p15
twoway outputs/erdc/w15_mega_fuse_l0p10/selected_brain w15_mega_brain
if [[ -d outputs/erdc/w15_dual_atm_bit_pair/fuse_run/selected_fused ]]; then
  twoway outputs/erdc/w15_dual_atm_bit_pair/fuse_run/selected_fused w15_dual_atm_bit_fuse
fi
for mode in retrieve shuffle misalign zero; do
  twoway "outputs/erdc/w14_merged_ctrl_${mode}/selected_fused" "w14_merged_ctrl_${mode}"
done

echo "################ [3] Bootstrap ################"
bootstrap_ci outputs/erdc/w7_official_flat official_atm_gen
bootstrap_ci outputs/erdc/w13_merged_fuse_l0p15/selected_fused w13_merged_fuse_l0p15
bootstrap_ci outputs/erdc/w15_mega_fuse_l0p15/selected_fused w15_mega_fuse_l0p15
bootstrap_ci outputs/erdc/w15_mega_fuse_l0p10/selected_brain w15_mega_brain
bootstrap_ci outputs/erdc/w12_official_atm_img_sub08/selected_brain w12_atm_brain
if [[ -d outputs/erdc/w15_dual_atm_bit_pair/fuse_run/selected_fused ]]; then
  bootstrap_ci outputs/erdc/w15_dual_atm_bit_pair/fuse_run/selected_fused w15_dual_atm_bit_fuse
fi

echo "################ [4] Panels ################"
python scripts/erdc_qualitative_panel.py \
  --b0-dir outputs/erdc/w7_official_flat \
  --random-dir outputs/erdc/w12_official_bit_img_sub08/selected_random \
  --brain-dir outputs/erdc/w13_merged_fuse_l0p15/selected_fused \
  --scores-npy outputs/erdc/w13_merged_fuse_l0p15/fused_scores.npy \
  --output-dir outputs/erdc/w16_panels_main_fuse \
  --n-panel 24 || true
python scripts/erdc_qualitative_panel.py \
  --b0-dir outputs/erdc/w7_official_flat \
  --random-dir outputs/erdc/w12_official_bit_img_sub08/selected_random \
  --brain-dir outputs/erdc/w15_mega_fuse_l0p10/selected_brain \
  --scores-npy outputs/erdc/w15_mega_fuse_l0p10/brain_scores.npy \
  --output-dir outputs/erdc/w16_panels_mega_brain \
  --n-panel 24 || true

echo "################ [5] Tables ################"
python scripts/erdc_paper_table.py \
  --metrics-dirs outputs/erdc/w15_metrics outputs/erdc/w14_metrics outputs/erdc/w12_metrics \
  --tags official_atm_gen w12_atm_brain w13_merged_fuse_l0p15 w13_merged_brain \
         w15_mega_brain w15_mega_fuse_l0p15 w15_mega_fuse_l0p20 \
         w15_mega_topk2_l0p15 w15_dual_atm_bit_fuse w10_sdxl_fuse \
         w14_merged_ctrl_shuffle w14_merged_ctrl_zero \
  --output-md outputs/erdc/paper_main_table.md \
  --output-tex outputs/erdc/paper_main_table.tex

python scripts/erdc_write_paper_freeze.py

echo "[OK] paper finalize script done"
