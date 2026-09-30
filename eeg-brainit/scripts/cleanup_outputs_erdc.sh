#!/usr/bin/env bash
# Safe cleanup: remove intermediate ERDC runs & heavy caches.
# Preserves paper tables, final images, metrics, key ckpts, embed npy.
set -euo pipefail
ROOT="$(cd "$(dirname "${0}")/.." && pwd)"
OUT="${ROOT}/outputs"
LOG="${OUT}/CLEANUP_LOG_20260823.txt"
exec >>"$LOG" 2>&1

echo ""
echo "===== SAFE cleanup phase 2 $(date -Iseconds) ====="
echo "Before: $(du -sh "$OUT" | awk '{print $1}')"

del() {
  local p="$1"
  if [[ -e "$p" ]]; then
    echo "DELETE $p ($(du -sh "$p" 2>/dev/null | awk '{print $1}'))"
    rm -rf "$p"
  fi
}

# --- Global: candidates + twoway caches (never needed after fuse) ---
while IFS= read -r d; do del "$d"; done < <(find "$OUT/erdc" -type d -name candidates 2>/dev/null)
while IFS= read -r d; do del "$d"; done < <(find "$OUT/erdc" -type d -name _twoway_cache 2>/dev/null)

trim_sel() {
  local base="$1" keep="$2"
  [[ -d "$base" ]] || return 0
  for d in "$base"/selected_*; do
    [[ -d "$d" ]] || continue
    [[ "$(basename "$d")" == "$keep" ]] || del "$d"
  done
}

trim_sel "$OUT/erdc/w13_merged_fuse_l0p15" "selected_fused"
trim_sel "$OUT/erdc/w12_official_atm_img_sub08" "selected_brain"
trim_sel "$OUT/erdc/w15_mega_fuse_l0p15" "selected_fused"
trim_sel "$OUT/erdc/w15_mega_fuse_l0p10" "selected_brain"

# --- ERDC whitelist (paper + qualitative) ---
ERDC_KEEP=(
  paper_main_table.md paper_main_table.tex
  paper_ten_subject_table.md paper_ten_subject_table.tex
  w16_paper_freeze.md
  w7_official_flat
  w12_official_atm_img_sub08
  w13_merged_fuse_l0p15
  w15_mega_fuse_l0p15
  w15_mega_fuse_l0p10
  w15_metrics
  w16_loso_metrics
  w16_panels_gt_vs_recon
  w16_panels_comparison_main
  w16_panels_main_fuse
  w16_panels_mega_brain
)

for item in "$OUT/erdc"/*; do
  [[ -e "$item" ]] || continue
  base="$(basename "$item")"
  keep=0
  for k in "${ERDC_KEEP[@]}"; do [[ "$base" == "$k" ]] && keep=1 && break; done
  if [[ $keep -eq 0 ]]; then del "$item"; fi
done

# --- outputs/ whitelist ---
NON_ERDC_KEEP=(
  atm_distill_s1_sub08
  atm_distill_s3_sub08
  atm_bridge
  eval
  CLEANUP_LOG_20260821.txt
  CLEANUP_LOG_20260823.txt
  顶会主路线_现行.txt
  顶会方案_EEG2Image重建_ERDC.txt
)

for item in "$OUT"/*; do
  [[ "$item" == "$OUT/erdc" ]] && continue
  base="$(basename "$item")"
  keep=0
  for k in "${NON_ERDC_KEEP[@]}"; do [[ "$base" == "$k" ]] && keep=1 && break; done
  if [[ $keep -eq 0 ]]; then del "$item"; fi
done

# --- Trim eval: keep npy/json, drop generated PNG dirs ---
del "$OUT/eval/atm_pipeline_sub08/generated"
del "$OUT/eval/atm_official_gen_sub08"
find "$OUT/eval" -type d -name generated -prune -exec rm -rf {} + 2>/dev/null || true

# --- Trim atm_bridge generated if any ---
find "$OUT/atm_bridge" -type d -name generated -prune -exec rm -rf {} + 2>/dev/null || true

echo "After: $(du -sh "$OUT" | awk '{print $1}')"
echo "===== kept erdc ====="
ls -1 "$OUT/erdc"
echo "===== kept outputs top ====="
ls -1 "$OUT"
echo "===== ckpt check ====="
ls -lh "$OUT/atm_distill_s3_sub08/checkpoints/" 2>/dev/null || true
ls -lh "$OUT/atm_distill_s1_sub08/checkpoints/" 2>/dev/null || true
test -f "$ROOT/checkpoints/atm_diffusion_prior/sub-08/diffusion_prior.pt" && echo "prior sub-08 OK"
echo "===== done ====="
