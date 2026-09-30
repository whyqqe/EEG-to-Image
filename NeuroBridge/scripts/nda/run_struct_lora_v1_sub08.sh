#!/usr/bin/env bash
# Phase B: SDXL UNet LoRA (structure from blur-init + IP distill) → EEG decode eval.
# Goal: SSIM≥0.28 & Pix≥0.15 under semantic gate (CLIP/A5/Inc/SwAV/FID).
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/struct_lora_v1/sub-08}"
HCMA10="${HCMA10:-${NB_ROOT}/outputs/hcma_10subj}"
OVN="${OVN:-${NB_ROOT}/outputs/overnight_struct_v2/sub-08}"
TCDA="${TCDA:-${NB_ROOT}/outputs/tcda/sub-08}"
LL="${LL:-${NB_ROOT}/outputs/lowlevel_decoder/sub-08}"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

SEM="${HCMA10}/sub-08/generation/hcma_full_a40/generated"
EMB="${HCMA10}/sub-08/ft/embeds/blend_nda_cfm_f_a40_test.npy"
PROMPT="${HCMA10}/prompts/prompts_full_hcma_test.json"
PC="${TCDA}/train/pred_pc_rgb_512"
LL_RGB="${LL}/vae_head/pred_lowlevel_rgb_512"
DEPTH="${OVN}/depth/depth_head/pred_depth_rgb_512"

mkdir -p "${OUT}/generation" "${OUT}/metrics" "${OUT}/logs" "${OUT}/lora" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" || true
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export DEVICE OUT NB_ROOT IMAGES_ROOT

echo "{\"pipeline\":\"struct_lora_v1\",\"started\":\"$(date -Iseconds)\"}" > "${OUT}/job_running.json"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
require "${SEM}/000.png"; require "${EMB}"; require "${PROMPT}"
require "${PC}/000.png"; require "${LL_RGB}/000.png"

link_tag() {
  local tag="$1" src="$2"
  local dst="${OUT}/generation/${tag}"
  mkdir -p "${dst}"
  [[ -e "${dst}/generated" ]] || ln -sfn "${src}" "${dst}/generated"
  echo "{\"tag\":\"${tag}\",\"source\":\"${src}\"}" > "${dst}/metrics.json"
}

# ---------- [0] baselines ----------
echo "===== [0] baselines @ $(date -Iseconds) ====="
link_tag "ref_hcma_full_a40" "${SEM}"
[[ -d "${OVN}/generation/combo_d40_luma_pc_a060/generated" ]] && \
  link_tag "combo_d40_luma_pc_a060" "${OVN}/generation/combo_d40_luma_pc_a060/generated"
ALEX048=/project/peilab/why/NeuroBridge/outputs/alex2_first/sub-08/generation/alex_luma_pc_a048/generated
[[ -d "${ALEX048}" ]] && link_tag "alex_luma_pc_a048" "${ALEX048}"

# ---------- [1] train LoRA ----------
echo "===== [1] train LoRA @ $(date -Iseconds) ====="
LORA_ROOT="${OUT}/lora"
if [[ ! -f "${LORA_ROOT}/lora_final.txt" ]]; then
  "${PYTHON}" scripts/nda/train_sdxl_struct_lora.py \
    --output-dir "${LORA_ROOT}" \
    --images-root "${IMAGES_ROOT}" \
    --n-train 2048 \
    --steps 2500 \
    --batch-size 1 \
    --lr 1e-4 \
    --lora-rank 8 \
    --lora-alpha 8 \
    --blur-radius 6.0 \
    --strength-min 0.26 \
    --strength-max 0.42 \
    --noise-frac 0.25 \
    --lambda-distill 0.35 \
    --save-every 500 \
    --seed 42
fi
LORA_DIR="$(cat "${LORA_ROOT}/lora_final.txt")"
require "${LORA_DIR}"
echo "[OK] LoRA=${LORA_DIR}"

# ---------- [2] decode with LoRA ----------
echo "===== [2] generate @ $(date -Iseconds) ====="

run_ip() {
  local tag="$1"; shift
  local gdir="${OUT}/generation/${tag}"
  [[ -f "${gdir}/generated/199.png" ]] && echo "[SKIP] ${tag}" && return 0
  "${PYTHON}" scripts/nda/generate_ip_txt2img.py \
    --embed-npy "${EMB}" --prompts-json "${PROMPT}" \
    --output-dir "${gdir}" --tag "${tag}" --seed 42 \
    --gen-steps 28 --gen-guidance 5.0 "$@"
}

run_i2i() {
  local tag="$1"; shift
  local gdir="${OUT}/generation/${tag}"
  [[ -f "${gdir}/generated/199.png" ]] && echo "[SKIP] ${tag}" && return 0
  "${PYTHON}" scripts/nda/generate_lowlevel_decode.py \
    --mode img2img --embed-npy "${EMB}" --prompts-json "${PROMPT}" \
    --output-dir "${gdir}" --tag "${tag}" --seed 42 \
    --gen-steps 28 --gen-guidance 5.0 --ip-scale 1.0 --skip-metrics "$@"
}

run_inj() {
  local tag="$1"; shift
  local gdir="${OUT}/generation/${tag}"
  [[ -f "${gdir}/generated/199.png" ]] && echo "[SKIP] ${tag}" && return 0
  "${PYTHON}" scripts/nda/generate_struct_inject_decode.py \
    --embed-npy "${EMB}" --prompts-json "${PROMPT}" \
    --output-dir "${gdir}" --tag "${tag}" --seed 42 \
    --gen-steps 28 --gen-guidance 5.0 --ip-scale 1.0 --skip-metrics "$@"
}

# semantic check: IP from noise + LoRA (should stay near HCMA)
run_ip "t2i_ip_lora100" --lora-dir "${LORA_DIR}" --lora-scale 1.0
run_ip "t2i_ip_lora070" --lora-dir "${LORA_DIR}" --lora-scale 0.7

# ATM-style mild init + LoRA (main bet)
run_i2i "i2i_pc_s028_lora100" --lowlevel-dir "${PC}" --strength 0.28 \
  --lora-dir "${LORA_DIR}" --lora-scale 1.0
run_i2i "i2i_pc_s032_lora100" --lowlevel-dir "${PC}" --strength 0.32 \
  --lora-dir "${LORA_DIR}" --lora-scale 1.0
run_i2i "i2i_pc_s028_lora070" --lowlevel-dir "${PC}" --strength 0.28 \
  --lora-dir "${LORA_DIR}" --lora-scale 0.7
run_i2i "i2i_ll_s028_lora100" --lowlevel-dir "${LL_RGB}" --strength 0.28 \
  --lora-dir "${LORA_DIR}" --lora-scale 1.0

# no-LoRA control at same strength (confirm LoRA effect)
run_i2i "i2i_pc_s028_nolor" --lowlevel-dir "${PC}" --strength 0.28

# timed canny + LoRA (Phase A winner path + LoRA)
run_inj "t2i_canny_pc_cn045_end040_lora100" --mode txt2img --control-type canny \
  --cond-dir "${PC}" --cn-scale 0.45 \
  --control-guidance-start 0.0 --control-guidance-end 0.40 \
  --lora-dir "${LORA_DIR}" --lora-scale 1.0

# mild i2i + timed depth CN + LoRA
if [[ -d "${DEPTH}" ]]; then
  run_inj "i2i_pc_s028_depth_cn030_end030_lora100" --mode img2img --control-type depth \
    --cond-dir "${DEPTH}" --init-dir "${PC}" --strength 0.28 --cn-scale 0.30 \
    --control-guidance-start 0.0 --control-guidance-end 0.30 \
    --lora-dir "${LORA_DIR}" --lora-scale 1.0
fi

# ---------- [3] official seven ----------
echo "===== [3] official seven @ $(date -Iseconds) ====="
for d in "${OUT}/generation"/*; do
  [[ -d "$d" ]] || continue
  tag="$(basename "$d")"
  gen="${d}/generated"
  [[ -f "${gen}/199.png" ]] || continue
  outj="${OUT}/metrics/${tag}_seven.json"
  [[ -f "${outj}" ]] && echo "[SKIP] seven ${tag}" && continue
  "${PYTHON}" scripts/nda/eval_official_seven_dir.py \
    --gen-dir "${gen}" --output-json "${outj}" --tag "${tag}" \
    --images-root "${IMAGES_ROOT}" --device "${DEVICE}" --batch-size 16
  find "${d}" -type d -name '_twoway_cache' -exec rm -rf {} + 2>/dev/null || true
done
find "${OUT}" -type d -name '_twoway_cache' -exec rm -rf {} + 2>/dev/null || true

# ---------- [4] gate + rank ----------
echo "===== [4] rank @ $(date -Iseconds) ====="
"${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT"])
rows = [json.loads(p.read_text()) for p in sorted((out / "metrics").glob("*_seven.json"))]
ref = next(r for r in rows if r["tag"] == "ref_hcma_full_a40")
ranked = []
for r in rows:
    gate = (
        float(r["clip"]) >= float(ref["clip"]) - 0.015
        and float(r["alex5"]) >= float(ref["alex5"]) - 0.015
        and float(r["inception"]) >= float(ref["inception"]) - 0.015
        and float(r["swav"]) <= float(ref["swav"]) + 0.025
        and float(r["fid"]) <= float(ref["fid"]) + 20
    )
    ranked.append({
        **{k: r[k] for k in ["tag", "pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "swav", "fid"]},
        "pass_gate": gate,
        "delta_ssim": float(r["ssim"]) - float(ref["ssim"]),
        "delta_pix": float(r["pixcorr"]) - float(ref["pixcorr"]),
        "hit_ssim_028": float(r["ssim"]) >= 0.28,
        "hit_pix_015": float(r["pixcorr"]) >= 0.15,
    })
ranked.sort(key=lambda x: (x["pass_gate"], x["ssim"], x["pixcorr"], x["alex2"]), reverse=True)
gated = [r for r in ranked if r["pass_gate"]]
best = gated[0] if gated else ranked[0]
summary = {
  "pipeline": "struct_lora_v1",
  "method": "SDXL UNet LoRA (blur-init structure + IP + base distill); EEG IP + Pc/LL at infer",
  "targets": {"ssim": 0.28, "pixcorr": 0.15},
  "ref": {k: ref[k] for k in ["tag", "pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "swav", "fid"]},
  "best_gated": best if best.get("pass_gate") else None,
  "best_overall_ssim": max(ranked, key=lambda x: x["ssim"]),
  "n_pass": len(gated),
  "n_hit_both_targets_gated": sum(1 for r in gated if r["hit_ssim_028"] and r["hit_pix_015"]),
  "all_ranked": ranked,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(out / "best_tag.txt").write_text(best["tag"] + "\n", encoding="utf-8")
lines = [
  "# Structure LoRA v1 (Phase B) — sub-08",
  "",
  "Train: THINGS train GT, IP=CLIP(GT), init=blur(GT) SDEdit + distill to frozen UNet.",
  "Infer: EEG HCMA IP/prompts + Pc/LL init and/or timed CN + LoRA.",
  f"Ref: SSIM={ref['ssim']:.3f} Pix={ref['pixcorr']:.3f} CLIP={ref['clip']:.3f} FID={ref['fid']:.1f}",
  "Targets: SSIM≥0.28, Pix≥0.15 under semantic gate.",
  "",
  "| tag | pass | SSIM≥.28 | Pix≥.15 | SSIM | ΔS | Pix | ΔP | A2 | CLIP | A5 | FID |",
  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in ranked:
    lines.append(
      f"| `{r['tag']}` | {int(r['pass_gate'])} | {int(r['hit_ssim_028'])} | {int(r['hit_pix_015'])} | "
      f"{r['ssim']:.3f} | {r['delta_ssim']:+.3f} | {r['pixcorr']:.3f} | {r['delta_pix']:+.3f} | "
      f"{r['alex2']:.3f} | {r['clip']:.3f} | {r['alex5']:.3f} | {r['fid']:.1f} |"
    )
(out / "STRUCT_LORA_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"best_gated": summary["best_gated"], "n_pass": len(gated),
                  "n_hit_both": summary["n_hit_both_targets_gated"]}, indent=2))
PY

echo "{\"pipeline\":\"struct_lora_v1\",\"finished\":\"$(date -Iseconds)\",\"best\":\"$(cat "${OUT}/best_tag.txt")\",\"lora\":\"${LORA_DIR}\"}" \
  > "${OUT}/job_done.json"
echo "[DONE] ${OUT}/STRUCT_LORA_TABLE.md"
