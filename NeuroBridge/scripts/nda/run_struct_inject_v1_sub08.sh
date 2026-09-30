#!/usr/bin/env bash
# Phase A structure injection (sub-08): ATM/CogCap-style first-class conditioning.
# - Timed Depth/Canny ControlNet (early layout) + HCMA IP/prompts (late semantics)
# - Optional mild Pc/LL img2img init (ATM latent start) under the same timed CN
# Frozen SDXL. Official seven + semantic gate; maximize SSIM then PixCorr.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/struct_inject_v1/sub-08}"
HCMA10="${HCMA10:-${NB_ROOT}/outputs/hcma_10subj}"
OVN="${OVN:-${NB_ROOT}/outputs/overnight_struct_v2/sub-08}"
DUAL="${DUAL:-${NB_ROOT}/outputs/dual_ctrl_struct/sub-08}"
TCDA="${TCDA:-${NB_ROOT}/outputs/tcda/sub-08}"
LL="${LL:-${NB_ROOT}/outputs/lowlevel_decoder/sub-08}"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

SEM="${HCMA10}/sub-08/generation/hcma_full_a40/generated"
EMB="${HCMA10}/sub-08/ft/embeds/blend_nda_cfm_f_a40_test.npy"
PROMPT="${HCMA10}/prompts/prompts_full_hcma_test.json"
DEPTH="${OVN}/depth/depth_head/pred_depth_rgb_512"
PC="${TCDA}/train/pred_pc_rgb_512"
LL_RGB="${LL}/vae_head/pred_lowlevel_rgb_512"
U_STR="${OVN}/depth/depth_head/u_str.npy"
U_ALEX="${NB_ROOT}/outputs/alex2_sota_official/sub-08/alex_mid/u_alex.npy"

mkdir -p "${OUT}/generation" "${OUT}/metrics" "${OUT}/schedules" "${OUT}/logs" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" || true
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export DEVICE OUT NB_ROOT IMAGES_ROOT

echo "{\"pipeline\":\"struct_inject_v1\",\"started\":\"$(date -Iseconds)\",\"goal\":\"SSIM≥0.28 Pix≥0.15 under semantic gate via timed CN + latent init\"}" \
  > "${OUT}/job_running.json"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
require "${SEM}/000.png"; require "${EMB}"; require "${PROMPT}"
require "${DEPTH}/000.png"; require "${PC}/000.png"; require "${LL_RGB}/000.png"
require "${U_STR}"; require "${U_ALEX}"

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
if [[ -d "${DUAL}/generation/rec_i2i_ll_gate_ualex_a070/generated" ]]; then
  link_tag "dual_rec_ll_gate_a070" "${DUAL}/generation/rec_i2i_ll_gate_ualex_a070/generated"
fi
ALEX048=/project/peilab/why/NeuroBridge/outputs/alex2_first/sub-08/generation/alex_luma_pc_a048/generated
[[ -d "${ALEX048}" ]] && link_tag "alex_luma_pc_a048" "${ALEX048}"

# ---------- [1] schedules ----------
echo "===== [1] schedules @ $(date -Iseconds) ====="
CN_GATE="${OUT}/schedules/cn_ustr_025_055.npy"
S_GATE="${OUT}/schedules/strength_ualex_022_034.npy"
if [[ ! -f "${CN_GATE}" ]]; then
  "${PYTHON}" scripts/nda/build_gated_cn_scale.py \
    --u-npy "${U_STR}" --output-npy "${CN_GATE}" \
    --cn-min 0.25 --cn-max 0.55 \
    --report-json "${OUT}/schedules/cn_ustr_report.json"
fi
if [[ ! -f "${S_GATE}" ]]; then
  "${PYTHON}" scripts/nda/build_gated_img2img_strength.py \
    --u-npy "${U_ALEX}" --output-npy "${S_GATE}" \
    --s-min 0.22 --s-max 0.34 \
    --report-json "${OUT}/schedules/strength_ualex_report.json"
fi

run_inj() {
  local tag="$1"; shift
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; return 0; fi
  "${PYTHON}" scripts/nda/generate_struct_inject_decode.py \
    --embed-npy "${EMB}" \
    --prompts-json "${PROMPT}" \
    --output-dir "${gdir}" \
    --tag "${tag}" \
    --seed 42 \
    --gen-steps 28 \
    --gen-guidance 5.0 \
    --ip-scale 1.0 \
    --skip-metrics \
    "$@"
}

# ---------- [2] timed Depth-CN from noise (CogCap-like spatial inject) ----------
echo "===== [2] timed depth CN txt2img @ $(date -Iseconds) ====="
# early CN only; keep scale mild so IP can finish semantics
run_inj "t2i_depth_cn040_end040" --mode txt2img --control-type depth \
  --cond-dir "${DEPTH}" --cn-scale 0.40 \
  --control-guidance-start 0.0 --control-guidance-end 0.40
run_inj "t2i_depth_cn045_end035" --mode txt2img --control-type depth \
  --cond-dir "${DEPTH}" --cn-scale 0.45 \
  --control-guidance-start 0.0 --control-guidance-end 0.35
run_inj "t2i_depth_cn050_end050" --mode txt2img --control-type depth \
  --cond-dir "${DEPTH}" --cn-scale 0.50 \
  --control-guidance-start 0.0 --control-guidance-end 0.50
run_inj "t2i_depth_cngate_end040" --mode txt2img --control-type depth \
  --cond-dir "${DEPTH}" --cn-scale 0.40 --cn-scale-npy "${CN_GATE}" \
  --control-guidance-start 0.0 --control-guidance-end 0.40

# ---------- [3] timed Canny from Pc (edge layout without depth head noise) ----------
echo "===== [3] timed canny-Pc @ $(date -Iseconds) ====="
run_inj "t2i_canny_pc_cn045_end040" --mode txt2img --control-type canny \
  --cond-dir "${PC}" --cn-scale 0.45 \
  --control-guidance-start 0.0 --control-guidance-end 0.40

# ---------- [4] ATM latent init + timed Depth-CN (dual first-class paths) ----------
echo "===== [4] img2img init + timed depth CN @ $(date -Iseconds) ====="
run_inj "i2i_pc_s028_depth_cn035_end035" --mode img2img --control-type depth \
  --cond-dir "${DEPTH}" --init-dir "${PC}" --strength 0.28 --cn-scale 0.35 \
  --control-guidance-start 0.0 --control-guidance-end 0.35
run_inj "i2i_pc_s032_depth_cn030_end030" --mode img2img --control-type depth \
  --cond-dir "${DEPTH}" --init-dir "${PC}" --strength 0.32 --cn-scale 0.30 \
  --control-guidance-start 0.0 --control-guidance-end 0.30
run_inj "i2i_ll_s028_depth_cn035_end035" --mode img2img --control-type depth \
  --cond-dir "${DEPTH}" --init-dir "${LL_RGB}" --strength 0.28 --cn-scale 0.35 \
  --control-guidance-start 0.0 --control-guidance-end 0.35
run_inj "i2i_pc_sgate_depth_cngate_end035" --mode img2img --control-type depth \
  --cond-dir "${DEPTH}" --init-dir "${PC}" \
  --strength 0.28 --strength-npy "${S_GATE}" \
  --cn-scale 0.35 --cn-scale-npy "${CN_GATE}" \
  --control-guidance-start 0.0 --control-guidance-end 0.35

# ---------- [5] official seven ----------
echo "===== [5] official seven @ $(date -Iseconds) ====="
for d in "${OUT}/generation"/*; do
  [[ -d "$d" ]] || continue
  tag="$(basename "$d")"
  gen="${d}/generated"
  [[ -f "${gen}/199.png" ]] || continue
  outj="${OUT}/metrics/${tag}_seven.json"
  if [[ -f "${outj}" ]]; then echo "[SKIP] seven ${tag}"; continue; fi
  "${PYTHON}" scripts/nda/eval_official_seven_dir.py \
    --gen-dir "${gen}" \
    --output-json "${outj}" \
    --tag "${tag}" \
    --images-root "${IMAGES_ROOT}" \
    --device "${DEVICE}" \
    --batch-size 16
  find "${d}" -type d -name '_twoway_cache' -exec rm -rf {} + 2>/dev/null || true
done
find "${OUT}" -type d -name '_twoway_cache' -exec rm -rf {} + 2>/dev/null || true

# ---------- [6] gate + rank ----------
echo "===== [6] rank @ $(date -Iseconds) ====="
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
        "delta_clip": float(r["clip"]) - float(ref["clip"]),
        "hit_ssim_028": float(r["ssim"]) >= 0.28,
        "hit_pix_015": float(r["pixcorr"]) >= 0.15,
    })
ranked.sort(key=lambda x: (x["pass_gate"], x["ssim"], x["pixcorr"], x["alex2"]), reverse=True)
gated = [r for r in ranked if r["pass_gate"]]
best = gated[0] if gated else ranked[0]
summary = {
  "pipeline": "struct_inject_v1",
  "method": "timed Depth/Canny-CN early + HCMA IP late; optional ATM mild Pc/LL latent init; frozen SDXL",
  "targets": {"ssim": 0.28, "pixcorr": 0.15},
  "selection": "max SSIM then Pix under CLIP/A5/Inc/SwAV/FID gate",
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
  "# Structure inject v1 (timed CN + ATM latent init) — sub-08",
  "",
  "Inspired by ATM (VAE/low-level img2img init) and CogCap (depth spatial inject via CN/Layout).",
  "CN only early (`control_guidance_end`); IP+HCMA prompts own late denoising. No RGB luma recovery as primary.",
  f"Ref: SSIM={ref['ssim']:.3f} Pix={ref['pixcorr']:.3f} CLIP={ref['clip']:.3f} FID={ref['fid']:.1f}",
  "Targets: SSIM≥0.28, PixCorr≥0.15 under semantic gate.",
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
(out / "STRUCT_INJECT_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"best_gated": summary["best_gated"], "n_pass": len(gated),
                  "n_hit_both": summary["n_hit_both_targets_gated"]}, indent=2))
PY

echo "{\"pipeline\":\"struct_inject_v1\",\"finished\":\"$(date -Iseconds)\",\"best\":\"$(cat "${OUT}/best_tag.txt")\"}" > "${OUT}/job_done.json"
echo "[DONE] ${OUT}/STRUCT_INJECT_TABLE.md"
