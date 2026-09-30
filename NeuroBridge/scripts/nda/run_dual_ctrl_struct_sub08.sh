#!/usr/bin/env bash
# Dual-control structure chase (sub-08): semantic IP + gated low-level img2img,
# then semantic-recovery luma fuse. Highest-probability path to lift PixCorr/SSIM
# without collapsing HCMA CLIP/Alex5/Inc/SwAV/FID.
#
# Frozen SDXL (no UNet FT). Eval = official seven; select by SSIM/Pix under semantic gate.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/dual_ctrl_struct/sub-08}"
HCMA10="${HCMA10:-${NB_ROOT}/outputs/hcma_10subj}"
A2O="${A2O:-${NB_ROOT}/outputs/alex2_sota_official/sub-08}"
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
U_STR="${OVN}/depth/depth_head/u_str.npy"
U_ALEX="${A2O}/alex_mid/u_alex.npy"

mkdir -p "${OUT}/generation" "${OUT}/metrics" "${OUT}/schedules" "${OUT}/logs" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" || true
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export DEVICE OUT NB_ROOT IMAGES_ROOT

echo "{\"pipeline\":\"dual_ctrl_struct\",\"started\":\"$(date -Iseconds)\",\"goal\":\"SSIM/Pix↑ under semantic gate; frozen SDXL\"}" > "${OUT}/job_running.json"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
require "${SEM}/000.png"; require "${EMB}"; require "${PROMPT}"
require "${PC}/000.png"; require "${LL_RGB}/000.png"
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
# prior best deployable structure-preserving candidate
if [[ -d "${OVN}/generation/combo_d40_luma_pc_a060/generated" ]]; then
  link_tag "combo_d40_luma_pc_a060" "${OVN}/generation/combo_d40_luma_pc_a060/generated"
fi
if [[ -d "${A2O}/generation/alex_luma_pc_a048/generated" ]] || [[ -d /project/peilab/why/NeuroBridge/outputs/alex2_first/sub-08/generation/alex_luma_pc_a048/generated ]]; then
  SRC048=/project/peilab/why/NeuroBridge/outputs/alex2_first/sub-08/generation/alex_luma_pc_a048/generated
  [[ -d "${SRC048}" ]] && link_tag "alex_luma_pc_a048" "${SRC048}"
fi

# ---------- [1] strength schedules (mild band; high conf → low strength) ----------
echo "===== [1] schedules @ $(date -Iseconds) ====="
S_ALEX="${OUT}/schedules/strength_ualex_020_036.npy"
S_STR="${OUT}/schedules/strength_ustr_020_036.npy"
if [[ ! -f "${S_ALEX}" ]]; then
  "${PYTHON}" scripts/nda/build_gated_img2img_strength.py \
    --u-npy "${U_ALEX}" --output-npy "${S_ALEX}" \
    --s-min 0.20 --s-max 0.36 \
    --report-json "${OUT}/schedules/strength_ualex_report.json"
fi
if [[ ! -f "${S_STR}" ]]; then
  "${PYTHON}" scripts/nda/build_gated_img2img_strength.py \
    --u-npy "${U_STR}" --output-npy "${S_STR}" \
    --s-min 0.20 --s-max 0.36 \
    --report-json "${OUT}/schedules/strength_ustr_report.json"
fi

run_i2i() {
  local tag="$1" ll="$2" extra=()
  shift 2
  extra=("$@")
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; return 0; fi
  "${PYTHON}" scripts/nda/generate_lowlevel_decode.py \
    --mode img2img \
    --lowlevel-dir "${ll}" \
    --embed-npy "${EMB}" \
    --prompts-json "${PROMPT}" \
    --ip-scale 1.0 \
    --gen-steps 28 \
    --gen-guidance 5.0 \
    --output-dir "${gdir}" \
    --tag "${tag}" \
    --seed 42 \
    --skip-metrics \
    "${extra[@]}"
}

# ---------- [2] gated + mild fixed img2img (dual control: IP semantic + LL/Pc layout) ----------
echo "===== [2] gated/mild img2img @ $(date -Iseconds) ====="
# Fixed mild strengths (safer than failed 0.35–0.55 band)
for s in 0.22 0.28 0.32; do
  run_i2i "i2i_pc_s$(echo $s | tr -d .)" "${PC}" --strength "$s"
done
for s in 0.24 0.30; do
  run_i2i "i2i_ll_s$(echo $s | tr -d .)" "${LL_RGB}" --strength "$s"
done
# Per-sample gated strength
run_i2i "i2i_pc_gate_ualex" "${PC}" --strength 0.28 --strength-npy "${S_ALEX}"
run_i2i "i2i_pc_gate_ustr" "${PC}" --strength 0.28 --strength-npy "${S_STR}"
run_i2i "i2i_ll_gate_ualex" "${LL_RGB}" --strength 0.28 --strength-npy "${S_ALEX}"

# ---------- [3] semantic-recovery luma (critical): blend i2i structure back toward HCMA ----------
echo "===== [3] semantic recovery fuse @ $(date -Iseconds) ====="
run_rec() {
  local tag="$1" struct_tag="$2" alpha="$3"
  local struct="${OUT}/generation/${struct_tag}/generated"
  local gdir="${OUT}/generation/${tag}"
  [[ -f "${struct}/199.png" ]] || { echo "[WARN] skip ${tag}: missing ${struct_tag}"; return 0; }
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; return 0; fi
  "${PYTHON}" scripts/nda/generate_luma_fuse.py \
    --struct-dir "${struct}" \
    --semantic-dir "${SEM}" \
    --output-dir "${gdir}" \
    --tag "${tag}" \
    --sem-alpha "${alpha}"
}

# Recover semantics from the most promising mild/gated i2i parents
for parent in i2i_pc_s028 i2i_pc_s032 i2i_pc_gate_ualex i2i_ll_s030 i2i_ll_gate_ualex; do
  for a in 0.70 0.80 0.88; do
    run_rec "rec_${parent}_a$(echo $a | tr -d .)" "${parent}" "$a"
  done
done

# ---------- [4] official seven ----------
echo "===== [4] official seven @ $(date -Iseconds) ====="
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

# ---------- [5] gate: protect semantics; maximize SSIM then PixCorr ----------
echo "===== [5] rank @ $(date -Iseconds) ====="
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
        "delta_alex2": float(r["alex2"]) - float(ref["alex2"]),
    })
# structure-first among gated
ranked.sort(key=lambda x: (x["pass_gate"], x["ssim"], x["pixcorr"], x["alex2"]), reverse=True)
gated = [r for r in ranked if r["pass_gate"]]
best = gated[0] if gated else ranked[0]
summary = {
  "pipeline": "dual_ctrl_struct",
  "method": "frozen SDXL; IP semantic + mild/gated low-level img2img; luma semantic-recovery",
  "selection": "max SSIM then PixCorr under CLIP/A5/Inc/SwAV/FID gate (drop≤1.5pt / SwAV+0.025 / FID+20)",
  "ref": {k: ref[k] for k in ["tag", "pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "swav", "fid"]},
  "targets": {"ssim": 0.28, "pixcorr": 0.15, "note": "structure targets while keeping HCMA semantics"},
  "best_gated": best if best.get("pass_gate") else None,
  "best_overall_ssim": max(ranked, key=lambda x: x["ssim"]),
  "n_pass": len(gated),
  "all_ranked": ranked,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(out / "best_tag.txt").write_text(best["tag"] + "\n", encoding="utf-8")
lines = [
  "# Dual-control structure chase — sub-08",
  "",
  "Frozen SDXL. Semantic = HCMA IP/prompts. Structure = mild/gated Pc/LL img2img + optional semantic-recovery luma.",
  f"Ref: SSIM={ref['ssim']:.3f} Pix={ref['pixcorr']:.3f} CLIP={ref['clip']:.3f} A2={ref['alex2']:.3f} FID={ref['fid']:.1f}",
  "Gate: CLIP/A5/Inc ≥ ref−0.015; SwAV ≤ ref+0.025; FID ≤ ref+20. Select max SSIM then PixCorr.",
  "",
  "| tag | pass | SSIM | ΔS | Pix | ΔP | A2 | CLIP | A5 | Inc | SwAV | FID |",
  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in ranked:
    lines.append(
      f"| `{r['tag']}` | {int(r['pass_gate'])} | {r['ssim']:.3f} | {r['delta_ssim']:+.3f} | {r['pixcorr']:.3f} | {r['delta_pix']:+.3f} | "
      f"{r['alex2']:.3f} | {r['clip']:.3f} | {r['alex5']:.3f} | {r['inception']:.3f} | {r['swav']:.3f} | {r['fid']:.1f} |"
    )
(out / "DUAL_CTRL_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"best_gated": summary["best_gated"], "n_pass": len(gated)}, indent=2))
PY

echo "{\"pipeline\":\"dual_ctrl_struct\",\"finished\":\"$(date -Iseconds)\",\"best\":\"$(cat "${OUT}/best_tag.txt")\"}" > "${OUT}/job_done.json"
echo "[DONE] ${OUT}/DUAL_CTRL_TABLE.md"
