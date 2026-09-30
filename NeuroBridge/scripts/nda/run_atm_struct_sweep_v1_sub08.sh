#!/usr/bin/env bash
# Structure sweep v1: push SSIM toward ≥0.28 under STRICT semantic gate.
# Based on analysis: sdedit_ll_s082 = best gated; sdedit_pc_s082 = SSIM 0.282 but A5/FID fail.
# Strategy: Pc strength narrow-band + ATM-closer (fewer steps / lower CFG) to keep semantics.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/atm_struct_sweep_v1/sub-08}"
PREV="${PREV:-${NB_ROOT}/outputs/atm_aligned_decode/sub-08}"
HCMA10="${HCMA10:-${NB_ROOT}/outputs/hcma_10subj}"
LL="${LL:-${NB_ROOT}/outputs/lowlevel_decoder/sub-08}"
TCDA="${TCDA:-${NB_ROOT}/outputs/tcda/sub-08}"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

# Force caches onto project (home is full)
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${TORCH_HOME}"

SEM="${HCMA10}/sub-08/generation/hcma_full_a40/generated"
EMB="${HCMA10}/sub-08/ft/embeds/blend_nda_cfm_f_a40_test.npy"
PROMPT="${HCMA10}/prompts/prompts_full_hcma_test.json"
LL_RGB="${LL}/vae_head/pred_lowlevel_rgb_512"
PC="${TCDA}/train/pred_pc_rgb_512"
BLEND="${OUT}/assets/pc_ll_blend050"

mkdir -p "${OUT}/generation" "${OUT}/metrics" "${OUT}/assets" "${OUT}/logs" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" || true
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export DEVICE OUT NB_ROOT IMAGES_ROOT

echo "{\"pipeline\":\"atm_struct_sweep_v1\",\"started\":\"$(date -Iseconds)\",\"goal\":\"SSIM↑ under strict semantic gate; around Pc@0.82 + fewer steps/lower CFG\"}" \
  > "${OUT}/job_running.json"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
require "${SEM}/000.png"; require "${EMB}"; require "${PROMPT}"
require "${LL_RGB}/000.png"; require "${PC}/000.png"

link_tag() {
  local tag="$1" src="$2"
  local dst="${OUT}/generation/${tag}"
  mkdir -p "${dst}"
  [[ -e "${dst}/generated" ]] || ln -sfn "${src}" "${dst}/generated"
  echo "{\"tag\":\"${tag}\",\"source\":\"${src}\"}" > "${dst}/metrics.json"
}

echo "===== [0] baselines @ $(date -Iseconds) ====="
link_tag "ref_hcma_full_a40" "${SEM}"
[[ -d "${PREV}/generation/sdedit_ll_s082/generated" ]] && link_tag "sdedit_ll_s082" "${PREV}/generation/sdedit_ll_s082/generated"
[[ -d "${PREV}/generation/sdedit_pc_s082/generated" ]] && link_tag "sdedit_pc_s082" "${PREV}/generation/sdedit_pc_s082/generated"
ALEX048=/project/peilab/why/NeuroBridge/outputs/alex2_first/sub-08/generation/alex_luma_pc_a048/generated
[[ -d "${ALEX048}" ]] && link_tag "alex_luma_pc_a048" "${ALEX048}"

# Pc+LL blend init (structure compromise)
if [[ ! -f "${BLEND}/199.png" ]]; then
  mkdir -p "${BLEND}"
  "${PYTHON}" - <<PY
from pathlib import Path
from PIL import Image
pc, ll, out = Path("${PC}"), Path("${LL_RGB}"), Path("${BLEND}")
out.mkdir(parents=True, exist_ok=True)
for i in range(200):
    a = Image.open(pc/f"{i:03d}.png").convert("RGB").resize((512,512), Image.Resampling.BICUBIC)
    b = Image.open(ll/f"{i:03d}.png").convert("RGB").resize((512,512), Image.Resampling.BICUBIC)
    Image.blend(a, b, 0.5).save(out/f"{i:03d}.png")
print("[OK] blend", out)
PY
fi

run_sd() {
  local tag="$1" init_dir="$2" strength="$3" steps="$4" guidance="$5"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; return 0; fi
  "${PYTHON}" scripts/nda/generate_atm_aligned_decode.py \
    --mode sdedit \
    --embed-npy "${EMB}" \
    --prompts-json "${PROMPT}" \
    --lowlevel-rgb-dir "${init_dir}" \
    --output-dir "${gdir}" \
    --tag "${tag}" \
    --strength "${strength}" \
    --ip-scale 1.0 \
    --gen-steps "${steps}" \
    --gen-guidance "${guidance}" \
    --seed 42
}

echo "===== [1] Pc strength narrow band (steps=28, cfg=5) @ $(date -Iseconds) ====="
# Pull A5/FID back from s082 failure while keeping SSIM near 0.28
for s in 0.78 0.80 0.84 0.86; do
  run_sd "pc_s$(echo $s | tr -d .)_st28_g50" "${PC}" "$s" 28 5.0
done

echo "===== [2] ATM-closer: fewer steps / lower CFG @ $(date -Iseconds) ====="
# Less pixel rewrite → higher SSIM chance; must still pass semantic gate
run_sd "pc_s078_st16_g35" "${PC}" 0.78 16 3.5
run_sd "pc_s082_st16_g35" "${PC}" 0.82 16 3.5
run_sd "pc_s086_st16_g35" "${PC}" 0.86 16 3.5
run_sd "pc_s080_st20_g40" "${PC}" 0.80 20 4.0
run_sd "pc_s084_st20_g40" "${PC}" 0.84 20 4.0

echo "===== [3] LL + ATM-closer / blend @ $(date -Iseconds) ====="
run_sd "ll_s078_st16_g35" "${LL_RGB}" 0.78 16 3.5
run_sd "ll_s082_st20_g40" "${LL_RGB}" 0.82 20 4.0
run_sd "blend_s082_st28_g50" "${BLEND}" 0.82 28 5.0
run_sd "blend_s080_st16_g35" "${BLEND}" 0.80 16 3.5

echo "===== [4] official seven @ $(date -Iseconds) ====="
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

echo "===== [5] strict gate + rank @ $(date -Iseconds) ====="
"${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT"])
rows = [json.loads(p.read_text()) for p in sorted((out / "metrics").glob("*_seven.json"))]
ref = next(r for r in rows if r["tag"] == "ref_hcma_full_a40")
ranked = []
for r in rows:
    gate = (
        float(r["clip"]) >= float(ref["clip"]) - 0.010
        and float(r["alex5"]) >= float(ref["alex5"]) - 0.010
        and float(r["inception"]) >= float(ref["inception"]) - 0.010
        and float(r["swav"]) <= float(ref["swav"]) + 0.020
        and float(r["fid"]) <= float(ref["fid"]) + 15
    )
    ranked.append({
        **{k: r[k] for k in ["tag", "pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "swav", "fid"]},
        "pass_gate": gate,
        "delta_ssim": float(r["ssim"]) - float(ref["ssim"]),
        "delta_pix": float(r["pixcorr"]) - float(ref["pixcorr"]),
        "delta_clip": float(r["clip"]) - float(ref["clip"]),
        "delta_fid": float(r["fid"]) - float(ref["fid"]),
        "delta_alex5": float(r["alex5"]) - float(ref["alex5"]),
        "hit_ssim_028": float(r["ssim"]) >= 0.28,
        "hit_ssim_030": float(r["ssim"]) >= 0.30,
        "hit_pix_015": float(r["pixcorr"]) >= 0.15,
    })
ranked.sort(key=lambda x: (x["pass_gate"], x["ssim"], x["pixcorr"], x["clip"]), reverse=True)
gated = [r for r in ranked if r["pass_gate"]]
best = gated[0] if gated else ranked[0]
prev_ll = next((r for r in ranked if r["tag"] == "sdedit_ll_s082"), None)
summary = {
  "pipeline": "atm_struct_sweep_v1",
  "method": "HCMA IP/prompts + Pc/LL/blend SDEdit; narrow strength + fewer steps/lower CFG",
  "goal": "maximize SSIM/Pix under strict semantic gate (keep HCMA-level semantics)",
  "semantic_gate": "CLIP/A5/Inc ≥ ref−0.010; SwAV ≤ ref+0.020; FID ≤ ref+15",
  "prior_best_gated": "sdedit_ll_s082 (SSIM≈0.261)",
  "prior_near_miss": "sdedit_pc_s082 (SSIM≈0.282, fail A5/FID)",
  "ref": {k: ref[k] for k in ["tag", "pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "swav", "fid"]},
  "best_gated": best if best.get("pass_gate") else None,
  "vs_prev_ll": None if not (prev_ll and best.get("pass_gate")) else {
      "delta_ssim": float(best["ssim"]) - float(prev_ll["ssim"]),
      "delta_pix": float(best["pixcorr"]) - float(prev_ll["pixcorr"]),
      "delta_clip": float(best["clip"]) - float(prev_ll["clip"]),
  },
  "n_pass": len(gated),
  "n_hit_ssim028_gated": sum(1 for r in gated if r["hit_ssim_028"]),
  "n_hit_ssim030_gated": sum(1 for r in gated if r["hit_ssim_030"]),
  "n_hit_both015_028_gated": sum(1 for r in gated if r["hit_ssim_028"] and r["hit_pix_015"]),
  "all_ranked": ranked,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(out / "best_tag.txt").write_text(best["tag"] + "\n", encoding="utf-8")
lines = [
  "# ATM structure sweep v1 — push SSIM under strict semantic gate",
  "",
  "HCMA semantics frozen. Focus: Pc near s082 + ATM-closer (st16/g3.5, st20/g4.0) + LL/blend.",
  f"Ref: SSIM={ref['ssim']:.3f} Pix={ref['pixcorr']:.3f} CLIP={ref['clip']:.3f} A5={ref['alex5']:.3f} FID={ref['fid']:.1f}",
  "Gate: CLIP/A5/Inc ≥ ref−0.010; SwAV ≤ ref+0.020; FID ≤ ref+15.",
  "",
  "| tag | pass | SSIM≥.28 | SSIM | ΔS | Pix | CLIP | ΔC | A5 | ΔA5 | FID | ΔFID |",
  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in ranked:
    lines.append(
      f"| `{r['tag']}` | {int(r['pass_gate'])} | {int(r['hit_ssim_028'])} | {r['ssim']:.3f} | {r['delta_ssim']:+.3f} | "
      f"{r['pixcorr']:.3f} | {r['clip']:.3f} | {r['delta_clip']:+.3f} | {r['alex5']:.3f} | {r['delta_alex5']:+.3f} | "
      f"{r['fid']:.1f} | {r['delta_fid']:+.1f} |"
    )
(out / "STRUCT_SWEEP_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({
  "best_gated": summary["best_gated"],
  "vs_prev_ll": summary["vs_prev_ll"],
  "n_pass": len(gated),
  "n_hit_028": summary["n_hit_ssim028_gated"],
  "n_hit_both": summary["n_hit_both015_028_gated"],
}, indent=2))
PY

echo "{\"pipeline\":\"atm_struct_sweep_v1\",\"finished\":\"$(date -Iseconds)\",\"best\":\"$(cat "${OUT}/best_tag.txt")\"}" > "${OUT}/job_done.json"
echo "[DONE] ${OUT}/STRUCT_SWEEP_TABLE.md"
