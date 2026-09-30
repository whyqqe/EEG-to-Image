#!/usr/bin/env bash
# ATM-aligned decode (sub-08): HCMA semantics + VAE low-level init.
# Innovation unchanged (HCMA embeds/prompts). Decode interface aligned to ATM dual-stream.
# Priority: do NOT hurt semantics — high strength SDEdit + strict gate.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/atm_aligned_decode/sub-08}"
HCMA10="${HCMA10:-${NB_ROOT}/outputs/hcma_10subj}"
OVN="${OVN:-${NB_ROOT}/outputs/overnight_struct_v2/sub-08}"
LL="${LL:-${NB_ROOT}/outputs/lowlevel_decoder/sub-08}"
TCDA="${TCDA:-${NB_ROOT}/outputs/tcda/sub-08}"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

SEM="${HCMA10}/sub-08/generation/hcma_full_a40/generated"
EMB="${HCMA10}/sub-08/ft/embeds/blend_nda_cfm_f_a40_test.npy"
PROMPT="${HCMA10}/prompts/prompts_full_hcma_test.json"
VAE_LAT="${LL}/vae_head/pred_vae_test.npy"
LL_RGB="${LL}/vae_head/pred_lowlevel_rgb_512"
PC="${TCDA}/train/pred_pc_rgb_512"

mkdir -p "${OUT}/generation" "${OUT}/metrics" "${OUT}/logs" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" || true
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export DEVICE OUT NB_ROOT IMAGES_ROOT

echo "{\"pipeline\":\"atm_aligned_decode\",\"started\":\"$(date -Iseconds)\",\"goal\":\"align ATM dual-stream decode; protect HCMA semantics\"}" \
  > "${OUT}/job_running.json"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
require "${SEM}/000.png"; require "${EMB}"; require "${PROMPT}"
require "${VAE_LAT}"; require "${LL_RGB}/000.png"

link_tag() {
  local tag="$1" src="$2"
  local dst="${OUT}/generation/${tag}"
  mkdir -p "${dst}"
  [[ -e "${dst}/generated" ]] || ln -sfn "${src}" "${dst}/generated"
  echo "{\"tag\":\"${tag}\",\"source\":\"${src}\"}" > "${dst}/metrics.json"
}

echo "===== [0] baselines @ $(date -Iseconds) ====="
link_tag "ref_hcma_full_a40" "${SEM}"
[[ -d "${OVN}/generation/combo_d40_luma_pc_a060/generated" ]] && \
  link_tag "combo_d40_luma_pc_a060" "${OVN}/generation/combo_d40_luma_pc_a060/generated"
ALEX048=/project/peilab/why/NeuroBridge/outputs/alex2_first/sub-08/generation/alex_luma_pc_a048/generated
[[ -d "${ALEX048}" ]] && link_tag "alex_luma_pc_a048" "${ALEX048}"

run_sdedit() {
  local tag="$1" strength="$2" init="$3"
  local gdir="${OUT}/generation/${tag}"
  [[ -f "${gdir}/generated/199.png" ]] && echo "[SKIP] ${tag}" && return 0
  local extra=()
  if [[ "$init" == "vae" ]]; then
    extra+=(--vae-latent-npy "${VAE_LAT}")
  elif [[ "$init" == "ll" ]]; then
    extra+=(--lowlevel-rgb-dir "${LL_RGB}")
  elif [[ "$init" == "pc" ]]; then
    extra+=(--lowlevel-rgb-dir "${PC}")
  else
    echo "[FATAL] bad init $init"; return 1
  fi
  "${PYTHON}" scripts/nda/generate_atm_aligned_decode.py \
    --mode sdedit \
    --embed-npy "${EMB}" \
    --prompts-json "${PROMPT}" \
    --output-dir "${gdir}" \
    --tag "${tag}" \
    --strength "${strength}" \
    --ip-scale 1.0 \
    --gen-steps 28 \
    --gen-guidance 5.0 \
    --seed 42 \
    "${extra[@]}"
}

run_atmexact() {
  local tag="$1" strength="$2"
  local gdir="${OUT}/generation/${tag}"
  [[ -f "${gdir}/generated/199.png" ]] && echo "[SKIP] ${tag}" && return 0
  set +e
  "${PYTHON}" scripts/nda/generate_atm_aligned_decode.py \
    --mode atmexact \
    --embed-npy "${EMB}" \
    --prompts-json "${PROMPT}" \
    --output-dir "${gdir}" \
    --tag "${tag}" \
    --vae-latent-npy "${VAE_LAT}" \
    --strength "${strength}" \
    --ip-scale 1.0 \
    --gen-steps 28 \
    --gen-guidance 5.0 \
    --seed 42
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[WARN] atmexact ${tag} failed rc=${rc}; continue"
    rm -rf "${gdir}"
  fi
}

echo "===== [1] semantic-safe SDEdit (high strength) @ $(date -Iseconds) ====="
# High strength → more IP/text denoising → protect CLIP/FID; still dual-stream init
for s in 0.75 0.82 0.88; do
  run_sdedit "sdedit_vae_s$(echo $s | tr -d .)" "$s" vae
done
run_sdedit "sdedit_ll_s082" 0.82 ll
run_sdedit "sdedit_pc_s082" 0.82 pc
# slightly lower strength only if we need structure probe (may fail gate — kept for analysis)
run_sdedit "sdedit_vae_s065" 0.65 vae

echo "===== [2] ATM-exact schedule (optional) @ $(date -Iseconds) ====="
# Official-like high strength for semantic room
run_atmexact "atmexact_vae_s080" 0.80
run_atmexact "atmexact_vae_s090" 0.90

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

echo "===== [4] strict semantic gate + rank @ $(date -Iseconds) ====="
"${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT"])
rows = [json.loads(p.read_text()) for p in sorted((out / "metrics").glob("*_seven.json"))]
ref = next(r for r in rows if r["tag"] == "ref_hcma_full_a40")
# STRICT semantic protection (tighter than prior structure chases)
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
        "hit_ssim_028": float(r["ssim"]) >= 0.28,
        "hit_pix_015": float(r["pixcorr"]) >= 0.15,
    })
ranked.sort(key=lambda x: (x["pass_gate"], x["ssim"], x["pixcorr"], x["clip"]), reverse=True)
gated = [r for r in ranked if r["pass_gate"]]
best = gated[0] if gated else ranked[0]
# among gated, also track least semantic damage
best_sem = max(gated, key=lambda x: (x["clip"], -x["fid"])) if gated else None
summary = {
  "pipeline": "atm_aligned_decode",
  "method": "HCMA IP/prompts + ATM-style VAE/low-level init (SDEdit high-strength; optional ATM-exact)",
  "innovation_note": "No new method claim; decode interface aligned to ATM dual-stream; HCMA semantics unchanged",
  "semantic_gate": "CLIP/A5/Inc ≥ ref−0.010; SwAV ≤ ref+0.020; FID ≤ ref+15",
  "targets_structure": {"ssim": 0.28, "pixcorr": 0.15},
  "ref": {k: ref[k] for k in ["tag", "pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "swav", "fid"]},
  "best_gated": best if best.get("pass_gate") else None,
  "best_gated_semantics": best_sem,
  "n_pass": len(gated),
  "n_hit_both_targets_gated": sum(1 for r in gated if r["hit_ssim_028"] and r["hit_pix_015"]),
  "all_ranked": ranked,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(out / "best_tag.txt").write_text(best["tag"] + "\n", encoding="utf-8")
lines = [
  "# ATM-aligned decode (HCMA semantics preserved) — sub-08",
  "",
  "Decode-only alignment: HCMA embeds/prompts via IP + EEG→VAE/Pc/LL init (ATM dual-stream).",
  "No SDXL FT; not claimed as a new method. Strict semantic gate.",
  f"Ref: SSIM={ref['ssim']:.3f} Pix={ref['pixcorr']:.3f} CLIP={ref['clip']:.3f} FID={ref['fid']:.1f}",
  "Gate: CLIP/A5/Inc ≥ ref−0.010; SwAV ≤ ref+0.020; FID ≤ ref+15.",
  "",
  "| tag | pass | SSIM | ΔS | Pix | ΔP | CLIP | ΔC | A5 | FID | ΔFID |",
  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in ranked:
    lines.append(
      f"| `{r['tag']}` | {int(r['pass_gate'])} | {r['ssim']:.3f} | {r['delta_ssim']:+.3f} | "
      f"{r['pixcorr']:.3f} | {r['delta_pix']:+.3f} | {r['clip']:.3f} | {r['delta_clip']:+.3f} | "
      f"{r['alex5']:.3f} | {r['fid']:.1f} | {r['delta_fid']:+.1f} |"
    )
(out / "ATM_ALIGNED_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({
  "best_gated": summary["best_gated"],
  "best_gated_semantics": best_sem,
  "n_pass": len(gated),
  "n_hit_both": summary["n_hit_both_targets_gated"],
}, indent=2))
PY

echo "{\"pipeline\":\"atm_aligned_decode\",\"finished\":\"$(date -Iseconds)\",\"best\":\"$(cat "${OUT}/best_tag.txt")\"}" > "${OUT}/job_done.json"
echo "[DONE] ${OUT}/ATM_ALIGNED_TABLE.md"
