#!/usr/bin/env bash
# HCMA-S (sub-08): freeze HCMA semantic + retrain Depth expert + LL init dual decode.
# Goal: raise gated SSIM via Depth-CN × LL-SDEdit under strict semantic gate.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/hcma_s/sub-08}"
HCMA10="${HCMA10:-${NB_ROOT}/outputs/hcma_10subj}"
LL="${LL:-${NB_ROOT}/outputs/lowlevel_decoder/sub-08}"
PREV="${PREV:-${NB_ROOT}/outputs/atm_aligned_decode/sub-08}"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${TORCH_HOME}"

SEM="${HCMA10}/sub-08/generation/hcma_full_a40/generated"
EMB="${HCMA10}/sub-08/ft/embeds/blend_nda_cfm_f_a40_test.npy"
PROMPT="${HCMA10}/prompts/prompts_full_hcma_test.json"
NEIGH="${HCMA10}/sub-08/memory/rag_soft5_neighbor_idx_test.npy"
ZTR="${HCMA10}/sub-08/zret/z_ret_train.npy"
ZTE="${HCMA10}/sub-08/zret/z_ret_test.npy"
LL_RGB="${LL}/vae_head/pred_lowlevel_rgb_512"

mkdir -p "${OUT}/gt_depth" "${OUT}/depth" "${OUT}/generation" "${OUT}/metrics" "${OUT}/logs" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" || true
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export DEVICE OUT NB_ROOT IMAGES_ROOT

echo "{\"pipeline\":\"HCMA-S\",\"started\":\"$(date -Iseconds)\",\"goal\":\"Depth-CN+LL-SDEdit under frozen HCMA semantics\"}" \
  > "${OUT}/job_running.json"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
require "${SEM}/000.png"; require "${EMB}"; require "${PROMPT}"
require "${ZTR}"; require "${ZTE}"; require "${LL_RGB}/000.png"
require "${NEIGH}"

echo "===== [1] GT depth cache @ $(date -Iseconds) ====="
DTR="${OUT}/gt_depth/train_depth_64.npy"
DTE="${OUT}/gt_depth/test_depth_64.npy"
if [[ ! -f "${DTR}" || ! -f "${DTE}" ]]; then
  "${PYTHON}" scripts/nda/build_gt_depth_cache.py \
    --output-dir "${OUT}/gt_depth" \
    --device "${DEVICE}" \
    --splits "train,test" \
    --batch-size 8
else
  echo "[SKIP] gt depth"
fi
require "${DTR}"; require "${DTE}"

echo "===== [2] Retrain Depth expert (HCMA z_ret) @ $(date -Iseconds) ====="
DEPTH_OUT="${OUT}/depth"
if [[ ! -f "${DEPTH_OUT}/depth_head_report.json" ]]; then
  "${PYTHON}" scripts/nda/train_eeg_depth_head.py \
    --eeg-train-npy "${ZTR}" \
    --eeg-test-npy "${ZTE}" \
    --depth-train-npy "${DTR}" \
    --depth-test-npy "${DTE}" \
    --output-dir "${DEPTH_OUT}" \
    --num-epochs 60 \
    --batch-size 256 \
    --lr 1e-3 \
    --lambda-grad 0.5 \
    --cn-min 0.25 \
    --cn-max 0.45 \
    --device "${DEVICE}"
else
  echo "[SKIP] depth train"
fi
DEPTH_RGB="${DEPTH_OUT}/pred_depth_rgb_512"
require "${DEPTH_RGB}/000.png"

# free large train depth after training (~270MB)
if [[ -f "${DTR}" ]]; then
  echo "[DISK] remove train_depth_64.npy"
  rm -f "${DTR}"
fi

link_tag() {
  local tag="$1" src="$2"
  local dst="${OUT}/generation/${tag}"
  mkdir -p "${dst}"
  [[ -e "${dst}/generated" ]] || ln -sfn "${src}" "${dst}/generated"
  echo "{\"tag\":\"${tag}\",\"source\":\"${src}\"}" > "${dst}/metrics.json"
}

echo "===== [3] baselines @ $(date -Iseconds) ====="
link_tag "ref_hcma_full_a40" "${SEM}"
[[ -d "${PREV}/generation/sdedit_ll_s082/generated" ]] && \
  link_tag "sdedit_ll_s082" "${PREV}/generation/sdedit_ll_s082/generated"

run_cn_only() {
  local tag="$1" cn="$2"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; return 0; fi
  # NO --cn-scale-npy (must not override swept scale)
  "${PYTHON}" scripts/nda/generate_cn_ip_decode.py \
    --embed-npy "${EMB}" \
    --neighbor-idx-npy "${NEIGH}" \
    --output-dir "${gdir}" \
    --tag "${tag}" \
    --seed 42 \
    --control-type depth \
    --depth-sample-dir "${DEPTH_RGB}" \
    --cn-scale "${cn}" \
    --ip-scale 1.0 \
    --gen-steps 28 \
    --gen-guidance 5.0 \
    --prompts-json "${PROMPT}" \
    --skip-metrics
}

run_dual() {
  local tag="$1" cn="$2" strength="$3"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; return 0; fi
  "${PYTHON}" scripts/nda/generate_hcma_s_decode.py \
    --embed-npy "${EMB}" \
    --prompts-json "${PROMPT}" \
    --depth-rgb-dir "${DEPTH_RGB}" \
    --lowlevel-rgb-dir "${LL_RGB}" \
    --output-dir "${gdir}" \
    --tag "${tag}" \
    --cn-scale "${cn}" \
    --ip-scale 1.0 \
    --strength "${strength}" \
    --gen-steps 28 \
    --gen-guidance 5.0 \
    --seed 42
}

echo "===== [4] Depth-CN only ablations @ $(date -Iseconds) ====="
run_cn_only "dcn_c025" 0.25
run_cn_only "dcn_c032" 0.32
run_cn_only "dcn_c040" 0.40

echo "===== [5] HCMA-S dual: Depth-CN × LL-SDEdit @ $(date -Iseconds) ====="
# Primary grid from analysis: cn∈[0.25,0.40], strength∈[0.82,0.88]
for cn in 0.25 0.32 0.40; do
  for s in 0.82 0.86 0.88; do
    ctag=$(echo "$cn" | tr -d .)
    stag=$(echo "$s" | tr -d .)
    run_dual "hs_c${ctag}_s${stag}" "$cn" "$s"
  done
done

echo "===== [6] official seven @ $(date -Iseconds) ====="
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

echo "===== [7] strict gate + summary @ $(date -Iseconds) ====="
"${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT"])
rows = [json.loads(p.read_text()) for p in sorted((out / "metrics").glob("*_seven.json"))]
ref = next(r for r in rows if r["tag"] == "ref_hcma_full_a40")
prior = next((r for r in rows if r["tag"] == "sdedit_ll_s082"), None)
depth_rep = {}
dr = out / "depth/depth_head_report.json"
if dr.is_file():
    depth_rep = json.loads(dr.read_text())
ranked = []
for r in rows:
    gate = (
        float(r["clip"]) >= float(ref["clip"]) - 0.010
        and float(r["alex5"]) >= float(ref["alex5"]) - 0.010
        and float(r["inception"]) >= float(ref["inception"]) - 0.010
        and float(r["swav"]) <= float(ref["swav"]) + 0.020
        and float(r["fid"]) <= float(ref["fid"]) + 15.0
    )
    ranked.append({
        "tag": r["tag"],
        "pass_gate": bool(gate) if r["tag"] != "ref_hcma_full_a40" else True,
        "ssim": float(r["ssim"]),
        "pixcorr": float(r["pixcorr"]),
        "clip": float(r["clip"]),
        "alex5": float(r["alex5"]),
        "inception": float(r.get("inception", 0)),
        "fid": float(r["fid"]),
        "swav": float(r.get("swav", 0)),
        "delta_ssim_vs_ref": float(r["ssim"]) - float(ref["ssim"]),
        "delta_ssim_vs_ll": (float(r["ssim"]) - float(prior["ssim"])) if prior else None,
    })
ranked.sort(key=lambda x: (-int(x["pass_gate"]), -x["ssim"]))
gated = [x for x in ranked if x["pass_gate"]]
best = gated[0] if gated else ranked[0]
summary = {
    "pipeline": "HCMA-S",
    "claim": "Frozen HCMA semantic + retrained Depth expert + LL SDEdit dual decode",
    "semantic_gate": "CLIP/A5/Inc ≥ ref−0.010; SwAV ≤ ref+0.020; FID ≤ ref+15",
    "depth_train": depth_rep,
    "ref": {"ssim": ref["ssim"], "clip": ref["clip"], "fid": ref["fid"], "inception": ref.get("inception")},
    "prior_best_gated": "sdedit_ll_s082",
    "best_gated": best,
    "all_ranked": ranked,
    "n_pass": sum(1 for x in ranked if x["pass_gate"]),
    "target_ssim": 0.28,
    "hit_target": bool(best.get("pass_gate") and best["ssim"] >= 0.28),
}
(out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
lines = [
    "# HCMA-S — Depth-CN + LL-SDEdit under frozen HCMA semantics",
    "",
    f"Depth pearson={depth_rep.get('best_pearson')}",
    f"Gate: CLIP/A5/Inc ≥ ref−0.010; SwAV ≤ ref+0.020; FID ≤ ref+15.",
    f"Best gated: `{best['tag']}` SSIM={best['ssim']:.3f} pass={best['pass_gate']}",
    "",
    "| tag | pass | SSIM | Pix | CLIP | A5 | Inc | FID |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for x in ranked:
    lines.append(
        f"| `{x['tag']}` | {int(x['pass_gate'])} | {x['ssim']:.3f} | {x['pixcorr']:.3f} | "
        f"{x['clip']:.3f} | {x['alex5']:.3f} | {x['inception']:.3f} | {x['fid']:.1f} |"
    )
(out / "HCMA_S_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

du -sh "${OUT}" 2>/dev/null || true
echo "{\"pipeline\":\"HCMA-S\",\"finished\":\"$(date -Iseconds)\"}" > "${OUT}/job_done.json"
echo "===== DONE HCMA-S @ $(date -Iseconds) ====="
