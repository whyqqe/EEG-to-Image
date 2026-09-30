#!/usr/bin/env bash
# R-CFM-LL (sub-08): L1 VAE mean + residual Cond-CFM → blur RGB → gated HCMA SDEdit.
# Semantic path FROZEN (HCMA embeds/prompts). Structure-only train + decode.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/rcfm_ll/sub-08}"
NDA_SS="${NDA_SS:-${NB_ROOT}/outputs/nda_ss/sub-08}"
LL="${LL:-${NB_ROOT}/outputs/lowlevel_decoder/sub-08}"
HCMA10="${HCMA10:-${NB_ROOT}/outputs/hcma_10subj}"
PREV="${PREV:-${NB_ROOT}/outputs/atm_aligned_decode/sub-08}"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

# Force caches onto project (home often full)
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${TORCH_HOME}"

DEC_TR="${NDA_SS}/train/z_decode_vith_train.npy"
DEC_TE="${NDA_SS}/train/z_decode_vith_test.npy"
SEM="${HCMA10}/sub-08/generation/hcma_full_a40/generated"
EMB="${HCMA10}/sub-08/ft/embeds/blend_nda_cfm_f_a40_test.npy"
PROMPT="${HCMA10}/prompts/prompts_full_hcma_test.json"
LL_RGB="${LL}/vae_head/pred_lowlevel_rgb_512"

mkdir -p "${OUT}/vae_cache" "${OUT}/train" "${OUT}/generation" "${OUT}/metrics" "${OUT}/logs" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" || true
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export DEVICE OUT NB_ROOT IMAGES_ROOT

echo "{\"pipeline\":\"R-CFM-LL\",\"started\":\"$(date -Iseconds)\",\"goal\":\"raise SSIM via residual CFM LL under strict HCMA gate\"}" \
  > "${OUT}/job_running.json"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
require "${DEC_TR}"; require "${DEC_TE}"
require "${SEM}/000.png"; require "${EMB}"; require "${PROMPT}"
require "${LL_RGB}/000.png"

VAE_CACHE="${OUT}/vae_cache"
# Prefer symlink/copy from existing LL test cache to save rebuild
if [[ -f "${LL}/vae_cache/test_vae_latents_f16.npy" && ! -f "${VAE_CACHE}/test_vae_latents_f16.npy" ]]; then
  cp -n "${LL}/vae_cache/test_vae_latents_f16.npy" "${VAE_CACHE}/test_vae_latents_f16.npy" || true
fi

echo "===== [1] GT VAE latents @ $(date -Iseconds) ====="
need_rebuild=0
[[ -f "${VAE_CACHE}/train_vae_latents_f16.npy" ]] || need_rebuild=1
[[ -f "${VAE_CACHE}/test_vae_latents_f16.npy" ]] || need_rebuild=1
if [[ "${need_rebuild}" == "1" ]]; then
  "${PYTHON}" scripts/nda/build_gt_vae_latents.py \
    --output-dir "${VAE_CACHE}" \
    --device "${DEVICE}" \
    --batch-size 8 \
    --splits "train,test"
else
  echo "[SKIP] VAE cache"
fi

echo "===== [2] Train R-CFM-LL @ $(date -Iseconds) ====="
TRAIN_OUT="${OUT}/train"
if [[ ! -f "${TRAIN_OUT}/rcfm_ll_train_report.json" ]]; then
  "${PYTHON}" scripts/nda/train_rcfm_ll.py \
    --eeg-train-npy "${DEC_TR}" \
    --eeg-test-npy "${DEC_TE}" \
    --vae-train-npy "${VAE_CACHE}/train_vae_latents_f16.npy" \
    --vae-test-npy "${VAE_CACHE}/test_vae_latents_f16.npy" \
    --output-dir "${TRAIN_OUT}" \
    --num-epochs 60 \
    --batch-size 40 \
    --lr 2e-4 \
    --device "${DEVICE}" \
    --w-l1 1.0 \
    --w-cfm 0.45 \
    --w-ctr 0.05 \
    --alpha 1.0 \
    --ode-steps 12 \
    --decode-rgb
else
  echo "[SKIP] train"
fi

BLUR="${TRAIN_OUT}/pred_blur_rgb_512"
BLUR_MU="${TRAIN_OUT}/pred_blur_mu_rgb_512"
require "${BLUR}/000.png"

# Free train latents (~large) after training
if [[ -f "${VAE_CACHE}/train_vae_latents_f16.npy" ]]; then
  echo "[DISK] removing train VAE latents"
  rm -f "${VAE_CACHE}/train_vae_latents_f16.npy"
fi

echo "===== [3] Phase-A blur structure vs legacy LL @ $(date -Iseconds) ====="
"${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_ski

import sys
sys.path.insert(0, "/project/peilab/why/eeg-brainit/scripts")
from eval_atm_pipeline import list_test_images

out = Path(os.environ["OUT"])
images_root = Path(os.environ["IMAGES_ROOT"])
blur = out / "train/pred_blur_rgb_512"
blur_mu = out / "train/pred_blur_mu_rgb_512"
ll = Path("/project/peilab/why/NeuroBridge/outputs/lowlevel_decoder/sub-08/vae_head/pred_lowlevel_rgb_512")
gt_list = list_test_images(images_root)

def pixcorr(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    a -= a.mean(); b -= b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / den) if den > 1e-8 else 0.0

def eval_dir(d: Path, tag: str):
    pixs, ssims = [], []
    for i in range(200):
        g = np.asarray(Image.open(d / f"{i:03d}.png").convert("RGB").resize((256, 256), Image.Resampling.BICUBIC))
        t = np.asarray(Image.open(gt_list[i]).convert("RGB").resize((256, 256), Image.Resampling.BICUBIC))
        pixs.append(pixcorr(g, t))
        ssims.append(float(ssim_ski(t, g, channel_axis=-1, data_range=255)))
    return {"tag": tag, "pixcorr": float(np.mean(pixs)), "ssim": float(np.mean(ssims)), "n": 200}

rows = [eval_dir(blur, "rcfm_blur"), eval_dir(ll, "legacy_ll")]
if (blur_mu / "000.png").is_file():
    rows.append(eval_dir(blur_mu, "rcfm_mu_only"))
by = {r["tag"]: r for r in rows}
go = by["rcfm_blur"]["ssim"] >= by["legacy_ll"]["ssim"] - 1e-4 or by["rcfm_blur"]["pixcorr"] >= by["legacy_ll"]["pixcorr"]
phase_a = {
    "phase": "A_blur_vs_ll",
    "rows": rows,
    "go_phase_b": True,  # always run gated SDEdit; mark relative quality
    "rcfm_beats_ll": bool(
        by["rcfm_blur"]["ssim"] > by["legacy_ll"]["ssim"]
        or by["rcfm_blur"]["pixcorr"] > by["legacy_ll"]["pixcorr"]
    ),
    "delta_ssim_vs_ll": by["rcfm_blur"]["ssim"] - by["legacy_ll"]["ssim"],
    "delta_pix_vs_ll": by["rcfm_blur"]["pixcorr"] - by["legacy_ll"]["pixcorr"],
}
(out / "phase_a_blur_compare.json").write_text(json.dumps(phase_a, indent=2), encoding="utf-8")
print(json.dumps(phase_a, indent=2))
PY

link_tag() {
  local tag="$1" src="$2"
  local dst="${OUT}/generation/${tag}"
  mkdir -p "${dst}"
  [[ -e "${dst}/generated" ]] || ln -sfn "${src}" "${dst}/generated"
  echo "{\"tag\":\"${tag}\",\"source\":\"${src}\"}" > "${dst}/metrics.json"
}

echo "===== [4] Phase-B gated SDEdit (HCMA frozen) @ $(date -Iseconds) ====="
link_tag "ref_hcma_full_a40" "${SEM}"
[[ -d "${PREV}/generation/sdedit_ll_s082/generated" ]] && \
  link_tag "sdedit_ll_s082" "${PREV}/generation/sdedit_ll_s082/generated"
link_tag "rcfm_blur_raw" "${BLUR}"

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

# Anchor around proven semantic-safe strength; also probe mild structure
run_sd "rcfm_s080_st28_g50" "${BLUR}" 0.80 28 5.0
run_sd "rcfm_s082_st28_g50" "${BLUR}" 0.82 28 5.0
run_sd "rcfm_s086_st28_g50" "${BLUR}" 0.86 28 5.0
# alpha ablations via mu-only blur if present
if [[ -f "${BLUR_MU}/000.png" ]]; then
  run_sd "rcfm_mu_s082_st28_g50" "${BLUR_MU}" 0.82 28 5.0
fi

echo "===== [5] official seven @ $(date -Iseconds) ====="
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

echo "===== [6] strict gate + summary @ $(date -Iseconds) ====="
"${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT"])
rows = [json.loads(p.read_text()) for p in sorted((out / "metrics").glob("*_seven.json"))]
ref = next(r for r in rows if r["tag"] == "ref_hcma_full_a40")
prior = next((r for r in rows if r["tag"] == "sdedit_ll_s082"), None)
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
phase_a = {}
pa = out / "phase_a_blur_compare.json"
if pa.is_file():
    phase_a = json.loads(pa.read_text())
summary = {
    "pipeline": "R-CFM-LL",
    "claim": "L1 μ + residual Cond-CFM blur init into frozen HCMA SDEdit",
    "semantic_gate": "CLIP/A5/Inc ≥ ref−0.010; SwAV ≤ ref+0.020; FID ≤ ref+15",
    "phase_a": phase_a,
    "ref": {"ssim": ref["ssim"], "clip": ref["clip"], "fid": ref["fid"], "inception": ref.get("inception")},
    "prior_best_gated": "sdedit_ll_s082",
    "best_gated": best,
    "all_ranked": ranked,
    "n_pass": sum(1 for x in ranked if x["pass_gate"]),
}
(out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
# markdown table
lines = [
    "# R-CFM-LL — residual Cond-CFM low-level under HCMA gate",
    "",
    f"Phase-A: rcfm_beats_ll={phase_a.get('rcfm_beats_ll')} ΔSSIM={phase_a.get('delta_ssim_vs_ll')}",
    f"Gate: CLIP/A5/Inc ≥ ref−0.010; SwAV ≤ ref+0.020; FID ≤ ref+15.",
    "",
    "| tag | pass | SSIM | Pix | CLIP | A5 | Inc | FID |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for x in ranked:
    lines.append(
        f"| `{x['tag']}` | {int(x['pass_gate'])} | {x['ssim']:.3f} | {x['pixcorr']:.3f} | "
        f"{x['clip']:.3f} | {x['alex5']:.3f} | {x['inception']:.3f} | {x['fid']:.1f} |"
    )
(out / "RCFM_LL_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

du -sh "${OUT}" 2>/dev/null || true
echo "{\"pipeline\":\"R-CFM-LL\",\"finished\":\"$(date -Iseconds)\"}" > "${OUT}/job_done.json"
echo "===== DONE R-CFM-LL @ $(date -Iseconds) ====="
