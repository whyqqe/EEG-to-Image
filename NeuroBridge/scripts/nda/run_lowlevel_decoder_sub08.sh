#!/usr/bin/env bash
# MindEye-style low-level decoder: EEG→VAE + fuse / img2img (disk-light)
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/lowlevel_decoder/sub-08}"
T1="${T1:-${NB_ROOT}/outputs/top1_structure/sub-08}"
NDA_SS="${NDA_SS:-${NB_ROOT}/outputs/nda_ss/sub-08}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
KEEP_TOP_K="${KEEP_TOP_K:-2}"

mkdir -p "${OUT}/vae_cache" "${OUT}/vae_head" "${OUT}/generation" "${OUT}/logs"
cd "${NB_ROOT}"

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"

SEM_GEN="${T1}/generation/pred_coca_ip1.0_cpa/generated"
EMB_IP="${NDA_SS}/blend/mem_decode_a50.npy"
DEC_TR="${NDA_SS}/train/z_decode_vith_train.npy"
DEC_TE="${NDA_SS}/train/z_decode_vith_test.npy"
PROMPT_CPA="${T1}/prompts/prompts_cpa.json"

test -f "${DEC_TR}" && test -f "${DEC_TE}" && test -f "${EMB_IP}"
test -f "${SEM_GEN}/000.png"
test -f "${PROMPT_CPA}"

echo "===== [1] GT VAE latents (float16) @ $(date -Iseconds) ====="
VAE_CACHE="${OUT}/vae_cache"
if [[ ! -f "${VAE_CACHE}/train_vae_latents_f16.npy" || ! -f "${VAE_CACHE}/test_vae_latents_f16.npy" ]]; then
  "${PYTHON}" scripts/nda/build_gt_vae_latents.py \
    --output-dir "${VAE_CACHE}" \
    --device "${DEVICE}" \
    --batch-size 8 \
    --splits "train,test"
else
  # rebuild if previous NaN cache
  "${PYTHON}" - <<PY
import numpy as np
from pathlib import Path
p=Path("${VAE_CACHE}/test_vae_latents_f16.npy")
a=np.asarray(np.load(p, mmap_mode="r")[:8], dtype=np.float32)
ok=bool(np.isfinite(a).all())
print("[CHECK] vae cache finite", ok)
open("${VAE_CACHE}/_finite_ok","w").write("1" if ok else "0")
PY
  if [[ "$(cat "${VAE_CACHE}/_finite_ok")" != "1" ]]; then
    echo "[WARN] rebuilding NaN VAE cache"
    "${PYTHON}" scripts/nda/build_gt_vae_latents.py \
      --output-dir "${VAE_CACHE}" \
      --device "${DEVICE}" \
      --batch-size 8 \
      --splits "train,test" \
      --force
  else
    echo "[SKIP] VAE cache"
  fi
fi

echo "===== [2] Train EEG→VAE head (ViT-H decode embeds) @ $(date -Iseconds) ====="
HEAD_OUT="${OUT}/vae_head"
if [[ ! -f "${HEAD_OUT}/vae_head_report.json" ]]; then
  "${PYTHON}" scripts/nda/train_eeg_vae_head.py \
    --eeg-train-npy "${DEC_TR}" \
    --eeg-test-npy "${DEC_TE}" \
    --vae-train-npy "${VAE_CACHE}/train_vae_latents_f16.npy" \
    --vae-test-npy "${VAE_CACHE}/test_vae_latents_f16.npy" \
    --output-dir "${HEAD_OUT}" \
    --num-epochs 80 \
    --batch-size 64 \
    --lr 3e-4 \
    --device "${DEVICE}" \
    --decode-rgb
else
  echo "[SKIP] VAE head"
fi
LL="${HEAD_OUT}/pred_lowlevel_rgb_512"
test -f "${LL}/000.png"

# Free ~0.5GB after training (keep test latents for diagnostics)
if [[ -f "${VAE_CACHE}/train_vae_latents_f16.npy" ]]; then
  rm -f "${VAE_CACHE}/train_vae_latents_f16.npy"
  echo "[DISK] removed train VAE latents"
fi

echo "===== [3] Generation: fuse + img2img @ $(date -Iseconds) ====="
run_fuse() {
  local tag="$1" alpha="$2"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" || -f "${gdir}/metrics.json" ]]; then
    echo "[SKIP] ${tag}"; return 0
  fi
  "${PYTHON}" scripts/nda/generate_lowlevel_decode.py \
    --mode fuse \
    --lowlevel-dir "${LL}" \
    --semantic-dir "${SEM_GEN}" \
    --fuse-alpha "${alpha}" \
    --output-dir "${gdir}" \
    --tag "${tag}" \
    --skip-metrics
}

run_i2i() {
  local tag="$1" strength="$2"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then
    echo "[SKIP] ${tag}"; return 0
  fi
  "${PYTHON}" scripts/nda/generate_lowlevel_decode.py \
    --mode img2img \
    --lowlevel-dir "${LL}" \
    --embed-npy "${EMB_IP}" \
    --prompts-json "${PROMPT_CPA}" \
    --strength "${strength}" \
    --ip-scale 1.0 \
    --gen-steps 28 \
    --gen-guidance 5.0 \
    --output-dir "${gdir}" \
    --tag "${tag}" \
    --seed 42 \
    --skip-metrics
}

# MindEye2 uses ~4:1 semantic:low → alpha≈0.8 on semantic
run_fuse "fuse_sem80_ll" 0.80
run_fuse "fuse_sem70_ll" 0.70
# img2img: keep layout from low-level, paint semantics
run_i2i "i2i_s55_cpa" 0.55
run_i2i "i2i_s45_cpa" 0.45
# also: fuse then lightly? skip to save disk — 4 tags enough

echo "===== [4] Paper metrics @ $(date -Iseconds) ====="
TAGS=""
for t in fuse_sem80_ll fuse_sem70_ll i2i_s55_cpa i2i_s45_cpa; do
  [[ -f "${OUT}/generation/${t}/generated/199.png" ]] && TAGS="${TAGS:+$TAGS,}${t}"
done
# include semantic ref path via symlink into generation for fair compare? copy metrics from known
# evaluate only new tags; summary will cite T1 champ numbers

"${PYTHON}" scripts/nda/eval_paper_metrics.py \
  --gen-root "${OUT}/generation" \
  --tags "${TAGS}" \
  --output-json "${OUT}/paper_metrics.json"

# also score low-level alone (should have high SSIM, low CLIP)
if [[ -f "${LL}/199.png" ]]; then
  mkdir -p "${OUT}/generation/lowlevel_only/generated"
  "${PYTHON}" - <<PY
from pathlib import Path
ll = Path("${LL}")
dst = Path("${OUT}/generation/lowlevel_only/generated")
for i in range(200):
    t = dst / f"{i:03d}.png"
    if not t.exists():
        t.symlink_to((ll / f"{i:03d}.png").resolve())
print("[OK] lowlevel_only symlinks")
PY
  "${PYTHON}" scripts/nda/eval_paper_metrics.py \
    --gen-root "${OUT}/generation" \
    --tags "lowlevel_only,${TAGS}" \
    --output-json "${OUT}/paper_metrics.json"
fi

echo "===== [5] Prune PNGs keep top-${KEEP_TOP_K} @ $(date -Iseconds) ====="
"${PYTHON}" - <<PY
import json
from pathlib import Path
out = Path("${OUT}/generation")
metrics = json.loads(Path("${OUT}/paper_metrics.json").read_text())
results = [r for r in metrics.get("results", []) if r["tag"] != "lowlevel_only"]
def score(r):
    return 0.45*float(r.get("clip_cosine") or 0) + 0.55*float(r.get("ssim") or 0)
ranked = sorted(results, key=score, reverse=True)
keep = {r["tag"] for r in ranked[: int("${KEEP_TOP_K}")]}
keep.add("lowlevel_only")  # tiny via symlinks
freed = 0
for d in out.iterdir():
    if not d.is_dir() or d.name in keep:
        continue
    for p in (d/"generated").glob("*.png"):
        if p.is_symlink():
            p.unlink()
            continue
        try:
            freed += p.stat().st_size
            p.unlink()
        except FileNotFoundError:
            pass
print(json.dumps({"kept": sorted(keep), "bytes_freed": freed}, indent=2))
PY

"${PYTHON}" - <<PY
import json
from pathlib import Path
out = Path("${OUT}")
metrics = json.loads((out/"paper_metrics.json").read_text())
results = metrics.get("results", [])
by = {r["tag"]: r for r in results}
head = json.loads((out/"vae_head/vae_head_report.json").read_text()) if (out/"vae_head/vae_head_report.json").is_file() else {}
# reference from prior champion (hardcoded from known summary)
ref = {
  "tag": "pred_coca_ip1.0_cpa (prior)",
  "clip_cosine": 0.5388,
  "ssim": 0.1988,
  "pixcorr": 0.1642,
  "note": "top1_structure champion; not re-scored here",
}
def score(r):
    return 0.45*float(r.get("clip_cosine") or 0)+0.55*float(r.get("ssim") or 0)
deploy = [r for r in results if r["tag"] != "lowlevel_only"]
best = max(deploy, key=score) if deploy else None
best_ssim = max(deploy, key=lambda r: r.get("ssim", -1)) if deploy else None
summary = {
  "pipeline": "lowlevel_decoder_mindeye",
  "plan": "EEG→SDXL-VAE + fuse/img2img; keep SDXL base",
  "vae_head": head,
  "prior_semantic_champ": ref,
  "best_combo": best,
  "best_ssim": best_ssim,
  "lowlevel_only": by.get("lowlevel_only"),
  "all_gen": results,
  "success_line_ssim": 0.28,
}
(out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

du -sh "${OUT}" "${OUT}/generation" 2>/dev/null || true
echo "===== DONE lowlevel decoder @ $(date -Iseconds) ====="
