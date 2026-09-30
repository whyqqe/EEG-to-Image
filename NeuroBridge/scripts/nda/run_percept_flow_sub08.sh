#!/usr/bin/env bash
# PerceptFlow: perception/structure CFM + low-strength img2img injection (sub-08)
# Semantic a40 path is FROZEN; only perc tower + injection modes are trained/evaluated.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/percept_flow/sub-08}"
NDA_SS="${NDA_SS:-${NB_ROOT}/outputs/nda_ss/sub-08}"
MG="${MG:-${NB_ROOT}/outputs/mg_flow/sub-08}"
T1="${T1:-${NB_ROOT}/outputs/top1_structure/sub-08}"
LL="${LL:-${NB_ROOT}/outputs/lowlevel_decoder/sub-08}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "${OUT}/vae_cache" "${OUT}/train" "${OUT}/generation" "${OUT}/compare" "${OUT}/logs"
cd "${NB_ROOT}"

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"

DEC_TR="${NDA_SS}/train/z_decode_vith_train.npy"
DEC_TE="${NDA_SS}/train/z_decode_vith_test.npy"
DEPTH_TR="${T1}/track_s/gt_depth/train_depth_64.npy"
DEPTH_TE="${T1}/track_s/gt_depth/test_depth_64.npy"
EMB_A40="${MG}/train/embeds/blend_nda_cfm_f_a40_test.npy"
SEM_A40="${MG}/generation/mg_blend_a40_dual/generated"
PROMPT_DUAL="${MG}/targets/prompts_dual_test.json"
NEIGH="${NDA_SS}/memory/rag_soft5_neighbor_idx_test.npy"

test -f "${DEC_TR}" && test -f "${DEC_TE}"
test -f "${DEPTH_TR}" && test -f "${DEPTH_TE}"
test -f "${EMB_A40}"
test -f "${SEM_A40}/000.png"
test -f "${PROMPT_DUAL}"

VAE_CACHE="${OUT}/vae_cache"
# reuse existing test cache if present to save disk/time
if [[ -f "${LL}/vae_cache/test_vae_latents_f16.npy" && ! -f "${VAE_CACHE}/test_vae_latents_f16.npy" ]]; then
  mkdir -p "${VAE_CACHE}"
  cp -n "${LL}/vae_cache/test_vae_latents_f16.npy" "${VAE_CACHE}/test_vae_latents_f16.npy" || true
  cp -n "${LL}/vae_cache/vae_latent_report.json" "${VAE_CACHE}/vae_latent_report.json" 2>/dev/null || true
fi

echo "===== [1] GT VAE latents @ $(date -Iseconds) ====="
need_train=0
if [[ ! -f "${VAE_CACHE}/train_vae_latents_f16.npy" ]]; then need_train=1; fi
if [[ ! -f "${VAE_CACHE}/test_vae_latents_f16.npy" ]]; then need_train=1; fi
if [[ "${need_train}" == "1" ]]; then
  "${PYTHON}" scripts/nda/build_gt_vae_latents.py \
    --output-dir "${VAE_CACHE}" \
    --device "${DEVICE}" \
    --batch-size 8 \
    --splits "train,test"
else
  "${PYTHON}" - <<PY
import numpy as np
from pathlib import Path
ok=True
for s in ("train","test"):
    p=Path("${VAE_CACHE}")/f"{s}_vae_latents_f16.npy"
    if not p.is_file():
        ok=False; print("[MISS]", p); continue
    a=np.asarray(np.load(p, mmap_mode="r")[:8], dtype=np.float32)
    print(s, "finite", bool(np.isfinite(a).all()), "shape", np.load(p, mmap_mode="r").shape)
    ok = ok and bool(np.isfinite(a).all())
open("${VAE_CACHE}/_finite_ok","w").write("1" if ok else "0")
PY
  if [[ "$(cat "${VAE_CACHE}/_finite_ok")" != "1" ]]; then
    echo "[WARN] rebuilding VAE cache"
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

echo "===== [2] Train PerceptFlow @ $(date -Iseconds) ====="
TRAIN_OUT="${OUT}/train"
if [[ ! -f "${TRAIN_OUT}/percept_flow_train_report.json" ]]; then
  "${PYTHON}" scripts/nda/train_percept_flow.py \
    --eeg-train-npy "${DEC_TR}" \
    --eeg-test-npy "${DEC_TE}" \
    --vae-train-npy "${VAE_CACHE}/train_vae_latents_f16.npy" \
    --vae-test-npy "${VAE_CACHE}/test_vae_latents_f16.npy" \
    --depth-train-npy "${DEPTH_TR}" \
    --depth-test-npy "${DEPTH_TE}" \
    --output-dir "${TRAIN_OUT}" \
    --num-epochs 50 \
    --batch-size 40 \
    --lr 2e-4 \
    --device "${DEVICE}" \
    --decode-rgb
else
  echo "[SKIP] train"
fi

BLUR="${TRAIN_OUT}/pred_blur_rgb_512"
DEPTH_RGB="${TRAIN_OUT}/pred_depth_rgb_512"
test -f "${BLUR}/000.png"
test -f "${DEPTH_RGB}/000.png"

# free ~540MB after train
if [[ -f "${VAE_CACHE}/train_vae_latents_f16.npy" ]]; then
  echo "[DISK] removing train VAE latents after training"
  rm -f "${VAE_CACHE}/train_vae_latents_f16.npy"
fi

echo "===== [3] Injection generation @ $(date -Iseconds) ====="
# 3a low-strength img2img (perception init + frozen a40 semantics)
run_i2i() {
  local tag="$1" strength="$2"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; return 0; fi
  "${PYTHON}" scripts/nda/generate_lowlevel_decode.py \
    --mode img2img \
    --lowlevel-dir "${BLUR}" \
    --embed-npy "${EMB_A40}" \
    --prompts-json "${PROMPT_DUAL}" \
    --output-dir "${gdir}" \
    --tag "${tag}" \
    --strength "${strength}" \
    --ip-scale 1.0 \
    --gen-steps 30 \
    --gen-guidance 5.0 \
    --skip-metrics
}

run_i2i "pf_i2i_s28" 0.28
run_i2i "pf_i2i_s35" 0.35
run_i2i "pf_i2i_s40" 0.40

# 3b frequency fuse: low-freq from blur / high-freq from a40 semantic
run_ff() {
  local tag="$1" sigma="$2"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; return 0; fi
  "${PYTHON}" scripts/nda/generate_freq_fuse.py \
    --struct-dir "${BLUR}" \
    --semantic-dir "${SEM_A40}" \
    --output-dir "${gdir}" \
    --sigma "${sigma}" \
    --tag "${tag}"
}
run_ff "pf_freq_s8" 8.0
run_ff "pf_freq_s12" 12.0

# 3c alpha fuse baselines
run_fuse() {
  local tag="$1" alpha="$2"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; return 0; fi
  "${PYTHON}" scripts/nda/generate_lowlevel_decode.py \
    --mode fuse \
    --lowlevel-dir "${BLUR}" \
    --semantic-dir "${SEM_A40}" \
    --fuse-alpha "${alpha}" \
    --output-dir "${gdir}" \
    --tag "${tag}" \
    --skip-metrics
}
run_fuse "pf_fuse_a75" 0.75
run_fuse "pf_fuse_a85" 0.85

# 3d weak Depth-CN with PerceptFlow depth + frozen a40 embeds
run_cn() {
  local tag="$1" cn_scale="$2"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; return 0; fi
  "${PYTHON}" scripts/nda/generate_cn_ip_decode.py \
    --embed-npy "${EMB_A40}" \
    --neighbor-idx-npy "${NEIGH}" \
    --output-dir "${gdir}" \
    --tag "${tag}" \
    --seed 42 \
    --control-type depth \
    --depth-sample-dir "${DEPTH_RGB}" \
    --cn-scale "${cn_scale}" \
    --ip-scale 1.0 \
    --gen-steps 30 \
    --gen-guidance 5.0 \
    --prompts-json "${PROMPT_DUAL}" \
    --skip-metrics
}
run_cn "pf_cn_d30" 0.30
run_cn "pf_cn_d45" 0.45

# symlink a40 ref for metrics
REF_DIR="${OUT}/generation/ref_a40_dual"
mkdir -p "${REF_DIR}"
if [[ ! -e "${REF_DIR}/generated" ]]; then
  ln -sfn "${SEM_A40}" "${REF_DIR}/generated"
fi
echo '{"tag":"ref_a40_dual","note":"frozen MG-Flow a40 semantic baseline"}' > "${REF_DIR}/metrics.json"

echo "===== [4] Paper metrics @ $(date -Iseconds) ====="
TAGS="ref_a40_dual,pf_i2i_s28,pf_i2i_s35,pf_i2i_s40,pf_freq_s8,pf_freq_s12,pf_fuse_a75,pf_fuse_a85,pf_cn_d30,pf_cn_d45"
VALID=""
IFS=',' read -ra ARR <<< "${TAGS}"
for t in "${ARR[@]}"; do
  [[ -f "${OUT}/generation/${t}/generated/199.png" ]] && VALID="${VALID:+$VALID,}${t}"
done
"${PYTHON}" scripts/nda/eval_paper_metrics.py \
  --gen-root "${OUT}/generation" \
  --tags "${VALID}" \
  --output-json "${OUT}/paper_metrics.json"

echo "===== [5] CLIP 2-way @ $(date -Iseconds) ====="
GEN_ARGS=""
IFS=',' read -ra ARR <<< "${VALID}"
for t in "${ARR[@]}"; do
  GEN_ARGS="${GEN_ARGS:+$GEN_ARGS,}${t}=${OUT}/generation/${t}/generated"
done
"${PYTHON}" scripts/nda/eval_clip_2way.py \
  --gen-dirs "${GEN_ARGS}" \
  --output-json "${OUT}/clip_2way_report.json" \
  --device "${DEVICE}" \
  --batch-size 16

echo "===== [6] Class consistency @ $(date -Iseconds) ====="
"${PYTHON}" scripts/nda/eval_class_consistency.py \
  --gen-dirs "${GEN_ARGS}" \
  --text-concept-npy "${NDA_SS}/clip_text/test/text_concept_clip.npy" \
  --concepts-json "${MG}/targets/concepts_test.json" \
  --output-json "${OUT}/class_consistency.json" \
  --device "${DEVICE}"

echo "===== [7] Quality gate vs a40 @ $(date -Iseconds) ====="
"${PYTHON}" - <<'PY'
import json
from pathlib import Path
out = Path("/project/peilab/why/NeuroBridge/outputs/percept_flow/sub-08")
paper = json.loads((out/"paper_metrics.json").read_text()) if (out/"paper_metrics.json").is_file() else {"results":[]}
tw = json.loads((out/"clip_2way_report.json").read_text()) if (out/"clip_2way_report.json").is_file() else {"generation_2way":[]}
cls = json.loads((out/"class_consistency.json").read_text()) if (out/"class_consistency.json").is_file() else {"results":[]}
by_p = {r["tag"]: r for r in paper.get("results", [])}
by_2 = {r["tag"]: r for r in tw.get("generation_2way", [])}
by_c = {r["tag"]: r for r in cls.get("results", [])}
ref = by_p.get("ref_a40_dual", {})
ref2 = by_2.get("ref_a40_dual", {})
ref_2way = float(ref2.get("clip_2way", 0))
ref_fid = float(ref.get("fid", 999))
ref_ssim = float(ref.get("ssim", 0))
rows = []
for tag in sorted(set(by_p)|set(by_2)|set(by_c)):
    p, w, c = by_p.get(tag, {}), by_2.get(tag, {}), by_c.get(tag, {})
    twoway = float(w.get("clip_2way", 0))
    clstop = float(c.get("class_top1", 0))
    fid = float(p.get("fid", 999))
    ssim = float(p.get("ssim", 0))
    # gate: keep semantic floor; reward SSIM/FID gains
    pass_gate = (twoway >= ref_2way - 0.02) and (fid <= ref_fid + 25.0)
    # structure-aware score with semantic protection
    score = (
        0.30 * twoway
        + 0.25 * clstop
        + 0.20 * max(0, (320 - fid) / 170)
        + 0.25 * ssim
    )
    if not pass_gate and tag != "ref_a40_dual":
        score -= 0.15
    rows.append({
        "tag": tag,
        "clip_2way": twoway,
        "class_top1": clstop,
        "fid": fid,
        "ssim": ssim,
        "pixcorr": p.get("pixcorr"),
        "clip_cosine": p.get("clip_cosine"),
        "pass_gate": pass_gate if tag != "ref_a40_dual" else True,
        "delta_2way": twoway - ref_2way,
        "delta_fid": fid - ref_fid,
        "delta_ssim": ssim - ref_ssim,
        "score": score,
    })
rows.sort(key=lambda x: -x["score"])
gated = [r for r in rows if r["pass_gate"]]
best = gated[0] if gated else rows[0]
summary = {
    "pipeline": "PerceptFlow",
    "claim": "Perception/structure Cond-CFM + low-s img2img/freq/CN inject into frozen a40 semantics",
    "ref_a40": {"clip_2way": ref_2way, "fid": ref_fid, "ssim": ref_ssim},
    "gate_rule": "2way >= ref-0.02 AND fid <= ref+25",
    "best_gated": best,
    "all_ranked": rows,
    "train_report": str(out/"train/percept_flow_train_report.json"),
    "compare_grid": str(out/"compare/compare_grid.png"),
}
(out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
# write best tag for compare
(out/"best_tag.txt").write_text(best["tag"], encoding="utf-8")
PY

echo "===== [8] Compare grid @ $(date -Iseconds) ====="
BEST_TAG="$(cat "${OUT}/best_tag.txt" 2>/dev/null || echo pf_i2i_s35)"
COLS="a40=${SEM_A40},blur=${BLUR}"
for pair in \
  "s28=${OUT}/generation/pf_i2i_s28/generated" \
  "s35=${OUT}/generation/pf_i2i_s35/generated" \
  "ff8=${OUT}/generation/pf_freq_s8/generated" \
  "cn=${OUT}/generation/pf_cn_d30/generated" \
  "best=${OUT}/generation/${BEST_TAG}/generated"; do
  name="${pair%%=*}"; path="${pair#*=}"
  [[ -f "${path}/000.png" ]] && COLS="${COLS},${name}=${path}"
done
"${PYTHON}" scripts/nda/make_compare_grid.py \
  --output-dir "${OUT}/compare" \
  --cell 140 \
  --indices "3,12,28,45,67,88,110,133,156,178,190,199" \
  --metrics-json "${OUT}/clip_2way_report.json" \
  --cols "${COLS}"

du -sh "${OUT}" 2>/dev/null || true
echo "===== DONE PerceptFlow @ $(date -Iseconds) ====="
