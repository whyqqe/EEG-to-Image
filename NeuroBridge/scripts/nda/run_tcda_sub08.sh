#!/usr/bin/env bash
# TCDA: Tri-Channel Decode Alignment on sub-08
# Freeze semantic S=a40; train multi-granular P (Pc/Pf) + relational R; gated inject.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/tcda/sub-08}"
NDA_SS="${NDA_SS:-${NB_ROOT}/outputs/nda_ss/sub-08}"
MG="${MG:-${NB_ROOT}/outputs/mg_flow/sub-08}"
T1="${T1:-${NB_ROOT}/outputs/top1_structure/sub-08}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "${OUT}/train" "${OUT}/generation" "${OUT}/compare" "${OUT}/logs"
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
test -f "${EMB_A40}" && test -f "${SEM_A40}/000.png" && test -f "${PROMPT_DUAL}"

echo "===== [1] Train TCDA (Pc/Pf/R) @ $(date -Iseconds) ====="
TRAIN_OUT="${OUT}/train"
if [[ ! -f "${TRAIN_OUT}/tcda_train_report.json" ]]; then
  "${PYTHON}" scripts/nda/train_tcda.py \
    --eeg-train-npy "${DEC_TR}" \
    --eeg-test-npy "${DEC_TE}" \
    --depth-train-npy "${DEPTH_TR}" \
    --depth-test-npy "${DEPTH_TE}" \
    --output-dir "${TRAIN_OUT}" \
    --num-epochs 30 \
    --batch-size 64 \
    --lr 2e-4 \
    --device "${DEVICE}" \
    --early-stop-patience 8
else
  echo "[SKIP] train"
fi

PC="${TRAIN_OUT}/pred_pc_rgb_512"
PF="${TRAIN_OUT}/pred_pf_depth_rgb_512"
PR="${TRAIN_OUT}/pred_r_sal_rgb_512"
STR_NPY="${TRAIN_OUT}/pred_strength_test.npy"
test -f "${PC}/000.png" && test -f "${PF}/000.png" && test -f "${PR}/000.png"

echo "===== [2] Injection generation @ $(date -Iseconds) ====="
# 2a fixed low-s img2img from Pc blur + frozen a40
run_i2i() {
  local tag="$1" strength="$2"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; return 0; fi
  "${PYTHON}" scripts/nda/generate_lowlevel_decode.py \
    --mode img2img \
    --lowlevel-dir "${PC}" \
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
run_i2i "tcda_i2i_s28" 0.28
run_i2i "tcda_i2i_s35" 0.35

# 2b mean gated strength (use mean of predicted strengths as global if per-sample pipe unsupported)
if [[ -f "${STR_NPY}" ]]; then
  STR_MEAN="$("${PYTHON}" - <<PY
import numpy as np
s=np.load("${STR_NPY}").astype(float)
print(f"{float(np.clip(s.mean(),0.22,0.42)):.4f}")
PY
)"
  run_i2i "tcda_i2i_sgate" "${STR_MEAN}"
fi

# 2c frequency fuse Pc + a40
run_ff() {
  local tag="$1" sigma="$2"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; return 0; fi
  "${PYTHON}" scripts/nda/generate_freq_fuse.py \
    --struct-dir "${PC}" \
    --semantic-dir "${SEM_A40}" \
    --output-dir "${gdir}" \
    --sigma "${sigma}" \
    --tag "${tag}"
}
run_ff "tcda_freq_s8" 8.0
run_ff "tcda_freq_s12" 12.0

# 2d relational saliency fuse (third branch)
run_sal() {
  local tag="$1" mode="$2" gamma="$3"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; return 0; fi
  "${PYTHON}" scripts/nda/generate_tcda_sal_fuse.py \
    --struct-dir "${PC}" \
    --semantic-dir "${SEM_A40}" \
    --saliency-dir "${PR}" \
    --output-dir "${gdir}" \
    --tag "${tag}" \
    --mode "${mode}" \
    --sal-gamma "${gamma}"
}
run_sal "tcda_sal_fuse" "sal_fuse" 1.0
run_sal "tcda_sal_fuse_g15" "sal_fuse" 1.5
# saliency over i2i result (structure protect on top of semantic refine)
if [[ -f "${OUT}/generation/tcda_i2i_s35/generated/000.png" ]]; then
  run_sal_i2i() {
    local tag="$1"
    local gdir="${OUT}/generation/${tag}"
    if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; return 0; fi
    "${PYTHON}" scripts/nda/generate_tcda_sal_fuse.py \
      --struct-dir "${PC}" \
      --semantic-dir "${OUT}/generation/tcda_i2i_s35/generated" \
      --saliency-dir "${PR}" \
      --output-dir "${gdir}" \
      --tag "${tag}" \
      --mode sal_fuse \
      --sal-gamma 1.0
  }
  run_sal_i2i "tcda_sal_on_i2i35"
fi

# 2e weak depth-CN with Pf + frozen a40
run_cn() {
  local tag="$1" cn="$2"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; return 0; fi
  "${PYTHON}" scripts/nda/generate_cn_ip_decode.py \
    --embed-npy "${EMB_A40}" \
    --neighbor-idx-npy "${NEIGH}" \
    --output-dir "${gdir}" \
    --tag "${tag}" \
    --seed 42 \
    --control-type depth \
    --depth-sample-dir "${PF}" \
    --cn-scale "${cn}" \
    --ip-scale 1.0 \
    --gen-steps 30 \
    --gen-guidance 5.0 \
    --prompts-json "${PROMPT_DUAL}" \
    --skip-metrics
}
run_cn "tcda_cn_pf30" 0.30
run_cn "tcda_cn_pf45" 0.45

# ref symlink
REF_DIR="${OUT}/generation/ref_a40_dual"
mkdir -p "${REF_DIR}"
[[ -e "${REF_DIR}/generated" ]] || ln -sfn "${SEM_A40}" "${REF_DIR}/generated"
echo '{"tag":"ref_a40_dual","note":"frozen MG-Flow a40"}' > "${REF_DIR}/metrics.json"

echo "===== [3] Metrics @ $(date -Iseconds) ====="
TAGS="ref_a40_dual,tcda_i2i_s28,tcda_i2i_s35,tcda_i2i_sgate,tcda_freq_s8,tcda_freq_s12,tcda_sal_fuse,tcda_sal_fuse_g15,tcda_sal_on_i2i35,tcda_cn_pf30,tcda_cn_pf45"
VALID=""
IFS=',' read -ra ARR <<< "${TAGS}"
for t in "${ARR[@]}"; do
  [[ -f "${OUT}/generation/${t}/generated/199.png" ]] && VALID="${VALID:+$VALID,}${t}"
done
"${PYTHON}" scripts/nda/eval_paper_metrics.py \
  --gen-root "${OUT}/generation" \
  --tags "${VALID}" \
  --output-json "${OUT}/paper_metrics.json"

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

"${PYTHON}" scripts/nda/eval_class_consistency.py \
  --gen-dirs "${GEN_ARGS}" \
  --text-concept-npy "${NDA_SS}/clip_text/test/text_concept_clip.npy" \
  --concepts-json "${MG}/targets/concepts_test.json" \
  --output-json "${OUT}/class_consistency.json" \
  --device "${DEVICE}"

echo "===== [4] Quality gate + summary @ $(date -Iseconds) ====="
"${PYTHON}" - <<'PY'
import json
from pathlib import Path
out = Path("/project/peilab/why/NeuroBridge/outputs/tcda/sub-08")
paper = json.loads((out/"paper_metrics.json").read_text()) if (out/"paper_metrics.json").is_file() else {"results":[]}
tw = json.loads((out/"clip_2way_report.json").read_text()) if (out/"clip_2way_report.json").is_file() else {"generation_2way":[]}
cls = json.loads((out/"class_consistency.json").read_text()) if (out/"class_consistency.json").is_file() else {"results":[]}
train = json.loads((out/"train/tcda_train_report.json").read_text()) if (out/"train/tcda_train_report.json").is_file() else {}
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
    pass_gate = (twoway >= ref_2way - 0.02) and (fid <= ref_fid + 25.0)
    score = 0.30*twoway + 0.25*clstop + 0.20*max(0,(320-fid)/170) + 0.25*ssim
    if not pass_gate and tag != "ref_a40_dual":
        score -= 0.15
    rows.append({
        "tag": tag, "clip_2way": twoway, "class_top1": clstop, "fid": fid, "ssim": ssim,
        "pixcorr": p.get("pixcorr"), "clip_cosine": p.get("clip_cosine"),
        "pass_gate": True if tag=="ref_a40_dual" else pass_gate,
        "delta_2way": twoway-ref_2way, "delta_fid": fid-ref_fid, "delta_ssim": ssim-ref_ssim,
        "score": score,
    })
rows.sort(key=lambda x: -x["score"])
gated = [r for r in rows if r["pass_gate"]]
best = gated[0] if gated else rows[0]
summary = {
  "pipeline": "TCDA",
  "claim": "Tri-channel decode alignment: frozen S(a40) + multi-granular P(Pc blur/Pf depth) + relational R(saliency)",
  "train": train,
  "ref_a40": {"clip_2way": ref_2way, "fid": ref_fid, "ssim": ref_ssim},
  "gate_rule": "2way >= ref-0.02 AND fid <= ref+25",
  "best_gated": best,
  "all_ranked": rows,
  "compare_grid": str(out/"compare/compare_grid.png"),
}
(out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(out/"best_tag.txt").write_text(best["tag"], encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "===== [5] Compare grid @ $(date -Iseconds) ====="
BEST_TAG="$(cat "${OUT}/best_tag.txt" 2>/dev/null || echo tcda_i2i_s35)"
COLS="a40=${SEM_A40},pc=${PC},r=${PR}"
for pair in \
  "i2i35=${OUT}/generation/tcda_i2i_s35/generated" \
  "freq=${OUT}/generation/tcda_freq_s8/generated" \
  "sal=${OUT}/generation/tcda_sal_fuse/generated" \
  "sal_i2i=${OUT}/generation/tcda_sal_on_i2i35/generated" \
  "cn=${OUT}/generation/tcda_cn_pf30/generated" \
  "best=${OUT}/generation/${BEST_TAG}/generated"; do
  name="${pair%%=*}"; path="${pair#*=}"
  [[ -f "${path}/000.png" ]] && COLS="${COLS},${name}=${path}"
done
"${PYTHON}" scripts/nda/make_compare_grid.py \
  --output-dir "${OUT}/compare" \
  --cell 130 \
  --indices "3,12,28,45,67,88,110,133,156,178,190,199" \
  --metrics-json "${OUT}/clip_2way_report.json" \
  --cols "${COLS}"

du -sh "${OUT}" 2>/dev/null || true
echo "===== DONE TCDA @ $(date -Iseconds) ====="
