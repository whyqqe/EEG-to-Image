#!/usr/bin/env bash
# TCDA-v2: highest-success path = improved R saliency + sal_fuse grid (no Pc-img2img)
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/tcda_salfuse_v2/sub-08}"
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

test -f "${DEC_TR}" && test -f "${EMB_A40}" && test -f "${SEM_A40}/000.png"

echo "===== [1] Train TCDA-v2 (better R, stronger w_r) @ $(date -Iseconds) ====="
TRAIN_OUT="${OUT}/train"
if [[ ! -f "${TRAIN_OUT}/tcda_train_report.json" ]]; then
  "${PYTHON}" scripts/nda/train_tcda.py \
    --eeg-train-npy "${DEC_TR}" \
    --eeg-test-npy "${DEC_TE}" \
    --depth-train-npy "${DEPTH_TR}" \
    --depth-test-npy "${DEPTH_TE}" \
    --output-dir "${TRAIN_OUT}" \
    --num-epochs 28 \
    --batch-size 64 \
    --lr 2e-4 \
    --device "${DEVICE}" \
    --early-stop-patience 7 \
    --w-r 0.60 \
    --w-pc 1.0 \
    --w-ssim 0.40 \
    --w-cfm-r 0.25
else
  echo "[SKIP] train"
fi

PC="${TRAIN_OUT}/pred_pc_rgb_512"
PF="${TRAIN_OUT}/pred_pf_depth_rgb_512"
PR="${TRAIN_OUT}/pred_r_sal_rgb_512"
test -f "${PC}/000.png" && test -f "${PR}/000.png"

echo "===== [2] sal_fuse grid (main success path) @ $(date -Iseconds) ====="
run_sal() {
  local tag="$1" mode="$2" gamma="$3" floor="$4" mmin="$5" mmax="$6"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; return 0; fi
  "${PYTHON}" scripts/nda/generate_tcda_sal_fuse.py \
    --struct-dir "${PC}" \
    --semantic-dir "${SEM_A40}" \
    --saliency-dir "${PR}" \
    --output-dir "${gdir}" \
    --tag "${tag}" \
    --mode "${mode}" \
    --sal-gamma "${gamma}" \
    --sem-floor "${floor}" \
    --m-min "${mmin}" \
    --m-max "${mmax}"
}

# v1 winners + protected variants (sem_floor protects 2-way)
run_sal "sf_g10" sal_fuse 1.0 0.0 0.0 1.0
run_sal "sf_g13" sal_fuse 1.3 0.0 0.0 1.0
run_sal "sf_g15" sal_fuse 1.5 0.0 0.0 1.0
run_sal "sf_g18" sal_fuse 1.8 0.0 0.0 1.0
run_sal "sf_floor35_g13" sal_floor 1.3 0.35 0.0 1.0
run_sal "sf_floor50_g13" sal_floor 1.3 0.50 0.0 1.0
run_sal "sf_floor35_g15" sal_floor 1.5 0.35 0.0 1.0
run_sal "sf_clip_g13" sal_fuse 1.3 0.0 0.25 0.90
run_sal "sf_floor40_clip" sal_floor 1.4 0.40 0.20 0.85

# secondary: weak Pf-CN (v1 semantic-friendly)
if [[ -f "${PF}/000.png" ]]; then
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
  run_cn "cn_pf30" 0.30
fi

REF_DIR="${OUT}/generation/ref_a40_dual"
mkdir -p "${REF_DIR}"
[[ -e "${REF_DIR}/generated" ]] || ln -sfn "${SEM_A40}" "${REF_DIR}/generated"
echo '{"tag":"ref_a40_dual"}' > "${REF_DIR}/metrics.json"

echo "===== [3] Metrics @ $(date -Iseconds) ====="
TAGS="ref_a40_dual,sf_g10,sf_g13,sf_g15,sf_g18,sf_floor35_g13,sf_floor50_g13,sf_floor35_g15,sf_clip_g13,sf_floor40_clip,cn_pf30"
VALID=""
IFS=',' read -ra ARR <<< "${TAGS}"
for t in "${ARR[@]}"; do
  [[ -f "${OUT}/generation/${t}/generated/199.png" ]] && VALID="${VALID:+$VALID,}${t}"
done

"${PYTHON}" scripts/nda/eval_paper_metrics.py \
  --gen-root "${OUT}/generation" --tags "${VALID}" --output-json "${OUT}/paper_metrics.json"

GEN_ARGS=""
IFS=',' read -ra ARR <<< "${VALID}"
for t in "${ARR[@]}"; do
  GEN_ARGS="${GEN_ARGS:+$GEN_ARGS,}${t}=${OUT}/generation/${t}/generated"
done
"${PYTHON}" scripts/nda/eval_clip_2way.py \
  --gen-dirs "${GEN_ARGS}" --output-json "${OUT}/clip_2way_report.json" \
  --device "${DEVICE}" --batch-size 16
"${PYTHON}" scripts/nda/eval_class_consistency.py \
  --gen-dirs "${GEN_ARGS}" \
  --text-concept-npy "${NDA_SS}/clip_text/test/text_concept_clip.npy" \
  --concepts-json "${MG}/targets/concepts_test.json" \
  --output-json "${OUT}/class_consistency.json" \
  --device "${DEVICE}"

echo "===== [4] Gate summary @ $(date -Iseconds) ====="
"${PYTHON}" - <<'PY'
import json
from pathlib import Path
out = Path("/project/peilab/why/NeuroBridge/outputs/tcda_salfuse_v2/sub-08")
paper = json.loads((out/"paper_metrics.json").read_text()) if (out/"paper_metrics.json").is_file() else {"results":[]}
tw = json.loads((out/"clip_2way_report.json").read_text()) if (out/"clip_2way_report.json").is_file() else {"generation_2way":[]}
cls = json.loads((out/"class_consistency.json").read_text()) if (out/"class_consistency.json").is_file() else {"results":[]}
train = json.loads((out/"train/tcda_train_report.json").read_text()) if (out/"train/tcda_train_report.json").is_file() else {}
by_p = {r["tag"]: r for r in paper.get("results", [])}
by_2 = {r["tag"]: r for r in tw.get("generation_2way", [])}
by_c = {r["tag"]: r for r in cls.get("results", [])}
ref2 = float(by_2.get("ref_a40_dual", {}).get("clip_2way", 0))
ref_fid = float(by_p.get("ref_a40_dual", {}).get("fid", 999))
ref_ssim = float(by_p.get("ref_a40_dual", {}).get("ssim", 0))
rows=[]
for tag in sorted(set(by_p)|set(by_2)|set(by_c)):
    p,w,c = by_p.get(tag,{}), by_2.get(tag,{}), by_c.get(tag,{})
    twoway=float(w.get("clip_2way",0)); clstop=float(c.get("class_top1",0))
    fid=float(p.get("fid",999)); ssim=float(p.get("ssim",0))
    pass_gate=(twoway>=ref2-0.02) and (fid<=ref_fid+25.0)
    # prefer structure gains under gate
    score=0.28*twoway+0.22*clstop+0.18*max(0,(320-fid)/170)+0.32*ssim
    if not pass_gate and tag!="ref_a40_dual": score-=0.20
    rows.append({"tag":tag,"clip_2way":twoway,"class_top1":clstop,"fid":fid,"ssim":ssim,
                 "pixcorr":p.get("pixcorr"),"clip_cosine":p.get("clip_cosine"),
                 "pass_gate":True if tag=="ref_a40_dual" else pass_gate,
                 "delta_2way":twoway-ref2,"delta_fid":fid-ref_fid,"delta_ssim":ssim-ref_ssim,"score":score})
rows.sort(key=lambda x:-x["score"])
gated=[r for r in rows if r["pass_gate"]]
# best gated by SSIM among those within 1pp of best 2way in gated set
best=gated[0] if gated else rows[0]
best_ssim=max(gated, key=lambda r:r["ssim"]) if gated else best
summary={
  "pipeline":"TCDA-salfuse-v2",
  "claim":"Improve R spectral-saliency + sal_fuse/sem_floor grid; drop failed Pc-img2img",
  "train":train,
  "ref_a40":{"clip_2way":ref2,"fid":ref_fid,"ssim":ref_ssim},
  "gate_rule":"2way>=ref-0.02 AND fid<=ref+25",
  "best_gated":best,
  "best_gated_ssim":best_ssim,
  "all_ranked":rows,
}
(out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
(out/"best_tag.txt").write_text(best["tag"],encoding="utf-8")
print(json.dumps(summary,indent=2))
PY

BEST_TAG="$(cat "${OUT}/best_tag.txt")"
COLS="a40=${SEM_A40},pc=${PC},r=${PR}"
for pair in \
  "g10=${OUT}/generation/sf_g10/generated" \
  "g15=${OUT}/generation/sf_g15/generated" \
  "fl35=${OUT}/generation/sf_floor35_g13/generated" \
  "fl50=${OUT}/generation/sf_floor50_g13/generated" \
  "cn=${OUT}/generation/cn_pf30/generated" \
  "best=${OUT}/generation/${BEST_TAG}/generated"; do
  name="${pair%%=*}"; path="${pair#*=}"
  [[ -f "${path}/000.png" ]] && COLS="${COLS},${name}=${path}"
done
"${PYTHON}" scripts/nda/make_compare_grid.py \
  --output-dir "${OUT}/compare" --cell 130 \
  --indices "3,12,28,45,67,88,110,133,156,178,190,199" \
  --metrics-json "${OUT}/clip_2way_report.json" --cols "${COLS}"

du -sh "${OUT}" || true
echo "===== DONE TCDA-salfuse-v2 @ $(date -Iseconds) ====="
