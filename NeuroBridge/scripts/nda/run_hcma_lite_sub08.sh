#!/usr/bin/env bash
# HCMA-lite (highest-success): freeze a40 semantics + hierarchical text + luma-matched Pc fuse
# No saliency R, no Pc-img2img, no long retrain.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/hcma_lite/sub-08}"
MG="${MG:-${NB_ROOT}/outputs/mg_flow/sub-08}"
TCDA="${TCDA:-${NB_ROOT}/outputs/tcda/sub-08}"
NDA_SS="${NDA_SS:-${NB_ROOT}/outputs/nda_ss/sub-08}"
T1="${T1:-${NB_ROOT}/outputs/top1_structure/sub-08}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "${OUT}/prompts" "${OUT}/generation" "${OUT}/compare" "${OUT}/logs"
cd "${NB_ROOT}"

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"

EMB_A40="${MG}/train/embeds/blend_nda_cfm_f_a40_test.npy"
SEM_A40="${MG}/generation/mg_blend_a40_dual/generated"
PC="${TCDA}/train/pred_pc_rgb_512"
NEIGH="${NDA_SS}/memory/rag_soft5_neighbor_idx_test.npy"
CONCEPTS="${MG}/targets/concepts_test.json"
PRED_DEPTH="${T1}/track_s/depth_head/pred_depth_rgb_512"
CN_COCA="${T1}/track_s/depth_head/cn_scale_coca.npy"
if [[ ! -d "${PRED_DEPTH}" ]]; then
  PRED_DEPTH="${NB_ROOT}/outputs/overnight_ablation/sub-08/depth_vith/pred_depth_rgb_512"
fi

test -f "${EMB_A40}" && test -f "${SEM_A40}/000.png" && test -f "${PC}/000.png" && test -f "${CONCEPTS}"

echo "===== [1] Hierarchical prompts @ $(date -Iseconds) ====="
if [[ ! -f "${OUT}/prompts/prompts_full_hcma_test.json" ]]; then
  "${PYTHON}" scripts/nda/build_hcma_prompts.py \
    --concepts-json "${CONCEPTS}" \
    --output-dir "${OUT}/prompts"
else
  echo "[SKIP] prompts"
fi

echo "===== [2] Semantic gens (frozen a40 IP + role prompts) @ $(date -Iseconds) ====="
run_sem() {
  local tag="$1" prompt="$2"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; return 0; fi
  local extra=()
  if [[ -f "${CN_COCA}" ]]; then extra+=(--cn-scale-npy "${CN_COCA}"); fi
  if [[ -d "${PRED_DEPTH}" ]]; then
    "${PYTHON}" scripts/nda/generate_cn_ip_decode.py \
      --embed-npy "${EMB_A40}" \
      --neighbor-idx-npy "${NEIGH}" \
      --output-dir "${gdir}" \
      --tag "${tag}" \
      --seed 42 \
      --control-type depth \
      --depth-sample-dir "${PRED_DEPTH}" \
      --cn-scale 0.45 \
      --ip-scale 1.0 \
      --gen-steps 30 \
      --gen-guidance 5.0 \
      --prompts-json "${prompt}" \
      --skip-metrics \
      "${extra[@]}"
  else
    # fallback: no depth dir — still run with neighbor canny? require depth for parity with a40
    echo "[ERR] missing pred depth"; exit 1
  fi
}

run_sem "sem_subj" "${OUT}/prompts/prompts_subj_test.json"
run_sem "sem_subj_det" "${OUT}/prompts/prompts_subj_det_test.json"
run_sem "sem_full_hcma" "${OUT}/prompts/prompts_full_hcma_test.json"

echo "===== [3] Luma-matched Pc fuse @ $(date -Iseconds) ====="
run_luma() {
  local tag="$1" semdir="$2" alpha="$3"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; return 0; fi
  "${PYTHON}" scripts/nda/generate_luma_fuse.py \
    --struct-dir "${PC}" \
    --semantic-dir "${semdir}" \
    --output-dir "${gdir}" \
    --tag "${tag}" \
    --sem-alpha "${alpha}"
}

# reproduce overnight winner with luma fix on frozen dual a40
run_luma "luma_a40_a055" "${SEM_A40}" 0.55
run_luma "luma_a40_a065" "${SEM_A40}" 0.65
run_luma "luma_a40_a075" "${SEM_A40}" 0.75

# hierarchical semantic + structure
for sem in sem_subj sem_subj_det sem_full_hcma; do
  run_luma "luma_${sem}_a055" "${OUT}/generation/${sem}/generated" 0.55
  run_luma "luma_${sem}_a065" "${OUT}/generation/${sem}/generated" 0.65
done

# refs
REF="${OUT}/generation/ref_a40_dual"
mkdir -p "${REF}"
[[ -e "${REF}/generated" ]] || ln -sfn "${SEM_A40}" "${REF}/generated"
echo '{"tag":"ref_a40_dual"}' > "${REF}/metrics.json"

# also keep overnight winner as external ref symlink if present
if [[ -f /project/peilab/why/NeuroBridge/outputs/overnight_struct_sweep/sub-08/generation/fuse_pc_a055/generated/000.png ]]; then
  W="${OUT}/generation/ref_fuse_pc_a055"
  mkdir -p "${W}"
  [[ -e "${W}/generated" ]] || ln -sfn /project/peilab/why/NeuroBridge/outputs/overnight_struct_sweep/sub-08/generation/fuse_pc_a055/generated "${W}/generated"
fi

echo "===== [4] Metrics @ $(date -Iseconds) ====="
VALID=""
for d in "${OUT}/generation"/*; do
  [[ -d "$d" ]] || continue
  t="$(basename "$d")"
  [[ -f "${d}/generated/199.png" ]] || continue
  VALID="${VALID:+$VALID,}${t}"
done
echo "[INFO] tags=${VALID}"

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
  --concepts-json "${CONCEPTS}" \
  --output-json "${OUT}/class_consistency.json" \
  --device "${DEVICE}"

echo "===== [5] Summary (semantic-first) @ $(date -Iseconds) ====="
"${PYTHON}" - <<'PY'
import json
from pathlib import Path
out = Path("/project/peilab/why/NeuroBridge/outputs/hcma_lite/sub-08")
paper = json.loads((out/"paper_metrics.json").read_text()) if (out/"paper_metrics.json").is_file() else {"results":[]}
tw = json.loads((out/"clip_2way_report.json").read_text()) if (out/"clip_2way_report.json").is_file() else {"generation_2way":[]}
cls = json.loads((out/"class_consistency.json").read_text()) if (out/"class_consistency.json").is_file() else {"results":[]}
by_p={r["tag"]:r for r in paper.get("results",[])}
by_2={r["tag"]:r for r in tw.get("generation_2way",[])}
by_c={r["tag"]:r for r in cls.get("results",[])}
ref2=float(by_2.get("ref_a40_dual",{}).get("clip_2way",0))
ref_fid=float(by_p.get("ref_a40_dual",{}).get("fid",999))
ref_ssim=float(by_p.get("ref_a40_dual",{}).get("ssim",0))
rows=[]
for tag in sorted(set(by_p)|set(by_2)|set(by_c)):
    p,w,c=by_p.get(tag,{}),by_2.get(tag,{}),by_c.get(tag,{})
    twoway=float(w.get("clip_2way",0)); clstop=float(c.get("class_top1",0))
    fid=float(p.get("fid",999)); ssim=float(p.get("ssim",0))
    pass_gate=(twoway>=ref2-0.015) and (fid<=ref_fid+20)
    # semantic-first score
    score=0.42*twoway+0.28*clstop+0.20*max(0,(320-fid)/170)+0.10*ssim
    if not pass_gate and tag!="ref_a40_dual": score-=0.20
    rows.append({"tag":tag,"clip_2way":twoway,"class_top1":clstop,"fid":fid,"ssim":ssim,
                 "pixcorr":p.get("pixcorr"),"clip_cosine":p.get("clip_cosine"),
                 "pass_gate":True if tag=="ref_a40_dual" else pass_gate,
                 "delta_2way":twoway-ref2,"delta_fid":fid-ref_fid,"delta_ssim":ssim-ref_ssim,"score":score})
rows.sort(key=lambda x:-x["score"])
gated=[r for r in rows if r["pass_gate"]]
best=gated[0] if gated else rows[0]
best_2way=max(gated,key=lambda r:r["clip_2way"]) if gated else best
summary={
  "pipeline":"HCMA-lite",
  "claim":"Frozen a40 + hierarchical text roles (subj/det/bg) + luma-matched Pc fuse; no R/saliency branch",
  "ref_a40":{"clip_2way":ref2,"fid":ref_fid,"ssim":ref_ssim},
  "gate_rule":"2way>=ref-0.015 AND fid<=ref+20",
  "best_gated":best,
  "best_gated_2way":best_2way,
  "all_ranked":rows,
}
(out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
(out/"best_tag.txt").write_text(best["tag"],encoding="utf-8")
print(json.dumps({k:summary[k] for k in summary if k!="all_ranked"},indent=2))
print("TOP8:")
for r in rows[:8]:
    print(f"  {r['tag']:28s} 2way={r['clip_2way']*100:5.1f}% cls={r['class_top1']*100:4.1f}% FID={r['fid']:6.1f} SSIM={r['ssim']:.3f} gate={r['pass_gate']}")
PY

BEST="$(cat "${OUT}/best_tag.txt")"
COLS="a40=${SEM_A40},subj=${OUT}/generation/sem_subj/generated,full=${OUT}/generation/sem_full_hcma/generated"
for pair in \
  "luma55=${OUT}/generation/luma_a40_a055/generated" \
  "luma_full=${OUT}/generation/luma_sem_full_hcma_a055/generated" \
  "best=${OUT}/generation/${BEST}/generated"; do
  name="${pair%%=*}"; path="${pair#*=}"
  [[ -f "${path}/000.png" ]] && COLS="${COLS},${name}=${path}"
done
"${PYTHON}" scripts/nda/make_compare_grid.py \
  --output-dir "${OUT}/compare" --cell 130 \
  --indices "3,12,28,45,67,88,110,133,156,178,190,199" \
  --metrics-json "${OUT}/clip_2way_report.json" --cols "${COLS}" || true

du -sh "${OUT}" || true
echo "===== DONE HCMA-lite @ $(date -Iseconds) ====="
