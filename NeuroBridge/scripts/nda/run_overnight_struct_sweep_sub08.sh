#!/usr/bin/env bash
# Overnight structure injection sweep — NO long training.
# Reuses frozen a40 + existing Pc/R/LL/ts assets; tries many fuse configs.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/overnight_struct_sweep/sub-08}"
MG="${MG:-${NB_ROOT}/outputs/mg_flow/sub-08}"
TCDA="${TCDA:-${NB_ROOT}/outputs/tcda/sub-08}"
LL="${LL:-${NB_ROOT}/outputs/lowlevel_decoder/sub-08}"
BAL="${BAL:-${NB_ROOT}/outputs/balanced_decoder/sub-08}"
NDA_SS="${NDA_SS:-${NB_ROOT}/outputs/nda_ss/sub-08}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "${OUT}/assets" "${OUT}/generation" "${OUT}/compare" "${OUT}/logs"
cd "${NB_ROOT}"

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"

SEM_A40="${MG}/generation/mg_blend_a40_dual/generated"
EMB_A40="${MG}/train/embeds/blend_nda_cfm_f_a40_test.npy"
PROMPT_DUAL="${MG}/targets/prompts_dual_test.json"
NEIGH="${NDA_SS}/memory/rag_soft5_neighbor_idx_test.npy"
PC="${TCDA}/train/pred_pc_rgb_512"
PR="${TCDA}/train/pred_r_sal_rgb_512"
PF="${TCDA}/train/pred_pf_depth_rgb_512"
LL_RGB="${LL}/vae_head/pred_lowlevel_rgb_512"
TS_LL="${BAL}/generation/ts_ll_a45_b70/generated"

test -f "${SEM_A40}/000.png"
test -f "${PC}/000.png"
test -f "${PR}/000.png"

echo "===== [0] Build R-post (Pc-modulated saliency) @ $(date -Iseconds) ====="
PR2="${OUT}/assets/r_post_pc"
if [[ ! -f "${PR2}/000.png" ]]; then
  "${PYTHON}" scripts/nda/build_r_postprocess.py \
    --sal-dir "${PR}" --pc-dir "${PC}" --output-dir "${PR2}"
else
  echo "[SKIP] r_post"
fi

run_fuse() {
  local tag="$1" struct="$2" alpha="$3"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; return 0; fi
  "${PYTHON}" scripts/nda/generate_lowlevel_decode.py \
    --mode fuse --lowlevel-dir "${struct}" --semantic-dir "${SEM_A40}" \
    --fuse-alpha "${alpha}" --output-dir "${gdir}" --tag "${tag}" --skip-metrics
}

run_sal() {
  local tag="$1" struct="$2" sal="$3" mode="$4" gamma="$5" floor="$6" mmin="$7" mmax="$8"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; return 0; fi
  "${PYTHON}" scripts/nda/generate_tcda_sal_fuse.py \
    --struct-dir "${struct}" --semantic-dir "${SEM_A40}" --saliency-dir "${sal}" \
    --output-dir "${gdir}" --tag "${tag}" --mode "${mode}" \
    --sal-gamma "${gamma}" --sem-floor "${floor}" --m-min "${mmin}" --m-max "${mmax}"
}

run_freq() {
  local tag="$1" struct="$2" sigma="$3"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; return 0; fi
  "${PYTHON}" scripts/nda/generate_freq_fuse.py \
    --struct-dir "${struct}" --semantic-dir "${SEM_A40}" \
    --output-dir "${gdir}" --sigma "${sigma}" --tag "${tag}"
}

echo "===== [1] Alpha fuse grid (Pc / LL / ts) @ $(date -Iseconds) ====="
for a in 0.55 0.65 0.75 0.80 0.85 0.90; do
  run_fuse "fuse_pc_a$(echo $a | tr -d .)" "${PC}" "$a"
done
if [[ -f "${LL_RGB}/000.png" ]]; then
  for a in 0.70 0.80 0.85; do
    run_fuse "fuse_ll_a$(echo $a | tr -d .)" "${LL_RGB}" "$a"
  done
fi
if [[ -f "${TS_LL}/000.png" ]]; then
  for a in 0.50 0.65 0.80; do
    run_fuse "fuse_ts_a$(echo $a | tr -d .)" "${TS_LL}" "$a"
  done
fi

echo "===== [2] sal_fuse grid (v1 R + R-post) @ $(date -Iseconds) ====="
# original R
run_sal "sal_v1_g10" "${PC}" "${PR}" sal_fuse 1.0 0.0 0.0 1.0
run_sal "sal_v1_g15" "${PC}" "${PR}" sal_fuse 1.5 0.0 0.0 1.0
run_sal "sal_v1_fl35_g13" "${PC}" "${PR}" sal_floor 1.3 0.35 0.0 1.0
run_sal "sal_v1_fl50_g13" "${PC}" "${PR}" sal_floor 1.3 0.50 0.0 1.0
# postprocessed R (higher success expectation)
run_sal "sal_post_g10" "${PC}" "${PR2}" sal_fuse 1.0 0.0 0.0 1.0
run_sal "sal_post_g13" "${PC}" "${PR2}" sal_fuse 1.3 0.0 0.0 1.0
run_sal "sal_post_g15" "${PC}" "${PR2}" sal_fuse 1.5 0.0 0.0 1.0
run_sal "sal_post_g18" "${PC}" "${PR2}" sal_fuse 1.8 0.0 0.0 1.0
run_sal "sal_post_fl30_g13" "${PC}" "${PR2}" sal_floor 1.3 0.30 0.0 1.0
run_sal "sal_post_fl40_g13" "${PC}" "${PR2}" sal_floor 1.3 0.40 0.0 1.0
run_sal "sal_post_fl50_g15" "${PC}" "${PR2}" sal_floor 1.5 0.50 0.0 1.0
run_sal "sal_post_clip_fl40" "${PC}" "${PR2}" sal_floor 1.4 0.40 0.20 0.85
if [[ -f "${LL_RGB}/000.png" ]]; then
  run_sal "sal_post_ll_fl40" "${LL_RGB}" "${PR2}" sal_floor 1.3 0.40 0.0 1.0
fi

echo "===== [3] freq fuse grid @ $(date -Iseconds) ====="
for s in 6 8 10 12 16; do
  run_freq "freq_pc_s${s}" "${PC}" "$s"
done
if [[ -f "${LL_RGB}/000.png" ]]; then
  run_freq "freq_ll_s8" "${LL_RGB}" 8
  run_freq "freq_ll_s12" "${LL_RGB}" 12
fi

echo "===== [4] Weak Pf-CN (semantic-friendly) @ $(date -Iseconds) ====="
if [[ -f "${PF}/000.png" && -f "${EMB_A40}" ]]; then
  for cn in 0.25 0.30 0.40; do
    tag="cn_pf$(echo $cn | tr -d .)"
    gdir="${OUT}/generation/${tag}"
    if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; continue; fi
    "${PYTHON}" scripts/nda/generate_cn_ip_decode.py \
      --embed-npy "${EMB_A40}" --neighbor-idx-npy "${NEIGH}" \
      --output-dir "${gdir}" --tag "${tag}" --seed 42 \
      --control-type depth --depth-sample-dir "${PF}" \
      --cn-scale "${cn}" --ip-scale 1.0 --gen-steps 30 --gen-guidance 5.0 \
      --prompts-json "${PROMPT_DUAL}" --skip-metrics
  done
fi

REF_DIR="${OUT}/generation/ref_a40_dual"
mkdir -p "${REF_DIR}"
[[ -e "${REF_DIR}/generated" ]] || ln -sfn "${SEM_A40}" "${REF_DIR}/generated"
echo '{"tag":"ref_a40_dual"}' > "${REF_DIR}/metrics.json"

echo "===== [5] Metrics @ $(date -Iseconds) ====="
# collect all tags with 199.png
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
  --concepts-json "${MG}/targets/concepts_test.json" \
  --output-json "${OUT}/class_consistency.json" \
  --device "${DEVICE}"

echo "===== [6] Rank + pick overnight winners @ $(date -Iseconds) ====="
"${PYTHON}" - <<'PY'
import json
from pathlib import Path
out = Path("/project/peilab/why/NeuroBridge/outputs/overnight_struct_sweep/sub-08")
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
    pass_gate=(twoway>=ref2-0.02) and (fid<=ref_fid+25)
    # overnight objective: maximize SSIM under semantic gate
    score=0.25*twoway+0.20*clstop+0.15*max(0,(320-fid)/170)+0.40*ssim
    if not pass_gate and tag!="ref_a40_dual": score-=0.25
    rows.append({"tag":tag,"clip_2way":twoway,"class_top1":clstop,"fid":fid,"ssim":ssim,
                 "pixcorr":p.get("pixcorr"),"clip_cosine":p.get("clip_cosine"),
                 "pass_gate":True if tag=="ref_a40_dual" else pass_gate,
                 "delta_2way":twoway-ref2,"delta_fid":fid-ref_fid,"delta_ssim":ssim-ref_ssim,"score":score})
rows.sort(key=lambda x:-x["score"])
gated=[r for r in rows if r["pass_gate"]]
best=gated[0] if gated else rows[0]
best_ssim=max(gated,key=lambda r:r["ssim"]) if gated else best
# also best SSIM with 2way drop <=1pp
strict=[r for r in gated if r["delta_2way"]>=-0.01]
best_strict=max(strict,key=lambda r:r["ssim"]) if strict else best_ssim
summary={
  "pipeline":"overnight_struct_sweep",
  "note":"No retrain; sweep fuse/sal/freq/cn on frozen a40 + existing Pc/R/LL/ts",
  "ref_a40":{"clip_2way":ref2,"fid":ref_fid,"ssim":ref_ssim},
  "gate_rule":"2way>=ref-0.02 AND fid<=ref+25",
  "best_gated":best,
  "best_gated_ssim":best_ssim,
  "best_ssim_within_1pp_2way":best_strict,
  "n_configs":len(rows),
  "n_pass_gate":len(gated),
  "all_ranked":rows,
}
(out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
(out/"best_tag.txt").write_text(best["tag"],encoding="utf-8")
print(json.dumps({k:summary[k] for k in summary if k!="all_ranked"},indent=2))
print("TOP10:")
for r in rows[:10]:
    print(f"  {r['tag']:28s} 2way={r['clip_2way']*100:5.1f}% FID={r['fid']:6.1f} SSIM={r['ssim']:.3f} gate={r['pass_gate']}")
PY

BEST="$(cat "${OUT}/best_tag.txt")"
COLS="a40=${SEM_A40},pc=${PC}"
for pair in \
  "postR=${PR2}" \
  "sal=${OUT}/generation/sal_post_fl40_g13/generated" \
  "fuse=${OUT}/generation/fuse_pc_a80/generated" \
  "freq=${OUT}/generation/freq_pc_s8/generated" \
  "cn=${OUT}/generation/cn_pf30/generated" \
  "best=${OUT}/generation/${BEST}/generated"; do
  name="${pair%%=*}"; path="${pair#*=}"
  [[ -f "${path}/000.png" || -f "${path}/000.png" ]] || true
  [[ -f "${path}/000.png" ]] && COLS="${COLS},${name}=${path}"
done
"${PYTHON}" scripts/nda/make_compare_grid.py \
  --output-dir "${OUT}/compare" --cell 120 \
  --indices "3,12,28,45,67,88,110,133,156,178,190,199" \
  --metrics-json "${OUT}/clip_2way_report.json" --cols "${COLS}" || true

du -sh "${OUT}" || true
echo "===== DONE overnight_struct_sweep @ $(date -Iseconds) ====="
