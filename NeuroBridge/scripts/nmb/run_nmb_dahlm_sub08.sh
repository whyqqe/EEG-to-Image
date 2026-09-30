#!/usr/bin/env bash
# DA-HLM-S: highest-probability SOTA scheme for sub-08
#
# Unified theory in practice:
#   DAH  — learnable mem⊕semantic binding (generalizes mem_linEns)
#   ATM DiffusionPrior — ViT-H manifold refine (cond = [z_nb, mem])
#   Hybrid — blend DAH + DiffPrior
#
# Generation paths vs control (mem_linEns_s04 @ 0.412)
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
BASE="${BASE:-${NB_ROOT}/outputs/nb_nmb_sota/sub-08}"
OUT="${OUT:-${NB_ROOT}/outputs/nb_nmb_dahlm/sub-08}"

DEVICE="${DEVICE:-cuda:0}"
PYTHON="${PYTHON:-python}"
MAX_IMAGES="${MAX_IMAGES:-0}"
KEEP_SAMPLES="${KEEP_SAMPLES:-8}"

EMBED_DIR="${BASE}/embeds"
MEM_TRAIN="${BASE}/memory/rag_soft5_train_clip_1024.npy"
MEM_TEST="${BASE}/memory/rag_soft5_test_clip_1024.npy"
FUSION_MEM_TRAIN="${BASE}/memory/fusion_mem_train.npy"
NEIGH="${BASE}/memory/rag_soft5_neighbor_idx_test.npy"
CLIP_TRAIN="${BRAINIT}/outputs/atm_bridge/clip_img_train_1024.npy"
CLIP_TEST="${BRAINIT}/outputs/atm_bridge/clip_img_test_1024.npy"
GALLERY="${CLIP_TRAIN}"
BRIDGE_PT="${BASE}/bridge_vith/linear_adapter.pt"
CONTROL_BLEND="${BASE}/blend_vith/mem_linEns_a50.npy"

DAHLM="${OUT}/dahlm"
GEN="${OUT}/generation"
FIG="${OUT}/figures"

mkdir -p "${OUT}" "${DAHLM}" "${GEN}" "${FIG}"
cd "${NB_ROOT}"

RESULTS_JSON="${OUT}/results_partial.json"
[[ -f "${RESULTS_JSON}" ]] || echo "[]" > "${RESULTS_JSON}"

append_result() {
  local tag="$1" eval_json="$2"
  "${PYTHON}" - <<PY
import json
from pathlib import Path
arr = json.loads(Path("${RESULTS_JSON}").read_text())
row = json.loads(Path("${eval_json}").read_text())["results"][0]
row["tag"] = "${tag}"
arr.append(row)
Path("${RESULTS_JSON}").write_text(json.dumps(arr, indent=2))
print(f"[result] ${tag} CLIP={row.get('clip_cosine')} FID={row.get('fid')}")
PY
}

eval_and_prune() {
  local tag="$1"
  local eval_one="${OUT}/_eval_${tag}.json"
  "${PYTHON}" scripts/nb_adapter/eval_clip_fid.py \
    --gen-root "${GEN}" --tags "${tag}" \
    --output-json "${eval_one}" --max-images "${MAX_IMAGES}"
  append_result "${tag}" "${eval_one}"
  rm -f "${eval_one}"
  "${PYTHON}" scripts/nmb/nmb_prune_generated.py \
    --gen-dir "${GEN}/${tag}/generated" --keep "${KEEP_SAMPLES}" \
    --seed 42 --report-json "${FIG}/${tag}_samples.json"
}

run_lowlevel() {
  local tag="$1" embed="$2" strength="${3:-0.4}" strength_npy="${4:-}"
  echo "[GEN] ${tag}"
  local -a extra=()
  if [[ -n "${strength_npy}" ]]; then extra+=(--strength-npy "${strength_npy}")
  else extra+=(--strength "${strength}"); fi
  "${PYTHON}" scripts/nb_adapter/generate_rag_lowlevel.py \
    --embed-npy "${embed}" --neighbor-idx-npy "${NEIGH}" \
    --output-dir "${GEN}/${tag}" --max-images "${MAX_IMAGES}" \
    --seed 42 --tag "${tag}" --skip-metrics "${extra[@]}"
  eval_and_prune "${tag}"
}

echo "===== [1] Train DA-HLM-S @ $(date -Iseconds) ====="
if [[ ! -f "${DAHLM}/dahlm_report.json" ]]; then
  "${PYTHON}" scripts/nmb/nmb_dahlm_train.py \
    --embed-dir "${EMBED_DIR}" \
    --mem-train "${MEM_TRAIN}" \
    --mem-test "${MEM_TEST}" \
    --fusion-mem-train "${FUSION_MEM_TRAIN}" \
    --clip-train "${CLIP_TRAIN}" \
    --clip-test "${CLIP_TEST}" \
    --bridge-pt "${BRIDGE_PT}" \
    --gallery "${GALLERY}" \
    --output-dir "${DAHLM}" \
    --dah-epochs 80 \
    --diff-epochs 40 \
    --patience 15 \
    --diff-patience 10 \
    --diff-steps 25 \
    --lambda-sota 0.3 \
    --sota-alpha 0.5 \
    --blend-dp 0.5 \
    --device "${DEVICE}"
else
  echo "[SKIP] DA-HLM-S trained"
fi

echo "===== [2] Adaptive strength for hybrid @ $(date -Iseconds) ====="
CONF="${OUT}/conf_hybrid"
mkdir -p "${CONF}"
if [[ ! -f "${CONF}/dsda_report.json" ]]; then
  "${PYTHON}" scripts/nmb/nmb_dsda_build.py \
    --mem-npy "${MEM_TEST}" \
    --proj-npy "${DAHLM}/hybrid_test.npy" \
    --output-dir "${CONF}" \
    --alpha-base 0.5 --conf-k 1.0 --s-hi 0.45 --s-lo 0.32
fi

echo "===== [3] Generation @ $(date -Iseconds) ====="

# A: current SOTA control
run_lowlevel "A_control_linEns_s04" "${CONTROL_BLEND}" 0.4

# B: SOTA teacher reference (recomputed mem⊕bridge)
run_lowlevel "B_sota_teacher_ref" "${DAHLM}/sota_teacher_test.npy" 0.4

# C: DAH only (unified learnable binding)
run_lowlevel "C_dah" "${DAHLM}/dah_test.npy" 0.4

# D: ATM DiffPrior only (cond=[z,mem])
run_lowlevel "D_diffprior" "${DAHLM}/diffprior_test.npy" 0.4

# E: Hybrid DAH + DiffPrior (flagship)
run_lowlevel "E_hybrid_dp50" "${DAHLM}/hybrid_test.npy" 0.4

# F: DAH + SOTA teacher ensemble
run_lowlevel "F_dah_sota_ens" "${DAHLM}/dah_sota_ens_test.npy" 0.4

# G: Hybrid + confidence blend + adaptive strength
run_lowlevel "G_hybrid_conf_adapt" \
  "${CONF}/dsda_conf_blend.npy" 0.4 \
  "${CONF}/dsda_adaptive_strength.npy"

# H: strength sweep on hybrid (best overnight was s=0.4)
run_lowlevel "H_hybrid_s035" "${DAHLM}/hybrid_test.npy" 0.35
run_lowlevel "I_hybrid_s045" "${DAHLM}/hybrid_test.npy" 0.45

echo "===== [4] Summary @ $(date -Iseconds) ====="
"${PYTHON}" - <<PY
import json
from pathlib import Path
out = Path("${OUT}")
rows = json.loads((out / "results_partial.json").read_text())
ranked = sorted(rows, key=lambda r: r.get("clip_cosine", 0), reverse=True)
summary = {
    "method": "DA-HLM-S",
    "theory": "Decode-Aligned Head + ATM DiffusionPrior, unified ViT-H space",
    "subject": "sub-08",
    "n_paths": len(rows),
    "best": ranked[0],
    "top5": ranked[:5],
    "baseline_ref": {"overnight_best": 0.412, "d2_fosa_fid": 146.33},
    "dahlm_report": json.loads((out / "dahlm/dahlm_report.json").read_text()) if (out / "dahlm/dahlm_report.json").is_file() else None,
    "all_results": rows,
}
(out / "summary_dahlm.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({"best": summary["best"], "top5": summary["top5"]}, indent=2))
PY

echo "===== DONE DA-HLM-S @ $(date -Iseconds) ====="
