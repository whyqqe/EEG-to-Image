#!/usr/bin/env bash
# NMB-DADEM: Decode-Aligned Diffusion with Episodic Memory (sub-08)
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
BASE="${BASE:-${NB_ROOT}/outputs/nb_nmb_sota/sub-08}"
OUT="${OUT:-${NB_ROOT}/outputs/nb_nmb_dadem/sub-08}"

DEVICE="${DEVICE:-cuda:0}"
PYTHON="${PYTHON:-python}"
MAX_IMAGES="${MAX_IMAGES:-0}"
KEEP_SAMPLES="${KEEP_SAMPLES:-8}"

EMBED_DIR="${BASE}/embeds"
MEM_TRAIN="${BASE}/memory/rag_soft5_train_clip_1024.npy"
MEM_TEST="${BASE}/memory/rag_soft5_test_clip_1024.npy"
NEIGH="${BASE}/memory/rag_soft5_neighbor_idx_test.npy"
CLIP_TRAIN="${BRAINIT}/outputs/atm_bridge/clip_img_train_1024.npy"
CLIP_TEST="${BRAINIT}/outputs/atm_bridge/clip_img_test_1024.npy"
GALLERY="${CLIP_TRAIN}"
CONTROL_BLEND="${BASE}/blend_vith/mem_linEns_a50.npy"

DDLEM_E2I="${OUT}/ddlem_e2i"
DDLEM_BI="${OUT}/ddlem_bi"
GEN="${OUT}/generation"
FIG="${OUT}/figures"

mkdir -p "${OUT}" "${DDLEM_E2I}" "${DDLEM_BI}" "${GEN}" "${FIG}"
cd "${NB_ROOT}"

RESULTS_JSON="${OUT}/results_partial.json"
[[ -f "${RESULTS_JSON}" ]] || echo "[]" > "${RESULTS_JSON}"

append_result() {
  local tag="$1"
  local eval_json="$2"
  "${PYTHON}" - <<PY
import json
from pathlib import Path
partial = Path("${RESULTS_JSON}")
arr = json.loads(partial.read_text())
data = json.loads(Path("${eval_json}").read_text())
row = data["results"][0] if data.get("results") else {"tag": "${tag}"}
row["tag"] = "${tag}"
arr.append(row)
partial.write_text(json.dumps(arr, indent=2))
print(f"[result] {row.get('tag')} CLIP={row.get('clip_cosine', 'n/a')} FID={row.get('fid', 'n/a')}")
PY
}

eval_and_prune() {
  local tag="$1"
  local out="${GEN}/${tag}"
  if [[ -f "${out}/generated/000.png" ]] && grep -q "\"${tag}\"" "${OUT}/metrics_dadem.json" 2>/dev/null; then
    echo "[SKIP] ${tag} already evaluated"
    return 0
  fi
  local eval_one="${OUT}/_eval_${tag}.json"
  echo "[EVAL] ${tag}"
  "${PYTHON}" scripts/nb_adapter/eval_clip_fid.py \
    --gen-root "${GEN}" \
    --tags "${tag}" \
    --output-json "${eval_one}" \
    --max-images "${MAX_IMAGES}"
  append_result "${tag}" "${eval_one}"
  rm -f "${eval_one}"
  echo "[PRUNE] ${tag} keep=${KEEP_SAMPLES}"
  "${PYTHON}" scripts/nmb/nmb_prune_generated.py \
    --gen-dir "${out}/generated" \
    --keep "${KEEP_SAMPLES}" \
    --seed 42 \
    --report-json "${FIG}/${tag}_samples.json"
}

run_lowlevel() {
  local tag="$1"
  local embed="$2"
  local strength="${3:-0.4}"
  local strength_npy="${4:-}"
  local out="${GEN}/${tag}"
  if [[ -f "${out}/generated/000.png" ]]; then
    echo "[SKIP GEN] ${tag} images exist"
    eval_and_prune "${tag}"
    return 0
  fi
  echo "[GEN] ${tag} embed=$(basename "${embed}") strength=${strength}"
  local -a extra=()
  if [[ -n "${strength_npy}" ]]; then
    extra+=(--strength-npy "${strength_npy}")
  else
    extra+=(--strength "${strength}")
  fi
  "${PYTHON}" scripts/nb_adapter/generate_rag_lowlevel.py \
    --embed-npy "${embed}" \
    --neighbor-idx-npy "${NEIGH}" \
    --output-dir "${out}" \
    --max-images "${MAX_IMAGES}" \
    --seed 42 \
    --tag "${tag}" \
    --skip-metrics \
    "${extra[@]}"
  eval_and_prune "${tag}"
}

train_or_infer_ddlem() {
  local dir="$1"
  local extra_flag="${2:-}"
  if [[ -f "${dir}/ddlem.pt" && ! -f "${dir}/ddlem_refine_align_test.npy" ]]; then
    echo "[INFER-ONLY] regenerate embeddings in ${dir}"
    "${PYTHON}" scripts/nmb/nmb_ddlem_train.py \
      --embed-dir "${EMBED_DIR}" \
      --mem-train "${MEM_TRAIN}" \
      --mem-test "${MEM_TEST}" \
      --clip-train "${CLIP_TRAIN}" \
      --clip-test "${CLIP_TEST}" \
      --gallery "${GALLERY}" \
      --output-dir "${dir}" \
      --checkpoint "${dir}/ddlem.pt" \
      --infer-only \
      --ddim-steps 30 \
      --warm-t-frac 0.5 \
      --device "${DEVICE}" \
      ${extra_flag}
    return 0
  fi
  if [[ -f "${dir}/ddlem_report.json" && -f "${dir}/ddlem_refine_align_test.npy" ]]; then
    echo "[SKIP] ${dir} ready"
    return 0
  fi
  echo "[TRAIN] ${dir}"
  "${PYTHON}" scripts/nmb/nmb_ddlem_train.py \
    --embed-dir "${EMBED_DIR}" \
    --mem-train "${MEM_TRAIN}" \
    --mem-test "${MEM_TEST}" \
    --clip-train "${CLIP_TRAIN}" \
    --clip-test "${CLIP_TEST}" \
    --gallery "${GALLERY}" \
    --output-dir "${dir}" \
    --epochs 80 \
    --patience 15 \
    --batch-size 512 \
    --timesteps 200 \
    --ddim-steps 30 \
    --warm-t-frac 0.5 \
    --lambda-align 1.0 \
    --lambda-e2i 0.5 \
    --device "${DEVICE}" \
    ${extra_flag}
}

echo "===== [1] DDLEM-E2I @ $(date -Iseconds) ====="
train_or_infer_ddlem "${DDLEM_E2I}"

echo "===== [2] DDLEM-Bi @ $(date -Iseconds) ====="
train_or_infer_ddlem "${DDLEM_BI}" "--bidirectional --lambda-i2e 0.5"

echo "===== [3] Confidence blend for flagship @ $(date -Iseconds) ====="
CONF_DIR="${OUT}/conf_blend"
mkdir -p "${CONF_DIR}"
if [[ ! -f "${CONF_DIR}/dsda_report.json" ]]; then
  "${PYTHON}" scripts/nmb/nmb_dsda_build.py \
    --mem-npy "${MEM_TEST}" \
    --proj-npy "${DDLEM_BI}/ddlem_refine_align_test.npy" \
    --output-dir "${CONF_DIR}" \
    --alpha-base 0.5 --conf-k 1.0 --s-hi 0.45 --s-lo 0.32
fi

echo "===== [4] Generation paths @ $(date -Iseconds) ====="

# A: overnight best control
if [[ -f "${CONTROL_BLEND}" ]]; then
  run_lowlevel "A_control_linEns_s04" "${CONTROL_BLEND}" 0.4
fi

# B: align head direct (memory-conditioned regression baseline)
run_lowlevel "B_ddlem_align" "${DDLEM_E2I}/ddlem_align_test.npy" 0.4

# C: DDIM refine from align_head (semantic init)
run_lowlevel "C_ddlem_refine_align" "${DDLEM_E2I}/ddlem_refine_align_test.npy" 0.4

# D: DDIM refine from mem + blend
run_lowlevel "D_ddlem_refine_mem_blend" "${DDLEM_E2I}/ddlem_blend_refine_mem_test.npy" 0.4

# E: DDLEM-Bi refine from align
run_lowlevel "E_ddlem_bi_refine_align" "${DDLEM_BI}/ddlem_refine_align_test.npy" 0.4

# F: flagship — Bi refine align/mem blend + adaptive strength
run_lowlevel "F_ddlem_bi_conf_adapt" \
  "${CONF_DIR}/dsda_conf_blend.npy" 0.4 \
  "${CONF_DIR}/dsda_adaptive_strength.npy"

echo "===== [5] Summary @ $(date -Iseconds) ====="
"${PYTHON}" - <<PY
import json
from pathlib import Path

out = Path("${OUT}")
rows = json.loads((out / "results_partial.json").read_text())
ranked = sorted(rows, key=lambda r: r.get("clip_cosine", 0), reverse=True)
summary = {
    "method": "NMB-DADEM",
    "subject": "sub-08",
    "n_paths": len(rows),
    "best": ranked[0] if ranked else None,
    "top3": ranked[:3],
    "baseline_ref": {"overnight_best": 0.412, "d2_fosa_fid": 146.33},
    "ddlem_e2i_report": json.loads((out / "ddlem_e2i/ddlem_report.json").read_text())
        if (out / "ddlem_e2i/ddlem_report.json").is_file() else None,
    "ddlem_bi_report": json.loads((out / "ddlem_bi/ddlem_report.json").read_text())
        if (out / "ddlem_bi/ddlem_report.json").is_file() else None,
    "all_results": rows,
}
(out / "summary_dadem.json").write_text(json.dumps(summary, indent=2))
(out / "metrics_dadem.json").write_text(json.dumps({"results": rows}, indent=2))
print(json.dumps({"best": summary["best"], "top3": summary["top3"]}, indent=2))
PY

echo "===== DONE NMB-DADEM @ $(date -Iseconds) ====="
