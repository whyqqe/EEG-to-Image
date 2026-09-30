#!/usr/bin/env bash
# AB-STRUCT sub-08 overnight pipeline.
#
# PRIMARY (Part 1) -- Spectral Assembly
#   Perm band-resolved structural anchoring of the latent trajectory, to break the
#   measured `strength` Pareto frontier (structure vs semantics/FID).
#
# SECONDARY (Part 2) -- structure-specialised EEG encoder
#   Tests whether a dedicated encoding branch can raise the structural ceiling at
#   all; the decision gate is the cross-sample `spread` (>0.55 viable, ~0.43 ceiling).
#
# All comparisons hold the frozen semantic path fixed, so any delta is structural.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
DEVICE="${DEVICE:-cuda:0}"
SUB="${SUB:-sub-08}"
NB_SUB="${SUB/-/}"                      # sub-08 -> sub08
OUT="${NB_ROOT}/outputs/ab_struct/${SUB}"
SRC="${NB_ROOT}/outputs/intra_hcma_s/${SUB}"   # frozen semantic + structure assets
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
STD7="${NB_ROOT}/outputs/standard7_protocol"
mkdir -p "${OUT}"/{anchors,generation,logs,prompts,struct} "${NB_ROOT}/outputs/slurm"

cd "${NB_ROOT}"
if [[ -f "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh"
else
  # shellcheck disable=SC1091
  source "/project/peilab/why/eeg-brainit/scripts/activate.sh"
fi
PYTHON="$(command -v python)"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
echo "[INFO] python=${PYTHON} device=${DEVICE}"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }

EMB="${SRC}/blend/mem_decode_a50.npy"
LL_RGB="${SRC}/vae_head/pred_lowlevel_rgb_512"
DEPTH_RGB="${SRC}/depth/pred_depth_rgb_512"
EEG_LAT="${SRC}/vae_head/pred_vae_test.npy"
TRAIN_LAT="${SRC}/vae_cache/train_vae_latents_f16.npy"
NEIGH="${SRC}/memory/rag_soft5_neighbor_idx_test.npy"
ORACLE_PROMPTS="${NB_ROOT}/outputs/hcma_10subj/prompts/prompts_full_hcma_test.json"
require "${EMB}"; require "${EEG_LAT}"; require "${ORACLE_PROMPTS}"
require "${LL_RGB}/000.png"; require "${DEPTH_RGB}/000.png"
require "${TRAIN_LAT}"; require "${NEIGH}"

# ------------------------------------------------------------------ prompts
# Two protocols, generated here so nothing is hidden:
#   oracle  = GT concept name (NOT deployable; kept only for comparability with
#             the existing intra_hs_* grid, which used it)
#   free    = empty prompt, semantics enter only through IP-Adapter (deployable)
FREE_PROMPTS="${OUT}/prompts/free.json"
if [[ ! -f "${FREE_PROMPTS}" ]]; then
  "${PYTHON}" - "${ORACLE_PROMPTS}" "${FREE_PROMPTS}" <<'PY'
import json, sys
n = len(json.load(open(sys.argv[1])))
json.dump([""] * n, open(sys.argv[2], "w"))
print(f"[OK] free prompts ({n}) -> {sys.argv[2]}")
PY
fi

# ============================================================ PART 1 anchors
echo "===== [1] spectral anchors @ $(date -Iseconds) ====="
ANC="${OUT}/anchors"
for cut in 0.0625 0.125 0.25; do
  c=$(echo "${cut}" | tr -d '.')
  [[ -f "${ANC}/lf_eeg_c${c}.npy" ]] && continue
  "${PYTHON}" scripts/nda/build_spectral_anchor.py \
    --eeg-latent-npy "${EEG_LAT}" --output-npy "${ANC}/lf_eeg_c${c}.npy" \
    --mode lf_eeg --cut "${cut}" --report-json "${ANC}/lf_eeg_c${c}.json"
done
[[ -f "${ANC}/full_eeg.npy" ]] || "${PYTHON}" scripts/nda/build_spectral_anchor.py \
  --eeg-latent-npy "${EEG_LAT}" --output-npy "${ANC}/full_eeg.npy" \
  --mode full_eeg --cut 0.0625 --report-json "${ANC}/full_eeg.json"
[[ -f "${ANC}/lf_eeg_hfretr_c00625.npy" ]] || "${PYTHON}" scripts/nda/build_spectral_anchor.py \
  --eeg-latent-npy "${EEG_LAT}" --train-latent-npy "${TRAIN_LAT}" \
  --neighbor-idx-npy "${NEIGH}" --output-npy "${ANC}/lf_eeg_hfretr_c00625.npy" \
  --mode lf_eeg_hf_retr --cut 0.0625 --retr-k 3 \
  --report-json "${ANC}/lf_eeg_hfretr_c00625.json"

# ======================================================== PART 1 generation
echo "===== [2] spectral-assembly generation grid @ $(date -Iseconds) ====="
GEN="${OUT}/generation"
run_sa() {  # tag anchor cut gamma strength cn prompts
  local tag="$1" anc="$2" cut="$3" gamma="$4" strength="$5" cn="$6" pr="$7"
  local gdir="${GEN}/${tag}"
  [[ -f "${gdir}/generated/199.png" ]] && { echo "[SKIP] ${tag}"; return 0; }
  echo "[GEN ] ${tag} cut=${cut} gamma=${gamma} strength=${strength} cn=${cn} prompts=$(basename "${pr}")"
  local a=()
  [[ -n "${anc}" ]] && a=(--anchor-latent-npy "${anc}")
  "${PYTHON}" scripts/nda/generate_spectral_decode.py \
    --embed-npy "${EMB}" --prompts-json "${pr}" "${a[@]}" \
    --depth-rgb-dir "${DEPTH_RGB}" --lowlevel-rgb-dir "${LL_RGB}" \
    --output-dir "${gdir}" --tag "${tag}" \
    --cut "${cut}" --gamma "${gamma}" --strength "${strength}" --cn-scale "${cn}" \
    --ip-scale 1.0 --gen-steps 28 --gen-guidance 5.0 --seed 42
}

# --- frontier controls (same script, anchoring disabled)
run_sa "sa_ctrl_gen_s100"   "" 0 0 1.00 0.00 "${ORACLE_PROMPTS}"   # pure prior, no structure at all
run_sa "sa_ctrl_old_s082"   "" 0 0 0.82 0.25 "${ORACLE_PROMPTS}"   # reproduces the shipped recipe
# --- band sweep, full denoising (prior supplies all texture)
run_sa "sa_lf0625_s100" "${ANC}/lf_eeg_c00625.npy" 0.0625 1.0 1.00 0.00 "${ORACLE_PROMPTS}"
run_sa "sa_lf125_s100"  "${ANC}/lf_eeg_c0125.npy"  0.1250 1.0 1.00 0.00 "${ORACLE_PROMPTS}"
run_sa "sa_lf25_s100"   "${ANC}/lf_eeg_c025.npy"   0.2500 1.0 1.00 0.00 "${ORACLE_PROMPTS}"
# --- mechanism ablations
run_sa "sa_lf0625_g05_s100"  "${ANC}/lf_eeg_c00625.npy" 0.0625 0.5 1.00 0.00 "${ORACLE_PROMPTS}"
run_sa "sa_full_eeg_s100"    "${ANC}/full_eeg.npy"      0.0625 1.0 1.00 0.00 "${ORACLE_PROMPTS}"
run_sa "sa_hfretr_s100"      "${ANC}/lf_eeg_hfretr_c00625.npy" 0.0625 1.0 1.00 0.00 "${ORACLE_PROMPTS}"
# --- complementarity with the depth ControlNet
run_sa "sa_lf0625_cn025_s100" "${ANC}/lf_eeg_c00625.npy" 0.0625 1.0 1.00 0.25 "${ORACLE_PROMPTS}"
# --- anchoring + the old moderate-strength recipe
run_sa "sa_lf0625_s082_cn025" "${ANC}/lf_eeg_c00625.npy" 0.0625 1.0 0.82 0.25 "${ORACLE_PROMPTS}"
# --- deployable-prompt bound (no GT concept leak)
run_sa "sa_lf0625_free_s100" "${ANC}/lf_eeg_c00625.npy" 0.0625 1.0 1.00 0.00 "${FREE_PROMPTS}"

# ==================================================== PART 2 structure head
echo "===== [3] structure-specialised encoder (A) @ $(date -Iseconds) ====="
ST="${OUT}/struct"
DTR="${OUT}/gt_depth/train_depth_64.npy"
DTE="${OUT}/gt_depth/test_depth_64.npy"
if [[ ! -f "${DTR}" ]]; then
  echo "[BUILD] GT depth cache for train (image side; ~16540 images)"
  "${PYTHON}" scripts/nda/build_gt_depth_cache.py \
    --images-root "${IMAGES_ROOT}" --output-dir "${OUT}/gt_depth" \
    --device "${DEVICE}" --splits "train" --batch-size 8 || echo "[WARN] depth cache failed; Part 2 skipped"
fi
[[ -f "${DTE}" ]] || ln -sfn "${NB_ROOT}/outputs/hcma_s_full10/shared/gt_depth/test_depth_64.npy" "${DTE}"

if [[ -f "${DTR}" && -f "${DTE}" ]]; then
  for v in multi_full vae_only; do
    [[ -f "${ST}/${v}/struct_encoder_report.json" ]] && { echo "[SKIP] ${v}"; continue; }
    echo "[TRAIN] structure encoder variant=${v}"
    "${PYTHON}" scripts/nda/ab_struct_encoder.py \
      --eeg-train-npy "${NB_ROOT}/data/things_eeg/preprocessed_eeg/${SUB}/train.npy" \
      --eeg-test-npy  "${NB_ROOT}/data/things_eeg/preprocessed_eeg/${SUB}/test.npy" \
      --feat-train-npy "${SRC}/train/z_decode_vith_train.npy" \
      --feat-test-npy  "${SRC}/train/z_decode_vith_test.npy" \
      --vae-train-npy "${TRAIN_LAT}" \
      --vae-test-npy  "${SRC}/vae_cache/test_vae_latents_f16.npy" \
      --depth-train-npy "${DTR}" --depth-test-npy "${DTE}" \
      --val-split-json "${OUT}/leakfree_split.json" \
      --output-dir "${ST}/${v}" --variant "${v}" \
      --num-epochs 120 --batch-size 128 --device "${DEVICE}" || echo "[WARN] variant ${v} failed"
  done
  [[ -f "${ST}/frozen_only/struct_encoder_report.json" ]] || \
    "${PYTHON}" scripts/nda/ab_struct_encoder.py \
      --eeg-train-npy "${NB_ROOT}/data/things_eeg/preprocessed_eeg/${SUB}/train.npy" \
      --eeg-test-npy  "${NB_ROOT}/data/things_eeg/preprocessed_eeg/${SUB}/test.npy" \
      --feat-train-npy "${SRC}/train/z_decode_vith_train.npy" \
      --feat-test-npy  "${SRC}/train/z_decode_vith_test.npy" \
      --vae-train-npy "${TRAIN_LAT}" \
      --vae-test-npy  "${SRC}/vae_cache/test_vae_latents_f16.npy" \
      --depth-train-npy "${DTR}" --depth-test-npy "${DTE}" \
      --val-split-json "${OUT}/leakfree_split.json" \
      --output-dir "${ST}/frozen_only" --variant frozen_only \
      --num-epochs 60 --batch-size 128 --device "${DEVICE}" || true
  rm -f "${DTR}"   # free ~4GB
else
  echo "[WARN] no depth targets -> Part 2 skipped"
fi

# ============================================================== evaluation
echo "===== [4] standard-7 + FID @ $(date -Iseconds) ====="
cp -f "${STD7}/results.json" "${OUT}/results_std7_backup.json" || true
MAN="${OUT}/manifest_ab_struct.json"
OUT_EVAL="${OUT}" MAN="${MAN}" "${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT_EVAL"])
tags = ["sa_ctrl_gen_s100", "sa_ctrl_old_s082",
        "sa_lf0625_s100", "sa_lf125_s100", "sa_lf25_s100",
        "sa_lf0625_g05_s100", "sa_full_eeg_s100", "sa_hfretr_s100",
        "sa_lf0625_cn025_s100", "sa_lf0625_s082_cn025", "sa_lf0625_free_s100"]
rows = []
for t in tags:
    d = out / "generation" / t / "generated"
    if (d / "199.png").exists():
        rows.append({"tag": t, "display": t,
                     "gen_dir": str(d), "config": str((out / "generation" / t / "metrics.json"))})
Path(os.environ["MAN"]).write_text(json.dumps({"protocol": "standard7", "rows": rows}, indent=2))
print(f"[OK] manifest rows={len(rows)}")
PY
"${PYTHON}" scripts/nda/eval_standard7.py --manifest "${MAN}" \
  --images-root "${IMAGES_ROOT}" --out-dir "${STD7}" --device "${DEVICE}" --batch-size 16
cp -f "${STD7}/results.json" "${OUT}/results_ab_struct.json"
"${PYTHON}" - <<PY
import json
from pathlib import Path
backup = json.loads(Path("${OUT}/results_std7_backup.json").read_text(encoding="utf-8"))
new = json.loads(Path("${OUT}/results_ab_struct.json").read_text(encoding="utf-8"))
by = {r["tag"]: r for r in backup["rows"]}
for r in new["rows"]:
    by[r["tag"]] = r
merged = dict(backup); merged["rows"] = [by[t] for t in by]
Path("${STD7}/results.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
print(f"[OK] merged results.json: {len(backup['rows'])} -> {len(merged['rows'])} rows")
PY

# ============================================================== final report
"${PYTHON}" - <<PY
import json
from pathlib import Path
import numpy as np
out = Path("${OUT}")
res = json.loads((out / "results_ab_struct.json").read_text(encoding="utf-8"))
by = {r["tag"]: r for r in res.get("rows", []) if r.get("n")}
P = {"sa_ctrl_old_s082": "control: shipped recipe (strength .82, cn .25)",
     "sa_ctrl_gen_s100": "control: pure prior, no structure",
     "sa_lf0625_s100": "LF eeg cut=.0625, strength 1.0",
     "sa_lf125_s100": "LF eeg cut=.125, strength 1.0",
     "sa_lf25_s100": "LF eeg cut=.25, strength 1.0",
     "sa_lf0625_g05_s100": "soft anchor gamma=.5",
     "sa_full_eeg_s100": "FULL eeg anchor (no band split)",
     "sa_hfretr_s100": "LF eeg + HF retrieved",
     "sa_lf0625_cn025_s100": "LF eeg + depth-CN .25",
     "sa_lf0625_s082_cn025": "LF eeg + shipped strength .82 + cn .25",
     "sa_lf0625_free_s100": "LF eeg, DEPLOYABLE free prompt"}
keys = ["pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "swav", "fid"]
lines = [f"{'tag':<26}" + "".join(f"{k:>10}" for k in keys)]
for t, d in P.items():
    r = by.get(t)
    if not r:
        continue
    lines.append(f"{t:<26}" + "".join(
        (f"{r.get(k, float('nan')):>10.4f}" if k != "fid" else f"{r.get(k, float('nan')):>10.1f}")
        for k in keys))
ctrl = by.get("sa_ctrl_old_s082"); best = by.get("sa_lf0625_s100")
verdict = "insufficient data"
if ctrl and best:
    d_ssim = best["ssim"] - ctrl["ssim"]; d_alex2 = best["alex2"] - ctrl["alex2"]
    d_fid = best["fid"] - ctrl["fid"]
    dom = d_ssim > 0 and d_alex2 > 0 and d_fid < 0
    verdict = ("BREAKS the Pareto frontier: better structure AND semantics AND FID"
               if dom else
               f"partial: dSSIM={d_ssim:+.4f} dAlex2={d_alex2:+.4f} dFID={d_fid:+.1f}")
rep = {"table": lines, "verdict_vs_shipped_control": verdict,
       "n_configs": len(by), "note": "semantic path frozen; only structural conditioning varies"}
(out / "ab_struct_final_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
print("\n".join(lines)); print(""); print("VERDICT:", verdict)
PY
echo "===== DONE @ $(date -Iseconds) ====="
