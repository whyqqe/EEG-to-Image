#!/usr/bin/env bash
# True LOSO for HCMA / MG-Flow:
#   For each held-out subject k ∈ {1..10}:
#     1) pretrain MG-Flow on the other 9 subjects (from scratch)
#     2) freeze DualSemanticHeads; FT CFM/gate/to_gen on subject k only
#     3) HCMA generate + MindEye/ATM metrics on subject k test
#
# Addresses cross-subject generalization critique vs pooled 10-subj pretrain.
# Reuses z_ret/memory/prompts from outputs/hcma_10subj (symlink); writes to new OUT_ROOT.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
SRC="${SRC:-${NB_ROOT}/outputs/hcma_10subj}"
OUT_ROOT="${OUT_ROOT:-${NB_ROOT}/outputs/hcma_loso}"
MG08="${MG08:-${NB_ROOT}/outputs/mg_flow/sub-08}"
NDA_SS="${NDA_SS:-${NB_ROOT}/outputs/nda_ss/sub-08}"
PRED_DEPTH="${PRED_DEPTH:-${NB_ROOT}/outputs/top1_structure/sub-08/track_s/depth_head/pred_depth_rgb_512}"
CN_COCA="${CN_COCA:-${NB_ROOT}/outputs/top1_structure/sub-08/track_s/depth_head/cn_scale_coca.npy}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
# Comma-separated holdouts to run (default: all 10). Example: HOLDOUTS=8 for pilot.
HOLDOUTS="${HOLDOUTS:-1,2,3,4,5,6,7,8,9,10}"
SUBJECTS_ALL=(1 2 3 4 5 6 7 8 9 10)

CLIP_TRAIN="${CLIP_TRAIN:-${BRAINIT}/outputs/atm_bridge/clip_img_train_1024.npy}"
CLIP_TEST="${CLIP_TEST:-${BRAINIT}/outputs/atm_bridge/clip_img_test_1024.npy}"
if [[ ! -f "${CLIP_TRAIN}" ]]; then
  CLIP_TRAIN="${NDA_SS}/train/decode_vith1024_train_clip_1024.npy"
  CLIP_TEST="${NDA_SS}/train/decode_vith1024_test_clip_1024.npy"
fi

mkdir -p "${OUT_ROOT}/logs" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" || true
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
require "${PRED_DEPTH}/000.png"
require "${CLIP_TRAIN}"; require "${CLIP_TEST}"
require "${NDA_SS}/clip_text/test/text_concept_clip.npy"

echo "{\"pipeline\":\"hcma_loso\",\"started\":\"$(date -Iseconds)\",\"holdouts\":\"${HOLDOUTS}\",\"protocol\":\"9-pretrain + 1-FT per fold\"}" \
  > "${OUT_ROOT}/job_running.json"

# ---------- shared prompts + rebuild train targets if pruned ----------
PROMPT_FULL="${SRC}/prompts/prompts_full_hcma_test.json"
CONCEPTS="${MG08}/targets/concepts_test.json"
require "${PROMPT_FULL}"
TARGETS="${OUT_ROOT}/shared_targets"
mkdir -p "${TARGETS}"
if [[ ! -f "${TARGETS}/t_coarse_train.npy" ]]; then
  echo "===== rebuild MG-Flow text targets @ $(date -Iseconds) ====="
  "${PYTHON}" scripts/nda/build_mg_flow_targets.py \
    --clip-text-root "${NDA_SS}/clip_text" \
    --output-dir "${TARGETS}"
fi
require "${TARGETS}/t_coarse_train.npy"
require "${TARGETS}/t_fine_train.npy"
require "${TARGETS}/t_coarse_test.npy"

TAG="hcma_loso_a40"
IFS=',' read -r -a HOLD_ARR <<< "${HOLDOUTS}"

run_fold() {
  local HOLD="$1"
  local HTAG
  HTAG=$(printf "holdout_%02d" "${HOLD}")
  local FOUT="${OUT_ROOT}/${HTAG}"
  local STAG
  STAG=$(printf "sub-%02d" "${HOLD}")
  echo "########## LOSO ${HTAG} (train on others, FT+eval on ${STAG}) @ $(date -Iseconds) ##########"
  mkdir -p "${FOUT}/shared/pool" "${FOUT}/${STAG}/"{ft,generation,metrics,compare}

  # symlink held-out subject assets
  [[ -e "${FOUT}/${STAG}/zret" ]] || ln -sfn "${SRC}/${STAG}/zret" "${FOUT}/${STAG}/zret"
  [[ -e "${FOUT}/${STAG}/memory" ]] || ln -sfn "${SRC}/${STAG}/memory" "${FOUT}/${STAG}/memory"
  require "${FOUT}/${STAG}/zret/z_ret_train.npy"
  require "${FOUT}/${STAG}/memory/rag_soft5_train_clip_1024.npy"

  local POOL="${FOUT}/shared/pool"
  local SHARED_TRAIN="${FOUT}/shared/mg_flow"
  local SHARED_CKPT="${SHARED_TRAIN}/checkpoints/best.pt"

  # build 9-subject pool (exclude HOLD)
  if [[ ! -f "${POOL}/z_ret_train.npy" ]]; then
    HOLD="${HOLD}" SRC="${SRC}" POOL="${POOL}" "${PYTHON}" - <<'PY'
import os
from pathlib import Path
import numpy as np
src = Path(os.environ["SRC"])
pool = Path(os.environ["POOL"]); pool.mkdir(parents=True, exist_ok=True)
hold = int(os.environ["HOLD"])
z_tr, z_te, n_tr, n_te = [], [], [], []
kept = []
for sid in range(1, 11):
    if sid == hold:
        continue
    stag = f"sub-{sid:02d}"
    zt = src / stag / "zret" / "z_ret_train.npy"
    ze = src / stag / "zret" / "z_ret_test.npy"
    nt = src / stag / "memory" / "rag_soft5_train_clip_1024.npy"
    ne = src / stag / "memory" / "rag_soft5_test_clip_1024.npy"
    assert zt.is_file() and nt.is_file(), stag
    z_tr.append(np.load(zt)); z_te.append(np.load(ze))
    n_tr.append(np.load(nt)); n_te.append(np.load(ne))
    kept.append(sid)
ztr = np.concatenate(z_tr, 0).astype(np.float32)
zte = np.concatenate(z_te, 0).astype(np.float32)
ntr = np.concatenate(n_tr, 0).astype(np.float32)
nte = np.concatenate(n_te, 0).astype(np.float32)
np.save(pool / "z_ret_train.npy", ztr)
np.save(pool / "z_ret_test.npy", zte)
np.save(pool / "nda_train.npy", ntr)
np.save(pool / "nda_test.npy", nte)
meta = {"holdout": hold, "train_subjects": kept, "z_ret_train": list(ztr.shape), "z_ret_test": list(zte.shape)}
(pool / "pool_meta.json").write_text(__import__("json").dumps(meta, indent=2))
print(meta)
PY
  else
    echo "[SKIP] pool ${HTAG}"
  fi

  # [1] pretrain on 9 subjects from scratch
  if [[ ! -f "${SHARED_CKPT}" ]]; then
    "${PYTHON}" scripts/nda/mg_flow_train.py \
      --z-ret-train "${POOL}/z_ret_train.npy" \
      --z-ret-test "${POOL}/z_ret_test.npy" \
      --clip-img-train "${CLIP_TRAIN}" \
      --clip-img-test "${CLIP_TEST}" \
      --t-coarse-train "${TARGETS}/t_coarse_train.npy" \
      --t-fine-train "${TARGETS}/t_fine_train.npy" \
      --t-coarse-test "${TARGETS}/t_coarse_test.npy" \
      --t-fine-test "${TARGETS}/t_fine_test.npy" \
      --nda-decode-train "${POOL}/nda_train.npy" \
      --nda-decode-test "${POOL}/nda_test.npy" \
      --text-concept-test "${NDA_SS}/clip_text/test/text_concept_clip.npy" \
      --output-dir "${SHARED_TRAIN}" \
      --tile-targets \
      --epochs 40 \
      --batch-size 512 \
      --lr 1e-4 \
      --early-stop 10 \
      --ode-steps 16 \
      --device "${DEVICE}"
    # prune bulky embeds
    if [[ -d "${SHARED_TRAIN}/embeds" ]]; then
      find "${SHARED_TRAIN}/embeds" -name '*_train.npy' -delete || true
    fi
  else
    echo "[SKIP] shared pretrain ${HTAG}"
  fi
  require "${SHARED_CKPT}"

  # free pool train arrays after pretrain (keep meta + can rebuild)
  rm -f "${POOL}/z_ret_train.npy" "${POOL}/nda_train.npy" || true

  # [2] FT held-out subject only
  local EMB="${FOUT}/${STAG}/ft/embeds/blend_nda_cfm_f_a40_test.npy"
  if [[ ! -f "${EMB}" ]]; then
    "${PYTHON}" scripts/nda/mg_flow_train.py \
      --z-ret-train "${FOUT}/${STAG}/zret/z_ret_train.npy" \
      --z-ret-test "${FOUT}/${STAG}/zret/z_ret_test.npy" \
      --clip-img-train "${CLIP_TRAIN}" \
      --clip-img-test "${CLIP_TEST}" \
      --t-coarse-train "${TARGETS}/t_coarse_train.npy" \
      --t-fine-train "${TARGETS}/t_fine_train.npy" \
      --t-coarse-test "${TARGETS}/t_coarse_test.npy" \
      --t-fine-test "${TARGETS}/t_fine_test.npy" \
      --nda-decode-train "${FOUT}/${STAG}/memory/rag_soft5_train_clip_1024.npy" \
      --nda-decode-test "${FOUT}/${STAG}/memory/rag_soft5_test_clip_1024.npy" \
      --text-concept-test "${NDA_SS}/clip_text/test/text_concept_clip.npy" \
      --output-dir "${FOUT}/${STAG}/ft" \
      --init-ckpt "${SHARED_CKPT}" \
      --freeze-backbone \
      --epochs 10 \
      --batch-size 512 \
      --lr 3e-5 \
      --early-stop 5 \
      --ode-steps 16 \
      --device "${DEVICE}"
    if [[ -d "${FOUT}/${STAG}/ft/embeds" ]]; then
      find "${FOUT}/${STAG}/ft/embeds" -type f ! -name 'blend_nda_cfm_f_a40_test.npy' -delete || true
    fi
  else
    echo "[SKIP] FT ${HTAG}"
  fi
  require "${EMB}"

  # [3] generate
  local GDIR="${FOUT}/${STAG}/generation/${TAG}"
  if [[ ! -f "${GDIR}/generated/199.png" ]]; then
    [[ -d "${GDIR}/generated" ]] && [[ ! -f "${GDIR}/generated/199.png" ]] && rm -rf "${GDIR}"
    local GEN_EXTRA=()
    [[ -f "${CN_COCA}" ]] && GEN_EXTRA+=(--cn-scale-npy "${CN_COCA}")
    "${PYTHON}" scripts/nda/generate_cn_ip_decode.py \
      --embed-npy "${EMB}" \
      --neighbor-idx-npy "${FOUT}/${STAG}/memory/rag_soft5_neighbor_idx_test.npy" \
      --output-dir "${GDIR}" \
      --tag "${TAG}" \
      --seed 42 \
      --control-type depth \
      --depth-sample-dir "${PRED_DEPTH}" \
      --cn-scale 0.45 \
      --ip-scale 1.0 \
      --gen-steps 30 \
      --gen-guidance 5.0 \
      --prompts-json "${PROMPT_FULL}" \
      --skip-metrics \
      "${GEN_EXTRA[@]}"
  else
    echo "[SKIP] gen ${HTAG}"
  fi

  # [4] metrics (same suite as HCMA 10subj)
  local SOUT="${FOUT}/${STAG}"
  if [[ ! -f "${SOUT}/metrics/paper_metrics.json" ]]; then
    "${PYTHON}" scripts/nda/eval_paper_metrics.py \
      --gen-root "${SOUT}/generation" --tags "${TAG}" \
      --output-json "${SOUT}/metrics/paper_metrics.json"
  fi
  if [[ ! -f "${SOUT}/metrics/clip_2way_report.json" ]]; then
    "${PYTHON}" scripts/nda/eval_clip_2way.py \
      --gen-dirs "${TAG}=${GDIR}/generated" \
      --output-json "${SOUT}/metrics/clip_2way_report.json" \
      --device "${DEVICE}" --batch-size 16
  fi
  if [[ ! -f "${SOUT}/metrics/class_consistency.json" ]]; then
    "${PYTHON}" scripts/nda/eval_class_consistency.py \
      --gen-dirs "${TAG}=${GDIR}/generated" \
      --text-concept-npy "${NDA_SS}/clip_text/test/text_concept_clip.npy" \
      --concepts-json "${CONCEPTS}" \
      --output-json "${SOUT}/metrics/class_consistency.json" \
      --device "${DEVICE}"
  fi
  if [[ ! -f "${SOUT}/metrics/erdc_2wc.json" ]]; then
    "${PYTHON}" "${BRAINIT}/scripts/erdc_twoway_metrics.py" \
      --gen-dir "${GDIR}/generated" \
      --images-root /project/peilab/why/data/images_set \
      --output-json "${SOUT}/metrics/erdc_2wc.json" \
      --tag "${STAG}_${TAG}"
  fi
  # drop regenerable cache
  find "${GDIR}" -type d -name '_twoway_cache' -exec rm -rf {} + 2>/dev/null || true

  HOLD="${HOLD}" TAG="${TAG}" SOUT="${SOUT}" HTAG="${HTAG}" "${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["SOUT"])
paper = json.loads((out / "metrics/paper_metrics.json").read_text())["results"][0]
tw = json.loads((out / "metrics/clip_2way_report.json").read_text())["generation_2way"][0]
cls = json.loads((out / "metrics/class_consistency.json").read_text())["results"][0]
twc = json.loads((out / "metrics/erdc_2wc.json").read_text())
summary = {
  "protocol": "LOSO",
  "holdout": int(os.environ["HOLD"]),
  "fold": os.environ["HTAG"],
  "tag": os.environ["TAG"],
  "pretrain": "9 subjects excluding holdout",
  "finetune": "holdout only, freeze DualSemanticHeads, delta CFM/gate/to_gen",
  "clip_2way_vitl": tw.get("clip_2way"),
  "class_top1": cls.get("class_top1"),
  "fid": paper.get("fid"),
  "ssim_skimage": paper.get("ssim"),
  "pixcorr": paper.get("pixcorr"),
  "clip_cosine": paper.get("clip_cosine"),
  "twoway": twc.get("twoway", twc),
}
(out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
}

for H in "${HOLD_ARR[@]}"; do
  H=$(echo "${H}" | tr -d ' ')
  [[ -n "${H}" ]] || continue
  run_fold "${H}"
done

# ---------- aggregate vs pooled-10subj HCMA ----------
echo "===== Aggregate LOSO @ $(date -Iseconds) ====="
OUT_ROOT="${OUT_ROOT}" SRC="${SRC}" "${PYTHON}" - <<'PY'
import json
from pathlib import Path
import numpy as np

root = Path(__import__("os").environ["OUT_ROOT"])
src = Path(__import__("os").environ["SRC"])
rows = []
for p in sorted(root.glob("holdout_*/sub-*/summary.json")):
    rows.append(json.loads(p.read_text()))

def mean_std(key, nested=None):
    vals = []
    for r in rows:
        if nested:
            cur = r
            ok = True
            for k in nested:
                if not isinstance(cur, dict) or k not in cur:
                    ok = False
                    break
                cur = cur[k]
            if ok and cur is not None:
                vals.append(float(cur))
        else:
            if r.get(key) is not None:
                vals.append(float(r[key]))
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))

# baseline non-LOSO
base_rows = []
for p in sorted(src.glob("sub-*/summary.json")):
    base_rows.append(json.loads(p.read_text()))

def bmean(key, nested=None):
    vals = []
    for r in base_rows:
        if nested:
            cur = r.get("twoway", {})
            if isinstance(cur, dict) and key in cur:
                vals.append(float(cur[key]))
        elif r.get(key) is not None:
            vals.append(float(r[key]))
    return float(np.mean(vals)) if vals else None

agg = {
  "pipeline": "hcma_loso",
  "n_folds": len(rows),
  "protocol": "For each holdout k: pretrain on 9 others from scratch → FT k (heads frozen) → eval k",
  "ours_loso_mean": {
    "clip_2way_vitl": mean_std("clip_2way_vitl")[0],
    "class_top1": mean_std("class_top1")[0],
    "fid": mean_std("fid")[0],
    "pixcorr": mean_std("pixcorr")[0],
    "ssim_skimage": mean_std("ssim_skimage")[0],
    "clip_cosine": mean_std("clip_cosine")[0],
    "twoway_clip": mean_std(None, ["twoway", "clip"])[0],
    "twoway_alex2": mean_std(None, ["twoway", "alex2"])[0],
    "twoway_alex5": mean_std(None, ["twoway", "alex5"])[0],
    "twoway_inception": mean_std(None, ["twoway", "inception"])[0],
  },
  "ours_loso_std": {
    "clip_2way_vitl": mean_std("clip_2way_vitl")[1],
    "twoway_clip": mean_std(None, ["twoway", "clip"])[1],
    "twoway_alex2": mean_std(None, ["twoway", "alex2"])[1],
    "fid": mean_std("fid")[1],
  },
  "baseline_pooled10_mean": {
    "clip_2way_vitl": bmean("clip_2way_vitl"),
    "class_top1": bmean("class_top1"),
    "fid": bmean("fid"),
    "pixcorr": bmean("pixcorr"),
    "ssim_skimage": bmean("ssim_skimage"),
    "twoway_clip": bmean("clip", nested=True),
    "twoway_alex2": bmean("alex2", nested=True),
    "twoway_alex5": bmean("alex5", nested=True),
    "twoway_inception": bmean("inception", nested=True),
  },
  "per_fold": rows,
}
(root / "summary_loso.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

lines = [
  "# HCMA LOSO (Leave-One-Subject-Out)",
  "",
  "Protocol: for held-out subject **k**, MG-Flow pretrained on the **other 9**, then FT on **k** (heads frozen), evaluate on **k** test.",
  "This is the cross-subject protocol answering generalization concerns (unlike pooled-10 pretrain that has seen k).",
  "",
  f"Folds completed: **{len(rows)}**/10",
  "",
  "## Mean over completed folds",
  "",
  "| Setting | CLIP2 (ViT-L) | erdc CLIP | Alex2 | Alex5 | Inc | Class | FID | PixCorr | SSIM |",
  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
ol, bl = agg["ours_loso_mean"], agg["baseline_pooled10_mean"]
def fmt(x):
    return f"{x:.3f}" if x is not None else "—"
lines.append(
  f"| **LOSO (9→1)** | {fmt(ol['clip_2way_vitl'])} | {fmt(ol['twoway_clip'])} | {fmt(ol['twoway_alex2'])} | "
  f"{fmt(ol['twoway_alex5'])} | {fmt(ol['twoway_inception'])} | {fmt(ol['class_top1'])} | {fmt(ol['fid'])} | "
  f"{fmt(ol['pixcorr'])} | {fmt(ol['ssim_skimage'])} |"
)
lines.append(
  f"| Pooled-10 pretrain + FT (baseline) | {fmt(bl['clip_2way_vitl'])} | {fmt(bl['twoway_clip'])} | {fmt(bl['twoway_alex2'])} | "
  f"{fmt(bl['twoway_alex5'])} | {fmt(bl['twoway_inception'])} | {fmt(bl['class_top1'])} | {fmt(bl['fid'])} | "
  f"{fmt(bl['pixcorr'])} | {fmt(bl['ssim_skimage'])} |"
)
lines += ["", "## Per-fold (held-out subject)", "",
  "| Holdout | CLIP2 | erdc CLIP | Alex2 | Alex5 | Inc | Class | FID |",
  "|---|---:|---:|---:|---:|---:|---:|---:|"]
for r in sorted(rows, key=lambda x: x["holdout"]):
    tw = r.get("twoway") or {}
    lines.append(
      f"| sub-{r['holdout']:02d} | {fmt(r.get('clip_2way_vitl'))} | {fmt(tw.get('clip'))} | {fmt(tw.get('alex2'))} | "
      f"{fmt(tw.get('alex5'))} | {fmt(tw.get('inception'))} | {fmt(r.get('class_top1'))} | {fmt(r.get('fid'))} |"
    )
(root / "LOSO_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"n_folds": len(rows), "loso_mean": ol, "baseline_mean": bl}, indent=2))
PY

echo "{\"pipeline\":\"hcma_loso\",\"finished\":\"$(date -Iseconds)\",\"holdouts\":\"${HOLDOUTS}\"}" > "${OUT_ROOT}/job_done.json"
echo "[DONE] ${OUT_ROOT}/LOSO_TABLE.md"
