#!/usr/bin/env bash
# LOSO for the FID≈129 HCMA model (hcma_full_a40).
#
# Baseline to defend: outputs/hcma_10subj — pooled FID 129.47
#   protocol: MG-Flow from-scratch (all 10) + per-subj FT(delta) + HCMA full prompts
#             + Depth-CN/IP gen tag=hcma_full_a40
#
# This script = TRUE leave-one-subject-out of that SAME recipe:
#   For held-out k: pretrain on other 9 (never sees k) → FT on k → gen/eval as hcma_full_a40
#   After all folds: pooled FID (2000 fake vs 200 GT) for direct comparison to 129.47
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
SRC="${SRC:-${NB_ROOT}/outputs/hcma_10subj}"
OUT_ROOT="${OUT_ROOT:-${NB_ROOT}/outputs/hcma_loso_fid129}"
MG08="${MG08:-${NB_ROOT}/outputs/mg_flow/sub-08}"
NDA_SS="${NDA_SS:-${NB_ROOT}/outputs/nda_ss/sub-08}"
PRED_DEPTH="${PRED_DEPTH:-${NB_ROOT}/outputs/top1_structure/sub-08/track_s/depth_head/pred_depth_rgb_512}"
CN_COCA="${CN_COCA:-${NB_ROOT}/outputs/top1_structure/sub-08/track_s/depth_head/cn_scale_coca.npy}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
HOLDOUTS="${HOLDOUTS:-1,2,3,4,5,6,7,8,9,10}"

# Exact claim tag from FID-129 HCMA
TAG="hcma_full_a40"

CLIP_TRAIN="${CLIP_TRAIN:-${BRAINIT}/outputs/atm_bridge/clip_img_train_1024.npy}"
CLIP_TEST="${CLIP_TEST:-${BRAINIT}/outputs/atm_bridge/clip_img_test_1024.npy}"
if [[ ! -f "${CLIP_TRAIN}" ]]; then
  CLIP_TRAIN="${NDA_SS}/train/decode_vith1024_train_clip_1024.npy"
  CLIP_TEST="${NDA_SS}/train/decode_vith1024_test_clip_1024.npy"
fi

mkdir -p "${OUT_ROOT}/folds" "${OUT_ROOT}/logs" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" || true
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
require "${PRED_DEPTH}/199.png"
require "${CLIP_TRAIN}"; require "${CLIP_TEST}"
require "${SRC}/prompts/prompts_full_hcma_test.json"
require "${SRC}/metrics_pooled_fid.json"

PROMPT_FULL="${SRC}/prompts/prompts_full_hcma_test.json"
CONCEPTS="${MG08}/targets/concepts_test.json"
BASE_POOLED_FID=$(python -c "import json;print(json.load(open('${SRC}/metrics_pooled_fid.json'))['pooled_fid_unique_gt'])")

echo "{\"pipeline\":\"hcma_loso_fid129\",\"baseline_pooled_fid\":${BASE_POOLED_FID},\"tag\":\"${TAG}\",\"holdouts\":\"${HOLDOUTS}\",\"started\":\"$(date -Iseconds)\"}" \
  > "${OUT_ROOT}/job_running.json"
echo "[INFO] LOSO of HCMA hcma_full_a40 (baseline pooled FID=${BASE_POOLED_FID})"

# Rebuild train text targets if pruned from disk
TARGETS="${OUT_ROOT}/shared_targets"
mkdir -p "${TARGETS}"
if [[ ! -f "${TARGETS}/t_coarse_train.npy" ]]; then
  "${PYTHON}" scripts/nda/build_mg_flow_targets.py \
    --clip-text-root "${NDA_SS}/clip_text" \
    --output-dir "${TARGETS}"
fi
require "${TARGETS}/t_coarse_train.npy"

IFS=',' read -r -a HOLD_ARR <<< "${HOLDOUTS}"

run_fold() {
  local HOLD="$1"
  local STAG HTAG FOLD
  STAG=$(printf "sub-%02d" "${HOLD}")
  HTAG=$(printf "holdout_%02d" "${HOLD}")
  FOLD="${OUT_ROOT}/folds/${HTAG}"
  local SOUT="${OUT_ROOT}/${STAG}"

  echo "########## LOSO ${HTAG}: pretrain≠${STAG}, FT+eval=${STAG} @ $(date -Iseconds) ##########"
  mkdir -p "${FOLD}/pool" "${FOLD}/mg_flow" "${SOUT}/"{ft,generation,metrics,compare}

  [[ -e "${SOUT}/zret" ]] || ln -sfn "${SRC}/${STAG}/zret" "${SOUT}/zret"
  [[ -e "${SOUT}/memory" ]] || ln -sfn "${SRC}/${STAG}/memory" "${SOUT}/memory"
  require "${SOUT}/zret/z_ret_train.npy"
  require "${SOUT}/memory/rag_soft5_neighbor_idx_test.npy"

  local POOL="${FOLD}/pool"
  local SHARED_CKPT="${FOLD}/mg_flow/checkpoints/best.pt"

  # 9-subject pool excluding HOLD
  if [[ ! -f "${POOL}/z_ret_train.npy" ]]; then
    HOLD="${HOLD}" SRC="${SRC}" POOL="${POOL}" "${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
import numpy as np
src, pool = Path(os.environ["SRC"]), Path(os.environ["POOL"])
pool.mkdir(parents=True, exist_ok=True)
hold = int(os.environ["HOLD"])
z_tr, z_te, n_tr, n_te, kept = [], [], [], [], []
for sid in range(1, 11):
    if sid == hold:
        continue
    stag = f"sub-{sid:02d}"
    z_tr.append(np.load(src / stag / "zret" / "z_ret_train.npy"))
    z_te.append(np.load(src / stag / "zret" / "z_ret_test.npy"))
    n_tr.append(np.load(src / stag / "memory" / "rag_soft5_train_clip_1024.npy"))
    n_te.append(np.load(src / stag / "memory" / "rag_soft5_test_clip_1024.npy"))
    kept.append(sid)
ztr = np.concatenate(z_tr, 0).astype(np.float32)
zte = np.concatenate(z_te, 0).astype(np.float32)
ntr = np.concatenate(n_tr, 0).astype(np.float32)
nte = np.concatenate(n_te, 0).astype(np.float32)
np.save(pool / "z_ret_train.npy", ztr)
np.save(pool / "z_ret_test.npy", zte)
np.save(pool / "nda_train.npy", ntr)
np.save(pool / "nda_test.npy", nte)
meta = {"holdout": hold, "train_subjects": kept, "z_train": list(ztr.shape), "z_test": list(zte.shape),
        "recipe": "HCMA hcma_full_a40 LOSO (FID-129 model)"}
(pool / "pool_meta.json").write_text(json.dumps(meta, indent=2))
print(meta)
PY
  else
    echo "[SKIP] pool ${HTAG}"
  fi

  # Pretrain on 9 (never sees holdout) — same hparams as FID-129 shared train
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
      --output-dir "${FOLD}/mg_flow" \
      --tile-targets \
      --epochs 40 --batch-size 512 --lr 1e-4 \
      --early-stop 10 --ode-steps 16 --device "${DEVICE}"
    find "${FOLD}/mg_flow/embeds" -name '*_train.npy' -delete 2>/dev/null || true
  else
    echo "[SKIP] pretrain ${HTAG}"
  fi
  require "${SHARED_CKPT}"
  rm -f "${POOL}/z_ret_train.npy" "${POOL}/nda_train.npy" || true

  # FT holdout only — same as FID-129 per-subj FT
  local EMB="${SOUT}/ft/embeds/blend_nda_cfm_f_a40_test.npy"
  if [[ ! -f "${EMB}" ]]; then
    "${PYTHON}" scripts/nda/mg_flow_train.py \
      --z-ret-train "${SOUT}/zret/z_ret_train.npy" \
      --z-ret-test "${SOUT}/zret/z_ret_test.npy" \
      --clip-img-train "${CLIP_TRAIN}" \
      --clip-img-test "${CLIP_TEST}" \
      --t-coarse-train "${TARGETS}/t_coarse_train.npy" \
      --t-fine-train "${TARGETS}/t_fine_train.npy" \
      --t-coarse-test "${TARGETS}/t_coarse_test.npy" \
      --t-fine-test "${TARGETS}/t_fine_test.npy" \
      --nda-decode-train "${SOUT}/memory/rag_soft5_train_clip_1024.npy" \
      --nda-decode-test "${SOUT}/memory/rag_soft5_test_clip_1024.npy" \
      --text-concept-test "${NDA_SS}/clip_text/test/text_concept_clip.npy" \
      --output-dir "${SOUT}/ft" \
      --init-ckpt "${SHARED_CKPT}" \
      --freeze-backbone \
      --epochs 10 --batch-size 512 --lr 3e-5 \
      --early-stop 5 --ode-steps 16 --device "${DEVICE}"
    find "${SOUT}/ft/embeds" -type f ! -name 'blend_nda_cfm_f_a40_test.npy' -delete 2>/dev/null || true
  else
    echo "[SKIP] FT ${STAG}"
  fi
  require "${EMB}"

  # Generate — identical to FID-129 hcma_full_a40
  local GDIR="${SOUT}/generation/${TAG}"
  if [[ ! -f "${GDIR}/generated/199.png" ]]; then
    [[ -d "${GDIR}/generated" && ! -f "${GDIR}/generated/199.png" ]] && rm -rf "${GDIR}"
    local GEN_EXTRA=()
    [[ -f "${CN_COCA}" ]] && GEN_EXTRA+=(--cn-scale-npy "${CN_COCA}")
    "${PYTHON}" scripts/nda/generate_cn_ip_decode.py \
      --embed-npy "${EMB}" \
      --neighbor-idx-npy "${SOUT}/memory/rag_soft5_neighbor_idx_test.npy" \
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
    echo "[SKIP] gen ${STAG}"
  fi

  # Metrics (same suite as hcma_10subj)
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
  find "${GDIR}" -type d -name '_twoway_cache' -exec rm -rf {} + 2>/dev/null || true

  HOLD="${HOLD}" TAG="${TAG}" SOUT="${SOUT}" "${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["SOUT"])
paper = json.loads((out / "metrics/paper_metrics.json").read_text())["results"][0]
tw = json.loads((out / "metrics/clip_2way_report.json").read_text())["generation_2way"][0]
cls = json.loads((out / "metrics/class_consistency.json").read_text())["results"][0]
twc = json.loads((out / "metrics/erdc_2wc.json").read_text())
summary = {
  "protocol": "LOSO_of_HCMA_fid129",
  "model": "hcma_full_a40",
  "holdout": int(os.environ["HOLD"]),
  "pretrain": "9 subjects excluding holdout (from scratch)",
  "finetune": "holdout only; freeze DualSemanticHeads; delta CFM/gate/to_gen",
  "tag": os.environ["TAG"],
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

# ---------- Pooled FID (same metric that gave 129.47) ----------
echo "===== Pooled FID (SOTA-style) @ $(date -Iseconds) ====="
"${PYTHON}" scripts/nda/eval_pooled_fid.py \
  --root "${OUT_ROOT}" \
  --tag "${TAG}" \
  --output-json "${OUT_ROOT}/metrics_pooled_fid.json" \
  --device "${DEVICE}" \
  --batch-size 32

# ---------- Aggregate vs FID-129 baseline ----------
echo "===== Aggregate @ $(date -Iseconds) ====="
OUT_ROOT="${OUT_ROOT}" SRC="${SRC}" BASE_POOLED_FID="${BASE_POOLED_FID}" TAG="${TAG}" "${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
import numpy as np

root = Path(os.environ["OUT_ROOT"])
src = Path(os.environ["SRC"])
base_pfid = float(os.environ["BASE_POOLED_FID"])
tag = os.environ["TAG"]

rows = []
for p in sorted(root.glob("sub-*/summary.json")):
    rows.append(json.loads(p.read_text()))

def ms(getter):
    vals = [getter(r) for r in rows]
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))

base_rows = [json.loads(p.read_text()) for p in sorted(src.glob("sub-*/summary.json"))]

def bms(getter):
    vals = [getter(r) for r in base_rows]
    vals = [float(v) for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None

loso_pfid = None
pf = root / "metrics_pooled_fid.json"
if pf.is_file():
    loso_pfid = float(json.loads(pf.read_text())["pooled_fid_unique_gt"])

agg = {
  "pipeline": "hcma_loso_fid129",
  "model_tag": tag,
  "n_folds": len(rows),
  "protocol": "LOSO of FID-129 HCMA: for each k, pretrain on 9 others → FT k → hcma_full_a40 gen/eval",
  "baseline_pooled_fid_129": base_pfid,
  "loso_pooled_fid": loso_pfid,
  "delta_pooled_fid": (loso_pfid - base_pfid) if loso_pfid is not None else None,
  "ours_loso_mean": {
    "clip_2way_vitl": ms(lambda r: r.get("clip_2way_vitl"))[0],
    "class_top1": ms(lambda r: r.get("class_top1"))[0],
    "fid_per_subj": ms(lambda r: r.get("fid"))[0],
    "pixcorr": ms(lambda r: r.get("pixcorr"))[0],
    "ssim": ms(lambda r: r.get("ssim_skimage"))[0],
    "twoway_clip": ms(lambda r: (r.get("twoway") or {}).get("clip"))[0],
    "twoway_alex2": ms(lambda r: (r.get("twoway") or {}).get("alex2"))[0],
    "twoway_alex5": ms(lambda r: (r.get("twoway") or {}).get("alex5"))[0],
    "twoway_inception": ms(lambda r: (r.get("twoway") or {}).get("inception"))[0],
  },
  "baseline_pooled10_mean": {
    "clip_2way_vitl": bms(lambda r: r.get("clip_2way_vitl")),
    "class_top1": bms(lambda r: r.get("class_top1")),
    "fid_per_subj": bms(lambda r: r.get("fid")),
    "twoway_clip": bms(lambda r: (r.get("twoway") or {}).get("clip")),
    "twoway_alex2": bms(lambda r: (r.get("twoway") or {}).get("alex2")),
    "twoway_alex5": bms(lambda r: (r.get("twoway") or {}).get("alex5")),
    "twoway_inception": bms(lambda r: (r.get("twoway") or {}).get("inception")),
  },
  "per_fold": rows,
}
(root / "summary_loso.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

def fmt(x):
    return f"{x:.3f}" if isinstance(x, float) else ("—" if x is None else str(x))

ol, bl = agg["ours_loso_mean"], agg["baseline_pooled10_mean"]
lines = [
  "# LOSO of HCMA (`hcma_full_a40`, FID≈129 model)",
  "",
  "Baseline: `outputs/hcma_10subj` — **pooled FID = {:.2f}** (10×200 gens vs 200 GT).".format(base_pfid),
  "LOSO: for held-out **k**, pretrain on other **9** (k never seen) → FT **k** → same `hcma_full_a40` decode.",
  "",
  f"## Pooled FID (primary SOTA number)",
  f"- Baseline (pooled-10 pretrain): **{base_pfid:.2f}**",
  f"- LOSO: **{fmt(loso_pfid)}**" + (f" (Δ {loso_pfid-base_pfid:+.2f})" if loso_pfid is not None else ""),
  "",
  "## Mean metrics over folds",
  "",
  "| Setting | Pooled FID↓ | CLIP2 | erdc CLIP | Alex2 | Alex5 | Inc | Class | FID/subj |",
  "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
  f"| **LOSO 9→1** | {fmt(loso_pfid)} | {fmt(ol['clip_2way_vitl'])} | {fmt(ol['twoway_clip'])} | {fmt(ol['twoway_alex2'])} | {fmt(ol['twoway_alex5'])} | {fmt(ol['twoway_inception'])} | {fmt(ol['class_top1'])} | {fmt(ol['fid_per_subj'])} |",
  f"| Pooled-10 + FT (FID-129) | {base_pfid:.2f} | {fmt(bl['clip_2way_vitl'])} | {fmt(bl['twoway_clip'])} | {fmt(bl['twoway_alex2'])} | {fmt(bl['twoway_alex5'])} | {fmt(bl['twoway_inception'])} | {fmt(bl['class_top1'])} | {fmt(bl['fid_per_subj'])} |",
  "",
  "## Per holdout",
  "",
  "| Holdout | CLIP2 | erdc CLIP | Alex2 | Alex5 | Inc | Class | FID |",
  "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for r in sorted(rows, key=lambda x: x["holdout"]):
    tw = r.get("twoway") or {}
    lines.append(
      f"| sub-{r['holdout']:02d} | {fmt(r.get('clip_2way_vitl'))} | {fmt(tw.get('clip'))} | {fmt(tw.get('alex2'))} | "
      f"{fmt(tw.get('alex5'))} | {fmt(tw.get('inception'))} | {fmt(r.get('class_top1'))} | {fmt(r.get('fid'))} |"
    )
(root / "LOSO_FID129_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"n_folds": len(rows), "loso_pooled_fid": loso_pfid, "baseline": base_pfid}, indent=2))
PY

echo "{\"pipeline\":\"hcma_loso_fid129\",\"finished\":\"$(date -Iseconds)\",\"tag\":\"${TAG}\",\"baseline_pooled_fid\":${BASE_POOLED_FID}}" \
  > "${OUT_ROOT}/job_done.json"
echo "[DONE] ${OUT_ROOT}/LOSO_FID129_TABLE.md"
