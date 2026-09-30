#!/usr/bin/env bash
# HCMA 10-subject: train cross-subject MG-Flow from scratch (no sub-08 warm-start),
# then per-subject FT that only updates/saves CFM+gate+to_gen (heads frozen).
# Recipe: HCMA full prompts + a40 blend; full MindEye/ATM metric suite; semantic auto-select grids.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
OUT_ROOT="${OUT_ROOT:-${NB_ROOT}/outputs/hcma_10subj}"
MG08="${MG08:-${NB_ROOT}/outputs/mg_flow/sub-08}"
NDA_SS="${NDA_SS:-${NB_ROOT}/outputs/nda_ss/sub-08}"
RGT_BANK="${RGT_BANK:-${NB_ROOT}/outputs/rgt_cfm/sub-08/bank}"
SS_CKPT="${SS_CKPT:-${NDA_SS}/ss/checkpoint_ss_calib_best.pth}"
# Prefer per-test-sample EEG/GT depth (HCMA-lite path). Neighbor depth_cache is incomplete.
PRED_DEPTH="${PRED_DEPTH:-${NB_ROOT}/outputs/top1_structure/sub-08/track_s/depth_head/pred_depth_rgb_512}"
if [[ ! -d "${PRED_DEPTH}" ]]; then
  PRED_DEPTH="${NB_ROOT}/outputs/overnight_ablation/sub-08/depth_vith/pred_depth_rgb_512"
fi
CN_COCA="${CN_COCA:-${NB_ROOT}/outputs/top1_structure/sub-08/track_s/depth_head/cn_scale_coca.npy}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
SUBJECTS="${SUBJECTS:-1,2,3,4,5,6,7,8,9,10}"

CLIP_TRAIN="${CLIP_TRAIN:-${BRAINIT}/outputs/atm_bridge/clip_img_train_1024.npy}"
CLIP_TEST="${CLIP_TEST:-${BRAINIT}/outputs/atm_bridge/clip_img_test_1024.npy}"
if [[ ! -f "${CLIP_TRAIN}" ]]; then
  CLIP_TRAIN="${NDA_SS}/train/decode_vith1024_train_clip_1024.npy"
  CLIP_TEST="${NDA_SS}/train/decode_vith1024_test_clip_1024.npy"
fi

mkdir -p "${OUT_ROOT}/shared" "${OUT_ROOT}/prompts" "${OUT_ROOT}/logs" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"

test -f "${SS_CKPT}"
test -d "${PRED_DEPTH}"
test -f "${PRED_DEPTH}/000.png" && test -f "${PRED_DEPTH}/199.png"
echo "[INFO] PRED_DEPTH=${PRED_DEPTH}"
echo "[INFO] protocol=cross_subj_from_scratch + per_subj_FT(delta-only)"

echo "===== [0] HCMA prompts @ $(date -Iseconds) ====="
CONCEPTS="${MG08}/targets/concepts_test.json"
if [[ ! -f "${OUT_ROOT}/prompts/prompts_full_hcma_test.json" ]]; then
  "${PYTHON}" scripts/nda/build_hcma_prompts.py \
    --concepts-json "${CONCEPTS}" \
    --output-dir "${OUT_ROOT}/prompts"
fi
PROMPT_FULL="${OUT_ROOT}/prompts/prompts_full_hcma_test.json"

# shared targets
TARGETS="${OUT_ROOT}/shared/targets"
if [[ ! -f "${TARGETS}/t_fine_train.npy" ]]; then
  mkdir -p "${TARGETS}"
  cp -a "${MG08}/targets/." "${TARGETS}/"
fi

echo "===== [1] Ensure z_ret for all subjects @ $(date -Iseconds) ====="
IFS=',' read -ra SUBJ_ARR <<< "${SUBJECTS}"
ZTR_LIST=()
ZTE_LIST=()
NDA_TR_LIST=()
NDA_TE_LIST=()

for SID in "${SUBJ_ARR[@]}"; do
  SID=$(echo "${SID}" | tr -d ' ')
  STAG=$(printf "sub-%02d" "${SID}")
  SOUT="${OUT_ROOT}/${STAG}"
  mkdir -p "${SOUT}/zret" "${SOUT}/memory" "${SOUT}/ft" "${SOUT}/generation" "${SOUT}/metrics" "${SOUT}/compare"

  ZTR="${RGT_BANK}/z_ret_sub$(printf '%02d' "${SID}")_train.npy"
  ZTE="${RGT_BANK}/z_ret_sub$(printf '%02d' "${SID}")_test.npy"
  # Prefer already-calibrated local zret (resume-safe for subjects missing from bank, e.g. sub-03)
  if [[ -f "${SOUT}/zret/z_ret_train.npy" && -f "${SOUT}/zret/z_ret_test.npy" ]]; then
    ZTR="${SOUT}/zret/z_ret_train.npy"
    ZTE="${SOUT}/zret/z_ret_test.npy"
  elif [[ ! -f "${ZTR}" || ! -f "${ZTE}" ]]; then
    echo "[INFO] calibrate/encode ${STAG}"
    "${PYTHON}" scripts/nda/calibrate_ss_new_subject.py \
      --checkpoint "${SS_CKPT}" \
      --subject "${SID}" \
      --output-dir "${SOUT}/zret" \
      --epochs 12 \
      --device "${DEVICE}"
    ZTR="${SOUT}/zret/z_ret_sub$(printf '%02d' "${SID}")_train.npy"
    ZTE="${SOUT}/zret/z_ret_sub$(printf '%02d' "${SID}")_test.npy"
  else
    cp -f "${ZTR}" "${SOUT}/zret/z_ret_train.npy"
    cp -f "${ZTE}" "${SOUT}/zret/z_ret_test.npy"
    cp -f "${ZTR}" "${SOUT}/zret/z_eeg_proj_train.npy"
    cp -f "${ZTE}" "${SOUT}/zret/z_eeg_proj_test.npy"
    ZTR="${SOUT}/zret/z_ret_train.npy"
    ZTE="${SOUT}/zret/z_ret_test.npy"
  fi
  # unify names
  [[ -f "${SOUT}/zret/z_ret_train.npy" ]] || cp -f "${ZTR}" "${SOUT}/zret/z_ret_train.npy"
  [[ -f "${SOUT}/zret/z_ret_test.npy" ]] || cp -f "${ZTE}" "${SOUT}/zret/z_ret_test.npy"
  [[ -f "${SOUT}/zret/z_eeg_proj_train.npy" ]] || cp -f "${SOUT}/zret/z_ret_train.npy" "${SOUT}/zret/z_eeg_proj_train.npy"
  [[ -f "${SOUT}/zret/z_eeg_proj_test.npy" ]] || cp -f "${SOUT}/zret/z_ret_test.npy" "${SOUT}/zret/z_eeg_proj_test.npy"
  ZTR="${SOUT}/zret/z_ret_train.npy"
  ZTE="${SOUT}/zret/z_ret_test.npy"

  if [[ ! -f "${SOUT}/memory/rag_soft5_test_clip_1024.npy" ]]; then
    "${PYTHON}" scripts/nmb/nmb_memory_router.py \
      --embed-dir "${SOUT}/zret" \
      --clip-train "${CLIP_TRAIN}" \
      --clip-test "${CLIP_TEST}" \
      --output-dir "${SOUT}/memory" \
      --input-key proj --soft-k 5 --soft-tau 0.07
  fi
  ZTR_LIST+=("${ZTR}")
  ZTE_LIST+=("${ZTE}")
  NDA_TR_LIST+=("${SOUT}/memory/rag_soft5_train_clip_1024.npy")
  NDA_TE_LIST+=("${SOUT}/memory/rag_soft5_test_clip_1024.npy")
done

echo "===== [2] Build pooled arrays @ $(date -Iseconds) ====="
POOL="${OUT_ROOT}/shared/pool"
mkdir -p "${POOL}"
if [[ ! -f "${POOL}/z_ret_train.npy" ]]; then
  "${PYTHON}" - <<PY
import numpy as np
from pathlib import Path
pool=Path("${POOL}")
ztrs=${ZTR_LIST@Q}
# bash array passed wrong — rebuild from env files
import os,glob
outs=sorted(Path("${OUT_ROOT}").glob("sub-*/zret/z_ret_train.npy"))
ztr=np.concatenate([np.load(p) for p in outs],0)
zte=np.concatenate([np.load(p.parent/"z_ret_test.npy") for p in outs],0)
nda_tr=np.concatenate([np.load(p.parent.parent/"memory"/"rag_soft5_train_clip_1024.npy") for p in outs],0)
nda_te=np.concatenate([np.load(p.parent.parent/"memory"/"rag_soft5_test_clip_1024.npy") for p in outs],0)
np.save(pool/"z_ret_train.npy", ztr.astype(np.float32))
np.save(pool/"z_ret_test.npy", zte.astype(np.float32))
np.save(pool/"nda_train.npy", nda_tr.astype(np.float32))
np.save(pool/"nda_test.npy", nda_te.astype(np.float32))
print("pool", ztr.shape, zte.shape, nda_tr.shape, nda_te.shape)
PY
fi

echo "===== [3] Cross-subject MG-Flow FROM SCRATCH (no sub-08 init) @ $(date -Iseconds) ====="
SHARED_TRAIN="${OUT_ROOT}/shared/mg_flow"
if [[ ! -f "${SHARED_TRAIN}/checkpoints/best.pt" ]]; then
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
else
  echo "[SKIP] shared train"
fi

# Drop bulky shared train embeds (keep test + ckpt) to free disk for 10x generation.
if [[ -d "${SHARED_TRAIN}/embeds" ]]; then
  find "${SHARED_TRAIN}/embeds" -name '*_train.npy' -delete || true
  echo "[INFO] pruned shared train embeds"
fi

echo "===== [4] Per-subject FT (freeze heads; save delta only) + HCMA gen + metrics @ $(date -Iseconds) ====="
TAG="hcma_full_a40"
SHARED_CKPT="${SHARED_TRAIN}/checkpoints/best.pt"
test -f "${SHARED_CKPT}"
for SID in "${SUBJ_ARR[@]}"; do
  SID=$(echo "${SID}" | tr -d ' ')
  STAG=$(printf "sub-%02d" "${SID}")
  SOUT="${OUT_ROOT}/${STAG}"
  echo "########## ${STAG} @ $(date -Iseconds) ##########"

  EMB="${SOUT}/ft/embeds/blend_nda_cfm_f_a40_test.npy"
  DELTA_CKPT="${SOUT}/ft/checkpoints/best_delta.pt"
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
      --epochs 10 \
      --batch-size 512 \
      --lr 3e-5 \
      --early-stop 5 \
      --ode-steps 16 \
      --device "${DEVICE}"
    # keep only claim embed + delta ckpt
    if [[ -d "${SOUT}/ft/embeds" ]]; then
      find "${SOUT}/ft/embeds" -type f ! -name 'blend_nda_cfm_f_a40_test.npy' -delete || true
    fi
    # drop full best.pt symlink duplicate if delta exists; keep delta + updated_keys
    if [[ -f "${DELTA_CKPT}" && -f "${SOUT}/ft/checkpoints/best.pt" ]]; then
      # best.pt is also delta payload (same content) — keep both for resume compatibility
      :
    fi
  else
    echo "[SKIP] ft ${STAG}"
  fi

  GDIR="${SOUT}/generation/${TAG}"
  if [[ ! -f "${GDIR}/generated/199.png" ]]; then
    # clear incomplete gen from prior failed runs
    if [[ -d "${GDIR}/generated" ]] && [[ ! -f "${GDIR}/generated/199.png" ]]; then
      rm -rf "${GDIR}"
    fi
    GEN_EXTRA=()
    if [[ -f "${CN_COCA}" ]]; then
      GEN_EXTRA+=(--cn-scale-npy "${CN_COCA}")
    fi
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

  # paper metrics
  if [[ ! -f "${SOUT}/metrics/paper_metrics.json" ]]; then
    "${PYTHON}" scripts/nda/eval_paper_metrics.py \
      --gen-root "${SOUT}/generation" --tags "${TAG}" \
      --output-json "${SOUT}/metrics/paper_metrics.json"
  fi
  # MindEye CLIP 2-way (ViT-L)
  if [[ ! -f "${SOUT}/metrics/clip_2way_report.json" ]]; then
    "${PYTHON}" scripts/nda/eval_clip_2way.py \
      --gen-dirs "${TAG}=${GDIR}/generated" \
      --output-json "${SOUT}/metrics/clip_2way_report.json" \
      --device "${DEVICE}" --batch-size 16
  fi
  # class consistency
  if [[ ! -f "${SOUT}/metrics/class_consistency.json" ]]; then
    "${PYTHON}" scripts/nda/eval_class_consistency.py \
      --gen-dirs "${TAG}=${GDIR}/generated" \
      --text-concept-npy "${NDA_SS}/clip_text/test/text_concept_clip.npy" \
      --concepts-json "${CONCEPTS}" \
      --output-json "${SOUT}/metrics/class_consistency.json" \
      --device "${DEVICE}"
  fi
  # ATM/MindEye 2WC suite
  if [[ ! -f "${SOUT}/metrics/erdc_2wc.json" ]]; then
    "${PYTHON}" "${BRAINIT}/scripts/erdc_twoway_metrics.py" \
      --gen-dir "${GDIR}/generated" \
      --images-root /project/peilab/why/data/images_set \
      --output-json "${SOUT}/metrics/erdc_2wc.json" \
      --tag "${STAG}_${TAG}"
  fi
  # full low/high-level correlations
  if [[ ! -f "${SOUT}/metrics/erdc_full.json" ]]; then
    "${PYTHON}" "${BRAINIT}/scripts/erdc_full_metrics.py" \
      --gen-dir "${GDIR}/generated" \
      --images-root /project/peilab/why/data/images_set \
      --output-json "${SOUT}/metrics/erdc_full.json" \
      --tag "${STAG}_${TAG}"
  fi

  # compare grid: semantically best rows
  if [[ ! -f "${SOUT}/compare/compare_grid.png" ]]; then
    "${PYTHON}" scripts/nda/make_compare_grid.py \
      --output-dir "${SOUT}/compare" \
      --cell 160 \
      --auto-select-gen-dir "${GDIR}/generated" \
      --auto-select-k 12 \
      --device "${DEVICE}" \
      --metrics-json "${SOUT}/metrics/clip_2way_report.json" \
      --cols "hcma=${GDIR}/generated"
  fi

  # per-subject summary
  "${PYTHON}" - <<PY
import json
from pathlib import Path
out=Path("${SOUT}")
paper=json.loads((out/"metrics/paper_metrics.json").read_text())["results"][0]
tw=json.loads((out/"metrics/clip_2way_report.json").read_text())["generation_2way"][0]
cls=json.loads((out/"metrics/class_consistency.json").read_text())["results"][0]
twc=json.loads((out/"metrics/erdc_2wc.json").read_text()) if (out/"metrics/erdc_2wc.json").is_file() else {}
full=json.loads((out/"metrics/erdc_full.json").read_text()) if (out/"metrics/erdc_full.json").is_file() else {}
cmp=json.loads((out/"compare/compare_report.json").read_text()) if (out/"compare/compare_report.json").is_file() else {}
summary={
  "subject": ${SID},
  "tag": "${TAG}",
  "clip_2way_vitl": tw.get("clip_2way"),
  "class_top1": cls.get("class_top1"),
  "fid": paper.get("fid"),
  "ssim_skimage": paper.get("ssim"),
  "pixcorr": paper.get("pixcorr"),
  "clip_cosine": paper.get("clip_cosine"),
  "twoway": twc.get("twoway", twc),
  "erdc_full": full,
  "compare_indices": cmp.get("indices"),
  "compare_scores": cmp.get("scores"),
  "compare_grid": cmp.get("grid"),
}
(out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps({k:summary[k] for k in summary if k not in ("erdc_full",)},indent=2))
PY
done

echo "===== [5] Aggregate vs SOTA @ $(date -Iseconds) ====="
"${PYTHON}" - <<'PY'
import json
from pathlib import Path
import numpy as np
root=Path("/project/peilab/why/NeuroBridge/outputs/hcma_10subj")
rows=[]
for p in sorted(root.glob("sub-*/summary.json")):
    rows.append(json.loads(p.read_text()))

def mean_key(key, default=None):
    vals=[]
    for r in rows:
        v=r.get(key)
        if v is None: continue
        vals.append(float(v))
    return float(np.mean(vals)) if vals else default

def mean_nested(path):
    vals=[]
    for r in rows:
        cur=r
        ok=True
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                ok=False; break
            cur=cur[k]
        if ok and cur is not None:
            try: vals.append(float(cur))
            except Exception: pass
    return float(np.mean(vals)) if vals else None

agg={
  "n_subjects": len(rows),
  "subjects": [r["subject"] for r in rows],
  "ours_mean": {
    "clip_2way_vitl": mean_key("clip_2way_vitl"),
    "class_top1": mean_key("class_top1"),
    "fid": mean_key("fid"),
    "ssim_skimage": mean_key("ssim_skimage"),
    "pixcorr": mean_key("pixcorr"),
    "clip_cosine": mean_key("clip_cosine"),
    "twoway_clip": mean_nested(["twoway","clip"]) or mean_nested(["twoway","twoway","clip"]),
    "twoway_alex2": mean_nested(["twoway","alex2"]) or mean_nested(["twoway","twoway","alex2"]),
    "twoway_alex5": mean_nested(["twoway","alex5"]) or mean_nested(["twoway","twoway","alex5"]),
    "twoway_inception": mean_nested(["twoway","inception"]) or mean_nested(["twoway","twoway","inception"]),
  },
  "per_subject": rows,
  "sota_reference": {
    "note": "Protocols differ: CogCap/ATM CLIP column ≠ our OpenCLIP paired cos; 2-way is MindEye/Ozcelik-style.",
    "CogCap_all_10subj_mean_Table3": {"pixcorr": 0.150, "ssim": 0.347, "clip_paper": 0.715},
    "CogCap_sub08_supp": {"pixcorr": 0.175, "ssim": 0.366, "clip_paper": 0.744},
    "ATM_eeg_recon_reported": {"pixcorr": 0.160, "ssim": 0.345, "clip_paper": 0.786},
    "MindEye_fmri_ref": {"pixcorr": 0.309, "ssim": 0.323, "clip_2way_approx": "see paper Alex/Incep/CLIP 2-way ~94%+"},
    "ours_sub08_hcma_lite_sem_full": {"clip_2way": 0.960, "class_top1": 0.745, "fid": 138.3, "ssim": 0.225},
  },
  "claim": "Cross-subject MG-Flow trained from scratch on all subjects (no sub-08 warm-start) + per-subject FT delta (heads frozen; only CFM/gate/to_gen saved) + HCMA full prompts; semantic-first metrics",
}
# also gather twoway from raw files more reliably
tw_clip, tw_a2, tw_a5, tw_inc = [], [], [], []
for r in rows:
    sid=r["subject"]
    p=root/f"sub-{sid:02d}/metrics/erdc_2wc.json"
    if not p.is_file():
        continue
    d=json.loads(p.read_text())
    tw=d.get("twoway", d)
    if "clip" in tw: tw_clip.append(float(tw["clip"]))
    if "alex2" in tw: tw_a2.append(float(tw["alex2"]))
    if "alex5" in tw: tw_a5.append(float(tw["alex5"]))
    if "inception" in tw: tw_inc.append(float(tw["inception"]))
if tw_clip:
    agg["ours_mean"]["twoway_clip"]=float(np.mean(tw_clip))
    agg["ours_mean"]["twoway_alex2"]=float(np.mean(tw_a2)) if tw_a2 else None
    agg["ours_mean"]["twoway_alex5"]=float(np.mean(tw_a5)) if tw_a5 else None
    agg["ours_mean"]["twoway_inception"]=float(np.mean(tw_inc)) if tw_inc else None

(root/"summary_all.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
# markdown table
md=["# HCMA 10-subject vs SOTA\n",
    f"Subjects: {agg['subjects']} (n={agg['n_subjects']})\n",
    "## Ours (mean)\n",
    "| Metric | Value |\n|---|---|\n"]
for k,v in agg["ours_mean"].items():
    if v is None: continue
    if "way" in k or "top1" in k or "cosine" in k or "ssim" in k or "pix" in k:
        md.append(f"| {k} | {v:.4f} |\n")
    else:
        md.append(f"| {k} | {v:.2f} |\n")
md += ["\n## SOTA reference (protocol caveats apply)\n",
       "| Method | PixCorr | SSIM | CLIP(paper) |\n|---|---|---|---|\n",
       "| CogCap 10subj mean | 0.150 | 0.347 | 0.715 |\n",
       "| CogCap sub-08 | 0.175 | 0.366 | 0.744 |\n",
       "| ATM | ~0.160 | 0.345 | 0.786 |\n",
       "\nPrimary semantic claim: **CLIP/Inception/Alex 2-way** (MindEye suite) + FID.\n"]
(root/"SOTA_COMPARE.md").write_text("".join(md), encoding="utf-8")
print(json.dumps(agg["ours_mean"], indent=2))
print("wrote", root/"summary_all.json")
PY

du -sh "${OUT_ROOT}" || true
echo "===== DONE HCMA 10-subj @ $(date -Iseconds) ====="
