#!/usr/bin/env bash
# inter_ll FULL-10-subject: pure zero-shot inter-subject semantic decode.
#
# Question answered: "if we DROP per-subject fine-tuning, is the model still SOTA?"
#   sdedit_ll_full10 (FT)   : HCMA per-subject FT embed  + per-subj LL-RGB  -> sdedit_ll
#   inter_ll_full10 (ZERO)  : LOSO fold MG-Flow ckpt pure-forward (NO FT)    -> sdedit_ll
#
# For EVERY subject 01..10 with holdout k = s:
#   1) mg_flow_inter_encode.py --ckpt hcma_loso_fid129/folds/holdout_XX/mg_flow/checkpoints/best.pt
#        (9-subject pretrained, k NEVER seen)  + holdout z_ret / RAG memory
#        => inter embeds (blend_nda_cfm_f_a40_test.npy etc.), pure inference, no FT.
#   2) sdedit_ll decode (same recipe as sdedit_ll_full10): LL-RGB init from
#        sdedit_ll_full10/<sub>/vae_head/pred_lowlevel_rgb_512 (subject structure expert)
#        + inter embed + HCMA prompts; strength 0.82, 28 steps, CFG 5.0, IP 1.0.
#   3) standard-7 eval on manifest (hcma refs cached; inter rows new) + pooled FID + SOTA table.
#
# NOTE: LL-RGB init (structure expert) is still per-subject here on purpose — this run
# isolates the SEMANTIC question (does dropping per-subject FT of the MG-Flow semantic
# tower keep SOTA semantic metrics?). Structural zero-shot is Phase 3 (pooled VAE head).
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/inter_ll_full10}"
LOSO="${LOSO:-${NB_ROOT}/outputs/hcma_loso_fid129}"
LL10="${LL10:-${NB_ROOT}/outputs/sdedit_ll_full10}"    # per-subject VAE head LL-RGB (structure expert)
HCMA10="${HCMA10:-${NB_ROOT}/outputs/hcma_10subj}"      # z_ret / RAG memory / prompts
NDA_SS="${NDA_SS:-${NB_ROOT}/outputs/nda_ss/sub-08}"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
STD7_OUT="${NB_ROOT}/outputs/standard7_protocol"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
SUBJECTS="${SUBJECTS:-1,2,3,4,5,6,7,8,9,10}"

export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${TORCH_HOME}"

mkdir -p "${OUT}/logs" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" || true
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export DEVICE OUT NB_ROOT IMAGES_ROOT

echo "{\"pipeline\":\"inter_ll_full10\",\"started\":\"$(date -Iseconds)\",\"goal\":\"pure zero-shot inter-subject semantic decode (NO per-subject FT), standard-7 + pooled FID\"}" \
  > "${OUT}/job_running.json"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }

PROMPT="${HCMA10}/prompts/prompts_full_hcma_test.json"
require "${PROMPT}"
CLIP_GALLERY="/project/peilab/why/eeg-brainit/outputs/atm_bridge/clip_img_test_1024.npy"
require "${CLIP_GALLERY}"
TGT="${LOSO}/shared_targets"
require "${TGT}/t_coarse_test.npy"; require "${TGT}/t_fine_test.npy"
TEXT_CONCEPT="${NDA_SS}/clip_text/test/text_concept_clip.npy"
require "${TEXT_CONCEPT}"

IFS=',' read -ra SUBJ_ARR <<< "${SUBJECTS}"

# ------------------------------------------------------------------ per-subject inter encode + sdedit decode
for SID in "${SUBJ_ARR[@]}"; do
  SID="$(echo "${SID}" | tr -d ' ')"
  STAG="$(printf "sub-%02d" "${SID}")"
  HTAG="$(printf "holdout_%02d" "${SID}")"
  SOUT="${OUT}/${STAG}"
  mkdir -p "${SOUT}/inter_embeds" "${SOUT}/generation"
  echo "########## ${STAG} (inter, NO FT) @ $(date -Iseconds) ##########"

  CKPT="${LOSO}/folds/${HTAG}/mg_flow/checkpoints/best.pt"
  ZTE="${HCMA10}/${STAG}/zret/z_ret_test.npy"
  NDA_TE="${HCMA10}/${STAG}/memory/rag_soft5_test_clip_1024.npy"
  LL_RGB="${LL10}/${STAG}/vae_head/pred_lowlevel_rgb_512"
  require "${CKPT}"; require "${ZTE}"; require "${NDA_TE}"; require "${LL_RGB}/199.png"

  # ---- [1] zero-shot inter encode (pure forward with LOSO fold ckpt)
  # NOTE: mg_flow_inter_encode.py writes to <output-dir>/embeds/ (mirrors mg_flow_train.py).
  INTER_EMB="${SOUT}/inter_embeds/embeds/blend_nda_cfm_f_a40_test.npy"
  if [[ ! -f "${INTER_EMB}" ]]; then
    "${PYTHON}" scripts/nda/mg_flow_inter_encode.py \
      --ckpt "${CKPT}" \
      --z-ret-test "${ZTE}" \
      --nda-decode-test "${NDA_TE}" \
      --clip-img-test "${CLIP_GALLERY}" \
      --t-coarse-test "${TGT}/t_coarse_test.npy" \
      --t-fine-test "${TGT}/t_fine_test.npy" \
      --text-concept-test "${TEXT_CONCEPT}" \
      --output-dir "${SOUT}/inter_embeds" \
      --ode-steps 16 \
      --device "${DEVICE}"
  else
    echo "[SKIP] inter encode ${STAG}"
  fi
  require "${INTER_EMB}"

  # ---- [2] sdedit_ll decode with inter embed (identical recipe to sdedit_ll_full10)
  GDIR="${SOUT}/generation/inter_ll"
  if [[ ! -f "${GDIR}/generated/199.png" ]]; then
    rm -rf "${GDIR}"
    "${PYTHON}" scripts/nda/generate_atm_aligned_decode.py \
      --mode sdedit \
      --embed-npy "${INTER_EMB}" \
      --prompts-json "${PROMPT}" \
      --lowlevel-rgb-dir "${LL_RGB}" \
      --output-dir "${GDIR}" \
      --tag "inter_ll" \
      --strength 0.82 \
      --ip-scale 1.0 \
      --gen-steps 28 \
      --gen-guidance 5.0 \
      --seed 42
  else
    echo "[SKIP] sdedit_ll gen ${STAG}"
  fi
  require "${GDIR}/generated/199.png"
done

echo "===== [3] Standard-7 eval (inter rows, refs cached) @ $(date -Iseconds) ====="
# Backup the FT/manifest results (33 rows) BEFORE eval_standard7 overwrites results.json.
cp -f "${STD7_OUT}/results.json" "${OUT}/results_ft_ref_backup.json" || true
# Build inter manifest: 10 inter rows.
MAN="${OUT}/manifest_inter_ll.json"
"${PYTHON}" - <<PY
import json
from pathlib import Path
man = {"protocol": "standard7", "rows": [], "avg_rows": []}
for sid in range(1, 11):
    stag = f"sub-{sid:02d}"
    man["rows"].append({
        "tag": f"inter_ll_{stag}",
        "group": "inter_ll-10subj",
        "display": f"inter_ll {stag}",
        "gen_dir": "${OUT}/" + stag + "/generation/inter_ll/generated",
    })
Path("${MAN}").write_text(json.dumps(man, indent=2), encoding="utf-8")
print(f"[OK] manifest rows={len(man['rows'])}")
PY
"${PYTHON}" scripts/nda/eval_standard7.py \
  --manifest "${MAN}" \
  --images-root "${IMAGES_ROOT}" \
  --out-dir "${STD7_OUT}" \
  --device "${DEVICE}" \
  --batch-size 16
# stash inter results separately, then restore FT backup into shared results.json
cp -f "${STD7_OUT}/results.json" "${OUT}/results_inter.json"
cp -f "${OUT}/results_ft_ref_backup.json" "${STD7_OUT}/results.json"
echo "[OK] FT results restored; inter results at ${OUT}/results_inter.json"

echo "===== [4] Pooled FID (inter_ll full10) @ $(date -Iseconds) ====="
"${PYTHON}" scripts/nda/eval_pooled_fid.py \
  --root "${OUT}" \
  --tag "inter_ll" \
  --images-root "${IMAGES_ROOT}" \
  --output-json "${OUT}/metrics_pooled_fid_inter_ll.json" \
  --device "${DEVICE}" \
  --batch-size 32

echo "===== [5] Compare table (FT vs inter zero-shot) @ $(date -Iseconds) ====="
"${PYTHON}" - <<'PY'
import json
from pathlib import Path
OUT = Path("/project/peilab/why/NeuroBridge/outputs/inter_ll_full10")
ft = json.loads((OUT / "results_ft_ref_backup.json").read_text(encoding="utf-8"))["rows"]
it = json.loads((OUT / "results_inter.json").read_text(encoding="utf-8"))["rows"]
by_ft = {r["tag"]: r for r in ft}
by_it = {r["tag"]: r for r in it}
keys = ["pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "swav"]

def avg(rows, key):
    vals = [r[key] for r in rows if key in r and r[key] is not None]
    return sum(vals) / len(vals) if vals else None

ft_rows, inter_rows = [], []
for s in range(1, 11):
    stag = f"sub-{s:02d}"
    if f"sdedit_ll_{stag}" in by_ft: ft_rows.append(by_ft[f"sdedit_ll_{stag}"])
    if f"inter_ll_{stag}" in by_it: inter_rows.append(by_it[f"inter_ll_{stag}"])
hdr = f"{'metric':>10s} {'FT-avg':>10s} {'INTER-avg':>10s} {'delta':>10s}"
print(hdr); print("-" * len(hdr))
out = []
for k in keys:
    f, i = avg(ft_rows, k), avg(inter_rows, k)
    d = (i - f) if (f is not None and i is not None) else None
    print(f"{k:>10s} {(f'{f:.4f}' if f is not None else '—'):>10s} "
          f"{(f'{i:.4f}' if i is not None else '—'):>10s} {(f'{d:+.4f}' if d is not None else '—'):>10s}")
    out.append({"metric": k, "ft_avg": f, "inter_avg": i, "delta": d})
pf = json.loads((OUT / "metrics_pooled_fid_inter_ll.json").read_text(encoding="utf-8"))
pfv = pf.get("pooled_fid_unique_gt")
print(f"{'pooledFID':>10s} {'131.34(FT)':>10s} {pfv:>10.2f}")
md = [
  "# Zero-shot inter-subject (`inter_ll`) vs per-subject FT (`sdedit_ll`) — 10-subject avg",
  "",
  "Protocol: standard-7 (Ozcelik/ATM/MindEye family) + pooled FID. Decode recipe identical "
  "(LL-RGB init from the same per-subject VAE structure expert + HCMA prompts, s=0.82).",
  "The ONLY difference is the semantic embed source:",
  "- `sdedit_ll` (FT): per-subject FT of the MG-Flow semantic tower on each subject.",
  "- `inter_ll` (ZERO): LOSO-fold MG-Flow ckpt (9-subject pretrain, subject never seen) pure forward, NO FT.",
  "",
  "| Metric | FT 10-subj avg | inter zero-shot avg | delta |",
  "|---|---:|---:|---:|",
]
for o in out:
    md.append(f"| {o['metric']} | {(f\"{o['ft_avg']:.4f}\" if o['ft_avg'] is not None else '—')} | "
              f"{(f\"{o['inter_avg']:.4f}\" if o['inter_avg'] is not None else '—')} | "
              f"{(f\"{o['delta']:+.4f}\" if o['delta'] is not None else '—')} |")
md.append(f"| pooled FID | 131.34 | {pfv:.2f} | {pfv-131.34:+.2f} |")
md += ["", "Verdict (semantic tower): if CLIP/Alex2/5/Inc/SwAV deltas are small (|d|<0.01), "
          "pure inter-subject (no per-subject FT) reaches the same semantic tier → the FT step is "
          "not what buys SOTA; if deltas are large, per-subject FT matters and inter needs Phase-2 fixes."]
(OUT / "FT_VS_INTER_TABLE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("wrote", OUT / "FT_VS_INTER_TABLE.md")
json5 = {"n_ft": len(ft_rows), "n_inter": len(inter_rows),
         "ft_avg": {k: avg(ft_rows, k) for k in keys},
         "inter_avg": {k: avg(inter_rows, k) for k in keys},
         "pooled_fid_inter": pfv}
(OUT / "ft_vs_inter.json").write_text(json.dumps(json5, indent=2), encoding="utf-8")
PY

echo "{\"pipeline\":\"inter_ll_full10\",\"finished\":\"$(date -Iseconds)\"}" > "${OUT}/job_done.json"
du -sh "${OUT}" 2>/dev/null || true
echo "===== DONE inter_ll full10 @ $(date -Iseconds) ====="
