#!/usr/bin/env bash
# Chase Alex2 toward ATM SOTA with ATM/CogCap-style low-level pathway,
# while protecting CLIP / Alex5 / Inception / SwAV / FID.
# Evaluation = official seven (ATM / MindEye / CogCap protocol).
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/alex2_sota_official/sub-08}"
HCMA10="${HCMA10:-${NB_ROOT}/outputs/hcma_10subj}"
A2F="${A2F:-${NB_ROOT}/outputs/alex2_first/sub-08}"
OVN="${OVN:-${NB_ROOT}/outputs/overnight_struct_v2/sub-08}"
TCDA="${TCDA:-${NB_ROOT}/outputs/tcda/sub-08}"
LL="${LL:-${NB_ROOT}/outputs/lowlevel_decoder/sub-08}"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

SEM="${HCMA10}/sub-08/generation/hcma_full_a40/generated"
EMB="${HCMA10}/sub-08/ft/embeds/blend_nda_cfm_f_a40_test.npy"
PROMPT="${HCMA10}/prompts/prompts_full_hcma_test.json"
ZTR="${HCMA10}/sub-08/zret/z_ret_train.npy"
ZTE="${HCMA10}/sub-08/zret/z_ret_test.npy"
PC="${TCDA}/train/pred_pc_rgb_512"
LL_RGB="${LL}/vae_head/pred_lowlevel_rgb_512"

mkdir -p "${OUT}/generation" "${OUT}/metrics" "${OUT}/alex_mid" "${OUT}/logs" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" || true
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export DEVICE OUT NB_ROOT IMAGES_ROOT PAPER_OVN="${OVN}/metrics/paper_metrics.json"

echo "{\"pipeline\":\"alex2_sota_official\",\"started\":\"$(date -Iseconds)\",\"eval\":\"ATM/MindEye/CogCap seven\"}" > "${OUT}/job_running.json"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
require "${SEM}/000.png"; require "${EMB}"; require "${PROMPT}"
require "${ZTR}"; require "${ZTE}"; require "${PC}/000.png"; require "${LL_RGB}/000.png"

link_tag() {
  local tag="$1" src="$2"
  local dst="${OUT}/generation/${tag}"
  mkdir -p "${dst}"
  [[ -e "${dst}/generated" ]] || ln -sfn "${src}" "${dst}/generated"
  echo "{\"tag\":\"${tag}\",\"source\":\"${src}\"}" > "${dst}/metrics.json"
}

# ---------- [0] baselines ----------
echo "===== [0] baselines @ $(date -Iseconds) ====="
link_tag "ref_hcma_full_a40" "${SEM}"
# best deployable from alex2_first
if [[ -f "${A2F}/generation/combo_d40_luma_pc_a060/generated/199.png" ]] || [[ -L "${A2F}/generation/combo_d40_luma_pc_a060/generated" ]]; then
  link_tag "combo_d40_luma_pc_a060" "$(readlink -f "${A2F}/generation/combo_d40_luma_pc_a060/generated" 2>/dev/null || echo "${OVN}/generation/combo_d40_luma_pc_a060/generated")"
fi
if [[ -d "${OVN}/generation/combo_d40_luma_pc_a060/generated" ]]; then
  link_tag "combo_d40_luma_pc_a060" "${OVN}/generation/combo_d40_luma_pc_a060/generated"
fi
if [[ -d "${A2F}/generation/alex_luma_pc_a048/generated" ]]; then
  link_tag "alex_luma_pc_a048" "${A2F}/generation/alex_luma_pc_a048/generated"
fi

# ---------- [1] EEG→Alex mid head (official Alex2 layer) ----------
echo "===== [1] Alex mid head @ $(date -Iseconds) ====="
ALEX_OUT="${OUT}/alex_mid"
U_ALEX="${ALEX_OUT}/u_alex.npy"
if [[ ! -f "${U_ALEX}" ]]; then
  "${PYTHON}" scripts/nda/train_eeg_alex_mid_head.py \
    --eeg-train-npy "${ZTR}" \
    --eeg-test-npy "${ZTE}" \
    --images-root "${IMAGES_ROOT}" \
    --output-dir "${ALEX_OUT}" \
    --num-epochs 25 \
    --batch-size 64 \
    --lr 1e-3 \
    --max-train 8000 \
    --device "${DEVICE}"
else
  echo "[SKIP] alex mid"
fi
require "${U_ALEX}"

# ---------- [2] ATM-style low-level img2img (Pc / LL init + HCMA IP/text) ----------
echo "===== [2] ATM img2img @ $(date -Iseconds) ====="
run_i2i() {
  local tag="$1" ll="$2" strength="$3"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; return 0; fi
  "${PYTHON}" scripts/nda/generate_lowlevel_decode.py \
    --mode img2img \
    --lowlevel-dir "${ll}" \
    --embed-npy "${EMB}" \
    --prompts-json "${PROMPT}" \
    --strength "${strength}" \
    --ip-scale 1.0 \
    --gen-steps 28 \
    --gen-guidance 5.0 \
    --output-dir "${gdir}" \
    --tag "${tag}" \
    --seed 42 \
    --skip-metrics
}
# literature: ATM Stage-II uses low-level init + guided diffusion / img2img
for s in 0.35 0.45 0.55; do
  run_i2i "atm_i2i_pc_s$(echo $s | tr -d .)" "${PC}" "$s"
done
for s in 0.40 0.50; do
  run_i2i "atm_i2i_ll_s$(echo $s | tr -d .)" "${LL_RGB}" "$s"
done

# ---------- [3] Alex-confidence gated luma (deployable) ----------
echo "===== [3] alex-gated luma @ $(date -Iseconds) ====="
GDIR="${OUT}/generation/alex_gated_luma_pc"
if [[ ! -f "${GDIR}/generated/199.png" ]]; then
  "${PYTHON}" scripts/nda/generate_alexfeat_guided_fuse.py \
    --struct-dir "${PC}" --semantic-dir "${SEM}" --output-dir "${GDIR}" \
    --tag alex_gated_luma_pc --mode u_str --u-str-npy "${U_ALEX}" \
    --alpha-min 0.45 --alpha-max 0.80 --device "${DEVICE}"
else echo "[SKIP] alex_gated_luma_pc"; fi

# also fixed Pc luma at prior best α
if [[ ! -f "${OUT}/generation/luma_pc_a048/generated/199.png" ]]; then
  if [[ -d "${A2F}/generation/alex_luma_pc_a048/generated" ]]; then
    link_tag "luma_pc_a048" "${A2F}/generation/alex_luma_pc_a048/generated"
  else
    "${PYTHON}" scripts/nda/generate_luma_fuse.py \
      --struct-dir "${PC}" --semantic-dir "${SEM}" \
      --output-dir "${OUT}/generation/luma_pc_a048" \
      --tag luma_pc_a048 --sem-alpha 0.48
  fi
fi

# ---------- [4] Official seven for every candidate ----------
echo "===== [4] official seven @ $(date -Iseconds) ====="
for d in "${OUT}/generation"/*; do
  [[ -d "$d" ]] || continue
  tag="$(basename "$d")"
  gen="${d}/generated"
  [[ -f "${gen}/000.png" || -L "${gen}" ]] || continue
  # resolve symlink completeness
  [[ -f "${gen}/199.png" ]] || continue
  outj="${OUT}/metrics/${tag}_seven.json"
  if [[ -f "${outj}" ]]; then echo "[SKIP] seven ${tag}"; continue; fi
  "${PYTHON}" scripts/nda/eval_official_seven_dir.py \
    --gen-dir "${gen}" \
    --output-json "${outj}" \
    --tag "${tag}" \
    --images-root "${IMAGES_ROOT}" \
    --device "${DEVICE}" \
    --batch-size 16
  # drop per-tag twoway cache promptly
  rm -rf "${d}/_twoway_cache" 2>/dev/null || true
done
# keep one shared gt cache under OUT/metrics if created oddly
find "${OUT}" -type d -name '_twoway_cache' -exec rm -rf {} + 2>/dev/null || true

# ---------- [5] Gate + rank (select on Alex2; report full seven; ignore Pix/SSIM for selection) ----------
echo "===== [5] rank @ $(date -Iseconds) ====="
"${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out=Path(os.environ["OUT"])
rows=[]
for p in sorted((out/"metrics").glob("*_seven.json")):
    rows.append(json.loads(p.read_text()))
ref=next(r for r in rows if r["tag"]=="ref_hcma_full_a40")
ATM={"alex2":0.776,"alex5":0.866,"inception":0.734,"clip":0.786,"swav":0.582,"pixcorr":0.160,"ssim":0.345}
COGCAP={"alex2":0.754,"alex5":0.623,"inception":0.669,"clip":0.715,"swav":0.590,"pixcorr":0.150,"ssim":0.347}
ranked=[]
for r in rows:
    gate=(
        float(r["clip"]) >= float(ref["clip"])-0.01
        and float(r["alex5"]) >= float(ref["alex5"])-0.01
        and float(r["inception"]) >= float(ref["inception"])-0.01
        and float(r["swav"]) <= float(ref["swav"])+0.02
        and float(r["fid"]) <= float(ref["fid"])+15
    )
    ranked.append({
        **{k:r[k] for k in ["tag","pixcorr","ssim","alex2","alex5","inception","clip","swav","fid"]},
        "pass_gate":gate,
        "delta_alex2":float(r["alex2"])-float(ref["alex2"]),
        "delta_clip":float(r["clip"])-float(ref["clip"]),
        "gap_atm_alex2":float(r["alex2"])-ATM["alex2"],
        "gap_cog_alex2":float(r["alex2"])-COGCAP["alex2"],
    })
ranked.sort(key=lambda x:(x["pass_gate"], x["alex2"], -x["swav"]), reverse=True)
gated=[r for r in ranked if r["pass_gate"]]
best=gated[0] if gated else ranked[0]
summary={
  "pipeline":"alex2_sota_official",
  "literature":{
    "ATM":"NeurIPS'24 dual pathway + MindEye seven (Reconstruction_Metrics)",
    "CogCap":"AAAI'25 modality experts (image/text/depth) + same seven",
    "ours_adaptation":"HCMA semantic frozen + ATM-style img2img low-level init + EEG→Alex mid confidence",
  },
  "eval_protocol":"official seven via erdc_twoway + skimage SSIM + SwAV-ResNet50",
  "selection":"max Alex2 under CLIP/A5/Inc/SwAV/FID gate; PixCorr/SSIM reported not used for selection",
  "ref": {k:ref[k] for k in ["tag","pixcorr","ssim","alex2","alex5","inception","clip","swav","fid"]},
  "sota":{"ATM_sub08":ATM,"CogCap_10subj_mean":COGCAP},
  "best_gated": best if best.get("pass_gate") else None,
  "best_overall_alex2": max(ranked, key=lambda x:x["alex2"]),
  "n_pass": len(gated),
  "all_ranked": ranked,
}
(out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
(out/"best_tag.txt").write_text(best["tag"]+"\n",encoding="utf-8")
lines=[
"# Alex2 SOTA chase — official seven (sub-08)",
"",
"Protocol: ATM/MindEye/CogCap **standard seven** (erdc 2-way + skimage SSIM + SwAV).",
f"Ref HCMA: A2={ref['alex2']:.3f} A5={ref['alex5']:.3f} CLIP={ref['clip']:.3f} Inc={ref['inception']:.3f} SwAV={ref['swav']:.3f} FID={ref['fid']:.1f}",
f"ATM: A2={ATM['alex2']:.3f} A5={ATM['alex5']:.3f} | CogCap: A2={COGCAP['alex2']:.3f} A5={COGCAP['alex5']:.3f}",
"",
"| tag | pass | Pix | SSIM | A2 | A5 | Inc | CLIP | SwAV | FID | ΔA2 | gapATM |",
"|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in ranked:
    lines.append(
      f"| `{r['tag']}` | {int(r['pass_gate'])} | {r['pixcorr']:.3f} | {r['ssim']:.3f} | {r['alex2']:.3f} | {r['alex5']:.3f} | "
      f"{r['inception']:.3f} | {r['clip']:.3f} | {r['swav']:.3f} | {r['fid']:.1f} | {r['delta_alex2']:+.3f} | {r['gap_atm_alex2']:+.3f} |"
    )
(out/"OFFICIAL_SEVEN_TABLE.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print(json.dumps({"best_gated":summary["best_gated"],"n_pass":len(gated)},indent=2))
PY

echo "{\"pipeline\":\"alex2_sota_official\",\"finished\":\"$(date -Iseconds)\",\"best\":\"$(cat "${OUT}/best_tag.txt")\"}" > "${OUT}/job_done.json"
echo "[DONE] see ${OUT}/OFFICIAL_SEVEN_TABLE.md"
