#!/usr/bin/env bash
# Overnight Structure v2 (sub-08): lift PixCorr/SSIM without collapsing HCMA semantics.
#
# Mechanisms (literature → local adaptation):
#   1) ATM/MLSP'25: dedicated low-level branch + tunable strength (here: Pc/LL + gated luma / freq)
#   2) CogCapPro: depth as structure prior (here: retrain EEG→DepthHead on HCMA z_ret, Depth-CN)
#   3) SGDM: explicit structure ControlNet (Depth-CN on EEG-predicted depth, NOT neighbor-only)
#   4) TOP1_STRUCTURE_PLAN: GT-supervised DepthHead; freeze HCMA semantic embeds/prompts
#
# Gate: keep CLIP 2-way within 1.5pt of HCMA ref; maximize SSIM then PixCorr.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/overnight_struct_v2/sub-08}"
HCMA10="${HCMA10:-${NB_ROOT}/outputs/hcma_10subj}"
T1="${T1:-${NB_ROOT}/outputs/top1_structure/sub-08}"
TCDA="${TCDA:-${NB_ROOT}/outputs/tcda/sub-08}"
LL="${LL:-${NB_ROOT}/outputs/lowlevel_decoder/sub-08}"
NDA_SS="${NDA_SS:-${NB_ROOT}/outputs/nda_ss/sub-08}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

SEM_DIR="${HCMA10}/sub-08/generation/hcma_full_a40/generated"
EMB="${HCMA10}/sub-08/ft/embeds/blend_nda_cfm_f_a40_test.npy"
PROMPT="${HCMA10}/prompts/prompts_full_hcma_test.json"
NEIGH="${HCMA10}/sub-08/memory/rag_soft5_neighbor_idx_test.npy"
ZTR="${HCMA10}/sub-08/zret/z_ret_train.npy"
ZTE="${HCMA10}/sub-08/zret/z_ret_test.npy"
DTR="${T1}/track_s/gt_depth/train_depth_64.npy"
DTE="${T1}/track_s/gt_depth/test_depth_64.npy"
PC="${TCDA}/train/pred_pc_rgb_512"
# Prefer real RGB dir (old generation/lowlevel_only/*.png are often broken symlinks after prune)
LL_RGB="${LL}/vae_head/pred_lowlevel_rgb_512"
LL_VAE_NPY="${LL}/vae_head/pred_vae_test.npy"
export LL_RGB LL_VAE_NPY DEVICE

mkdir -p "${OUT}/depth" "${OUT}/generation" "${OUT}/metrics" "${OUT}/compare" "${OUT}/logs" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"

echo "{\"pipeline\":\"overnight_struct_v2\",\"started\":\"$(date -Iseconds)\",\"mechanisms\":[\"eeg_depth_head\",\"depth_cn\",\"luma_pc\",\"freq_fuse\",\"gated_luma\"]}" \
  > "${OUT}/job_running.json"

require() {
  local p="$1"
  if [[ ! -e "$p" ]]; then
    echo "[FATAL] missing required asset: $p" >&2
    exit 1
  fi
}

# Restore pruned low-level RGB from cached VAE latents if needed
if [[ ! -f "${LL_RGB}/000.png" ]]; then
  echo "[INFO] LL RGB missing; decode from ${LL_VAE_NPY}"
  require "${LL_VAE_NPY}"
  "${PYTHON}" - <<'PY'
import os, sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
sys.path.insert(0, "scripts/nda")
from train_eeg_vae_head import resolve_vae, decode_latents
hub = Path(os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
device = torch.device(os.environ.get("DEVICE", "cuda:0") if torch.cuda.is_available() else "cpu")
lat = torch.from_numpy(np.load(os.environ["LL_VAE_NPY"]).astype(np.float32))
out = Path(os.environ["LL_RGB"]); out.mkdir(parents=True, exist_ok=True)
vae = resolve_vae(hub, device)
bs = 8
for s in tqdm(range(0, len(lat), bs), desc="restore-ll-rgb"):
    chunk = lat[s:s+bs].to(device)
    imgs = decode_latents(vae, chunk, 0.13025)
    for j, im in enumerate(imgs):
        im.save(out / f"{s+j:03d}.png")
print(f"[OK] restored {len(lat)} -> {out}")
PY
fi

require "${SEM_DIR}/000.png"
require "${EMB}"
require "${PROMPT}"
require "${ZTR}"; require "${ZTE}"
require "${DTR}"; require "${DTE}"
require "${PC}/000.png"
require "${LL_RGB}/000.png"
require "${NEIGH}"
# ---------- [0] symlink semantic reference ----------
REF="${OUT}/generation/ref_hcma_full_a40"
mkdir -p "${REF}"
[[ -e "${REF}/generated" ]] || ln -sfn "${SEM_DIR}" "${REF}/generated"
echo '{"tag":"ref_hcma_full_a40"}' > "${REF}/metrics.json"

# ---------- [1] Retrain EEG→DepthHead on HCMA z_ret (GT depth labels) ----------
echo "===== [1] EEG→DepthHead @ $(date -Iseconds) ====="
DEPTH_OUT="${OUT}/depth/depth_head"
PRED_DEPTH="${DEPTH_OUT}/pred_depth_rgb_512"
U_STR="${DEPTH_OUT}/u_str.npy"
CN_COCA="${DEPTH_OUT}/cn_scale_coca.npy"
if [[ ! -f "${PRED_DEPTH}/199.png" ]]; then
  "${PYTHON}" scripts/nda/train_eeg_depth_head.py \
    --eeg-train-npy "${ZTR}" \
    --eeg-test-npy "${ZTE}" \
    --depth-train-npy "${DTR}" \
    --depth-test-npy "${DTE}" \
    --output-dir "${DEPTH_OUT}" \
    --num-epochs 50 \
    --batch-size 256 \
    --lr 1e-3 \
    --lambda-grad 0.5 \
    --device "${DEVICE}"
else
  echo "[SKIP] depth head"
fi
test -f "${PRED_DEPTH}/199.png"
test -f "${U_STR}"

# ---------- [2] Depth-CN generations (SGDM/CogCapPro-style structure prior) ----------
echo "===== [2] Depth-CN gens @ $(date -Iseconds) ====="
run_depth_cn() {
  local tag="$1" cn="$2"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; return 0; fi
  local extra=()
  if [[ -f "${CN_COCA}" ]]; then extra+=(--cn-scale-npy "${CN_COCA}"); fi
  "${PYTHON}" scripts/nda/generate_cn_ip_decode.py \
    --embed-npy "${EMB}" \
    --neighbor-idx-npy "${NEIGH}" \
    --output-dir "${gdir}" \
    --tag "${tag}" \
    --seed 42 \
    --control-type depth \
    --depth-sample-dir "${PRED_DEPTH}" \
    --cn-scale "${cn}" \
    --ip-scale 1.0 \
    --gen-steps 30 \
    --gen-guidance 5.0 \
    --prompts-json "${PROMPT}" \
    --skip-metrics \
    "${extra[@]}"
}

# fixed cn sweeps (COCA npy overrides per-sample when present — also run pure fixed)
run_depth_cn "depth_cn035" 0.35
run_depth_cn "depth_cn045" 0.45
run_depth_cn "depth_cn055" 0.55

# fixed cn without coca npy for cleaner ablation
run_depth_cn_fixed() {
  local tag="$1" cn="$2"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; return 0; fi
  "${PYTHON}" scripts/nda/generate_cn_ip_decode.py \
    --embed-npy "${EMB}" \
    --neighbor-idx-npy "${NEIGH}" \
    --output-dir "${gdir}" \
    --tag "${tag}" \
    --seed 42 \
    --control-type depth \
    --depth-sample-dir "${PRED_DEPTH}" \
    --cn-scale "${cn}" \
    --ip-scale 1.0 \
    --gen-steps 30 \
    --gen-guidance 5.0 \
    --prompts-json "${PROMPT}" \
    --skip-metrics
}
run_depth_cn_fixed "depth_fix_cn040" 0.40
run_depth_cn_fixed "depth_fix_cn050" 0.50

# ---------- [3] Post-hoc structure injection on frozen HCMA semantics ----------
echo "===== [3] Luma / freq / gated fuse @ $(date -Iseconds) ====="
run_luma() {
  local tag="$1" struct="$2" alpha="$3"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; return 0; fi
  "${PYTHON}" scripts/nda/generate_luma_fuse.py \
    --struct-dir "${struct}" \
    --semantic-dir "${SEM_DIR}" \
    --output-dir "${gdir}" \
    --tag "${tag}" \
    --sem-alpha "${alpha}"
}
run_freq() {
  local tag="$1" struct="$2" sigma="$3"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; return 0; fi
  "${PYTHON}" scripts/nda/generate_freq_fuse.py \
    --struct-dir "${struct}" \
    --semantic-dir "${SEM_DIR}" \
    --output-dir "${gdir}" \
    --tag "${tag}" \
    --sigma "${sigma}"
}

# Pc (blurry low-level from TCDA) — ATM-style low-level pipe
for a in 0.55 0.65 0.75; do
  run_luma "luma_pc_a$(echo $a | tr -d .)" "${PC}" "$a"
done
# VAE/lowlevel-only
for a in 0.70 0.80; do
  run_luma "luma_ll_a$(echo $a | tr -d .)" "${LL_RGB}" "$a"
done
# frequency: layout from Pc, texture from HCMA
for s in 8 12; do
  run_freq "freq_pc_s${s}" "${PC}" "$s"
done
# confidence-gated luma (MLSP strength control)
GATED="${OUT}/generation/gated_luma_pc"
if [[ ! -f "${GATED}/generated/199.png" ]]; then
  "${PYTHON}" scripts/nda/generate_gated_luma_fuse.py \
    --struct-dir "${PC}" \
    --semantic-dir "${SEM_DIR}" \
    --u-str-npy "${U_STR}" \
    --output-dir "${GATED}" \
    --tag gated_luma_pc \
    --alpha-min 0.50 \
    --alpha-max 0.78
fi

# combo: take best expected depth-cn gen then luma with Pc
COMBO_SRC="${OUT}/generation/depth_fix_cn040/generated"
if [[ -f "${COMBO_SRC}/000.png" ]]; then
  for a in 0.60 0.70; do
    tag="combo_d40_luma_pc_a$(echo $a | tr -d .)"
    gdir="${OUT}/generation/${tag}"
    if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; continue; fi
    "${PYTHON}" scripts/nda/generate_luma_fuse.py \
      --struct-dir "${PC}" \
      --semantic-dir "${COMBO_SRC}" \
      --output-dir "${gdir}" \
      --tag "${tag}" \
      --sem-alpha "${a}"
  done
fi

# ---------- [4] Metrics ----------
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
  --gen-root "${OUT}/generation" --tags "${VALID}" \
  --output-json "${OUT}/metrics/paper_metrics.json"

# eval_clip_2way expects a single comma-separated string (not argv list)
GEN_CSV=""
for t in ${VALID//,/ }; do
  GEN_CSV="${GEN_CSV:+$GEN_CSV,}${t}=${OUT}/generation/${t}/generated"
done
"${PYTHON}" scripts/nda/eval_clip_2way.py \
  --gen-dirs "${GEN_CSV}" \
  --output-json "${OUT}/metrics/clip_2way_report.json" \
  --device "${DEVICE}" --batch-size 16

# ---------- [5] Gate + ranking ----------
echo "===== [5] Gate vs HCMA ref @ $(date -Iseconds) ====="
"${PYTHON}" - <<PY
import json
from pathlib import Path
out=Path("${OUT}")
paper={r["tag"]:r for r in json.loads((out/"metrics/paper_metrics.json").read_text())["results"]}
tw={r["tag"]:r for r in json.loads((out/"metrics/clip_2way_report.json").read_text())["generation_2way"]}
ref=paper.get("ref_hcma_full_a40") or paper[list(paper)[0]]
# prefer known ref tw
ref_tw=tw.get("ref_hcma_full_a40", list(tw.values())[0])
ref2=float(ref_tw.get("clip_2way", 0.96))
ref_fid=float(ref.get("fid", 150))
ref_ssim=float(ref.get("ssim", 0.22))
rows=[]
for tag,p in paper.items():
    if tag not in tw: continue
    twoway=float(tw[tag]["clip_2way"])
    fid=float(p["fid"]); ssim=float(p["ssim"]); pix=float(p["pixcorr"])
    # structure-first gate: protect semantics
    pass_gate=(twoway >= ref2-0.015) and (fid <= ref_fid+25)
    # score: prioritize SSIM/PixCorr while keeping 2way
    score=(
      0.28*twoway
      + 0.32*ssim
      + 0.25*min(pix/0.20, 1.0)
      + 0.15*max(0,(320-fid)/170)
    )
    rows.append({
      "tag":tag,"clip_2way":twoway,"fid":fid,"ssim":ssim,"pixcorr":pix,
      "pass_gate":pass_gate,
      "delta_2way":twoway-ref2,"delta_ssim":ssim-ref_ssim,"delta_pix":pix-float(ref.get("pixcorr",0)),
      "score":score,
    })
rows.sort(key=lambda r: (r["pass_gate"], r["score"]), reverse=True)
best=rows[0] if rows else None
best_struct=sorted([r for r in rows if r["pass_gate"]], key=lambda r: (r["ssim"], r["pixcorr"]), reverse=True)
summary={
  "pipeline":"overnight_struct_v2",
  "mechanisms":[
    "EEG→DepthHead (GT DepthAnything labels; SGDM/CogCapPro depth prior)",
    "Depth-ControlNet on EEG-predicted depth (not neighbor-only)",
    "ATM/MLSP low-level: Pc/LL luma fuse + freq fuse + confidence-gated strength",
    "Frozen HCMA semantic embeds/prompts as IP",
  ],
  "ref_hcma":{"clip_2way":ref2,"fid":ref_fid,"ssim":ref_ssim,"pixcorr":float(ref.get("pixcorr",0))},
  "gate_rule":"2way>=ref-0.015 AND fid<=ref+25",
  "best_gated":best,
  "best_struct_among_gated": best_struct[0] if best_struct else None,
  "all_ranked":rows,
  "targets":{"pixcorr":0.15,"ssim":0.30,"note":"structure targets; semantics protected by gate"},
}
(out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
(out/"best_tag.txt").write_text((best or {}).get("tag",""),encoding="utf-8")
print(json.dumps({"best_gated":best,"best_struct":best_struct[0] if best_struct else None},indent=2))
for r in rows[:12]:
    print(f"  {r['tag']:28s} 2way={r['clip_2way']*100:5.1f}% SSIM={r['ssim']:.3f} Pix={r['pixcorr']:.3f} FID={r['fid']:6.1f} gate={r['pass_gate']}")
PY

# ---------- [6] Compare grids for top gated ----------
echo "===== [6] Compare grids @ $(date -Iseconds) ====="
BEST=$(cat "${OUT}/best_tag.txt" || true)
if [[ -n "${BEST}" && -d "${OUT}/generation/${BEST}/generated" ]]; then
  "${PYTHON}" scripts/nda/make_compare_grid.py \
    --output-dir "${OUT}/compare" \
    --cell 160 \
    --auto-select-gen-dir "${OUT}/generation/${BEST}/generated" \
    --auto-select-k 12 \
    --device "${DEVICE}" \
    --metrics-json "${OUT}/metrics/clip_2way_report.json" \
    --cols "ref=${SEM_DIR},best=${OUT}/generation/${BEST}/generated,pc=${PC}"
fi

# standard seven on ref + best + top3 gated structure
"${PYTHON}" - <<'PY'
import json, subprocess, sys
from pathlib import Path
out=Path("/project/peilab/why/NeuroBridge/outputs/overnight_struct_v2/sub-08")
s=json.loads((out/"summary.json").read_text())
tags=["ref_hcma_full_a40"]
if s.get("best_gated"): tags.append(s["best_gated"]["tag"])
if s.get("best_struct_among_gated"):
    t=s["best_struct_among_gated"]["tag"]
    if t not in tags: tags.append(t)
# add top 2 gated by ssim
gated=[r for r in s["all_ranked"] if r["pass_gate"]]
gated=sorted(gated, key=lambda r: r["ssim"], reverse=True)
for r in gated[:3]:
    if r["tag"] not in tags: tags.append(r["tag"])
print("eval tags", tags)
# run erdc 2wc + swav-lite via paper already; write STRUCT table
rows=[]
paper={r["tag"]:r for r in json.loads((out/"metrics/paper_metrics.json").read_text())["results"]}
tw={r["tag"]:r for r in json.loads((out/"metrics/clip_2way_report.json").read_text())["generation_2way"]}
md=["# Overnight Structure v2 — results\n\n",
    "Mechanisms: EEG→DepthHead + Depth-CN; Pc/LL luma & freq fuse; confidence-gated luma.\n\n",
    "| Tag | CLIP2way↑ | FID↓ | SSIM↑ | PixCorr↑ | gate |\n|---|---|---|---|---|---|\n"]
for t in tags:
    if t not in paper or t not in tw: continue
    p,w=paper[t],tw[t]
    gated=next((r for r in s["all_ranked"] if r["tag"]==t),{})
    md.append(f"| {t} | {w['clip_2way']:.4f} | {p['fid']:.1f} | {p['ssim']:.3f} | {p['pixcorr']:.3f} | {gated.get('pass_gate','')} |\n")
md.append("\nFull ranking in `summary.json`.\n")
(out/"STRUCT_OVERNIGHT_TABLE.md").write_text("".join(md),encoding="utf-8")
print("wrote", out/"STRUCT_OVERNIGHT_TABLE.md")
PY

echo "{\"pipeline\":\"overnight_struct_v2\",\"finished\":\"$(date -Iseconds)\"}" > "${OUT}/job_done.json"
du -sh "${OUT}" || true
echo "===== DONE overnight_struct_v2 @ $(date -Iseconds) ====="
