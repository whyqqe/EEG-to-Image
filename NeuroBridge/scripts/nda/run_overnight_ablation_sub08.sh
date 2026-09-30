#!/usr/bin/env bash
# Overnight ablation: untested mechanisms, disk-light (prune PNGs after eval).
# Reuses top1_structure assets; does NOT rebuild GT depth or copy champions.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/overnight_ablation/sub-08}"
BASE="${BASE:-${NB_ROOT}/outputs/top1_structure/sub-08}"
NDA_SS="${NDA_SS:-${NB_ROOT}/outputs/nda_ss/sub-08}"
COCA_PREV="${COCA_PREV:-${NB_ROOT}/outputs/coca_depth/sub-08}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
KEEP_TOP_K="${KEEP_TOP_K:-2}"  # keep PNG sets for top-K tags only

mkdir -p "${OUT}/generation" "${OUT}/routing" "${OUT}/depth_vith" "${OUT}/track_t_ablate" "${OUT}/logs"
cd "${NB_ROOT}"

prune_tag_pngs() {
  local tag="$1"
  local gdir="${OUT}/generation/${tag}/generated"
  if [[ -d "${gdir}" ]]; then
    find "${gdir}" -type f -name '*.png' -delete 2>/dev/null || true
    echo "[DISK] pruned PNGs for ${tag}"
  fi
}

prune_all_but_keep() {
  local keep_csv="$1"
  "${PYTHON}" - <<PY
import json, shutil
from pathlib import Path
out = Path("${OUT}/generation")
keep = set(x for x in "${keep_csv}".split(",") if x)
metrics_path = Path("${OUT}/paper_metrics.json")
if not metrics_path.is_file():
    raise SystemExit(0)
results = json.loads(metrics_path.read_text()).get("results", [])
# rank by 0.5*clip + 0.5*ssim
def score(r):
    return 0.5 * float(r.get("clip_cosine") or 0) + 0.5 * float(r.get("ssim") or 0)
ranked = sorted(results, key=score, reverse=True)
auto_keep = {r["tag"] for r in ranked[: int("${KEEP_TOP_K}")]}
keep |= auto_keep
freed = 0
for d in out.iterdir():
    if not d.is_dir():
        continue
    tag = d.name
    gen = d / "generated"
    if tag in keep:
        continue
    if gen.is_dir():
        for p in gen.glob("*.png"):
            try:
                freed += p.stat().st_size
                p.unlink()
            except FileNotFoundError:
                pass
print(json.dumps({"kept_tags": sorted(keep), "approx_bytes_freed": freed}, indent=2))
PY
}

echo "===== [0] Wait/check base assets @ $(date -Iseconds) ====="
PRED_DEPTH="${BASE}/track_s/depth_head/pred_depth_rgb_512"
U_STR="${BASE}/track_s/depth_head/u_str.npy"
CN_COCA="${BASE}/track_s/depth_head/cn_scale_coca.npy"
DEPTH64_TR="${BASE}/track_s/gt_depth/train_depth_64.npy"
DEPTH64_TE="${BASE}/track_s/gt_depth/test_depth_64.npy"
NEIGH="${NDA_SS}/memory/rag_soft5_neighbor_idx_test.npy"
NEIGH_DEPTH="${COCA_PREV}/depth_cache"
EMB_IP="${NDA_SS}/blend/mem_decode_a50.npy"
PROMPT_CPA="${BASE}/prompts/prompts_cpa.json"
PROMPT_CLEAN="${BASE}/prompts/prompts_clean.json"
MARGINS_CLEAN="${BASE}/retrieval/clean/margins.npy"
NB_CKPT="${NB_ROOT}/results/things_eeg/intra-subjects/20260901-074319-sub-08/checkpoint_test_best.pth"
T_CKPT="${BASE}/track_t/checkpoint_clean_dual_best.pth"

for f in "${PRED_DEPTH}/000.png" "${U_STR}" "${DEPTH64_TR}" "${DEPTH64_TE}" "${NEIGH}" "${EMB_IP}" "${PROMPT_CPA}" "${PROMPT_CLEAN}"; do
  if [[ ! -e "${f}" ]]; then
    echo "[FATAL] missing base asset: ${f}"
    exit 1
  fi
done
# neighbor depth optional for hybrid
HAS_NB_DEPTH=0
[[ -d "${NEIGH_DEPTH}" ]] && HAS_NB_DEPTH=1

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"

# ---------- A) Routing packs (tiny) ----------
echo "===== [A] COCA text/IP/fuse routing @ $(date -Iseconds) ====="
if [[ ! -f "${OUT}/routing/routing_report.json" ]]; then
  "${PYTHON}" scripts/nda/build_coca_text_routing.py \
    --prompts-json "${PROMPT_CLEAN}" \
    --margins-npy "${MARGINS_CLEAN}" \
    --u-str-npy "${U_STR}" \
    --output-dir "${OUT}/routing"
fi
PROMPT_GATED="${OUT}/routing/prompts_gated.json"
CN_R="${OUT}/routing/cn_scale.npy"
IP_R="${OUT}/routing/ip_scale.npy"
FUSE_R="${OUT}/routing/fuse_beta.npy"

# ---------- B) Hybrid depth (symlinks) ----------
echo "===== [B] Hybrid pred/neighbor depth @ $(date -Iseconds) ====="
HYBRID="${OUT}/hybrid_depth"
if [[ "${HAS_NB_DEPTH}" -eq 1 && ! -f "${HYBRID}/hybrid_report.json" ]]; then
  "${PYTHON}" scripts/nda/build_hybrid_depth_maps.py \
    --u-str-npy "${U_STR}" \
    --pred-depth-dir "${PRED_DEPTH}" \
    --neighbor-idx-npy "${NEIGH}" \
    --neighbor-depth-dir "${NEIGH_DEPTH}" \
    --output-dir "${HYBRID}" \
    --thresh 0.45
fi

# ---------- C) DepthHead from ViT-H decode (untested input) ----------
echo "===== [C] DepthHead on ViT-H decode embeds @ $(date -Iseconds) ====="
DEC_TR="${NDA_SS}/train/z_decode_vith_train.npy"
DEC_TE="${NDA_SS}/train/z_decode_vith_test.npy"
VITH_OUT="${OUT}/depth_vith"
if [[ -f "${DEC_TR}" && -f "${DEC_TE}" && ! -f "${VITH_OUT}/depth_head_report.json" ]]; then
  "${PYTHON}" scripts/nda/train_eeg_depth_head.py \
    --eeg-train-npy "${DEC_TR}" \
    --eeg-test-npy "${DEC_TE}" \
    --depth-train-npy "${DEPTH64_TR}" \
    --depth-test-npy "${DEPTH64_TE}" \
    --output-dir "${VITH_OUT}" \
    --num-epochs 50 \
    --batch-size 256 \
    --device "${DEVICE}"
fi
PRED_VITH="${VITH_OUT}/pred_depth_rgb_512"

# ---------- D) Clean-heavier dual finetune (metrics only, no gen) ----------
echo "===== [D] λ ablation (clean-heavy, mid-strong) @ $(date -Iseconds) ====="
HCF_TRAIN="${NDA_SS}/train/hcf_train.npy"
HCF_TEST="${NDA_SS}/train/hcf_test.npy"
if [[ ! -f "${OUT}/track_t_ablate/clean_heavy/clean_dual_report.json" ]]; then
  "${PYTHON}" scripts/nda/nda_clean_dual_finetune.py \
    --init-checkpoint "${NB_CKPT}" \
    --output-dir "${OUT}/track_t_ablate/clean_heavy" \
    --subject 8 --num-epochs 30 --batch-size 512 --lr 3e-5 \
    --lambda-clean 1.5 --lambda-cpa 0.25 --lambda-mid 0.5 \
    --hcf-train "${HCF_TRAIN}" --hcf-test "${HCF_TEST}" \
    --device "${DEVICE}"
fi
# drop bulky embeds from ablate run (keep ckpt+report)
rm -f "${OUT}/track_t_ablate/clean_heavy/embeds/"*.npy 2>/dev/null || true

# ---------- E) Generation ablations (untested mechanisms) ----------
echo "===== [E] Generation ablations @ $(date -Iseconds) ====="
run_gen() {
  local tag="$1"; shift
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" || -f "${gdir}/metrics.json" ]]; then
    # if metrics exist but pngs pruned, still skip regen
    if [[ -f "${gdir}/metrics.json" && ! -f "${gdir}/generated/199.png" ]]; then
      echo "[SKIP] ${tag} (metrics kept, pngs pruned earlier)"
      return 0
    fi
    if [[ -f "${gdir}/generated/199.png" ]]; then
      echo "[SKIP] ${tag}"
      return 0
    fi
  fi
  "${PYTHON}" scripts/nda/generate_cn_ip_decode.py \
    --embed-npy "${EMB_IP}" \
    --neighbor-idx-npy "${NEIGH}" \
    --output-dir "${gdir}" \
    --tag "${tag}" --seed 42 \
    --control-type depth \
    --gen-steps 28 --gen-guidance 5.0 \
    --skip-metrics "$@"
}

# E1: MindEye fuse + pred depth + CPA (SSIM probe)
run_gen "fuse_pred_cn0.5_cpa" \
  --depth-sample-dir "${PRED_DEPTH}" --cn-scale 0.5 --ip-scale 1.0 \
  --enable-fuse --fuse-beta-npy "${FUSE_R}" \
  --prompts-json "${PROMPT_CPA}"

# E2: gated clean text + COCA cn/ip (honest abstain)
run_gen "gated_clean_coca" \
  --depth-sample-dir "${PRED_DEPTH}" \
  --cn-scale-npy "${CN_R}" --ip-scale-npy "${IP_R}" --cn-scale 0.5 --ip-scale 1.0 \
  --prompts-json "${PROMPT_GATED}"

# E3: hybrid depth + CPA
if [[ -f "${HYBRID}/hybrid_report.json" ]]; then
  run_gen "hybrid_cn0.5_cpa" \
    --depth-sample-dir "${HYBRID}" --cn-scale 0.5 --ip-scale 1.0 \
    --prompts-json "${PROMPT_CPA}"
fi

# E4: ViT-H depth head + COCA cn + CPA
if [[ -d "${PRED_VITH}" ]]; then
  run_gen "vith_depth_coca_cpa" \
    --depth-sample-dir "${PRED_VITH}" \
    --cn-scale-npy "${CN_COCA}" --cn-scale 0.5 --ip-scale 1.0 \
    --prompts-json "${PROMPT_CPA}"
fi

# E5: full stack — pred depth + gated clean + cn/ip routing + mild fuse
run_gen "fullstack_gated_fuse" \
  --depth-sample-dir "${PRED_DEPTH}" \
  --cn-scale-npy "${CN_R}" --ip-scale-npy "${IP_R}" \
  --cn-scale 0.5 --ip-scale 1.0 \
  --enable-fuse --fuse-beta-npy "${FUSE_R}" \
  --prompts-json "${PROMPT_GATED}"

echo "===== [F] Paper metrics @ $(date -Iseconds) ====="
TAGS=""
for t in fuse_pred_cn0.5_cpa gated_clean_coca hybrid_cn0.5_cpa vith_depth_coca_cpa fullstack_gated_fuse; do
  if [[ -f "${OUT}/generation/${t}/generated/199.png" || -f "${OUT}/generation/${t}/metrics.json" ]]; then
    # re-eval needs PNGs; if pruned, skip unless present
    if [[ -f "${OUT}/generation/${t}/generated/199.png" ]]; then
      TAGS="${TAGS:+$TAGS,}${t}"
    fi
  fi
done
if [[ -z "${TAGS}" ]]; then
  echo "[FATAL] no generation tags with PNGs to eval"
  exit 1
fi

"${PYTHON}" scripts/nda/eval_paper_metrics.py \
  --gen-root "${OUT}/generation" \
  --tags "${TAGS}" \
  --output-json "${OUT}/paper_metrics.json"

# prune PNGs except top-K (+ optional keep list empty)
prune_all_but_keep ""

# also drop ViT-H pred rgb after eval (large); keep npy+report
if [[ -d "${PRED_VITH}" ]]; then
  find "${PRED_VITH}" -type f -name '*.png' -delete 2>/dev/null || true
  echo "[DISK] pruned ViT-H pred depth PNGs"
fi
# hybrid dir is mostly symlinks — delete only real PNG files (not symlinks)
if [[ -d "${HYBRID}" ]]; then
  find "${HYBRID}" -type f -name '*.png' -delete 2>/dev/null || true
fi

"${PYTHON}" - <<PY
import json
from pathlib import Path
out = Path("${OUT}")
metrics = json.loads((out/"paper_metrics.json").read_text())
results = metrics.get("results", [])
def score(r):
    return 0.5*float(r.get("clip_cosine") or 0)+0.5*float(r.get("ssim") or 0)
best = max(results, key=score) if results else None
t_heavy = {}
p = out/"track_t_ablate/clean_heavy/clean_dual_report.json"
if p.is_file():
    t_heavy = json.loads(p.read_text())
vith = {}
vp = out/"depth_vith/depth_head_report.json"
if vp.is_file():
    vith = json.loads(vp.read_text())
base_sum = {}
bp = Path("${BASE}/summary.json")
if bp.is_file():
    base_sum = json.loads(bp.read_text())
summary = {
  "pipeline": "overnight_ablation_disklight",
  "base": str(Path("${BASE}")),
  "mechanisms": [
    "MindEye fuse + pred depth",
    "margin-gated clean text + COCA cn/ip",
    "hybrid pred/neighbor depth",
    "DepthHead on ViT-H decode embeds",
    "fullstack gated+fuse",
    "clean-heavy dual finetune (metrics only)",
  ],
  "disk_policy": "prune PNGs keep top-${KEEP_TOP_K}; no GT rebuild; symlink hybrid",
  "track_t_clean_heavy": {
    "best_top1_clean": t_heavy.get("best_top1_clean"),
    "best_top1_cpa": t_heavy.get("best_top1_cpa"),
    "baseline": t_heavy.get("baseline"),
  },
  "depth_vith": {"best_pearson": vith.get("best_pearson"), "final_test": vith.get("final_test")},
  "base_track_t_top1_clean": (base_sum.get("track_t") or {}).get("train_report", {}).get("best_top1_clean"),
  "best_combo": best,
  "all_gen": results,
}
(out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

# disk usage note
du -sh "${OUT}" "${OUT}/generation" 2>/dev/null || true
echo "===== DONE overnight ablation @ $(date -Iseconds) ====="
