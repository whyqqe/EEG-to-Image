#!/usr/bin/env bash
# Alex2-first overnight (sub-08): raise AlexNet(2) 2-way without hurting
# CLIP / Alex5 / Inception / SwAV / FID. PixCorr & SSIM are NOT selection criteria.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/alex2_first/sub-08}"
HCMA10="${HCMA10:-${NB_ROOT}/outputs/hcma_10subj}"
OVN="${OVN:-${NB_ROOT}/outputs/overnight_struct_v2/sub-08}"
TCDA="${TCDA:-${NB_ROOT}/outputs/tcda/sub-08}"
LL="${LL:-${NB_ROOT}/outputs/lowlevel_decoder/sub-08}"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
ERDC_TW="/project/peilab/why/eeg-brainit/scripts/erdc_twoway_metrics.py"

SEM="${HCMA10}/sub-08/generation/hcma_full_a40/generated"
PC="${TCDA}/train/pred_pc_rgb_512"
LL_RGB="${LL}/vae_head/pred_lowlevel_rgb_512"
U_STR="${OVN}/depth/depth_head/u_str.npy"
PAPER_OVN="${OVN}/metrics/paper_metrics.json"

mkdir -p "${OUT}/generation" "${OUT}/metrics" "${OUT}/logs" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" || true
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export DEVICE OUT PAPER_OVN IMAGES_ROOT NB_ROOT

echo "{\"pipeline\":\"alex2_first\",\"started\":\"$(date -Iseconds)\",\"goal\":\"Alex2↑ under semantic gate\"}" > "${OUT}/job_running.json"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
require "${SEM}/000.png"
require "${PC}/000.png"
require "${LL_RGB}/000.png"
require "${ERDC_TW}"
require "${PAPER_OVN}"

# ---------- [0] register / symlink reusable gens ----------
echo "===== [0] register candidates @ $(date -Iseconds) ====="
link_tag() {
  local tag="$1" src="$2"
  local dst="${OUT}/generation/${tag}"
  mkdir -p "${dst}"
  if [[ ! -e "${dst}/generated" ]]; then
    ln -sfn "${src}" "${dst}/generated"
  fi
  echo "{\"tag\":\"${tag}\",\"source\":\"${src}\"}" > "${dst}/metrics.json"
}

link_tag "ref_hcma_full_a40" "${SEM}"
# overnight candidates that preserved semantics in paper metrics
for t in luma_pc_a055 luma_pc_a065 luma_pc_a075 luma_ll_a070 luma_ll_a080 gated_luma_pc \
         combo_d40_luma_pc_a060 combo_d40_luma_pc_a070 depth_fix_cn040; do
  if [[ -f "${OVN}/generation/${t}/generated/199.png" ]]; then
    link_tag "${t}" "${OVN}/generation/${t}/generated"
  fi
done

# ---------- [1] new Alex2-oriented fuses (disk-light) ----------
echo "===== [1] new fuses @ $(date -Iseconds) ====="
run_luma() {
  local tag="$1" struct="$2" alpha="$3"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; return 0; fi
  "${PYTHON}" scripts/nda/generate_luma_fuse.py \
    --struct-dir "${struct}" --semantic-dir "${SEM}" \
    --output-dir "${gdir}" --tag "${tag}" --sem-alpha "${alpha}"
}
# fine sweep around likely Alex2 sweet spot
for a in 0.48 0.52 0.58 0.62 0.68; do
  run_luma "alex_luma_pc_a$(echo $a | tr -d .)" "${PC}" "$a"
done
run_luma "alex_luma_ll_a065" "${LL_RGB}" 0.65
run_luma "alex_luma_ll_a075" "${LL_RGB}" 0.75

# AlexNet-feature guided (deployable u_str + oracle ceiling)
if [[ -f "${U_STR}" ]]; then
  GDIR="${OUT}/generation/alex_ustr_pc"
  if [[ ! -f "${GDIR}/generated/199.png" ]]; then
    "${PYTHON}" scripts/nda/generate_alexfeat_guided_fuse.py \
      --struct-dir "${PC}" --semantic-dir "${SEM}" --output-dir "${GDIR}" \
      --tag alex_ustr_pc --mode u_str --u-str-npy "${U_STR}" \
      --alpha-min 0.48 --alpha-max 0.82 --device "${DEVICE}"
  else echo "[SKIP] alex_ustr_pc"; fi
fi
GDIR="${OUT}/generation/alex_oracle_pc"
if [[ ! -f "${GDIR}/generated/199.png" ]]; then
  "${PYTHON}" scripts/nda/generate_alexfeat_guided_fuse.py \
    --struct-dir "${PC}" --semantic-dir "${SEM}" --output-dir "${GDIR}" \
    --tag alex_oracle_pc --mode oracle --alphas "0.45,0.55,0.65,0.75,0.85" \
    --images-root "${IMAGES_ROOT}" --device "${DEVICE}"
else echo "[SKIP] alex_oracle_pc"; fi

# ---------- [2] erdc 2-way (Alex2/5/CLIP/Inc) ----------
echo "===== [2] erdc_twoway @ $(date -Iseconds) ====="
# reuse HCMA official erdc for ref
REF_ERDC="${HCMA10}/sub-08/metrics/erdc_2wc.json"
if [[ -f "${REF_ERDC}" && ! -f "${OUT}/metrics/ref_hcma_full_a40_erdc_2wc.json" ]]; then
  "${PYTHON}" - <<PY
import json
from pathlib import Path
src=json.loads(Path("${REF_ERDC}").read_text())
out={"n":src.get("n",200),"tag":"sub08_ref_hcma_full_a40","gen_dir":"${SEM}","twoway":src["twoway"]}
Path("${OUT}/metrics/ref_hcma_full_a40_erdc_2wc.json").write_text(json.dumps(out,indent=2))
print("[OK] reused HCMA erdc for ref")
PY
fi
for d in "${OUT}/generation"/*; do
  [[ -d "$d" ]] || continue
  tag="$(basename "$d")"
  gen="${d}/generated"
  [[ -f "${gen}/199.png" ]] || continue
  outj="${OUT}/metrics/${tag}_erdc_2wc.json"
  if [[ -f "${outj}" ]]; then echo "[SKIP] erdc ${tag}"; continue; fi
  "${PYTHON}" "${ERDC_TW}" \
    --gen-dir "${gen}" --images-root "${IMAGES_ROOT}" \
    --output-json "${outj}" --tag "sub08_${tag}"
done

# ---------- [3] SwAV + FID bundle ----------
echo "===== [3] SwAV/FID bundles @ $(date -Iseconds) ====="
"${PYTHON}" - <<'PY'
import json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image

nb = Path(os.environ.get("NB_ROOT", "/project/peilab/why/NeuroBridge"))
out = Path(os.environ["OUT"])
ovn_paper = Path(os.environ["PAPER_OVN"])
images_root = Path(os.environ["IMAGES_ROOT"])
device_s = os.environ.get("DEVICE", "cuda:0")

sys.path.insert(0, str(nb / "scripts" / "nda"))
sys.path.insert(0, "/project/peilab/why/eeg-brainit/scripts")
from eval_standard_seven_table import compute_swav_distance  # type: ignore
from eval_atm_pipeline import list_test_images  # type: ignore
from eval_paper_metrics import eval_folder  # type: ignore
import torch

fid_map = {}
if ovn_paper.is_file():
    for r in json.loads(ovn_paper.read_text())["results"]:
        fid_map[r["tag"]] = float(r["fid"])
# HCMA ref FID
hcma_paper = nb / "outputs/hcma_10subj/sub-08/metrics/paper_metrics.json"
if hcma_paper.is_file():
    hp = json.loads(hcma_paper.read_text())
    if "results" in hp:
        for r in hp["results"]:
            if "hcma_full" in r.get("tag", "") or r.get("tag") == "hcma_full_a40":
                fid_map["ref_hcma_full_a40"] = float(r["fid"])
    elif "fid" in hp:
        fid_map["ref_hcma_full_a40"] = float(hp["fid"])
# known from overnight summary
fid_map.setdefault("ref_hcma_full_a40", 146.43370056152344)

# also copy ref SwAV if present
ref_swav_p = nb / "outputs/hcma_10subj/sub-08/metrics/swav.json"
ref_swav = float(json.loads(ref_swav_p.read_text())["swav"]) if ref_swav_p.is_file() else None

gt = list_test_images(images_root)
device = torch.device(device_s if torch.cuda.is_available() else "cpu")

for d in sorted((out / "generation").iterdir()):
    if not d.is_dir():
        continue
    tag = d.name
    gen = d / "generated"
    if not (gen / "199.png").is_file():
        continue
    bundle_p = out / "metrics" / f"{tag}_bundle.json"
    erdc_p = out / "metrics" / f"{tag}_erdc_2wc.json"
    if not erdc_p.is_file():
        print(f"[WARN] skip bundle {tag}: no erdc")
        continue
    if bundle_p.is_file():
        print(f"[SKIP] bundle {tag}")
        continue
    tw = json.loads(erdc_p.read_text())["twoway"]
    gens = [gen / f"{i:03d}.png" for i in range(200)]
    if tag == "ref_hcma_full_a40" and ref_swav is not None:
        swav = ref_swav
    else:
        swav = compute_swav_distance(gens, gt[:200], device, batch_size=16)
    fid = fid_map.get(tag)
    if fid is None:
        try:
            row = eval_folder(gen, gt[:200], device, tag=tag, batch_size=16)
            fid = float(row["fid"])
        except Exception as e:
            print(f"[WARN] FID failed for {tag}: {e}")
            fid = None
    bundle = {
        "tag": tag,
        "alex2": float(tw["alex2"]),
        "alex5": float(tw["alex5"]),
        "inception": float(tw["inception"]),
        "clip": float(tw["clip"]),
        "swav": float(swav),
        "fid": fid,
        "gen_dir": str(gen),
    }
    bundle_p.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"[OK] {tag} A2={bundle['alex2']:.3f} CLIP={bundle['clip']:.3f} SwAV={bundle['swav']:.3f} FID={fid}")
PY

# ---------- [4] gate + rank (ignore PixCorr/SSIM) ----------
echo "===== [4] gate rank @ $(date -Iseconds) ====="
"${PYTHON}" scripts/nda/eval_alex2_gate_rank.py \
  --metrics-dir "${OUT}/metrics" \
  --ref-tag ref_hcma_full_a40 \
  --output-json "${OUT}/summary.json" \
  --output-md "${OUT}/ALEX2_FIRST_TABLE.md" \
  --alex2-target 0.776

BEST=$(python -c "import json;print(json.load(open('${OUT}/summary.json')).get('best_gated',{}) or {}).get('tag','')")
echo "${BEST}" > "${OUT}/best_tag.txt"
echo "{\"pipeline\":\"alex2_first\",\"finished\":\"$(date -Iseconds)\",\"best\":\"${BEST}\"}" > "${OUT}/job_done.json"
echo "[DONE] best_gated=${BEST}  see ${OUT}/ALEX2_FIRST_TABLE.md"
