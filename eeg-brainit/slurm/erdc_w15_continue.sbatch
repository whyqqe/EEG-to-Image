#!/usr/bin/env bash
#SBATCH --job-name=erdc-w15c
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --exclude=dgx-10,dgx-11
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# W15 continuation — skip completed steps from 527787; FID + fuse + panels + paper table

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/eeg-brainit/open_clip
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
export TMPDIR=/project/peilab/why/cache/eeg-brainit/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XFORMERS_DISABLED=1
mkdir -p "$TMPDIR" outputs/erdc/w15_metrics outputs/slurm

# FID dependency (fixed from 527787 failure)
pip install -q torch-fidelity
python -c "from torchmetrics.image.fid import FrechetInceptionDistance; print('[OK] FID deps')"

MAX_IMAGES="${ERDC_MAX_IMAGES:-0}"
EXTRA=()
[[ "${MAX_IMAGES}" != "0" ]] && EXTRA+=(--max-images "${MAX_IMAGES}")

BIT_NPY="outputs/eval/atm_pipeline_sub08/sub-08_bit_clip_1024.npy"
MET_DIR="outputs/erdc/w15_metrics"
FUSE_LAM="0.15"
MEGA_CAND="outputs/erdc/w15_mega_turbo/candidates"

full_metrics () {
  local gen="$1"; local tag="$2"
  local out="${MET_DIR}/${tag}.json"
  [[ -f "${out}" ]] && echo "[SKIP] ${tag}" && return 0
  [[ -d "${gen}" ]] || { echo "[WARN] miss ${gen}"; return 0; }
  python scripts/erdc_full_metrics.py --gen-dir "${gen}" --output-json "${out}" --tag "${tag}" "${EXTRA[@]}"
}

fid_metrics () {
  local gen="$1"; local tag="$2"
  local out="${MET_DIR}/${tag}_fid.json"
  [[ -f "${out}" ]] && echo "[SKIP] fid ${tag}" && return 0
  [[ -d "${gen}" ]] || return 0
  python scripts/erdc_fid_metrics.py --gen-dir "${gen}" --output-json "${out}" --tag "${tag}" "${EXTRA[@]}"
}

twoway () {
  local gen="$1"; local tag="$2"
  local out="${MET_DIR}/${tag}_2wc.json"
  [[ -f "${out}" ]] && echo "[SKIP] 2wc ${tag}" && return 0
  [[ -d "${gen}" ]] || return 0
  python scripts/erdc_twoway_metrics.py --gen-dir "${gen}" --output-json "${out}" --tag "${tag}" "${EXTRA[@]}"
}

bootstrap_ci () {
  local gen="$1"; local tag="$2"
  local out="${MET_DIR}/${tag}_bootstrap.json"
  [[ -f "${out}" ]] && echo "[SKIP] boot ${tag}" && return 0
  [[ -d "${gen}" ]] || return 0
  python scripts/erdc_bootstrap_ci.py --gen-dir "${gen}" --output-json "${out}" --tag "${tag}" "${EXTRA[@]}"
}

run_fuse () {
  local lam="$1"
  local tag="w15_mega_fuse_l${lam/./p}"
  local out="outputs/erdc/${tag}"
  if [[ ! -f "${out}/metrics.json" ]]; then
    echo "----- fuse λ=${lam} -----"
    python scripts/erdc_fuse_reselect.py \
      --cand-dir "${MEGA_CAND}" \
      --eeg-npy "${BIT_NPY}" \
      --lambda-struct "${lam}" \
      --output-dir "${out}" \
      "${EXTRA[@]}"
  else
    echo "[SKIP] fuse ${tag}"
  fi
  full_metrics "${out}/selected_fused" "${tag}"
  fid_metrics "${out}/selected_fused" "${tag}"
}

echo "################ [1] Verify mega bank ################"
if [[ ! -d "${MEGA_CAND}" ]]; then
  echo "[ERROR] missing ${MEGA_CAND}; run full W15 first"
  exit 1
fi
echo "[OK] mega bank k=$(python -c "import json;print(len(json.load(open('${MEGA_CAND}/specs.json'))))")"

# mega brain (best Pix from partial run: ~0.163)
full_metrics outputs/erdc/w15_mega_fuse_l0p10/selected_brain w15_mega_brain
fid_metrics outputs/erdc/w15_mega_fuse_l0p10/selected_brain w15_mega_brain

echo "################ [2] Fuse λ=0.10/0.15/0.20 ################"
run_fuse 0.10
run_fuse 0.15
run_fuse 0.20

echo "################ [3] Top-K struct ################"
for TK in 2 3; do
  TAG="w15_mega_topk${TK}_l${FUSE_LAM/./p}"
  OUT="outputs/erdc/${TAG}"
  if [[ ! -f "${OUT}/metrics.json" ]]; then
    python scripts/erdc_fuse_reselect.py \
      --cand-dir "${MEGA_CAND}" \
      --eeg-npy "${BIT_NPY}" \
      --lambda-struct "${FUSE_LAM}" \
      --top-k-brain "${TK}" \
      --output-dir "${OUT}" \
      "${EXTRA[@]}"
  fi
  full_metrics "${OUT}/selected_topk_struct" "${TAG}"
  fid_metrics "${OUT}/selected_topk_struct" "${TAG}"
done

echo "################ [4] Dual-path fuse ################"
BIT_FUSE="outputs/erdc/w13_merged_fuse_l0p15/selected_fused"
if [[ -d outputs/erdc/w15_mega_fuse_l0p15/selected_fused ]]; then
  BIT_FUSE="outputs/erdc/w15_mega_fuse_l0p15/selected_fused"
fi
if [[ ! -f outputs/erdc/w15_dual_atm_bit_pair/fuse_run/metrics.json ]]; then
  python scripts/erdc_pair_bank_fuse.py \
    --dir-a outputs/erdc/w12_official_atm_img_sub08/selected_brain \
    --dir-b "${BIT_FUSE}" \
    --label-a atm_brain \
    --label-b bit_merged_fuse \
    --eeg-npy "${BIT_NPY}" \
    --lambda-struct "${FUSE_LAM}" \
    --output-dir outputs/erdc/w15_dual_atm_bit_pair \
    "${EXTRA[@]}"
fi
full_metrics outputs/erdc/w15_dual_atm_bit_pair/fuse_run/selected_fused w15_dual_atm_bit_fuse
fid_metrics outputs/erdc/w15_dual_atm_bit_pair/fuse_run/selected_fused w15_dual_atm_bit_fuse

echo "################ [5] References FID + metrics ################"
for tag_gen in \
  "official_atm_gen:outputs/erdc/w7_official_flat" \
  "w12_atm_brain:outputs/erdc/w12_official_atm_img_sub08/selected_brain" \
  "w13_merged_fuse:outputs/erdc/w13_merged_fuse_l0p15/selected_fused" \
  "w10_sdxl_fuse:outputs/erdc/w10_fuse_lowstr_l0p35_sub08/selected_fused"; do
  tag="${tag_gen%%:*}"
  gen="${tag_gen##*:}"
  full_metrics "${gen}" "${tag}"
  fid_metrics "${gen}" "${tag}"
done

echo "################ [6] 2WC + bootstrap ################"
twoway outputs/erdc/w7_official_flat official_atm_gen
twoway outputs/erdc/w13_merged_fuse_l0p15/selected_fused w13_merged_fuse_l0p15
twoway outputs/erdc/w15_mega_fuse_l0p15/selected_fused w15_mega_fuse_l0p15
twoway outputs/erdc/w15_mega_fuse_l0p10/selected_brain w15_mega_brain
twoway outputs/erdc/w15_dual_atm_bit_pair/fuse_run/selected_fused w15_dual_atm_bit_fuse
for TK in 2 3; do
  twoway "outputs/erdc/w15_mega_topk${TK}_l${FUSE_LAM/./p}/selected_topk_struct" "w15_mega_topk${TK}_l${FUSE_LAM/./p}"
done

bootstrap_ci outputs/erdc/w13_merged_fuse_l0p15/selected_fused w13_merged_fuse_l0p15
bootstrap_ci outputs/erdc/w15_mega_fuse_l0p15/selected_fused w15_mega_fuse_l0p15
bootstrap_ci outputs/erdc/w15_mega_fuse_l0p10/selected_brain w15_mega_brain
bootstrap_ci outputs/erdc/w15_dual_atm_bit_pair/fuse_run/selected_fused w15_dual_atm_bit_fuse
bootstrap_ci outputs/erdc/w12_official_atm_img_sub08/selected_brain w12_atm_brain

echo "################ [7] Qualitative panels ################"
python scripts/erdc_qualitative_panel.py \
  --b0-dir outputs/erdc/w7_official_flat \
  --random-dir outputs/erdc/w12_official_bit_img_sub08/selected_random \
  --brain-dir outputs/erdc/w15_mega_fuse_l0p10/selected_brain \
  --scores-npy outputs/erdc/w15_mega_fuse_l0p10/brain_scores.npy \
  --output-dir outputs/erdc/w15_panels_mega_brain \
  --n-panel 24 || true

python scripts/erdc_qualitative_panel.py \
  --b0-dir outputs/erdc/w7_official_flat \
  --random-dir outputs/erdc/w12_official_bit_img_sub08/selected_random \
  --brain-dir outputs/erdc/w13_merged_fuse_l0p15/selected_fused \
  --scores-npy outputs/erdc/w13_merged_fuse_l0p15/fused_scores.npy \
  --output-dir outputs/erdc/w15_panels_main_fuse \
  --n-panel 24 || true

echo "################ [8] Paper table + freeze ################"
python scripts/erdc_paper_table.py \
  --metrics-dirs outputs/erdc/w15_metrics outputs/erdc/w14_metrics outputs/erdc/w12_metrics \
  --tags official_atm_gen w12_atm_brain w13_merged_fuse_l0p15 w15_mega_brain \
         w15_mega_fuse_l0p15 w15_mega_fuse_l0p20 w15_mega_topk2_l0p15 \
         w15_dual_atm_bit_fuse w10_sdxl_fuse \
  --output-md outputs/erdc/paper_main_table.md \
  --output-tex outputs/erdc/paper_main_table.tex

python - <<'PY'
import json
from pathlib import Path

MET = Path("outputs/erdc/w15_metrics")
rows = ["# W15 续跑汇总", ""]
for p in sorted(MET.glob("*.json")):
    if "_fid" in p.name or "_2wc" in p.name or "_bootstrap" in p.name or "test" in p.name:
        continue
    d = json.loads(p.read_text())
    tag = p.stem
    fid_p = MET / f"{tag}_fid.json"
    tw_p = MET / f"{tag}_2wc.json"
    fid = json.loads(fid_p.read_text())["fid"] if fid_p.is_file() else None
    tw = json.loads(tw_p.read_text())["twoway"]["clip"]*100 if tw_p.is_file() else None
    fs = f"{fid:.1f}" if fid else "—"
    ts = f"{tw:.1f}%" if tw else "—"
    rows.append(f"{tag:40s} Pix={d['pixcorr']:.4f} CLIP={d['clip_cosine']:.4f} FID={fs} 2WC={ts}")

Path("outputs/erdc/w15_final_freeze.md").write_text("\n".join(rows)+"\n", encoding="utf-8")
print("\n".join(rows))
print("[OK] w15_final_freeze.md")
PY

echo "[OK] W15 continuation finished"
date
