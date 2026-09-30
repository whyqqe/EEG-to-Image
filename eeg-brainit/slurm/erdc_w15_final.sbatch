#!/usr/bin/env bash
#SBATCH --job-name=erdc-w15
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --requeue
#SBATCH --exclude=dgx-10,dgx-11
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# W15 — remaining optimization + FID + bootstrap + dual-path + freeze
#  A) Expanded bit Turbo bank
#  B) Mega merge + fuse λ + top-K struct
#  C) Dual-path pair-bank fuse (atm brain + bit fuse)
#  D) FID / full metrics / 2WC / bootstrap CI
#  E) Freeze summary

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
pip install -q torch-fidelity

MAX_IMAGES="${ERDC_MAX_IMAGES:-0}"
EXTRA=()
[[ "${MAX_IMAGES}" != "0" ]] && EXTRA+=(--max-images "${MAX_IMAGES}")

BIT_NPY="outputs/eval/atm_pipeline_sub08/sub-08_bit_clip_1024.npy"
MET_DIR="outputs/erdc/w15_metrics"
FUSE_LAM="0.15"

full_metrics () {
  local gen="$1"; local tag="$2"
  local out="${MET_DIR}/${tag}.json"
  [[ -f "${out}" ]] && echo "[SKIP] ${tag}" && return 0
  [[ -d "${gen}" ]] || return 0
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

echo "################ [A] Expanded bit Turbo bank ################"
python scripts/erdc_official_atm_pipeline.py \
  --subject sub-08 \
  --embed-source bit_clip \
  --bit-npy "${BIT_NPY}" \
  --low-level-mode neighbor_image \
  --top-m 2 \
  --strengths 0.20,0.25,0.30,0.35,0.50,0.65,0.85 \
  --use-turbo --gen-steps 4 --gen-guidance 0.0 \
  --output-dir outputs/erdc/w15_bit_expanded_sub08 \
  "${EXTRA[@]}"

echo "################ [B] Mega merge + fuse ################"
python scripts/erdc_merge_banks.py \
  --bank-dirs outputs/erdc/w15_bit_expanded_sub08 \
              outputs/erdc/w12_official_atm_img_sub08 \
              outputs/erdc/w12_official_prior_latent_sub08 \
  --output-dir outputs/erdc/w15_mega_turbo/candidates \
  "${EXTRA[@]}"

for LAM in 0.10 0.15 0.20; do
  TAG="w15_mega_fuse_l${LAM/./p}"
  OUT="outputs/erdc/${TAG}"
  python scripts/erdc_fuse_reselect.py \
    --cand-dir outputs/erdc/w15_mega_turbo/candidates \
    --eeg-npy "${BIT_NPY}" \
    --lambda-struct "${LAM}" \
    --output-dir "${OUT}" \
    "${EXTRA[@]}"
  full_metrics "${OUT}/selected_fused" "${TAG}"
  fid_metrics "${OUT}/selected_fused" "${TAG}"
done

for TK in 2 3; do
  TAG="w15_mega_topk${TK}_l${FUSE_LAM/./p}"
  OUT="outputs/erdc/${TAG}"
  python scripts/erdc_fuse_reselect.py \
    --cand-dir outputs/erdc/w15_mega_turbo/candidates \
    --eeg-npy "${BIT_NPY}" \
    --lambda-struct "${FUSE_LAM}" \
    --top-k-brain "${TK}" \
    --output-dir "${OUT}" \
    "${EXTRA[@]}"
  full_metrics "${OUT}/selected_topk_struct" "${TAG}"
  fid_metrics "${OUT}/selected_topk_struct" "${TAG}"
done

echo "################ [C] Dual-path pair fuse ################"
python scripts/erdc_pair_bank_fuse.py \
  --dir-a outputs/erdc/w12_official_atm_img_sub08/selected_brain \
  --dir-b outputs/erdc/w13_merged_fuse_l0p15/selected_fused \
  --label-a atm_brain \
  --label-b bit_merged_fuse \
  --eeg-npy "${BIT_NPY}" \
  --lambda-struct "${FUSE_LAM}" \
  --output-dir outputs/erdc/w15_dual_atm_bit_pair \
  "${EXTRA[@]}"
full_metrics outputs/erdc/w15_dual_atm_bit_pair/fuse_run/selected_fused w15_dual_atm_bit_fuse
fid_metrics outputs/erdc/w15_dual_atm_bit_pair/fuse_run/selected_fused w15_dual_atm_bit_fuse

echo "################ [D] FID + metrics on references ################"
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

echo "################ [E] 2WC + bootstrap ################"
twoway outputs/erdc/w7_official_flat official_atm_gen
twoway outputs/erdc/w13_merged_fuse_l0p15/selected_fused w13_merged_fuse_l0p15
twoway outputs/erdc/w15_mega_fuse_l0p15/selected_fused w15_mega_fuse_l0p15
twoway outputs/erdc/w15_dual_atm_bit_pair/fuse_run/selected_fused w15_dual_atm_bit_fuse
for TK in 2 3; do
  twoway "outputs/erdc/w15_mega_topk${TK}_l${FUSE_LAM/./p}/selected_topk_struct" "w15_mega_topk${TK}_l${FUSE_LAM/./p}"
done

bootstrap_ci outputs/erdc/w13_merged_fuse_l0p15/selected_fused w13_merged_fuse_l0p15
bootstrap_ci outputs/erdc/w15_mega_fuse_l0p15/selected_fused w15_mega_fuse_l0p15
bootstrap_ci outputs/erdc/w15_dual_atm_bit_pair/fuse_run/selected_fused w15_dual_atm_bit_fuse
bootstrap_ci outputs/erdc/w12_official_atm_img_sub08/selected_brain w12_atm_brain

echo "################ [F] Freeze W15 ################"
python - <<'PY'
import json
from pathlib import Path

MET = Path("outputs/erdc/w15_metrics")
W14 = Path("outputs/erdc/w14_metrics")
rows = ["# W15 实验汇总", "", "## 主结果（含 FID）", ""]
rows.append("| 方法 | Pix | CLIP | FID | 2WC |")
rows.append("|------|-----|------|-----|-----|")

def load(tag):
    p = MET / f"{tag}.json"
    if not p.is_file():
        p = W14 / f"{tag}.json"
    return json.loads(p.read_text()) if p.is_file() else None

def load_fid(tag):
    p = MET / f"{tag}_fid.json"
    return json.loads(p.read_text()) if p.is_file() else None

def load_2wc(tag):
    p = MET / f"{tag}_2wc.json"
    if p.is_file():
        return json.loads(p.read_text())["twoway"]["clip"] * 100
    p = W14 / f"{tag}_2wc.json"
    if p.is_file():
        return json.loads(p.read_text())["twoway"]["clip"] * 100
    return None

tags = [
    "official_atm_gen", "w12_atm_brain", "w13_merged_fuse_l0p15",
    "w15_mega_fuse_l0p10", "w15_mega_fuse_l0p15", "w15_mega_fuse_l0p20",
    "w15_mega_topk2_l0p15", "w15_mega_topk3_l0p15", "w15_dual_atm_bit_fuse",
    "w10_sdxl_fuse",
]
best_pix, best_tag = None, None
for tag in tags:
    d = load(tag)
    if not d:
        continue
    fid = load_fid(tag)
    tw = load_2wc(tag)
    fid_s = f"{fid['fid']:.1f}" if fid else "—"
    tw_s = f"{tw:.1f}%" if tw else "—"
    rows.append(f"| {tag} | {d['pixcorr']:.4f} | {d['clip_cosine']:.4f} | {fid_s} | {tw_s} |")
    if best_pix is None or d["pixcorr"] > best_pix:
        best_pix, best_tag = d["pixcorr"], tag

rows.append("")
rows.append("## Bootstrap 95% CI")
for tag in ["w13_merged_fuse_l0p15", "w15_mega_fuse_l0p15", "w15_dual_atm_bit_fuse"]:
    p = MET / f"{tag}_bootstrap.json"
    if p.is_file():
        b = json.loads(p.read_text())
        px, cx = b["pixcorr"], b["clip_cosine"]
        rows.append(
            f"- **{tag}**: Pix {px['mean']:.4f} [{px['ci95_lo']:.4f},{px['ci95_hi']:.4f}] | "
            f"CLIP {cx['mean']:.4f} [{cx['ci95_lo']:.4f},{cx['ci95_hi']:.4f}]"
        )

if best_tag:
    rows.append("")
    rows.append(f"**Best Pix**: {best_tag} = {best_pix:.4f}")

out = Path("outputs/erdc/w15_final_freeze.md")
out.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(out.read_text())
print("[OK]", out)
PY

echo "[OK] W15 finished"
date
