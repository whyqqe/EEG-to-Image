#!/usr/bin/env bash
#SBATCH --job-name=erdc-w11
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

# ERDC W11 — PixCorr push:
#   A) Zero-regen: merge banks + fuse λ grid + struct mechanism controls
#   B) Dense low-strength RAS (atm embed, finer grid)
#   C) S4 fine-tune → export embeds → RAS + fuse
#   D) Metrics + freeze

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export HUGGINGFACE_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export TRANSFORMERS_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export DIFFUSERS_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/eeg-brainit/open_clip
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
export TMPDIR=/project/peilab/why/cache/eeg-brainit/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XFORMERS_DISABLED=1
mkdir -p "$TMPDIR" outputs/erdc/w11_metrics outputs/slurm

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
print("GPU", torch.cuda.get_device_name(0), flush=True)
PY

MAX_IMAGES="${ERDC_MAX_IMAGES:-0}"
EXTRA=()
[[ "${MAX_IMAGES}" != "0" ]] && EXTRA+=(--max-images "${MAX_IMAGES}")

full_metrics () {
  local gen="$1"; local tag="$2"
  local out="outputs/erdc/w11_metrics/${tag}.json"
  [[ -f "${out}" ]] && echo "[SKIP] ${tag}" && return 0
  [[ -d "${gen}" ]] || { echo "[WARN] miss ${gen}"; return 0; }
  python scripts/erdc_full_metrics.py --gen-dir "${gen}" --output-json "${out}" --tag "${tag}" "${EXTRA[@]}"
}

EEG_ATM="outputs/atm_bridge/sub-08_test_eeg_1024.npy"
EEG_BIT="outputs/eval/atm_pipeline_sub08/sub-08_bit_clip_1024.npy"

echo "################ A) Merge banks + fuse (no regen) ################"
python scripts/erdc_merge_banks.py \
  --bank-dirs outputs/erdc/w8_ras_bit_lowstr_sub08 \
              outputs/erdc/w10_ras_ultralow_bit_sub08 \
  --output-dir outputs/erdc/w11_merged_lowstr_ultralow/candidates

for LAM in 0.25 0.30 0.35 0.40 0.45; do
  echo "----- merged fuse λ=${LAM} -----"
  python scripts/erdc_fuse_reselect.py \
    --cand-dir outputs/erdc/w11_merged_lowstr_ultralow/candidates \
    --eeg-npy "${EEG_BIT}" \
    --lambda-struct "${LAM}" \
    --output-dir "outputs/erdc/w11_merged_fuse_l${LAM/./p}" \
    "${EXTRA[@]}"
  full_metrics "outputs/erdc/w11_merged_fuse_l${LAM/./p}/selected_fused" "w11_merged_fuse_l${LAM/./p}"
done

echo "----- fuse mechanism controls (λ=0.35) -----"
for MODE in zero shuffle misalign; do
  python scripts/erdc_fuse_reselect.py \
    --cand-dir outputs/erdc/w8_ras_bit_lowstr_sub08/candidates \
    --eeg-npy "${EEG_BIT}" \
    --lambda-struct 0.35 \
    --struct-mode "${MODE}" \
    --output-dir "outputs/erdc/w11_fuse_ctrl_${MODE}" \
    "${EXTRA[@]}"
  full_metrics "outputs/erdc/w11_fuse_ctrl_${MODE}/selected_fused" "w11_fuse_ctrl_${MODE}"
done

echo "################ B) Dense low-strength RAS (atm embed) ################"
python scripts/erdc_ras_closed_loop.py \
  --subject sub-08 \
  --embed-source atm \
  --neighbor-mode retrieve \
  --top-m 3 \
  --strengths 0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.22,0.25,0.30 \
  --output-dir outputs/erdc/w11_ras_dense_atm_sub08 \
  --gen-steps 30 --gen-guidance 4.0 --gen-size 512 \
  "${EXTRA[@]}"
full_metrics outputs/erdc/w11_ras_dense_atm_sub08/selected_brain w11_dense_atm_brain

python scripts/erdc_fuse_reselect.py \
  --cand-dir outputs/erdc/w11_ras_dense_atm_sub08/candidates \
  --eeg-npy "${EEG_ATM}" \
  --lambda-struct 0.35 \
  --output-dir outputs/erdc/w11_dense_atm_fuse_l0p35 \
  "${EXTRA[@]}"
full_metrics outputs/erdc/w11_dense_atm_fuse_l0p35/selected_fused w11_dense_atm_fuse

# guidance ablation on ultralow recipe
python scripts/erdc_ras_closed_loop.py \
  --subject sub-08 \
  --embed-source bit_clip \
  --neighbor-mode retrieve \
  --top-m 2 \
  --strengths 0.10,0.15,0.20 \
  --output-dir outputs/erdc/w11_ultralow_g3_bit_sub08 \
  --gen-steps 30 --gen-guidance 3.0 --gen-size 512 \
  "${EXTRA[@]}"
full_metrics outputs/erdc/w11_ultralow_g3_bit_sub08/selected_brain w11_ultralow_g3_brain

echo "################ C) S4 fine-tune + export + RAS ################"
if [[ -f outputs/atm_distill_s4_pix_sub08/checkpoints/atm_stage4_best.pt ]]; then
  S4_CKPT=outputs/atm_distill_s4_pix_sub08/checkpoints/atm_stage4_best.pt
elif [[ -f outputs/atm_distill_s4_pix_sub08/checkpoints/atm_stage4_last.pt ]]; then
  S4_CKPT=outputs/atm_distill_s4_pix_sub08/checkpoints/atm_stage4_last.pt
else
  echo "[RUN] S4 training inline (no pre-trained S4 ckpt)"
  python scripts/train_atm_bridge.py \
    --config configs/base.yaml \
    --override configs/atm_distill_s4_pix_sub08.yaml
  S4_CKPT=outputs/atm_distill_s4_pix_sub08/checkpoints/atm_stage4_best.pt
fi

python scripts/export_atm_bit_clip.py \
  --checkpoint "${S4_CKPT}" \
  --subject sub-08 \
  --output-npy outputs/eval/atm_pipeline_sub08/sub-08_bit_clip_s4_pix_1024.npy

S4_EEG="outputs/eval/atm_pipeline_sub08/sub-08_bit_clip_s4_pix_1024.npy"

python scripts/erdc_ras_closed_loop.py \
  --subject sub-08 \
  --embed-source bit_clip \
  --bit-npy "${S4_EEG}" \
  --neighbor-mode retrieve \
  --top-m 2 \
  --strengths 0.10,0.15,0.20,0.25,0.35 \
  --output-dir outputs/erdc/w11_ras_s4_bit_sub08 \
  --gen-steps 30 --gen-guidance 4.0 --gen-size 512 \
  "${EXTRA[@]}"
full_metrics outputs/erdc/w11_ras_s4_bit_sub08/selected_brain w11_s4_ras_brain

python scripts/erdc_fuse_reselect.py \
  --cand-dir outputs/erdc/w11_ras_s4_bit_sub08/candidates \
  --eeg-npy "${S4_EEG}" \
  --lambda-struct 0.35 \
  --output-dir outputs/erdc/w11_s4_fuse_l0p35 \
  "${EXTRA[@]}"
full_metrics outputs/erdc/w11_s4_fuse_l0p35/selected_fused w11_s4_fuse

echo "################ D) References ################"
full_metrics outputs/erdc/w10_fuse_lowstr_l0p35_sub08/selected_fused w10_best_fuse_ref
full_metrics outputs/erdc/w8_ras_bit_lowstr_sub08/selected_brain w8_lowstr_ref
full_metrics outputs/erdc/w7_official_flat official_atm_gen

echo "################ E) Freeze summary ################"
python - <<'PY'
import json
from pathlib import Path

def load(tag):
    p = Path("outputs/erdc/w11_metrics") / f"{tag}.json"
    if p.is_file():
        return json.loads(p.read_text())
    return None

rows = []
best = None
best_tag = None
for p in sorted(Path("outputs/erdc/w11_metrics").glob("*.json")):
    if "_2wc" in p.name:
        continue
    d = json.loads(p.read_text())
    tag = p.stem
    rows.append(f"{tag:40s} Pix={d['pixcorr']:.4f} CLIP={d['clip_cosine']:.4f}")
    if best is None or d["pixcorr"] > best["pixcorr"] or (
        abs(d["pixcorr"] - best["pixcorr"]) < 1e-4 and d["clip_cosine"] > best["clip_cosine"]
    ):
        best, best_tag = d, tag

off = load("official_atm_gen")
w10 = load("w10_best_fuse_ref")
checks = []
if best and off:
    checks.append(
        f"BEST_W11={best_tag} Pix={best['pixcorr']:.4f} CLIP={best['clip_cosine']:.4f} "
        f"vs_official dPix={best['pixcorr']-off['pixcorr']:+.4f} PASS_PIX={best['pixcorr']>=0.15}"
    )
if best and w10:
    checks.append(f"vs_W10_best dPix={best['pixcorr']-w10['pixcorr']:+.4f} dCLIP={best['clip_cosine']-w10['clip_cosine']:+.4f}")

ctrl = load("w11_fuse_ctrl_shuffle")
main = load("w11_merged_fuse_l0p35") or load("w10_best_fuse_ref")
if ctrl and main:
    checks.append(
        f"FUSE_CTRL shuffle Pix={ctrl['pixcorr']:.4f} vs main={main['pixcorr']:.4f} "
        f"PASS={main['pixcorr']>ctrl['pixcorr']}"
    )

out = Path("outputs/erdc/w11_pix_push_freeze.txt")
out.write_text("\n".join(rows) + "\n\n" + "\n".join(checks) + "\n", encoding="utf-8")
print(out.read_text())
print("[OK]", out)
PY

echo "[OK] ERDC W11 PixCorr push finished"
date
