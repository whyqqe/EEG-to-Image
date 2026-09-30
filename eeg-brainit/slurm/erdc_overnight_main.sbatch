#!/usr/bin/env bash
#SBATCH --job-name=erdc-night
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

# Overnight paper-main pack (starts after W8 via dependency when submitted).
# A) Finish any missing W8 full-metrics / summary
# B) Low-strength RAS (PixCorr push) on bit sub-08
# C) RAS atm for remaining subjects sub-03..10 (10-subject main table)
# D) Full metrics + ENIGMA-style 2WC on key conditions
# E) Aggregate paper freeze table + panels
# F) Note ControlNet download (parallel cpu job)

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
mkdir -p "$TMPDIR" outputs/erdc/overnight_metrics outputs/erdc/w8_metrics outputs/slurm

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable — requeue"
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
print("GPU", torch.cuda.get_device_name(0), flush=True)
PY

full_metrics () {
  local gen="$1"; local tag="$2"; local outdir="$3"
  local out="${outdir}/${tag}.json"
  if [[ -f "${out}" ]]; then echo "[SKIP] metrics ${tag}"; return 0; fi
  if [[ ! -d "${gen}" ]]; then echo "[WARN] miss ${gen}"; return 0; fi
  python scripts/erdc_full_metrics.py --gen-dir "${gen}" --output-json "${out}" --tag "${tag}"
}

twoway () {
  local gen="$1"; local tag="$2"; local outdir="$3"
  local out="${outdir}/${tag}_2wc.json"
  if [[ -f "${out}" ]]; then echo "[SKIP] 2wc ${tag}"; return 0; fi
  if [[ ! -d "${gen}" ]]; then echo "[WARN] miss ${gen}"; return 0; fi
  python scripts/erdc_twoway_metrics.py --gen-dir "${gen}" --output-json "${out}" --tag "${tag}"
}

echo "################ A) Complete W8 metrics if needed ################"
# W8 may have finished gens; ensure metrics/summary exist
if [[ -d outputs/erdc/w8_ras_bit_sub08/selected_brain ]]; then
  full_metrics outputs/erdc/w8_ras_bit_sub08/selected_brain  w8_ras_bit_brain  outputs/erdc/w8_metrics
  full_metrics outputs/erdc/w8_ras_bit_sub08/selected_random w8_ras_bit_random outputs/erdc/w8_metrics
  full_metrics outputs/erdc/w8_ras_bit_sub08/selected_first  w8_ras_bit_first  outputs/erdc/w8_metrics
  full_metrics outputs/erdc/w8_ras_bit_sub08/selected_oracle w8_ras_bit_oracle outputs/erdc/w8_metrics
fi
if [[ -d outputs/erdc/w8_ras_bit_shuffle_sub08/selected_brain ]]; then
  full_metrics outputs/erdc/w8_ras_bit_shuffle_sub08/selected_brain w8_ras_bit_shuffle_brain outputs/erdc/w8_metrics
fi
for SUB in sub-08 sub-01 sub-02; do
  if [[ -d "outputs/erdc/w8_ras_atm_${SUB}/selected_brain" ]]; then
    full_metrics "outputs/erdc/w8_ras_atm_${SUB}/selected_brain"  "w8_ras_atm_${SUB}_brain"  outputs/erdc/w8_metrics
    full_metrics "outputs/erdc/w8_ras_atm_${SUB}/selected_random" "w8_ras_atm_${SUB}_random" outputs/erdc/w8_metrics
  fi
done
# also keep official/w7 refs
full_metrics outputs/erdc/w7_loop_bit_k8_sub08/selected_brain w7_bit_k8_brain outputs/erdc/w8_metrics || true
full_metrics outputs/erdc/w7_official_flat official_atm_gen outputs/erdc/w8_metrics || true

echo "################ B) Low-strength RAS (PixCorr push) ################"
python scripts/erdc_ras_closed_loop.py \
  --subject sub-08 \
  --embed-source bit_clip \
  --top-m 2 \
  --strengths 0.25,0.35,0.50 \
  --output-dir outputs/erdc/w8_ras_bit_lowstr_sub08 \
  --gen-steps 30 --gen-guidance 5.0 --gen-size 512

full_metrics outputs/erdc/w8_ras_bit_lowstr_sub08/selected_brain  w8_ras_bit_lowstr_brain  outputs/erdc/overnight_metrics
full_metrics outputs/erdc/w8_ras_bit_lowstr_sub08/selected_random w8_ras_bit_lowstr_random outputs/erdc/overnight_metrics
full_metrics outputs/erdc/w8_ras_bit_lowstr_sub08/selected_first  w8_ras_bit_lowstr_first  outputs/erdc/overnight_metrics

echo "################ C) RAS atm multi-subject (remaining) ################"
for SUB in sub-03 sub-04 sub-05 sub-06 sub-07 sub-09 sub-10; do
  echo "----- overnight RAS atm ${SUB} -----"
  test -f "outputs/atm_bridge/${SUB}_test_eeg_1024.npy"
  python scripts/erdc_ras_closed_loop.py \
    --subject "${SUB}" \
    --embed-source atm \
    --top-m 2 \
    --strengths 0.45,0.60,0.75 \
    --output-dir "outputs/erdc/overnight_ras_atm_${SUB}" \
    --gen-steps 30 --gen-guidance 5.0 --gen-size 512
  full_metrics "outputs/erdc/overnight_ras_atm_${SUB}/selected_brain"  "overnight_ras_atm_${SUB}_brain"  outputs/erdc/overnight_metrics
  full_metrics "outputs/erdc/overnight_ras_atm_${SUB}/selected_random" "overnight_ras_atm_${SUB}_random" outputs/erdc/overnight_metrics
done

# Symlink/copy W8 subjects into overnight naming for aggregation mean
for SUB in sub-08 sub-01 sub-02; do
  if [[ -f "outputs/erdc/w8_metrics/w8_ras_atm_${SUB}_brain.json" ]]; then
    cp -f "outputs/erdc/w8_metrics/w8_ras_atm_${SUB}_brain.json" \
      "outputs/erdc/overnight_metrics/overnight_ras_atm_${SUB}_brain.json"
    cp -f "outputs/erdc/w8_metrics/w8_ras_atm_${SUB}_random.json" \
      "outputs/erdc/overnight_metrics/overnight_ras_atm_${SUB}_random.json" 2>/dev/null || true
  fi
done
# also copy lowstr tag expected by aggregator from overnight path already

echo "################ D) ENIGMA-style 2WC on paper-critical gens ################"
twoway outputs/erdc/w8_ras_bit_sub08/selected_brain           w8_ras_bit_brain           outputs/erdc/overnight_metrics
twoway outputs/erdc/w8_ras_bit_sub08/selected_random          w8_ras_bit_random          outputs/erdc/overnight_metrics
twoway outputs/erdc/w7_loop_bit_k8_sub08/selected_brain       w7_bit_k8_brain            outputs/erdc/overnight_metrics
twoway outputs/erdc/w7_official_flat                          official_atm_gen           outputs/erdc/overnight_metrics
twoway outputs/erdc/w8_ras_bit_lowstr_sub08/selected_brain    w8_ras_bit_lowstr_brain    outputs/erdc/overnight_metrics
twoway outputs/erdc/w1_b0_atm_sub08/generated                 w1_b0_atm                  outputs/erdc/overnight_metrics
# one more subject for multi-subj 2WC spot-check
if [[ -d outputs/erdc/overnight_ras_atm_sub-03/selected_brain ]]; then
  twoway outputs/erdc/overnight_ras_atm_sub-03/selected_brain overnight_ras_atm_sub-03_brain outputs/erdc/overnight_metrics
fi

echo "################ E) Panels + aggregate ################"
if [[ -d outputs/erdc/w8_ras_bit_sub08/selected_brain ]]; then
  python scripts/erdc_qualitative_panel.py \
    --b0-dir outputs/erdc/w6_b0_bit_sub08/generated \
    --random-dir outputs/erdc/w8_ras_bit_sub08/selected_random \
    --brain-dir outputs/erdc/w8_ras_bit_sub08/selected_brain \
    --scores-npy outputs/erdc/w8_ras_bit_sub08/candidates/brain_scores.npy \
    --output-dir outputs/erdc/overnight_panels_ras_bit \
    --n-panel 24 || true
fi
if [[ -d outputs/erdc/w8_ras_bit_lowstr_sub08/selected_brain ]]; then
  python scripts/erdc_qualitative_panel.py \
    --b0-dir outputs/erdc/w6_b0_bit_sub08/generated \
    --random-dir outputs/erdc/w8_ras_bit_lowstr_sub08/selected_random \
    --brain-dir outputs/erdc/w8_ras_bit_lowstr_sub08/selected_brain \
    --scores-npy outputs/erdc/w8_ras_bit_lowstr_sub08/candidates/brain_scores.npy \
    --output-dir outputs/erdc/overnight_panels_ras_lowstr \
    --n-panel 16 || true
fi

# ensure aggregator sees w8_ras_bit_* from w8_metrics
mkdir -p outputs/erdc/overnight_metrics
for t in w8_ras_bit_brain w8_ras_bit_random w8_ras_bit_shuffle_brain w8_ras_bit_lowstr_brain \
         w7_bit_k8_brain official_atm_gen w1_b0_atm w6_b0_bit; do
  if [[ -f "outputs/erdc/w8_metrics/${t}.json" && ! -f "outputs/erdc/overnight_metrics/${t}.json" ]]; then
    cp -f "outputs/erdc/w8_metrics/${t}.json" "outputs/erdc/overnight_metrics/${t}.json"
  fi
  if [[ -f "outputs/erdc/w7_metrics/${t}.json" && ! -f "outputs/erdc/overnight_metrics/${t}.json" ]]; then
    cp -f "outputs/erdc/w7_metrics/${t}.json" "outputs/erdc/overnight_metrics/${t}.json"
  fi
done

python scripts/erdc_aggregate_main_table.py

echo "################ F) ControlNet readiness note ################"
if [[ -f outputs/erdc/controlnet_canny_ready.txt ]]; then
  echo "[OK] ControlNet already downloaded:"
  cat outputs/erdc/controlnet_canny_ready.txt
else
  echo "[INFO] ControlNet download is a parallel CPU job (erdc-cn-dl); not blocking night pack."
fi

echo "[OK] ERDC overnight paper-main pack finished"
date
