#!/usr/bin/env bash
#SBATCH --job-name=erdc-w7
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

# ERDC W7 full pack — everything runnable now without new training:
#   A) Full ATM-style metrics on existing W1/W6 outputs + official ATM gen
#   B) bit_clip closed-loop K=8 (sub-08)
#   C) bit_clip TTS scale-search bank K=8 denser scales (sub-08)
#   D) atm closed-loop K=8 multi-subject (sub-08/01/02)
#   E) Full metrics on all new selected_* folders
#   F) Qualitative panels
# Summary -> outputs/erdc/w7_full_pack_summary.txt

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
mkdir -p "$TMPDIR" "$HF_HUB_CACHE/.locks" outputs/erdc/w7_metrics outputs/slurm

python - <<'PY'
import torch, torchvision
assert torch.cuda.is_available(), "CUDA unavailable — exclude/requeue"
from diffusers import StableDiffusionXLPipeline
print("GPU", torch.cuda.get_device_name(0), "tv", torchvision.__version__, "SDXL OK", flush=True)
PY

MAX_IMAGES="${ERDC_MAX_IMAGES:-0}"
EXTRA=()
if [[ "${MAX_IMAGES}" != "0" ]]; then
  EXTRA+=(--max-images "${MAX_IMAGES}")
fi

full_metrics () {
  local gen="$1"
  local tag="$2"
  local out="outputs/erdc/w7_metrics/${tag}.json"
  if [[ -f "${out}" ]]; then
    echo "[SKIP] metrics ${tag} exists"
    return 0
  fi
  if [[ ! -d "${gen}" ]]; then
    echo "[WARN] missing gen dir ${gen}; skip ${tag}"
    return 0
  fi
  python scripts/erdc_full_metrics.py \
    --gen-dir "${gen}" \
    --output-json "${out}" \
    --tag "${tag}" \
    "${EXTRA[@]}"
}

echo "################ A) Full metrics on existing artifacts ################"
full_metrics outputs/erdc/w1_b0_atm_sub08/generated              w1_b0_atm
full_metrics outputs/erdc/w6_b0_bit_sub08/generated              w6_b0_bit
full_metrics outputs/erdc/w6_loop_bit_sub08/selected_first       w6_bit_k4_first
full_metrics outputs/erdc/w6_loop_bit_sub08/selected_random      w6_bit_k4_random
full_metrics outputs/erdc/w6_loop_bit_sub08/selected_brain       w6_bit_k4_brain
full_metrics outputs/erdc/w6_loop_bit_sub08/selected_oracle      w6_bit_k4_oracle
full_metrics outputs/erdc/w6_loop_atm_sub08/selected_brain       w6_atm_k4_brain
full_metrics outputs/erdc/w1_teacher_ceiling_sub08/generated     w1_teacher
# official ATM extracted gens if present
OFF_ROOT=outputs/eval/atm_official_gen_sub08/extracted/generated_imgs/sub-08
if [[ -d "${OFF_ROOT}" ]]; then
  mkdir -p outputs/erdc/w7_official_flat
  python - <<'PY'
from pathlib import Path
import shutil
import sys
sys.path.insert(0, "scripts")
from eval_atm_pipeline import list_test_images

gt = list_test_images(Path("/project/peilab/why/data/images_set"))
src = Path("outputs/eval/atm_official_gen_sub08/extracted/generated_imgs/sub-08")
dst = Path("outputs/erdc/w7_official_flat")
dst.mkdir(parents=True, exist_ok=True)
n = 0
for i, gp in enumerate(gt):
    # GT parent: 00001_aircraft_carrier -> class key aircraft_carrier
    folder = gp.parent.name
    key = folder.split("_", 1)[1] if "_" in folder and folder.split("_", 1)[0].isdigit() else folder
    cands = sorted((src / key).glob("*.png"))
    if not cands:
        # fallback: any dir ending with key
        hits = [d for d in src.iterdir() if d.is_dir() and (d.name == key or d.name.endswith("_" + key))]
        cands = sorted(hits[0].glob("*.png")) if hits else []
    if not cands:
        raise FileNotFoundError(f"no official gen for {folder} key={key}")
    shutil.copy2(cands[0], dst / f"{i:03d}.png")
    n += 1
print("flattened official", n)
PY
  full_metrics outputs/erdc/w7_official_flat official_atm_gen
fi

echo "################ B) bit_clip closed-loop K=8 ################"
python scripts/erdc_closed_loop.py \
  --subject sub-08 \
  --embed-source bit_clip \
  --k 8 \
  --scale-grid 0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.0 \
  --seed0 42 \
  --output-dir outputs/erdc/w7_loop_bit_k8_sub08 \
  --gen-steps 30 \
  --gen-guidance 5.0 \
  --gen-size 512 \
  "${EXTRA[@]}"

echo "################ C) bit_clip TTS denser scale bank K=8 ################"
# Same K but seed-fixed scale sweep emphasis (different seed0 to diversify vs B)
python scripts/erdc_closed_loop.py \
  --subject sub-08 \
  --embed-source bit_clip \
  --k 8 \
  --scale-grid 0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.2 \
  --seed0 7 \
  --output-dir outputs/erdc/w7_tts_bit_k8_sub08 \
  --gen-steps 30 \
  --gen-guidance 5.0 \
  --gen-size 512 \
  "${EXTRA[@]}"

echo "################ D) atm closed-loop K=8 multi-subject ################"
for SUB in sub-08 sub-01 sub-02; do
  echo "----- atm K=8 ${SUB} -----"
  test -f "outputs/atm_bridge/${SUB}_test_eeg_1024.npy"
  python scripts/erdc_closed_loop.py \
    --subject "${SUB}" \
    --embed-source atm \
    --k 8 \
    --scale-grid 0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.0 \
    --seed0 42 \
    --output-dir "outputs/erdc/w7_loop_atm_k8_${SUB}" \
    --gen-steps 30 \
    --gen-guidance 5.0 \
    --gen-size 512 \
    "${EXTRA[@]}"
done

echo "################ E) Full metrics on new selections ################"
full_metrics outputs/erdc/w7_loop_bit_k8_sub08/selected_first    w7_bit_k8_first
full_metrics outputs/erdc/w7_loop_bit_k8_sub08/selected_random   w7_bit_k8_random
full_metrics outputs/erdc/w7_loop_bit_k8_sub08/selected_brain    w7_bit_k8_brain
full_metrics outputs/erdc/w7_loop_bit_k8_sub08/selected_oracle   w7_bit_k8_oracle
full_metrics outputs/erdc/w7_tts_bit_k8_sub08/selected_first     w7_tts_bit_first
full_metrics outputs/erdc/w7_tts_bit_k8_sub08/selected_random    w7_tts_bit_random
full_metrics outputs/erdc/w7_tts_bit_k8_sub08/selected_brain     w7_tts_bit_brain
full_metrics outputs/erdc/w7_tts_bit_k8_sub08/selected_oracle    w7_tts_bit_oracle
for SUB in sub-08 sub-01 sub-02; do
  full_metrics "outputs/erdc/w7_loop_atm_k8_${SUB}/selected_first"  "w7_atm_k8_${SUB}_first"
  full_metrics "outputs/erdc/w7_loop_atm_k8_${SUB}/selected_random" "w7_atm_k8_${SUB}_random"
  full_metrics "outputs/erdc/w7_loop_atm_k8_${SUB}/selected_brain"  "w7_atm_k8_${SUB}_brain"
done

echo "################ F) Qualitative panels ################"
python scripts/erdc_qualitative_panel.py \
  --b0-dir outputs/erdc/w6_b0_bit_sub08/generated \
  --random-dir outputs/erdc/w7_loop_bit_k8_sub08/selected_random \
  --brain-dir outputs/erdc/w7_loop_bit_k8_sub08/selected_brain \
  --scores-npy outputs/erdc/w7_loop_bit_k8_sub08/candidates/brain_scores.npy \
  --output-dir outputs/erdc/w7_panels_bit_k8 \
  --n-panel 24

python scripts/erdc_qualitative_panel.py \
  --b0-dir outputs/erdc/w1_b0_atm_sub08/generated \
  --random-dir outputs/erdc/w7_loop_atm_k8_sub08/selected_random \
  --brain-dir outputs/erdc/w7_loop_atm_k8_sub08/selected_brain \
  --scores-npy outputs/erdc/w7_loop_atm_k8_sub08/candidates/brain_scores.npy \
  --output-dir outputs/erdc/w7_panels_atm_k8 \
  --n-panel 24

echo "################ Summary ################"
python - <<'PY'
import json
from pathlib import Path

def load(tag):
    p = Path(f"outputs/erdc/w7_metrics/{tag}.json")
    if not p.is_file():
        return None
    return json.loads(p.read_text())

def line(tag):
    d = load(tag)
    if d is None:
        return f"{tag}: MISSING"
    return (
        f"{tag:28s} "
        f"Pix={d['pixcorr']:.4f} SSIM={d['ssim']:.4f} CLIP={d['clip_cosine']:.4f} "
        f"A2={d['alexnet2']:.4f} A5={d['alexnet5']:.4f} "
        f"Inc={d['inception']:.4f} Eff={d['effnet_b1']:.4f}"
    )

tags = [
    "w1_b0_atm", "w6_b0_bit",
    "w6_bit_k4_first", "w6_bit_k4_random", "w6_bit_k4_brain", "w6_bit_k4_oracle",
    "w1_teacher", "official_atm_gen",
    "w7_bit_k8_first", "w7_bit_k8_random", "w7_bit_k8_brain", "w7_bit_k8_oracle",
    "w7_tts_bit_first", "w7_tts_bit_random", "w7_tts_bit_brain", "w7_tts_bit_oracle",
    "w7_atm_k8_sub-08_first", "w7_atm_k8_sub-08_random", "w7_atm_k8_sub-08_brain",
    "w7_atm_k8_sub-01_first", "w7_atm_k8_sub-01_random", "w7_atm_k8_sub-01_brain",
    "w7_atm_k8_sub-02_first", "w7_atm_k8_sub-02_random", "w7_atm_k8_sub-02_brain",
]
rows = [line(t) for t in tags]

# pass checks
checks = []
def clip(tag):
    d = load(tag)
    return None if d is None else d["clip_cosine"]

b = clip("w7_bit_k8_brain"); f = clip("w7_bit_k8_first"); r = clip("w7_bit_k8_random")
if b is not None:
    checks.append(f"PASS_bit_k8_brain_gt_first_random={b>f and b>r} brain={b:.4f} first={f:.4f} rand={r:.4f}")
for sub in ["sub-08", "sub-01", "sub-02"]:
    b = clip(f"w7_atm_k8_{sub}_brain"); f = clip(f"w7_atm_k8_{sub}_first"); r = clip(f"w7_atm_k8_{sub}_random")
    if b is not None:
        checks.append(f"PASS_atm_k8_{sub}={b>f and b>r} brain={b:.4f} first={f:.4f} rand={r:.4f}")
tb = clip("w7_tts_bit_brain"); tk4 = clip("w6_bit_k4_brain"); tk8 = clip("w7_bit_k8_brain")
if tb is not None and tk8 is not None:
    checks.append(f"TTS_vs_K8 dCLIP={tb-tk8:+.4f} TTS={tb:.4f} K8={tk8:.4f}")
if tk8 is not None and tk4 is not None:
    checks.append(f"K8_vs_K4 dCLIP={tk8-tk4:+.4f} K8={tk8:.4f} K4={tk4:.4f}")

out = Path("outputs/erdc/w7_full_pack_summary.txt")
out.write_text("\n".join(rows + [""] + checks) + "\n", encoding="utf-8")
print(out.read_text())
print("[OK]", out)
PY

echo "[OK] ERDC W7 full pack finished"
