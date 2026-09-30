#!/usr/bin/env bash
#SBATCH --job-name=nwret-scan
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/nwret-scan-%A_%a.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/nwret-scan-%A_%a.err
#
# NW-Retrieval phase 1: tokenization variants + layer scan, one arm per array task.
#
# Why this is the first thing to run
# ----------------------------------
# Layer depth is reported as the single largest lever in this literature
# (+31.7 intra-subject Top-1, 50.9 -> 82.6), bigger than any backbone choice.
# EEGiT's ablation additionally prices the tokenization interface at +16.4 and
# the pretrained weights at +6.8. Both are cheap to measure and both are
# prerequisites for interpreting anything else, so they run first.
#
# Protocol discipline
# -------------------
#   * 200-way retrieval, train reps 4 / test reps 80 averaged -- identical to
#     third_party/SAMGA so numbers are comparable to the published 91.3/98.8.
#   * checkpoint is chosen on a concept-level holdout of the *training* concepts
#     (150 of 1654). The test split is scored exactly once, at the end.
#     SAMGA selects on test Top-1; we deliberately do not.
#
# Arm selection: pass an array range, e.g.
#   sbatch --array=0-9      nwret_scan.sbatch   # phase A + ViT-B/16 scan
#   sbatch --array=15-18    nwret_scan.sbatch   # DINOv2-L scan
#   sbatch --array=19-22    nwret_scan.sbatch   # CLIP ViT-H-14 scan
set -uo pipefail

ROOT="/project/peilab/why/eeg-retrieval"
cd "${ROOT}"

export PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
export HOME="${XDG_CACHE_HOME}"     # never let a library write to /home

mkdir -p outputs/slurm outputs

# arm index -> arm name (must match ARMS in scripts/run_arm.sh)
ARMS=(A0 A1 A2 A3 A4 B02 B04 B06 B08 B10 B12 C06 C12 C18 C24 D08 D16 D24 D32 E1 E2)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX >= ${#ARMS[@]} )); then
  echo "[FATAL] array index ${IDX} out of range (0..$((${#ARMS[@]}-1)))"
  exit 2
fi
ARM="${ARMS[$IDX]}"

echo "########## nwret arm=${ARM} (idx ${IDX}) job ${SLURM_JOB_ID:-local} @ $(date -Iseconds) ##########"

# Hard CUDA gate: a silent CPU fallback would waste the whole allocation and,
# worse, produce numbers that look like results.
"${PY}" - <<'PY' || { echo "[FATAL] CUDA unavailable"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("[FATAL] torch.cuda.is_available() is False"); sys.exit(1)
print(f"[gpu] {torch.cuda.get_device_name(0)}  torch={torch.__version__}  "
      f"mem={torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GiB")
PY

ARM="${ARM}" bash scripts/run_arm.sh
RC=$?

echo "########## nwret arm=${ARM} done @ $(date -Iseconds) rc=${RC} ##########"
exit "${RC}"
