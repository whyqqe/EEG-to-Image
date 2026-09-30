#!/usr/bin/env bash
#SBATCH --job-name=nwret-dual2
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/nwret-dual2-%A_%a.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/nwret-dual2-%A_%a.err
#
# One array over the two subjects the experiment was asked for: sub-08 and sub-10.
# Each task runs the whole pipeline -- structural target caches, training, condition
# export, four decode arms, seven metrics -- for its own subject. See
# scripts/run_patch_dual.sh for what is fixed and why.
#
# Why an array and not two sbatch calls
# -------------------------------------
# The two subjects share everything that is subject-independent: the semantic CLIP
# feature bank, the VAE latent cache, and the topology of the run. Running them from
# one script with SUBJ as the only difference is what keeps their configurations
# provably identical -- the failure mode of two hand-edited scripts is a comparison
# between two experiments rather than between two subjects.
#
# Prerequisites, and why they are checked rather than assumed
# -----------------------------------------------------------
# `outputs/sub<NN>/probe_targets_all.json` must exist. It is the closed-form ridge
# probe that measures how much of each candidate target space the EEG carries, and
# it is what the structural side of this pipeline is built on: the depth head was
# removed and the VAE head kept because of those numbers, per subject. Re-running
# the pipeline with a stale or absent probe would silently detach the configuration
# from its evidence, so the job refuses rather than proceeding. Build it with
# `sbatch slurm/nwret_probe_targets.sbatch` first (~10 min of one GPU per arm).
#
# The gate that matters at run time is not here but in `export_conds.py`: a
# structural head that has collapsed to the conditional mean aborts the export
# unless `--allow-collapse` is passed. Stage 0 predicts which heads CAN collapse;
# that gate catches the case where one DID. Both are needed, because the probe
# bounds what is reachable in principle and says nothing about whether this
# particular training run reached it.
#
# Cost, per task
# --------------
#   [0] VAE target cache      ~12 min   one-off, shared across subjects, skipped after
#   [1] training, 100 epochs  ~2.5 h    2 towers; the depth head is gone so the
#                                       structural side is smaller than it was
#   [2] condition export      ~10 min   includes the VAE decode for the init images
#   [3] four decode arms      ~45 min   2 img2img + 2 txt2img, 200 images each
#   [4] seven metrics, 4 arms ~25 min
# so ~4 h of GPU against a 6 h wall clock. The margin is deliberate: stage 0 is the
# only stage whose duration is not bounded by a step count, and a requeue resumes
# rather than restarts, because every stage is skipped when its artefact exists.
#
# Memory
# ------
# 128G host / one GPU. The VAE latent cache is memmapped, not read, so host memory
# is not the constraint; the structure tower's activations at batch 128 are what
# needs the room, since freezing blocks 1..20 does not stop autograd from storing
# their activations (the path from the trainable patch_embed runs through every one).
set -uo pipefail

ROOT="/project/peilab/why/eeg-retrieval"
cd "${ROOT}"

export PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# array index -> subject
SUBJECTS=(8 10)
i="${SLURM_ARRAY_TASK_ID:-0}"
export SUBJ="${SUBJECTS[$i]}"

echo "########## job ${SLURM_JOB_ID:-local}.${i} sub-${SUBJ} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds) ##########"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

mkdir -p outputs/slurm outputs/logs

# ---- the stage-0 evidence must be on disk, for THIS subject ------------------
PROBE="outputs/sub$(printf '%02d' "${SUBJ}")/probe_targets_all.json"
if [[ ! -f "${PROBE}" ]]; then
  echo "[FATAL] no stage-0 probe at ${PROBE}." >&2
  echo "        The structural configuration is derived from those numbers, so" >&2
  echo "        running without them would detach the run from its evidence." >&2
  echo "        Build it first:  sbatch slurm/nwret_probe_targets.sbatch" >&2
  exit 2
fi
"${PYTHON}" - <<PY
import json, sys
d = json.loads(open("${PROBE}").read())
by = {r["target"]: r for r in d["results"]}
vae, dep = by.get("vae"), by.get("depth")
print(f"[gate ] stage-0 probe for sub-{d['subject']:02d} "
      f"({d['n_channels']}ch, {d['n_fit_concepts']} fit concepts)")
if vae:
    ok = vae.get("val_r_margin", -1) > 0.02 and vae.get("val_pred_var_ratio", 0) > 0.20
    print(f"[gate ] vae   : val@1 {vae['val_top1']:.2f}  margin "
          f"{vae['val_r_margin']:+.4f}  var ratio {vae['val_pred_var_ratio']:.4f}  "
          f"-> {'above floor, head is justified' if ok else 'AT OR BELOW FLOOR'}")
if dep:
    print(f"[gate ] depth : margin {dep['val_r_margin']:+.4f} "
          f"-> below floor, head removed from the pipeline")
dino = by.get("dino_l")
ref = by.get("clip_block26")
if dino and ref:
    print(f"[gate ] dino_l: val@1 {dino['val_top1']:.2f} vs clip_block26 "
          f"{ref['val_top1']:.2f} -> NOT used as the structural target")
PY

# Model weights live on /project, NOT on /home: /home is at 100% capacity and a
# download attempt there fails with ENOSPC mid-run. HF_HUB_OFFLINE is left unset so
# a genuinely missing artefact surfaces as a download error rather than as a
# confusing cache miss; every artefact this job needs is already cached.
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"

bash scripts/run_patch_dual.sh
RC=$?

echo "########## job ${SLURM_JOB_ID:-local}.${i} sub-${SUBJ} rc=${RC} @ $(date -Iseconds) ##########"
exit "${RC}"
