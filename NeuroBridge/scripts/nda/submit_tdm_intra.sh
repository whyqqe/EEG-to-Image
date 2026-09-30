#!/usr/bin/env bash
# Submit the single TDM-DT overnights job for sub-08. One submission only.
set -euo pipefail
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
mkdir -p outputs/slurm outputs/tdm_intra/logs

echo "===== preflight ====="
fail=0
chk() { if [[ -e "$1" ]]; then echo "  [ok]   $1"; else echo "  [MISS] $1"; fail=1; fi; }

chk scripts/nmb/nmb_sota_v2_env.sh
chk scripts/nda/run_tdm_intra.sh
chk scripts/nda/tdm_train.py
chk scripts/nda/tdm_clip_patch.py
chk scripts/nda/tdm_gate0.py
chk scripts/nda/ocf_export_intra_z.py
chk scripts/nda/idg_frla_decode.py
chk scripts/nda/nda_ss_pretrain.py
chk scripts/nda/train_eeg_vae_head.py
chk scripts/nda/generate_atm_aligned_decode.py
chk scripts/nda/eval_official_seven_dir.py
chk slurm/tdm_intra.sbatch
chk outputs/g2/targets/sem_concept_tmpl_train.npy
chk outputs/g2/targets/sem_concept_tmpl_test.npy
chk outputs/g2/targets/perc_struct_train.npy
chk outputs/sdedit_ll_full10/shared/vae_cache/train_vae_latents_f16.npy
chk outputs/sdedit_ll_full10/shared/vae_cache/test_vae_latents_f16.npy
chk outputs/nda_ss/sub-08/clip_text/train/concept_phrases.json
chk outputs/g2/captions/captions_train.jsonl
chk outputs/leakfree/split.json
chk outputs/ocf/intra_enc/checkpoint_ss_calib_best.pth
chk outputs/ocf/intra_z/sub-08/shared_r_train.npy
chk outputs/ocf/intra_z/sub-08/shared_r_test.npy
# reference rows that will be re-evaluated in the same pass
chk outputs/sdedit_ll_full10/sub-08/generation/sdedit_ll/generated/199.png

echo "===== syntax/consistency ====="
python3 -c "import ast
for f in ('scripts/nda/tdm_train.py','scripts/nda/tdm_clip_patch.py','scripts/nda/tdm_gate0.py'):
    ast.parse(open(f).read()); print('  [ok] syntax', f)"
bash -n scripts/nda/run_tdm_intra.sh && echo "  [ok] bash -n run_tdm_intra.sh"

echo "===== interpreter/import check under the project venv ====="
source scripts/nmb/nmb_sota_v2_env.sh >/dev/null 2>&1 || true
python - <<'PY'
import importlib
for m in ("torch", "diffusers", "transformers", "open_clip", "skimage", "scipy"):
    try:
        mod = importlib.import_module(m)
        print(f"  [ok] {m} {getattr(mod,'__version__','?')}")
    except Exception as e:
        raise SystemExit(f"  [FATAL] {m} import failed under the project venv: {e}")
import torch
if not torch.cuda.is_available():
    print("  [warn] no GPU on this login node (expected); the guard runs inside the job")
PY

echo "===== the raw-EEG cache the front-end needs ====="
# tdm_train.py hard-fails without this, and tdm_gate0.py is the only writer, so
# the schedule is checked here rather than discovered 3 stages into the job.
if [[ -f outputs/tdm/cache/sub08_train_eeg.npy && -f outputs/tdm/cache/sub08_test_eeg.npy ]]; then
  echo "  [ok] outputs/tdm/cache present"
else
  echo "  [info] raw cache missing; stage [2b] will build it inside the job"
fi

if (( fail )); then echo "[FATAL] missing assets; not submitting"; exit 1; fi

echo "===== submit ====="
# NODE SELECTION: `--exclude` rather than `-w` pinning.  The previous pinned OCF
# submission sat in PENDING for hours because dgx-10/dgx-38 were fully occupied,
# and the first unpinned attempt landed on dgx-09 whose driver (12.8) is older
# than this venv's torch build (cu130) -- it failed in 7 s, which is the guard
# working, not a silent CPU fallback.  Excluding only the known-bad node keeps the
# scheduler free to use the other ~20 GPU nodes while making the failure mode
# fast and visible on anything else with an old driver.  `TDM_EXCLUDE` lets the
# bad-node list be extended without editing this script.
EXCL="${TDM_EXCLUDE:-dgx-09}"
JID="$(sbatch --parsable --exclude="${EXCL}" slurm/tdm_intra.sbatch)"
echo "JOBID=${JID}  (excluded: ${EXCL})"
echo "${JID}" > outputs/tdm_intra_sub08.jobid
squeue -j "${JID}" -o "%.12i %.24j %.10T %.10M %.20R" || true
