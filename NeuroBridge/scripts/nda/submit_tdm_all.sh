#!/usr/bin/env bash
# Submit the single TDM-DT all-subject overnight job. One submission only.
set -euo pipefail
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
mkdir -p outputs/slurm outputs/tdm_all/logs

echo "===== preflight ====="
fail=0
chk() { if [[ -e "$1" ]]; then echo "  [ok]   $1"; else echo "  [MISS] $1"; fail=1; fi; }

chk scripts/nmb/nmb_sota_v2_env.sh
chk scripts/nda/run_tdm_all.sh
chk scripts/nda/tdm_train.py
chk scripts/nda/tdm_clip_patch.py
chk scripts/nda/tdm_gate0.py
chk scripts/nda/ocf_export_intra_z.py
chk scripts/nda/idg_frla_decode.py
chk scripts/nda/nda_ss_pretrain.py
chk scripts/nda/train_eeg_vae_head.py
chk scripts/nda/generate_atm_aligned_decode.py
chk scripts/nda/eval_official_seven_dir.py
chk slurm/tdm_all.sbatch
# shared, image-level assets: the same for every subject
chk outputs/tdm/clip_patch/train_patch_f16.npy
chk outputs/g2/targets/sem_concept_tmpl_train.npy
chk outputs/g2/targets/sem_concept_tmpl_test.npy
chk outputs/g2/targets/perc_struct_train.npy
chk outputs/g2/targets/perc_struct_test.npy
chk outputs/sdedit_ll_full10/shared/vae_cache/train_vae_latents_f16.npy
chk outputs/sdedit_ll_full10/shared/vae_cache/test_vae_latents_f16.npy
chk outputs/nda_ss/sub-08/clip_text/train/concept_phrases.json
chk outputs/g2/captions/captions_train.jsonl
chk outputs/leakfree/split.json
# the reuse check that matters: the pipeline must NOT retrain sub-08 or rebuild
# the 2.7 GB patch cache, so both must already be on disk
chk outputs/ocf/intra_enc/sub-08/checkpoint_ss_calib_best.pth
chk outputs/ocf/intra_z/sub-08/shared_r_test.npy
# reference rows (sub-08 only, re-scored by the same code in every pass)
chk outputs/sdedit_ll_full10/sub-08/generation/sdedit_ll/generated/199.png
chk outputs/g3f/gen/sub-08/g3f_ll_selfgate/generated/199.png

echo "===== syntax/consistency ====="
python3 -c "import ast
for f in ('scripts/nda/tdm_train.py','scripts/nda/tdm_clip_patch.py','scripts/nda/tdm_gate0.py'):
    ast.parse(open(f).read()); print('  [ok] syntax', f)"
bash -n scripts/nda/run_tdm_all.sh && echo "  [ok] bash -n run_tdm_all.sh"

echo "===== the freeze guard is actually in place ====="
# The previous run looked healthy in its logs while every gradient was multiplied
# by zero.  These two greps are the tripwire: both the bounded norm floor (so the
# grad-norm cannot overflow) and the PREFLIGHT abort must be present.
grep -q "clamp_min(1e-3)" scripts/nda/ocf_train.py \
  && echo "  [ok] l2t norm floor is 1e-3 (bounded Jacobian)" \
  || { echo "  [FAIL] l2t still has an unbounded Jacobian"; fail=1; }
grep -q "PREFLIGHT FAILED" scripts/nda/tdm_train.py \
  && echo "  [ok] tdm_train.py aborts on a non-finite/none gradient" \
  || { echo "  [FAIL] no preflight guard"; fail=1; }
grep -q "head_init_std" scripts/nda/tdm_train.py \
  && echo "  [ok] read-out heads no longer start at exactly zero" \
  || { echo "  [FAIL] zero-init read-out still present"; fail=1; }

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

echo "===== disk budget ====="
# /project is a SHARED 50 T filesystem that reports 100% use.  The run is built so
# that per-subject scratch is deleted as soon as it is consumed; this prints the
# state before the night starts so a regression is visible in the log.
echo "  avail: $(df -h "${NB_ROOT}" | awk 'NR==2{print $4}')"
echo "  my outputs: $(du -sh ${NB_ROOT}/outputs 2>/dev/null | cut -f1)"
echo "  expected steady state: ~0.2 GB kept per subject, ~0.5 GB scratch at a time"

if (( fail )); then echo "[FATAL] missing assets; not submitting"; exit 1; fi

echo "===== submit ====="
# `--exclude` rather than `-w` pinning: the pinned submission waited hours behind
# other users, and the first unpinned one landed on dgx-09 whose driver (12.8) is
# older than this venv's torch (cu130) -- which failed in 7 s, i.e. the guard
# working as designed rather than a silent CPU fallback.
EXCL="${TDM_EXCLUDE:-dgx-09}"
JID="$(sbatch --parsable --exclude="${EXCL}" slurm/tdm_all.sbatch)"
echo "JOBID=${JID}  (excluded: ${EXCL})"
echo "${JID}" > outputs/tdm_all_subjects.jobid
squeue -j "${JID}" -o "%.12i %.24j %.10T %.10M %.20R" || true
