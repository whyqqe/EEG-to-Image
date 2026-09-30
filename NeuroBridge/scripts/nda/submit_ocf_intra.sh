#!/usr/bin/env bash
# Submit the single OCF-INTRA job for sub-08. One submission only.
set -euo pipefail
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
mkdir -p outputs/slurm outputs/ocf_intra/logs

echo "===== preflight ====="
fail=0
chk() { if [[ -e "$1" ]]; then echo "  [ok]   $1"; else echo "  [MISS] $1"; fail=1; fi; }

chk scripts/nmb/nmb_sota_v2_env.sh
chk scripts/nda/run_ocf_intra.sh
chk scripts/nda/ocf_train.py
chk scripts/nda/ocf_export_intra_z.py
chk scripts/nda/idg_frla_decode.py
chk scripts/nda/nda_ss_pretrain.py
chk scripts/nda/train_eeg_vae_head.py
chk scripts/nda/generate_atm_aligned_decode.py
chk scripts/nda/eval_official_seven_dir.py
chk outputs/g2/targets/sem_concept_tmpl_train.npy
chk outputs/g2/targets/sem_concept_tmpl_test.npy
chk outputs/sdedit_ll_full10/shared/vae_cache/train_vae_latents_f16.npy
chk outputs/sdedit_ll_full10/shared/vae_cache/test_vae_latents_f16.npy
chk outputs/nda_ss/sub-08/clip_text/train/concept_phrases.json
chk outputs/g2/captions/captions_train.jsonl
chk outputs/leakfree/split.json
# reference rows that will be re-evaluated in the same pass
chk outputs/sdedit_ll_full10/sub-08/generation/sdedit_ll/generated/199.png

echo "===== syntax/consistency ====="
python3 -c "import ast
for f in ('scripts/nda/ocf_train.py','scripts/nda/ocf_export_intra_z.py','scripts/nda/idg_frla_decode.py'):
    ast.parse(open(f).read()); print('  [ok] syntax', f)"
bash -n scripts/nda/run_ocf_intra.sh && echo "  [ok] bash -n run_ocf_intra.sh"

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

if (( fail )); then echo "[FATAL] missing assets; not submitting"; exit 1; fi

echo "===== submit ====="
# -w pins to the two nodes whose driver is confirmed new enough for this venv.
JID="$(sbatch --parsable -w dgx-10 -w dgx-38 slurm/ocf_intra.sbatch)"
echo "JOBID=${JID}"
echo "${JID}" > outputs/ocf_intra_sub08.jobid
squeue -j "${JID}" -o "%.12i %.24j %.10T %.10M %.20R" || true
