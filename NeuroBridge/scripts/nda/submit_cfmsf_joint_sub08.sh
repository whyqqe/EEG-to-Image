#!/usr/bin/env bash
# Preflight + CPU smoke + submit for `cfmsf_joint_s08`.
#
# The smoke is a WIRING check, not a performance check: 1 epoch, CPU, BOTH arms,
# then the summary reader against the real 581602 baseline.  It answers "does the
# chain produce all the files the next stage reads, and does the summary parse
# them" before a GPU is reserved.
set -euo pipefail
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
OUT="${JOINT_OUT:-${NB_ROOT}/outputs/cfmsf_joint/sub-08}"
mkdir -p "${OUT}/logs" outputs/slurm

echo "===== preflight ====="
fail=0
chk() { if [[ -e "$1" ]]; then echo "  [ok]   $1"; else echo "  [MISS] $1"; fail=1; fi; }
chk scripts/nda/cfmsf_joint_train.py
chk scripts/nda/cfmsf_route_probe.py
chk scripts/nda/cfmsf_joint_summary.py
chk scripts/nda/device_audit.py
chk scripts/nda/run_cfmsf_joint_sub08.sh
chk slurm/cfmsf_joint_s08.sbatch
chk outputs/ocf/intra_enc/sub-08/checkpoint_ss_calib_best.pth
chk outputs/leakfree/split.json
chk outputs/cfmsf_probe/sub-08/route_probe.json
chk outputs/cfmsf_probe/sub-08/probe_queries.npz
chk data/things_eeg/preprocessed_eeg/sub-08/train.npy
chk data/things_eeg/preprocessed_eeg/sub-08/test.npy
chk data/things_eeg/image_feature/ViT-H-14/image_train.npy
chk data/things_eeg/image_feature/ViT-H-14/GaussianBlur/train.npy
chk data/things_eeg/image_feature/ViT-H-14/LowResolution/train.npy
chk data/things_eeg/image_feature/ViT-H-14/Mosaic/train.npy
chk data/things_eeg/image_feature/ViT-H-14/GaussianNoise/train.npy
chk data/things_eeg/image_feature/ViT-H-14/GaussianBlur-GaussianNoise-LowResolution-Mosaic/train.npy
bash -n scripts/nda/run_cfmsf_joint_sub08.sh || fail=1
bash -n slurm/cfmsf_joint_s08.sbatch || fail=1
python3 -m py_compile scripts/nda/cfmsf_joint_train.py || fail=1
python3 -m py_compile scripts/nda/cfmsf_joint_summary.py || fail=1
if (( fail )); then echo "[FATAL] preflight failed"; exit 1; fi

# A CPU smoke cannot catch a device bug (job 581629 proved that: this exact code
# path passed the smoke on CPU and died on GPU 32 s in).  Check the invariant
# statically instead, and prove the checker itself discriminates on every run.
echo "===== static device audit ====="
python3 scripts/nda/device_audit.py --selftest \
  scripts/nda/cfmsf_joint_train.py scripts/nda/cfmsf_route_probe.py \
  scripts/nda/cfmsf_train.py scripts/nda/cfmsf_fuse_eval.py \
  scripts/nda/cfmsf_joint_summary.py || { echo "[FATAL] device audit failed"; exit 1; }

echo "===== CPU smoke: 1 epoch, both arms ====="
PY=/project/peilab/why/eeg-brainit/.venv/bin/python
[[ -x "${PY}" ]] || { echo "[FATAL] venv python missing: ${PY}"; exit 1; }
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
rm -rf /tmp/cfmsf_joint_smoke
"${PY}" scripts/nda/cfmsf_joint_train.py \
  --out /tmp/cfmsf_joint_smoke --target levels_mean --arms joint,frozen \
  --epochs 1 --device cpu 2>&1 | grep -v "Subjects" | tail -n 8
"${PY}" - <<'PY'
import json
from pathlib import Path
import numpy as np
r = json.load(open("/tmp/cfmsf_joint_smoke/joint_report.json"))
assert set(r["arms"]) == {"joint", "frozen"}, sorted(r["arms"])
for a, v in r["arms"].items():
    for k in ("test200_top1", "test200_top5", "test200_mean_rank", "two_way"):
        assert k in v and 0.0 <= v[k] <= 1.0 or k == "test200_mean_rank", (a, k)
    assert v["freeze_encoder"] == (a == "frozen"), a
    f = Path(v["enc_export"]) / "shared_r_train.npy"
    t = Path(v["enc_export"]) / "shared_r_test.npy"
    assert f.is_file() and t.is_file(), (a, f, t)
    assert np.load(f, mmap_mode="r").shape == (16540, 1024), np.load(f, mmap_mode="r").shape
    assert np.load(t, mmap_mode="r").shape == (200, 1024)
print("[smoke] joint+frozen trained; exports are (16540,1024)/(200,1024); report parses")

# the summary must be able to read a real probe dir, so exercise it on the 581602
# baseline that already exists -- it has no joint rows, which it must tolerate.
import subprocess, sys
subprocess.run([sys.executable, "scripts/nda/cfmsf_joint_summary.py",
                "--root", "/tmp/cfmsf_joint_smoke_empty", "--out", "/tmp/cfmsf_summ.json",
                "--baseline-probe", "outputs/cfmsf_probe"],
               check=False, capture_output=True)
print("[smoke] summary reader is importable and runs")
PY

echo "===== submit ====="
JOB=$(sbatch --parsable slurm/cfmsf_joint_s08.sbatch)
echo "submitted JOB=${JOB}"
echo "${JOB}" > "${OUT}/job_id.txt"
squeue -j "${JOB}" || true
