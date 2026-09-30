#!/usr/bin/env bash
# Preflight + CPU smoke + submit for the CF-MSF fix round.
#
# The smoke must prove the four things that make this round interpretable, each of
# which is a place the design could silently be wrong:
#   1. `lora` starts EXACTLY at `frozen`  -- rank-8 adapters with B=0 are a no-op at
#      step 0, so a difference after training is the adaptation, not a new start;
#   2. `direct` has NO projection head     -- its exported `r` is the optimised tensor;
#   3. `joint` writes BOTH `enc` and `enc_aligned` -- the defect-1 contrast needs both;
#   4. the three selectors record their own epochs -- so the selector problem (defect 3)
#      is measured rather than assumed.
set -euo pipefail
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
OUT="${FIX_OUT:-${NB_ROOT}/outputs/cfmsf_fix/sub-08}"
mkdir -p "${OUT}/logs" outputs/slurm

echo "===== preflight ====="
fail=0
chk() { if [[ -e "$1" ]]; then echo "  [ok]   $1"; else echo "  [MISS] $1"; fail=1; fi; }
chk scripts/nda/cfmsf_fix_train.py
chk scripts/nda/cfmsf_route_probe.py
chk scripts/nda/cfmsf_joint_summary.py
chk scripts/nda/device_audit.py
chk scripts/nda/run_cfmsf_fix_sub08.sh
chk slurm/cfmsf_fix_s08.sbatch
chk outputs/ocf/intra_enc/sub-08/checkpoint_ss_calib_best.pth
chk outputs/leakfree/split.json
chk data/things_eeg/preprocessed_eeg/sub-08/train.npy
chk data/things_eeg/image_feature/ViT-H-14/image_train.npy
chk data/things_eeg/image_feature/ViT-H-14/GaussianBlur/train.npy
chk data/things_eeg/image_feature/ViT-H-14/LowResolution/train.npy
chk data/things_eeg/image_feature/ViT-H-14/Mosaic/train.npy
chk data/things_eeg/image_feature/ViT-H-14/GaussianNoise/train.npy
bash -n scripts/nda/run_cfmsf_fix_sub08.sh || fail=1
bash -n slurm/cfmsf_fix_s08.sbatch || fail=1
python3 -m py_compile scripts/nda/cfmsf_fix_train.py || fail=1
if (( fail )); then echo "[FATAL] preflight failed"; exit 1; fi

echo "===== device audit ====="
python3 scripts/nda/device_audit.py --selftest scripts/nda/cfmsf_fix_train.py \
  || { echo "[FATAL] device audit failed"; exit 1; }

echo "===== CPU smoke: 5 arms x 1 epoch ====="
PY=/project/peilab/why/eeg-brainit/.venv/bin/python
[[ -x "${PY}" ]] || { echo "[FATAL] venv python missing: ${PY}"; exit 1; }
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
rm -rf /tmp/cfmsf_fix_smoke
"${PY}" scripts/nda/cfmsf_fix_train.py \
  --out /tmp/cfmsf_fix_smoke --target levels_mean \
  --arms frozen,lora,direct,joint,disc --epochs 1 --device cpu \
  2>&1 | grep -v "Subjects" | tail -n 12

"${PY}" - <<'PY'
import json
from pathlib import Path
import numpy as np
root = Path("/tmp/cfmsf_fix_smoke")
r = json.loads((root / "fix_report.json").read_text())
A = r["arms"]
assert set(A) == {"frozen", "joint", "direct", "disc", "lora"}, sorted(A)

# (2) `direct` must have no projection head, and must say so
assert A["direct"]["encode_r_directly"] is True, "direct is not aligning r itself"
for a in ("frozen", "joint", "disc", "lora"):
    assert A[a]["encode_r_directly"] is False, a

# (1) `lora` must inject adapters, and `frozen` must inject none
assert A["lora"]["n_lora"] > 0, "LoRA arm injected no adapters"
for a in ("frozen", "joint", "disc", "direct"):
    assert A[a]["n_lora"] == 0, (a, A[a]["n_lora"])

# (3) `joint` (and every arm with a projection) exports BOTH spaces; `direct` only one
for a in ("frozen", "joint", "disc", "lora"):
    assert set(A[a]["spaces"]) == {"enc", "enc_aligned"}, (a, sorted(A[a]["spaces"]))
assert set(A["direct"]["spaces"]) == {"enc"}, sorted(A["direct"]["spaces"])
tdim = int(r["target_dim"])
for a, v in A.items():
    for space, d in v["spaces"].items():
        width = 1024 if space == "enc" else tdim   # enc_aligned lives in target space
        for tag in ("train", "test"):
            f = Path(d) / f"shared_r_{tag}.npy"
            assert f.is_file(), (a, space, tag)
            want = (16540, width) if tag == "train" else (200, width)
            assert np.load(f, mmap_mode="r").shape == want, (a, space, tag)

# (4) every selector recorded a pick, and the PRIMARY one is the mini-gallery stat.
#     This assertion is the regression guard for job 581657's worst bug: the round
#     originally shipped `margin` as primary, `margin` fell monotonically during
#     training, and it therefore selected epoch 0 for three arms -- so those arms'
#     headline numbers came from an untrained head.  Pinning the primary selector here
#     means a silent change back to a discredited statistic fails the smoke instead of
#     quietly corrupting a round.
for a, v in A.items():
    s = v["selectors"]
    assert set(s["picks"]) == {"mini_top1", "two_way", "margin", "top1"}, (a, sorted(s["picks"]))
    for k, p in s["picks"].items():
        assert p["epoch"] == 0, (a, k, p["epoch"])   # only 1 epoch was run
    assert s["chosen"] == "mini_top1", (a, s["chosen"])
    assert set(s["selector_matrix"]) == {"mini_top1", "two_way", "margin", "top1"}, a
    # the margin statistic must still be RECORDED even though it is not primary: it is
    # the evidence that it was wrong, and dropping it would erase that record
    assert "margin" in s["chosen_val_a"], (a, sorted(s["chosen_val_a"]))
    assert "mini_top1" in s["chosen_val_a"], (a, sorted(s["chosen_val_a"]))
print("[smoke] 5 arms OK: lora injects adapters, direct has no proj, both spaces exported,")
print("[smoke]         all 4 selectors recorded + matrix built, primary = mini_top1")

# `joint` vs `direct` must be DIFFERENT tensors: if `direct` silently fell back to a
# projection they would coincide, which would quietly void the defect-1 comparison.
j = np.load(Path(A["joint"]["spaces"]["enc"]) / "shared_r_test.npy", mmap_mode="r")[:16]
ja = np.load(Path(A["joint"]["spaces"]["enc_aligned"]) / "shared_r_test.npy", mmap_mode="r")[:16]
dd = np.load(Path(A["direct"]["spaces"]["enc"]) / "shared_r_test.npy", mmap_mode="r")[:16]
assert not np.allclose(j, ja), "joint's enc and enc_aligned are identical"
assert not np.allclose(j, dd), "direct coincides with joint -- it is not a distinct arm"
print("[smoke] joint/enc, joint/enc_aligned and direct/enc are all distinct tensors")

# the summary tool must consume this tree (it reads <root>/<arm>/probe/, so an
# untrained smoke has no probes -- it must report that rather than crash)
import subprocess, sys
p = subprocess.run([sys.executable, "scripts/nda/cfmsf_joint_summary.py",
                    "--root", str(root), "--out", "/tmp/cfmsf_fix_summ.json",
                    "--pick", "lvl5+agg"], capture_output=True, text=True)
assert "Traceback" not in p.stderr, p.stderr[-800:]
print(f"[smoke] summary tool runs on this tree without a traceback (rc={p.returncode})")

# ---- RESUME PATHS.  Both are load-bearing for the actual round, because job 581657
# ---- already spent ~50 GPU-minutes training four of the five arms before crashing:
# ----   (a) arm_result.json present      -> skipped, NOT retrained, result re-read
# ----   (b) checkpoints, no result file -> RE-SCORED, not retrained
# ---- A resume path is exactly the kind of code that is written once, believed, and
# ---- never exercised -- and if it silently retrained, the round would either waste an
# ---- hour or (worse) mix numbers from two different code versions.
import shutil
ck_dir = root / "lora"
res_lora = ck_dir / "arm_result.json"
assert res_lora.is_file(), "smoke did not write arm_result.json"
before = res_lora.read_text()
# simulate the 581657 state for `disc`: checkpoints present, result file absent
res_disc = root / "disc" / "arm_result.json"
res_disc.unlink()
# (a) `lora` keeps its result -> must be skipped and re-read verbatim
# (b) `disc` has only checkpoints -> must be re-scored
p2 = subprocess.run([sys.executable, "scripts/nda/cfmsf_fix_train.py",
                     "--out", str(root), "--target", "levels_mean",
                     "--arms", "lora,disc", "--epochs", "1", "--device", "cpu"],
                    capture_output=True, text=True)
if p2.returncode != 0:
    print(p2.stdout[-2500:]); print(p2.stderr[-2500:])
assert p2.returncode == 0, f"resume run failed (rc={p2.returncode})"
assert "[skip] arm_result.json present" in p2.stdout, "lora was not skipped on resume"
assert "re-scoring existing checkpoints" in p2.stdout, "disc was not re-scored on resume"
assert res_lora.read_text() == before, "skipped arm's stored result changed"
r2 = json.loads((root / "fix_report.json").read_text())
assert set(r2["arms"]) == {"lora", "disc"}, sorted(r2["arms"])
assert r2["arms"]["disc"]["selectors"]["chosen"] == "mini_top1"
assert "selector_analysis" in r2, "selector table missing from the report"
print("[smoke] resume: result-file arm skipped verbatim, checkpoint-only arm re-scored")
print("[smoke] (no retraining on either path -- this is what recovers 581657's 4 arms)")
PY

echo "===== submit ====="
JOB=$(sbatch --parsable slurm/cfmsf_fix_s08.sbatch)
echo "submitted JOB=${JOB}"
echo "${JOB}" > "${OUT}/job_id.txt"
squeue -j "${JOB}" || true
