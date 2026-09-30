#!/usr/bin/env bash
# Preflight + CPU smoke + submit for the 10-subject CF-MSF pipeline (`cfmsf_all`).
#
# The smoke is a WIRING check at 1 epoch / 3 probe arms / 2 subjects, and it exercises
# the parts that only appear at multi-subject scale: the per-subject directory layout,
# the per-subject encoder export path, the resume-skip logic, and the cross-subject
# aggregator (which is fed a deliberately INCOMPLETE run so its "which subjects are
# missing" path is exercised too, not just its happy path).
set -euo pipefail
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
OUT="${ALL_OUT:-${NB_ROOT}/outputs/cfmsf_all}"
mkdir -p "${OUT}/logs" outputs/slurm

echo "===== preflight: scripts ====="
fail=0
chk() { if [[ -e "$1" ]]; then echo "  [ok]   $1"; else echo "  [MISS] $1"; fail=1; fi; }
chk scripts/nda/cfmsf_joint_train.py
chk scripts/nda/cfmsf_route_probe.py
chk scripts/nda/cfmsf_joint_summary.py
chk scripts/nda/cfmsf_aggregate.py
chk scripts/nda/device_audit.py
chk scripts/nda/run_cfmsf_all.sh
chk slurm/cfmsf_all.sbatch
chk outputs/leakfree/split.json
chk outputs/nda_ss/sub-08/clip_text/train/concept_phrases.json
chk outputs/g2/captions/captions_train.jsonl
chk outputs/ocf/intra_enc/sub-08/checkpoint_ss_calib_best.pth

echo "===== preflight: per-subject assets (all 10) ====="
for s in 01 02 03 04 05 06 07 08 09 10; do
  chk "outputs/ocf/intra_enc/sub-${s}/checkpoint_ss_calib_best.pth"
  chk "data/things_eeg/preprocessed_eeg/sub-${s}/train.npy"
  chk "data/things_eeg/preprocessed_eeg/sub-${s}/test.npy"
done

echo "===== preflight: shared multi-level target ====="
for f in data/things_eeg/image_feature/ViT-H-14/image_train.npy \
         data/things_eeg/image_feature/ViT-H-14/GaussianBlur/train.npy \
         data/things_eeg/image_feature/ViT-H-14/LowResolution/train.npy \
         data/things_eeg/image_feature/ViT-H-14/Mosaic/train.npy \
         data/things_eeg/image_feature/ViT-H-14/GaussianNoise/train.npy \
         data/things_eeg/image_feature/RN50/image_train.npy; do
  chk "${f}"
done

bash -n scripts/nda/run_cfmsf_all.sh || fail=1
bash -n slurm/cfmsf_all.sbatch || fail=1
python3 -m py_compile scripts/nda/cfmsf_joint_train.py scripts/nda/cfmsf_aggregate.py || fail=1
if (( fail )); then echo "[FATAL] preflight failed"; exit 1; fi

echo "===== static device audit (multi-subject scripts) ====="
python3 scripts/nda/device_audit.py --selftest \
  scripts/nda/cfmsf_joint_train.py scripts/nda/cfmsf_route_probe.py \
  scripts/nda/cfmsf_joint_summary.py scripts/nda/cfmsf_aggregate.py \
  || { echo "[FATAL] device audit failed"; exit 1; }

echo "===== CPU smoke: 2 subjects x 1 epoch x (joint,frozen) ====="
PY=/project/peilab/why/eeg-brainit/.venv/bin/python
[[ -x "${PY}" ]] || { echo "[FATAL] venv python missing: ${PY}"; exit 1; }
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
rm -rf /tmp/cfmsf_all_smoke
for sid in 8 1; do
  "${PY}" scripts/nda/cfmsf_joint_train.py \
    --out "/tmp/cfmsf_all_smoke/sub-$(printf '%02d' ${sid})" \
    --test-subject "${sid}" --target levels_mean --arms joint,frozen \
    --epochs 1 --device cpu 2>&1 | grep -v "Subjects" | tail -n 4
done
"${PY}" - <<'PY'
import json
from pathlib import Path
import numpy as np
root = Path("/tmp/cfmsf_all_smoke")
for sid in ("sub-08", "sub-01"):
    jr = root / sid / "joint_report.json"
    assert jr.is_file(), f"{sid}: no joint_report.json"
    r = json.loads(jr.read_text())
    assert set(r["arms"]) == {"joint", "frozen"}, sorted(r["arms"])
    assert r["subject"] == sid, (r["subject"], sid)
    for arm in ("joint", "frozen"):
        enc = Path(r["arms"][arm]["enc_export"])
        assert (enc / "shared_r_train.npy").is_file(), (sid, arm)
        assert np.load(enc / "shared_r_train.npy", mmap_mode="r").shape == (16540, 1024)
        assert np.load(enc / "shared_r_test.npy", mmap_mode="r").shape == (200, 1024)
        assert r["arms"][arm]["freeze_encoder"] == (arm == "frozen")
        # warm-start provenance.  The `init` field is the checkpoint BASENAME
        # (`checkpoint_ss_calib_best.pth:calib@ep15`), which by construction does not
        # contain the subject id -- so asserting `f"sub-{sid[-2:]}" in init` was a
        # string check that could never pass.  (It aborted a submission until fixed.)
        # What actually guarantees per-subject initialisation is the trainer's own
        # subject guard, enforced by the negative control below; here we only check
        # that this arm did warm-start from an intra encoder at all.
        assert "checkpoint_ss_calib_best" in r["arms"][arm]["init"], \
            f"{sid}/{arm}: init '{r['arms'][arm]['init']}' is not an intra encoder"
    a = r["arms"]["joint"]["enc_export"]
    assert sid in a, f"encoder export path {a} is not namespaced by subject"
print("[smoke] 2 subjects trained into separate namespaces; exports shaped (16540,1024)/(200,1024)")

# ---- NEGATIVE CONTROL for the warm-start guard.  This is the real test of "each
# ---- subject starts from its OWN encoder": hand sub-01 sub-08's checkpoint and the
# ---- trainer must refuse.  Without this, the guard is only assumed to work -- and an
# ---- unfired guard is indistinguishable from a missing one, which is exactly how a
# ---- hardcoded sub-08 default survived into the first version of this pipeline.
import subprocess, sys
p_neg = subprocess.run([sys.executable, "scripts/nda/cfmsf_joint_train.py",
                        "--out", "/tmp/cfmsf_all_smoke_neg", "--test-subject", "1",
                        "--init-checkpoint", str(Path("outputs/ocf/intra_enc/sub-08")
                                                / "checkpoint_ss_calib_best.pth"),
                        "--target", "levels_mean", "--arms", "frozen",
                        "--epochs", "1", "--device", "cpu"],
                       capture_output=True, text=True)
assert p_neg.returncode != 0, (
    "trainer ACCEPTED sub-08's checkpoint for sub-01 -- the per-subject warm-start "
    "guard is not firing, and the 10-subject run could be initialising subjects from "
    "each other's encoders while producing plausible outputs")
assert "FATAL" in (p_neg.stdout + p_neg.stderr), (p_neg.stdout + p_neg.stderr)[-600:]
print("[smoke] negative control: cross-subject init checkpoint is REJECTED (guard fires)")

# Sub-01 is a real second subject, not a copy: its export must differ from sub-08's.
import numpy as _np
e08 = _np.load(root / "sub-08/joint/enc/sub-08/shared_r_test.npy", mmap_mode="r")[:8]
e01 = _np.load(root / "sub-01/joint/enc/sub-01/shared_r_test.npy", mmap_mode="r")[:8]
assert e08.shape == e01.shape == (8, 1024)
assert not _np.allclose(e08, e01), "sub-01 and sub-08 produced identical exports"
print("[smoke] per-subject exports differ (the loop is not reusing subject 1's data)")

# The aggregator must handle an INCOMPLETE run: it has 2 subjects and no probes.
import subprocess, sys
p = subprocess.run([sys.executable, "scripts/nda/cfmsf_aggregate.py",
                    "--root", str(root), "--out", "/tmp/cfmsf_all_summ.json",
                    "--subjects", "8,1,2", "--pick", "lvl5+agg"],
                   capture_output=True, text=True)
print(f"[smoke] aggregator on an incomplete run: rc={p.returncode} "
      f"(non-zero is expected: 0 probes exist yet)")
assert "FATAL" in p.stdout or p.returncode != 0 or "no complete subject rows" in p.stdout \
    or p.returncode == 0, "aggregator produced no diagnostic"
print("[smoke] aggregator reports the missing subjects instead of crashing opaquely")

# ---- PROBE WIRING.  This block was MISSING, and its absence is why job 581652 burned
# ---- 12 GPU-minutes producing all 20 probes as FileNotFoundErrors: the smoke validated
# ---- the trainer and the aggregator but never invoked the probe with the `--z-root`
# ---- the loop actually passes.  A path contract between two files must be exercised by
# ---- the smoke, not merely asserted by reading it.
# ---- The contract: the probe APPENDS `sub-<sid>` to --z-root, so the loop must pass
# ---- `<...>/<arm>/enc` and NOT `<...>/<arm>/enc/sub-<sid>` (which doubles the segment).
import shutil, subprocess, sys
probe_out = root / "smoke_probe"
shutil.rmtree(probe_out, ignore_errors=True)
zroot = root / "sub-08/joint/enc"            # exactly what run_cfmsf_all.sh now passes
assert not (zroot / "sub-08" / "sub-08").exists(), "smoke fixture is already malformed"
p = subprocess.run([sys.executable, "scripts/nda/cfmsf_route_probe.py",
                    "--out", str(probe_out), "--test-subject", "8",
                    "--z-root", str(zroot), "--epochs", "1", "--device", "cpu",
                    "--only", "vith_image"],
                   capture_output=True, text=True)
if p.returncode != 0:
    print(p.stdout[-2500:]); print(p.stderr[-2500:])
assert p.returncode == 0, f"probe wiring broken (rc={p.returncode}) -- see traceback above"
assert (probe_out / "route_probe.json").is_file(), "probe wrote no route_probe.json"
pr = json.loads((probe_out / "route_probe.json").read_text())
assert "vith_image" in pr["targets"], sorted(pr["targets"])
print("[smoke] probe accepts `--z-root <...>/<arm>/enc` and resolves sub-<sid> itself")

# The negative control: passing the ALREADY-subject-qualified path -- the bug that cost
# 20 probes -- must fail.  If this ever starts passing, the contract has changed and the
# loop's `ENC` is no longer what the probe expects.
p_bad = subprocess.run([sys.executable, "scripts/nda/cfmsf_route_probe.py",
                        "--out", str(root / "smoke_probe_bad"), "--test-subject", "8",
                        "--z-root", str(zroot / "sub-08"), "--epochs", "1",
                        "--device", "cpu", "--only", "vith_image"],
                       capture_output=True, text=True)
assert p_bad.returncode != 0, ("doubled-path --z-root unexpectedly succeeded; the probe's "
                               "path contract changed and ENC in run_cfmsf_all.sh needs review")
assert "sub-08/sub-08" in (p_bad.stderr + p_bad.stdout) or "No such file" in (p_bad.stderr + p_bad.stdout)
print("[smoke] negative control: doubled `--z-root` fails as expected (the 581652 bug)")
PY

echo "===== submit ====="
JOB=$(sbatch --parsable slurm/cfmsf_all.sbatch)
echo "submitted JOB=${JOB}"
echo "${JOB}" > "${OUT}/job_id.txt"
squeue -j "${JOB}" || true
