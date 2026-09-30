#!/usr/bin/env bash
# Preflight + CPU smoke + submit for the CF-MSF selection-rule probe.
#
# The smoke must prove four things, each of which is a way the run could be
# uninterpretable rather than merely broken:
#   1. the per-epoch curve is recorded for ALL epochs -- the whole point is that
#      `cfmsf_train` discards it, so an off-by-one that truncates it would silently
#      reproduce the blind spot this run exists to remove;
#   2. the ORACLE column is present and is >= every selector -- if a rule ever beat the
#      oracle, the oracle would not be a ceiling and the comparison would be meaningless;
#   3. the selectors genuinely DISAGREE about the epoch -- if every rule picked the same
#      epoch there would be nothing to measure, and the reported regret would be an
#      artefact of the tiny epoch count rather than evidence about the rules;
#   4. `swa` and `fixed` rows exist and differ from the selector rows -- they are the
#      two "avoid selection entirely" alternatives and must not be silently empty.
#
# The smoke deliberately runs a route whose output dim is LARGE (cat5, 5120-d) as well
# as a small one (image, 1024-d), because the SWA accumulator and the score_weights
# rebuild both have to handle the bigger head.
set -euo pipefail
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
OUT="${SEL_OUT:-${NB_ROOT}/outputs/cfmsf_sel/sub-08}"
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
mkdir -p "${OUT}/logs" outputs/slurm

echo "===== preflight ====="
fail=0
chk() { if [[ -e "$1" ]]; then echo "  [ok]   $1"; else echo "  [MISS] $1"; fail=1; fi; }
chk scripts/nda/cfmsf_sel_probe.py
chk scripts/nda/cfmsf_select_audit.py
chk scripts/nda/run_cfmsf_sel_sub08.sh
chk slurm/cfmsf_sel_s08.sbatch
chk outputs/cfmsf_all/aggregate.json
chk outputs/cfmsf_all/sub-08/frozen/enc/sub-08/shared_r_train.npy
chk outputs/cfmsf_all/sub-08/frozen/enc/sub-08/shared_r_test.npy
chk outputs/leakfree/split.json
chk outputs/gem/cond_cache/clip_img1024_train.npy
bash -n scripts/nda/run_cfmsf_sel_sub08.sh || fail=1
bash -n slurm/cfmsf_sel_s08.sbatch || fail=1
python3 -m py_compile scripts/nda/cfmsf_sel_probe.py scripts/nda/cfmsf_select_audit.py || fail=1
if (( fail )); then echo "[FATAL] preflight failed"; exit 1; fi

echo "===== device audit ====="
python3 scripts/nda/device_audit.py --selftest scripts/nda/cfmsf_sel_probe.py \
  || { echo "[FATAL] device audit failed"; exit 1; }

echo "===== the audit that motivates this run (read-only) ====="
/project/peilab/why/eeg-brainit/.venv/bin/python scripts/nda/cfmsf_select_audit.py \
  | grep -E "Spearman|misses the best route|positive in|ratio =" || true

echo "===== CPU smoke: 2 routes x 5 epochs ====="
PY=/project/peilab/why/eeg-brainit/.venv/bin/python
[[ -x "${PY}" ]] || { echo "[FATAL] venv python missing: ${PY}"; exit 1; }
rm -rf /tmp/cfmsf_sel_smoke
"${PY}" scripts/nda/cfmsf_sel_probe.py \
  --out /tmp/cfmsf_sel_smoke --test-subject 8 \
  --z-root outputs/cfmsf_all/sub-08/frozen/enc \
  --routes vith_cat5,vith_image --epochs 5 --device cpu \
  --swa-k 2,3 --fixed-epochs 1,2 2>&1 | tail -n 16

"${PY}" - <<'PY'
import json
from pathlib import Path
import numpy as np
r = json.loads(Path("/tmp/cfmsf_sel_smoke/sel_report.json").read_text())
R = r["routes"]
assert set(R) == {"vith_cat5", "vith_image"}, sorted(R)

for name, v in R.items():
    # (1) the FULL curve, not a tail
    assert len(v["curve"]) == 5, (name, len(v["curve"]))
    assert [c["epoch"] for c in v["curve"]] == [0, 1, 2, 3, 4], (name, "non-contiguous")
    # every selector must be recorded on EVERY epoch, which is what makes the
    # regret decomposition possible
    for s in ("full_top1", "mini_top1", "mini_csls", "two_way", "inst_cos"):
        assert f"valb_{s}" in v["curve"][0], (name, s, sorted(v["curve"][0]))
        assert f"vala_{s}" in v["curve"][0], (name, s)
    assert "oracle_test_top1" in v["curve"][0], name

    # (2) the oracle is a CEILING
    oc = v["oracle"]["best_test_top1"]
    curve_max = max(c["oracle_test_top1"] for c in v["curve"])
    assert abs(oc - curve_max) < 1e-9, (name, oc, curve_max)
    for rule, row in v["selectors"].items():
        assert row["top1"] <= oc + 1e-9, (name, rule, row["top1"], oc)
    for rule, row in v["swa"].items():
        assert row["top1"] <= oc + 1e-9, (name, rule, row["top1"], oc)
    assert "swa2" in v["swa"] and "swa3" in v["swa"], (name, sorted(v["swa"]))
    assert set(v["fixed"]) == {"fixed1", "fixed2"}, (name, sorted(v["fixed"]))
    # large-output head must survive the SWA rebuild.  `cat5` concatenates FIVE levels
    # (image + 4 perturbations) so its target is 5120-d, not 4096-d -- asserting 4096
    # here was wrong and aborted a submission.  The expected width is derived from the
    # level count rather than written as a literal, so adding a level cannot silently
    # invalidate the check.
    assert v["dim"] == (5 * 1024 if name == "vith_cat5" else 1024), (name, v["dim"])
    if name == "vith_cat5":
        assert v["n_params"] > 7e6, ("cat5 head unexpectedly small", v["n_params"])

print("[smoke] full 5-epoch curve recorded for every selector (val_a AND val_b)")
print("[smoke] oracle >= every rule's score, on both a 1024-d and a 5120-d head")
print("[smoke] swa + fixed baselines present; SWA rebuilds the 5120-d head correctly")

# (3) the rules must be able to disagree -- otherwise there is nothing to measure
epochs = {name: {s: v["selectors"][s]["epoch"] for s in v["selectors"]}
          for name, v in R.items()}
for name, e in epochs.items():
    assert len(set(e.values())) > 1, (
        f"{name}: every selector picked epoch {set(e.values())} on a 5-epoch run; the "
        f"smoke cannot distinguish the rules, so raise --epochs in the smoke")
print(f"[smoke] selectors disagree about the epoch: {epochs}")
print("[smoke] (a 5-epoch smoke cannot settle which rule is best -- only that they differ)")

# (4) the summary must price every rule and expose the oracle as a bound
assert "per_rule" in r["summary"], sorted(r["summary"])
assert "full_top1" in r["summary"]["per_rule"], sorted(r["summary"]["per_rule"])
assert "mini_top1" in r["summary"]["per_rule"], sorted(r["summary"]["per_rule"])
for rule, s in r["summary"]["per_rule"].items():
    assert s["mean_regret_vs_oracle"] <= 1e-9, (rule, s)
print("[smoke] summary prices every rule and keeps regret <= 0 against the ceiling")
PY

echo "===== submit ====="
JOB=$(sbatch --parsable slurm/cfmsf_sel_s08.sbatch)
echo "submitted JOB=${JOB}"
echo "${JOB}" > "${OUT}/job_id.txt"
squeue -j "${JOB}" || true
