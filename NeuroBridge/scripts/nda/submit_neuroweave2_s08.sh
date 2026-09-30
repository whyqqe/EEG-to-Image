#!/usr/bin/env bash
# Preflight + CPU smoke for NeuroWeave ROUND 2, then submit.
#
# The smoke is here for one reason: round 1 submitted TWO jobs of the same name
# that raced on one output directory, corrupted several JSON files, and killed the
# summary stage.  Every check below is (a) an assertion about the new code paths,
# or (b) a NEGATIVE control that must fail, plus (c) a guard that refuses to
# create that race again.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"

PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}/hub"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
SMOKE="$(mktemp -d /tmp/nweave2_smoke.XXXXXX)"
trap 'rm -rf "${SMOKE}"' EXIT

chk() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
say() { echo; echo "===== $* ====="; }

say "0/6 sources + assets"
for f in scripts/nda/neuroweave_s1_train.py scripts/nda/neuroweave_anytime_eval.py \
         scripts/nda/neuroweave_cycle_score.py scripts/nda/neuroweave_summary.py \
         scripts/nda/run_neuroweave2_s08.sh slurm/nweave2_s08.sbatch \
         scripts/nda/cfmsf_route_probe.py scripts/nda/device_audit.py; do
  chk "$f"
done
chk outputs/ocf/intra_enc/sub-08/checkpoint_ss_calib_best.pth
chk outputs/leakfree/split.json
chk outputs/cfmsf_all/sub-08/frozen/probe/probe_queries.npz
chk outputs/atm_aligned_decode/sub-08/generation/combo_d40_luma_pc_a060/generated/000.png
chk outputs/mb_s08/gen/mb_p3_i3_cn/generated/000.png

say "1/6 device audit"
"${PYTHON}" scripts/nda/device_audit.py \
  scripts/nda/neuroweave_s1_train.py \
  scripts/nda/neuroweave_anytime_eval.py \
  scripts/nda/neuroweave_cycle_score.py

say "2/6 new arms registered + sample_end schedules (incl. negative control)"
"${PYTHON}" - <<'PY' >"${SMOKE}/sched.log" 2>&1
import sys, collections
import numpy as np
sys.path.insert(0, "scripts/nda")
from neuroweave_s1_train import ARM_CFG, WIN, sample_end, ANYTIME_ARMS, ROUND1_ARMS

for a in ("anytime_soft", "anytime_curric", "anytime_consist", "hier_aux"):
    assert a in ARM_CFG, a
for a in ROUND1_ARMS:
    assert a in ARM_CFG, a

class A:
    anytime_full_p = 0.5
    anytime_max_p = 0.6
    curric_ramp = 20

# soft: P(full) must be near the requested 0.5, NOT 0.25 (round-1 uniform).
rng = np.random.default_rng(0)
c = collections.Counter(sample_end("soft", 0, A(), rng) for _ in range(20000))
assert abs(c[WIN["full"]] / 20000 - 0.5) < 0.03, c
# round-1 rule kept, and it must still be the uniform one.
c1 = collections.Counter(sample_end("anytime", 0, A(), rng) for _ in range(20000))
assert all(abs(c1[w] / 20000 - 0.25) < 0.03 for w in WIN.values()), c1
# curric: epoch 0 must truncate NOTHING, late epochs must truncate.
c0 = collections.Counter(sample_end("curric", 0, A(), rng) for _ in range(4000))
assert c0[WIN["full"]] / 4000 > 0.98, c0
c9 = collections.Counter(sample_end("curric", 39, A(), rng) for _ in range(4000))
assert 0.35 < c9[WIN["full"]] / 4000 < 0.45, c9
# NEGATIVE CONTROL: an unknown mode must fail loudly, not silently return "full",
# which would have made a mis-typed arm look like a clean baseline.
try:
    sample_end("nonsense", 0, A(), rng)
    raise SystemExit("[FATAL] unknown temporal mode did not raise")
except KeyError:
    pass
print("[ok] schedules: soft~0.50 full, anytime~0.25, curric 1.00->~0.40, bad mode raises")
PY
cat "${SMOKE}/sched.log"

say "3/6 GT image ordering (the control the cycle numbers depend on)"
"${PYTHON}" - <<'PY' >"${SMOKE}/gt.log" 2>&1
import sys
from pathlib import Path
sys.path.insert(0, "scripts/nda")
import numpy as np
from neuroweave_cycle_score import gt_test_images, IMG_EXT
imgs = gt_test_images(200)
assert len(imgs) == 200
meta = np.load("/project/peilab/why/data/images_set/image_metadata.npy",
               allow_pickle=True).item()
cs = list(meta["test_img_concepts"])
assert len(cs) == 200 and len(set(cs)) == 200
# the dataset order is the numeric-prefix order; if this breaks, the GT bank the
# routes were trained against would no longer line up with these images.
assert cs == sorted(cs), "test_img_concepts are not in sorted order"
root = Path("/project/peilab/why/data/images_set/test_images")
assert {p.name for p in root.iterdir()} == set(cs)
print(f"[ok] 200 GT images, order == sorted(test_img_concepts), dirs match")
PY
cat "${SMOKE}/gt.log"

say "4/6 cycle: aligned EEG bank present and in CLIP space (not raw shared_r)"
"${PYTHON}" - <<'PY' >"${SMOKE}/bank.log" 2>&1
import sys
import numpy as np
sys.path.insert(0, "scripts/nda")
from cfmsf_joint_train import l2n, load_level, VITH
d = np.load("outputs/cfmsf_all/sub-08/frozen/probe/probe_queries.npz")
k = "vith_levels_mean__mlp_q"
assert k in d, list(d.keys())[:5]
q = l2n(d[k].astype(np.float32))
gt = load_level(VITH, "image", "test")
top1 = float((q @ gt.T).argmax(1).eq(np.arange(len(q))).mean())
# Sanity: the ALIGNED bank must be far better than chance here; the raw encoder
# feature round 1 mistakenly used scored top1=0.0000, which is what made its
# eeg_agree identically zero.
assert top1 > 0.25, f"aligned EEG bank looks wrong: top1={top1}"
raw = l2n(np.load("outputs/ocf/intra_z/sub-08/shared_r_test.npy").astype(np.float32))
raw_top1 = float((raw @ gt.T).argmax(1).eq(np.arange(len(raw))).mean())
print(f"[ok] aligned bank top1={top1:.4f} vs raw shared_r top1={raw_top1:.4f} "
      f"(round-1 bug reproduced and avoided)")
PY
cat "${SMOKE}/bank.log"

say "5/6 orchestration parses; new arms wired end to end"
bash -n scripts/nda/run_neuroweave2_s08.sh
bash -n slurm/nweave2_s08.sbatch
for a in anytime_soft anytime_curric anytime_consist hier_aux; do
  grep -q "$a" scripts/nda/run_neuroweave2_s08.sh || { echo "[FATAL] $a not in run script"; exit 1; }
done
grep -q "13 routes" scripts/nda/run_neuroweave2_s08.sh
grep -q "neuroweave_cycle_score" scripts/nda/run_neuroweave2_s08.sh
echo "[ok] orchestration"

say "6/6 DUPLICATE-JOB GUARD (round 1 lost a stage to exactly this race)"
if squeue -u "${USER}" -h -o "%j" 2>/dev/null | grep -qx "nweave2_s08"; then
  echo "[FATAL] a nweave2_s08 job is already queued or running. Round 1 submitted"
  echo "        the same job twice; both wrote one output directory, corrupted"
  echo "        several JSON files and killed the summary stage. Cancel the old"
  echo "        job first (scancel <id>) or wait for it."
  exit 1
fi
echo "[ok] no nweave2_s08 in queue"

say "SUBMIT"
mkdir -p outputs/slurm outputs/neuroweave2/sub-08/logs
JOB=$(sbatch --parsable slurm/nweave2_s08.sbatch)
echo "[submitted] job ${JOB}"
echo "${JOB}" > outputs/neuroweave2/sub-08/job_id.txt
squeue -j "${JOB}" || true
