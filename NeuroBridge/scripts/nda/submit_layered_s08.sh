#!/usr/bin/env bash
# Preflight + GPU-submission for the layer-wise injection experiment.
#
# Two failure modes justify every check below:
#   1. A layer spec that silently does NOT land renders perfectly plausible
#      images.  The null would then be reported as evidence against layered
#      injection when nothing was ever injected.  So the audit's negative
#      controls must RAISE, the plan must resolve against the real UNet config,
#      and the job itself re-proves the assignment by read-back and by a pixel
#      diff between two specs on one pipeline.
#   2. A job submitted twice against one output directory corrupts the shared
#      JSON artifacts (this already cost the NeuroWeave round-1 summary stage).
set -uo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"

PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
export LAYERED_OUT="${LAYERED_OUT:-${NB_ROOT}/outputs/layered/sub-08}"

SMOKE="$(mktemp -d /tmp/layered_smoke.XXXXXX)"
trap 'rm -rf "${SMOKE}"' EXIT

chk() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
say() { echo; echo "===== $* ====="; }
fail=0

say "0/7 sources + assets"
for f in scripts/nda/generate_layered_decode.py scripts/nda/layered_arms.py \
         scripts/nda/layer_plan_audit.py scripts/nda/layered_summary.py \
         scripts/nda/run_layered_sub08.sh slurm/layered_s08.sbatch \
         scripts/nda/device_audit.py scripts/nda/eval_official_seven_dir.py; do
  chk "$f"
done
for f in outputs/uck/sub-08/full/conds/ip_mem_test.npy \
         outputs/mb_s08/heads/conds/depth_pred_test_cal.npy \
         outputs/mb_s08/heads/conds/edge_pred_test_cal.npy \
         outputs/g2f/prompts/prompts_deploy.json \
         outputs/uck/sub-08/full/spatial/pred_depth_rgb_512/199.png \
         outputs/sdedit_ll_full10/sub-08/vae_head/pred_lowlevel_rgb_512/199.png \
         outputs/atm_aligned_decode/sub-08/generation/combo_d40_luma_pc_a060/generated/199.png \
         outputs/mb_s08/gen/mb_p3_i3_cn/generated/199.png \
         outputs/mb_s08/eval/s08_mb_p3_i3_cn.json ; do
  chk "$f"
done
echo "[ok] all sources and the four reference artifacts are present"

say "1/7 layer plan audit (pure logic + 5 negative controls that must raise)"
"${PYTHON}" scripts/nda/layer_plan_audit.py > "${SMOKE}/audit.log" 2>&1
if [[ $? -ne 0 ]]; then
  echo "[FATAL] layer plan audit failed:"; cat "${SMOKE}/audit.log"; exit 1
fi
grep -c "^\[ok\]" "${SMOKE}/audit.log" | sed 's/^/[ok] audit assertions passed: /'
grep -q "rejected" "${SMOKE}/audit.log" \
  || { echo "[FATAL] the audit reported no negative-control rejections"; exit 1; }
echo "[ok] negative controls are active, not vacuously true"

say "2/7 plan resolves against the REAL unet config, with the expected masses"
CONDS="outputs/uck/sub-08/full/conds/ip_mem_test.npy,outputs/mb_s08/heads/conds/depth_pred_test_cal.npy,outputs/mb_s08/heads/conds/edge_pred_test_cal.npy"
PF="outputs/g2f/prompts/prompts_deploy.json"
"${PYTHON}" - "$CONDS" "$PF" "$SMOKE" <<'PY'
import json, subprocess, sys
from pathlib import Path
sys.path.insert(0, "scripts/nda")
from layered_arms import GENERATE_ARMS, LAYERED_ARMS

conds, pf, smoke = sys.argv[1], sys.argv[2], Path(sys.argv[3])
want = {"layered": (7, 4, 4, 15.0), "rev": (4, 7, 7, 18.0),
        "lowall": (11, 11, 11, 15.0)}
for arm in GENERATE_ARMS:
    spec = LAYERED_ARMS[arm]["spec"]
    out = smoke / f"plan_{arm}"
    r = subprocess.run([sys.executable, "scripts/nda/generate_layered_decode.py",
                        "--cond-npys", conds, "--branch-spec", spec,
                        "--prompts-json", pf, "--output-dir", str(out),
                        "--tag", arm, "--plan-only"],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"{arm} plan-only rc={r.returncode}\n{r.stderr[-1500:]}"
    plan = json.loads((out / "plan.json").read_text())
    lv = tuple(plan["plan"]["active_levels_per_branch"])
    mass = plan["plan"]["total_mass"]
    e_lv, e_mass = want[arm][:3], want[arm][3]
    assert lv == e_lv, f"{arm}: levels {lv} != {e_lv}"
    assert abs(mass - e_mass) < 1e-9, f"{arm}: mass {mass} != {e_mass}"
    print(f"[ok] {arm:<8} levels={lv} mass={mass:.6f} "
          f"uniform_equiv={plan['plan']['uniform_equivalent_scale']:.6f}")

# NEGATIVE CONTROL: a bad mode must fail loudly instead of collapsing to a
# zero-scale no-op arm that would render "clean" images and a fake null.
r = subprocess.run([sys.executable, "scripts/nda/generate_layered_decode.py",
                    "--cond-npys", conds, "--branch-spec", "bogus:1.0,early:1.0,early:1.0",
                    "--prompts-json", pf, "--output-dir", str(smoke / "bad"),
                    "--tag", "bad", "--plan-only"], capture_output=True, text=True)
assert r.returncode != 0, "a bogus mode was ACCEPTED"
assert "mode must be one of" in (r.stdout + r.stderr), (r.stdout + r.stderr)[-500:]
print("[ok] bogus mode rejected loudly (no silent zero-scale arm)")

# NEGATIVE CONTROL: lowall must be mass-matched to layered, or the strongest
# alternative explanation for a layered win ("just less conditioning") is untested.
m_lay = json.loads((smoke / "plan_layered/plan.json").read_text())["plan"]["total_mass"]
m_low = json.loads((smoke / "plan_lowall/plan.json").read_text())["plan"]["total_mass"]
assert abs(m_lay - m_low) < 1e-9, f"matched-mass control broken: {m_lay} vs {m_low}"
print(f"[ok] matched-mass control holds exactly: lowall == layered == {m_lay:.6f}")
PY
[[ $? -ne 0 ]] && { echo "[FATAL] plan resolution smoke failed"; exit 1; }

say "3/7 orchestrator cannot drift from the source of truth"
bash -n scripts/nda/run_layered_sub08.sh || { echo "[FATAL] run script syntax"; exit 1; }
bash -n slurm/layered_s08.sbatch || { echo "[FATAL] sbatch syntax"; exit 1; }
for a in layered rev lowall; do
  grep -q "spec_of ${a}" scripts/nda/run_layered_sub08.sh \
    || { echo "[FATAL] run script does not pull ${a}'s spec from layered_arms"; exit 1; }
done
grep -q -- "--layer-report" scripts/nda/run_layered_sub08.sh \
  || { echo "[FATAL] run script does not persist the verified assignment"; exit 1; }
grep -q -- "--layer-sanity" scripts/nda/run_layered_sub08.sh \
  || { echo "[FATAL] run script does not run the pixel-diff no-op guard"; exit 1; }
grep -q "LAYER_SPEC_ACTIVE" scripts/nda/run_layered_sub08.sh \
  || { echo "[FATAL] run script does not gate on the no-op guard verdict"; exit 1; }
echo "[ok] specs come from layered_arms; assignment + no-op guards are wired"

say "3b/7 every flag the run script passes actually exists (this cost a job once)"
"${PYTHON}" - <<'PY' || exit 1
import re, subprocess, sys

lines = open("scripts/nda/run_layered_sub08.sh").read().splitlines()
used, blocks = set(), 0
for i, ln in enumerate(lines):
    if "generate_layered_decode.py" not in ln:
        continue
    blocks += 1
    buf, j = [ln], i
    while j + 1 < len(lines) and lines[j].rstrip().endswith("\\"):
        j += 1
        buf.append(lines[j].strip())
    used |= set(re.findall(r"--[a-z0-9-]+", " ".join(buf)))
assert blocks >= 4, f"expected >=4 generator invocations, parsed {blocks}"

h = subprocess.run([sys.executable, "scripts/nda/generate_layered_decode.py", "--help"],
                   capture_output=True, text=True)
assert h.returncode == 0, h.stderr[-800:]
known = set(re.findall(r"--[a-z0-9-]+", h.stdout))
missing = sorted(used - known)
assert not missing, (f"run script passes flags the generator does not accept: {missing}\n"
                     f"generator accepts: {sorted(known)}")
print(f"[ok] {blocks} invocations, {len(used)} distinct flags, all accepted by the generator")
PY

say "4/7 device audit"
"${PYTHON}" scripts/nda/device_audit.py scripts/nda/generate_layered_decode.py \
  || { echo "[FATAL] device audit failed"; exit 1; }

say "5/7 summary refuses to emit verdicts before the artifacts exist"
"${PYTHON}" scripts/nda/layered_summary.py --root "${SMOKE}/empty" > "${SMOKE}/sum.log" 2>&1
rc=$?
if [[ "${rc}" -eq 0 ]]; then
  echo "[FATAL] summary emitted verdicts with NO artifacts present"; exit 1
fi
grep -q "refusing to emit verdicts" "${SMOKE}/sum.log" \
  || { echo "[FATAL] summary failed for the wrong reason:"; cat "${SMOKE}/sum.log"; exit 1; }
echo "[ok] summary gates on artifacts (rc=${rc}) and names the missing inputs"

say "6/7 DUPLICATE-JOB GUARD (two writers on one output dir corrupted round 1)"
if squeue -u "${USER}" -h -o "%j" 2>/dev/null | grep -qx "layered_s08"; then
  echo "[FATAL] a layered_s08 job is already queued or running against the same"
  echo "        output directory. Cancel it first (scancel <id>) or wait."
  exit 1
fi
echo "[ok] no layered_s08 in queue"

say "SUBMIT"
mkdir -p outputs/slurm "${LAYERED_OUT}/logs"
JOB=$(sbatch --parsable slurm/layered_s08.sbatch)
echo "[submitted] job ${JOB}"
echo "${JOB}" > "${LAYERED_OUT}/job_id.txt"
squeue -j "${JOB}" || true
