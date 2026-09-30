#!/usr/bin/env bash
set -euo pipefail
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
mkdir -p outputs/slurm outputs/ack_s08r/logs

echo "===== preflight ====="
fail=0
chk() { if [[ -e "$1" ]]; then echo "  [ok]   $1"; else echo "  [MISS] $1"; fail=1; fi; }
chk scripts/nda/ack_rebuild_prompts.py
chk scripts/nda/run_ack_reprompt_sub08.sh
chk scripts/nda/ack_heads_train.py
chk slurm/ack_reprompt_sub08.sbatch
chk outputs/ack_s08/heads/conds/ip_q_test.npy
chk outputs/ack_s08/heads/conds/ip_ack_test.npy
chk outputs/ack_s08/heads/proto/mu_all.npy
chk outputs/uck/sub-08/full/conds/ip_mem_test.npy
chk outputs/uck/sub-08/full/conds/ip_q_test.npy
chk outputs/uck/sub-08/full/spatial/pred_depth_rgb_512/199.png
chk outputs/gem/cond_cache/clip_img1024_test.npy
chk outputs/gem/cond_cache/clip_img1024_train.npy
chk outputs/g2f/prompts/prompts_oracle.json
bash -n scripts/nda/run_ack_reprompt_sub08.sh || fail=1
bash -n scripts/nda/submit_ack_reprompt_sub08.sh || fail=1
python3 -m py_compile scripts/nda/ack_rebuild_prompts.py || fail=1
if (( fail )); then echo "[FATAL] preflight failed"; exit 1; fi

echo "===== CPU rebuild smoke ====="
python3 scripts/nda/ack_rebuild_prompts.py \
  --heads-dir outputs/ack_s08/heads \
  --out-prompts outputs/ack_s08r/prompts \
  --uck-q outputs/uck/sub-08/full/conds/ip_q_test.npy \
  --mu-npy outputs/ack_s08/heads/proto/mu_all.npy \
  --gate-margin 0.02 \
  --report outputs/ack_s08r/rebuild_report.json \
  | tee outputs/ack_s08r/logs/rebuild_smoke.txt | tail -n 35
# copy report into prompts dir so run script SKIP works consistently
cp -f outputs/ack_s08r/rebuild_report.json outputs/ack_s08r/prompts/rebuild_report.json
python3 - <<'PY'
import json
d=json.load(open("outputs/ack_s08r/rebuild_report.json"))
assert d["test200_top1_q"] >= 0.05, d
print(f"  [ok]   top1_q={d['test200_top1_q']:.3f} gated={d['n_gated_to_object']} "
      f"correct={d['n_correct_pred']}")
PY

echo "===== submit ====="
EXCL="${ACK_EXCLUDE:-dgx-09}"
JID="$(sbatch --parsable --exclude="${EXCL}" slurm/ack_reprompt_sub08.sbatch)"
echo "JOBID=${JID}  (excluded: ${EXCL})"
echo "  log: outputs/slurm/ack_s08r_${JID}.out"
echo "  out: outputs/ack_s08r"
echo "  fix: naming = q·G_test (+ uck_q ablation); v1 kept in outputs/ack_s08"
