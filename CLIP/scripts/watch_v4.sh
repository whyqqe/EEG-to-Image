#!/bin/bash
# Wait for the v4 pipeline to drain, then write a summary next to the logs.
#
# Why a script and not "read the logs in the morning": the jobs run for hours with no
# supervision, and the ONE thing that has to survive is which configuration produced
# which number. The per-seed numbers are re-read from the eval JSON (not from the Slurm
# logs) so the summary cannot disagree with the artefact, and `summarize_arms.py` prints
# the seeds' SPREAD next to the mean -- at n=200 one trial is 0.5pp, so a mean without
# its spread is how a one-trial difference gets reported as an improvement.
#
# Run detached:  nohup bash scripts/watch_v4.sh > outputs/watch_v4.log 2>&1 &
ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
TAG="${1:-v4-sub-08}"
SEEDS="${2:-2025 2026 2027}"

cd "${ROOT}"

echo "[watch] started $(date -Is) for tag=${TAG} seeds=${SEEDS}"
# Poll instead of `squeue --wait`: --wait returns on the FIRST job that exits, which for
# a 7-job DAG means it returns while three trainings are still running.
while squeue -u "${USER}" -h -o "%j" 2>/dev/null | grep -q "samclip"; do
    sleep 120
done
echo "[watch] queue drained at $(date -Is)"

# ---- did every stage actually produce an artefact? ------------------------------
echo
echo "=============== exit status ==============="
for s in ${SEEDS}; do
    d="${ROOT}/outputs/stage1/${TAG}_seed${s}"
    echo "-- seed ${s}"
    if [[ -f "${d}/last.pt" ]]; then
        echo "   ckpt     OK   $(stat -c '%y  %s bytes' "${d}/last.pt")"
    else
        echo "   ckpt     MISSING (${d}/last.pt) -- check outputs/slurm/samclip-s1-*.err"
    fi
    if [[ -f "${ROOT}/outputs/eval/${TAG}_seed${s}.json" ]]; then
        echo "   eval     OK   ${TAG}_seed${s}.json"
    else
        echo "   eval     MISSING -- check outputs/slurm/samclip-eval-*.err"
    fi
done

# ---- the run record, straight out of the checkpoint -----------------------------
echo
echo "=============== what was actually trained ==============="
"${PY}" - <<PY
import json, torch
from pathlib import Path
for s in "${SEEDS}".split():
    p = Path("${ROOT}/outputs/stage1/${TAG}_seed$s/config.json")
    if not p.is_file():
        print(f"seed {s}: no config.json"); continue
    cfg = json.loads(p.read_text())
    ck = Path("${ROOT}/outputs/stage1/${TAG}_seed$s/last.pt")
    ep = torch.load(ck, map_location="cpu", weights_only=False).get("epoch") if ck.is_file() else "?"
    print(f"seed {s}: arch={cfg.get('arch')} objective={cfg.get('objective')} "
          f"d_latent={cfg.get('d_latent')} d_align={cfg.get('d_align')} "
          f"share_head_hidden={cfg.get('share_head_hidden')} epochs={cfg.get('epochs')} "
          f"final_epoch={ep} lr={cfg.get('lr')} dropout={cfg.get('dropout')}")
    print(f"         smn={cfg.get('smn')}")
    print(f"         loss_weights={cfg.get('loss_weights')}")
    print(f"         schedule={cfg.get('schedule')}")
PY

# ---- the numbers -----------------------------------------------------------------
echo
echo "=============== arm table (mean +/- seed std) ==============="
"${PY}" scripts/summarize_arms.py --glob "${TAG}*" 2>&1

echo
echo "=============== per-seed ladder ==============="
"${PY}" - <<PY
import json, glob
from pathlib import Path
for f in sorted(glob.glob("${ROOT}/outputs/eval/${TAG}*.json")):
    d = json.loads(Path(f).read_text())
    for name, c in (d.get("checkpoints") or {}).items():
        print(f"{Path(f).stem}  (epoch {c.get('epoch')}, fusion={c.get('target_fusion')})")
        print(f"    smn_gate={c.get('smn_gate')}  offset_ratio={c.get('offset_ratio')}  "
              f"spec_top_frac={c.get('spec_top_frac')}")
        for r, v in (c.get("rows") or {}).items():
            print(f"    {r:<18} top1 {v.get('top1'):>6.2f}  top5 {v.get('top5'):>6.2f}  "
                  f"mean_rank {v.get('mean_rank'):>7.2f}")
print()
print("[reference] SAMGA official, this fold, final epoch: 22.00 / 50.00")
PY
echo "[watch] done $(date -Is)"
