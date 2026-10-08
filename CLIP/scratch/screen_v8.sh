#!/bin/bash
# Screening sweep on the BANKED v8 encoder: alpha/tau grid + score matrices for ensembling.
# Writes everything under CLIP/. Runs on whatever node executes it.
set -u
ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
cd "${ROOT}"
export PYTHONNOUSERSITE=1
export TMPDIR="${ROOT}/scratch/tmp"
export OMP_NUM_THREADS=8
export PYTHONPATH="${ROOT}/src"
mkdir -p "${TMPDIR}" outputs/eval/v8scr outputs/scores

# configs: "alpha tau"
for cfg in "0.5 0.01" "0.625 0.01" "0.875 0.01" "0.75 0.03" "0.75 0.01"; do
  set -- ${cfg}
  A="$1"; T="$2"
  EV="${ROOT}/outputs/eval/v8scr/a${A}_t${T}"
  SC="${ROOT}/outputs/scores/v8scr_a${A}_t${T}"
  mkdir -p "${EV}" "${SC}"
  for s in 1 2 3 4 5 6 7 8 9 10; do
    tag=$(printf 'sub%02d' "${s}")
    if [ "${s}" -eq 8 ]; then REF="--ref-top1 22.00 --ref-top5 50.00"; else REF="--ref-top1 unknown"; fi
    for seed in 2025 2026 2027; do
      ck="${ROOT}/outputs/stage1/v8/${tag}_k20_seed${seed}/last.pt"
      ev="${EV}/${tag}_seed${seed}.json"
      sc="${SC}/${tag}_seed${seed}.npz"
      [ -s "${ck}" ] || continue
      [ -s "${ev}" ] && [ -s "${sc}" ] && continue
      "${PY}" scripts/run_eval.py --ckpts "${ck}" \
        --target-subject "${s}" --out "${ev}" \
        --recovery --reps --csls-k 10 --rho 0.1 \
        --recovery-operator fgw --recovery-alpha "${A}" --recovery-tau "${T}" \
        --save-scores "${sc}" ${REF} >/dev/null 2>&1 \
        && echo "[ok] a=${A} t=${T} ${tag}/seed${seed}" \
        || echo "[FAIL] a=${A} t=${T} ${tag}/seed${seed}"
    done
  done
done
echo "SCREENING DONE"
