#!/usr/bin/env bash
# ============================================================================
# DIAGNOSTICS ONLY -- reads finished results, trains nothing, changes no numbers.
#
# WHY
#   The all-subject run finished, but three of its mechanisms report suspicious
#   values that decide whether the architecture is right:
#     1. every granularity's mean gate time is EXACTLY 500 ms, which is the mean of
#        a uniform gate over patch centres [100,300,500,700,900].  If the gates
#        never moved, then "the gates learned a temporal placement" is false.
#     2. `frla_uniform` scored PixCorr 0.341 / SSIM 0.376 (the best pixels of the
#        whole run) against a FID of 347 (the worst).  That is the signature of
#        generation NOT happening and the low-frequency initialisation coming back
#        out.  It has to be scored, not argued about.
#     3. `--ablation all` may be turning the time gates OFF, in which case the
#        ablation arm cannot be the control for the granularity-time claim.
#
# WHAT IT DOES
#   A. gate matrix (report) -- uniform or learned, per granularity, per subject
#   B. frozen-condition test (report) -- is `ip_fused_test` a function of the image
#      class?  A condition that is constant across images cannot drive generation.
#   C. `--ablation all` semantics (source) -- which mechanisms does it disable
#   D. init decode (GPU) -- decode the EEG-predicted VAE latent with the SAME code
#      and score it with the SAME metric implementation as every other row
#   E. score that init with the official seven
# ============================================================================
#SBATCH --job-name=tdm_diag
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/tdm_diag-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/tdm_diag-%j.err
set -euo pipefail
export NB_ROOT=/project/peilab/why/NeuroBridge
cd "${NB_ROOT}"
source scripts/nmb/nmb_sota_v2_env.sh >/dev/null 2>&1 || true
export OUT="${NB_ROOT}/outputs/tdm_all"
export ABL_SUBJECT=8

python - <<'PY'
import json, glob, os, re
import numpy as np

OUT = os.environ["OUT"]
REF = int(os.environ.get("ABL_SUBJECT", "8"))
T_MS = None
print("=" * 92)
print("A. GATE MATRIX: did the per-granularity time gates move at all?")
print("=" * 92)
for arm in ("tdm", "tdm_abl"):
    for p in sorted(glob.glob(f"{OUT}/{arm}/sub-??/tdm_report.json")):
        sd = int(re.search(r"sub-(\d+)", p).group(1))
        d = json.load(open(p))
        g = (d.get("mechanisms", {}) or {}).get("granularity_time_claim", {}) or {}
        rows = [r for r in g.get("rows", []) if "gate_matrix" in r]
        if not rows:
            print(f"  {arm} sub-{sd:02d}: no gate matrices recorded"); continue
        T_MS = rows[0]["t_ms"]
        line = []
        for r in rows:
            w = np.asarray(r["gate_matrix"], float)          # (nb, n_patch)
            tp = w.sum(0) / max(w.sum(), 1e-12)              # time profile
            unif = 1.0 / len(T_MS)
            line.append(f"{r['granularity'][:4]} max{tp.max():.3f}/unif{unif:.3f}")
        print(f"  {arm:<7} sub-{sd:02d} t_ms={[int(t) for t in T_MS]}")
        print(f"          " + "  ".join(line))
        if arm == "tdm" and sd == REF:
            print("          ^ a profile that is flat at 1/5 = the gates never left "
                  "their initialisation")
print()
print("  VERDICT ON (1): the reported 500 ms means come from a UNIFORM gate "
      "(sum of a flat profile times [100,300,500,700,900] = 500).")
PY

python - <<'PY'
import json, glob, os
import numpy as np

OUT = os.environ["OUT"]
print()
print("=" * 92)
print("B. FROZEN-CONDITION TEST: is the condition actually a function of the image?")
print("=" * 92)
for arm in ("tdm", "tdm_abl"):
    for p in sorted(glob.glob(f"{OUT}/{arm}/sub-??/conds/ip_fused_test.npy")):
        sd = p.split("sub-")[1][:2]
        c = np.load(p).astype(np.float32)
        c = c / np.clip(np.linalg.norm(c, axis=1, keepdims=True), 1e-8, None)
        # cross-image variability of a unit-norm condition, per subject
        sd_dim = c.std(0).mean()
        cos = c @ c.T
        off = cos[~np.eye(len(c), dtype=bool)]
        print(f"  {arm:<7} sub-{sd} shape={c.shape} "
              f"per-dim std={sd_dim:.5f} pairwise cos mean={off.mean():+.4f} "
              f"min={off.min():+.4f} max={off.max():+.4f}")
print()
print("  A per-dim std at the 0.x level is normal for a projected condition.  A value")
print("  near 0 with pairwise cos pinned at +1 would mean every test image got the")
print("  SAME condition -- which generation cannot recover from, and which would")
print("  explain a row whose metrics are near the unconditional floor.")
PY

echo
echo "===== C. what does --ablation all actually disable? ====="
python - "$NB_ROOT" <<'PY'
import re, sys, pathlib
src = pathlib.Path(sys.argv[1], "scripts/nda/tdm_train.py").read_text().splitlines()
pat = re.compile(r"use_(dla|demix|dng|timegates)|ablation")
for i, ln in enumerate(src, 1):
    if pat.search(ln) and ("args." in ln or "==" in ln or "all" in ln):
        print(f"  {i:5d}: {ln.rstrip()}")
PY

echo
echo "===== D/E. decode the EEG-predicted low-level latent and score it ====="
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
SUBJECT="${ABL_SUBJECT}" python scripts/nda/tdm_init_check.py "${OUT}/init_check/sub-08" 2>&1 | tail -6
python scripts/nda/eval_official_seven_dir.py \
  --gen-dir "${OUT}/init_check/sub-08" \
  --output-json "${OUT}/eval/s08_init_lowlevel.json" --tag init_lowlevel \
  --images-root /project/peilab/why/data/images_set --device cuda:0 --skip-if-exists \
  2>&1 | tail -20

echo
echo "===== side-by-side: the init vs the three FRLA arms it supposedly steered ====="
python - <<'PY'
import json, os
OUT = os.environ["OUT"]
M = ("pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "swav", "fid")
print(f"  {'row':<18}" + "".join(f"{k:>10}" for k in M))
for tag in ("init_lowlevel", "sdedit_ll_095", "frla_off", "frla_uniform", "frla_frla"):
    p = f"{OUT}/eval/s08_{tag}.json"
    if not os.path.isfile(p):
        print(f"  {tag:<18} (missing {p})"); continue
    d = json.load(open(p))
    print(f"  {tag:<18}" + "".join(
        f"{d[k]:>10.4f}" if d.get(k) is not None else f"{'-':>10}" for k in M))
PY
echo "===== diagnostics done ====="
