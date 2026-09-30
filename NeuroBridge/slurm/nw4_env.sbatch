#!/bin/bash
#SBATCH --job-name=nw4_env
#SBATCH --partition=cpu
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_env_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_env_%j.err
#
# Why did `from diffusers import AutoencoderKL` start failing once
# PYTHONPATH included NB_ROOT?  nw3 decoded 200 RGBs without that export; nw4
# failed with "cannot import name 'EncoderDecoderCache' from 'transformers'".
# Run the same import twice and diff sys.path / resolved module files.

set -uo pipefail
cd /project/peilab/why/NeuroBridge
export NB_ROOT=/project/peilab/why/NeuroBridge
PY=python

probe() {
  "$PY" - <<'PY'
import sys, importlib
print("  PYTHONPATH =", repr(__import__("os").environ.get("PYTHONPATH", "")))
print("  sys.path[:6] =", sys.path[:6])
for mod in ("transformers", "diffusers"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:<13} {getattr(m,'__version__','?')}  {getattr(m,'__file__','?')}")
    except Exception as e:
        print(f"  {mod:<13} IMPORT FAILED: {type(e).__name__}: {e}")
try:
    from transformers import EncoderDecoderCache  # noqa: F401
    print("  EncoderDecoderCache: OK")
except Exception as e:
    print(f"  EncoderDecoderCache: MISSING ({type(e).__name__})")
try:
    from diffusers import AutoencoderKL  # noqa: F401
    print("  diffusers.AutoencoderKL: OK")
except Exception as e:
    print(f"  diffusers.AutoencoderKL: FAILED -> {type(e).__name__}")
PY
}

echo "=== A. clean env (what nw3 did) ==="
unset PYTHONPATH
probe

echo
echo "=== B. with PYTHONPATH=scripts/nda:NB_ROOT (what nw4 did) ==="
export PYTHONPATH="${NB_ROOT}/scripts/nda:${NB_ROOT}"
probe

echo
echo "=== C. PYTHONPATH=NB_ROOT only ==="
export PYTHONPATH="${NB_ROOT}"
probe

echo
echo "=== D. PYTHONPATH=scripts/nda only ==="
export PYTHONPATH="${NB_ROOT}/scripts/nda"
probe

echo
echo "=== E. does NB_ROOT shadow anything importable? ==="
for name in transformers diffusers torch numpy typing json; do
  for d in "${NB_ROOT}" "${NB_ROOT}/scripts/nda"; do
    if [[ -e "${d}/${name}.py" || -d "${d}/${name}" ]]; then
      echo "  SHADOW: ${d}/${name}"
    fi
  done
done
echo "  (no SHADOW lines above = nothing shadows those top-level names)"
