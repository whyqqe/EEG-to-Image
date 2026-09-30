#!/usr/bin/env bash
#SBATCH --job-name=epd-da2-oracle
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-da2-oracle-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-da2-oracle-%j.err
#
# THE ATTRIBUTION CONTROL for `epd_da2_depth8`.
#
# Result it is testing: `depth_cn070` scored 0.086 BELOW `sem_only` on the official
# CLIP 2-way (paired bootstrap CI [-0.125, -0.047]), and the exported EEG depth
# condition turned out to sit within r = +0.018 per-sample of the constant fit-set
# mean field. Two very different stories explain that:
#
#   (a) the EEG depth map is too close to a constant to be useful, and a GOOD depth
#       map would help            -> the structural tower is worth more work
#   (b) ANY depth map at this ControlNet scale overrides the IP-Adapter condition and
#       costs semantics           -> the structural branch is mis-routed and no
#                                    amount of tower work will fix it
#
# This job separates them with one variable. The semantic condition, the CN scale, the
# seed (42), the step count and the whole generation stack are held fixed; the ONLY
# change is that the depth condition is the ground-truth depth of the test image
# (`scripts/epd/build_oracle_depth_cond.py`, r = +0.9215 per-sample against the target
# versus the EEG map's +0.6720) instead of the EEG prediction.
#
# Reading it, before the metrics come in:
#   oracle_cn070 << sem_only   -> (b). The route is the problem. Stop tuning the tower.
#   oracle_cn070 ~= sem_only   -> the depth CN is inert here; the structural branch
#                                 cannot help by this route even when its map is right.
#   oracle_cn070 >  sem_only   -> (a). The map was the problem and the ceiling is real,
#                                 which prices how much tower work is worth.
#
# `oracle_cn035` is included because the EEG arm was scale-dependent: a no-op at 0.35
# and harmful at 0.70. If the oracle behaves the same way, the scale (not the map) sets
# the damage and the comparison at 0.70 is the informative one.
#
# Only generation and metrics run; nothing is trained and nothing is exported.
set -uo pipefail

ROOT="${ROOT:-/project/peilab/why/eeg-retrieval}"
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${ROOT}"

export PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
export TOKENIZERS_PARALLELISM=false
export SLURM_CPU_BIND=none

mkdir -p outputs/slurm outputs/logs

EXP="${ROOT}/outputs/sub08/epd_da2_depth8_export"
GEN="${ROOT}/outputs/sub08/epd_da2_depth8_oracle_gen"
MET="${ROOT}/outputs/sub08/epd_da2_depth8_oracle_metrics"
COND="${EXP}/spatial/cond_depth_oracle_g1"
IP="${EXP}/conds/ip_deploy_test.npy"

echo "########## oracle control ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds) ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

[[ -f "${COND}/199.png" ]] || { echo "[FATAL] no oracle condition under ${COND}"; exit 2; }
[[ -f "${IP}" ]] || { echo "[FATAL] no IP condition at ${IP}"; exit 2; }
[[ -f "${EXP}/conds/ip_noise_test.npy" ]] || { echo "[FATAL] no noise IP condition"; exit 2; }

mkdir -p "${GEN}" "${MET}"

gen_arm() {
  local arm="$1" cn="$2" iparm="$3"
  # The generator writes to `<output-dir>/generated`, so the PNGs live one level
  # below. Checked here as well as in the metric loop because a wrong guard is
  # silent: it makes a completed arm look absent and the run ends with an empty
  # table instead of an error.
  #
  # `--output-dir` must therefore be `<arm>/generated`, NOT `gdir`. An earlier
  # revision passed `gdir`, which made the generator write to
  # `<arm>/generated/generated/generated` while every guard read
  # `<arm>/generated/generated` -- the job burned two full generation stages and
  # then died at the metric loop with "no 200 PNGs". The assertion below is the
  # guard against reintroducing that: the two paths are derived, not restated.
  local gdir="${GEN}/${arm}/generated/generated"
  local gen_out="${GEN}/${arm}/generated"
  [[ "${gdir}" == "${gen_out}/generated" ]] || {
    echo "[FATAL] gdir ${gdir} is not <output-dir>/generated for output-dir ${gen_out}"; exit 2; }
  if [[ -f "${gdir}/199.png" ]]; then
    echo "[gen ] ${arm}: present, skipped"
    return 0
  fi
  echo "[gen ] ${arm}: txt2img cn=${cn} ip=${iparm} cond=ORACLE-GT-DEPTH @ $(date -Iseconds)"
  "${PYTHON}" -u "${NB_ROOT}/scripts/nda/generate_struct_inject_decode.py" \
    --mode txt2img \
    --embed-npy "${EXP}/conds/ip_${iparm}_test.npy" \
    --cond-dir "${COND}" \
    --output-dir "${gen_out}" \
    --tag "${arm}" \
    --control-type depth \
    --cn-scale "${cn}" \
    --ip-scale 1.0 \
    --control-guidance-start 0.0 \
    --control-guidance-end 0.5 \
    --gen-steps 28 \
    --gen-guidance 5.0 \
    --gen-size 512 \
    --seed 42 \
    --skip-metrics || { echo "[FATAL] generation failed for ${arm}"; exit 1; }
}

gen_arm "oracle_cn070" 0.70 deploy
gen_arm "oracle_cn035" 0.35 deploy
# The same oracle map driven by the noise IP embedding: separates "the oracle map
# helps because it is a correct scene layout" from "the oracle map helps because the
# deploy embedding was in it all along". Without this, an oracle gain is ambiguous.
gen_arm "oracle_null_cn070" 0.70 noise

ARMS="oracle_cn070 oracle_cn035 oracle_null_cn070"

for arm in ${ARMS}; do
  gdir="${GEN}/${arm}/generated/generated"
  [[ -f "${gdir}/199.png" ]] || { echo "[FATAL] ${arm}: no 200 PNGs under ${gdir}"; exit 1; }
  outj="${MET}/${arm}_seven.json"
  [[ -f "${outj}" ]] && { echo "[met ] ${arm}: present, skipped"; continue; }
  echo "[met ] ${arm} @ $(date -Iseconds)"
  "${PYTHON}" -u "${NB_ROOT}/scripts/nda/eval_official_seven_dir.py" \
    --gen-dir "${gdir}" \
    --output-json "${outj}" \
    --tag "${arm}" \
    --images-root /project/peilab/why/data/images_set \
    --device cuda:0 \
    --batch-size 16 || { echo "[FATAL] metrics failed for ${arm}"; exit 1; }
done

# The summary, with the arms it has to be read against. `sem_only` is the reference
# the whole comparison is anchored on and comes from the main run's metrics.
echo
echo "## Oracle control, official (Pearson) seven metrics"
"${PYTHON}" - <<PY
import json
from pathlib import Path
met_main = Path("${ROOT}/outputs/sub08/epd_da2_depth8_metrics")
met_orac = Path("${MET}")
rows = {}
for p in list(met_main.glob("*_seven.json")) + list(met_orac.glob("*_seven.json")):
    rows[p.stem.replace("_seven", "")] = json.loads(p.read_text(encoding="utf-8"))
order = ["sem_only", "depth_cn035", "depth_cn070", "null_txt2img", "null_cn070",
         "oracle_cn035", "oracle_cn070", "oracle_null_cn070"]
keys = [("pixcorr","PixCorr↑"),("ssim","SSIM↑"),("alex2","AlexNet(2)↑"),
        ("alex5","AlexNet(5)↑"),("inception","Inception↑"),("clip","CLIP↑"),
        ("swav","SwAV↓"),("fid","FID↓")]
print("| arm | " + " | ".join(n for _, n in keys) + " |")
print("|---|" + "---:|" * len(keys))
for arm in order:
    if arm not in rows:
        continue
    r = rows[arm]
    print(f"| {arm} | " + " | ".join(
        "—" if r.get(k) is None else f"{float(r[k]):.3f}" for k, _ in keys) + " |")

# The one number the job exists to produce.
def g(a, k):
    return None if a not in rows or rows[a].get(k) is None else float(rows[a][k])
print()
for k, nm in (("clip", "CLIP"), ("inception", "Inception"), ("alex5", "AlexNet(5)")):
    for arm, lab in (("oracle_cn070", "oracle CN0.70"), ("oracle_cn035", "oracle CN0.35"),
                     ("depth_cn070", "EEG CN0.70"), ("null_cn070", "null CN0.70")):
        v, s = g(arm, k), g("sem_only", k)
        if v is not None and s is not None:
            print(f"- {nm:12s} {lab:14s} vs sem_only: {v - s:+.3f}")
print()
print("Interpretation key (from the sbatch header): oracle_cn070 below sem_only means")
print("the depth ControlNet route costs semantics at ANY depth map; above means the")
print("EEG map was the limiter and the structural ceiling is real but unreached.")
PY

echo "########## oracle control rc=$? @ $(date -Iseconds) ##########"
