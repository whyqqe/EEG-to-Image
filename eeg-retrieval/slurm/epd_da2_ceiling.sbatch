#!/usr/bin/env bash
#SBATCH --job-name=epd-da2-ceiling
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-da2-ceiling-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/outputs/slurm/epd-da2-ceiling-%j.err
#
# THE CEILING ARMS: what this generation stack scores when its conditions are RIGHT.
#
# The five arms of the main run bracket the pipeline on the floor side only. The
# `null_txt2img` arm (IP built from zeroed EEG) sits at CLIP 0.4962 / AlexNet(2)
# 0.4981, i.e. exactly chance, which proves the EEG is doing all the work -- but it
# does not say how much work there was to do. Without a ceiling, "sem_only 0.786" is
# unreadable: it could mean the semantic tower is nearly optimal, or that the decoder
# discards most of what the tower provides. Those two diagnoses call for opposite
# work, and the experiment as designed cannot separate them.
#
# The ceiling is built from ground truth rather than from a better model, so it does
# not depend on any claim about how good the towers are:
#
#   ip_oracle         IP = the test image's OWN CLIP ViT-H-14 joint embedding
#                     (`build_oracle_ip_cond.py`), CN 0.  The semantic ceiling.
#   ip_oracle_cn070   the same oracle IP plus the ORACLE (ground-truth) depth map at
#                     CN 0.70.  The joint ceiling, and the arm that settles whether
#                     the depth-ControlNet route is harmful even with a perfect map:
#                     if ip_oracle_cn070 < ip_oracle, no structural work can pay off
#                     through this route and the branch has to be re-routed.
#
# Read against the arms already on disk:
#
#   null_txt2img      0.496   the floor (no EEG anywhere)
#   sem_only          0.786   the deployed semantic condition
#   ip_oracle         ???     the semantic ceiling
#   oracle_cn070      ???     GT depth through the deployed route
#   ip_oracle_cn070   ???     the combination
#
# The three numbers that decide the architecture:
#   (ip_oracle      - sem_only)      how much of the semantic signal is still on the
#                                    table -> is the semantic tower the bottleneck?
#   (ip_oracle_cn070 - ip_oracle)    does the depth route cost semantics even when the
#                                    map is perfect -> is the branch salvageable?
#   (oracle_cn070   - sem_only)      already measured by the sibling job: does a
#                                    perfect map help the DEPLOYED semantic condition
#
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
GEN="${ROOT}/outputs/sub08/epd_da2_depth8_ceiling_gen"
MET="${ROOT}/outputs/sub08/epd_da2_depth8_ceiling_metrics"
COND_EEG="${EXP}/spatial/cond_depth_test_g1"
COND_ORACLE="${EXP}/spatial/cond_depth_oracle_g1"
IP_ORACLE="${EXP}/conds/ip_oracle_test.npy"

echo "########## ceiling arms ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} @ $(date -Iseconds) ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for f in "${IP_ORACLE}" "${COND_ORACLE}/199.png" "${EXP}/conds/ip_deploy_test.npy"; do
  [[ -e "$f" ]] || { echo "[FATAL] missing prerequisite ${f}"; exit 2; }
done

mkdir -p "${GEN}" "${MET}"

gen_arm() {
  local arm="$1" cn="$2" ipnpy="$3" conddir="$4"
  local gdir="${GEN}/${arm}/generated/generated"
  local gen_out="${GEN}/${arm}/generated"
  [[ "${gdir}" == "${gen_out}/generated" ]] || {
    echo "[FATAL] gdir ${gdir} is not <output-dir>/generated for output-dir ${gen_out}"; exit 2; }
  if [[ -f "${gdir}/199.png" ]]; then
    echo "[gen ] ${arm}: present, skipped"
    return 0
  fi
  echo "[gen ] ${arm}: cn=${cn} ip=$(basename "${ipnpy}") cond=$(basename "${conddir}") @ $(date -Iseconds)"
  "${PYTHON}" -u "${NB_ROOT}/scripts/nda/generate_struct_inject_decode.py" \
    --mode txt2img \
    --embed-npy "${ipnpy}" \
    --cond-dir "${conddir}" \
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

# The ceiling pair. The first is the semantic ceiling; the second asks whether the
# structural route costs anything once BOTH of its inputs are perfect.
gen_arm "ip_oracle"       0.0  "${IP_ORACLE}" "${COND_EEG}"
gen_arm "ip_oracle_cn070" 0.70 "${IP_ORACLE}" "${COND_ORACLE}"

for arm in ip_oracle ip_oracle_cn070; do
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

echo
echo "## The full ladder (official Pearson two-way)"
"${PYTHON}" - <<PY
import json
from pathlib import Path
dirs = [
    Path("${ROOT}/outputs/sub08/epd_da2_depth8_metrics"),
    Path("${ROOT}/outputs/sub08/epd_da2_depth8_oracle_metrics"),
    Path("${MET}"),
]
rows = {}
for d in dirs:
    for p in d.glob("*_seven.json"):
        rows[p.stem.replace("_seven", "")] = json.loads(p.read_text(encoding="utf-8"))
order = ["null_txt2img", "null_cn070", "sem_only", "depth_cn035", "depth_cn070",
         "oracle_cn035", "oracle_cn070", "ip_oracle", "ip_oracle_cn070"]
keys = [("pixcorr","PixCorr"),("ssim","SSIM"),("alex2","AlexNet(2)"),("alex5","AlexNet(5)"),
        ("inception","Inception"),("clip","CLIP"),("swav","SwAV"),("fid","FID")]
print("| arm | " + " | ".join(n for _, n in keys) + " |")
print("|---|" + "---:|" * len(keys))
for a in order:
    if a in rows:
        print(f"| {a} | " + " | ".join(
            "—" if rows[a].get(k) is None else f"{float(rows[a][k]):.3f}" for k, _ in keys) + " |")

def g(a, k):
    return None if a not in rows or rows[a].get(k) is None else float(rows[a][k])

print()
print("## What the ceiling says")
print()
for k, nm in (("clip", "CLIP"), ("inception", "Inception"), ("alex5", "AlexNet(5)")):
    fl, dep, orc, ceil = g("null_txt2img", k), g("sem_only", k), g("oracle_cn070", k), g("ip_oracle", k)
    both = g("ip_oracle_cn070", k)
    if None in (fl, dep, orc, ceil):
        continue
    rng = ceil - fl
    print(f"- {nm}: floor {fl:.3f} | deployed {dep:.3f} | GT-depth {orc:.3f} | "
          f"CEILING {ceil:.3f}")
    print(f"    the deployed semantic condition captures {100*(dep-fl)/rng:.0f}% of the "
          f"range the decoder can reach ({rng:+.3f} wide); "
          f"{100*(ceil-dep)/rng:.0f}% is left on the table")
    if both is not None:
        print(f"    adding GT depth on top of the oracle IP: {both:.3f} "
              f"({both - ceil:+.3f}) -> the depth route "
              f"{'COSTS' if both < ceil - 0.01 else 'is neutral'} even with perfect input")
PY

echo "########## ceiling arms done @ $(date -Iseconds) ##########"
