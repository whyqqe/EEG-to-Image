#!/usr/bin/env bash
# =============================================================================
# epd_lowlevel -- the img2img strength frontier on sub-08
#
# Why THIS is the experiment the project needs next
# -------------------------------------------------
# The architecture question was "semantic vs structural", and it was being asked
# with the wrong instrument. Re-deriving where the deficit actually is, from the
# project's own recorded numbers (REPORT.md:375-379) and this run's arms:
#
#                                 PixCorr   SSIM   CLIP
#   sem_only / txt2img              0.084   0.327  0.786     <- us
#   deploy_sdedit  (init, s=0.80)   0.109   0.323  0.781     <- us
#   ATM, sub-08                    ~0.160   0.345  0.786     <- the paper to beat
#   ENIGMA (SOTA), sub-08           0.167   0.426   ...
#
# We already MATCH ATM on CLIP (0.786 vs 0.786) and lose the axis that decides the
# comparison: PixCorr, by a factor of two. Every "improve the semantic tower" arm
# therefore buys headroom on an axis we are not losing, while the axis we ARE
# losing has an untested knob. This run spends its GPU on that knob.
#
# The knob, and why it is untested rather than rejected
# ----------------------------------------------------
# `deploy_sdedit` initialises the denoiser from the VAE-decoded prediction of the
# structural tower, and it was only ever run at `--strength 0.80`. The two ENDS of
# that axis are already known, and one of them is not a generation:
#
#   strength -> 1   the init is fully noised away = the txt2img arm   PixCorr 0.088
#   strength 0.80   measured                                          PixCorr 0.109
#   strength -> 0   the output IS the init     r(init, GT) = +0.2191  (see below)
#
# That third number is the point of this script. `spatial/pred_lowlevel_rgb_512` is
# on disk, so its PixCorr against the ground truth can be computed with PIL and
# numpy at no GPU cost, in the official protocol (RGB @ 425 BILINEAR, Pearson of
# the flattened arrays):
#
#     measured   PixCorr(pred_lowlevel_rgb_512, GT) = +0.2191   (dino3 tower)
#                PixCorr(patch_dual lowlevel,    GT) = +0.1568
#                PixCorr(pred_depth_rgb_512,     GT) = -0.0952   <- anti-correlated
#                PixCorr(random training image,  GT) = +0.0420   <- the floor
#
# So the init carries MORE low-level information than ATM's whole reported PixCorr
# (0.219 > 0.160), and the pipeline at s=0.80 converts only 0.109 of it. The
# strength knob is therefore not marginal: it is the difference between 21.9 points
# of available signal and the ~12.1 the shipped configuration extracts. The region
# between 0 and 0.80 is completely unmapped, and mapping it costs generation time
# only -- no training, no new model, no new cache.
#
# The depth finding, restated in this frame
# -----------------------------------------
# `pred_depth_rgb_512` is ANTI-correlated with the ground truth (-0.095). That is
# the cleanest available explanation of the previous conclusion, which was reached
# from the other direction: a ControlNet-depth condition built from the EEG is inert
# at CN 0.35 and harmful at CN 0.70, and a ground-truth depth map through the same
# route is no better. A depth map is a good GEOMETRIC constraint and a terrible
# LOW-LEVEL image: as an img2img init it is worse than noise. The `ll_s050_depthinit`
# arm below turns that into a measurement rather than an interpretation.
#
# The arms
# --------
# Every arm shares one IP condition, one init, one generator, one seed. The only
# variable is `--strength`, except for the two controls, which hold strength at 0.50
# and change the INPUT instead:
#
#   ll_txt2img         txt2img, no init              FREE re-score of deploy_txt2img
#   ll_s020..ll_s065   img2img, strength .20 .35 .50 .65     NEW generations
#   ll_s080            img2img, strength .80         FREE re-score of deploy_sdedit
#   ll_s050_noiseip    strength 0.50, IP built from ZERO EEG
#   ll_s050_depthinit  strength 0.50, init = pred_depth_rgb_512 (the bad init)
#
# The two free arms are re-SCORED rather than regenerated because the metric script
# has changed since they were made: their `*_seven.json` carries no `statistic`
# field, i.e. they predate the switch to the official Pearson similarity, and
# reading them against the new arms would compare two different statistics. Their
# PNGs are deterministic given the seed, so re-scoring them costs about a minute and
# puts all eight arms on one axis.
#
# What the controls buy
# ---------------------
# `ll_s050_noiseip` consumes no EEG at all. At strength 0.80 it scored PixCorr 0.105
# against `deploy_sdedit`'s 0.109 while its CLIP collapsed from 0.781 to 0.508 -- so
# at 0.80 the PixCorr is coming from the INIT and not from the EEG. At 0.50, where
# the init has more influence, that separation should widen, and it is what tells us
# whether a low-strength arm's PixCorr is a result or an artifact.
#
# Cost
# ----
# 6 generations x ~6.5 min + 8 metric runs x ~1 min + 2 free re-scores ~= 50 min GPU.
# The sbatch asks for 2 h.
# =============================================================================
set -uo pipefail

ROOT="${ROOT:-/project/peilab/why/eeg-retrieval}"
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
PY="${PY:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
SUBJ="${SUBJ:-8}"

TAG="${TAG:-epd_lowlevel}"
OUT="${OUT:-${ROOT}/outputs/sub$(printf '%02d' "${SUBJ}")}"
GEN="${GEN:-${OUT}/${TAG}_gen}"
MET="${MET:-${OUT}/${TAG}_metrics}"

# The source of every input. This arm trains nothing: the init and the IP condition
# are the artifacts of the last dual-tower run, held fixed while strength varies.
SRC_EXP="${SRC_EXP:-${OUT}/epd_opt_dino3_eegit_export}"
SRC_GEN="${SRC_GEN:-${OUT}/epd_opt_dino3_eegit_gen}"
INIT_DIR="${INIT_DIR:-${SRC_EXP}/spatial/pred_lowlevel_rgb_512}"
DEPTH_INIT_DIR="${DEPTH_INIT_DIR:-${OUT}/patch_dual_export/spatial/pred_depth_rgb_512}"
IP_DEPLOY="${IP_DEPLOY:-${SRC_EXP}/conds/ip_deploy_test.npy}"
IP_NOISE="${IP_NOISE:-${SRC_EXP}/conds/ip_noise_test.npy}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '[FATAL] %s\n' "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# The arm table. `strength` is the only scientific variable; `kind` selects which
# input varies for the two controls.
# -----------------------------------------------------------------------------
ARMS=(ll_txt2img ll_s020 ll_s035 ll_s050 ll_s065 ll_s080 ll_s050_noiseip ll_s050_depthinit)

arm_strength() {
  case "$1" in
    ll_txt2img)         echo "" ;;      # txt2img, no init
    ll_s020)            echo "0.20" ;;
    ll_s035)            echo "0.35" ;;
    ll_s050)            echo "0.50" ;;
    ll_s065)            echo "0.65" ;;
    ll_s080)            echo "0.80" ;;
    ll_s050_noiseip)    echo "0.50" ;;
    ll_s050_depthinit)  echo "0.50" ;;
    *) die "unknown arm $1" ;;
  esac
}

# The directory holding the 200 finished PNGs for an arm. Two arms are re-scores of
# artifacts from the previous run, so their images are NOT under ${GEN}: keep the
# mapping in one function rather than restating it at each read.
arm_png_dir() {
  case "$1" in
    ll_txt2img) echo "${SRC_GEN}/deploy_txt2img/generated" ;;
    ll_s080)    echo "${SRC_GEN}/deploy_sdedit/generated" ;;
    *)          echo "${GEN}/$1/generated/generated" ;;
  esac
}

# The directory to hand `--output-dir`. The generator appends `generated`, so the
# PNGs land one level below this. Kept as a function because the previous revision of
# this family of scripts passed the DEEP path here and the PNGs went to
# `<dir>/generated/generated`, while every guard read `<dir>/generated` -- which made
# a finished arm look absent and the run end with an empty table instead of an error.
arm_out_dir() {
  case "$1" in
    ll_txt2img|ll_s080) echo "" ;;      # re-scored, never generated
    *)                  echo "${GEN}/$1/generated" ;;
  esac
}

# Everything except the variable under study. Written once so no arm can drift.
COMMON=(
  --control-type depth
  --cn-scale 0.0
  --ip-scale 1.0
  --control-guidance-start 0.0
  --control-guidance-end 0.5
  --gen-steps 28
  --gen-guidance 5.0
  --gen-size 512
  --seed 42
  --skip-metrics
)

# =============================================================================
# [1] prerequisites
# =============================================================================
log "[1] checking inputs (this arm trains nothing; it reuses the last run's artifacts)"
for f in "${INIT_DIR}/199.png" "${DEPTH_INIT_DIR}/199.png" \
         "${SRC_GEN}/deploy_txt2img/generated/199.png" \
         "${SRC_GEN}/deploy_sdedit/generated/199.png"; do
  [[ -f "$f" ]] || die "missing input image: $f"
done
for f in "${IP_DEPLOY}" "${IP_NOISE}"; do
  [[ -f "$f" ]] || die "missing IP condition: $f"
done
for f in "${NB_ROOT}/scripts/nda/generate_struct_inject_decode.py" \
         "${NB_ROOT}/scripts/nda/eval_official_seven_dir.py" \
         "${ROOT}/scripts/epd/stats.py"; do
  [[ -f "$f" ]] || die "pipeline references a script that does not exist: $f"
done
# The free re-scores are only meaningful if the generator is deterministic given the
# seed. It is invoked with `--seed 42` in both arms, and these are the artifacts that
# call produced -- asserted here so a re-scored arm cannot silently be a re-score of
# something else.
for a in ll_txt2img ll_s080; do
  d="$(arm_png_dir "$a")"
  n=$(find "$d" -maxdepth 1 -name '*.png' | wc -l)
  [[ "$n" -eq 200 ]] || die "$a: expected 200 PNGs under $d, found $n"
done
log "[1] inputs ok: init ${INIT_DIR##*/}, ${#ARMS[@]} arms"

if [[ "${1:-}" == "--dry-run" ]]; then
  log "[1] dry run: every input exists; the arm table is"
  for a in "${ARMS[@]}"; do
    printf '        %-18s strength=%-5s pngs=%s\n' "$a" "$(arm_strength "$a")" "$(arm_png_dir "$a")"
  done
  exit 0
fi

mkdir -p "${GEN}" "${MET}"

# =============================================================================
# [2] generate the six new arms
# =============================================================================
# `SKIP_GEN`/`SKIP_METRICS` exist so the summary block can be exercised against an
# already-scored metrics directory without a GPU -- which is how the analysis was
# tested before this job was submitted, rather than by submitting it and reading the
# traceback in the log.
gen_arm() {
  local arm="$1" sval="$2" ipnpy="$3" initdir="$4"
  local pdir outdir; pdir="$(arm_png_dir "$arm")"; outdir="$(arm_out_dir "$arm")"
  # Derived, not restated: the guard and the generator's output must be the same
  # place, and this is the assertion that keeps them so.
  [[ "${pdir}" == "${outdir}/generated" ]] || {
    echo "[FATAL] ${arm}: guard ${pdir} is not <output-dir>/generated for ${outdir}"; exit 2; }

  if [[ -f "${pdir}/199.png" ]]; then
    log "[2] ${arm}: 200 images present, skipping generation"
    return 0
  fi
  log "[2] ${arm}: img2img strength=${sval} init=${initdir##*/} ip=${ipnpy##*/}"
  "${PY}" -u "${NB_ROOT}/scripts/nda/generate_struct_inject_decode.py" \
    --mode img2img \
    --embed-npy "${ipnpy}" \
    --cond-dir "${initdir}" \
    --init-dir "${initdir}" \
    --strength "${sval}" \
    --output-dir "${outdir}" \
    --tag "${arm}" \
    "${COMMON[@]}" || die "generation failed for arm ${arm}"
  [[ -f "${pdir}/199.png" ]] || die "${arm}: expected 200 PNGs under ${pdir} after generation"
}

if [[ "${SKIP_GEN:-0}" == "1" ]]; then
  log "[2] SKIP_GEN=1: not generating"
else
  for arm in "${ARMS[@]}"; do
    case "${arm}" in
      ll_txt2img|ll_s080) : ;;   # pre-existing artifacts, re-scored in [3]
      ll_s050_noiseip)    gen_arm "${arm}" "$(arm_strength "${arm}")" "${IP_NOISE}" "${INIT_DIR}" ;;
      ll_s050_depthinit)  gen_arm "${arm}" "$(arm_strength "${arm}")" "${IP_DEPLOY}" "${DEPTH_INIT_DIR}" ;;
      *)                  gen_arm "${arm}" "$(arm_strength "${arm}")" "${IP_DEPLOY}" "${INIT_DIR}" ;;
    esac
  done
fi

# =============================================================================
# [3] seven metrics for every arm, on one axis
# =============================================================================
log "[3] seven metrics (Pearson two-way, the official protocol)"
if [[ "${SKIP_METRICS:-0}" == "1" ]]; then
  log "[3] SKIP_METRICS=1: not scoring"
fi
for arm in "${ARMS[@]}"; do
  [[ "${SKIP_METRICS:-0}" == "1" ]] && break
  pdir="$(arm_png_dir "${arm}")"
  outj="${MET}/${arm}_seven.json"
  if [[ -f "${outj}" ]]; then
    log "[3] ${arm}: metrics present, skipping"
    continue
  fi
  # A missing PNG set is a hard failure, never a skip. The previous revision of this
  # family of scripts read a path the generator did not write to, so every arm
  # reported "no generations, skipped" and the run finished with an empty table.
  [[ -f "${pdir}/199.png" ]] || die "[3] ${arm}: expected 200 PNGs under ${pdir}"
  log "[3] ${arm}: scoring ${pdir}"
  "${PY}" -u "${NB_ROOT}/scripts/nda/eval_official_seven_dir.py" \
    --gen-dir "${pdir}" \
    --output-json "${outj}" \
    --tag "${arm}" \
    --images-root "${IMAGES_ROOT}" \
    --device cuda:0 \
    --batch-size 16 || die "seven-metric evaluation failed for arm ${arm}"
done

# =============================================================================
# [4] the frontier
# =============================================================================
log "[4] frontier"
TAG="${TAG}" MET="${MET}" ROOT="${ROOT}" \
"${PY}" - <<'PY'
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ["ROOT"])
sys.path.insert(0, str(ROOT / "scripts"))
from epd.stats import verdict                                    # noqa: E402

MET = Path(os.environ["MET"])
TAG = os.environ["TAG"]

# Ordered by strength; the two controls are held to the side because they vary the
# INPUT at a fixed strength and are not points on the frontier.
FRONTIER = [("ll_txt2img", None), ("ll_s020", 0.20), ("ll_s035", 0.35),
            ("ll_s050", 0.50), ("ll_s065", 0.65), ("ll_s080", 0.80)]
CONTROLS = [("ll_s050_noiseip", "strength 0.50, IP from ZERO EEG"),
            ("ll_s050_depthinit", "strength 0.50, init = depth-as-RGB")]

KEYS = ["pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "swav", "fid"]


def load(arm):
    p = MET / f"{arm}_seven.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def qvec(arm, key):
    """Per-concept vector for a metric, from whichever decomposition it has.

    The two-way metrics are identification scores and decompose per CONCEPT (`q`).
    PixCorr and SSIM are not identification scores and decompose per IMAGE, written
    alongside under `per_image_lowlevel`. Both are indexed by the same test concept,
    so both pair against another arm the same way -- which is what lets the decisive
    low-level comparison carry an interval instead of a bare difference of means.
    """
    p = MET / f"{arm}_seven_persample.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    if key in ("pixcorr", "ssim"):
        v = d.get("per_image_lowlevel", {}).get(key)
    else:
        v = d.get("q", {}).get(key)
    return None if v is None else np.asarray(v, dtype=np.float64)


rows = [(a, s, load(a)) for a, s in FRONTIER]
have = [(a, s, d) for a, s, d in rows if d is not None]
if not have:
    print("no arms scored -- nothing to report")
    raise SystemExit(1)

print(f"## {TAG}: the img2img strength frontier, sub-08, 200 concepts")
print()
print("All arms share one IP condition (`ip_deploy`), one init "
      "(`pred_lowlevel_rgb_512`, PixCorr vs GT = +0.2191), one generator and "
      "`--seed 42`. Only `--strength` varies.")
print()
print("| arm | strength | PixCorr | SSIM | AlexNet(2) | AlexNet(5) | Inception | "
      "CLIP | SwAV | FID |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for arm, s, d in have:
    if d is None:
        print(f"| `{arm}` | {'' if s is None else f'{s:.2f}'} | MISSING | | | | | | | |")
        continue
    lab = "txt2img" if s is None else f"{s:.2f}"
    print(f"| `{arm}` | {lab} | {d['pixcorr']:.3f} | {d['ssim']:.3f} | {d['alex2']:.3f} | "
          f"{d['alex5']:.3f} | {d['inception']:.3f} | {d['clip']:.3f} | "
          f"{d['swav']:.3f} | {d['fid']:.1f} |")

print()
print("### Is the init's information actually being extracted?")
print()
print("At strength 0 the output IS the init (PixCorr +0.2191, measured directly on "
      "the PNGs at no GPU cost). At strength 1 it is `ll_txt2img`. Everything below "
      "is how much of that 0.2191 survives into the final image.")
print()
print("| strength | PixCorr | fraction of the init's 0.2191 reached | CLIP |")
print("|---:|---:|---:|---:|")
INIT_PIXCORR = 0.2191
txt = load("ll_txt2img")
if txt is not None:
    print(f"| txt2img (s->1) | {txt['pixcorr']:.3f} | "
          f"{txt['pixcorr'] / INIT_PIXCORR:.1%} | {txt['clip']:.3f} |")
for arm, s, d in have:
    if d is None or s is None:
        continue
    print(f"| {s:.2f} | {d['pixcorr']:.3f} | {d['pixcorr'] / INIT_PIXCORR:.1%} | "
          f"{d['clip']:.3f} |")

print()
print("### Paired tests vs the shipped configuration (strength 0.80)")
print()
print("Paired over the 200 test concepts: the two-way metrics on the official Pearson "
      "per-concept `q_i`, and PixCorr/SSIM on their per-IMAGE vectors (they are not "
      "identification scores, so they do not decompose per concept -- but they are "
      "indexed by the same concept, so they pair the same way). The unpaired threshold "
      "for a 200-way score is ~10 points; pairing is what buys the resolution to see "
      "0.02 here, and PixCorr is the metric this run is decided on.")
print()
print("| comparison | metric | delta | 95% CI | sign p | verdict |")
print("|---|---|---:|---|---:|---|")
base = "ll_s080"
for arm, s, d in have:
    if arm == base or d is None:
        continue
    lab = f"{arm} vs {base}"
    for key in ("pixcorr", "ssim", "clip", "inception"):
        qa, qb = qvec(arm, key), qvec(base, key)
        if qa is None or qb is None or qa.shape != qb.shape:
            continue
        v = verdict(qa, qb)
        print(f"| {lab} | {key} | {v['delta']:+.3f} | [{v['lo']:+.3f}, {v['hi']:+.3f}] | "
              f"{v['sign_p']:.3f} | {v['verdict']} |")

print()
print("### The two controls at strength 0.50")
print()
check = load("ll_s050")
for arm, why in CONTROLS:
    d = load(arm)
    if d is None or check is None:
        print(f"- `{arm}` ({why}): not scored")
        continue
    print(f"- **`{arm}`** ({why}) vs `ll_s050`: PixCorr "
          f"{d['pixcorr'] - check['pixcorr']:+.3f} ({check['pixcorr']:.3f} -> "
          f"{d['pixcorr']:.3f}), CLIP {d['clip'] - check['clip']:+.3f} "
          f"({check['clip']:.3f} -> {d['clip']:.3f})")
    for key in ("pixcorr", "clip"):
        qa, qb = qvec(arm, key), qvec("ll_s050", key)
        if qa is None or qb is None or qa.shape != qb.shape:
            continue
        v = verdict(qa, qb)
        print(f"  - paired on {key}: {v['delta']:+.3f} "
              f"[{v['lo']:+.3f}, {v['hi']:+.3f}], sign p {v['sign_p']:.3f}, "
              f"{v['verdict']}")
print()
print("`ll_s050_noiseip` consumes no EEG. If its PixCorr stays at `ll_s050`'s level "
      "while its CLIP collapses, then at that strength the PixCorr is being supplied "
      "by the INIT and not by the EEG -- which is the attribution the strength sweep "
      "has to establish before any low-strength arm can be called a result.")
print()
print("`ll_s050_depthinit` swaps the init for `pred_depth_rgb_512`, whose own "
      "PixCorr against the ground truth is -0.095 (anti-correlated). If PixCorr falls "
      "to the txt2img level while CLIP is unharmed, the init's CONTENT is what drives "
      "the low-level metric -- which is the measurement behind removing the "
      "ControlNet-depth branch.")

# FID and SwAV are reported in the table but not tested here: SwAV is a mean
# correlation DISTANCE (lower is better) and FID is not a per-concept score, so
# neither has the decomposition these tests are built on.
print()
print("SwAV and FID appear in the table but are not tested: SwAV is a mean "
      "correlation distance and FID is not a per-concept score, so neither has the "
      "decomposition the paired tests require.")
PY

log "[done] $(date -Iseconds)"
