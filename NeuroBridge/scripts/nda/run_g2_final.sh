#!/usr/bin/env bash
# G2 FINAL: ONE job, ONE complete, correctly controlled experiment.
#
# WHAT THIS ANSWERS
#   Does an INDEPENDENT parallel perceptual tower produce a better structural
#   conditioning signal than the proven serial VAE head, with semantics held fixed?
#
# WHY THIS DESIGN (measured, 2026-09-11)
#   The earlier G2 configuration trained BOTH towers from scratch off `z_eeg_proj`
#   and re-derived the IP-Adapter embedding with an MLP. Measured against the same
#   CLIP-image target on sub-08:
#
#       HCMA rag_soft5 (retrieved memory)               cos 0.5909
#       HCMA blend_nda_cfm_f_a40  <- what sdedit_ll uses cos 0.5451
#       G2 own semantic head, LOSO                      cos 0.4026  (-14.3pp)
#
#   A 14pp semantic deficit cannot be trained away: HCMA's condition is the output
#   of a whole pretrained stack (NDA-SS encoder + semantic tower + RAG + CFM blend),
#   while G2 replaced it with a small MLP regressing 512 -> 1024 from scratch. Any
#   comparison built on that also compares the semantic path, so it cannot attribute
#   anything to the structural innovation.
#
#   The perceptual side is a different story: our structural head reaches
#   std_ratio 0.275-0.530 while sdedit_ll's own VAE head sits at 0.279-0.415, i.e.
#   the parallel tower is comparable or better. So the experiment is worth running,
#   but ONLY if the semantic condition is held at the proven value.
#
# WHAT IS HELD FIXED / WHAT IS VARIED
#   Fixed:   frozen EEG encoder, cross-subject module, HCMA semantic condition
#            (blend_nda_cfm_f_a40), prompts, generator, all sampling parameters,
#            seed, and the anchored band (r < 0.0625).
#   Varied:  the structural anchor only.
#              hcma_ll : sdedit_ll's predicted VAE latent  (the proven predictor)
#              g2_ll   : our parallel perceptual tower      (the innovation)
#              nc0     : no anchor at all                   (value of anchoring)
#              g2sem_ll: our own semantic head as the IP condition, our anchor
#                        (measures the parallel semantic tower, sub-08 only)
#              g2_intra_ll: our perceptual tower trained intra-subject (sub-08 only)
#
#   Both anchors are amplitude-equalised to the SAME low-band std before use. Their
#   native amplitudes differ (ours 29% of target, sdedit_ll's 65.7%), so comparing
#   them raw would mostly measure gain, not spatial accuracy.
#
#   A faithful reproduction of sdedit_ll is NOT regenerated: its 10 folds of images
#   already exist and are evaluated here in the same pass, so the reference number is
#   produced by the same code on the same GT cache as ours.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/g2}"
DEVICE="${DEVICE:-cuda:0}"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-512}"

Z_ROOT="${NB_ROOT}/outputs/hcma_10subj"
IP_TRAIN="/project/peilab/why/eeg-brainit/outputs/atm_bridge/clip_img_train_1024.npy"
IP_TEST="/project/peilab/why/eeg-brainit/outputs/atm_bridge/clip_img_test_1024.npy"
HCMA_EMB_ROOT="${Z_ROOT}"
SDEDIT_ROOT="${NB_ROOT}/outputs/sdedit_ll_full10"
PROMPTS="${Z_ROOT}/prompts/prompts_full_hcma_test.json"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
SOTA_TABLE="${NB_ROOT}/outputs/standard7_protocol/results.json"

CUT="${CUT:-0.0625}"
GEN_STEPS="${GEN_STEPS:-28}"
GEN_GUIDANCE="${GEN_GUIDANCE:-5.0}"
IP_SCALE="${IP_SCALE:-1.0}"
# Low-band amplitude every anchor is equalised to, in unscaled-latent units.
# 0.4641 = mean per-sample r<cut band std of perc_struct_TRAIN (the test split reads
# 0.4506; the train value is used so no test statistic enters the pipeline).
ANCHOR_STD="${ANCHOR_STD:-0.4641}"

# MANDATORY. Without this the job runs on the system interpreter plus ~/.local
# site-packages (transformers 4.36 + diffusers 0.30) instead of the project venv
# (transformers 4.46.3 + diffusers 0.31.0), and every generation stage dies at
# import with "cannot import name 'EncoderDecoderCache'" -- which is exactly how the
# 2026-09-11 run trained 11 folds and then produced zero images.
# shellcheck disable=SC1091
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh"

PYTHON="${PYTHON:-python}"

# ------------------------------------------------------------------- preflight
# Two cheap assertions that each correspond to a failure that already cost a night.
# Both run before any GPU time is spent, and both are hard failures: a wrong
# interpreter or an unusable diffusers build cannot be worked around by continuing.
#   1. the interpreter must be the project venv. On 2026-09-11 `python` resolved to
#      the system interpreter plus ~/.local site-packages, so generation died at
#      import and 11 trained folds produced zero images.
#   2. the exact pipeline class the generator imports must import. This is the check
#      that would have caught it, because training imports nothing from diffusers.
PY_EXE="$("${PYTHON}" -c 'import sys; print(sys.executable)')"
case "${PY_EXE}" in
  *eeg-brainit/.venv/*) echo "[preflight] interpreter OK: ${PY_EXE}" ;;
  *) echo "[FATAL] interpreter is ${PY_EXE}, not the project venv. The generator" \
          "imports diffusers, which is incompatible with the system transformers in" \
          "~/.local. Aborting before spending GPU time." >&2; exit 1 ;;
esac
"${PYTHON}" - <<'PY' || { echo "[FATAL] diffusers pipeline import failed; the generator cannot run." >&2; exit 1; }
import transformers, diffusers, torch
print(f"[preflight] transformers {transformers.__version__} | "
      f"diffusers {diffusers.__version__} | torch {torch.__version__}")
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline  # noqa: F401
print("[preflight] SDXL ControlNet img2img import OK")
PY

MROOT="${OUT}/multi"
mkdir -p "${OUT}/logs" "${MROOT}" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"
echo "{\"pipeline\":\"g2_final\",\"subjects\":\"${SUBJECTS}\",\"epochs\":${EPOCHS},\"started\":\"$(date -Iseconds)\"}" \
  > "${OUT}/final_running.json"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
stag_of() { printf "sub-%02d" "$1"; }

require "${PROMPTS}"
require "${IP_TRAIN}"
require "${IP_TEST}"
for f in perc_struct_train.npy perc_texture_train.npy sem_image_train.npy sem_overall_train.npy; do
  require "${OUT}/targets/${f}"
done

# ---------------------------------------------------------------- train
train_one() {                      # proto subject outdir
  local proto="$1" S="$2" odir="$3"
  if [[ -f "${odir}/g2_report.json" ]]; then echo "[SKIP] train ${proto} sub-$(stag_of "$S")"; return 0; fi
  local extra=()
  if [[ "${proto}" == "intra" ]]; then
    extra=(--z-intra-dir "${NB_ROOT}/outputs/intra_hcma_s/$(stag_of "$S")/train")
  fi
  echo "===== train ${proto} $(stag_of "$S") @ $(date -Iseconds) ====="
  # --resume 1: the runner writes last.pth every epoch, so a preemption continues
  # instead of restarting. This single job is long enough to meet one.
  "${PYTHON}" scripts/nda/g2_train.py \
    --protocol "${proto}" --subject "${S}" \
    --z-root "${Z_ROOT}" --targets-dir "${OUT}/targets" \
    --ip-train-npy "${IP_TRAIN}" --ip-test-npy "${IP_TEST}" \
    --out "${odir}" --epochs "${EPOCHS}" --batch-size "${BATCH}" \
    --device "${DEVICE}" --cut "${CUT}" --resume 1 \
    "${extra[@]}" 2>&1 | tee "${OUT}/logs/train_${proto}_$(stag_of "$S").log"
}

# ---------------------------------------------------------------- generate
gen_row() {                        # tag embed anchor cut extra...
  local tag="$1" emb="$2" anchor="$3" cut="$4"; shift 4
  local gdir="${MROOT}/${STAG}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] gen ${tag} ${STAG}"; return 0; fi
  local a=()
  if [[ -n "${anchor}" && "${cut}" != "0" ]]; then
    a=(--anchor-latent-npy "${anchor}" --anchor-std-target "${ANCHOR_STD}")
  fi
  echo "===== gen ${tag} ${STAG} @ $(date -Iseconds) ====="
  "${PYTHON}" scripts/nda/generate_spectral_decode.py \
    --embed-npy "${emb}" --prompts-json "${PROMPTS}" \
    --output-dir "${gdir}" --tag "${tag}" \
    --cut "${cut}" --gamma 1.0 --start-step 0 \
    --strength 1.0 --ip-scale "${IP_SCALE}" \
    --gen-steps "${GEN_STEPS}" --gen-guidance "${GEN_GUIDANCE}" --seed 42 \
    "${a[@]}" "$@" 2>&1 | tee "${OUT}/logs/gen_${tag}_${STAG}.log"
}

for S in ${SUBJECTS}; do
  STAG="$(stag_of "${S}")"
  HCMA_EMB="${HCMA_EMB_ROOT}/${STAG}/ft/embeds/blend_nda_cfm_f_a40_test.npy"
  SDEDIT_VAE="${SDEDIT_ROOT}/${STAG}/vae_head/pred_vae_test.npy"
  TDIR="${OUT}/inter_${STAG}"
  echo "########## ${STAG} @ $(date -Iseconds) ##########"
  require "${HCMA_EMB}"
  require "${SDEDIT_VAE}"

  # A training failure must cost only this subject, not the whole sweep: the
  # manifest is completeness-filtered downstream, so a missing fold degrades the
  # result honestly (n_folds is reported) instead of killing ten good folds.
  if ! train_one inter "${S}" "${TDIR}"; then
    echo "[WARN] inter training failed for ${STAG}; skipping its rows"
    continue
  fi
  if [[ ! -f "${TDIR}/g2_report.json" ]]; then
    echo "[WARN] no report for ${STAG}; skipping its rows"
    continue
  fi

  gen_row hcma_ll  "${HCMA_EMB}"      "${SDEDIT_VAE}"        "${CUT}" || echo "[WARN] hcma_ll failed ${STAG}"
  gen_row g2_ll    "${HCMA_EMB}"      "${TDIR}/lf_latent_test.npy" "${CUT}" || echo "[WARN] g2_ll failed ${STAG}"

  # sub-08-only ablations and the intra-subject contrast
  if [[ "${S}" == "8" ]]; then
    train_one intra 8 "${OUT}/intra_sub-08" || echo "[WARN] intra training failed"
    gen_row nc0         "${HCMA_EMB}" ""                              "0"      || echo "[WARN] nc0 failed"
    gen_row g2sem_ll    "${TDIR}/ip_direct_test.npy" "${TDIR}/lf_latent_test.npy" "${CUT}" || echo "[WARN] g2sem_ll failed"
    if [[ -f "${OUT}/intra_sub-08/lf_latent_test.npy" ]]; then
      gen_row g2_intra_ll "${HCMA_EMB}" "${OUT}/intra_sub-08/lf_latent_test.npy" "${CUT}" || echo "[WARN] g2_intra_ll failed"
    fi
  fi
done

# ---------------------------------------------------------------- manifest
# Built by direct globbing with an explicit completeness test. An empty manifest is
# a hard failure: on 2026-09-11 this stage "succeeded" with zero rows and the
# failure only surfaced an hour later in the dependent eval job.
MAN="${OUT}/manifest_final.json"
OUT_EVAL="${OUT}" MROOT="${MROOT}" SDEDIT_ROOT="${SDEDIT_ROOT}" MAN="${MAN}" \
"${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out, mroot = Path(os.environ["OUT_EVAL"]), Path(os.environ["MROOT"])
sedit = Path(os.environ["SDEDIT_ROOT"])
rows = []
for sub in sorted(mroot.glob("sub-*")):
    if not sub.is_dir():
        continue
    gen_root = sub / "generation"
    if not gen_root.is_dir():
        continue
    for v in sorted(p.name for p in gen_root.iterdir() if p.is_dir()):
        g = gen_root / v / "generated"
        if (g / "199.png").is_file():
            rows.append({"tag": f"{v}_{sub.name}", "display": f"{v} {sub.name}",
                         "fold": sub.name, "variant": v, "gen_dir": str(g)})
# the already-generated sdedit_ll folds, so the reference number comes from the same
# evaluation code and GT cache in the same run
for sub in sorted(sedit.glob("sub-*")):
    g = sub / "generation" / "sdedit_ll" / "generated"
    if (g / "199.png").is_file():
        rows.append({"tag": f"sdedit_ll_{sub.name}", "display": f"sdedit_ll {sub.name}",
                     "fold": sub.name, "variant": "sdedit_ll", "gen_dir": str(g)})
Path(os.environ["MAN"]).write_text(
    json.dumps({"protocol": "standard7", "rows": rows, "avg_rows": []}, indent=2),
    encoding="utf-8")
print(f"[manifest] {len(rows)} complete rows")
if not rows:
    raise SystemExit("[FATAL] no complete generation rows; nothing to evaluate")
PY

# ---------------------------------------------------------------- standard-7
# Isolated from ../../../standard7_protocol, which holds the 61-row SOTA reference
# table: eval_standard7.py rewrites results.json wholesale rather than merging, so
# sharing the path would replace the reference with this run's rows.
STD7="${OUT}/standard7_final"
mkdir -p "${STD7}"
echo "===== standard-7 @ $(date -Iseconds) ====="
"${PYTHON}" scripts/nda/eval_standard7.py \
  --manifest "${MAN}" --images-root "${IMAGES_ROOT}" \
  --out-dir "${STD7}" --device "${DEVICE}" --batch-size 16 \
  2>&1 | tee "${OUT}/logs/eval_standard7_final.log"

# ---------------------------------------------------------------- pooled FID
# fake = all 10 folds' generations concatenated (10 x 200), real = the 200 unique
# test images. Only complete folds are symlinked in, so one failed fold cannot
# destroy the pooled number for a variant.
for v in g2_ll hcma_ll sdedit_ll; do
  if [[ "${v}" == "sdedit_ll" ]]; then
    SRC_ROOT="${SDEDIT_ROOT}"; TAGNAME="sdedit_ll"
  else
    SRC_ROOT="${MROOT}"; TAGNAME="${v}"
  fi
  PROOT="${OUT}/pooled_${v}"
  rm -rf "${PROOT}"; mkdir -p "${PROOT}"
  NFOLD=0
  for sub in "${SRC_ROOT}"/sub-*; do
    [[ -d "${sub}" ]] || continue
    g="${sub}/generation/${TAGNAME}/generated"
    if [[ -f "${g}/199.png" ]]; then
      ln -sfn "$(cd "${sub}" && pwd)" "${PROOT}/$(basename "${sub}")"
      NFOLD=$((NFOLD + 1))
    fi
  done
  echo "[pooled] ${v}: ${NFOLD} complete folds"
  if (( NFOLD >= 2 )); then
    "${PYTHON}" scripts/nda/eval_pooled_fid.py \
      --root "${PROOT}" --tag "${TAGNAME}" --images-root "${IMAGES_ROOT}" \
      --output-json "${OUT}/fid_pooled_${v}.json" --device "${DEVICE}" \
      2>&1 | tee -a "${OUT}/logs/eval_pooled_fid_final.log" \
      || echo "[WARN] pooled FID failed for ${v}"
  else
    echo "[SKIP] pooled FID ${v}: only ${NFOLD} complete fold(s)"
  fi
done

# ---------------------------------------------------------------- summary
echo "===== summary @ $(date -Iseconds) ====="
"${PYTHON}" scripts/nda/g2_compare_sota.py \
  --g2 "${STD7}/results.json" \
  --sota "${SOTA_TABLE}" \
  --fid-glob "${OUT}/fid_pooled_*.json" \
  --out "${OUT}/final_vs_sota.json" \
  2>&1 | tee "${OUT}/logs/final_comparison.log" || echo "[WARN] comparison failed"

echo "{\"pipeline\":\"g2_final\",\"finished\":\"$(date -Iseconds)\"}" > "${OUT}/final_done.json"
echo "===== G2 FINAL complete @ $(date -Iseconds) ====="
