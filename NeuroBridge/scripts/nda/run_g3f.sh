#!/usr/bin/env bash
# ============================================================================
# G3F -- leak-free dual tower, centered discriminative readout, SELF-PROMPT.
#
# ONE job. Trains intra + all 10 LOSO folds, generates the controlled rows, then
# evaluates everything (ours AND the existing reference rows) in one pass with
# the official standard-7 protocol plus pooled FID.
#
# WHAT IS NEW VS THE g2f RUN (each item tied to a measurement, see g3f_train.py)
#   1. DELETE CFM       -- 3rd disconfirmation; image-side g2f_cfm was the worst
#                          row of the run (CLIP 0.600, FID 223.48).
#   2. DELETE h_texture -- fine/coarse = 0.0765 of explainable variance; the head
#                          regressed an unidentifiable component.
#   3. DELETE h_ip      -- cos-to-target 0.613-0.634 against a CONSTANT baseline
#                          of 0.6147; h_fuse (2-way 0.87-0.955) becomes the sole
#                          IP producer.
#   4. CENTERED RESIDUAL heads over the TRAIN mean, so capacity is not spent
#                          re-learning the common direction that eats the whole
#                          cosine budget.
#   5. InfoNCE on h_image -- it regressed under the same mechanism that collapsed
#                          h_ip.
#   6. SELF-PROMPT      -- replaces HCMA's oracle class name. Retrieval over the
#                          1654 TRAIN concepts indexed by an EEG concept head.
#
# TWO SEPARABLE FACTS THIS RUN EXPLOITS (measured on sub-08, 2026-09-11)
#   prompt buys high-level semantics + FID:
#       hcma_deploy generic  CLIP 0.736  Inception 0.675  FID 179.98
#       hcma_oracle GT name  CLIP 0.948  Inception 0.909  FID 149.22
#   the low-level pathway buys PixCorr/SSIM and barely moves FID:
#       hcma_oracle (weak path)  PixCorr 0.074  FID 149.22
#       sdedit_ll   (RGB init)   PixCorr 0.165  FID 145.76
#   So the rows below vary exactly one of the two at a time.
#
# LEAK-FREE, BY CONSTRUCTION AND BY AUDIT
#   * the prompt gallery holds the 1654 TRAIN concepts only; the 200 test
#     concepts are DISJOINT from it (verified at run time, printed below and
#     hard-failed if the intersection is non-empty), so no emitted string can
#     name a test class even when the EEG prediction is wrong;
#   * `sem_concept_tmpl_test.npy` (the oracle target) is never read for training,
#     selection or prompt construction;
#   * checkpoint selection uses a held-out slice of TRAIN target indices;
#   * the memory bank and concept gallery are train rows only;
#   * generation hyper-parameters are FIXED A PRIORI (not swept on test): the
#     low-level grid in intra_hcma_s selected 0.86/0.88 on test metrics, which is
#     exactly the bias this run avoids. The 0.88 and 0.86 settings collapse to
#     the SAME init_timestep under diffusers' integer truncation, so that grid
#     never even tested what it claimed to.
#
# ROWS
#   all 10 subjects (LOSO: the target subject is unseen during training):
#     g3f_ll_gen    ip_fused + GENERIC prompt + LL-SDEdit 0.82   (pathway only)
#     g3f_ll_self   ip_fused + SELF prompt  + LL-SDEdit 0.82   (headline)
#   sub-08 only:
#     g3f_mem_self  ip_mem   + SELF prompt  + LL-SDEdit 0.82
#     g3f_cn_self   ip_fused + SELF prompt  + Depth-CN 0.25
#   sub-08 intra (trained on sub-08 alone):
#     g3f_intra_self / g3f_intra_gen
#   plus hcma_deploy / g2f_mem / sdedit_ll folds, which already exist, evaluated
#   here by the same code on the same GT cache in the same pass.
# ============================================================================
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/g3f}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-26}"
BATCH="${BATCH:-512}"

Z_ROOT="${NB_ROOT}/outputs/hcma_10subj"
G2F_ROOT="${NB_ROOT}/outputs/g2f"
SDEDIT_ROOT="${NB_ROOT}/outputs/sdedit_ll_full10"
INTRA_S="${NB_ROOT}/outputs/intra_hcma_s/sub-08"
CLIP_TEXT="${NB_ROOT}/outputs/nda_ss/sub-08/clip_text"
CAPTIONS="${NB_ROOT}/outputs/g2/captions/captions_train.jsonl"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
SOTA_TABLE="${NB_ROOT}/outputs/standard7_protocol/results.json"

# A PRIORI FIXED, never selected on test.
LL_STRENGTH=0.82          # identical to the sdedit_ll reference rows
CN_SCALE=0.25             # conservative end of the historical grid
CN_STRENGTH=0.86
GEN_STEPS=28
GEN_GUIDANCE=5.0
IP_SCALE=1.0
SEED=42

GENERIC="a photo of an object, clearly showing its shape, color, and distinctive parts, natural lighting"

# MANDATORY: the system interpreter dies at import (transformers/diffusers
# mismatch). Omitting this once cost a whole night: 11 folds trained, 0 images.
# shellcheck disable=SC1091
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh"
PYTHON="$(command -v python)"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
mkdir -p "${XDG_CACHE_HOME}" "${OUT}/logs" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"

# HARD GUARD: a GPU job must actually get a GPU.
# On 2026-09-11 the scheduler placed this job on dgx-09, whose driver (12.8) is
# older than the venv's torch build (cu130), so torch fell back to CPU silently:
# `[g3f] params 34.05M device cpu`. 148860 rows/epoch/fold on CPU is ~8 h per
# fold, i.e. ~87 h for the 11 folds, against ~1 h on a GPU. Fail in seconds
# instead, and let the submission pin a node whose driver is known good.
"${PYTHON}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("[FATAL] no usable CUDA device -- refusing to train on CPU. "
                     "Check the node's driver version against torch's CUDA build.")
print(f"[env] torch {torch.__version__} cuda_rt {torch.version.cuda} | "
      f"{torch.cuda.get_device_name(0)} "
      f"({torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GiB)")
PY

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
echo "{\"pipeline\":\"g3f\",\"started\":\"$(date -Iseconds)\",\"job\":\"${SLURM_JOB_ID:-local}\"}" \
  > "${OUT}/running.json"

# ------------------------------------------------------------------ [0] leak audit
echo "===== [0] leak audit: train concepts vs test concepts @ $(date -Iseconds) ====="
"${PYTHON}" - "${CLIP_TEXT}" "${CAPTIONS}" "${IMAGES_ROOT}" <<'PY'
import json, sys
from pathlib import Path
clip_text, captions, images_root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
phrases = json.loads((clip_text / "train" / "concept_phrases.json").read_text(encoding="utf-8"))
vocab = set(phrases)
def dirs(p):
    return [json.loads(l)["path"].rsplit("/", 1)[0].rsplit("/", 1)[-1]
            for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
tr = {d.split("_", 1)[1].replace("_", " ") for d in dirs(captions)}
test_dirs = sorted({p.parent.name for p in (images_root / "test_images").glob("*/*")})
te = {d.split("_", 1)[1].replace("_", " ") for d in test_dirs}
inter = vocab & te
print(f"[audit] train concepts {len(tr)}  test concepts {len(te)}  "
      f"test concepts inside the prompt gallery = {len(inter)}")
if inter:
    raise SystemExit(f"[FATAL] the prompt gallery leaks {len(inter)} test concepts, "
                     f"e.g. {sorted(inter)[:5]}")
print("[audit] gallery is train-only and disjoint from the test concepts -> a prompt "
      "built from it CANNOT name a test class")
PY

# ------------------------------------------------------------------ [1] training
train_fold() {                    # out_dir proto test_subj
  local odir="$1" proto="$2" S="$3"
  if [[ -f "${odir}/g3f_report.json" ]]; then echo "[SKIP] train ${proto} sub-0${S}"; return 0; fi
  local tr=(--train-subjects)
  if [[ "${proto}" == "intra" ]]; then
    tr+=("${S}")
  else
    for k in 1 2 3 4 5 6 7 8 9 10; do [[ "${k}" != "${S}" ]] && tr+=("${k}"); done
  fi
  echo "===== train ${proto} sub-0${S} @ $(date -Iseconds) ====="
  "${PYTHON}" scripts/nda/g3f_train.py \
    "${tr[@]}" --test-subject "${S}" \
    --z-root "${Z_ROOT}" --targets-dir "${NB_ROOT}/outputs/g2/targets" \
    --clip-text-dir "${CLIP_TEXT}" --captions-jsonl "${CAPTIONS}" \
    --out "${odir}" --epochs "${EPOCHS}" --batch-size "${BATCH}" \
    --device "${DEVICE}" --resume 1 \
    2>&1 | tee "${OUT}/logs/train_${proto}_sub-0${S}.log"
}

# ------------------------------------------------------------------ [2] generation
# sdedit mode: EEG low-level head decodes a blurry RGB init, then img2img.
gen_ll() {                        # tag cond_npy prompt aux_root
  local tag="$1" cond="$2" prompt="$3" aux="$4"
  local gdir="${OUT}/gen/${STAG}/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] gen ${tag} ${STAG}"; return 0; fi
  [[ -f "${cond}" ]] || { echo "[WARN] ${tag}: missing cond ${cond}"; return 1; }
  local ll="${aux}/vae_head/pred_lowlevel_rgb_512"
  require "${ll}/199.png"
  echo "===== gen ${tag} ${STAG} @ $(date -Iseconds) ====="
  "${PYTHON}" scripts/nda/generate_atm_aligned_decode.py \
    --mode sdedit --embed-npy "${cond}" --prompts-json "${prompt}" \
    --lowlevel-rgb-dir "${ll}" \
    --output-dir "${gdir}" --tag "${tag}" \
    --strength "${LL_STRENGTH}" --ip-scale "${IP_SCALE}" \
    --gen-steps "${GEN_STEPS}" --gen-guidance "${GEN_GUIDANCE}" --seed "${SEED}" \
    2>&1 | tee "${OUT}/logs/gen_${tag}_${STAG}.log"
}

# depth-ControlNet mode: adds an explicit structural control branch in addition
# to the low-level RGB init.
gen_cn() {                        # tag cond_npy prompt aux_root
  local tag="$1" cond="$2" prompt="$3" aux="$4"
  local gdir="${OUT}/gen/${STAG}/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] gen ${tag} ${STAG}"; return 0; fi
  [[ -f "${cond}" ]] || { echo "[WARN] ${tag}: missing cond ${cond}"; return 1; }
  local ll="${aux}/vae_head/pred_lowlevel_rgb_512"
  local dp="${aux}/depth/pred_depth_rgb_512"
  require "${ll}/199.png"; require "${dp}/199.png"
  echo "===== gen ${tag} ${STAG} @ $(date -Iseconds) ====="
  "${PYTHON}" scripts/nda/generate_hcma_s_decode.py \
    --embed-npy "${cond}" --prompts-json "${prompt}" \
    --depth-rgb-dir "${dp}" --lowlevel-rgb-dir "${ll}" \
    --output-dir "${gdir}" --tag "${tag}" \
    --cn-scale "${CN_SCALE}" --ip-scale "${IP_SCALE}" --strength "${CN_STRENGTH}" \
    --gen-steps "${GEN_STEPS}" --gen-guidance "${GEN_GUIDANCE}" --seed "${SEED}" \
    2>&1 | tee "${OUT}/logs/gen_${tag}_${STAG}.log"
}

for S in 1 2 3 4 5 6 7 8 9 10; do
  STAG="$(printf 'sub-%02d' "${S}")"
  ODIR="${OUT}/${STAG}"
  AUX="${SDEDIT_ROOT}/${STAG}"          # per-subject intra low-level head (existing)
  require "${Z_ROOT}/${STAG}/zret/z_eeg_proj_test.npy"
  require "${AUX}/vae_head/pred_lowlevel_rgb_512/199.png"
  echo "########## ${STAG} @ $(date -Iseconds) ##########"

  if ! train_fold "${ODIR}" loso "${S}"; then
    echo "[WARN] LOSO training failed for ${STAG}; skipping its rows"; continue
  fi
  [[ -f "${ODIR}/g3f_report.json" ]] || { echo "[WARN] no report ${STAG}"; continue; }

  SELF="${ODIR}/prompts/prompts_self.json"        # UNGATED, threshold free
  GATED="${ODIR}/prompts/prompts_selfgate.json"   # val-threshold gated
  GENC="${ODIR}/prompts/prompts_generic.json"
  require "${SELF}"; require "${GATED}"; require "${GENC}"

  # Manifold calibration of the condition (train-statistic reference, no labels,
  # no test data). See g3f_calibrate_cond.py: every one of our conditions is
  # off-manifold for IP-Adapter, which was trained on REAL CLIP image embeddings.
  CAL="${ODIR}/conds/ip_fused_cal_test.npy"
  if [[ ! -f "${CAL}" ]]; then
    "${PYTHON}" scripts/nda/g3f_calibrate_cond.py \
      --in-npy "${ODIR}/conds/ip_fused_test.npy" --out-npy "${CAL}" \
      --ref-npy "${IP_TRAIN}" --mode quantile \
      --report-json "${ODIR}/conds/ip_fused_cal_report.json" \
      2>&1 | tee "${OUT}/logs/cal_${STAG}.log"
  fi
  require "${CAL}"

  HCMA_EMB="${Z_ROOT}/${STAG}/ft/embeds/blend_nda_cfm_f_a40_test.npy"
  require "${HCMA_EMB}"

  gen_ll g3f_ll_gen  "${ODIR}/conds/ip_fused_test.npy" "${GENC}" "${AUX}" \
    || echo "[WARN] g3f_ll_gen failed ${STAG}"
  # primary prompt row: ungated, so its effect is not diluted by the gate
  gen_ll g3f_ll_self "${ODIR}/conds/ip_fused_test.npy" "${SELF}" "${AUX}" \
    || echo "[WARN] g3f_ll_self failed ${STAG}"
  # deployable variant (gate threshold from TRAIN-val margins only)
  gen_ll g3f_ll_selfgate "${ODIR}/conds/ip_fused_test.npy" "${GATED}" "${AUX}" \
    || echo "[WARN] g3f_ll_selfgate failed ${STAG}"
  # calibrated condition, generic prompt: isolates the manifold effect
  gen_ll g3f_ll_cal "${CAL}" "${GENC}" "${AUX}" \
    || echo "[WARN] g3f_ll_cal failed ${STAG}"
  # HCMA condition on the SAME pathway and the SAME prompt as g3f_ll_gen: this is
  # the clean condition comparison. Without it, g3f_ll_gen vs hcma_deploy differs
  # in BOTH the condition and the low-level pathway and attributes nothing.
  gen_ll hcma_ll_gen "${HCMA_EMB}" "${GENC}" "${AUX}" \
    || echo "[WARN] hcma_ll_gen failed ${STAG}"

  if [[ "${S}" == "8" ]]; then
    gen_ll g3f_mem_self "${ODIR}/conds/ip_mem_test.npy" "${SELF}" "${AUX}" || true
    gen_cn g3f_cn_self  "${ODIR}/conds/ip_fused_test.npy" "${SELF}" "${INTRA_S}" || true

    # intra contrast: every EEG weight trained on sub-08 alone
    H8="${OUT}/intra_sub-08"
    if train_fold "${H8}" intra 8; then
      gen_ll g3f_intra_gen  "${H8}/conds/ip_fused_test.npy" "${GENC}" "${INTRA_S}" || true
      gen_ll g3f_intra_self "${H8}/conds/ip_fused_test.npy" "${SELF}" "${INTRA_S}" || true
    else
      echo "[WARN] intra training failed"
    fi
  fi
done

# ------------------------------------------------------------------ [3] manifest
MAN="${OUT}/manifest.json"
OUT_EVAL="${OUT}" G2F_ROOT="${G2F_ROOT}" SDEDIT_ROOT="${SDEDIT_ROOT}" MAN="${MAN}" \
"${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out, g2f, sedit = Path(os.environ["OUT_EVAL"]), Path(os.environ["G2F_ROOT"]), Path(os.environ["SDEDIT_ROOT"])
rows = []
for sub in sorted((out / "gen").glob("sub-*")):
    for v in sorted(p.name for p in sub.iterdir() if p.is_dir()):
        g = sub / v / "generated"
        if (g / "199.png").is_file():
            rows.append({"tag": f"{v}_{sub.name}", "display": f"{v} {sub.name}",
                         "fold": sub.name, "variant": v, "gen_dir": str(g)})
# reference rows that already exist: evaluated here, same code, same GT cache
for root, subl, tag in ((g2f / "gen", "{v}", None),):
    for sub in sorted(root.glob("sub-*")):
        for v in ("hcma_deploy", "g2f_mem"):
            g = sub / v / "generated"
            if (g / "199.png").is_file():
                rows.append({"tag": f"{v}_{sub.name}", "display": f"{v} {sub.name}",
                             "fold": sub.name, "variant": v, "gen_dir": str(g)})
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
    raise SystemExit("[FATAL] no complete generation rows")
PY

# ------------------------------------------------------------------ [4] standard-7
STD7="${OUT}/standard7"
mkdir -p "${STD7}"
echo "===== standard-7 @ $(date -Iseconds) ====="
"${PYTHON}" scripts/nda/eval_standard7.py \
  --manifest "${MAN}" --images-root "${IMAGES_ROOT}" \
  --out-dir "${STD7}" --device "${DEVICE}" --batch-size 16 \
  2>&1 | tee "${OUT}/logs/eval_standard7.log"

# ------------------------------------------------------------------ [5] pooled FID
# eval_pooled_fid.py looks for <root>/sub-XX/generation/<tag>/generated, so the
# link tree is built to that exact shape. The g2f run linked <root>/sub-XX
# straight at the fold dir and died with FileNotFoundError -- do not repeat that.
pool_links() {                    # src_layout tag root fold_glob
  local src="$1" tag="$2" root="$3" glob="$4"
  rm -rf "${root}"; mkdir -p "${root}"
  local n=0
  for sub in "${src}"/${glob}; do
    [[ -d "${sub}" ]] || continue
    local g=""
    [[ -f "${sub}/generation/${tag}/generated/199.png" ]] && g="${sub}/generation/${tag}/generated"
    [[ -n "${g}" ]] || continue
    mkdir -p "${root}/$(basename "${sub}")/generation/${tag}"
    ln -sfn "$(cd "${g}" && pwd)" "${root}/$(basename "${sub}")/generation/${tag}/generated"
    n=$((n + 1))
  done
  echo "${n}"
}
# ours live at gen/sub-XX/<tag>/generated, so link through an intermediate shape
pool_links_ours() {               # tag root
  local tag="$1" root="$2"
  rm -rf "${root}"; mkdir -p "${root}"
  local n=0
  for sub in "${OUT}"/gen/sub-*; do
    [[ -f "${sub}/${tag}/generated/199.png" ]] || continue
    mkdir -p "${root}/$(basename "${sub}")/generation/${tag}"
    ln -sfn "$(cd "${sub}/${tag}/generated" && pwd)" \
            "${root}/$(basename "${sub}")/generation/${tag}/generated"
    n=$((n + 1))
  done
  echo "${n}"
}
for v in g3f_ll_self g3f_ll_gen; do
  PROOT="${OUT}/pooled_${v}"
  NFOLD="$(pool_links_ours "${v}" "${PROOT}")"
  echo "[pooled] ${v}: ${NFOLD} complete folds"
  if (( NFOLD >= 2 )); then
    "${PYTHON}" scripts/nda/eval_pooled_fid.py \
      --root "${PROOT}" --tag "${v}" --images-root "${IMAGES_ROOT}" \
      --output-json "${OUT}/fid_pooled_${v}.json" --device "${DEVICE}" \
      2>&1 | tee -a "${OUT}/logs/eval_pooled_fid.log" \
      || echo "[WARN] pooled FID failed for ${v}"
  fi
done
for v in hcma_deploy g2f_mem; do
  PROOT="${OUT}/pooled_${v}"
  rm -rf "${PROOT}"; mkdir -p "${PROOT}"
  n=0
  for sub in "${G2F_ROOT}"/gen/sub-*; do
    [[ -f "${sub}/${v}/generated/199.png" ]] || continue
    mkdir -p "${PROOT}/$(basename "${sub}")/generation/${v}"
    ln -sfn "$(cd "${sub}/${v}/generated" && pwd)" \
            "${PROOT}/$(basename "${sub}")/generation/${v}/generated"
    n=$((n + 1))
  done
  echo "[pooled] ${v}: ${n} complete folds"
  if (( n >= 2 )); then
    "${PYTHON}" scripts/nda/eval_pooled_fid.py \
      --root "${PROOT}" --tag "${v}" --images-root "${IMAGES_ROOT}" \
      --output-json "${OUT}/fid_pooled_${v}.json" --device "${DEVICE}" \
      2>&1 | tee -a "${OUT}/logs/eval_pooled_fid.log" \
      || echo "[WARN] pooled FID failed for ${v}"
  fi
done
PROOT="${OUT}/pooled_sdedit_ll"
NFOLD="$(pool_links "${SDEDIT_ROOT}" sdedit_ll "${PROOT}" "sub-*")"
echo "[pooled] sdedit_ll: ${NFOLD} complete folds"
if (( NFOLD >= 2 )); then
  "${PYTHON}" scripts/nda/eval_pooled_fid.py \
    --root "${PROOT}" --tag sdedit_ll --images-root "${IMAGES_ROOT}" \
    --output-json "${OUT}/fid_pooled_sdedit_ll.json" --device "${DEVICE}" \
    2>&1 | tee -a "${OUT}/logs/eval_pooled_fid.log" \
    || echo "[WARN] pooled FID failed for sdedit_ll"
fi

# ------------------------------------------------------------------ [6] comparison
echo "===== comparison vs the 61-row SOTA table @ $(date -Iseconds) ====="
"${PYTHON}" scripts/nda/g2_compare_sota.py \
  --g2 "${STD7}/results.json" --sota "${SOTA_TABLE}" \
  --fid-glob "${OUT}/fid_pooled_*.json" --out "${OUT}/final_vs_sota.json" \
  2>&1 | tee "${OUT}/logs/comparison.log" || echo "[WARN] comparison failed"

"${PYTHON}" scripts/nda/g3f_summary.py --out "${OUT}" \
  2>&1 | tee "${OUT}/logs/summary.log" || echo "[WARN] summary failed"

echo "{\"pipeline\":\"g3f\",\"finished\":\"$(date -Iseconds)\"}" > "${OUT}/done.json"
echo "===== G3F complete @ $(date -Iseconds) ====="
