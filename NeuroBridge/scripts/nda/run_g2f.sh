#!/usr/bin/env bash
# G2F: the user-designed dual tower, trained with leak-free objectives, evaluated
# under the official standard-7 protocol with a DEPLOYABLE prompt.
#
# ============================================================================
# THE QUESTION
#   Under a protocol that neither leaks the test class name nor selects anything
#   on the test set, does the user-designed semantic tower produce a better
#   image-decoding condition than HCMA's semantic path?
#
# THE FOUR DEFECTS THIS RUN ADDRESSES (all measured 2026-09-11, sub-08)
#   1. The generation prompt is oracle text. HCMA's prompt is
#      "a photo of a <GT test concept name>" for 200/200 rows. The 200 test
#      concepts are DISJOINT from the 1655 train concepts (intersection empty),
#      so no EEG-derived model can produce that name. No row in the 61-row SOTA
#      table isolates this contribution.
#      Direct evidence of its size, from two pipeline-faithful rows:
#        official_atm_sub08  prompt=""          CLIP 0.7845  Inception 0.7292  Alex2 0.7845
#        hcma_s08            prompt=<GT class>  CLIP 0.9323  Inception 0.9205  Alex2 0.7539
#      The advantage sits in the two metrics that a class name can supply
#      (+14.8pp CLIP, +19.1pp Inception) and vanishes on lower-level features.
#      This run measures it directly (hcma_deploy vs hcma_oracle) instead of
#      assuming it.
#   2. cos-to-target is not a usable metric here. A CONSTANT vector (train CLIP
#      centroid) scores 0.6147, above HCMA's own 0.5451, because random image
#      pairs already sit at 0.378. The usable metric is 200-way discrimination:
#        constant          top1 0.005  top5 0.025  2way 0.520
#        HCMA rag_soft5    top1 0.060  top5 0.200  2way 0.760
#        HCMA blend_a40    top1 0.060  top5 0.185  2way 0.755
#        NDA-SS decoder    top1 0.290  top5 0.585  2way 0.940
#   3. 9 of 10 subjects run with NDA_SRC=memory_soft5, i.e. their HCMA semantic
#      condition is a retrieval table, not a decoder; only sub-08 uses the real
#      decoder. Those two differ 5x in top-1. The 10-subject average therefore
#      mixes a decoder with nine retrieval tables.
#   4. The 5x gap in (2) cannot be blamed on the encoder or the input. All three
#      pipelines read the SAME cached latent: z_eeg_proj and z_ret are
#      byte-identical, and nda_ss/embeds/z_eeg_proj is byte-identical to
#      hcma_10subj/.../z_eeg_proj (np.array_equal == True). It is the objective:
#      NDA-SS carries a DIFFERENTIABLE soft memory plus class supervision, while
#      HCMA's memory is a non-differentiable lookup, so its encoder was never
#      trained to be retrieval-friendly.
#
# WHAT THIS RUN CHANGES
#   * semantic objective -> differentiable soft memory (inside autograd) +
#     class-level contrastive over the 1655 train concepts + multi-teacher
#     regression + multi-granularity fusion into the IP embedding, so the user's
#     multi-granularity design is functionally required, not decorative.
#   * differentiable TOP-K soft memory (k=16). A full softmax over all 16540
#     entries averages the whole bank and returns its mean: measured mem_to_ip
#     0.6426 against a CONSTANT baseline of 0.6147 and 2-way 0.65 vs 0.52.
#     No leave-one-out mask: that mask is only meaningful when query and gallery
#     are the SAME object in the SAME space (HCMA's rag_soft5, where row i
#     matches itself at sim 1.0 and returns its own target, hence 0.8891 against
#     its own target). Here the query is a learned projection and the gallery
#     holds image embeddings, so masking the true positive removes the only
#     correct answer.
#   * in-batch InfoNCE with multi-positive masking + class-level contrastive.
#     A pure "be close to your own target" objective is satisfiable while every
#     output drifts to the bank mean, because nothing forces separation.
#   * checkpoint selection on a held-out slice of TRAIN target indices only.
#   * prompts: a class-name-free DEPLOYABLE prompt for every subject row, plus
#     ORACLE rows on sub-08 to size the text leak.
#
# WHAT IS HELD FIXED
#   the frozen EEG latent, the cross-subject module, the generator, the sampler,
#   seed, steps/guidance/ip-scale, the anchored band (r<0.0625), and the
#   structural anchor with its amplitude. For the headline pair
#   (hcma_deploy vs g2f_mem) the prompt AND the anchor are identical, so the only
#   difference is which semantic condition feeds IP-Adapter.
#
# ROWS
#   all 10 subjects (LOSO: trained on the other nine, target subject unseen):
#     hcma_deploy    HCMA condition + generic prompt + HCMA anchor   (reference)
#     g2f_mem        G2F memory     + generic prompt + HCMA anchor   (ours)
#   sub-08 only:
#     hcma_oracle    HCMA condition + GT-concept prompt              (leak on)
#     g2f_mem_oracle G2F memory     + GT-concept prompt
#     g2f_fused      G2F fused head + generic prompt
#     g2f_cfm        G2F CFM sample + generic prompt
#     g2f_struct     G2F memory     + generic prompt + G2F own LF anchor
#     g2f_mem_intra  G2F memory trained on sub-08 alone (intra contrast)
#     nda_deploy     NDA-SS decoder + generic prompt                 (strong ref)
#   plus the existing sdedit_ll folds, evaluated by the same code on the same GT
#   cache in the same pass.
# ============================================================================
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/g2f}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-26}"
BATCH="${BATCH:-512}"

Z_ROOT="${NB_ROOT}/outputs/hcma_10subj"
SDEDIT_ROOT="${NB_ROOT}/outputs/sdedit_ll_full10"
NDA_SS="${NB_ROOT}/outputs/nda_ss"
G2T="${NB_ROOT}/outputs/g2/targets"
IP_TRAIN="/project/peilab/why/eeg-brainit/outputs/atm_bridge/clip_img_train_1024.npy"
IP_TEST="/project/peilab/why/eeg-brainit/outputs/atm_bridge/clip_img_test_1024.npy"
PROMPTS_ORACLE_SRC="${Z_ROOT}/prompts/prompts_full_hcma_test.json"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
SOTA_TABLE="${NB_ROOT}/outputs/standard7_protocol/results.json"

CUT="${CUT:-0.0625}"
GEN_STEPS="${GEN_STEPS:-28}"
GEN_GUIDANCE="${GEN_GUIDANCE:-5.0}"
IP_SCALE="${IP_SCALE:-1.0}"
# Low-band amplitude every structural anchor is equalised to, in unscaled-latent
# units. 0.4641 is the TRAIN-split mean per-sample r<cut band std, so no test
# statistic enters the pipeline. Anchors have different native amplitudes (G2F
# ~29% of target, sdedit_ll's VAE head ~65.7%), so comparing them raw would
# measure gain rather than spatial accuracy.
ANCHOR_STD="${ANCHOR_STD:-0.4641}"

# MANDATORY. Without this the run uses the system interpreter and every
# generation stage dies at import (transformers/diffusers mismatch). This single
# omission is what cost the 2026-09-11 g2 night: 11 folds trained, 0 images.
# shellcheck disable=SC1091
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh"
PYTHON="${PYTHON:-python}"

mkdir -p "${OUT}/logs" "${OUT}/prompts" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"

# ------------------------------------------------------------------ preflight
PY_EXE="$("${PYTHON}" -c 'import sys; print(sys.executable)')"
case "${PY_EXE}" in
  *eeg-brainit/.venv/*) echo "[preflight] interpreter OK: ${PY_EXE}" ;;
  *) echo "[FATAL] interpreter ${PY_EXE} is not the project venv" >&2; exit 1 ;;
esac
"${PYTHON}" - <<'PY' || { echo "[FATAL] diffusers import failed" >&2; exit 1; }
import transformers, diffusers, torch
print(f"[preflight] transformers {transformers.__version__} | "
      f"diffusers {diffusers.__version__} | torch {torch.__version__}")
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline  # noqa: F401
print("[preflight] SDXL ControlNet img2img import OK")
PY

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
require "${PROMPTS_ORACLE_SRC}"; require "${IP_TRAIN}"; require "${IP_TEST}"
for f in sem_image_train.npy sem_overall_train.npy sem_subject_train.npy \
         sem_background_train.npy sem_detail_train.npy sem_concept_tmpl_train.npy \
         perc_struct_train.npy perc_texture_train.npy; do
  require "${G2T}/${f}"
done

# ------------------------------------------------- deployable vs oracle prompt
# The test concepts are unseen, so the honest prompt carries no class name and
# all semantic content must come from the IP embedding.
DEPLOY="${OUT}/prompts/prompts_deploy.json"
if [[ ! -f "${DEPLOY}" ]]; then
  "${PYTHON}" - "${DEPLOY}" "${PROMPTS_ORACLE_SRC}" <<'PY'
import json, sys
out, oracle = sys.argv[1], sys.argv[2]
n = len(json.load(open(oracle, encoding="utf-8")))
p = "a photo of an object, clearly showing its shape, color, and distinctive parts, natural lighting"
json.dump([p] * n, open(out, "w", encoding="utf-8"), indent=1)
print(f"[prompts] deploy prompt x{n} (class-name free) -> {out}")
PY
fi
ORACLE="${OUT}/prompts/prompts_oracle.json"
[[ -f "${ORACLE}" ]] || cp -f "${PROMPTS_ORACLE_SRC}" "${ORACLE}"

# ------------------------------------------------------------------- training
train_fold() {                    # out_dir proto test_subj
  local odir="$1" proto="$2" S="$3"
  if [[ -f "${odir}/g2f_report.json" ]]; then echo "[SKIP] train ${proto} sub-0${S}"; return 0; fi
  local tr=(--train-subjects)
  if [[ "${proto}" == "intra" ]]; then
    tr+=("${S}")
  else
    for k in 1 2 3 4 5 6 7 8 9 10; do [[ "${k}" != "${S}" ]] && tr+=("${k}"); done
  fi
  echo "===== train ${proto} sub-0${S} @ $(date -Iseconds) ====="
  "${PYTHON}" scripts/nda/g2f_train.py \
    "${tr[@]}" --test-subject "${S}" \
    --z-root "${Z_ROOT}" --targets-dir "${G2T}" \
    --ip-train-npy "${IP_TRAIN}" --ip-test-npy "${IP_TEST}" \
    --out "${odir}" --epochs "${EPOCHS}" --batch-size "${BATCH}" \
    --device "${DEVICE}" --resume 1 \
    2>&1 | tee "${OUT}/logs/train_${proto}_sub-0${S}.log"
}

# ------------------------------------------------------------------- generate
gen_row() {                       # tag embed prompt anchor cut
  local tag="$1" emb="$2" prompt="$3" anchor="$4" cut="$5"
  local gdir="${OUT}/gen/${STAG}/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] gen ${tag} ${STAG}"; return 0; fi
  [[ -f "${emb}" ]] || { echo "[WARN] ${tag}: missing embed ${emb}"; return 1; }
  local a=()
  if [[ -n "${anchor}" && "${cut}" != "0" ]]; then
    a=(--anchor-latent-npy "${anchor}" --anchor-std-target "${ANCHOR_STD}")
  fi
  echo "===== gen ${tag} ${STAG} @ $(date -Iseconds) ====="
  "${PYTHON}" scripts/nda/generate_spectral_decode.py \
    --embed-npy "${emb}" --prompts-json "${prompt}" \
    --output-dir "${gdir}" --tag "${tag}" \
    --cut "${cut}" --gamma 1.0 --start-step 0 --strength 1.0 \
    --ip-scale "${IP_SCALE}" --gen-steps "${GEN_STEPS}" \
    --gen-guidance "${GEN_GUIDANCE}" --seed 42 "${a[@]}" \
    2>&1 | tee "${OUT}/logs/gen_${tag}_${STAG}.log"
}

echo "{\"pipeline\":\"g2f\",\"started\":\"$(date -Iseconds)\"}" > "${OUT}/running.json"

for S in 1 2 3 4 5 6 7 8 9 10; do
  STAG="$(printf 'sub-%02d' "${S}")"
  ODIR="${OUT}/${STAG}"
  HCMA_EMB="${Z_ROOT}/${STAG}/ft/embeds/blend_nda_cfm_f_a40_test.npy"
  HCMA_ANCHOR="${SDEDIT_ROOT}/${STAG}/vae_head/pred_vae_test.npy"
  echo "########## ${STAG} @ $(date -Iseconds) ##########"
  require "${HCMA_EMB}"; require "${HCMA_ANCHOR}"

  # a training failure must degrade one fold honestly, not kill the sweep
  if ! train_fold "${ODIR}" loso "${S}"; then
    echo "[WARN] LOSO training failed for ${STAG}; skipping its rows"; continue
  fi
  [[ -f "${ODIR}/g2f_report.json" ]] || { echo "[WARN] no report ${STAG}"; continue; }

  # headline pair: same prompt, same anchor, only the IP condition differs
  gen_row hcma_deploy "${HCMA_EMB}" "${DEPLOY}" "${HCMA_ANCHOR}" "${CUT}" \
    || echo "[WARN] hcma_deploy failed ${STAG}"
  gen_row g2f_mem "${ODIR}/conds/ip_mem_test.npy" "${DEPLOY}" "${HCMA_ANCHOR}" "${CUT}" \
    || echo "[WARN] g2f_mem failed ${STAG}"

  if [[ "${S}" == "8" ]]; then
    H8="${OUT}/intra_sub-08"
    train_fold "${H8}" intra 8 || echo "[WARN] intra training failed"
    ND="${NDA_SS}/sub-08/train/z_decode_vith_test.npy"
    gen_row nda_deploy "${ND}" "${DEPLOY}" "${HCMA_ANCHOR}" "${CUT}" \
      || echo "[WARN] nda_deploy skipped"
    # oracle pair: identical except the prompt names the GT class
    gen_row hcma_oracle "${HCMA_EMB}" "${ORACLE}" "${HCMA_ANCHOR}" "${CUT}" || true
    gen_row g2f_mem_oracle "${ODIR}/conds/ip_mem_test.npy" "${ORACLE}" "${HCMA_ANCHOR}" "${CUT}" || true
    # component ablations
    gen_row g2f_fused "${ODIR}/conds/ip_fused_test.npy" "${DEPLOY}" "${HCMA_ANCHOR}" "${CUT}" || true
    gen_row g2f_cfm "${ODIR}/conds/ip_cfm_test.npy" "${DEPLOY}" "${HCMA_ANCHOR}" "${CUT}" || true
    gen_row g2f_mem_intra "${H8}/conds/ip_mem_test.npy" "${DEPLOY}" "${HCMA_ANCHOR}" "${CUT}" || true
    # perceptual tower as its own structural anchor
    gen_row g2f_struct "${ODIR}/conds/ip_mem_test.npy" "${DEPLOY}" \
      "${ODIR}/conds/lf_latent_test.npy" "${CUT}" || true
  fi
done

# ------------------------------------------------------------------- manifest
MAN="${OUT}/manifest.json"
OUT_EVAL="${OUT}" SDEDIT_ROOT="${SDEDIT_ROOT}" MAN="${MAN}" \
"${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out, sedit = Path(os.environ["OUT_EVAL"]), Path(os.environ["SDEDIT_ROOT"])
rows = []
for sub in sorted((out / "gen").glob("sub-*")):
    for v in sorted(p.name for p in sub.iterdir() if p.is_dir()):
        g = sub / v / "generated"
        if (g / "199.png").is_file():
            rows.append({"tag": f"{v}_{sub.name}", "display": f"{v} {sub.name}",
                         "fold": sub.name, "variant": v, "gen_dir": str(g)})
# existing sdedit_ll folds: same evaluation code, same GT cache, same pass
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

# --------------------------------------------------------------- standard-7
# isolated from ../../../standard7_protocol, which holds the 61-row SOTA
# reference table: eval_standard7.py rewrites results.json wholesale rather than
# merging, so sharing the path would overwrite the reference.
STD7="${OUT}/standard7"
mkdir -p "${STD7}"
echo "===== standard-7 @ $(date -Iseconds) ====="
"${PYTHON}" scripts/nda/eval_standard7.py \
  --manifest "${MAN}" --images-root "${IMAGES_ROOT}" \
  --out-dir "${STD7}" --device "${DEVICE}" --batch-size 16 \
  2>&1 | tee "${OUT}/logs/eval_standard7.log"

# --------------------------------------------------------------- pooled FID
# fake = the folds' generations concatenated, real = the 200 unique test images.
# Only complete folds are linked in, so one bad fold cannot destroy a variant's
# pooled number; the fold count is reported next to it.
pool_links() {                    # srcdir subglob tag
  local src="$1" subg="$2" tag="$3" root="$4"
  rm -rf "${root}"; mkdir -p "${root}"
  local n=0
  for sub in "${src}"/sub-*; do
    [[ -d "${sub}" ]] || continue
    [[ -f "${sub}/${subg}/generated/199.png" ]] || continue
    ln -sfn "$(cd "${sub}" && pwd)" "${root}/$(basename "${sub}")"; n=$((n + 1))
  done
  echo "${n}"
}
for v in g2f_mem hcma_deploy; do
  PROOT="${OUT}/pooled_${v}"
  NFOLD="$(pool_links "${OUT}/gen" "${v}" "${v}" "${PROOT}")"
  echo "[pooled] ${v}: ${NFOLD} complete folds"
  if (( NFOLD >= 2 )); then
    "${PYTHON}" scripts/nda/eval_pooled_fid.py \
      --root "${PROOT}" --tag "${v}" --images-root "${IMAGES_ROOT}" \
      --output-json "${OUT}/fid_pooled_${v}.json" --device "${DEVICE}" \
      2>&1 | tee -a "${OUT}/logs/eval_pooled_fid.log" \
      || echo "[WARN] pooled FID failed for ${v}"
  fi
done
PROOT="${OUT}/pooled_sdedit_ll"
NFOLD="$(pool_links "${SDEDIT_ROOT}" "generation/sdedit_ll" sdedit_ll "${PROOT}")"
echo "[pooled] sdedit_ll: ${NFOLD} complete folds"
if (( NFOLD >= 2 )); then
  "${PYTHON}" scripts/nda/eval_pooled_fid.py \
    --root "${PROOT}" --tag sdedit_ll --images-root "${IMAGES_ROOT}" \
    --output-json "${OUT}/fid_pooled_sdedit_ll.json" --device "${DEVICE}" \
    2>&1 | tee -a "${OUT}/logs/eval_pooled_fid.log" \
    || echo "[WARN] pooled FID failed for sdedit_ll"
fi

# --------------------------------------------------------------- comparison
"${PYTHON}" scripts/nda/g2_compare_sota.py \
  --g2 "${STD7}/results.json" --sota "${SOTA_TABLE}" \
  --fid-glob "${OUT}/fid_pooled_*.json" --out "${OUT}/final_vs_sota.json" \
  2>&1 | tee "${OUT}/logs/comparison.log" || echo "[WARN] comparison failed"

echo "{\"pipeline\":\"g2f\",\"finished\":\"$(date -Iseconds)\"}" > "${OUT}/done.json"
echo "===== G2F complete @ $(date -Iseconds) ====="
