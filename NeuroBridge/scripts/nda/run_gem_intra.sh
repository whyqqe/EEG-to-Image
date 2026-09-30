#!/usr/bin/env bash
# ============================================================================
# UGE / sub-08 INTRA -- unified EEG2Text + EEG2Image, leak-free, disk-frugal.
#
# Unification is at the SHARED ENCODER and at the SCOREBOARD, not in a fused
# vector.  Language targets stay in CLIP-text; the generation condition stays in
# CLIP-image; VAE is the img2img init only.
#
#   E   intra `shared_r` (PurifiedFront OFF -- it did not beat this baseline)
#   L   EEG2Text ladder: L0 train-concept retrieval, L1 K=5 MI anchors,
#       L_emb = CLIP-text of the description.  Scored, never injected as IP.
#   V   EEG2Image: V_img = encode_image() 1024-d (MSE + cosine + in-batch NCE);
#       V_vae = low-freq latent used ONLY as SDEdit init.
#   G   IP-Adapter(calibrated V_img) + SDEdit(V_vae) + prompt in
#       {5-word, empty, generic}.
#
# ROWS
#   gem_ll_self      V_img + 5-word prompt          <- headline
#   gem_ll_noprompt  V_img, no prompt               <- prompt contribution
#   gem_generic      V_img + generic prompt
#   gem_sem_noprompt language embedding mapped to IP, no prompt  <- SPACE control
#   gem_ll_rawcal    uncalibrated V_img             <- calibration contribution
#   gem_stat / gem_noise / gem_oracle / gem_unrel / gem_swap
#
# EVERY ROW ABOVE IS LEAK-FREE.  The oracle row is a CEILING, never an input: it
# is built from the test image's description by a ridge fitted on TRAIN rows only,
# and it is reported as a ceiling.  Nothing is selected on it.
#
# DISK DISCIPLINE.  Generated PNGs (133 MB/row) are deleted the moment their
# metrics land; `last.pth` is deleted after the export; the raw EEG cache (281 MB)
# goes after training.  The CLIP image cache (68 MB) is kept -- it is shared.
# ============================================================================
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/uge/sub-08}"
COND="${NB_ROOT}/outputs/gem/cond_cache"
RAW="${NB_ROOT}/outputs/tdm/cache"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-26}"
SID="${SID:-8}"
SD="$(printf '%02d' "${SID}")"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
YEAR_TAG="$(date -Iseconds)"

# ---- A PRIORI FIXED.  Never selected on any test metric.
SD_STRENGTH="0.82"       # the established sdedit_ll operating point
GEN_STEPS="28"
GEN_GUIDANCE="5.0"
IP_SCALE="1.0"
# ---- GVM.  These are the a-priori settings of the new mechanisms.  None of them is
# chosen on a test metric: the arbitration gains are derived from HELD-IN TRAIN
# reliability spread inside `gem_train.py`, and the schedule is measured on held-in
# rows there too.
W_TEXT_DECODE="0.0"      # M2: the teacher-forced token CE is OFF.  See gem_train.py.
W_ANCHOR="1.0"           # AUXILIARY objective: the word-level read-out that makes this
                         # comparable to EEG-to-text work and makes `prompts_self`
                         # inspectable.  The generation condition does NOT come from
                         # here -- it comes from the embedding-level alignment
                         # (`conds["sem"]`), which is also generated as a PROMPT-FREE
                         # arm (`gem_sem_noprompt`) so its sufficiency is measured.
ANCHOR_TOPK="2"          # PER-FIELD CAP in an assembled prompt
ANCHOR_TOTAL="5"         # TOTAL words per prompt, ranked across fields.  5 is
                         # Brain-CLIPLM's measured optimum: only a handful of ordered
                         # semantic anchors survive EEG decoding, so a 20-word budget
                         # (the old 5-per-field rule) spends 15 slots on words the head
                         # predicted at chance level.
N_ANCHOR="384"
ANCHOR_MIN_COUNT="12"
W_NVOL="0.0"
NVOL_DIM="1280"
W_ARB="0.0"
W_FUSE_ROW="0.0"
W_FUSE_CLS="0.0"
W_CLIP_MSE="1.0"
W_CLIP_NCE="1.0"
ARB_SENS="4.0"
ARB_LO="0.66"            # the per-row strength range, symmetric about 0.82
ARB_HI="0.98"
ARB_IP_LO="0.70"
ARB_IP_HI="1.30"
W_SCHED="1"              # innovation 4: weights follow measured recoverability
SCHED_WARMUP="10"
SCHED_CLAMP="4.0"
SCHED_FLOOR="0.02"
ANCHOR_THR="0.0"
# `W_BALANCE` was removed together with the norm-balancing loss it weighted.

export HF_HOME="${HF_HOME:-/project/peilab/why/cache/hf_gem}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
mkdir -p "${OUT}/logs" "${OUT}/eval" "${COND}" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"

source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" >/dev/null 2>&1 || true
PYTHON="$(command -v python)"
echo "[env] PYTHON=${PYTHON} HF_HOME=${HF_HOME}"
"${PYTHON}" - <<'PY'
import sys
try:
    import torch, diffusers, transformers
except Exception as e:                                    # noqa: BLE001
    sys.exit(f"[FATAL] import failed: {e}")
if not torch.cuda.is_available():
    sys.exit("[FATAL] CUDA unavailable. A CPU fallback would train at a different "
             "speed and, for the CLIP cache stage, at a different cost.")
print(f"[env] torch {torch.__version__} diffusers {diffusers.__version__} "
      f"transformers {transformers.__version__} gpu {torch.cuda.get_device_name(0)}")
PY

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
have()    { [[ -e "$1" ]]; }

VC="${NB_ROOT}/outputs/sdedit_ll_full10/shared/vae_cache"
CLIP_TEXT="${NB_ROOT}/outputs/nda_ss/sub-08/clip_text"
CAPS="${NB_ROOT}/outputs/g2/captions"
ZC="${NB_ROOT}/outputs/ocf/intra_z/sub-${SD}"
ENC_CK="${NB_ROOT}/outputs/ocf/intra_enc/sub-${SD}/checkpoint_ss_calib_best.pth"
require "${VC}/train_vae_latents_f16.npy"
require "${CAPS}/captions_train.jsonl"
require "${CAPS}/captions_test.jsonl"
require "${CLIP_TEXT}/train/concept_phrases.json"

echo "===== [0] shared assets @ ${YEAR_TAG} ====="
# ---- CLIP image features: the PROJECTED 1024-d feature, with the row-order audit
if [[ ! -f "${COND}/clip_img1024_train.npy" ]]; then
  echo "===== [0a] CLIP ViT-H-14 image features (train+test, row order audited) ====="
  "${PYTHON}" scripts/nda/gem_clip_img.py --out-dir "${COND}" \
    --images-root "${IMAGES_ROOT}" --captions-dir "${CAPS}" \
    --targets-dir "${NB_ROOT}/outputs/g2/targets" --device "${DEVICE}" \
    2>&1 | tee "${OUT}/logs/clip_img.log"
else
  echo "[SKIP] CLIP image features exist ($(du -h "${COND}/clip_img1024_train.npy" | cut -f1))"
fi
require "${COND}/clip_img1024_train.npy"
require "${COND}/clip_img1024_test.npy"

# ---- ground-truth descriptions for the TEST rows.  These are the ORACLE row's
# input and the reference for what the decoded prompt should have said.  They are
# never a model input and nothing is selected on them.
PROMPT_BAK="${NB_ROOT}/outputs/gem/prompts_cache"
mkdir -p "${PROMPT_BAK}"
if [[ ! -f "${PROMPT_BAK}/prompts_true_test.json" ]]; then
  "${PYTHON}" - <<PY
import json
from pathlib import Path
rows = [json.loads(l) for l in Path("${CAPS}/captions_test.jsonl").read_text(
    encoding="utf-8").splitlines() if l.strip()]
out = ["%s. %s. %s. %s. %s." % (
    Path(r["path"]).parent.name.split("_", 1)[1].replace("_", " "),
    r.get("overall", ""), r.get("subject", ""), r.get("background", ""),
    r.get("detail", "")) for r in rows]
Path("${PROMPT_BAK}/prompts_true_test.json").write_text(json.dumps(out, indent=1),
                                                       encoding="utf-8")
print(f"[prompts] backed up {len(out)} ground-truth test descriptions")
print("  e.g.", out[0][:130])
PY
fi
require "${PROMPT_BAK}/prompts_true_test.json"
# the TRAIN descriptions and their concepts, so the pool is re-derivable from a
# permanent location instead of only from the scratch captions directory
if [[ ! -f "${PROMPT_BAK}/prompts_train.json" ]]; then
  cp "${CAPS}/captions_train.jsonl" "${PROMPT_BAK}/captions_train.jsonl"
fi

# ---- intra-subject encoder (this subject alone) and the shared_r export
if [[ ! -f "${ENC_CK}" ]]; then
  echo "===== [0b] intra EEG encoder (sub-${SD} alone) ====="
  "${PYTHON}" scripts/nda/nda_ss_pretrain.py \
    --train-subjects "${SID}" --calib-subject "${SID}" \
    --output-dir "${NB_ROOT}/outputs/ocf/intra_enc/sub-${SD}" \
    --pretrain-epochs 30 --calib-epochs 15 --batch-size 512 --device "${DEVICE}" \
    2>&1 | tee "${OUT}/logs/enc_s${SD}.log"
fi
require "${ENC_CK}"
if [[ ! -f "${ZC}/shared_r_test.npy" ]]; then
  echo "===== [0c] export shared_r ====="
  "${PYTHON}" scripts/nda/ocf_export_intra_z.py --subject "${SID}" \
    --checkpoint "${ENC_CK}" --out "${NB_ROOT}/outputs/ocf/intra_z" \
    --device "${DEVICE}" 2>&1 | tee "${OUT}/logs/export_z_s${SD}.log"
fi
require "${ZC}/shared_r_train.npy"
require "${ZC}/shared_r_test.npy"

# ---- raw EEG cache (hard requirement for both arms)
if [[ ! -f "${RAW}/sub${SD}_test_eeg.npy" || ! -f "${RAW}/sub${SD}_train_eeg.npy" ]]; then
  echo "===== [0d] raw EEG cache ====="
  "${PYTHON}" scripts/nda/tdm_gate0.py --subject "${SID}" --device "${DEVICE}" \
    --targets-dir "${NB_ROOT}/outputs/g2/targets" --cache-dir "${RAW}" \
    --out-json "${NB_ROOT}/outputs/tdm/gate0_sub${SD}.json" --cache-only \
    2>&1 | tee "${OUT}/logs/raw_cache_s${SD}.log"
fi
require "${RAW}/sub${SD}_train_eeg.npy"
require "${RAW}/sub${SD}_test_eeg.npy"

echo "===== [1] leak audit: prompt gallery vs test concepts ====="
"${PYTHON}" - <<PY
import json, sys
from pathlib import Path
gal = json.loads((Path("${CLIP_TEXT}") / "train" / "concept_phrases.json").read_text(
    encoding="utf-8"))
te = {Path(json.loads(l)["path"]).parent.name.split("_", 1)[1].replace("_", " ")
      for l in Path("${CAPS}/captions_test.jsonl").read_text(encoding="utf-8"
      ).splitlines() if l.strip()}
inter = {str(c).strip().lower() for c in gal} & {c.lower() for c in te}
print(f"[audit] gallery {len(gal)} | test concepts {len(te)} | intersection {len(inter)}")
if inter:
    sys.exit(f"[FATAL] the pool shares {len(inter)} concepts with the test set: "
             f"{sorted(inter)[:5]}. Refusing to generate.")
print("[audit] ok")
PY

# ============================================================================
train_arm() {                     # tag use_front noise_arm
  local tag="$1" uf="$2" na="$3"
  local d="${OUT}/${tag}"
  mkdir -p "${d}/conds" "${d}/prompts"
  if [[ -f "${d}/conds/ip_clip_test.npy" ]]; then
    echo "[SKIP] train ${tag} (condition already exported)"; return 0
  fi
  echo "===== [2] train ${tag} (front=${uf} noise=${na}) @ $(date -Iseconds) ====="
  "${PYTHON}" scripts/nda/gem_train.py \
    --train-subjects "${SID}" --test-subject "${SID}" \
    --raw-cache "${RAW}" --z-root "${NB_ROOT}/outputs/ocf/intra_z" \
    --targets-dir "${NB_ROOT}/outputs/g2/targets" --clip-img-dir "${COND}" \
    --captions-dir "${CAPS}" --vae-cache "${VC}" \
    --clip-text-dir "${CLIP_TEXT}" \
    --clip-patch-npy "${NB_ROOT}/outputs/tdm/clip_patch/train_patch_f16.npy" \
    --out "${d}" --epochs "${EPOCHS}" --device "${DEVICE}" \
    --use-front "${uf}" --noise-arm "${na}" \
    --w-text-decode "${W_TEXT_DECODE}" \
    --w-anchor "${W_ANCHOR}" --anchor-topk "${ANCHOR_TOPK}" \
    --anchor-total "${ANCHOR_TOTAL}" \
    --n-anchor "${N_ANCHOR}" --anchor-min-count "${ANCHOR_MIN_COUNT}" \
    --w-nvol "${W_NVOL}" --nvol-proj-dim "${NVOL_DIM}" \
    --w-arb "${W_ARB}" --arb-sens "${ARB_SENS}" \
    --w-fuse-row "${W_FUSE_ROW}" --w-fuse-cls "${W_FUSE_CLS}" \
    --w-clip-mse "${W_CLIP_MSE}" --w-clip-nce "${W_CLIP_NCE}" \
    --arb-lo "${ARB_LO}" --arb-hi "${ARB_HI}" \
    --arb-ip-lo "${ARB_IP_LO}" --arb-ip-hi "${ARB_IP_HI}" \
    --arb-center "${SD_STRENGTH}" \
    --w-sched "${W_SCHED}" --sched-warmup "${SCHED_WARMUP}" \
    --sched-clamp "${SCHED_CLAMP}" --sched-floor "${SCHED_FLOOR}" \
    --anchor-thr "${ANCHOR_THR}" \
    2>&1 | tee "${OUT}/logs/train_${tag}.log"
  require "${d}/conds/ip_clip_test.npy"
}

train_arm full  0 0
train_arm noise 0 1

echo "===== [3] did each arm train? ====="
# The delimiter is QUOTED (`<<'PY'`), and `${OUT}` is passed through the environment
# instead of being interpolated into the source.  This is not style: with an unquoted
# `<<PY` bash performs command substitution on the body, so any BACKTICK in a printed
# message gets executed as a command.  That is exactly how job 571020 died -- the note
# string "`gem_arb` and `gem_fixalpha` will differ only by noise" made bash run
# `gem_arb` and `gem_fixalpha`, print "command not found", and substitute empty
# strings, and the diagnostic then raised `KeyError` on the next line.  A quoted
# delimiter makes the whole class of bug impossible, at the cost of having to pass
# every variable explicitly.
# THE `if ! ... then` IS NOT DECORATION.  This block is a DIAGNOSTIC, and on job
# 571020 a `KeyError` in it aborted the script after all three arms had trained and
# before a single image was generated or scored -- the GPU budget was spent and the
# run produced no evaluation metrics at all.  A failure in a diagnostic must never
# cost the metrics.  `if !` also keeps the shell from exiting under `set -e`, which
# a bare `|| echo` on a multi-line heredoc does not reliably do.
if ! GEM_OUT="${OUT}" "${PYTHON}" - <<'PY'
import json, os, pathlib
OUT = os.environ["GEM_OUT"]
for tag in ("full", "noise"):
    p = pathlib.Path(OUT) / tag / "gem_report.json"
    if not p.is_file():
        print(f"  [WARN] {tag}: no report"); continue
    d = json.load(open(p)); f = d["frozen_condition_check"]
    print(f"  {tag:<8} trained={d['trained']} skips={d['grad_skips']} "
          f"epoch={d['best']['epoch']} score={d['best']['score']:.4f} | "
          f"pool->CLIPtext cos {f['pool_to_clip_text_cos']:.4f} | "
          f"row-identity {f['row_identity_acc_pool']:.4f} "
          f"(chance {f['chance_row_identity']:.4f}) | "
          f"clip cos {f['clip_tower_cos']:.4f}")
    if not d["trained"] or d["grad_skips"] > 0:
        print(f"  [WARN] {tag}: did NOT train cleanly; its rows are still generated so "
              f"the failure is visible in the metrics")
    cc = d.get("condition_concentration", {})
    # POST-TRAINING degeneracy.  The init guard cannot see a collapse that DEVELOPS
    # during training, and that is what happened on the first clean sub-08 run: the
    # guard passed at init (0.7837) and the trained model exported a CONSTANT
    # condition (fused row-cos ~1.0, c_self 0.999995).  Every semantic metric then
    # came out at chance while the log said the training was healthy.  This is the
    # same measurement, after training, on what was actually exported.
    if cc:
        print(f"  {tag:<8} condition: fused row-cos {cc.get('fused_rowcos', float('nan')):.4f} "
              f"c_self {cc.get('c_self_predicted', float('nan')):.6f} "
              f"clip row-cos {cc.get('clip_rowcos', float('nan')):.4f}")
        if cc.get("degenerate"):
            print(f"  [WARN] {tag}: CONDITION IS DEGENERATE (row-cos "
                  f"{cc.get('fused_rowcos'):.4f} > 0.90). It is nearly a constant, so "
                  f"it cannot be row-specific and its generated images cannot be "
                  f"EEG-driven. Read every semantic metric below as chance-by-"
                  f"construction, not as a result about this architecture.")
    # ---- GVM mechanism verdicts, printed next to the training numbers so a
    # mechanism that did NOT engage is visible before any metric is read.
    a = d.get("M2_anchor_ladder")
    if a:
        print(f"  {tag:<8} M2 anchors: vocab {a['n_vocab']} topk {a['topk_per_field']} "
              f"| prompt unique {a['prompt_unique']}/{a['prompt_n']} "
              f"jaccard {a['prompt_jaccard']:.4f} "
              f"rows-with-anchor {a['rows_with_any_anchor']:.3f} | "
              f"recall " + " ".join(f"{g}={a['per_field_recall'][g]:.3f}"
                                    for g in ("overall", "subject", "background", "detail")))
    v = d.get("M3_neural_visibility")
    if v:
        print(f"  {tag:<8} M3 layer A/B (held-in): direct1024 {v['cos_direct_1024']:.4f} "
              f"vs penult1280 {v['cos_penultimate_1280_through_proj']:.4f} -> "
              f"{v['chosen']} (margin {v['margin']:.4f})")
    r = d.get("M4_arbitration")
    if r:
        # TOLERANT BY DESIGN.  This block reads keys that a previous version of
        # `gem_train.py` wrote and that a later version removed, and a missing
        # diagnostic key must never kill a job whose only remaining work is to
        # generate and evaluate -- which is exactly what happened to 571020, where a
        # `KeyError` here aborted the run AFTER all three arms had trained.
        def _g(k, fmt="{:.4f}", dflt="n/a"):
            v = r.get(k, None)
            return dflt if v is None else fmt.format(v)
        print(f"  {tag:<8} M4 arbitration: alpha {_g('alpha_mean')}"
              f"+-{_g('alpha_sd')} | ip_scale {_g('ipscale_mean')}"
              f"+-{_g('ipscale_sd')}")
        # the CALIBRATION numbers are the ones that carry information; the old
        # corr(alpha, r_est) was +-1.0 by construction (alpha is an affine function of
        # r_est) and is gone
        if "corr_r_est_sem_vs_achieved" in r:
            print(f"  {tag:<8}   calibration r_est vs ACHIEVED: sem "
                  f"{_g('corr_r_est_sem_vs_achieved', '{:+.3f}')} (null "
                  f"{_g('corr_r_est_sem_vs_permuted', '{:+.3f}')}) | str "
                  f"{_g('corr_r_est_str_vs_achieved', '{:+.3f}')} (null "
                  f"{_g('corr_r_est_str_vs_permuted', '{:+.3f}')})")
        if r.get("alpha_sd", 1.0) < 0.005:
            print(f"  [WARN] {tag}: M4 arbitration is INERT (alpha sd "
                  f"{_g('alpha_sd')}); the arbitrated and fixed arms will differ only "
                  f"by noise and M4 must be reported as untested, not as a null.")
    sc = d.get("innovation4_weight_schedule", {})
    if sc.get("applied_epoch") is not None:
        print(f"  {tag:<8} innov4 schedule at ep {sc['applied_epoch']}: R2 "
              + " ".join(f"{k}={vv:.4f}" for k, vv in sc["r2"].items())
              + " -> mult " + " ".join(f"{k}={vv:.3f}" for k, vv in sc["mult"].items()))
PY
then
  echo "[WARN] stage [3] diagnostics FAILED. This is NON-FATAL: generation and"
  echo "[WARN] evaluation still run. A diagnostic must never spend the GPU budget"
  echo "[WARN] and then produce no metrics (that is how job 571020 died)."
fi

# ============================================================================
# `eval_row` must be DEFINED before `gen_row` calls it.  In bash a function is only
# visible after its definition is executed, so the previous order ran the
# generation and then failed with `eval_row: command not found` -- the images would
# have been produced and thrown away unscored.
eval_row() {                      # tag gen_dir
  local tag="$1" gdir="$2"
  local ev="${OUT}/eval/s${SD}_${tag}.json"
  [[ -f "${ev}" ]] && { echo "[SKIP] eval ${tag}"; return 0; }
  [[ -f "${gdir}/199.png" ]] || { echo "[WARN] eval ${tag}: no images"; return 0; }
  echo "--- eval ${tag}"
  "${PYTHON}" scripts/nda/eval_official_seven_dir.py \
    --gen-dir "${gdir}" --output-json "${ev}" --tag "${tag}" \
    --images-root "${IMAGES_ROOT}" --device "${DEVICE}" --skip-if-exists \
    2>&1 | tee -a "${OUT}/logs/eval_s${SD}.log" \
    || { echo "[WARN] eval ${tag} failed"; return 0; }
  # delete the images ONLY once the metrics exist AND the file is non-empty: a
  # failed evaluation must not cost the generation it was about to score
  if [[ -s "${ev}" && "${KEEP_IMAGES:-0}" != "1" ]]; then
    rm -rf "${gdir}"
  fi
}

gen_row() {                       # tag cond prompts|none vae_scaled [strength_npy] [ipscale_npy]
  local tag="$1" cond="$2" pf="$3" vd="$4"
  local snpy="${5:-}" isnpy="${6:-}"
  local gdir="${OUT}/gen/${tag}"
  local ev="${OUT}/eval/s${SD}_${tag}.json"
  if [[ -f "${ev}" ]]; then echo "[SKIP] ${tag} (already scored)"; return 0; fi
  if [[ ! -f "${cond}" ]]; then echo "[WARN] ${tag}: missing condition ${cond}"; return 0; fi
  # `none` means "generate from the CONDITION ALONE, with no text prompt".  This is a
  # first-class arm, not an omission: the semantic tower's primary objective is the
  # embedding-level alignment (see `M2_anchor_ladder` in the report), and the only way
  # to show that the read-out carries the semantics on its own is to run a row where
  # the assembled words are absent.  `--prompts-json ""` makes the generator build the
  # negative prompt as "" too (see `neg = args.negative_prompt if prompt else ""`), so
  # both conditioning strings are empty and the IP-Adapter embedding is the sole input.
  local pflag=()
  if [[ "${pf}" == "none" || -z "${pf}" ]]; then
    pflag=(--prompts-json "")
  else
    if [[ ! -f "${pf}" ]]; then echo "[WARN] ${tag}: missing prompts ${pf}"; return 0; fi
    pflag=(--prompts-json "${pf}")
  fi
  if [[ ! -f "${vd}" ]]; then echo "[WARN] ${tag}: missing vae init ${vd}"; return 0; fi
  echo "===== [4] gen ${tag} @ $(date -Iseconds) ====="
  local extra=()
  if [[ -n "${snpy}" ]]; then
    [[ -f "${snpy}" ]] || { echo "[WARN] ${tag}: missing ${snpy}"; return 0; }
    extra+=(--strength-npy "${snpy}")
  fi
  if [[ -n "${isnpy}" ]]; then
    [[ -f "${isnpy}" ]] || { echo "[WARN] ${tag}: missing ${isnpy}"; return 0; }
    extra+=(--ip-scale-npy "${isnpy}")
  fi
  "${PYTHON}" scripts/nda/generate_atm_aligned_decode.py \
    --mode sdedit --embed-npy "${cond}" "${pflag[@]}" \
    --vae-latent-npy "${vd}" \
    --output-dir "${gdir}" --tag "${tag}" \
    --strength "${SD_STRENGTH}" --ip-scale "${IP_SCALE}" \
    "${extra[@]}" \
    --gen-steps "${GEN_STEPS}" --gen-guidance "${GEN_GUIDANCE}" \
    --device "${DEVICE}" 2>&1 | tee "${OUT}/logs/gen_${tag}.log" \
    || { echo "[WARN] gen ${tag} failed"; return 0; }
  eval_row "${tag}" "${gdir}/generated"
}

# quantile calibration: the IP adapter responds to a condition's CONCENTRATION, so
# an uncalibrated condition is partly a statement about the read-out's own
# shrinkage.  Label-free: the reference is the TRAIN concept bank, never test data.
#
# APPLIED TO EVERY ARM, not just `full`.  The first version calibrated `full` only
# and generated the `nofront` and `noise` arms from their RAW conditions, which is
# a confounded ablation: if calibration helps, then part of any "full beats
# nofront" gap is the calibration, and part is the front end, and the run cannot
# tell them apart.  Calibrating all three against the SAME reference bank removes
# the confound, so the arms differ only in how the condition was produced.
F="${OUT}/full"
P="${OUT}/noise"
V="${F}/pred_vae_test_scaled.npy"
require "${F}/conds/ip_clip_test.npy"
require "${F}/conds/ip_oracle_test.npy"
require "${F}/conds/ip_static_test.npy"
require "${P}/conds/ip_clip_test.npy"
require "${V}"
# concentration is matched onto the TRAIN CLIP-IMAGE bank: V_img lives in that
# space.  The previous reference was `text_concept_clip.npy` (CLIP-text), which
# quantile-matched an image condition onto a text distribution.

cal_arm() {                       # arm_dir tag -> calibrated condition path
  # The RETURN VALUE is the path, so NOTHING ELSE may write to stdout: the first
  # version let `tee` echo the calibration log into the command substitution, so
  # `CC="$(cal_arm ...)"` captured the log text and every downstream `np.load`
  # got a Python SyntaxError.  The log goes to its file and to stderr; stdout
  # carries the path and nothing else.
  local d="$1" tag="$2"
  local o="${d}/conds/ip_cal_test.npy"
  if [[ ! -f "${o}" ]]; then
    "${PYTHON}" scripts/nda/gem_calib.py \
      --in "${d}/conds/ip_clip_test.npy" --out "${o}" \
      --ref "${COND}/clip_img1024_train.npy" --tag "${tag}" \
      >> "${OUT}/logs/calib_s${SD}.log" 2>&1 \
      || echo "[WARN] calib ${tag} failed; falling back to the raw condition" >&2
  fi
  if [[ -f "${o}" ]]; then printf '%s\n' "${o}"
  else printf '%s\n' "${d}/conds/ip_clip_test.npy"; fi
}

CC="$(cal_arm "${F}" self)"
CN="$(cal_arm "${P}" noise)"

# the embedding-alignment condition, calibrated through the same reference bank as
# every other arm.  `conds['sem']` is `ridge(W_ci, ridge(pool -> CLIP-text))`: the EEG
# latent mapped to the CLIP-text encoding of the description and then into the
# IP-Adapter image space.  No word is generated anywhere on this path, so calibrating
# it with the SAME function (rather than a bespoke one) keeps it comparable to `full`.
cal_file() {                      # in_file tag -> calibrated path, stdout only
  local src="$1" tag="$2"
  local o="${F}/conds/ip_cal_${tag}.npy"
  if [[ ! -f "${o}" ]]; then
    "${PYTHON}" scripts/nda/gem_calib.py \
      --in "${src}" --out "${o}" \
      --ref "${COND}/clip_img1024_train.npy" --tag "${tag}" \
      >> "${OUT}/logs/calib_s${SD}.log" 2>&1 \
      || echo "[WARN] calib ${tag} failed; falling back to the raw condition" >&2
  fi
  if [[ -f "${o}" ]]; then printf '%s\n' "${o}"
  else printf '%s\n' "${src}"; fi
}
CS="$(cal_file "${F}/conds/ip_sem_test.npy" sem)"

gen_row gem_ll_self  "${CC}"                              "${F}/prompts/prompts_self.json"    "${V}"
# ---- THE CALIBRATION ABLATION, AND WHY IT IS NOW A HEADLINE ROW RATHER THAN A
# FOOTNOTE.  `gem_ll_self` consumes a CALIBRATED condition; this row consumes the
# model's RAW `ip_fused_auto_test.npy` with the same prompt and the same VAE init, so
# the difference between the two rows IS the calibration's effect on generation.
#
# It is not optional.  Measured on the sub-08 conditions, matching the concentration
# onto the train bank LOWERS row-cosine from 0.8792 to 0.3939 and RAISES row-identity
# retrieval from 0.0500 to 0.1450 (2.9x), while `cos` to the true image embedding
# FALLS from 0.6657 to 0.5695.  That divergence is the point: cos is dominated by the
# row-independent constant, so it PREFERS the uncalibrated condition and cannot be
# used to judge this change.  Only the generation metrics can, so they are measured.
gen_row gem_ll_rawcal "${F}/conds/ip_clip_test.npy"       "${F}/prompts/prompts_self.json"    "${V}"
# ---- the two PROMPT-FREE arms.  `gem_ll_noprompt` re-runs the main condition with
# both conditioning strings empty, so the difference from `gem_ll_self` is the text
# condition's contribution and nothing else; `gem_sem_noprompt` runs the
# embedding-alignment condition with no prompt, which is the read-out the semantic
# tower is primarily trained for.  Together they answer "does this need generated
# text at all" as a measurement rather than as an assumption.
gen_row gem_ll_noprompt  "${CC}"                          none                                "${V}"
gen_row gem_sem_noprompt "${CS}"                          none                                "${V}"
gen_row gem_stat     "${F}/conds/ip_static_test.npy"      "${F}/prompts/prompts_self.json"    "${V}"
gen_row gem_noise    "${CN}"                              "${P}/prompts/prompts_self.json"    "${V}"
gen_row gem_generic  "${CC}"                              "${F}/prompts/prompts_generic.json" "${V}"
gen_row gem_oracle   "${F}/conds/ip_oracle_test.npy"      "${PROMPT_BAK}/prompts_true_test.json" "${V}"

# ---- COUNTERFACTUALS, built by PERMUTATION of the model's own outputs.
#
# No oracle text is involved: the prompts used here are the model's OWN decoded
# prompts from other rows, and the conditions are the model's own exported
# conditions from other rows.  They enter as generation-only permutations of
# already-exported arrays, so they cost no training and cannot leak.
#
#   gem_unrel  prompt_i with condition from a DIFFERENT row.  This is the sharpest
#              control in the run: the text is exactly right, only the EEG-derived
#              condition is somebody else's.  Anything it scores above chance is
#              attributable to the TEXT and not to the condition, so
#              `self - unrel` is the honest measure of whether the condition is read.
#   gem_swap   condition_i with a DIFFERENT row's decoded prompt.  Same idea from
#              the other side: if the condition overrode the prompt, this would
#              still score well on the concept; if it does not, `self - swap`
#              measures how much the prompt is doing.
PERM_DIR="${OUT}/permuted"
mkdir -p "${PERM_DIR}/conds"
if [[ ! -f "${PERM_DIR}/conds/ip_unrel_test.npy" ]]; then
  "${PYTHON}" - <<PY
import json
from pathlib import Path
import numpy as np
rng = np.random.default_rng(20260912)
c = np.load("${CC}")
P = Path("${PERM_DIR}"); P.mkdir(parents=True, exist_ok=True)
(P / "conds").mkdir(parents=True, exist_ok=True)
n = len(c)
# a derangement: every row gets ANOTHER row's condition, nobody keeps their own
perm = rng.permutation(n)
for i in np.where(perm == np.arange(n))[0]:
    j = (i + 1) % n
    perm[i], perm[j] = perm[j], perm[i]
assert (perm != np.arange(n)).all(), "permutation is not a derangement"
np.save(P / "conds" / "ip_unrel_test.npy", c[perm])
np.save(P / "conds" / "ip_swap_test.npy", c)          # same condition, other prompt
pr = json.loads(Path("${F}/prompts/prompts_self.json").read_text(encoding="utf-8"))
(P / "prompts").mkdir(parents=True, exist_ok=True)
(P / "prompts" / "prompts_unrel.json").write_text(json.dumps(pr, indent=1),
                                                 encoding="utf-8")
(P / "prompts" / "prompts_swap.json").write_text(
    json.dumps([pr[j] for j in perm], indent=1), encoding="utf-8")
print(f"[permute] derangement over {n} rows: unrel condition shuffled, "
      f"swap prompt shuffled (fixed seed, no test text involved)")
PY
fi
gen_row gem_unrel "${PERM_DIR}/conds/ip_unrel_test.npy"      "${PERM_DIR}/prompts/prompts_unrel.json" "${V}"
gen_row gem_swap  "${PERM_DIR}/conds/ip_swap_test.npy"       "${PERM_DIR}/prompts/prompts_swap.json"  "${V}"

# reference rows: the SAME evaluation code and GT cache, so the columns are
# comparable.  They live outside OUT, so this neither moves nor deletes them.
if [[ ! -f "${OUT}/eval/s00_sdedit_ll.json" ]]; then
  "${PYTHON}" scripts/nda/eval_official_seven_dir.py \
    --gen-dir "${NB_ROOT}/outputs/sdedit_ll_full10/sub-08/generation/sdedit_ll/generated" \
    --output-json "${OUT}/eval/s00_sdedit_ll.json" --tag sdedit_ll \
    --images-root "${IMAGES_ROOT}" --device "${DEVICE}" --skip-if-exists \
    2>&1 | tee -a "${OUT}/logs/eval_refs.log" || echo "[WARN] ref eval failed"
fi
if [[ ! -f "${OUT}/eval/s00_g3f_ll_selfgate.json" ]]; then
  "${PYTHON}" scripts/nda/eval_official_seven_dir.py \
    --gen-dir "${NB_ROOT}/outputs/g3f/gen/sub-08/g3f_ll_selfgate/generated" \
    --output-json "${OUT}/eval/s00_g3f_ll_selfgate.json" --tag g3f_ll_selfgate \
    --images-root "${IMAGES_ROOT}" --device "${DEVICE}" --skip-if-exists \
    2>&1 | tee -a "${OUT}/logs/eval_refs.log" || echo "[WARN] ref eval failed"
fi

echo "===== [5] analysis (towers / grounding / attribution) @ $(date -Iseconds) ====="
"${PYTHON}" scripts/nda/gem_ground.py --out "${OUT}" --sid "${SID}" \
  --captions-test "${CAPS}/captions_test.jsonl" \
  --report "${OUT}/grounding_report.txt" \
  2>&1 | tee "${OUT}/logs/ground_s${SD}.log" \
  || echo "[WARN] grounding analysis failed; the eval JSONs are untouched"

echo "===== [5b] control validity + retrieval (numerical, always) ====="
# THE NUMBERS THAT COS ALONE CANNOT GIVE, AND WHY THEY ARE NOT LEFT TO A HUMAN.
#
# Job 571020 trained all three arms and then died in a DIAGNOSTIC before stage [4],
# so a run that had already spent its GPU budget produced zero evaluation metrics.
# A diagnostic must not be able to do that.  Two structural changes:
#   * every diagnostic block in this script is non-fatal (`|| echo [WARN]`), and the
#     Python bodies are quoted-delimiter heredocs;
#   * THIS stage is unconditional, runs AFTER generation and eval, and returns 0
#     even when a section cannot be computed -- an uncomputable measurement is a
#     `null` in the JSON, never a failed job.
#
# It writes: the centreline (a row-independent constant scores +0.6275 in the image
# space, so cos +0.66 is a +0.03 margin, not a result), the ROW-WISE agreement
# between the ablation arms and the full arm (the regression test for a control
# that silently does not control anything), and row-identity retrieval on both the
# full condition and its constant-removed residual, which is the honest headline.
"${PYTHON}" scripts/nda/gvm_baseline_check.py \
  --out-dir "${OUT}" --bank "${COND}/clip_img1024_test.npy" \
  --captions "${CAPS}" --json "${OUT}/reports/measure.json" \
  2>&1 | tee "${OUT}/logs/measure_s${SD}.log" \
  || echo "[WARN] measurement stage failed; eval JSONs and reports are untouched"

echo "===== [6] cleanup ====="
# resume artifacts: the exported conditions and the reports are what matter
rm -f "${OUT}"/*/last.pth
# the activation dumps were consumed by stage [5] (~250 MB); the reports carry the
# statistics they were saved for
rm -f "${OUT}"/*/acts_train.npz "${OUT}"/*/acts_test.npz
# the raw EEG cache is regenerable in ~90 s per split
rm -f "${RAW}/sub${SD}_train_eeg.npy" "${RAW}/sub${SD}_test_eeg.npy" \
      "${RAW}/sub${SD}_train_row.npy" "${RAW}/sub${SD}_test_row.npy"
echo "[cleanup] kept: reports, fingerprints, conditions, prompts, eval JSONs, best.pth"
du -sh "${OUT}" | sed 's/^/[cleanup] OUT now /'
echo "===== done @ $(date -Iseconds) ====="
