#!/usr/bin/env bash
# G2 pipeline: granularity-factorised PARALLEL dual tower + CFM condition synthesiser.
#
# Protocol pair on the same test subject (sub-08):
#   intra : fit on sub-08 only            (pure intra-subject)
#   inter : fit on the other 9 subjects   (leave-one-subject-out; sub-08 never seen)
#
# Both share the SAME frozen encoder latent (`z_eeg_proj`, 512-d) and the same
# targets, so the only difference is the fitting set. The cross-subject module and
# EEG encoder are untouched, as requested.
#
# Stage list (each stage is skipped when its outputs already exist, so the job is
# resumable across preemption):
#   [0] VLM captions  (4 granularities per image, one structured call per image)
#   [1] targets       (4 text granularities + image encoding + structure + texture)
#   [2] train intra
#   [3] train inter (LOSO)
#   [4] generation    (condition variants, see below)
#   [5] evaluation    (standard-7 + per-row FID + pooled FID)
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/g2}"
CAPS="${CAPS:-${OUT}/captions}"
TARGETS="${TARGETS:-${OUT}/targets}"
SUBJECT="${SUBJECT:-8}"
DEVICE="${DEVICE:-cuda:0}"
STAGES="${STAGES:-012345}"
PYTHON="${PYTHON:-python}"
PY_CAP="${PY_CAP:-${NB_ROOT}/.venv-cap/bin/python}"
CAP_MODEL="${CAP_MODEL:-/home/sbaiae/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct}"
CAP_BATCH="${CAP_BATCH:-8}"
CAP_LIMIT="${CAP_LIMIT:-0}"
EPOCHS="${EPOCHS:-60}"
BATCH="${BATCH:-512}"
Z_ROOT="${Z_ROOT:-${NB_ROOT}/outputs/hcma_10subj}"
IP_TRAIN="${IP_TRAIN:-/project/peilab/why/eeg-brainit/outputs/atm_bridge/clip_img_train_1024.npy}"
IP_TEST="${IP_TEST:-/project/peilab/why/eeg-brainit/outputs/atm_bridge/clip_img_test_1024.npy}"
HCMA_PROMPTS="${HCMA_PROMPTS:-${NB_ROOT}/outputs/hcma_10subj/prompts/prompts_full_hcma_test.json}"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
CUT="${CUT:-0.0625}"
GEN_STEPS="${GEN_STEPS:-28}"
GEN_GUIDANCE="${GEN_GUIDANCE:-5.0}"
IP_SCALE="${IP_SCALE:-1.0}"

# ------------------------------------------------------------------- environment
# MANDATORY, and the single cause of the 2026-09-11 failure. Without this the job
# runs on the SYSTEM interpreter plus the user's site-packages rather than the
# project venv: `python` resolved to transformers 4.36.0 + diffusers 0.30.0 +
# torch 2.5.0+cu124 from ~/.local/lib/python3.11/site-packages, and every
# generation stage died with "cannot import name 'EncoderDecoderCache' from
# 'transformers'", because that diffusers requires a newer transformers than 4.36.
# All 11 folds trained, then all 11 generation stages failed at import: the night
# produced zero images. Sources the venv, PYTHONNOUSERSITE=1, and the HF/Triton
# caches under /project.
# shellcheck disable=SC1091
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh"

SID="$(printf "sub-%02d" "${SUBJECT}")"
if [[ "${SUBJECT}" -ge 10 ]]; then STAG="sub-${SUBJECT}"; else STAG="sub-0${SUBJECT}"; fi
# NB: never use the `[[ ... ]] && VAR=...` idiom under `set -e`; a false test
# returns 1 and aborts the whole job.

mkdir -p "${OUT}" "${CAPS}" "${TARGETS}" "${NB_ROOT}/outputs/slurm" "${OUT}/logs"
cd "${NB_ROOT}"
echo "{\"pipeline\":\"g2\",\"subject\":${SUBJECT},\"stages\":\"${STAGES}\",\"started\":\"$(date -Iseconds)\"}" \
  > "${OUT}/job_running.json"

has_stage() { [[ "${STAGES}" == *"$1"* ]]; }
require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }

# ===================================================================== [0] captions
# Runs in an ISOLATED environment: the Qwen2.5-VL weights live in the user's default
# HF cache, while the shared pipeline env repoints HF_HUB_CACHE (and HOME) elsewhere
# and pins transformers 4.46, which is too old for Qwen2.5-VL. Hence a separate venv
# with transformers 4.57 and an explicitly preserved cache path.
if has_stage 0; then
  echo "===== [0] captions @ $(date -Iseconds) ====="
  "${PY_CAP}" -c "import transformers; print('caption venv transformers', transformers.__version__)" \
    || { echo "[FATAL] caption venv missing; create with: python -m venv --system-site-packages .venv-cap && .venv-cap/bin/pip install 'transformers>=4.49' accelerate qwen-vl-utils" >&2; exit 1; }
  require "${CAP_MODEL}"
  for SPLIT in train test; do
    CL="${CAPS}/captions_${SPLIT}.jsonl"
    # Always invoke: the caption script skips rows already present, so a resumed
    # job continues where it stopped instead of being blocked by a partial file.
    #
    # Cache redirection is mandatory here: this stage deliberately runs with the
    # user's real HOME (so the Qwen2.5-VL weights in ~/.cache/huggingface can be
    # read), but /home was measured at 100% full (0 bytes available). Triton JIT
    # and the torch kernel cache both write, and they aborted the run with
    # "OSError: [Errno 28] No space left on device: /home/sbaiae/.triton/cache/...".
    # All writable caches are therefore pointed at the project filesystem.
    # PYTHONNOUSERSITE is unset because the pipeline-wide `source` above sets it:
    # .venv-cap was built with --system-site-packages off the SYSTEM interpreter,
    # so it needs the user site-packages visible, which is the state in which the
    # 16740 captions were successfully generated.
    CAP_CACHE="${CAP_CACHE:-/project/peilab/why/cache/eeg-brainit/capcache}"
    mkdir -p "${CAP_CACHE}"/{triton,torch_kernels,cuda_jit,inductor,xdg}
    env -u PYTHONPATH -u HF_HOME -u TRANSFORMERS_CACHE -u PYTHONNOUSERSITE \
      HOME=/home/sbaiae \
      HF_HUB_CACHE=/home/sbaiae/.cache/huggingface/hub \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      TRITON_CACHE_DIR="${CAP_CACHE}/triton" \
      TORCH_KERNEL_CACHE="${CAP_CACHE}/torch_kernels" \
      TORCHINDUCTOR_CACHE_DIR="${CAP_CACHE}/inductor" \
      CUDA_CACHE_PATH="${CAP_CACHE}/cuda_jit" \
      XDG_CACHE_HOME="${CAP_CACHE}/xdg" \
      OMP_NUM_THREADS=8 \
      "${PY_CAP}" scripts/nda/g2_build_captions.py \
        --images-root "${IMAGES_ROOT}" --split "${SPLIT}" --out "${CL}" \
        --model "${CAP_MODEL}" --device "${DEVICE}" --batch-size "${CAP_BATCH}" \
        --limit "${CAP_LIMIT}" 2>&1 | tee "${OUT}/logs/captions_${SPLIT}.log"
    require "${CL}"
  done
  # Quality gate. A caption file that is present but mostly failed would otherwise
  # silently become the concept-name fallback for the whole corpus and waste the run,
  # so the error rate is checked before any training starts.
  for SPLIT in train test; do
    "${PYTHON}" - "${CAPS}/captions_${SPLIT}.jsonl" "${SPLIT}" <<'PY'
import json, sys
from pathlib import Path
p, split = Path(sys.argv[1]), sys.argv[2]
n = bad = 0
for line in p.open(encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except Exception:
        continue
    n += 1
    bad += ("error" in r)
if n == 0:
    raise SystemExit(f"[FATAL] {p} has no records")
rate = bad / n
print(f"[cap] {split}: {n} records, {bad} failed ({rate:.2%})")
if rate > 0.02:
    raise SystemExit(
        f"[FATAL] {bad}/{n} ({rate:.1%}) captions failed; refusing to build targets "
        f"on a mostly-fallback corpus. Inspect the log and rerun stage 0.")
PY
  done
fi

# ===================================================================== [1] targets
if has_stage 1; then
  echo "===== [1] targets @ $(date -Iseconds) ====="
  # image-side CLIP layer features (pooled ViT-H-14) reused from the NDA-SS run
  CLIP_LAYERS="${CLIP_LAYERS:-${NB_ROOT}/outputs/nda_ss/sub-08/clip_layers}"
  require "${CLIP_LAYERS}/train/layer_08.npy"
  VAE_TR="${VAE_TR:-${NB_ROOT}/outputs/sdedit_ll_full10/shared/vae_cache/train_vae_latents_f16.npy}"
  VAE_TE="${VAE_TE:-${NB_ROOT}/outputs/sdedit_ll_full10/shared/vae_cache/test_vae_latents_f16.npy}"
  require "${VAE_TR}"; require "${VAE_TE}"
  if [[ ! -f "${TARGETS}/g2_targets_report.json" ]]; then
    "${PYTHON}" scripts/nda/g2_build_targets.py \
      --images-root "${IMAGES_ROOT}" --out "${TARGETS}" \
      --captions-dir "${CAPS}" \
      --clip-layer-dir "${CLIP_LAYERS}" --clip-layer-subdirs "train,test" \
      --vae-train "${VAE_TR}" --vae-test "${VAE_TE}" \
      --cut "${CUT}" --n-radial 8 --n-angular 8 \
      --device "${DEVICE}" 2>&1 | tee "${OUT}/logs/targets.log"
  else
    echo "[SKIP] targets"
  fi
  require "${TARGETS}/g2_targets_report.json"
fi

# ===================================================================== [2][3] training
train_one() {
  local proto="$1"; local odir="$2"
  [[ -f "${odir}/g2_report.json" ]] && { echo "[SKIP] train ${proto}"; return 0; }
  local extra=()
  if [[ "${proto}" == "intra" ]]; then
    extra=(--z-intra-dir "${NB_ROOT}/outputs/intra_hcma_s/${SID}/train")
  fi
  "${PYTHON}" scripts/nda/g2_train.py \
    --protocol "${proto}" --subject "${SUBJECT}" \
    --z-root "${Z_ROOT}" --targets-dir "${TARGETS}" \
    --ip-train-npy "${IP_TRAIN}" --ip-test-npy "${IP_TEST}" \
    --out "${odir}" --epochs "${EPOCHS}" --batch-size "${BATCH}" \
    --device "${DEVICE}" --cut "${CUT}" \
    "${extra[@]}" 2>&1 | tee "${OUT}/logs/train_${proto}.log"
}

if has_stage 2; then
  echo "===== [2] train INTRA @ $(date -Iseconds) ====="
  train_one intra "${OUT}/intra_${STAG}"
fi
if has_stage 3; then
  echo "===== [3] train INTER (LOSO) @ $(date -Iseconds) ====="
  train_one inter "${OUT}/inter_${STAG}"
fi

# ===================================================================== [4] generation
# Condition variants per protocol. Each isolates one design claim:
#   direct      : deterministic mean head + LF anchor       (the main row)
#   cfm0        : CFM sample instead of the mean head       (is flow matching worth it?)
#   direct_nc0  : mean head, anchoring OFF (cut=0)           (is the structural tower worth it?)
#   direct_np   : mean head, EMPTY prompts                  (removes the oracle-prompt shortcut)
gen_one() {
  local tag="$1"; local emb="$2"; local anchor="$3"; local cut="$4"; local prompts="$5"
  local gdir="${OUT}/generation/${tag}"
  [[ -f "${gdir}/generated/199.png" ]] && { echo "[SKIP] ${tag}"; return 0; }
  local a=()
  [[ -n "${anchor}" && "${cut}" != "0" ]] && a=(--anchor-latent-npy "${anchor}")
  "${PYTHON}" scripts/nda/generate_spectral_decode.py \
    --embed-npy "${emb}" --prompts-json "${prompts}" \
    --output-dir "${gdir}" --tag "${tag}" \
    --cut "${cut}" --gamma 1.0 --start-step 0 \
    --strength 1.0 --ip-scale "${IP_SCALE}" \
    --gen-steps "${GEN_STEPS}" --gen-guidance "${GEN_GUIDANCE}" --seed 42 \
    "${a[@]}" 2>&1 | tee "${OUT}/logs/gen_${tag}.log"
}

if has_stage 4; then
  echo "===== [4] generation @ $(date -Iseconds) ====="
  require "${HCMA_PROMPTS}"
  EMPTY_PROMPTS="${OUT}/prompts_empty.json"
  if [[ ! -f "${EMPTY_PROMPTS}" ]]; then
    NPROMPT="$("${PYTHON}" -c "import json;print(len(json.load(open('${HCMA_PROMPTS}'))))")"
    "${PYTHON}" -c "import json;json.dump(['']*${NPROMPT}, open('${EMPTY_PROMPTS}','w'))"
  fi
  for proto in intra inter; do
    D="${OUT}/${proto}_${STAG}"
    [[ -f "${D}/g2_report.json" ]] || { echo "[WARN] no training output for ${proto}, skip"; continue; }
    gen_one "g2_${proto}_direct"     "${D}/ip_direct_test.npy" "${D}/lf_latent_test.npy" "${CUT}" "${HCMA_PROMPTS}"
    gen_one "g2_${proto}_cfm0"       "${D}/ip_cfm0_test.npy"   "${D}/lf_latent_test.npy" "${CUT}" "${HCMA_PROMPTS}"
    gen_one "g2_${proto}_direct_nc0" "${D}/ip_direct_test.npy" ""                       "0"   "${HCMA_PROMPTS}"
    gen_one "g2_${proto}_direct_np"  "${D}/ip_direct_test.npy" "${D}/lf_latent_test.npy" "${CUT}" "${EMPTY_PROMPTS}"
  done
fi

# ===================================================================== [5] evaluation
if has_stage 5; then
  echo "===== [5] evaluation @ $(date -Iseconds) ====="
  # ISOLATED from outputs/standard7_protocol on purpose. That path holds the SOTA /
  # baseline REFERENCE table (61 rows plus the HCMA-10subj pooled FID 129.47), and
  # eval_standard7.py writes results.json wholesale instead of merging (line 461).
  # Pointing this stage at the shared path would have replaced the reference with
  # this job's 8 rows, silently destroying the comparison baseline. The eval job
  # already writes to its own ${OUT}/standard7_eval for the same reason; this stage
  # now does the same.
  STD7="${OUT}/standard7_sub08"
  SOTA_TABLE="${NB_ROOT}/outputs/standard7_protocol/results.json"
  MAN="${OUT}/manifest_g2.json"
  OUT_EVAL="${OUT}" MAN="${MAN}" "${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT_EVAL"])
man = {"protocol": "standard7", "rows": [], "avg_rows": []}
for proto in ("intra", "inter"):
    for v in ("direct", "cfm0", "direct_nc0", "direct_np"):
        t = f"g2_{proto}_{v}"
        d = out / "generation" / t / "generated"
        if (d / "199.png").exists():
            man["rows"].append({"tag": t, "display": t, "gen_dir": str(d)})
Path(os.environ["MAN"]).write_text(json.dumps(man, indent=2), encoding="utf-8")
print("[OK] manifest rows", len(man["rows"]))
PY
  require "${MAN}"
  # A present-but-empty manifest would otherwise let this stage "succeed" while
  # evaluating nothing -- exactly what hid the 2026-09-11 failure until the
  # dependent eval job reported 0 rows an hour later. Fail loudly instead.
  "${PYTHON}" -c "
import json, sys
n = len(json.load(open(sys.argv[1]))['rows'])
print('[OK] manifest rows', n)
sys.exit(1 if n == 0 else 0)" "${MAN}" \
    || { echo "[FATAL] no generated rows to evaluate; generation produced nothing" >&2; exit 1; }
  mkdir -p "${STD7}"
  "${PYTHON}" scripts/nda/eval_standard7.py \
    --manifest "${MAN}" --images-root "${IMAGES_ROOT}" \
    --out-dir "${STD7}" --device "${DEVICE}" --batch-size 16 \
    2>&1 | tee "${OUT}/logs/eval_standard7.log"
  cp -f "${STD7}/results.json" "${OUT}/results_g2.json"

  # pooled FID per row for direct comparison against the SOTA table
  for t in $(OUT_EVAL="${OUT}" "${PYTHON}" -c "
import json,os
m=json.load(open(os.path.join(os.environ['OUT_EVAL'],'manifest_g2.json')))
print(' '.join(r['tag'] for r in m['rows']))"); do
    "${PYTHON}" scripts/nda/eval_pooled_fid.py \
      --root "${OUT}/generation/${t}" --tag "${t}" \
      --images-root "${IMAGES_ROOT}" \
      --output-json "${OUT}/fid_${t}.json" --device "${DEVICE}" \
      2>&1 | tee -a "${OUT}/logs/eval_pooled_fid.log" || \
      echo "[WARN] pooled FID failed for ${t}"
  done

  # merge the G2 rows with the PRESERVED SOTA reference into one comparison table
  if [[ -f "${SOTA_TABLE}" ]]; then
    "${PYTHON}" scripts/nda/g2_compare_sota.py \
      --g2 "${STD7}/results.json" --sota "${SOTA_TABLE}" \
      --fid-glob "${OUT}/fid_g2_*.json" \
      --out "${OUT}/results_g2_vs_sota.json" \
      2>&1 | tee "${OUT}/logs/comparison.log" || echo "[WARN] comparison table failed"
  else
    echo "[WARN] SOTA reference table missing: ${SOTA_TABLE}"
  fi
  cp -f "${OUT}/results_g2_vs_sota.json" "${OUT}/results_merged.json" 2>/dev/null || true
fi

echo "{\"pipeline\":\"g2\",\"finished\":\"$(date -Iseconds)\"}" > "${OUT}/job_done.json"
echo "===== G2 pipeline complete @ $(date -Iseconds) ====="
