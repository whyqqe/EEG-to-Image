#!/usr/bin/env bash
# ============================================================================
# BRDT (Band-Routed Dual Tower) : sub-08, gate -> encode -> train -> generate -> eval
#
# HCMA's main architecture is PRESERVED: cross-subject backbone, EEG encoder slot,
# structure/semantic dual tower, multi-condition injection decoder. Two things change,
# and both are decided by measurement rather than by design preference:
#
#   (1) EEG encoder -> band x window tangent-space features, ROUTED to the towers.
#       gamma -> structure tower, alpha/theta -> semantic tower, beta split,
#       early window -> structure, late -> semantic.
#       Basis: gamma is retinotopically tuned (carries WHERE), alpha is not (carries
#       global state). Gate 1 TESTS this on THINGS-EEG2 instead of assuming it.
#
#   (2) A rank-r bottleneck between the towers, where r = the MEASURED number of
#       gamma<->alpha/theta canonical components above threshold. Initialised as an
#       exact no-op, so any benefit has to be earned. The `overexchange` arm tests the
#       falsifiable prediction that r above the measured value hurts BOTH towers.
#
# Each tower carries two granularities and two modalities:
#   SEM    global: CLIP-text + ViT-H image      local: DINOv2 patch-cell grid
#   STRUCT global: SDXL-VAE low band + depth    local: high band, ABSTAINS (models Sigma)
#
# The semantic side of the GENERATION remains HCMA's own frozen condition in Arm A, so
# the structural effect is isolated; Arm B uses the new encoder's own image-space
# prediction end to end.
# ============================================================================

set -uo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
SUB="${SUB:-sub-08}"
IMG_ROOT="${IMG_ROOT:-/project/peilab/why/data/images_set}"
EEG_ROOT="${EEG_ROOT:-${NB_ROOT}/data/things_eeg/preprocessed_eeg}"
INTRA="${INTRA:-${NB_ROOT}/outputs/intra_hcma_s/${SUB}}"
C0="${C0:-${NB_ROOT}/outputs/clean_p0p1/${SUB}}"
H2G="${H2G:-${NB_ROOT}/outputs/h2g/${SUB}}"          # reused DINOv2 local targets
OUT="${OUT:-${NB_ROOT}/outputs/brdt/${SUB}}"
STD7="${STD7:-${OUT}/std7}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON="${PYTHON:-python}"

mkdir -p "${OUT}"/{logs,bandfeat,heads,gen,std7,gates,gt_depth}
echo "{\"pipeline\":\"brdt_sub08\",\"started\":\"$(date -Iseconds)\",\"innovation\":\"band-routed evidence + measured-rank inter-tower bottleneck\",\"preserves\":\"HCMA cross-subject backbone, dual tower, injection decoder\"}" > "${OUT}/job_running.json"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1"; return 1; }; }

EEG_TR="${EEG_ROOT}/${SUB}/train.npy"
EEG_TE="${EEG_ROOT}/${SUB}/test.npy"
VAE_TR="${INTRA}/vae_cache/train_vae_latents_f16.npy"
VAE_TE="${INTRA}/vae_cache/test_vae_latents_f16.npy"
CT_TR="${NB_ROOT}/outputs/hcma_loso_fid129/shared_targets/t_coarse_train.npy"
CT_TE="${NB_ROOT}/outputs/hcma_loso_fid129/shared_targets/t_coarse_test.npy"
VITH_TR="${INTRA}/train/decode_vith1024_train_clip_1024.npy"
VITH_TE="${INTRA}/train/decode_vith1024_test_clip_1024.npy"
DLOC_TR="${H2G}/local_targets/dinov2_local_train.npy"
DLOC_TE="${H2G}/local_targets/dinov2_local_test.npy"
DEPTH_TE="${INTRA}/gt_depth/test_depth_64.npy"
for f in "${EEG_TR}" "${EEG_TE}" "${VAE_TR}" "${VAE_TE}" "${CT_TR}" "${CT_TE}" \
         "${VITH_TR}" "${VITH_TE}" "${DLOC_TR}" "${DLOC_TE}" "${DEPTH_TE}"; do
  require "${f}" || exit 1
done

# ---------------------------------------------------------------- 0. split
SPLIT="${OUT}/leakfree_split.json"
[[ -f "${SPLIT}" ]] || "${PYTHON}" "${NB_ROOT}/scripts/nda/leakfree.py" --out "${SPLIT}"

# --------------------------------------------- 1. Gate 0-2 feasibility probes
echo "===== [1] routing/bottleneck gates @ $(date -Iseconds) ====="
GATES="${OUT}/gates/gates.json"
if [[ ! -f "${GATES}" ]]; then
  "${PYTHON}" "${NB_ROOT}/scripts/nda/brdt_probe.py" \
    --eeg-train-npy "${EEG_TR}" --vae-train-npy "${VAE_TR}" \
    --out-json "${GATES}" --cut 0.125 --cca-thresh 0.50 \
    --device "${DEVICE}" 2>&1 | tail -60 || echo "[WARN] probe failed"
fi
RANK_R=4
ROUTE_FLAG=""
if [[ -f "${GATES}" ]]; then
  RANK_R=$("${PYTHON}" -c "import json;print(json.load(open('${GATES}'))['gates'].get('gate2_rank_r',4))")
  G1=$("${PYTHON}" -c "import json;print(int(json.load(open('${GATES}'))['gates'].get('gate1_pass',False)))")
  echo "[GATE] rank_r=${RANK_R}  gate1_pass=${G1}"
  if [[ "${G1}" != "1" ]]; then
    echo "[GATE] routing claim NOT supported on this subject -> merged-band fallback"
    ROUTE_FLAG="merged"
  fi
fi

# --------------------------------------- 2. optional depth cache (structure modality)
echo "===== [2] GT depth cache (structure tower modality 2) @ $(date -Iseconds) ====="
DEPTH_TR="${OUT}/gt_depth/train_depth_64.npy"
if [[ ! -f "${DEPTH_TR}" ]]; then
  timeout 3600 "${PYTHON}" "${NB_ROOT}/scripts/nda/build_gt_depth_cache.py" \
    --images-root "${IMG_ROOT}" --output-dir "${OUT}/gt_depth" \
    --splits train --low-res 64 --rgb-size 512 \
    --device "${DEVICE}" --batch-size 8 2>&1 | tail -8 \
    || { echo "[WARN] depth cache failed -> structure tower runs VAE-only"; DEPTH_TR=""; }
fi
[[ -n "${DEPTH_TR}" && ! -f "${DEPTH_TR}" ]] && DEPTH_TR=""

# --------------------------------------- 3. band x window tangent-space features
echo "===== [3] band features (new encoder front end) @ $(date -Iseconds) ====="
FEAT="${OUT}/bandfeat"
if [[ ! -f "${FEAT}/bandfeat_test.npz" ]]; then
  "${PYTHON}" "${NB_ROOT}/scripts/nda/brdt_build_bandfeat.py" \
    --eeg-train-npy "${EEG_TR}" --eeg-test-npy "${EEG_TE}" \
    --out-dir "${FEAT}" --d-pca 256 --device "${DEVICE}" 2>&1 | tail -25 \
    || echo "[FATAL] band features failed"
fi
require "${FEAT}/bandfeat_test.npz" || exit 1

# ------------------------------------------------------- 4. train BRDT variants
echo "===== [4] BRDT training @ $(date -Iseconds) ====="
DP_ARGS=()
[[ -n "${DEPTH_TR}" ]] && DP_ARGS=(--depth-train-npy "${DEPTH_TR}" --depth-test-npy "${DEPTH_TE}")

run_train() {  # variant
  local v="$1"
  # If Gate 1 did not support the routing claim we run the `full` arm as `merged`
  # (bands pooled). The substitution is recorded so a negative gate cannot be
  # laundered into a positive routing result.
  local real="${v}"
  [[ -n "${ROUTE_FLAG}" && "${v}" == "full" ]] && real="merged"
  local D="${OUT}/heads/${v}"
  [[ -f "${D}/checkpoint_brdt_best.pth" ]] && { echo "[SKIP] ${v}"; return 0; }
  echo "[TRAIN] variant=${v} (effective=${real})"
  "${PYTHON}" "${NB_ROOT}/scripts/nda/brdt_train.py" \
    --feat-dir "${FEAT}" \
    --vae-train-npy "${VAE_TR}" --vae-test-npy "${VAE_TE}" \
    --clip-text-train-npy "${CT_TR}" --clip-text-test-npy "${CT_TE}" \
    --vith-train-npy "${VITH_TR}" --vith-test-npy "${VITH_TE}" \
    --dino-local-train-npy "${DLOC_TR}" --dino-local-test-npy "${DLOC_TE}" \
    "${DP_ARGS[@]}" \
    --val-split-json "${SPLIT}" --gate-json "${GATES}" \
    --output-dir "${D}" --variant "${real}" \
    --cut 0.125 --code 768 --num-epochs 100 --batch-size 128 \
    --patience 20 --device "${DEVICE}" 2>&1 | tail -30 \
    || echo "[WARN] variant ${v} failed"
}

for v in full noexchange overexchange wrongroute nolocal noabstain; do
  run_train "${v}"
done
require "${OUT}/heads/full/pred_anchor_lowband_test.npy" || exit 1

# ------------------------------------------------------------- 5. generation
echo "===== [5] generation (intra + inter) @ $(date -Iseconds) ====="
GEN="${OUT}/gen"

run_gen() {  # tag embed prompts anchor cut gamma strength cn
  local tag="$1" emb="$2" pr="$3" anc="$4" cut="$5" gamma="$6" s="$7" cn="$8"
  local gdir="${GEN}/${tag}"
  [[ -f "${gdir}/generated/199.png" ]] && { echo "[SKIP] ${tag}"; return 0; }
  echo "[GEN ] ${tag} cut=${cut} gamma=${gamma} s=${s} cn=${cn} emb=$(basename "${emb}")"
  local a=()
  [[ -n "${anc}" ]] && a=(--anchor-latent-npy "${anc}")
  "${PYTHON}" "${NB_ROOT}/scripts/nda/generate_spectral_decode.py" \
    --embed-npy "${emb}" --prompts-json "${pr}" "${a[@]}" \
    --output-dir "${gdir}" --tag "${tag}" \
    --cut "${cut}" --gamma "${gamma}" --strength "${s}" --cn-scale "${cn}" \
    --ip-scale 1.0 --gen-steps 28 --gen-guidance 5.0 --gen-size 512 \
    --seed 42 --max-images 200 2>&1 | tail -6 || echo "[WARN] ${tag} failed"
}

EMB_A00="${C0}/conditions/embeds/a00.npy"
EMB_WHITEN="${C0}/conditions/embeds/al_whiten.npy"
EMB_BRDT="${OUT}/heads/full/sem_img_test.npy"
P_FREE="${C0}/conditions/prompts/free.json"
P_NEUTRAL="${C0}/conditions/prompts/neutral.json"
P_ORACLE="${C0}/conditions/prompts/oracle.json"
for f in "${EMB_A00}" "${EMB_WHITEN}" "${P_FREE}" "${P_NEUTRAL}" "${P_ORACLE}"; do
  require "${f}" || exit 1
done

A_FULL="${OUT}/heads/full/pred_anchor_lowband_test.npy"
A_RES="${OUT}/heads/full/pred_anchor_low_plus_res_test.npy"

run_gen "brdt_prior_a00"        "${EMB_A00}"    "${P_FREE}"    ""        0 0 1.00 0.00
run_gen "brdt_anc_a00"          "${EMB_A00}"    "${P_FREE}"    "${A_FULL}" 0.125 1.0 1.00 0.00
run_gen "brdt_anc_whiten"       "${EMB_WHITEN}" "${P_FREE}"    "${A_FULL}" 0.125 1.0 1.00 0.00
run_gen "brdt_anc_c0625_whiten" "${EMB_WHITEN}" "${P_FREE}"    "${A_FULL}" 0.0625 1.0 1.00 0.00
run_gen "brdt_anc_c25_whiten"   "${EMB_WHITEN}" "${P_FREE}"    "${A_FULL}" 0.250 1.0 1.00 0.00
run_gen "brdt_anc_res_whiten"   "${EMB_WHITEN}" "${P_FREE}"    "${A_RES}"  0.125 1.0 1.00 0.00

# Arm B: the new encoder supplies the image-space condition end to end
run_gen "brdt_b_brdtemb_free"    "${EMB_BRDT}"  "${P_FREE}"    "${A_FULL}" 0.125 1.0 1.00 0.00
run_gen "brdt_b_brdtemb_neutral" "${EMB_BRDT}"  "${P_NEUTRAL}" "${A_FULL}" 0.125 1.0 1.00 0.00

# ablations
for v in noexchange overexchange wrongroute nolocal noabstain; do
  A="${OUT}/heads/${v}/pred_anchor_lowband_test.npy"
  [[ -f "${A}" ]] && run_gen "brdt_abl_${v}_whiten" "${EMB_WHITEN}" "${P_FREE}" "${A}" 0.125 1.0 1.00 0.00
done
# leak quantification only (NOT deployable)
run_gen "brdt_anc_oracle" "${EMB_WHITEN}" "${P_ORACLE}" "${A_FULL}" 0.125 1.0 1.00 0.00

# ------------------------------------------------------------- 6. manifest
echo "===== [6] manifest @ $(date -Iseconds) ====="
MAN="${OUT}/manifest_brdt.json"
OUT_EVAL="${OUT}" C0_E="${C0}" MAN="${MAN}" "${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out, c0 = Path(os.environ["OUT_EVAL"]), Path(os.environ["C0_E"])
rows = []
for d in sorted((out / "gen").iterdir()):
    g = d / "generated"
    if (g / "199.png").is_file():
        rows.append({"tag": d.name, "display": d.name, "gen_dir": str(g)})
cgen = c0 / "conditions" / "generation"
if cgen.is_dir():
    for d in sorted(cgen.iterdir()):
        g = d / "generated"
        if (g / "199.png").is_file():
            rows.append({"tag": d.name, "display": d.name, "gen_dir": str(g)})
Path(os.environ["MAN"]).write_text(json.dumps({"protocol": "standard7", "rows": rows}, indent=2))
print(f"[OK] manifest rows={len(rows)}")
PY

# ------------------------------------------------------------- 7. evaluate
echo "===== [7] standard-7 + FID @ $(date -Iseconds) ====="
"${PYTHON}" "${NB_ROOT}/scripts/nda/eval_standard7.py" \
  --manifest "${MAN}" --images-root "${IMG_ROOT}" --out-dir "${STD7}" \
  --device "${DEVICE}" --batch-size 16 2>&1 | tail -20
cp -f "${STD7}/results.json" "${OUT}/results_brdt.json"

# ------------------------------------------------------- 8. table + gates
echo "===== [8] table @ $(date -Iseconds) ====="
OUT_EVAL="${OUT}" GATES_E="${GATES}" "${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT_EVAL"])
rep = json.loads((out / "results_brdt.json").read_text(encoding="utf-8"))
rows = rep["rows"]
keys = ["pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "swav", "fid"]
hdr = f"{'tag':<32}" + "".join(f"{k:>10}" for k in keys)
print("\n" + "=" * len(hdr)); print(hdr); print("-" * len(hdr))
for r in sorted(rows, key=lambda x: -(x.get("ssim") or 0)):
    line = f"{r['tag']:<32}"
    for k in keys:
        v = r.get(k)
        line += f"{v:>10.4f}" if (v is not None and k != "fid") else (f"{v:>10.1f}" if v is not None else f"{'--':>10}")
    print(line)
print("=" * len(hdr))
by = {r["tag"]: r for r in rows}
ref = by.get("brdt_prior_a00")
if ref:
    print(f"\n[GATE vs {ref['tag']}]")
    for t in sorted(by):
        if t == ref["tag"] or any(by[t].get(k) is None for k in ("clip", "alex5", "fid", "ssim")):
            continue
        d_s = float(by[t]["ssim"]) - float(ref["ssim"])
        d_a5 = float(by[t]["alex5"]) - float(ref["alex5"])
        d_f = float(by[t]["fid"]) - float(ref["fid"])
        ok = d_a5 >= -0.010 and d_f <= 15.0
        print(f"  {'PASS' if ok else 'fail'} {t:<32} dSSIM={d_s:+.4f} dA5={d_a5:+.4f} dFID={d_f:+.1f}")
(out / "brdt_table.txt").write_text(hdr + "\n", encoding="utf-8")
import datetime
(out / "job_done.json").write_text(json.dumps(
    {"pipeline": "brdt_sub08", "finished": datetime.datetime.now().isoformat(),
     "n_arms": len(rows), "gates": os.environ.get("GATES_E", "")}, indent=2), encoding="utf-8")
PY

echo "===== DONE @ $(date -Iseconds) ====="
