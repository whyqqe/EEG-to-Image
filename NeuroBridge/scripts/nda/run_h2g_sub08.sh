#!/usr/bin/env bash
# ============================================================================
# HCMA-2G (Granularity axis) : sub-08, intra + inter(LOSO)
#
# WHAT IS NEW HERE (and why it is not in clean_p0p1)
# --------------------------------------------------
# clean_p0p1 already solved two measurable defects, and its outputs are REUSED:
#   * protocol  : deployable prompts (free/neutral) instead of GT-concept oracle
#   * RAG       : alpha ablation; a00 (memory OFF) gives the best retrieval
#                 (top1 0.325 vs 0.225 at a50) -> memory was degrading the condition
#   * geometry  : label-free calibration, measured on the LOSO source features
#                 raw   top1 0.160  hub_skew 2.999
#                 whiten top1 0.235 hub_skew 0.630   (+7.5pp, hubness repaired)
#                 flow (CFM) top1 0.080; flow_gal 0.010 spread 0.8417  -> CFM collapses
#   It was cancelled BEFORE evaluation, so its arms have never been scored.
#
# What clean_p0p1 does NOT have is the GRANULARITY axis. Every alignment target in
# HCMA is a global pooled vector (RN50 / ViT-H / DINOv2 via timm num_classes=0 all
# return only the CLS token; the 1369 patch tokens are discarded). So the chain rule
#
#     I(e;y) = I(e;y_glob) + I(e;y_loc | y_glob)
#
# has its second term UNSUPERVISED -- not zero, just never asked for. This pipeline
# adds that axis (2x2 modality x granularity) and three consequences of it:
#
#   (1) collapse is the L1/L2 optimum, not a tuning bug:
#       argmin = E[y|e] and Var[E[y|e]] < Var[y] strictly. Shipped VAE head reaches
#       std_ratio 0.390 against an achievable ~0.514 (full) / ~0.773 (low) band.
#       -> heteroscedastic residual head (mean + logvar) so irreducible variance is
#          MODELLED instead of averaged away, and the high band ABSTAINS.
#   (2) two independent structure heads are unconstrained (12 vs 5 epochs, measured
#       cn_scale effect small) -> coarse-to-fine RESIDUAL factorisation + explicit
#       consistency term.
#   (3) the alignment objective is blind to hubness: cosine/InfoNCE is invariant to
#       orthogonal transforms, and hubness lives in the covariance geometry, which is
#       exactly the axis on which the model fails (raw margin 0.1329 > whiten margin
#       0.0970 yet raw top1 is 7.5pp WORSE) -> VICReg variance+covariance terms,
#       applied per granularity.
#
# Band allocation is MEASURED, not tuned: each band's weight is its Fourier-domain
# R^2 from the fit split, so a band the signal cannot carry gets ~zero weight.
#
# Deployment design: the semantic side stays FROZEN (shipped/calibrated IP embeds),
# protecting the already-saturated high-level metrics; only the structural condition
# is replaced by the band-anchored 2x2 decoder. No depth ControlNet and no img2img
# init are used on the anchor arms (strength=1.0), so the anchor's effect is isolated.
# ============================================================================

set -uo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
SUB="${SUB:-sub-08}"
IMG_ROOT="${IMG_ROOT:-/project/peilab/why/data/images_set}"
INTRA="${INTRA:-${NB_ROOT}/outputs/intra_hcma_s/${SUB}}"
C0="${C0:-${NB_ROOT}/outputs/clean_p0p1/${SUB}}"
OUT="${OUT:-${NB_ROOT}/outputs/h2g/${SUB}}"
STD7="${STD7:-${OUT}/std7}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON="${PYTHON:-python}"

mkdir -p "${OUT}"/{logs,anchors,gen,std7,heads}
echo "{\"pipeline\":\"h2g_sub08\",\"started\":\"$(date -Iseconds)\",\"innovation\":\"granularity axis (2x2) + heteroscedastic abstention + VICReg geometry\",\"reuses\":\"clean_p0p1 conditions (protocol + calibration), shipped frozen semantics\"}" > "${OUT}/job_running.json"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1"; exit 1; }; }

# ---------------------------------------------------------------- inputs
VAE_TR="${INTRA}/vae_cache/train_vae_latents_f16.npy"
VAE_TE="${INTRA}/vae_cache/test_vae_latents_f16.npy"
Z_TR="${INTRA}/train/z_decode_vith_train.npy"
Z_TE="${INTRA}/train/z_decode_vith_test.npy"
require "${VAE_TR}"; require "${VAE_TE}"; require "${Z_TR}"; require "${Z_TE}"
require "${C0}/conditions/embeds/a00.npy"
require "${C0}/conditions/prompts/free.json"

# frozen semantic conditions (calibrated / protocol-fixed)
EMB_A00="${C0}/conditions/embeds/a00.npy"
EMB_WHITEN="${C0}/conditions/embeds/al_whiten.npy"
EMB_RAW="${C0}/conditions/embeds/al_raw.npy"
P_FREE="${C0}/conditions/prompts/free.json"
P_NEUTRAL="${C0}/conditions/prompts/neutral.json"
P_ORACLE="${C0}/conditions/prompts/oracle.json"

# ---------------------------------------------------------------- 0. split
SPLIT="${OUT}/leakfree_split.json"
[[ -f "${SPLIT}" ]] || "${PYTHON}" "${NB_ROOT}/scripts/nda/leakfree.py" --out "${SPLIT}"

# ------------------------------------------------- 1. local semantic targets
echo "===== [1] DINOv2 patch-grid local targets @ $(date -Iseconds) ====="
LDIR="${OUT}/local_targets"
if [[ ! -f "${LDIR}/dinov2_local_train.npy" ]]; then
  "${PYTHON}" "${NB_ROOT}/scripts/nda/h2g_build_local_targets.py" \
    --images-root "${IMG_ROOT}" --output-dir "${LDIR}" \
    --splits train test --grid 6 --input-size 224 \
    --batch-size 16 --device "${DEVICE}" 2>&1 | tail -20
else
  echo "[SKIP] local targets cached"
fi
require "${LDIR}/dinov2_local_train.npy"; require "${LDIR}/dinov2_local_test.npy"

# ------------------------------------------------------- 2. H2G 2x2 heads
echo "===== [2] H2G granularity heads (4 variants) @ $(date -Iseconds) ====="
for v in full nogeom noloc noresid; do
  D="${OUT}/heads/${v}"
  if [[ -f "${D}/checkpoint_h2g_best.pth" ]]; then echo "[SKIP] ${v}"; continue; fi
  echo "[TRAIN] variant=${v}"
  "${PYTHON}" "${NB_ROOT}/scripts/nda/h2g_train_heads.py" \
    --sem-train-npy "${Z_TR}" --sem-test-npy "${Z_TE}" \
    --vae-train-npy "${VAE_TR}" --vae-test-npy "${VAE_TE}" \
    --local-train-npy "${LDIR}/dinov2_local_train.npy" \
    --local-test-npy  "${LDIR}/dinov2_local_test.npy" \
    --global-train-npy "${LDIR}/dinov2_global_train.npy" \
    --global-test-npy  "${LDIR}/dinov2_global_test.npy" \
    --val-split-json "${SPLIT}" --output-dir "${D}" --variant "${v}" \
    --cut 0.125 --code 768 --num-epochs 100 --batch-size 128 \
    --lambda-loc 1.0 --lambda-glob-sem 0.3 --lambda-struct 1.0 \
    --lambda-cons 0.5 --lambda-geo 0.1 --patience 20 \
    --device "${DEVICE}" 2>&1 | tail -25 || echo "[WARN] variant ${v} failed"
done
require "${OUT}/heads/full/pred_anchor_lowband_test.npy"

# ------------------------------------------------------- 3. generation grid
GEN="${OUT}/gen"
echo "===== [3] generation (intra + inter) @ $(date -Iseconds) ====="

run_gen() {  # tag embed prompts anchor cut gamma strength cn
  local tag="$1" emb="$2" pr="$3" anc="$4" cut="$5" gamma="$6" s="$7" cn="$8"
  local gdir="${GEN}/${tag}"
  [[ -f "${gdir}/generated/199.png" ]] && { echo "[SKIP] ${tag}"; return 0; }
  echo "[GEN ] ${tag} cut=${cut} gamma=${gamma} s=${s} cn=${cn} prompts=$(basename "${pr}")"
  local a=()
  [[ -n "${anc}" ]] && a=(--anchor-latent-npy "${anc}")
  "${PYTHON}" "${NB_ROOT}/scripts/nda/generate_spectral_decode.py" \
    --embed-npy "${emb}" --prompts-json "${pr}" "${a[@]}" \
    --output-dir "${gdir}" --tag "${tag}" \
    --cut "${cut}" --gamma "${gamma}" --strength "${s}" --cn-scale "${cn}" \
    --ip-scale 1.0 --gen-steps 28 --gen-guidance 5.0 --gen-size 512 \
    --seed 42 --max-images 200 2>&1 | tail -6 || echo "[WARN] ${tag} failed"
}

A_FULL="${OUT}/heads/full/pred_anchor_lowband_test.npy"
A_RES="${OUT}/heads/full/pred_anchor_low_plus_res_test.npy"
A_NOGEOM="${OUT}/heads/nogeom/pred_anchor_lowband_test.npy"
A_NOLOC="${OUT}/heads/noloc/pred_anchor_lowband_test.npy"
A_NORES="${OUT}/heads/noresid/pred_anchor_lowband_test.npy"

# --- pure-prior controls (no structural condition at all)
run_gen "h2g_prior_a00_free"    "${EMB_A00}"    "${P_FREE}" "" 0 0 1.00 0.00
run_gen "h2g_prior_whiten_free" "${EMB_WHITEN}" "${P_FREE}" "" 0 0 1.00 0.00

# --- main arms: 2x2 low-band anchor, deployable prompts
run_gen "h2g_low_a00_free"       "${EMB_A00}"    "${P_FREE}"    "${A_FULL}" 0.0625 1.0 1.00 0.00
run_gen "h2g_low_a00_neutral"    "${EMB_A00}"    "${P_NEUTRAL}" "${A_FULL}" 0.0625 1.0 1.00 0.00
run_gen "h2g_low_whiten_free"    "${EMB_WHITEN}" "${P_FREE}"    "${A_FULL}" 0.0625 1.0 1.00 0.00
run_gen "h2g_low_whiten_neutral" "${EMB_WHITEN}" "${P_NEUTRAL}" "${A_FULL}" 0.0625 1.0 1.00 0.00
run_gen "h2g_low_raw_free"       "${EMB_RAW}"    "${P_FREE}"    "${A_FULL}" 0.0625 1.0 1.00 0.00

# --- band-width sweep (does a wider MI-positive band help?)
run_gen "h2g_low_c125_whiten_free" "${EMB_WHITEN}" "${P_FREE}" "${A_FULL}" 0.1250 1.0 1.00 0.00
run_gen "h2g_low_c25_whiten_free"  "${EMB_WHITEN}" "${P_FREE}" "${A_FULL}" 0.2500 1.0 1.00 0.00

# --- ABLATION: heteroscedastic abstention (adding the high-band mean back)
run_gen "h2g_cf_whiten_free" "${EMB_WHITEN}" "${P_FREE}" "${A_RES}" 0.1250 1.0 1.00 0.00

# --- ABLATION: geometry term (VICReg)
run_gen "h2g_nogeom_whiten_free" "${EMB_WHITEN}" "${P_FREE}" "${A_NOGEOM}" 0.0625 1.0 1.00 0.00
# --- ABLATION: local-semantic supervision (the chain-rule term)
run_gen "h2g_noloc_whiten_free"  "${EMB_WHITEN}" "${P_FREE}" "${A_NOLOC}"  0.0625 1.0 1.00 0.00
# --- ABLATION: residual head entirely absent
run_gen "h2g_noresid_whiten_free" "${EMB_WHITEN}" "${P_FREE}" "${A_NORES}" 0.0625 1.0 1.00 0.00

# --- leak quantification only (GT concept prompt; NOT deployable)
run_gen "h2g_low_whiten_oracle" "${EMB_WHITEN}" "${P_ORACLE}" "${A_FULL}" 0.0625 1.0 1.00 0.00

# ------------------------------------------------------------- 4. manifest
echo "===== [4] manifest (new arms + clean_p0p1 arms) @ $(date -Iseconds) ====="
MAN="${OUT}/manifest_h2g.json"
OUT_EVAL="${OUT}" C0_E="${C0}" MAN="${MAN}" "${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out, c0 = Path(os.environ["OUT_EVAL"]), Path(os.environ["C0_E"])
rows = []
# new H2G arms
for d in sorted((out / "gen").iterdir()):
    g = d / "generated"
    if (g / "199.png").is_file():
        rows.append({"tag": d.name, "display": d.name, "gen_dir": str(g)})
# clean_p0p1 arms (protocol + calibration) -- never scored before
cgen = c0 / "conditions" / "generation"
if cgen.is_dir():
    for d in sorted(cgen.iterdir()):
        g = d / "generated"
        if (g / "199.png").is_file():
            rows.append({"tag": d.name, "display": d.name, "gen_dir": str(g)})
Path(os.environ["MAN"]).write_text(json.dumps({"protocol": "standard7", "rows": rows}, indent=2))
print(f"[OK] manifest rows={len(rows)}")
for r in rows: print("  ", r["tag"])
PY

# ------------------------------------------------------------- 5. evaluate
echo "===== [5] standard-7 + FID @ $(date -Iseconds) ====="
"${PYTHON}" "${NB_ROOT}/scripts/nda/eval_standard7.py" \
  --manifest "${MAN}" --images-root "${IMG_ROOT}" --out-dir "${STD7}" \
  --device "${DEVICE}" --batch-size 16 2>&1 | tail -20
cp -f "${STD7}/results.json" "${OUT}/results_h2g.json"

# ------------------------------------------------------- 6. table + gates
echo "===== [6] table + gate @ $(date -Iseconds) ====="
OUT_EVAL="${OUT}" "${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT_EVAL"])
rep = json.loads((out / "results_h2g.json").read_text(encoding="utf-8"))
rows = rep["rows"]
keys = ["pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "swav", "fid"]
hdr = f"{'tag':<34}" + "".join(f"{k:>10}" for k in keys)
print("\n" + "=" * len(hdr)); print(hdr); print("-" * len(hdr))
for r in sorted(rows, key=lambda x: -(x.get("ssim") or 0)):
    line = f"{r['tag']:<34}"
    for k in keys:
        v = r.get(k)
        line += f"{v:>10.4f}" if (v is not None and k != "fid") else \
                (f"{v:>10.1f}" if v is not None else f"{'--':>10}")
    print(line)
print("=" * len(hdr))

# gate relative to the shipped reference where present
by = {r["tag"]: r for r in rows}
ref = by.get("cl_ref_oracle_a00") or by.get("h2g_prior_a00_free")
if ref:
    print(f"\n[GATE] reference = {ref['tag']}")
    ranked = []
    for r in rows:
        if r["tag"] == ref["tag"]:
            continue
        try:
            ok = (float(r["clip"]) >= float(ref["clip"]) - 0.010
                  and float(r["alex5"]) >= float(ref["alex5"]) - 0.010
                  and float(r["fid"]) <= float(ref["fid"]) + 15.0)
        except (TypeError, KeyError):
            continue
        ranked.append((bool(ok), float(r.get("ssim") or 0), r["tag"],
                       float(r["ssim"]) - float(ref["ssim"]),
                       float(r["clip"]) - float(ref["clip"]),
                       float(r["fid"]) - float(ref["fid"])))
    ranked.sort(key=lambda x: (not x[0], -x[1]))
    for ok, _s, tag, dssim, dclip, dfid in ranked[:12]:
        print(f"  {'PASS' if ok else 'fail'} {tag:<34} dSSIM={dssim:+.4f} dCLIP={dclip:+.4f} dFID={dfid:+.1f}")
(out / "h2g_table.txt").write_text(hdr + "\n", encoding="utf-8")
(out / "job_done.json").write_text(json.dumps(
    {"pipeline": "h2g_sub08", "finished": __import__("datetime").datetime.now().isoformat(),
     "n_arms": len(rows)}, indent=2), encoding="utf-8")
PY

echo "===== DONE @ $(date -Iseconds) ====="
