#!/usr/bin/env bash
# =============================================================================
# NW5 sub-08: test the OPTIMIZED architecture, i.e. the mechanism that CogCapPro
# actually uses, against the pixel-constrained chain we have been shipping.
#
# WHAT CHANGED AND WHY IT IS THE WHOLE BALLGAME
# ---------------------------------------------
# Reading CogCapPro's public code (XiaoZhangYES/CognitionCapturerPro) settles the
# question that the local oracle experiment had only bounded from the inside:
#
#   src/cogcappro/generate_image/generator.py
#       _MODALITY_SCALES = {
#         "image": {"down": {"block_2": [1.0, 1.0]}, "up": {"block_0": [1.0, 1.0, 1.0]}},
#         "depth": {"down": {"block_2": [0.0, 0.5]}, "up": {"block_0": [0.0, 0.0, 0.0]}},
#         "edge":  {"down": {"block_2": [0.0, 0.5]}, "up": {"block_0": [0.0, 0.0, 0.0]}}}
#   src/cogcappro/generate_image/batch_generate.py
#       IPAdapterGenerator(..., num_inference_steps=15, guidance_scale=0.0)
#       prompt=""                       # no text at all
#       modalities = ["image", "depth", "edge"]
#
# Three facts fall out, and each one invalidates an assumption our chain was built on:
#
#  (1) THERE IS NO PIXEL CONSTRAINT.  No ControlNet, no img2img init, nothing that
#      says "put this pixel here".  Structure is expressed as a CLIP embedding injected
#      through IP-Adapter, i.e. in FEATURE space.  That is how it can hold SSIM 0.398
#      and Inception 0.779 at the same time: a feature-space structural prior asks for
#      the right layout without spending the generator's freedom on per-pixel
#      obedience.  Our chain spends that freedom twice (depth ControlNet + a 0.82-0.92
#      img2img init) and then wonders why semantics cannot get through.
#
#  (2) THE INJECTION LAYOUT IS SPECIFIC AND ASYMMETRIC.  Semantic travels on
#      down_blocks.2 + up_blocks.0 at full strength; structure is admitted on exactly
#      ONE level -- down_blocks.2.attentions.1 -- at 0.5, and is off everywhere else.
#      Our own `layered` arm put semantics on `late`, which includes all three
#      up_blocks.1 levels (the highest-resolution, most spatial levels), and structure
#      on all four `down` levels.  Those are different mechanisms, not a tuning gap.
#
#  (3) CFG IS OFF.  guidance_scale=0.0.  Our chain runs 28 steps at guidance 5.0.
#
# `IPAdapterGenerator._MODALITY_SCALES` is verbatim the example in diffusers'
# `set_ip_adapter_scale` docstring, so the layout is reproducible exactly -- which is
# what `--ip-scale-json` in generate_layered_decode.py was added for.  Our coarser
# {down,mid,up}x scalar form cannot express "one level, 0.5" or "no up blocks".
#
# THE ARMS -- each answers one question, and the cheap decisive ones come first
# ---------------------------------------------------------------------------
#  A1  pure IP + our semantic condition, CogCapPro layout  -> can out conditional
#      travel on their mechanism at all?  (the unlock test)
#  A2  A1 + GT depth + GT edge, CogCapPro layout           -> does structure in
#      feature space add anything, and does it hold SSIM?  (mechanism ceiling)
#  A3  A2 but uniform across all 11 levels                 -> is their ASYMMETRIC
#      layout load-bearing, or is multi-modal enough?
#  A4  same condition as A1 on SDXL-base 28 steps CFG 5    -> separates the OPERATOR
#      (turbo/CFG0 vs base/CFG5) from the PIXEL MACHINE
#  A5  A2 + init-only anchor, init low-passed sigma 3      -> our own L3 contribution:
#      keep only the low band of the init (PixCorr/SSIM are dominated by it) and hand
#      the high band back to the generator
#  A6  A2 layout but inside our shipped CN+init chain       -> does the layout help
#      even when the pixel machine is present?  (isolates layout from pixel path)
#  A7  the a_hi recipe exactly (branch-spec all:0.9)        -> HARNESS CONTROL: must
#      reproduce the recorded a_hi sub-08 numbers, otherwise nothing else is readable
#  A8  A2 with EEG-derived depth instead of GT depth         -> the deployable version:
#      our UCK spatial path predicts depth, we CLIP-encode it and calibrate to the depth
#      train bank.  Edge is NOT included: Canny over our predicted low-level image gives
#      near-identical maps across trials (raw rowcos 0.9375 vs the bank's 0.5295, and
#      quantile matching cannot close that), so its CLIP rows carry almost no trial
#      information (cos to the true edge row 0.127).  A8b keeps it as evidence of that.
#  A9  GT image + GT depth + GT edge, CogCapPro layout      -> the honest CEILING of
#      this mechanism on this data; no EEG anywhere
#
# Read A1/A2/A9 against A7.  If A9 does not clear the bars, the ceiling is the
# mechanism and no amount of decoder work helps; if A9 clears them but A2 does not, the
# gap is our condition quality; if A2 clears them and A7 does not, the pixel machine was
# the suppressant all along and the architecture should drop it outright.
# =============================================================================

set -uo pipefail

NB_ROOT="/project/peilab/why/NeuroBridge"
cd "${NB_ROOT}"

PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
STAG="${STAG:-sub-08}"
SEED="${SEED:-42}"
OUT_ROOT="${OUT_ROOT:-${NB_ROOT}/outputs/nw5_s08}"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
CC="${NB_ROOT}/outputs/gem/cond_cache"

# these four keep transformers/open_clip/torch off $HOME, which is not writable here
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
export HOME="${XDG_CACHE_HOME}"

LOG="${OUT_ROOT}/logs"
SPECS="${OUT_ROOT}/specs"
CONDS="${OUT_ROOT}/conds"
mkdir -p "${LOG}" "${SPECS}" "${CONDS}" "${OUT_ROOT}/arms"

log() { echo "[$(date +%H:%M:%S)] $*"; }
hr()  { echo "------------------------------------------------------------------------"; }

# our best semantic condition for this subject, carried over from the 10-subject run
SEM_COND="${NB_ROOT}/outputs/nw4_10s/arms/a_hi/conds/${STAG}/cal_test.npy"
DEPTH_RGB="${NB_ROOT}/outputs/uck/${STAG}/full/spatial/pred_depth_rgb_512"
INIT_RGB="${NB_ROOT}/outputs/sdedit_ll_full10/${STAG}/vae_head/pred_lowlevel_rgb_512"
DEPLOY_PROMPTS="${NB_ROOT}/outputs/g2f/prompts/prompts_deploy.json"

for f in "${SEM_COND}" "${DEPLOY_PROMPTS}"; do
  [[ -f "${f}" ]] || { echo "[FATAL] missing ${f}" >&2; exit 1; }
done

# ---------------------------------------------------------------------------
log "===== [0] fail fast on CUDA ====="
"${PYTHON}" - <<'PY' || exit 1
import torch
if not torch.cuda.is_available():
    raise SystemExit("[FATAL] CUDA unavailable - refusing to silently fall back to CPU")
print(f"[gpu] {torch.cuda.get_device_name(0)}  "
      f"cuda={torch.version.cuda}  torch={torch.__version__}")
PY

# ---------------------------------------------------------------------------
log "===== [1] scale specs + prompt files ====="
"${PYTHON}" - "${SPECS}" "${CONDS}" <<'PY'
import json, sys
from pathlib import Path
specs, conds = Path(sys.argv[1]), Path(sys.argv[2])

# verbatim from CogCapPro src/cogcappro/generate_image/generator.py::_MODALITY_SCALES
CC_IMG    = {"down": {"block_2": [1.0, 1.0]}, "up": {"block_0": [1.0, 1.0, 1.0]}}
CC_STRUCT = {"down": {"block_2": [0.0, 0.5]}, "up": {"block_0": [0.0, 0.0, 0.0]}}
# every one of the 11 SDXL cross-attention levels, uniformly on.
# NOTE `mid` must be a scalar or a 1-element list: diffusers' _maybe_expand_lora_scales
# only special-cases `isinstance(scales["mid"], list)`, so a DICT there is passed through
# unexpanded and lands on the attention processor verbatim -- which fails much later, in
# the readback, as `float() argument must be ... not 'dict'`.  That is how the first run
# of this script lost its A3 arm; see _validate_raw_scales() in generate_layered_decode.py.
ALL_BR = {"down": {"block_1": [1.0, 1.0], "block_2": [1.0, 1.0]},
          "mid": 1.0,
          "up": {"block_0": [1.0, 1.0, 1.0], "block_1": [1.0, 1.0, 1.0]}}
# Mass-matched uniform control.  cc3's total injection mass is 5*1.0 + 0.5 + 0.5 = 6.0
# spread over 3 branches; all3 puts 33 cells at 1.0, i.e. 5.5x that, so comparing the two
# only shows "more conditioning differs".  This spreads the SAME total mass (6.0) over
# the same 33 cells: 33*s = 6  ->  s = 6/33.  Comparing cc3 against this isolates the
# SHAPE of the distribution (deep+one structural level vs uniform) at fixed strength.
UNI_S = 6.0 / 33.0
UNI_BR = {"down": {"block_1": [UNI_S, UNI_S], "block_2": [UNI_S, UNI_S]},
          "mid": UNI_S,
          "up": {"block_0": [UNI_S] * 3, "block_1": [UNI_S] * 3}}

for name, payload in {
    "cc1":  [CC_IMG],
    "cc2":  [CC_IMG, CC_STRUCT],
    "cc3":  [CC_IMG, CC_STRUCT, CC_STRUCT],
    "all3": [ALL_BR, ALL_BR, ALL_BR],
    "uni3": [UNI_BR, UNI_BR, UNI_BR],
}.items():
    (specs / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[specs] {name}.json")

# truly empty prompts (CogCapPro generates with prompt=""), plus the deploy prompts
n = 200
(conds / "prompts_empty.json").write_text(json.dumps([""] * n), encoding="utf-8")
deploy = json.loads(Path("/project/peilab/why/NeuroBridge/outputs/g2f/prompts/"
                         "prompts_deploy.json").read_text(encoding="utf-8"))
print(f"[prompts] empty={n}  deploy={len(deploy)}")
PY

EMPTY_PROMPTS="${CONDS}/prompts_empty.json"

# ---------------------------------------------------------------------------
log "===== [2] verify the injection layout lands where CogCapPro says ====="
"${PYTHON}" - "${SPECS}" <<'PY' || exit 1
import json, sys
from pathlib import Path
specs = Path(sys.argv[1])
L = ["down_blocks.1.attentions.0", "down_blocks.1.attentions.1",
     "down_blocks.2.attentions.0", "down_blocks.2.attentions.1", "mid_block.attentions.0",
     "up_blocks.0.attentions.0", "up_blocks.0.attentions.1", "up_blocks.0.attentions.2",
     "up_blocks.1.attentions.0", "up_blocks.1.attentions.1", "up_blocks.1.attentions.2"]

def expand(cfg):
    """diffusers nested dict -> {level: scale}, mirroring its own indexing.

    `mid` is handled the way `_maybe_expand_lora_scales` really handles it -- a scalar,
    or a 1-element list that gets unwrapped.  A dict is NOT expanded there, so this
    raised form is rejected rather than silently accepted.
    """
    out = {k: 0.0 for k in L}
    for side, blocks in cfg.items():
        if side == "mid":
            if isinstance(blocks, dict):
                raise SystemExit(f"[FATAL] 'mid' must be a scalar or 1-element list, "
                                 f"not a dict: {blocks!r}")
            v = blocks[0] if isinstance(blocks, list) else blocks
            out["mid_block.attentions.0"] = float(v)
            continue
        for bname, vals in blocks.items():
            idx = int(bname.split("_")[-1])
            for j, v in enumerate(vals):
                key = f"{side}_blocks.{idx}.attentions.{j}"
                if key in out:
                    out[key] = float(v)
    return out

SEM5 = ["down_blocks.2.attentions.0", "down_blocks.2.attentions.1",
        "up_blocks.0.attentions.0", "up_blocks.0.attentions.1", "up_blocks.0.attentions.2"]

def semantic_expectation(name):
    """Branch 0's intended map, for every spec we ship."""
    if name in ("cc1", "cc2", "cc3"):
        return {k: (1.0 if k in SEM5 else 0.0) for k in L}
    if name == "all3":
        return {k: 1.0 for k in L}
    if name == "uni3":
        return {k: 6.0 / 33.0 for k in L}     # mass-matched uniform
    raise SystemExit(f"[FATAL] no expectation defined for spec '{name}'")

bad = 0
for name in ("cc1", "cc2", "cc3", "all3", "uni3"):
    cfgs = json.loads((specs / f"{name}.json").read_text())
    maps = [expand(c) for c in cfgs]
    exp0 = semantic_expectation(name)
    diff = {k: (maps[0][k], exp0[k]) for k in L if abs(maps[0][k] - exp0[k]) > 1e-9}
    if diff:
        print(f"[FATAL] {name} branch0 mismatch: {diff}"); bad += 1
    for bi, m in enumerate(maps[1:], start=1):
        if name in ("all3", "uni3"):
            continue          # uniform controls: every branch has the same shape
        if abs(m["down_blocks.2.attentions.1"] - 0.5) > 1e-9 or \
                sum(1 for k in L if m[k] > 0) != 1:
            print(f"[FATAL] {name} structure branch {bi} is not 'one level @0.5': "
                  f"{ {k: m[k] for k in L if m[k] > 0} }"); bad += 1
    mass = sum(sum(mm.values()) for mm in maps)
    active = sum(1 for k in L if any(mm[k] > 0 for mm in maps))
    print(f"[layout] {name}: {len(maps)} branches, {active}/11 levels active, "
          f"total mass {mass:.4f}  {'OK' if not bad else 'MISMATCH'}")
if bad:
    raise SystemExit("[FATAL] scale specs do not express the intended mechanism")
print("[layout] specs PASS - cc1/cc3 reproduce CogCapPro exactly "
      "(5 semantic levels full, 1 structure level at 0.5)")
PY

# ---------------------------------------------------------------------------
log "===== [3] encoder self-check + EEG-derived structure conditions ====="
if [[ ! -f "${CONDS}/eeg_edge1024_cal_${STAG}_test.npy" ]]; then
  if ! "${PYTHON}" "${NB_ROOT}/scripts/nda/nw5_make_struct_conds.py" \
      --self-check --device cuda:0 > "${LOG}/struct_selfcheck.log" 2>&1; then
    log "WARN encoder self-check failed:"; tail -n 12 "${LOG}/struct_selfcheck.log"
  fi
  grep -E "self-check|encoder" "${LOG}/struct_selfcheck.log" | tail -4

  if ! "${PYTHON}" "${NB_ROOT}/scripts/nda/nw5_make_struct_conds.py" \
      --subject "${STAG}" --out-dir "${CONDS}" --device cuda:0 \
      > "${LOG}/struct_conds.log" 2>&1; then
    log "WARN EEG-derived structure conditions failed - A8 will be skipped"
    tail -n 20 "${LOG}/struct_conds.log"
  fi
fi

cond_path() {
  case "$1" in
    SEM)   echo "${SEM_COND}" ;;
    IMGB)  echo "${CC}/clip_img1024_test.npy" ;;
    DEPG)  echo "${CC}/clip_depth1024_test.npy" ;;
    EDGEG) echo "${CC}/clip_edge1024_test.npy" ;;
    DEPE)  echo "${CONDS}/eeg_depth1024_cal_${STAG}_test.npy" ;;
    EDGEE) echo "${CONDS}/eeg_edge1024_cal_${STAG}_test.npy" ;;
    *)     echo "" ;;
  esac
}

# ---------------------------------------------------------------------------
# name | cond keys | scale json (empty = use spec) | branch spec | pipeline |
# steps | guidance | use_cn | use_init | blur | strength | cn_scale | prompts
ARMS=(
  "A1_pure_cc1|SEM|cc1||turbo|15|0.0|0|0|0.0|0.82|0.28|empty"
  "A2_pure_cc3|SEM,DEPG,EDGEG|cc3||turbo|15|0.0|0|0|0.0|0.82|0.28|empty"
  "A3_pure_all3|SEM,DEPG,EDGEG|all3||turbo|15|0.0|0|0|0.0|0.82|0.28|empty"
  "A3b_pure_uni3|SEM,DEPG,EDGEG|uni3||turbo|15|0.0|0|0|0.0|0.82|0.28|empty"
  "A4_base_cc1|SEM|cc1||base|28|5.0|0|0|0.0|0.82|0.28|deploy"
  "A5_band_cc3|SEM,DEPG,EDGEG|cc3||turbo|15|0.0|0|1|3.0|0.92|0.28|empty"
  "A6_cn_cc3|SEM,DEPG,EDGEG|cc3||base|28|5.0|1|1|0.0|0.92|0.28|deploy"
  "A7_ref_ahi|SEM||all:0.9|base|28|5.0|1|1|0.0|0.92|0.28|deploy"
  "A8_eegdep|SEM,DEPE|cc2||turbo|15|0.0|0|0|0.0|0.82|0.28|empty"
  "A8b_eegdepe|SEM,DEPE,EDGEE|cc3||turbo|15|0.0|0|0|0.0|0.82|0.28|empty"
  "A9_orc_cc3|IMGB,DEPG,EDGEG|cc3||turbo|15|0.0|0|0|0.0|0.82|0.28|empty"
)

# ---------------------------------------------------------------------------
# ON-NODE SMOKE -- the three configurations the arm loop relies on are new code paths
# (raw scale dicts; ControlNet and init decoupled; init low-passed).  A pipeline-class
# or shape mistake would otherwise surface an hour in, after the cheap decisive arms had
# already spent GPU time.  Two images each, on the real device.
log "===== [3.5] smoke the new generation paths (2 images each, on this node) ====="
SMOKE="${OUT_ROOT}/_smoke"
rm -rf "${SMOKE}"
smoke_one() {
  local name="$1"; shift
  if ! "${PYTHON}" "${NB_ROOT}/scripts/nda/generate_layered_decode.py" \
      --cond-npys "${CLIST_SMOKE}" --prompts-json "${EMPTY_PROMPTS}" \
      --output-dir "${SMOKE}/${name}" --tag "smoke_${name}" \
      --max-images 2 --gen-size 512 --seed "${SEED}" "$@" \
      > "${LOG}/smoke_${name}.log" 2>&1; then
    log "FATAL smoke ${name} FAILED"; tail -n 25 "${LOG}/smoke_${name}.log"; return 1
  fi
  local nimg bad
  nimg=$(ls "${SMOKE}/${name}/generated" 2>/dev/null | wc -l)
  bad=$(grep -cE "FATAL|Traceback" "${LOG}/smoke_${name}.log" || true)
  log "smoke ${name}: ${nimg} images, ${bad} error lines"
  [[ "${nimg}" -ge 2 && "${bad}" -eq 0 ]]
}

CLIST_SMOKE="${SEM_COND},$(cond_path DEPG),$(cond_path EDGEG)"
SMOKE_OK=1
smoke_one pure_turbo_cc3 --ip-scale-json "${SPECS}/cc3.json" \
  --pipeline turbo --gen-steps 15 --gen-guidance 0.0 --use-cn 0 --use-init 0 \
  --device cuda:0 || SMOKE_OK=0
smoke_one initonly_blur_cc3 --ip-scale-json "${SPECS}/cc3.json" \
  --pipeline turbo --gen-steps 15 --gen-guidance 0.0 --use-cn 0 --use-init 1 \
  --lowlevel-rgb-dir "${INIT_RGB}" --init-blur-sigma 3.0 --strength 0.92 \
  --device cuda:0 || SMOKE_OK=0
smoke_one cn_init_cc3 --ip-scale-json "${SPECS}/cc3.json" \
  --pipeline base --gen-steps 28 --gen-guidance 5.0 --use-cn 1 --use-init 1 \
  --depth-rgb-dir "${DEPTH_RGB}" --lowlevel-rgb-dir "${INIT_RGB}" \
  --cn-scale 0.28 --strength 0.92 --device cuda:0 || SMOKE_OK=0
# the legacy spec path is what A7 uses, and A7 is ONE branch (one spec entry); smoke_one
# always passes the three-branch CLIST_SMOKE, so this one overrides it
CLIST_SMOKE="${SEM_COND}"
smoke_one specpath_all09 --branch-spec "all:0.9" \
  --pipeline base --gen-steps 28 --gen-guidance 5.0 --use-cn 1 --use-init 1 \
  --depth-rgb-dir "${DEPTH_RGB}" --lowlevel-rgb-dir "${INIT_RGB}" \
  --cn-scale 0.28 --strength 0.92 --device cuda:0 || SMOKE_OK=0
CLIST_SMOKE="${SEM_COND},$(cond_path DEPG),$(cond_path EDGEG)"

if [[ "${SMOKE_OK}" -ne 1 ]]; then
  echo "[FATAL] smoke failed - refusing to spend hours of GPU time on a broken path" >&2
  echo "logs: ${LOG}/smoke_*.log" >&2
  exit 1
fi
log "smoke PASS: raw scale dicts, init-only+blur and the legacy spec path all render"

# ---- and prove the mechanism actually landed on the LIVE UNet ---------------
# The expansion was verified offline (`down.block_2.0` -> `down_blocks.2.attentions.0`,
# unlisted groups -> default_scale 0.0).  That is theory.  `apply_raw_scales` reads the
# scales back off `unet.attn_processors`, so if the nesting is wrong the readback would
# show it -- and a silent no-op here would make every arm below uninterpretable.
log "--- verifying the CogCapPro layout landed on the live UNet (readback)"
"${PYTHON}" - "${LOG}/smoke_pure_turbo_cc3.log" <<'PY' || exit 1
import re, sys
from pathlib import Path
txt = Path(sys.argv[1]).read_text(errors="ignore")
rows = re.findall(r"^\s+(down_blocks\.\d\.attentions\.\d|up_blocks\.\d\.attentions\.\d|"
                  r"mid_block\.attentions\.\d)\s+\[([-0-9.,\s]+)\]", txt, re.M)
if not rows:
    raise SystemExit("[FATAL] no readback in the smoke log - cannot confirm the layout")
got = {k: [float(x) for x in v.replace(" ", "").split(",")] for k, v in rows}

# the intended live map, cell by cell: semantic full on its five levels, structure 0.5 on
# down_blocks.2.attentions.1 ONLY -- a level that carries BOTH branches, which is exactly
# the asymmetry that makes the layout worth reproducing
SEM = ["down_blocks.2.attentions.0", "down_blocks.2.attentions.1",
       "up_blocks.0.attentions.0", "up_blocks.0.attentions.1", "up_blocks.0.attentions.2"]
STRUCT = "down_blocks.2.attentions.1"

print(f"{'level':<30}{'semantic':>10}{'depth':>8}{'edge':>8}   (live UNet)")
bad = []
for k in sorted(got):
    v = got[k]
    print(f"{k:<30}{v[0]:>10.2f}"
          f"{v[1] if len(v) > 1 else 0.0:>8.2f}{v[2] if len(v) > 2 else 0.0:>8.2f}")
    exp = [1.0 if k in SEM else 0.0,
           0.5 if k == STRUCT else 0.0,
           0.5 if k == STRUCT else 0.0]
    for i, e in enumerate(exp):
        g = v[i] if i < len(v) else 0.0
        if abs(g - e) > 1e-6:
            bad.append(f"{k}[branch{i}]: got {g} want {e}")

nsem = sum(1 for v in got.values() if v[0] > 0)
nstr = sum(1 for v in got.values() if len(v) >= 3 and v[1] > 0)
if nsem != 5:
    bad.append(f"semantic active on {nsem} levels, want 5")
if nstr != 1:
    bad.append(f"structure active on {nstr} levels, want 1")
if bad:
    print("[FATAL] readback does not match CogCapPro's layout:")
    for b in bad:
        print("   ", b)
    raise SystemExit(1)
print("\n[verify] PASS - semantic on 5 deep levels @1.0, structure on exactly "
      "down_blocks.2.attentions.1 @0.5, everything else off")
PY

log "===== [4] generate + score ${#ARMS[@]} arms on ${STAG} ====="
DONE=()
for spec_line in "${ARMS[@]}"; do
  IFS='|' read -r ARM CKEYS SJSON BSPEC PIPE STEPS GUID UCN UINIT BLUR STRENGTH CNSC PROMPTS EXTRA <<< "${spec_line}"
  # a stray '|' silently shifts every later value into the wrong slot -- which is how A7
  # first ran as `--pipeline all:0.9`.  Refuse to run a malformed definition.
  if [[ -n "${EXTRA:-}" || -z "${ARM}" || -z "${PIPE}" || -z "${PROMPTS}" ]]; then
    echo "[FATAL] arm definition is not 13 fields: '${spec_line}'" >&2
    exit 1
  fi
  if [[ -n "${SJSON}" && -n "${BSPEC}" ]]; then
    echo "[FATAL] ${ARM}: both a scale json (${SJSON}) and a branch spec (${BSPEC})" >&2
    exit 1
  fi
  hr; log "arm ${ARM}: keys=${CKEYS} scales=${SJSON:-spec:${BSPEC}} ${PIPE} ${STEPS}step g=${GUID} cn=${UCN} init=${UINIT} blur=${BLUR}"

  CDIR="${OUT_ROOT}/arms/${ARM}/conds/${STAG}"
  GEN="${OUT_ROOT}/arms/${ARM}/gen/${STAG}"
  EV="${OUT_ROOT}/arms/${ARM}/eval/${STAG}.json"
  mkdir -p "${CDIR}" "${OUT_ROOT}/arms/${ARM}/eval"

  N=0; CLIST=""
  for k in ${CKEYS//,/ }; do
    p="$(cond_path "${k}")"
    if [[ -z "${p}" || ! -f "${p}" ]]; then
      log "WARN ${ARM}: condition ${k} unavailable (${p})"; N=-1; break
    fi
    CLIST="${CLIST:+${CLIST},}${p}"; N=$((N+1))
  done
  [[ "${N}" -eq -1 ]] && { log "WARN ${ARM}: skipped"; continue; }

  PF="${EMPTY_PROMPTS}"
  [[ "${PROMPTS}" == "deploy" ]] && PF="${DEPLOY_PROMPTS}"

  # ---- generation ---------------------------------------------------------
  if [[ -f "${GEN}/generated/199.png" ]]; then
    log "--- ${ARM}: generation already present - skip"
  else
    SCALE_ARGS=()
    if [[ -n "${BSPEC}" ]]; then
      SCALE_ARGS=(--branch-spec "${BSPEC}")
    else
      SCALE_ARGS=(--ip-scale-json "${SPECS}/${SJSON}.json")
    fi
    PIX_ARGS=(--use-cn "${UCN}" --use-init "${UINIT}")
    if [[ "${UCN}" == "1" ]]; then PIX_ARGS+=(--depth-rgb-dir "${DEPTH_RGB}"); fi
    if [[ "${UINIT}" == "1" ]]; then PIX_ARGS+=(--lowlevel-rgb-dir "${INIT_RGB}"); fi

    if ! "${PYTHON}" "${NB_ROOT}/scripts/nda/generate_layered_decode.py" \
        --cond-npys "${CLIST}" "${SCALE_ARGS[@]}" \
        --prompts-json "${PF}" --output-dir "${GEN}" --tag "${ARM}_${STAG}" \
        --pipeline "${PIPE}" "${PIX_ARGS[@]}" \
        --cn-scale "${CNSC}" --strength "${STRENGTH}" \
        --init-blur-sigma "${BLUR}" \
        --gen-steps "${STEPS}" --gen-guidance "${GUID}" \
        --gen-size 512 --seed "${SEED}" --device cuda:0 \
        --layer-report "${OUT_ROOT}/arms/${ARM}/layer_report.json" \
        > "${LOG}/${ARM}_gen.log" 2>&1; then
      log "WARN ${ARM}: generation FAILED"; tail -n 20 "${LOG}/${ARM}_gen.log"; continue
    fi
    grep -E "^\[layout|^\[plan\] tag|^\[INFO\] pipeline|^\[init-only|^\[pure" \
      "${LOG}/${ARM}_gen.log" | head -4
  fi

  # ---- seven-metric eval --------------------------------------------------
  if [[ -f "${EV}" ]]; then
    log "--- ${ARM}: eval present - skip"
  elif ! "${PYTHON}" "${NB_ROOT}/scripts/nda/eval_official_seven_dir.py" \
      --gen-dir "${GEN}/generated" --output-json "${EV}" --tag "${ARM}_${STAG}" \
      --images-root "${IMAGES_ROOT}" --device cuda:0 \
      > "${LOG}/${ARM}_eval.log" 2>&1; then
    log "WARN ${ARM}: eval FAILED"; tail -n 20 "${LOG}/${ARM}_eval.log"; continue
  fi
  DONE+=("${ARM}")
  "${PYTHON}" - "${EV}" "${ARM}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  [{sys.argv[2]}] pixcorr {d['pixcorr']:.4f}  ssim {d['ssim']:.4f}  "
      f"incep {d['inception']:.4f}  clip {d['clip']:.4f}  alex2 {d['alex2']:.4f}  "
      f"alex5 {d['alex5']:.4f}  swav {d['swav']:.4f}  fid {d['fid']:.2f}")
PY
done

# ---------------------------------------------------------------------------
log "===== [5] official (Pearson) 2-way re-scoring of every finished arm ====="
"${PYTHON}" "${NB_ROOT}/scripts/nda/nw4_official_twoway.py" \
  --gen-root "${OUT_ROOT}/arms" --arms "$(IFS=,; echo "${DONE[*]}")" \
  --subjects "8" --images-root "${IMAGES_ROOT}" \
  --out "${OUT_ROOT}/official_twoway.json" --device cuda:0 \
  > "${LOG}/official_twoway.log" 2>&1 || { log "WARN twoway failed"; tail -n 20 "${LOG}/official_twoway.log"; }
tail -n 14 "${LOG}/official_twoway.log" 2>/dev/null

# ---------------------------------------------------------------------------
log "===== [6] summary vs baselines and published bars ====="
"${PYTHON}" - "${OUT_ROOT}" "${STAG}" <<'PY'
import json, sys
from pathlib import Path
root, stag = Path(sys.argv[1]), sys.argv[2]

# reference points, all measured with these same scorers on this same data
REF = {
    "a_hi (shipped, recorded)": Path("/project/peilab/why/NeuroBridge/outputs/nw4_10s/"
                                     "arms/a_hi/eval/sub-08.json"),
}
SOTA = {
    # published numbers, with the scope each was reported at
    "ATM sub-08":       {"pixcorr": 0.160, "ssim": 0.345, "alex2": 0.776,
                         "alex5": 0.866, "inception": 0.734, "clip": 0.786},
    "CogCap 10subj":    {"pixcorr": 0.150, "ssim": 0.347, "alex2": 0.754,
                         "alex5": 0.623, "inception": 0.669, "clip": 0.715},
    "CogCapPro 10subj": {"pixcorr": 0.163, "ssim": 0.398, "inception": 0.779,
                         "clip": 0.830},
    "D2-FOSA":          {"pixcorr": 0.193, "ssim": 0.350, "fid": 146.33},
    "MB2C":             {"pixcorr": 0.188, "ssim": 0.333, "fid": 163.94},
}
KEYS = ["pixcorr", "ssim", "inception", "clip", "alex2", "alex5", "swav", "fid"]

rows = {}
for d in sorted((root / "arms").glob("*/eval")):
    f = d / f"{stag}.json"
    if f.is_file():
        rows[d.parent.name] = json.load(open(f))
for name, p in REF.items():
    if p.is_file():
        rows[name] = json.load(open(p))

print()
print("=" * 118)
print(f"NW5 {stag}: does the CogCapPro mechanism (pure IP, no pixel constraint) "
      f"beat the pixel-constrained chain?")
print("=" * 118)
hdr = f"{'arm':<26}" + "".join(f"{k:>9}" for k in KEYS)
print(hdr); print("-" * len(hdr))
for arm, r in rows.items():
    print(f"{arm:<26}" + "".join(
        f"{r[k]:>9.4f}" if k != "fid" else f"{r[k]:>9.2f}" for k in KEYS))

print()
print("published bars (10-subject means; our FID has no comparable published value):")
for name, bar in SOTA.items():
    print(f"  {name:<20}" + "".join(f"{k}={bar[k]:<7}" for k in bar if k in bar))

if "a_hi (shipped, recorded)" in rows:
    print()
    print("delta vs the shipped a_hi chain (the thing we have been calling SOTA-adjacent):")
    base = rows["a_hi (shipped, recorded)"]
    for arm, r in rows.items():
        if arm.startswith("a_hi") or arm.startswith("A7"):
            continue
        d = "  ".join(
            f"{k}{r[k] - base[k]:+.4f}" for k in ["pixcorr", "ssim", "inception", "clip"]
            if k in r and k in base)
        win = sum(1 for k in ["pixcorr", "ssim", "inception", "clip", "alex2", "alex5"]
                  if k in r and k in base and r[k] > base[k])
        print(f"  {arm:<26} {d}   ({win}/6 axes better)")

# ---- how good were the STRUCTURE conditions this run actually fed in? -------
# A8/A8b are the only arms whose structure comes from EEG rather than from GT.  Their
# result is only readable next to the concentration and correspondence of the rows they
# consumed, so those numbers are printed here instead of being left in a side file.
sq = root / "conds" / f"eeg_struct_{stag}_report.json"
if sq.is_file():
    r = json.load(open(sq))
    print()
    print(f"EEG-derived structure conditions used by A8/A8b "
          f"(encoder self-check cos={r.get('self_check_cos')}):")
    print(f"  {'variant':<12}{'rowcos':>9}{'rowcos_cal':>12}{'bank':>9}"
          f"{'cos_own_gt':>12}{'nn_gt':>9}")
    for k, v in r.get("variants", {}).items():
        print(f"  {k:<12}{v['rowcos']:>9.4f}{v['rowcos_cal']:>12.4f}"
              f"{v['bank_rowcos_test']:>9.4f}{v['diag_cos_to_gt']:>12.4f}"
              f"{v['nn_cos_to_gt_bank']:>9.4f}")
    de = r.get("variants", {}).get("eeg_edge", {})
    dd = r.get("variants", {}).get("eeg_depth", {})
    if dd:
        verdict_d = "USABLE" if dd["diag_cos_to_gt"] > 0.30 else "WEAK"
        print(f"  -> depth: calibrated {dd['rowcos']:.4f} -> {dd['rowcos_cal']:.4f} "
              f"against bank {dd['bank_rowcos_test']:.4f}, "
              f"cos to its own true depth row {dd['diag_cos_to_gt']:.4f}  {verdict_d}")
    if de:
        ok = (de["rowcos_cal"] < 0.70 and de["diag_cos_to_gt"] > 0.30)
        verdict_e = "USABLE" if ok else "DEGENERATE"
        print(f"  -> edge:  calibrated {de['rowcos']:.4f} -> {de['rowcos_cal']:.4f} "
              f"against bank {de['bank_rowcos_test']:.4f}, "
              f"cos to its own true edge row {de['diag_cos_to_gt']:.4f}  {verdict_e}")
        if not ok:
            print("     Edge rows carry almost no trial information: Canny over the "
                  "predicted low-level")
            print("     image is near-constant across trials, so no EEG->edge CLIP head can "
                  "be faked")
            print("     from the spatial path.  Read A8b as a negative control on structure "
                  "quality,")
            print("     NOT as a test of whether an edge branch helps.  That is a result, "
                  "not a bug.")

tw = root / "official_twoway.json"
if tw.is_file():
    o = json.load(open(tw))
    print()
    print("2-way under the OFFICIAL Pearson protocol (the axis CogCap/CogCapPro publish):")
    # pixcorr/ssim/fid are recorded as {value,std,n} while the 2-way metrics are
    # {cos,pearson,pearson_std,n}; only the latter carry a 'pearson' key
    for arm, rec in o.get("arms", {}).items():
        m = rec.get("mean", {})
        cells = []
        for k in ("inception", "clip", "alex2", "alex5"):
            v = m.get(k)
            if isinstance(v, dict) and "pearson" in v:
                cells.append(f"{k} {v['pearson']:.4f} (cos {v['cos']:.4f})")
        if cells:
            print(f"  {arm:<26}" + "   ".join(cells))
    print("  published: ATM sub-08 incep 0.7340 clip 0.7860 | "
          "CogCapPro incep 0.7790 clip 0.8300 | CogCap incep 0.6690 clip 0.7150")

Path(root / "NW5_S08_SUMMARY.json").write_text(
    json.dumps({"arms": rows, "sota_bars": SOTA}, indent=2), encoding="utf-8")
print(f"\n[wrote] {root}/NW5_S08_SUMMARY.json")
PY

hr
log "NW5 ${STAG} finished: ${#DONE[@]}/${#ARMS[@]} arms generated and scored -> ${OUT_ROOT}"
