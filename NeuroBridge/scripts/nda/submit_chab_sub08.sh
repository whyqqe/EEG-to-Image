#!/usr/bin/env bash
# Preflight + CPU smoke test for CHAB (17 vs 63 electrodes), then submit.
#
# The smoke test is not ceremony here.  A montage change is the one edit that can
# fail SILENTLY: the encoder's input is a flat weight, so a wrong electrode order,
# a stale checkpoint field, or an unguardeded exporter all run to completion and
# emit plausible features that no downstream metric can flag as wrong.  So every
# check below is either (a) an assertion that the dilation is exact, or (b) a
# NEGATIVE control that must fail.  A check that cannot fail is not a check.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"

PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
SMOKE="$(mktemp -d /tmp/chab_smoke.XXXXXX)"
trap 'rm -rf "${SMOKE}"' EXIT

chk() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
say() { echo; echo "===== $* ====="; }

say "0/6 sources present"
chk scripts/nda/ss_modules.py
chk scripts/nda/nda_ss_pretrain.py
chk scripts/nda/ocf_export_intra_z.py
chk scripts/nda/channel_ablation_audit.py
chk scripts/nda/cfmsf_route_probe.py
chk scripts/nda/run_chab_sub08.sh
chk slurm/chab_s08.sbatch
chk data/things_eeg/preprocessed_eeg/info.json

say "1/6 baseline assets the experiment is measured against"
chk outputs/ocf/intra_enc/sub-08/checkpoint_ss_calib_best.pth
chk outputs/ocf/intra_z/sub-08/shared_r_train.npy
chk outputs/ocf/intra_z/sub-08/shared_r_test.npy
chk outputs/cfmsf_all/sub-08/frozen/probe/route_probe.json
# the targets the 13 legacy routes read
for lv in image GaussianBlur LowResolution Mosaic GaussianNoise; do
  if [[ "${lv}" == "image" ]]; then
    chk data/things_eeg/image_feature/ViT-H-14/image_train.npy
    chk data/things_eeg/image_feature/ViT-H-14/image_test.npy
  else
    chk "data/things_eeg/image_feature/ViT-H-14/${lv}/train.npy"
    chk "data/things_eeg/image_feature/ViT-H-14/${lv}/test.npy"
  fi
done
chk "data/things_eeg/image_feature/ViT-H-14/GaussianBlur-GaussianNoise-LowResolution-Mosaic/train.npy"
chk outputs/gem/cond_cache/clip_img1024_train.npy
chk outputs/gem/cond_cache/clip_depth1024_train.npy
chk outputs/gem/cond_cache/clip_edge1024_train.npy
chk outputs/nda_ss/sub-08/clip_text/train/concept_phrases.json
chk outputs/leakfree/split.json

say "2/6 the baseline we must reproduce (read, not assumed)"
"${PYTHON}" - <<'PY'
import json, sys
r = json.load(open("outputs/cfmsf_all/sub-08/frozen/probe/route_probe.json"))
best = max(r["targets"].items(), key=lambda kv: kv[1]["mlp"]["top1"])
print(f"  baseline best single route: {best[0]} top1={best[1]['mlp']['top1']:.4f} "
      f"csls={best[1]['mlp']['top1_csls']:.4f}")
f = r["fusion"]["mlp"]
print(f"  baseline 4-route fusion:    csls={f['csls']['top1']:.4f} "
      f"+sinkhorn={f['csls']['sinkhorn_top1']:.4f} over {len(f['routes'])} routes")
assert len(r["targets"]) == 13, f"expected 13 legacy routes, got {len(r['targets'])}"
assert best[0] == "vith_cat5", (
    f"the pre-registered verdict in run_chab_sub08.sh names vith_cat5 as the best "
    f"single route, but the baseline file says {best[0]}. Fix the header before "
    f"submitting, or the verdict text will describe a different experiment."
)
print("  [ok] 13 legacy routes, best single route == vith_cat5 as documented")
PY

say "3/6 static wiring: the montage flags actually exist where they are used"
"${PYTHON}" - <<'PY'
import ast, sys
from pathlib import Path

def arg_flags(path):
    tree = ast.parse(Path(path).read_text())
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument":
            for a in n.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    out.add(a.value)
    return out

needs = {
    "scripts/nda/nda_ss_pretrain.py": {"--channels", "--init-ss-checkpoint", "--warm-start-only"},
    "scripts/nda/ocf_export_intra_z.py": {"--channels"},
    "scripts/nda/cfmsf_route_probe.py": {"--z-root", "--epochs", "--out"},
}
bad = []
for f, want in needs.items():
    have = arg_flags(f)
    miss = want - have
    print(f"  [{'ok  ' if not miss else 'FAIL'}] {f}: {sorted(want)}")
    if miss:
        bad.append((f, sorted(miss)))
if bad:
    sys.exit(f"[FATAL] missing arguments: {bad}")

# the orchestrator must pass the flags the scripts now demand
orch = Path("scripts/nda/run_chab_sub08.sh").read_text()
for must in ("--channels all", "--init-ss-checkpoint", "--warm-start-only",
             "--z-root"):
    ok = must in orch
    print(f"  [{'ok  ' if ok else 'FAIL'}] run_chab_sub08.sh contains {must!r}")
    if not ok:
        bad.append(("run_chab_sub08.sh", must))
if bad:
    sys.exit(f"[FATAL] orchestrator/script mismatch: {bad}")
PY

say "4/6 the dilation helpers behave (tiny, no dataset)"
"${PYTHON}" - <<'PY'
import json, sys
sys.path.insert(0, "scripts/nda")
import torch
from ss_modules import (CHANNEL_SETS, POSTERIOR_17, channel_indices,
                        dilate_first_linear, resolve_channels)

allch = json.load(open("data/things_eeg/preprocessed_eeg/info.json"))["ch_names"]
assert len(allch) == 63, len(allch)
assert resolve_channels("posterior") == POSTERIOR_17
assert resolve_channels("all") == [], "empty means whole montage, per the dataset"
assert sorted(CHANNEL_SETS) == ["all", "posterior"]

idx = channel_indices(POSTERIOR_17, allch)
assert idx == list(range(46, 63)), f"posterior must be the tail block, got {idx}"
print(f"  [ok  ] posterior electrodes are the tail block {idx[0]}..{idx[-1]} of 63")

T, Fd = 250, 64
W17, b17 = torch.randn(Fd, 17 * T), torch.randn(Fd)
W63, b63 = dilate_first_linear(W17, b17, idx, 63, T)
x17 = torch.randn(3, 17 * T)
x63 = torch.zeros(3, 63 * T)
for j, c in enumerate(idx):
    x63[:, c * T:(c + 1) * T] = x17[:, j * T:(j + 1) * T]
assert torch.allclose(x17 @ W17.T + b17, x63 @ W63.T + b63, atol=1e-4)
assert float(W63[:, :46 * T].abs().sum()) == 0.0
print("  [ok  ] dilation is exact on kept channels and zero elsewhere")

# negative controls: these MUST raise, otherwise the checks above are decorative
for bad_call, why in (
    (lambda: dilate_first_linear(torch.randn(Fd, 17 * T), b17, idx, 63, T - 1),
     "a sample-count mismatch must be rejected, not silently reshaped"),
    (lambda: channel_indices(["P7", "ZZZ_absent"], allch),
     "a channel absent from the montage must be rejected, not dropped"),
):
    try:
        bad_call()
    except Exception:
        print(f"  [ok  ] negative control fired: {why}")
    else:
        sys.exit(f"[FATAL] negative control did NOT fire: {why}")
PY

say "5/6 the MONTAGE GUARD: the exporter must inherit, and must refuse a mismatch"
# The guard is the only thing between a typo and a silently scrambled feature file.
# Positive control: NO --channels, so the montage is inherited from the checkpoint
# (17ch) and the export must succeed. Negative control: ask for a montage the
# checkpoint was never trained on and require a non-zero exit.
if "${PYTHON}" scripts/nda/ocf_export_intra_z.py \
     --subject 8 --checkpoint outputs/ocf/intra_enc/sub-08/checkpoint_ss_calib_best.pth \
     --out "${SMOKE}/guard_pos" --device cpu > "${SMOKE}/guard_pos.log" 2>&1; then
  echo "  [ok  ] control: an omitted --channels inherits the checkpoint montage (17ch)"
  grep -o "channels=posterior n_ch=17" "${SMOKE}/guard_pos.log" | head -1 | sed 's/^/         /'
else
  echo "  [FAIL] the control export failed, so the mismatch check below proves nothing"
  tail -20 "${SMOKE}/guard_pos.log"; exit 1
fi

if "${PYTHON}" scripts/nda/ocf_export_intra_z.py \
     --subject 8 --checkpoint outputs/ocf/intra_enc/sub-08/checkpoint_ss_calib_best.pth \
     --out "${SMOKE}/guard_neg" --channels all --device cpu > "${SMOKE}/guard.log" 2>&1; then
  echo "  [FAIL] exporter ACCEPTED a 63-channel request for a 17-channel checkpoint."
  echo "         That is the silent-failure mode this whole exercise exists to avoid."
  exit 1
else
  echo "  [ok  ] exporter refused the mismatch:"
  grep -o "\[FATAL\].*" "${SMOKE}/guard.log" | head -1 | sed 's/^/         /'
fi

say "6/6 orchestration dry checks (no training)"
bash -n scripts/nda/run_chab_sub08.sh && echo "  [ok  ] run_chab_sub08.sh parses"
bash -n slurm/chab_s08.sbatch && echo "  [ok  ] chab_s08.sbatch parses"
"${PYTHON}" -c "
import ast,sys
src=open('scripts/nda/run_chab_sub08.sh').read()
for must in ('set -euo pipefail','channel_ablation_audit.py','warm-start-only',
             'cfmsf_route_probe.py','PRE-REGISTERED VERDICT'):
    assert must in src, must
print('  [ok  ] orchestrator has the audit gate, the equivalence stage, both arms, the verdict')
"
grep -q "TIME BUDGET" slurm/chab_s08.sbatch && echo "  [ok  ] sbatch records its time budget"
# the verdict thresholds are the whole point: if they drift after the fact the
# result is uninterpretable, so they are asserted to be present as written
for t in "0.15" "0.04" "0.02"; do
  grep -q "${t}" scripts/nda/run_chab_sub08.sh || { echo "[FATAL] verdict threshold ${t} missing"; exit 1; }
done
echo "  [ok  ] pre-registered thresholds present in the verdict block"

say "ALL PREFLIGHT CHECKS PASSED"
echo "  baseline to beat: vith_cat5 0.4050 top1 / 0.4950 csls (job 581704, sub-08)"
echo "  submitting chab_s08 ..."
sbatch slurm/chab_s08.sbatch
squeue -u "${USER}" -o "%.10i %.9P %.16j %.8T %.10M %R" | head -6
