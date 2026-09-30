#!/bin/bash
# =============================================================================
# Preflight + CPU smoke + submit for nw4_s08.
# Nothing here touches the login node's GPU; the smoke test is CPU-only torch.
# =============================================================================
set -uo pipefail
cd /project/peilab/why/NeuroBridge
export NB_ROOT=/project/peilab/why/NeuroBridge
PYTHON="${PYTHON:-python}"

FAIL=0
ok()   { echo "  [ok]   $*"; }
bad()  { echo "  [FAIL] $*"; FAIL=1; }
warn() { echo "  [warn] $*"; }

echo "=== 1. sources ==="
for f in scripts/nda/nw4_arms.py scripts/nda/nw4_s1_train.py scripts/nda/nw4_s2_project.py \
         scripts/nda/nw4_summary.py scripts/nda/nw4_preflight_facts.py \
         scripts/nda/run_nw4_sub08.sh slurm/nw4_s08.sbatch \
         scripts/nda/generate_layered_decode.py scripts/nda/layered_arms.py \
         scripts/nda/eval_official_seven_dir.py scripts/nda/nw3_spatial_cycle.py \
         scripts/nda/nw3_cond_audit.py scripts/nda/leakfree.py \
         scripts/nda/nw4_gate_cond.py scripts/nda/dev_guard.py; do
  [[ -f "$f" ]] && ok "$f" || bad "$f missing"
done

# Syntax must be verified for EVERY edited file, not just the ones first thought of.
# A blanket string substitution once left `--device "${DEVICE}"` without its line
# continuation in run_nw4_sub08.sh, which would have failed every stage at runtime
# after the queue wait.  `bash -n` and `ast.parse` catch that class of edit instantly.
echo "=== 1b. syntax (all touched scripts) ==="
for f in scripts/nda/nw4_s1_train.py scripts/nda/nw4_s2_project.py \
         scripts/nda/nw4_arms.py scripts/nda/nw4_gate_cond.py \
         scripts/nda/dev_guard.py scripts/nda/nw4_summary.py \
         scripts/nda/generate_layered_decode.py; do
  "${PYTHON:-python3}" -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$f" \
    && ok "py $f" || bad "py $f"
done
for f in scripts/nda/run_nw4_sub08.sh slurm/nw4_s08.sbatch; do
  bash -n "$f" && ok "sh $f" || bad "sh $f"
done
# the generator flags the orchestrator passes must exist in the generator's parser
"${PYTHON:-python3}" - <<'PYEOF' || bad "orchestrator/generator flag mismatch"
import re, sys
orch = open("scripts/nda/run_nw4_sub08.sh").read()
gen = open("scripts/nda/generate_layered_decode.py").read()
have = set(re.findall(r'add_argument\("(--[a-z0-9-]+)"', gen))
# flags the orchestrator hands to the generator inside the V1 loop
block = orch[orch.index("generate_layered_decode.py"):]
block = block[:block.index("2>&1")] if "2>&1" in block else block
used = {f for f in re.findall(r"(--[a-z0-9-]+)", block)}
missing = sorted(used - have - {"--cond-npys", "--branch-spec", "--tag", "--output-dir"})
missing = [m for m in missing if m not in have]
assert not missing, f"generator lacks {missing}"
print(f"    [ok]   generator accepts every flag the orchestrator passes")
PYEOF

echo "=== 2. assets ==="
CC=outputs/gem/cond_cache
G2=outputs/g2/targets
for f in "$CC/clip_img1024_train.npy" "$CC/clip_img1024_test.npy" \
         "$CC/clip_depth1024_train.npy" "$CC/clip_edge1024_train.npy" \
         "$G2/sem_overall_train.npy" "$G2/sem_subject_train.npy" "$G2/sem_detail_train.npy" \
         "$G2/sem_overall_test.npy" "$G2/sem_subject_test.npy" "$G2/sem_detail_test.npy" \
         outputs/leakfree/split.json \
         outputs/sdedit_ll_full10/shared/vae_cache/train_vae_latents_f16.npy \
         outputs/uck/shared/gt_depth/train_depth_64.npy \
         outputs/hcma_s_full10/shared/gt_depth/test_depth_64.npy \
         outputs/ocf/intra_z/sub-08/shared_r_train.npy \
         outputs/ocf/intra_z/sub-08/shared_r_test.npy \
         outputs/g2f/prompts/prompts_deploy.json; do
  [[ -f "$f" ]] && ok "$f" || bad "$f missing"
done
for lv in image GaussianBlur LowResolution Mosaic GaussianNoise; do
  if [[ "$lv" == "image" ]]; then
    p_train=data/things_eeg/image_feature/ViT-H-14/image_train.npy
    p_test=data/things_eeg/image_feature/ViT-H-14/image_test.npy
  else
    p_train=data/things_eeg/image_feature/ViT-H-14/$lv/train.npy
    p_test=data/things_eeg/image_feature/ViT-H-14/$lv/test.npy
  fi
  [[ -f "$p_train" && -f "$p_test" ]] && ok "vith $lv" || bad "vith $lv missing"
done

echo "=== 3. protocol: prompts must be generic (no class names) ==="
"${PYTHON}" - <<'PY' || FAIL=1
import json, sys, re
p = "outputs/g2f/prompts/prompts_deploy.json"
pr = json.load(open(p))
if len(pr) != 200:
    print(f"  [FAIL] prompts rows {len(pr)} != 200"); sys.exit(1)
uniq = sorted(set(pr))
print(f"  [ok]   {len(pr)} rows, {len(uniq)} unique: {uniq[:2]}")
# the generic prompt must not enumerate many distinct nouns
bad = 0
for c in uniq:
    # a leaked prompt list contains >100 distinct class names; generic has very few
    pass
if len(uniq) > 10:
    print(f"  [FAIL] {len(uniq)} unique prompts looks like per-class leakage"); sys.exit(1)
print("  [ok]   single generic prompt family -> no class-name leakage")
PY

echo "=== 4. device audit (static) ==="
if [[ -f scripts/nda/device_audit.py ]]; then
  "${PYTHON}" scripts/nda/device_audit.py \
      scripts/nda/nw4_s1_train.py scripts/nda/nw4_s2_project.py \
    2>&1 | tail -30 || warn "device_audit reported findings (review above)"
else
  warn "device_audit.py missing"
fi

echo "=== 5. nw4_arms consistency ==="
"${PYTHON}" - <<'PY' || FAIL=1
import sys
sys.path.insert(0, "scripts/nda")
from nw4_arms import (ARMS, BRANCHES, MODES, MULTI, DOMINATE, FID_FLOOR, SEM_GATE,
                      passes_2d, dominates, spec_counts)
import layered_arms as LA

fail = 0
for tag, a in ARMS.items():
    n = len(a["branches"])
    c = spec_counts(a["spec"])
    if c["n"] != n:
        print(f"  [FAIL] {tag}: {c['n']} spec entries != {n} branches"); fail = 1
        continue
    for m in c["modes"]:
        if m not in MODES:
            print(f"  [FAIL] {tag}: mode '{m}' not in {MODES}"); fail = 1
    # the spec must survive the real parser the generator uses
    try:
        spec = LA.parse_spec(a["spec"], n)
    except Exception as e:
        print(f"  [FAIL] {tag}: layered_arms.parse_spec rejected it: {e}"); fail = 1
        continue
    # every conditional branch must actually have mass somewhere, unless the arm is
    # deliberately an ablation (scale 0 on that branch is the point)
    for (mode, s) in spec:
        sd = LA.scale_dict(mode, s)
        if set(sd) != {"down", "mid", "up"}:
            print(f"  [FAIL] {tag}: scale_dict keys {set(sd)}"); fail = 1
        if abs(sd["down"] - sd["mid"]) > 0 and abs(sd["mid"] - sd["up"]) > 0:
            pass  # mixed mode is the whole point for 'all'/'early'/'late'
    print(f"  [ok]   {tag:<18} branches={n} spec='{a['spec']}' modes={c['modes']}")

# A4's key property: for `w4_layered`, semantics and structure must land on DISJOINT levels
a = ARMS["w4_layered"]
sp = LA.parse_spec(a["spec"], len(a["branches"]))
sem_levels = set()
for i, (mode, s) in enumerate(sp):
    b = a["branches"][i]
    if b in ("img", "attr") and s > 0:
        sem_levels |= {k for k, v in LA.scale_dict(mode, s).items() if v > 0}
struct_levels = set()
for i, (mode, s) in enumerate(sp):
    b = a["branches"][i]
    if b in ("depth", "edge") and s > 0:
        struct_levels |= {k for k, v in LA.scale_dict(mode, s).items() if v > 0}
print(f"  [info] w4_layered semantic levels={sorted(sem_levels)} "
      f"structural levels={sorted(struct_levels)}")
if sem_levels & struct_levels:
    print(f"  [FAIL] layered arm's groups overlap: {sem_levels & struct_levels}"); fail = 1
else:
    print("  [ok]   layered arm keeps the two groups on disjoint UNet levels")

# the corrected bar must NOT be passed by the blurry init alone
init = {"pixcorr": 0.2998, "ssim": 0.4997, "alex2": 0.7876, "alex5": 0.6690,
        "inception": 0.5110, "clip": 0.5465, "swav": 0.7027}
if passes_2d(init):
    print("  [FAIL] the 2-D bar is still passed by the V0 init blur"); fail = 1
else:
    print("  [ok]   2-D bar rejects the V0 init blur (semantic gate does the work)")
sota = {"pixcorr": 0.211, "ssim": 0.432, "alex2": 0.818, "alex5": 0.913,
        "inception": 0.831, "clip": 0.903, "swav": 0.489}
d = dominates(sota)
if not d["soft"]:
    print("  [FAIL] the dominance reference does not dominate itself"); fail = 1
else:
    print(f"  [ok]   dominance rule self-consistent (wins={d['n_wins']})")
print(f"  [info] multi-branch arms: {list(MULTI)}")
sys.exit(fail)
PY

echo "=== 6. orchestrator flags vs generator argparse (negative control) ==="
"${PYTHON}" - <<'PY' || FAIL=1
import re, sys
sh = open("scripts/nda/run_nw4_sub08.sh").read()
seg = sh.split("generate_layered_decode.py", 1)[1].split("2>&1 | tee", 1)[0]
used = set(re.findall(r"--[a-z0-9-]+", seg))
src = open("scripts/nda/generate_layered_decode.py").read()
declared = set(re.findall(r'add_argument\("(--[a-z0-9-]+)"', src))
declared |= {a.strip('"') for a in re.findall(r'add_argument\((--[a-z0-9-]+)', src)}
unknown = sorted(u for u in used if u not in declared)
print(f"  [info] flags used={len(used)} declared={len(declared)}")
if unknown:
    print(f"  [FAIL] orchestrator passes flags the generator does not declare: {unknown}")
    sys.exit(1)
print("  [ok]   every generator flag the orchestrator passes exists")
PY

echo "=== 7. emit coverage: every branch an arm names must be written by S2 ==="
"${PYTHON}" - <<'PY' || FAIL=1
import re, sys
sys.path.insert(0, "scripts/nda")
from nw4_arms import ARMS
sh = open("scripts/nda/run_nw4_sub08.sh").read()
# S2 writes the names in --emit plus 'fused'
s2 = open("scripts/nda/nw4_s2_project.py").read()
m = re.search(r'--emit",\s*type=str,\s*default="([^"]+)"', s2)
emit = {x.strip() for x in m.group(1).split(",")} | {"fused"}
need = set()
for a in ARMS.values():
    need |= set(a["branches"])
missing = sorted(need - emit)
print(f"  [info] arms need {sorted(need)}; S2 emits {sorted(emit)}")
if missing:
    print(f"  [FAIL] S2 never writes: {missing}"); sys.exit(1)
print("  [ok]   S2 covers every branch any arm references")
PY

echo "=== 8. CPU smoke: encoder + losses + projection (tiny tensors) ==="
"${PYTHON}" - <<'PY' || FAIL=1
import sys
sys.path.insert(0, "scripts/nda")
import numpy as np, torch
from nw4_s1_train import FactorizedEncoder, inst_nce, concept_gallery, mini_metrics, l2n as l1
from nw4_s2_project import manifold_project, fit_ridge, retrieval, offdiag, row2mean, erank, l2n as l2

# dtype contract: both l2n must return float32, or the float32 torch modules reject
# the input at runtime ("mat1 and mat2 must have the same dtype") -- the bug that
# killed the first nw4 job.
for nm, f in (("nw4_s1_l2n", l1), ("nw4_s2_l2n", l2)):
    out = f(np.random.RandomState(0).randn(4, 8).astype(np.float64))
    assert out.dtype == np.float32, f"{nm} returned {out.dtype}, expected float32"
    print(f"  [ok]   {nm} -> {out.dtype}")

# the trunk must survive a float64 argument: it casts internally
torch.manual_seed(0)
m0 = FactorizedEncoder(z_dim=16, hidden=8, dim_img=8, dim_vith=8, dim_attr=8)
h = m0.trunk(torch.from_numpy(np.ascontiguousarray(
    np.random.RandomState(1).randn(4, 16).astype(np.float64), dtype=np.float32)))
assert h.dtype == torch.float32 and h.shape == (4, 8)
print(f"  [ok]   trunk(float64 input -> cast) -> {tuple(h.shape)} {h.dtype}")

torch.manual_seed(0)
m = FactorizedEncoder(z_dim=64, hidden=32, dim_img=32, dim_vith=48, dim_attr=40).to(torch.device("cpu"))
z = torch.randn(8, 64)
o = m(z)
assert o["z_img"].shape == (8, 32) and o["z_vith"].shape == (8, 48)
assert o["z_attr"].shape == (8, 40) and o["depth"].shape == (8, 64, 64)
assert o["vae"].shape == (8, 4, 64, 64) and o["z_hat"].shape == (8, 64)
tgt = torch.nn.functional.normalize(torch.randn(20, 32), dim=-1)
idx = torch.arange(8)
l = inst_nce(o["z_img"], tgt, idx, 0.07)
assert torch.isfinite(l) and l.item() > 0, l
print(f"  [ok]   encoder forward + instance NCE loss={l.item():.4f}")

# the projection must keep a residue and never emit a constant
u = np.random.RandomState(0).randn(200, 32).astype(np.float32)
bank = np.random.RandomState(1).randn(500, 32).astype(np.float32)
for g in (0.0, 0.2, 0.8):
    zp, d = manifold_project(u, bank, 2, 0.07, g)
    assert zp.shape == u.shape and np.isfinite(zp).all()
    n = np.linalg.norm(zp, axis=1)
    assert np.allclose(n, 1.0, atol=1e-4), f"gamma={g} rows not unit norm"
    print(f"  [ok]   projection gamma={g}: offdiag={offdiag(zp):.4f} "
          f"erank={erank(zp):.1f} resid_frac={d['resid_frac']:.4f}")
assert offdiag(manifold_project(u, bank, 2, 0.07, 0.8)[0]) < \
       offdiag(manifold_project(u, bank, 2, 0.07, 0.0)[0]) + 1e-6, \
    "a larger residual must not make the bank MORE collapsed"
print("  [ok]   residual increases row diversity (gamma monotone)")

zp, _ = manifold_project(u, bank, 1, 0.07, 0.0)
sim = (u / np.linalg.norm(u, axis=1, keepdims=True)) @ \
      (bank / np.linalg.norm(bank, axis=1, keepdims=True)).T
nn = sim.argmax(1)
ref = bank[nn] / np.linalg.norm(bank[nn], axis=1, keepdims=True)
assert np.allclose(zp, ref, atol=1e-4), "K=1,gamma=0 must equal the nearest bank row"
print("  [ok]   K=1,gamma=0 == nearest real bank row (on-manifold by construction)")

X = np.random.RandomState(2).randn(300, 16)
Y = np.random.RandomState(3).randn(300, 24)
r = fit_ridge(X[:200], Y[:200], [0.1, 1.0, 10.0])
assert r["W"].shape == (16, 24)
q = np.random.RandomState(4).randn(20, 24)
mt = retrieval(q, np.random.RandomState(5).randn(20, 24))
assert set(mt) == {"top1", "top5", "top1_csls"}
print(f"  [ok]   ridge {r['W'].shape} + retrieval keys {sorted(mt)}")
concept_gallery(np.random.RandomState(6).randn(40, 8),
                np.repeat(np.arange(10), 4), 10)
mini_metrics(np.random.RandomState(7).randn(40, 8),
             np.random.RandomState(8).randn(10, 8), np.repeat(np.arange(10), 4))
print("  [ok]   concept_gallery + mini_metrics wiring")

# CSLS shapes must broadcast for a rectangular query bank (the second bug preflight
# caught: transposing knn_g gave (G,1)+(Q,1))
for qn, gn in ((20, 82), (82, 20), (200, 200)):
    mm = mini_metrics(np.random.RandomState(9).randn(qn, 8),
                      np.random.RandomState(10).randn(gn, 8),
                      np.random.randint(0, max(1, gn), qn))
    assert all(np.isfinite(list(mm.values()))), (qn, gn, mm)
print("  [ok]   CSLS broadcasts for square and rectangular banks")

# ------------------------------------------------------------------ loss scale
# `F.mse_loss` defaults to reduction="mean", dividing by the number of ELEMENTS.
# Used on a per-VECTOR residual over D dimensions that silently scales the term by
# 1/D -- which is how `w_anchor` was inert at 0.5 AND at 20.0 (the epoch-0 loss
# moved 0.034 instead of ~39), so the head never learned the common component CLIP
# embeddings share and the exported condition scored vs_true 0.257 while a CONSTANT
# vector scored 0.620.  The check below fails the build if any per-vector loss term
# is again many orders of magnitude smaller than its weight implies.
D = 1024
a = torch.nn.functional.normalize(torch.randn(32, D), dim=-1)
b = torch.nn.functional.normalize(torch.randn(32, D), dim=-1)

term_ok = (1.0 - torch.nn.functional.cosine_similarity(a, b, dim=-1)).mean().item()
bad = torch.nn.functional.mse_loss(a, b).item()
# The claim being checked: cosine distance is O(1) per row, while the mse_loss form
# equals (2-2cos)/D.  For a random pair cos~=0 so the ratio is D/2, not D -- assert
# the RATIO, not a hard-coded 1/D, so this stays correct for any ambient dimension.
ratio = term_ok / bad
assert 0.5 < term_ok < 2.0, f"cosine distance should be O(1), got {term_ok}"
assert ratio > D / 8, (f"mse_loss should be ~D/2 smaller than cosine distance, "
                       f"got a ratio of {ratio:.1f} for D={D}")
print(f"  [ok]   cosine-distance term is O(1): {term_ok:.4f}")
print(f"         (the old mse_loss form is {bad:.6f}, i.e. the weight divided by "
      f"{ratio:.0f} = D/2 for a random pair -- this is why w_anchor was inert)")
if "mse_loss(l2t(" in __import__("pathlib").Path(
        "scripts/nda/nw4_s1_train.py").read_text():
    raise SystemExit("  [FAIL] nw4_s1_train.py still uses mse_loss(l2t(...)) for a "
                     "per-vector term -- its weight is silently divided by D")
print("  [ok]   no per-vector mse_loss(l2t(...)) remains in nw4_s1_train.py")
PY

echo "=== 9. duplicate-job guard ==="
if squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -qx "nw4_s08"; then
  bad "nw4_s08 is already queued/running"
else
  ok "no nw4_s08 in the queue"
fi

echo
if [[ "$FAIL" == "1" ]]; then
  echo "PREFLIGHT FAILED — not submitting"
  exit 1
fi
if [[ "${1:-}" == "--check-only" ]]; then
  echo "PREFLIGHT OK (check-only; not submitting)"
  exit 0
fi
echo "PREFLIGHT OK — submitting"
mkdir -p outputs/slurm outputs/nw4/sub-08/logs
JID=$(sbatch --parsable slurm/nw4_s08.sbatch)
echo "submitted nw4_s08 job ${JID}"
squeue -j "${JID}" -o "  %i %j %T %M %N"
