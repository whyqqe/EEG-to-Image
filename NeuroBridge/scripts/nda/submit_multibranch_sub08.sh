#!/usr/bin/env bash
# Preflight + CPU-safe smoke + submit for the multi-branch IP pipeline (sub-08).
set -euo pipefail
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"
OUT="${MB_OUT:-${NB_ROOT}/outputs/mb_s08}"
mkdir -p "${OUT}"/logs outputs/slurm

echo "===== preflight: files ====="
fail=0
chk() { if [[ -e "$1" ]]; then echo "  [ok]   $1"; else echo "  [MISS] $1"; fail=1; fi; }
chk scripts/nda/multibranch_train_heads.py
chk scripts/nda/generate_multibranch_decode.py
chk scripts/nda/uck_nat_select.py
chk scripts/nda/run_multibranch_sub08.sh
chk slurm/mb_s08.sbatch
chk scripts/nda/eval_official_seven_dir.py
chk scripts/nda/gem_calib.py
chk scripts/nda/ocf_train.py
chk scripts/nda/leakfree.py
chk outputs/gem/cond_cache/clip_depth1024_train.npy
chk outputs/gem/cond_cache/clip_depth1024_test.npy
chk outputs/gem/cond_cache/clip_edge1024_train.npy
chk outputs/gem/cond_cache/clip_edge1024_test.npy
chk outputs/gem/cond_cache/clip_img1024_train.npy
chk outputs/uck/sub-08/full/conds/ip_mem_test.npy
chk outputs/uck/sub-08/full/spatial/pred_depth_rgb_512/199.png
chk outputs/uck_nat_s08/conds/anchor_idx_selfex.npy
chk outputs/uck_nat_s08/conds/ip_lam0.5_k7_test.npy
chk outputs/uck_nat_s08/conds/ip_rand_lam0.5_k3_test.npy
chk outputs/ocf/intra_z/sub-08/shared_r_train.npy
chk outputs/ocf/intra_z/sub-08/shared_r_test.npy
chk outputs/leakfree/split.json
chk outputs/g2f/prompts/prompts_deploy.json
chk outputs/nda_ss/sub-08/clip_text/train/concept_phrases.json
if [[ ! -f outputs/sdedit_ll_full10/sub-08/vae_head/pred_lowlevel_rgb_512/199.png ]]; then
  echo "  [warn] sdedit_ll_full10 LL missing; fallback = uck spatial pred_lowlevel_rgb_512"
  chk outputs/uck/sub-08/full/spatial/pred_lowlevel_rgb_512/199.png
fi
bash -n scripts/nda/run_multibranch_sub08.sh || fail=1
bash -n slurm/mb_s08.sbatch || fail=1
python3 -m py_compile scripts/nda/multibranch_train_heads.py || fail=1
python3 -m py_compile scripts/nda/generate_multibranch_decode.py || fail=1
python3 -m py_compile scripts/nda/uck_nat_select.py || fail=1
if (( fail )); then echo "[FATAL] preflight failed"; exit 1; fi

echo "===== CPU smoke: depth/edge heads (3 epochs, seconds) ====="
if [[ ! -f /tmp/mb_head_smoke/report.json ]]; then
  python3 scripts/nda/multibranch_train_heads.py \
    --out /tmp/mb_head_smoke --test-subject 8 --modalities depth,edge \
    --epochs 3 --device cpu 2>&1 | tail -n 5
fi
python3 - <<'PY'
import json, sys
r = json.load(open("/tmp/mb_head_smoke/report.json"))["modalities"]
bad = []
for m, d in r.items():
    g = d["val_cos_gain_over_shuffle"]
    print(f"[smoke] {m}: val_cos={d['val_cos_held']:.4f} "
          f"shuffled={d['val_cos_shuffled_control']:.4f} gain={g:+.4f} "
          f"ret_top1={d['val_retrieval_top1_within_concept']:.4f} "
          f"(chance {d['chance_retrieval']:.4f})")
    if g <= 0:
        bad.append(m)
if bad:
    print(f"[FATAL] heads carry no paired signal: {bad}"); sys.exit(2)
print("[smoke] depth/edge heads carry paired signal OK")
PY

echo "===== CPU smoke: branch plumbing (no GPU, no pipeline download) ====="
python3 - <<'PY'
# Check the *mechanism* the decode script relies on: N registered adapters must
# exist and each projection must map (B,1,1024) -> (B,4,2048). Pure torch.
import sys
from pathlib import Path
import torch
sys.path.insert(0, "/project/peilab/why/eeg-brainit/scripts")
from eval_atm_pipeline import resolve_ip_adapter_dir  # type: ignore
from diffusers.models.embeddings import ImageProjection
import numpy as np

hub = Path("/project/peilab/why/cache/eeg-brainit/hf/hub")
ip = resolve_ip_adapter_dir(hub)
sd = torch.load(ip / "sdxl_models" / "ip-adapter_sdxl_vit-h.bin",
                map_location="cpu", weights_only=True)["image_proj"]
n_embed = 4
cross = sd["proj.weight"].shape[0] // n_embed
proj = ImageProjection(image_embed_dim=sd["proj.weight"].shape[1],
                       cross_attention_dim=cross, num_image_text_embeds=n_embed)
proj.load_state_dict({k.replace("proj", "image_embeds"): v.float() for k, v in sd.items()})
proj.eval()
rng = np.random.default_rng(0)
e = rng.normal(size=(1, 1, 1024)).astype(np.float32)
e /= np.linalg.norm(e, axis=-1, keepdims=True)
with torch.no_grad():
    out = proj(torch.from_numpy(e))
print(f"[smoke] ImageProjection (1,1,1024) -> {tuple(out.shape)}  n_embed={n_embed} cross={cross}")
assert out.shape == (1, n_embed, cross), "unexpected projection output shape"
print("[smoke] branch plumbing OK (3 adapters => 3 such call sites)")
PY

echo "===== submit ====="
JOB=$(sbatch --parsable slurm/mb_s08.sbatch)
echo "submitted JOB=${JOB}"
echo "${JOB}" > "${OUT}/job_id.txt"
squeue -j "${JOB}" || true
