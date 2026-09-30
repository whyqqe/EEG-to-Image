#!/usr/bin/env bash
# Finetune Fusion Prior on THINGS from pretrained H14+B32+VAE checkpoint.
set -euo pipefail

BRAIN_HIVE="${BRAIN_HIVE:-/project/peilab/why/Brain-HIVE}"
FUSION_PRIOR="${FUSION_PRIOR:-/project/peilab/why/cache/fusion_prior/H14_B32_VAE}"
IMAGE_DIR="${IMAGE_DIR:-/project/peilab/why/data/images_set}"
OUT_EMB="${OUT_EMB:-/project/peilab/why/NeuroBridge/outputs/nb_nmb_sota/sub-08/things_embeddings}"
OUT_PRIOR="${OUT_PRIOR:-/project/peilab/why/NeuroBridge/outputs/nb_nmb_sota/sub-08/fusion_prior_finetuned}"
CACHE_HUB="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
CACHE_ROOT="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"

FT_EPOCHS="${FT_EPOCHS:-3}"
FT_MAX_STEPS="${FT_MAX_STEPS:-2500}"
FT_BATCH="${FT_BATCH:-8}"
FT_LR="${FT_LR:-5e-5}"

cd "${BRAIN_HIVE}"
export PYTHONPATH="${BRAIN_HIVE}:${PYTHONPATH:-}"
mkdir -p "${OUT_PRIOR}"

echo "[preflight] import train_prior..."
python3 -c "import train_prior; print('[OK] train_prior import')"

resolve_sdxl_train() {
  python3 - <<'PY'
import os
from pathlib import Path

def find_repo(repo: str) -> str | None:
    roots = []
    for key in ("HF_HUB_CACHE", "HF_HOME"):
        v = os.environ.get(key)
        if v:
            roots.append(Path(v))
    roots.extend(
        [
            Path("/project/peilab/why/cache/eeg-brainit/hf"),
            Path("/project/peilab/why/cache/eeg-brainit/hf/hub"),
        ]
    )
    seen = set()
    for root in roots:
        key = str(root.resolve())
        if key in seen:
            continue
        seen.add(key)
        snap_root = root / f"models--stabilityai--{repo}" / "snapshots"
        if snap_root.is_dir():
            for snap in sorted(snap_root.iterdir(), reverse=True):
                if (snap / "model_index.json").is_file():
                    return str(snap)
    return None

for repo in ("stable-diffusion-xl-base-1.0", "sdxl-turbo"):
    hit = find_repo(repo)
    if hit:
        print(hit)
        raise SystemExit
print("stabilityai/stable-diffusion-xl-base-1.0")
PY
}

resolve_sdxl_eval() {
  python3 - <<'PY'
import os
from pathlib import Path

roots = []
for key in ("HF_HUB_CACHE", "HF_HOME"):
    v = os.environ.get(key)
    if v:
        roots.append(Path(v))
roots.extend(
    [
        Path("/project/peilab/why/cache/eeg-brainit/hf"),
        Path("/project/peilab/why/cache/eeg-brainit/hf/hub"),
    ]
)
for root in roots:
    snap_root = root / "models--stabilityai--sdxl-turbo" / "snapshots"
    if snap_root.is_dir():
        for snap in sorted(snap_root.iterdir(), reverse=True):
            if (snap / "model_index.json").is_file():
                print(snap)
                raise SystemExit
print("stabilityai/sdxl-turbo")
PY
}

resolve_ip_adapter() {
  python3 - <<'PY'
import os
from pathlib import Path
roots = []
for key in ("HF_HUB_CACHE", "HF_HOME"):
    v = os.environ.get(key)
    if v:
        roots.append(Path(v))
roots.append(Path("/project/peilab/why/cache/eeg-brainit/hf/hub"))
for root in roots:
    snap_root = root / "models--h94--IP-Adapter" / "snapshots"
    if snap_root.is_dir():
        for snap in sorted(snap_root.iterdir(), reverse=True):
            for name in ("ip-adapter_sdxl_vit-h.safetensors", "ip-adapter_sdxl_vit-h.bin"):
                p = snap / "sdxl_models" / name
                if p.is_file():
                    print(p)
                    raise SystemExit
print("")
PY
}

SDXL_TRAIN="$(resolve_sdxl_train)"
SDXL_EVAL="$(resolve_sdxl_eval)"
IP_ADAPTER="$(resolve_ip_adapter)"

if [[ -z "${IP_ADAPTER}" ]]; then
  echo "[WARN] IP-Adapter weight not found; using init_fusion_prior_path only"
  IP_ADAPTER="${FUSION_PRIOR}/proj/config.json"
fi

TMP_CFG="${OUT_PRIOR}/train_prior.resolved.yaml"
PROJ_META="{vae: 1024, CLIP-ViT-B-32-laion2B-s34B-b79K: 512, CLIP-ViT-H-14-laion2B-s32B-b79K: 1024}"

python3 build_config.py \
  --config_file configs/train_prior.yaml \
  --output_file "${TMP_CFG}" \
  "output_dir=${OUT_PRIOR}" \
  "run_name=nmb-things-ft" \
  "proj_meta=${PROJ_META}" \
  "dataset_name=things" \
  "image_directory=${IMAGE_DIR}" \
  "embedding_directory=${OUT_EMB}" \
  "eval_dataset_name=things" \
  "eval_image_directory=${IMAGE_DIR}" \
  "eval_embedding_directory=${OUT_EMB}" \
  "eval_split=test" \
  "diffusion_model_name_or_path=${SDXL_TRAIN}" \
  "eval_diffusion_model_name_or_path=${SDXL_EVAL}" \
  "ip_adapter_name_or_path=${IP_ADAPTER}" \
  "init_fusion_prior_path=${FUSION_PRIOR}" \
  "num_train_epochs=${FT_EPOCHS}" \
  "max_steps=${FT_MAX_STEPS}" \
  "per_device_train_batch_size=${FT_BATCH}" \
  "per_device_eval_batch_size=4" \
  "learning_rate=${FT_LR}" \
  "checkpointing_steps=1000" \
  "validation_steps=500" \
  "save_total_limit=2" \
  "report_to=tensorboard" \
  "do_eval=false"

echo "[finetune] SDXL train=${SDXL_TRAIN} eval=${SDXL_EVAL}"
echo "[finetune] init from ${FUSION_PRIOR} -> ${OUT_PRIOR}"

accelerate launch --config_file configs/gpu_cfg_1gpu.yaml train_prior.py --config "${TMP_CFG}"
rm -f "${TMP_CFG}"

test -f "${OUT_PRIOR}/fusion_encoder/config.json" || {
  # fallback: copy from latest accelerate checkpoint if final save raced with NFS
  latest_ckpt="$(ls -d "${OUT_PRIOR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)"
  if [[ -n "${latest_ckpt}" && -f "${latest_ckpt}/fusion_encoder/config.json" ]]; then
    echo "[WARN] using checkpoint fusion_encoder from ${latest_ckpt}"
    mkdir -p "${OUT_PRIOR}/fusion_encoder" "${OUT_PRIOR}/proj" "${OUT_PRIOR}/attn_adapter"
    cp -a "${latest_ckpt}/fusion_encoder/." "${OUT_PRIOR}/fusion_encoder/"
    cp -a "${latest_ckpt}/proj/." "${OUT_PRIOR}/proj/"
    cp -a "${latest_ckpt}/attn_adapter/." "${OUT_PRIOR}/attn_adapter/"
  fi
}
test -f "${OUT_PRIOR}/fusion_encoder/config.json" || {
  echo "[ERROR] finetune did not produce fusion_encoder in ${OUT_PRIOR}" >&2
  exit 1
}
echo "[OK] Fusion Prior finetuned -> ${OUT_PRIOR}"
