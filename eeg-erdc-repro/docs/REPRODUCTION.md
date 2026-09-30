# 复现指南

## 0. 前提

- GPU 节点（建议 H800/A100，≥96GB RAM）
- 已有 `/project/peilab/why/eeg-brainit` 主项目与环境
- 已下载外部资产（见 `EXTERNAL_ASSETS.md`）

本包 `eeg-erdc-repro/` 可单独拷到本地阅读；**实际跑实验需回到主项目**。

---

## 1. 环境

```bash
cd /project/peilab/why/eeg-brainit
source scripts/activate.sh
export XFORMERS_DISABLED=1
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=$HF_HOME/hub
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/eeg-brainit/open_clip
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
```

---

## 2. 下载官方解码资产

```bash
python scripts/download_atm_official_assets.py
```

需要：
- `stabilityai/sdxl-turbo`
- IP-Adapter SDXL
- `train_image_latent_512.pt` → `checkpoints/_hf_atm_ds/`

---

## 3. 编码器 embed（sub-08）

```bash
python scripts/eval_atm_pipeline.py \
  --subject sub-08 \
  --prior-ckpt checkpoints/atm_diffusion_prior/sub-08/diffusion_prior.pt \
  --s3-ckpt outputs/atm_distill_s3_sub08/checkpoints/atm_stage3_best.pt \
  --output-dir outputs/eval/atm_pipeline_sub08 \
  --skip-generate \
  --gen-sources bit_clip
```

输出：`outputs/eval/atm_pipeline_sub08/sub-08_bit_clip_1024.npy`

---

## 4. W12 官方 Turbo 生成 + fuse

```bash
# 生成 bit bank
python scripts/erdc_official_atm_pipeline.py \
  --subject sub-08 \
  --embed-source bit_clip \
  --bit-npy outputs/eval/atm_pipeline_sub08/sub-08_bit_clip_1024.npy \
  --low-level-mode neighbor_image \
  --top-m 2 \
  --strengths 0.35,0.50,0.65,0.85 \
  --use-turbo --gen-steps 4 --gen-guidance 0.0 \
  --output-dir outputs/erdc/w12_official_bit_img_sub08

# merge 3 banks
python scripts/erdc_merge_banks.py \
  --bank-dirs outputs/erdc/w12_official_bit_img_sub08 \
              outputs/erdc/w12_official_atm_img_sub08 \
              outputs/erdc/w12_official_prior_latent_sub08 \
  --output-dir outputs/erdc/w13_merged_turbo/candidates

# fuse
python scripts/erdc_fuse_reselect.py \
  --cand-dir outputs/erdc/w13_merged_turbo/candidates \
  --eeg-npy outputs/eval/atm_pipeline_sub08/sub-08_bit_clip_1024.npy \
  --lambda-struct 0.15 \
  --output-dir outputs/erdc/w13_merged_fuse_l0p15
```

---

## 5. 指标

```bash
python scripts/erdc_full_metrics.py \
  --gen-dir outputs/erdc/w13_merged_fuse_l0p15/selected_fused \
  --output-json outputs/erdc/w13_metrics/w13_merged_fuse_l0p15.json \
  --tag w13_merged_fuse_l0p15

python scripts/erdc_twoway_metrics.py \
  --gen-dir outputs/erdc/w13_merged_fuse_l0p15/selected_fused \
  --output-json outputs/erdc/w14_metrics/w13_merged_fuse_l0p15_2wc.json \
  --tag w13_merged_fuse_l0p15
```

GT 图像：`/project/peilab/why/data/images_set/test_images/`

---

## 6. Slurm 一键跑

```bash
sbatch slurm/erdc_w12_official_continue.sbatch   # W12 续跑
sbatch slurm/erdc_w13_merge_fuse.sbatch          # W13 merge
sbatch slurm/erdc_w14_paper_finalize.sbatch       # W14 收尾
```

脚本副本在本包 `code/slurm/`。

---

## 7. 预期结果（sub-08）

| 步骤 | 预期 Pix | 预期 CLIP |
|------|----------|-----------|
| w12 bit brain | ~0.144 | ~0.403 |
| w13 merged fuse λ=0.15 | ~0.159 | ~0.418 |
| w12 atm brain | ~0.173 | ~0.379 |

若 Pix 仍在 0.13 左右 → 检查是否误用 SDXL-base 而非 Turbo。

---

## 8. 同步本复现包脚本到主项目（可选）

```bash
cp eeg-erdc-repro/code/scripts/erdc_*.py eeg-brainit/scripts/
cp eeg-erdc-repro/code/slurm/erdc_w14_paper_finalize.sbatch eeg-brainit/slurm/
```
