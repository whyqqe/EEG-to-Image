# 外部资产清单（不包含在本复现包内）

## 1. 数据集

| 资产 | 路径 | 大小级 | 获取 |
|------|------|--------|------|
| THINGS-EEG2 图像 | `/project/peilab/why/data/images_set/` | ~GB | 官方 THINGS-EEG2 |
| THINGS-EEG2 EEG | 主项目 data 配置 | ~GB | 同上 |

## 2. HuggingFace 模型（cache）

| 模型 | 用途 | 脚本 |
|------|------|------|
| `stabilityai/sdxl-turbo` | W12 官方生成 | `download_atm_official_assets.py` |
| `stabilityai/stable-diffusion-xl-base-1.0` | W10 SDXL 路线 | `download_pretrained.sbatch` |
| IP-Adapter SDXL | 条件生成 | 同上 |
| OpenCLIP ViT-H-14 | 指标 / embed | 自动下载到 OPENCLIP_CACHE_DIR |

**Cache 根目录**：`/project/peilab/why/cache/eeg-brainit/`

## 3. ATM 官方资产

| 文件 | 路径 |
|------|------|
| train VAE latents | `checkpoints/_hf_atm_ds/train_image_latent_512.pt` |
| 官方 Generation 代码 | `third_party/EEG_Image_decode/`（git clone） |

```bash
cd eeg-brainit
git clone <EEG_Image_decode repo> third_party/EEG_Image_decode
python scripts/download_atm_official_assets.py
```

## 4. 训练 ckpt（主项目内，未打入复现包）

见 `artifacts/LARGE_CHECKPOINT_PATHS.txt`

## 5. 生成产物（可重跑，勿同步）

- `outputs/erdc/*/candidates/*.png` — 每张 bank 数百 MB
- `outputs/erdc/w14_turbo_*` — 10 被试全量生成

仅指标 JSON 已保存在 `results/metrics/`。
