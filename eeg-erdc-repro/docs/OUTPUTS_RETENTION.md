# outputs 清理说明（2026-08-23）

> 脚本：`scripts/cleanup_outputs_erdc.sh`  
> 日志：`outputs/CLEANUP_LOG_20260823.txt`

## 保留内容（论文 / 复现必需）

### 训练 checkpoint（sub-08 主实验）

| 路径 | 说明 |
|------|------|
| `outputs/atm_distill_s1_sub08/checkpoints/` | S1 bridge |
| `outputs/atm_distill_s3_sub08/checkpoints/atm_stage3_best.pt` | S3 bit_clip（主用） |
| `checkpoints/atm_diffusion_prior/sub-{01..10}/` | diffusion prior（在项目根 checkpoints，非 outputs） |

### 嵌入与 gallery

| 路径 | 说明 |
|------|------|
| `outputs/atm_bridge/clip_img_train_1024.npy` | fuse 结构项 gallery |
| `outputs/eval/atm_pipeline_sub08/*.npy` | sub-08 bit/bridge/prior embed |
| `outputs/eval/atm_pipeline_all/sub-*_bit_clip_1024.npy` | 十被试 bit_clip（W16） |

### 最终重建图（200 张 PNG）

| 路径 | 说明 |
|------|------|
| `outputs/erdc/w7_official_flat/` | Official ATM 基线 |
| `outputs/erdc/w13_merged_fuse_l0p15/selected_fused/` | **主方法** fuse λ=0.15 |
| `outputs/erdc/w12_official_atm_img_sub08/selected_brain/` | 最高 PixCorr 行 |
| `outputs/erdc/w15_mega_fuse_l0p15/selected_fused/` | mega fuse |
| `outputs/erdc/w15_mega_fuse_l0p10/selected_brain/` | mega brain |

### 指标与论文表

- `outputs/erdc/paper_main_table.md` / `.tex`
- `outputs/erdc/paper_ten_subject_table.md` / `.tex`
- `outputs/erdc/w16_paper_freeze.md`
- `outputs/erdc/w15_metrics/`
- `outputs/erdc/w16_loso_metrics/`

### 对比图

- `outputs/erdc/w16_panels_gt_vs_recon/`

## 已删除（可重跑生成）

- 全部 `candidates/` 候选库（每张测试图 × K 候选，占绝大部分空间）
- W7–W16 中间实验目录（w10/w11/w14 per-subject turbo 全量输出等）
- `selected_random` / `selected_first` 等消融选图（指标已写入 JSON）
- `eval/*/generated/` 中间生成 PNG
- 非 ERDC 实验：`aria/`、`nod_*`、`nb_*`、`subject_align_loso/`、`slurm/` 日志等
- `atm_distill_s4_pix_sub08`、`atm_distill_*_sub-01`（非主表必需）

## 如需完整重跑 fuse

需重新执行 W12 三路 bank 生成 → merge → fuse（见 `eeg-erdc-repro/docs/REPRODUCTION.md`）。  
主结果 PNG 与指标 JSON 已保留，**不必重跑即可写论文表与 qualitative**。
