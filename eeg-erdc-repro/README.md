# EEG-ERDC Repro Archive

THINGS-EEG2 上 **EEG→Image** 实验的精简复现包，对应主项目 `/project/peilab/why/eeg-brainit`。

**方法名**：ERDC（Evidence-Routed Diffusion Control）— 检索结构 + 脑一致闭环 + 融合证据重选。

## 目录结构

```
eeg-erdc-repro/
├── README.md                 # 本文件
├── docs/                     # 架构、实验时间线、对话记录、复现指南
├── code/                     # 核心脚本、配置、Slurm、模型源码片段
├── results/                  # 指标 JSON、freeze 汇总、W14 日志
└── artifacts/                # 小体积关键 ckpt 与 embed（不含 SDXL/Turbo 权重）
    ├── checkpoints/
    └── embeds/
```

## 当前最佳结果（sub-08，官方 Turbo 栈）

| 方法 | PixCorr | CLIP | 2WC (CLIP) |
|------|---------|------|------------|
| 官方 ATM 预生成 | 0.159 | 0.371 | 78.5% |
| **W13 merged fuse λ=0.15**（主方法候选） | **0.159** | **0.418** | **84.4%** |
| W12 atm + brain | **0.173** | 0.379 | 81.6% |
| W10 SDXL fuse（CLIP 上限） | 0.133 | **0.449** | 85.2% |

详见 `results/w14_paper_finalize_freeze.txt` 与 `docs/RESULTS_W14.md`。

## 快速复现（在集群上）

1. 克隆/挂载主项目 `eeg-brainit`，或使用本包 `code/` 覆盖脚本。
2. 下载外部资产（见 `docs/EXTERNAL_ASSETS.md`）：THINGS-EEG2 数据、SDXL-Turbo、IP-Adapter、官方 latent 等。
3. 指向主项目中的 S3 ckpt：`outputs/atm_distill_s3_sub08/checkpoints/atm_stage3_best.pt`（约 1.7GB，未打入本包）。
4. 激活环境：`source /project/peilab/why/eeg-brainit/scripts/activate.sh`
5. 跑 W12 官方栈 + W13 merge fuse：
   ```bash
   cd /project/peilab/why/eeg-brainit
   sbatch slurm/erdc_w12_official_continue.sbatch
   sbatch slurm/erdc_w13_merge_fuse.sbatch
   sbatch slurm/erdc_w14_paper_finalize.sbatch
   ```

## 不包含（需从主项目/网络获取）

- SDXL / SDXL-Turbo / IP-Adapter 权重（HF cache）
- THINGS-EEG2 原始 EEG 与图像
- 全量生成图像（`outputs/erdc/*/candidates/*.png`）
- S3/S1 蒸馏 ckpt（1.7GB×2，路径见 `artifacts/LARGE_CHECKPOINT_PATHS.txt`）
- `brain_it` 50GB 权重

## 文档索引

| 文件 | 内容 |
|------|------|
| `docs/ARCHITECTURE.md` | ERDC 原理与流水线 |
| `docs/EXPERIMENT_TIMELINE.md` | W7–W14 实验历程 |
| `docs/RESULTS_W14.md` | W14 收尾结果解读 |
| `docs/科研成果报告.md` | **完整科研成果**（灵感、架构、实验、对照严谨性） |
| `docs/CONVERSATION_LOG.md` | 与 Agent 的主要决策与对话摘要 |
| `docs/REPRODUCTION.md` | 逐步复现说明 |
| `docs/EXTERNAL_ASSETS.md` | 外部依赖清单 |

## 体量

本复现包约 **140MB**（含 diffusion prior sub-08 + CLIP gallery + 指标 JSON）。

主项目 `eeg-brainit` + cache 体量 **>50GB**，请勿整目录同步到本地。
