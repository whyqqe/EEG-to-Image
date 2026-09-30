# 对话与决策记录（Agent 协作摘要）

本文档整理与 Cursor Agent 在本项目中的主要讨论、决策与结论，供后续 Agent 或合作者接续。

---

## 一、项目背景与用户目标

- **项目路径**：`/project/peilab/why/eeg-brainit`
- **数据集**：THINGS-EEG2
- **任务**：EEG → Image 生成/重建
- **方法品牌**：**ERDC**（Evidence-Routed Diffusion Control）
- **顶会目标**：NeurIPS / ICML / ICLR 主会
- **硬指标**：PixCorr ≥ 0.15，同时 CLIP 优于官方 ATM（~0.37）

用户要求全程使用**简体中文**交流。

---

## 二、关键对话节点

### 1. W12 完成后的结果分析

用户：「好像结束了，请分析结果。」

**结论**：
- 换 **官方 Turbo + 低层 img2img** 后 Pix 从 ~0.133 跳到 0.15–0.17
- **atm + brain** 双超官方（Pix 0.173, CLIP 0.379）
- **bit + fuse λ=0.25** 更均衡（Pix 0.150, CLIP 0.417）
- brain 选优 **不总赢 Pix**（random 往往 Pix 更高）→ 论文需如实写 trade-off

### 2. 能否 CLIP 和 Pix 都优于官方？

用户：「能否在保持最优 CLIP 的同时让 Pix 达到官方水平？」

**结论**：
- **双超官方**：atm + Turbo 已达成
- **bit_clip 主线**：fuse 可 Pix≈0.15、CLIP≈0.42；oracle 上界 0.164/0.456 说明候选库够好，瓶颈在选优
- W10 CLIP 0.45 与 Pix 0.13 不可兼得（SDXL-base 天花板）
- 最高 ROI：**merge 3 bank + fuse 细扫**（零重生成）

### 3. 执行 W13 实验

用户：「请执行你认为最可能成功的实验。」

**执行**：Job 527543 — merge bit+atm+prior_latent + fuse λ grid + atm fuse + top2struct

**结果**：
- merged fuse λ=0.15：**Pix 0.159, CLIP 0.418**（主方法候选）
- merged brain：**Pix 0.162, CLIP 0.406**
- atm fuse λ=0.15：**Pix 0.170, CLIP 0.386**

### 4. 顶会水准评估

用户：「现在的成果是否已经达到顶会水准？」

**结论**：
- **指标**：sub-08 已达可投稿线
- **完备性**：缺多被试 LOSO、人类 2AFC、强机制对照
- **估计**：borderline **55–65%** → W13 后 **65–75%**（仍非稳收）

### 5. 收尾流水线

用户：「请帮我提交完成收尾工作的流水线。」

**执行**：Job 527571 — W14 `erdc_w14_paper_finalize.sbatch`
- fuse shuffle/misalign/zero
- 2WC 全主线
- 10 被试 Turbo 生成
- freeze 汇总

### 6. 保存进度 / 复现包

用户（修订）：「先检查 W14 结果；在 why 下建复制项目，关键代码+文档+少量 ckpt，可复现即可。」

**执行**：创建 `/project/peilab/why/eeg-erdc-repro/`（本目录）

---

## 三、Agent 重要技术决策

| 决策 | 理由 |
|------|------|
| 放弃 W11 SDXL-only Pix 扫参 | 天花板 ~0.133，ROI 低 |
| 放弃 S4 继续训 encoder | Top-1 不变，Pix 无增益 |
| W12 采用官方 `custom_pipeline_low_level` | 与 ATM 论文一致，Pix 质变 |
| 主方法用 **merge fuse** 而非纯 brain | Pix–CLIP 更均衡 |
| W14 10 被试用 **shared sub-08 S3** | 快速摸底；**不可作 main multi-subject claim** |
| shuffle 机制弱 | 论文弱化「结构证据」表述，强调 decoder-agnostic ERDC |

---

## 四、已知 Bug 与修复

1. **latent dict 加载**：`train_image_latent_512.pt` 需取 `["image_latent"]`
2. **fuse neighbor_idx**：路径可能在 bank 父目录
3. **Slurm**：排除 `dgx-10, dgx-11`；`XFORMERS_DISABLED=1`

---

## 五、给下一个 Agent 的接续清单

1. **若要强化主会**：LOSO 训练 S3 或 per-subject head → 重跑 W14-D
2. **机制**：尝试 top-K brain → max struct、λ 自适应、strength 惩罚（见 `erdc_fuse_reselect.py` 扩展）
3. **论文**：定稿 Table 1 用 W13 merged fuse；Figure 用 `w14_panels_main`
4. **勿做**：SDXL-base 大规模扫参、无 GT 的 oracle 当主结果

---

## 六、相关路径速查

```
主项目：     /project/peilab/why/eeg-brainit
HF cache：   /project/peilab/why/cache/eeg-brainit
数据：       /project/peilab/why/data/images_set
S3 ckpt：    eeg-brainit/outputs/atm_distill_s3_sub08/checkpoints/atm_stage3_best.pt
官方代码：   eeg-brainit/third_party/EEG_Image_decode/Generation/
W14 freeze： eeg-brainit/outputs/erdc/w14_paper_finalize_freeze.txt
本复现包：   /project/peilab/why/eeg-erdc-repro/
```

---

*文档生成时间：2026-08-23。对话来源：Cursor Agent 会话（EEG-ERDC 实验线程）。*
