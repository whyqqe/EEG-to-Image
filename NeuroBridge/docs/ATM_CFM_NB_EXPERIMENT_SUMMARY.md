# ATM + CFM + NeuroBridge 实验总结

> **更新**：2026-09-02  
> **范围**：THINGS-EEG sub-08 主实验 + eeg-thor-a 十被试 CFM/OT 流水线  
> **评估协议**：生成侧采用 eeg-brainit `erdc_full_metrics`（与 ATM 官方复现一致）

---

## 1. 执行摘要

| 赛道 | 最佳方法 | 主指标 | vs ATM 官方 |
|------|----------|--------|-------------|
| **生成 CLIP** | NB ERDC (`nb_erdc_brain`) | **0.422** | +0.052 |
| **生成 CLIP** | NB DecodeAligner | 0.418 | +0.048 |
| **生成 CLIP** | NB v1 linEns (mem⊕bridge) | 0.412 | +0.041 |
| **生成 CLIP** | Official ATM | 0.370 | 基线 |
| **检索 Top-1** | NB RN50 intra (512-d proj) | **57%** | — |
| **检索 Top-1** | NB ViT-H intra (512-d proj) | 57% | — |
| **检索 Top-1** | NB-ATM ATMS (1024-d direct) | 20.5% | — |
| **检索 Top-1** | ENIGMA sub-08 | 37.5% | — |
| **检索 Top-1** | ATM raw EEG 1024 | ~34.5% | — |
| **跨被试 OT** | OT-Sinkhorn ensemble (10subj) | 20.5% | — |
| **跨被试 CFM** | ENIGMA + CFM (10subj 均值) | 24.4% | — |

**核心结论**：

1. **NB 在生成上已稳定超越 ATM 官方**（CLIP +0.04~0.05），但主要靠后验工程（memory RAG + blend + DecodeAligner），而非端到端训练。
2. **NB 检索强、生成弱的「解码间隙」**在 RN50→ViT-H adapter 映射时尤为明显（57% → 6.5% Top-1）；直接在 1024-d 训 ATMS 可把 gallery Top-1 提到 20.5%，仍远低于 RN50 空间。
3. **CFM 对被试内检索有稳定小幅增益**（+1.5% Top-1 均值），但在 OT 对齐后的 Z 上训 CFM **打不过**原生 Path A+CFM。
4. **NB-ATM 端到端流水线（Job 543326）** Stage A/B 已完成，Stage C DecodeAligner 因空 `img_projector_state_dict` 崩溃；已修复，可重跑。

---

## 2. ATM 官方基线

### 2.1 生成（Official ATM，200 test）

来源：`eeg-brainit/outputs/erdc/w7_official_flat` + `outputs/atm_official_eval_sub08/full_report.json`

| 指标 | Official ATM | 论文 Table3 参考 |
|------|-------------|-----------------|
| PixCorr | 0.159 | — |
| SSIM (simple) | 0.038 | 0.345* |
| **CLIP paired cos** | **0.370** | — |
| FID↓ | 182.2 | — |
| 2WC CLIP | 78.5% | 78.6% |
| 2WC Alex2 | 79.1% | 77.6% |
| 2WC Inception | 67.2% | 73.4% |

\* SSIM 实现与论文可能不同（本仓库用 `ssim_simple` 256×256）。

### 2.2 检索（sub-08，ViT-H 1024-d gallery）

| 来源 | Top-1 | Top-5 | paired cos |
|------|-------|-------|------------|
| ATM raw EEG 1024 | ~34.5% | — | — |
| ATM + Diffusion Prior | ~14.5% | — | 0.338 |

ATM 的 Diffusion Prior 提升 paired cos 但**损害** gallery Top-1，与 NB 侧观察一致。

---

## 3. CFM 实验（eeg-thor-a）

来源：`eeg-thor-a/docs/week_experiments_10subj_results.md`

### 3.1 被试内 ENIGMA + CFM（10 被试）

| subject | ENIGMA raw | + CFM | Δ |
|---------|-----------|-------|---|
| sub-01 | 0.210 | 0.245 | +0.035 |
| sub-02 | 0.250 | 0.250 | 0.000 |
| sub-03 | 0.090 | 0.105 | +0.015 |
| sub-04 | 0.285 | 0.290 | +0.005 |
| sub-05 | 0.140 | 0.155 | +0.015 |
| sub-06 | 0.205 | 0.205 | 0.000 |
| sub-07 | 0.215 | 0.245 | +0.030 |
| **sub-08** | **0.375** | **0.395** | **+0.020** |
| sub-09 | 0.180 | 0.190 | +0.010 |
| sub-10 | 0.335 | 0.355 | +0.020 |
| **均值** | **0.229** | **0.244** | **+0.015** |

**结论**：CFM 在强基线上仍有稳定小幅增益；应在**原生 Z 空间**做 Path B，而非 OT 对齐后。

### 3.2 跨被试 OT + CFM

| 方法 | Top-1 (10subj) |
|------|----------------|
| LOSO encoder transfer | 0.025 |
| OT-Sinkhorn ensemble | **0.205** |
| Paired Procrustes ensemble | 0.203 |
| OT 对齐后 + CFM | 0.215 |
| 原生 ENIGMA + CFM | **0.244** |

**关键负结果**：OT 对齐后的 CFM（0.215）< 被试内 CFM（0.244）；跨被试 OT→源 CFM ensemble（0.208）≈ OT-only（0.205）。

### 3.3 sub-08 CFM on OT-aligned Z

来源：`eeg-thor-a/outputs/week4_cfm_ot_10subj/cfm_ot_aligned_sub-08/summary.json`

| 阶段 | Top-1 | Top-5 | cos |
|------|-------|-------|-----|
| OT aligned raw | 28.5% | 59.0% | 0.426 |
| OT aligned + CFM | 29.0% | 59.5% | 0.430 |

增益极小（+0.5% Top-1），与十被试汇总一致。

---

## 4. NeuroBridge 检索实验

### 4.1 RN50 空间（NeuroBridge 原文设定）

| 方法 | Top-1 | Top-5 | 备注 |
|------|-------|-------|------|
| NB intra sub-08 (RN50 proj 512-d) | **57%** | 85% | `checkpoint_test_best.pth` |
| NB 原文 SOTA | 63.2% | — | 全被试/设定略有差异 |

### 4.2 ViT-H 空间映射（Adapter）

来源：`outputs/nb_adapter/sub-08-vit-h/summary.json`

| 路径 | Gallery Top-1 | paired cos (GT) |
|------|-------------|-----------------|
| NB ViT-H intra (512-d proj) | 57% | — |
| Linear Adapter → 1024 | **3.5%** | 0.650 |
| MLP Adapter → 1024 | **6.5%** | 0.659 |
| Linear raw EEG → 1024 | 4.0% | 0.650 |
| MLP raw EEG → 1024 | 6.0% | 0.657 |
| ATM Prior on NB cond | — | 0.346 (gen 50) |

**解码间隙**：高 GT paired cos（0.65+）≠ 高 gallery Top-1（<7%）。Adapter 学到的是「与 GT 对齐」而非「与 gallery 可区分」。

### 4.3 NB-ATM ATMS 直接 1024-d（Job 543326 Stage A）

来源：`outputs/nb_atm_vith/sub-08/`

| 阶段 | Top-1 | Top-5 | paired cos |
|------|-------|-------|------------|
| 训练 val（50 ep） | 30.0% | 57.0% | — |
| Gallery raw embed | **20.5%** | 49.0% | 0.209 |
| + Diffusion Prior | 12.0% | 36.5% | 0.376 |

直接 1024-d 训练比 Adapter 映射好很多（20.5% vs 6.5%），但仍远低于 RN50 空间 57%。Prior 再次损害 Top-1 但提升 cos。

---

## 5. NeuroBridge 生成实验（ATM 官方指标）

来源：`outputs/atm_official_eval_sub08/summary_table.md`

### 5.1 论文主表（sub-08，200 test）

| Method | PixCorr | CLIP | Δ CLIP | FID↓ | 2WC CLIP |
|--------|---------|------|--------|------|----------|
| Official ATM | **0.159** | 0.370 | +0.000 | 182.2 | **78.5%** |
| **nb_erdc_brain** | 0.152 | **0.422** | +0.051 | 177.5 | 69.2% |
| **nb_decode_aligner** | 0.157 | 0.418 | +0.048 | **169.3** | 73.0% |
| nb_v1_linEns | 0.148 | 0.412 | +0.041 | 176.9 | 68.3% |
| nb_r2fosa_align | 0.146 | 0.412 | +0.042 | 176.7 | 67.7% |

### 5.2 各方法配置要点

| 方法 | 核心配置 | CLIP |
|------|----------|------|
| **ERDC** | mem RAG + pair fuse rerank + brain selection | **0.422** |
| **DecodeAligner** | probe 监督 + mem blend, s=0.4 | 0.418 |
| **v1 linEns** | `blend(mem, linear_bridge(ens), α=0.5)` + img2img s=0.4 | 0.412 |
| **R²-FOSA align** | DDLG warm-start + align head | 0.412 |
| NB rag_soft5_lowlevel | 纯 NB 基线 memory img2img | 0.389 |
| Adapter MLP direct | 无 memory，直接 IP-Adapter | ~0.40 (50) |
| Teacher 上界 | GT ViT-H embed 直接解码 | **0.635** |

### 5.3 平台效应

7+ 种后验方法收敛到 CLIP ≈ **0.412 ± 0.002**；仅 DecodeAligner（0.418）与 ERDC（0.422）小幅突破。在冻结 NB encoder 设定下，sub-08 后验工程接近局部最优。

### 5.4 嵌入 vs 生成间隙

| 嵌入来源 | paired cos | 生成 CLIP | 间隙 |
|----------|------------|-----------|------|
| CFT → Fusion | 0.886 | 0.36 | 巨大 |
| DecodeAligner direct | 0.424 | 0.418 | 小 |
| mem_vith | 0.53~0.59 | 0.39~0.42 | 中等 |
| DiffPrior / DDIM | 0.04~0.38 | 0.34~0.40 | 不稳定 |

---

## 6. NB-ATM 端到端流水线（Job 543326）

**Job**：543326 `nb-atm-vith` | **状态**：FAILED（8min）| **输出**：`outputs/nb_atm_vith/sub-08/`

### 6.1 流水线设计

```
Stage A   ATMS encoder + NB 训练策略 → 1024-d direct CLIP
Stage A-e 提取 embed + gallery retrieval
Stage B   ATM Diffusion Prior（warm-start）
Stage B2  encoder ⊕ prior blend (α=0.5)
Stage C   DecodeAligner（probe 监督）
Stage D/E 生成 + ATM 官方指标
```

### 6.2 已完成阶段结果

| Stage | Top-1 | Top-5 | paired cos | 备注 |
|-------|-------|-------|------------|------|
| A raw | 20.5% | 49.0% | 0.209 | 50 ep ATMS |
| B prior | 12.0% | 36.5% | 0.376 | val cos 0.265 |
| B2 blend α=0.5 | — | — | 0.447 (RAG) | blend cos |

### 6.3 失败原因与修复

**Stage C** 在加载 checkpoint 时崩溃：

```
RuntimeError: Missing key(s): linear.weight, linear.bias
```

根因：NB-ATM checkpoint 中 `img_projector_state_dict` 为空 `OrderedDict()`（direct projector 无独立 img projector），但脚本仍尝试 `load_state_dict`。

**已修复**：`nmb_decode_aligner_train.py` 仅在 state dict 非空时加载 img projector。

### 6.4 初步分析

1. **直接 1024-d 训练有效但未闭合检索 gap**：20.5% Top-1 显著优于 Adapter（6.5%），说明「在目标空间端到端训 encoder」方向正确，但 50 epoch 不足以追上 RN50 57%。
2. **Prior 仍损害 Top-1**：12% vs 20.5%，与历史实验一致；blend 可部分挽回 cos（0.447）但未见生成结果。
3. **待重跑 Stage C~F**：修复后预计 DecodeAligner 可在 1024-d ATMS backbone 上验证「检索→生成」闭环。

重跑命令：

```bash
cd /project/peilab/why/NeuroBridge
bash scripts/nb_atm/submit_nb_atm_sub08.sh
# 或跳过 Stage A/B，从 Stage C 起：
# 编辑 run_nb_atm_pipeline_sub08.sh 设置 SKIP_STAGE_A=1 SKIP_STAGE_B=1
```

---

## 7. 横向对比：谁擅长什么

| 能力 | ATM | CFM/OT | NB (后验) | NB-ATM (进行中) |
|------|-----|--------|-----------|-----------------|
| 被试内检索 | 中 (~35%) | 强 (ENIGMA 37.5%, CFM 39.5%) | **最强 RN50 57%** | 中 (20.5% 1024-d) |
| 跨被试检索 | 弱 | **强 OT 20.5%** | 未系统做 | — |
| 生成 CLIP | 0.370 | 未报 | **0.422** | 待测 |
| 端到端训练 | 有 | 检索 only | 冻结 encoder + 后验 | 尝试中 |
| 解码间隙闭合 | — | — | 部分（DA/ERDC） | 目标 |

---

## 8. 成功路径归纳

### 8.1 检索侧「成功」

1. **NB RN50 intra 对比学习** → 57% Top-1（512-d proj）
2. **ENIGMA + CFM** → 39.5% Top-1 sub-08（1024-d CLIP）
3. **OT-Sinkhorn 跨被试** → 20.5% Top-1 ensemble（闭合 ~88% in-transfer gap）

### 8.2 生成侧「成功」

1. **ViT-H episodic memory RAG**（soft k=5）→ 感知锚定，cos ≈ 0.59
2. **mem ⊕ bridge linear blend**（α≈0.5）→ 语义-感知绑定
3. **低 strength img2img**（s=0.4）→ 保留邻居结构
4. **DecodeAligner probe 监督** → 显式优化解码友好性，CLIP 0.418
5. **ERDC rerank fuse** → 多候选重排，CLIP 0.422

### 8.3 反复失败

| 路径 | 原因 |
|------|------|
| Fusion Prior 主解码 | Fusion 空间与 ViT-H 评估脱节 |
| RN50→ViT-H Adapter 直接生成 | gallery Top-1 <7%，解码间隙 |
| Diffusion Prior 单独用 | Top-1 下降，cos 升但不可区分 |
| OT 对齐后训 CFM | 打不过原生 Path A+CFM |
| 纯后验 blend 平台 | 0.412 饱和，需训练侧突破 |

---

## 9. 下一步建议

| 优先级 | 行动 | 预期收益 |
|--------|------|----------|
| P0 | 重跑 NB-ATM Stage C~F（已修复 bug） | 验证 1024-d 端到端 + DA 生成 |
| P1 | NB-ATM 加长训练 / 数据增强对齐 NB RN50 | 检索 20%→30%+ |
| P1 | 10 被试 DecodeAligner + ERDC | 生成泛化 |
| P2 | 端到端 ViT-H 对比学习（非后验） | 闭合解码间隙根本路径 |
| P2 | NB encoder + CFM（原生 Z）| 检索进一步提升 |

---

## 10. 关键路径索引

| 内容 | 路径 |
|------|------|
| ATM 官方生成 eval | `eeg-brainit/outputs/erdc/w7_official_flat` |
| NB ATM 官方对比 | `NeuroBridge/outputs/atm_official_eval_sub08/` |
| NB DecodeAligner | `NeuroBridge/outputs/nb_decode_aligner/sub-08/` |
| NB ERDC | `NeuroBridge/outputs/nb_nmb_sota_v2/sub-08/erdc/` |
| NB v1 linEns | `NeuroBridge/outputs/nb_nmb_sota/sub-08/generation_overnight/` |
| NB Adapter | `NeuroBridge/outputs/nb_adapter/sub-08-vit-h/` |
| NB-ATM pipeline | `NeuroBridge/outputs/nb_atm_vith/sub-08/` |
| CFM 十被试 | `eeg-thor-a/docs/week_experiments_10subj_results.md` |
| CFM OT sub-08 | `eeg-thor-a/outputs/week4_cfm_ot_10subj/cfm_ot_aligned_sub-08/` |
| NMB 详细报告 | `NeuroBridge/docs/NMB_EXPERIMENT_SUMMARY.md` |
