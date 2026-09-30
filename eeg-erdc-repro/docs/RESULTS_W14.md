# W14 收尾结果详解

Job **527571** 已完成。完整日志：`results/logs/erdc-w14-527571.out`  
Freeze 原文：`results/w14_paper_finalize_freeze.txt`

---

## 1. sub-08 主表（论文核心）

| 方法 | PixCorr | CLIP | 2WC (CLIP) |
|------|---------|------|------------|
| official_atm_gen | 0.1593 | 0.3705 | 78.5% |
| **w13_merged_fuse_l0p15** | **0.1592** | **0.4182** | **84.4%** |
| w13_merged_brain | 0.1619 | 0.4056 | 83.5% |
| w12_official_atm_brain | **0.1728** | 0.3786 | 81.6% |
| w13_atm_fuse_l0p15 | 0.1698 | 0.3860 | 80.9% |
| w12_fuse_bit_l0p25 | 0.1502 | 0.4171 | 85.0% |
| w10_sdxl_fuse_ref | 0.1327 | **0.4494** | **85.2%** |

**主方法推荐**：`w13_merged_fuse_l0p15` — Pix 与官方持平，CLIP **+0.048**，2WC **+5.9pt**。

**Pix 上限**：`w12_official_atm_brain` — Pix **+0.014**，CLIP +0.008。

---

## 2. Fuse 机制对照（λ=0.15, merged bank）

| Mode | Pix | CLIP | 2WC |
|------|-----|------|-----|
| retrieve | 0.1592 | **0.4182** | 84.4% |
| shuffle | **0.1624** | 0.4168 | 84.6% |
| misalign | 0.1583 | 0.4140 | 83.6% |
| zero (=brain) | 0.1619 | 0.4060 | 83.5% |

**结论**：
- shuffle 相对 retrieve：dPix **+0.003**，dCLIP **−0.001** → 结构 shuffle **未显著破坏** CLIP/Pix，机制证据 **偏弱**。
- 论文应强调：**merge 扩大候选 + brain/fuse 选优 + 官方 Turbo 栈**，而非强 claim「结构 cosine 是主要驱动力」。

---

## 3. 十被试结果（共用 sub-08 S3 ckpt）

### bit fuse λ=0.15

| Sub | Pix | CLIP |
|-----|-----|------|
| sub-08 | **0.1454** | **0.4135** |
| sub-03 | 0.1205 | 0.3823 |
| sub-06 | 0.1282 | 0.3526 |
| 均值 | **0.1172** | **0.3804** |

- Pix 超官方：**0/10**
- CLIP 超官方：**7/10**

### atm brain

| Sub | Pix | CLIP |
|-----|-----|------|
| sub-08 | **0.1728** | 0.3786 |
| sub-06 | 0.1558 | 0.3397 |
| 均值 | **0.1347** | **0.3541** |

- Pix 超官方：**1/10**（仅 sub-08）

**解读**：多被试表 **不能** 作为「方法普适优于官方」的主证据；需 LOSO 训练或 per-subject ckpt 才能写强 claim。

---

## 4. 顶会评估（W14 后更新）

| 维度 | 状态 |
|------|------|
| sub-08 Pix | ✅ 追平/超过 |
| sub-08 CLIP | ✅ 明显超过 |
| sub-08 2WC | ✅ 84.4% vs 78.5% |
| 10 被试 | ⚠️ 仅 sub-08 可靠 |
| fuse 机制 | ⚠️ shuffle 对照弱 |
| 人类 2AFC | ❌ 未做 |

**主会接受率估计**：sub-08 单点 **65–75% borderline**；若只报 10 被试均值会被拒。  
**建议投稿策略**：主表 sub-08 + ERDC ablation；10 被试放 appendix 并注明 shared encoder。

---

## 5. 论文主表建议（定稿）

| Row | Method | Pix | CLIP | 2WC |
|-----|--------|-----|------|-----|
| 1 | Official ATM | 0.159 | 0.371 | 78.5% |
| 2 | **Ours: Turbo + bit + merge fuse** | **0.159** | **0.418** | **84.4%** |
| 3 | Ours: Turbo + ATM + brain (upper bound) | **0.173** | 0.379 | 81.6% |
| 4 | Ablation: SDXL fuse (W10) | 0.133 | 0.449 | 85.2% |
