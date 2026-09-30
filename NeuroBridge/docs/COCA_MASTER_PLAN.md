# COCA 完整方案：协议净化 · 结构解码 · 可信度条件装配

> 目标：把 sub-08「能打局部指标」推进到可投主会的整表叙事。  
> 原则：**冻结编码器与跨被试骨干；主攻评测协议 + 条件→解码器注入；路由替代跨流形 transport。**

---

## 0. 问题诊断（已证实）

| 问题 | 含义 | 证据 |
|------|------|------|
| Top-1「不干净」 | 测时用 CPA 增强图库 → 73%；干净 RN50 → 35% | 同 checkpoint 双协议 |
| SSIM「假低」 | `ssim_simple` 非论文 SSIM | skimage ≈0.19–0.21 vs proxy≈0.05 |
| SSIM「真低」 | 相对 CogCap≈0.35 仍差 | 仅有邻居 Canny，无 depth/低层支路 |
| 解码瓶颈 | turbo img2img 无空间条件 | 换 CN+IP 后 PixCorr 0.18、CLIP 0.48 |
| 有害路径 | 强制 CFM transport | 伤生成 CLIP |
| 覆盖不足 | 仅 sub-08 | 主会要多被试 |

当前最强单点配置（保留为语义基线）：  
`NDA-SS mem⊕decode` + `ControlNet cn=0.5` + `IP=1.0` + `CPA text`  
→ CLIP≈0.480，PixCorr≈0.182，SSIM(skimage)≈0.192。

---

## 1. 一句话总方案

**COCA（Credibility-Optimal Condition Assembly）**

> EEG 对齐到不可互通的多条件流形（检索概念 / 生成语义 / 像素结构）；  
> **禁止**学 transport 改骨干；  
> 用 **可信度门控** 把 text / IP / depth(edge) 按样本与扩散时间装配进 SDXL；  
> 主表用 **干净检索协议 + 论文级七件套**；跨被试骨干冻结后原样外推。

---

## 2. 架构（最终形态）

```
                    ┌─ M_ret: 干净/CPA 双协议检索 ──► text（仅高 u_txt）
EEG ─► NDA-SS ─────┼─ M_gen: mem⊕decode (冻结) ──► IP-Adapter
(跨被试可冻结)      └─ M_str: EEG→depth/edge 或邻居depth ──► ControlNet(s)

                         ▼
              COCA Router R(u_txt, u_str, t)
              α_txt(t), α_ip(t), α_cn(t), α_ll
                         ▼
         SDXL: IP + Depth-CN [+ Edge-CN] + gated text
         可选: 低层模糊/VAE 与输出加权 (MindEye-style)
```

### 可信度定义

- \(u_{txt}\)：概念检索 margin（Top1−Top2）；主表用**干净图库**算；生成可用 CPA 但附录声明  
- \(u_{str}\)：EEG 语义 ↔ 结构条件（邻居/预测 depth 的 CLIP 或深度特征）对齐度  
- 低可信 → \(\alpha \to 0\)（dustbin / abstain），避免错概念、错布局污染  

### 分时先验（可先规则，后可学习）

| 扩散阶段 | 结构 CN | IP | Text |
|----------|---------|-----|------|
| 早 | 高（若 \(u_{str}\) 高） | 中 | 低/关 |
| 中 | 中 | 高 | 按 \(u_{txt}\) |
| 晚 | 低 | 稳 | 高（仅高 \(u_{txt}\)） |

### 明确不做

- 默认路径再引入 RGT-CFM 改 `mem⊕decode`  
- 主表只报 CPA Top-1  
- 继续用 `ssim_simple` 当论文 SSIM  

---

## 3. 分期执行（完整路线）

### Phase A — 协议与尺子（1 周内，必须先做）

**目的：** 去掉假阴性/假阳性，和对标对齐。

1. **评测脚本统一**（已有 `eval_paper_metrics.py` 为起点）  
   - PixCorr、**skimage SSIM**、AlexNet(2/5)、Inception、CLIP **2-way**、SwAV、FID  
2. **检索双报表**  
   - 主表：干净 RN50/约定视觉骨干 200-way Top-1/5  
   - 附录：CPA gallery（写明 `image_test_aug`）  
3. **重评现有冠军** `cn0.5_ip1.0_cpa` 全七件套 → 定基线  

**完成标准：** 一篇内部「基线卡」：干净 Top-1、七件套、与 CogCap 同口径备注。

---

### Phase B — 解码器结构升级（主攻，2–3 周）

**目的：** paper SSIM → 0.28–0.35 区间，CLIP 不明显低于 0.48。

| 步 | 内容 | 成功率 |
|----|------|--------|
| B1 | 训练集/测试邻居 **DepthAnything** 深度图；**Depth-ControlNet** 替换/并联 Canny | 高 |
| B2 | 固定网格扫 `cn_depth ∈ {0.4,0.6,0.8}` × `ip ∈ {0.85,1.0}` × ±CPA | 高 |
| B3 | 轻量 **EEG→depth**（回归 DepthAnything 伪标签或 depth-CLIP 对比） | 中高 |
| B4 | 推理用预测 depth；邻居 depth 作消融 | 中 |
| B5 | 可选 MindEye 式低层融合（仅高 \(u_{str}\)） | 中 |

**完成标准：** sub-08 上 skimage SSIM≥0.28 且 CLIP≥0.46；或 SSIM≥0.30。

---

### Phase C — COCA 路由（与 B 并行后半段）

**目的：** 结构↑时少掉语义；形成相对 CogCap「静态 multi-IP」的创新点。

1. 规则版 COCA（扩展现有 SCR）：\(u_{txt},u_{str}\) → \(\alpha\)  
2. 可学习轻量 router（可选）：验证集优化  
   \(\mathcal{L} = -\mathrm{CLIP2way} + \lambda_1(1-\mathrm{SSIM}) + \lambda_2\mathrm{KL}(R\|R_{prior})\)  
3. 消融：无路由 / 无 depth / 强制 CFM（负面） / 仅干净 prompt  

**完成标准：** 相对固定 cn=0.5，SSIM↑ 或同 SSIM 下 CLIP↑；消融表完整。

---

### Phase D — 干净检索补强（按需，不阻塞 B）

**目的：** 主表干净 Top-1 具备可比性。

1. 关闭测时 image aug，只报干净 gallery  
2. 可选：Shallow/中层对齐、训练混合干净+CPA 目标  
3. 生成 prompt 两条线对比：干净检索 vs CPA 检索  

**完成标准：** 干净 Top-1 显著高于当前 35%（争取 ≥50%），或诚实报现状并强调生成主贡献在解码。

---

### Phase E — 多被试外推（系统文必备）

**目的：** 主会表。

1. **冻结** Phase B/C 冠军配方（含跨被试 NDA-SS 权重策略）  
2. 在 sub-01…10（或至少 5 个）跑同一解码+评测  
3. 报 mean±std；跨被试模块保留，**不再为涨 sub-08 而改**  

**完成标准：** 多被试七件套 + 检索表。

---

## 4. 资源与模块冻结策略

| 模块 | 策略 |
|------|------|
| NDA-SS / 跨被试 shared+specific | **冻结**（E 阶段只推理） |
| NB RN50 检索 checkpoint | 冻结；协议分开报 |
| SDXL + IP + ControlNet | 主迭代区 |
| DepthAnything | 冻结教师，产伪标签 |
| EEG→depth / COCA router | **唯一主要新训模块** |
| RGT-CFM | 仅负面消融 |

---

## 5. 成功判据（能否「够顶会」）

**最低可投系统文门槛（建议）：**

- 评测：论文级 SSIM + CLIP 2-way + 干净 200-way  
- sub-08：PixCorr≥0.15，SSIM≥0.30，高层不低于 CogCap image-only 量级  
- 多被试：≥5 subjects 同配方  
- 贡献三条以内：  
  1. 多流形分裂 + 禁 transport 的实证  
  2. COCA 可信度条件装配  
  3. depth 结构解码在 THINGS-EEG 上的整表增益  

未达 SSIM/多被试前：定位为强 workshop / 扩展，不宣称全面 SOTA。

---

## 6. 近期立刻执行顺序（下一枪）

1. Phase A：七件套重评 `cn0.5_ip1.0_cpa` + 干净 Top-1 报表  
2. Phase B1–B2：Depth-ControlNet 替换 Canny，扫规模  
3. Phase C 规则 COCA 接到 depth  
4. 达标后再 E 多被试；并行 D 视审稿方向决定  

---

## 7. 文档与代码落点（已有/将有）

| 内容 | 路径 |
|------|------|
| 本方案 | `docs/COCA_MASTER_PLAN.md` |
| 论文级指标 | `scripts/nda/eval_paper_metrics.py` |
| CN+IP 解码 | `scripts/nda/generate_cn_ip_decode.py` |
| SCR 路由（COCA 雏形） | `scripts/nda/compute_scr_routing.py` |
| 当前冠军产物 | `outputs/cn_ip_decode/sub-08/…/cn0.5_ip1.0_cpa` |
| SCR 实验 | `outputs/scr_decode/sub-08/` |

---

**决策一句话：**  
协议洗干净、尺子换论文级；编码器与跨被试冻结；全力做 **Depth 结构条件 + COCA 可信度注入**；用多被试同一配方收束成顶会表。
