# NeuroWeave v3 — Working Brief（紧凑版，新对话起点）

> 详细版见 `NeuroWeave_v3.md`。本文件是**唯一必需上下文**，目标 <2k tokens。

## 0. 目标

sub-08 intra-subject，**Pareto 支配** ATM / BrainAE / CogCapPro。（7/7 全赢受前沿约束，不可达，不作目标）

## 1. 铁律（防泄露）

- **禁止**任何含 GT 类名的 prompt。事故：`prompts_full_hcma_test.json` 200/200 行含类名 → 41 个 `run_*.sh`、49 个七指标文件已作废
- prompt 主表 = **generic 无类名模板**（`prompts_deploy.json`）；若报预测类名行（`prompts_pred.json`）必须**并列** generic 行
- 生成主路径保持**零样本**；gallery 只用于检索
- 检索主表报 **raw top1 / CSLS**；`sinkhorn` 是转导式，另列不与 SOTA 混比

## 2. 已确证事实（不要再验证）

**协议**：PixCorr = RGB@425 BILINEAR；SSIM = gray@425 gaussian σ=1.5, `use_sample_covariance=False`, `data_range=1.0`

**协议偏差**：官方 ATM sub-08 图（`eeg-brainit/outputs/erdc/w7_official_flat`）在我们打分器下 6/7 指标落在 ±0.009，**唯 SSIM 有 −0.035 系统偏差**（0.3101 vs 论文 0.345）

**量级（决定什么东西值得做）**

| 干预 | 量级 |
|---|---|
| 注入**层位置** | ΔSSIM ±0.018 / ΔIncep ±0.027 |
| 条件**机制**（LL-init→CN+IP） | ΔSSIM **+0.109** / ΔIncep **±0.200** |
| **prompt 泄露** | ΔIncep **+0.17** |

**关键读数**

- 语义 **oracle 上界**（完美类名）：Incep **0.912** / CLIP 0.945 —— **超过所有 SOTA**，且此时 SSIM 跌到 0.2611（干净臂 0.37–0.39）
  ⇒ 语义**交付机制够用**；SSIM **不来自语义通路**；两条指标互相挤占
- 通道：17 vs 63 **无统计显著差异** ⇒ 用 17
- cycle 在 CLIP/DINO/RN50 三空间排序一致且与 SSIM 反序 ⇒ CLIP 空间**可证退化**
- 路由融合：composite **0.6025** vs 单层图 0.4550（**+0.15**，配对显著）
- 已证伪：`hier_aux`、`multi_head`、检索记忆库、分层注入（预登记 FAIL）

## 3. 干净臂最好值（sub-08，prompt=generic 已验证）

| 指标 | 最好 | 臂 |
|---|---|---|
| PixCorr | 0.1729 | `layered` |
| SSIM | 0.3880 | `rev` |
| Alex2 | 0.7818 | `mb_p2_i3` |
| Alex5 | 0.8669 | `mb_p3_i3_cn` |
| Incep | 0.7404 | `mb_p2_i3_k2` |
| CLIP | 0.8597 | `mb_p2_i3_sel8` |
| SwAV↓ | 0.5874 | `mb_p1_i1` |

## 4. SOTA 靶子（sub-08）

| | PixCorr | SSIM | Alex2 | Alex5 | Incep | CLIP | SwAV↓ |
|---|---|---|---|---|---|---|---|
| BrainAE | **0.211** | **0.432** | 0.768 | 0.869 | 0.753 | 0.816 | 0.541 |
| CogCapPro | 0.166 | 0.409 | **0.818** | **0.913** | **0.831** | **0.903** | **0.489** |
| ATM | 0.160 | 0.345 | 0.776 | 0.866 | 0.734 | 0.786 | 0.582 |

## 5. 缺口归属（决定重心）

| 指标 | 缺口 | 归属 |
|---|---|---|
| PixCorr | 0.038 | 空间 |
| SSIM | 0.044 → **校准后 ≈0.009** | 空间（**接近闭合**） |
| Alex2 / Alex5 | 0.036 / 0.046 | 语义 |
| **Incep** | **0.091** | **语义** |
| CLIP | 0.043 | 语义 |
| **SwAV↓** | **0.098** | **语义** |

⇒ **重心在语义侧，不在空间侧。**

## 6. v3 架构（4 条）

单通道必然满足 \(f(q_s)+g(q_p)=C\) ⇒ 一升必一降。破法 = **分池 + 分目标**。

| 模块 | 内容 |
|---|---|
| **M1 因子化潜变量** | `f_sem : h→z_sem(1024)` [NCE **+ MSE 锚定**] ；`f_spa : h→z_spa(4×64×64)` [MSE→GT VAE 潜变量]；`L_dec` 去相关保证容量不相交 |
| **M2 空间通路** | `z_spa` → VAE decode → img2img init；→ depth → ControlNet（早期 `[0,τ)`） |
| **M3 先验精炼** | 轻量扩散 \(\epsilon(z_I^t,t,z_{sem})\) **采样** \(z_I\)（落流形上）；兜底 CMP |
| **M4 融合** | 交叉注意力融合 {image,text,depth,edge} → `m_fused`，再注入 IP-Adapter |
| M6/M7 保留 | 路由 bank 分数级融合；空间 cycle（depth/VAE latent，不用 CLIP） |

`L = L_sem + λ_a·L_anchor + λ_s·L_spa + λ_d·L_dec + λ_c·L_cyc`
**不变式**：`L_sem`/`L_anchor` 只碰 `z_sem`，`L_spa` 只碰 `z_spa` ⇒ 目标自动不冲突。

独有项：**任务因子化潜变量**、**双通路容量分离**、**分数级路由融合**（SOTA 均无）

## 7. 判据

| ID | 比较 | 通过 |
|---|---|---|
| **V1** | 双通路 vs 最佳单通路 | **SSIM ≥+0.01 且 Incep ≥+0.03 同时 → (+,+) 支配成立** |
| V2 | 因子化 vs 单潜变量（等容量） | 两轴不差，至少一轴显著更好 |
| V3 | 去 `L_dec` | 前沿塌回一升一降 |
| V4 | 加 `L_anchor` | 检索不降 **且** 生成 ≥+0.02 |
| V8 | 融合 vs 并联 | 融合 ≥ 并联 |

**证伪**：V1 出现一升一降 ⇒ Pareto 主张作废，转检索主线

## 8. 阻塞（未解）

1. **INVALID 标记**：41 个泄露脚本 / 49 个污染文件写禁入清单
2. **SSIM 校准**：确认 −0.035 偏差是否随质量变化
3. **`sdedit_ll_s082` 矛盾**：0.2611（json）vs 0.3850（表）

## 9. 复用资产

`ss_modules.py`(POSTERIOR_17) · `ocf_train.py`(NCE/diff_loss/split) · `cfmsf_route_probe.py`(路由bank) · `cfmsf_fuse_eval.py`(CSLS) · `eval_official_seven_dir.py` · `device_audit.py` · split: `outputs/leakfree/split.json`

## 10. 下一步

M1 因子化编码器 `scripts/nda/nw3_encoder.py`
