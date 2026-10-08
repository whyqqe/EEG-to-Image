# 跨被试 EEG→图像解码：完整方案 v2.1

> 状态：**设计定稿，代码未改**
> 取代：`docs/eeg2image_cross_subject_plan.md` §4.4「被试即模态」路线
> 日期：2026-10-03

---

## 0. TL;DR

范式从「**提取被试特征向量 → 条件化主网络**」转向「**训练被试无关的通用 EEG 编码器
→ 再映射到图像空间**」。

这不是口味偏好，而是三条独立证据同时指向的结论：
① 本项目的 `diag_zcollapse` / `diag_subject_statistic` 实测条件化塌缩且无收益；
② 外部研究实测「抑制被试身份不提升检索，拉近同刺激跨被试才提升」；
③ 2026 榜首方法的增益**几乎全部来自编码器之外**（目标侧/损失侧/校准侧）。

| 维度 | 决定 | 关键依据 |
|---|---|---|
| **EEG 主干** | **ATM-S 家族**（通道 token Transformer + 时空卷积）= 现有 `EEGTrunk`，**不改 block** | 榜首 SAMGA(34.4) 的 EEG 侧**同属 ATM 家族**；增益来自目标/损失/校准 |
| **被试处理** | **无被试向量**。跨被试一致性损失 + 测试时几何校准 | XS-NCE **+2.23pp**（CI [1.28, 3.19]）；对抗抑制**有害** |
| **视觉目标** | **结构化多层目标** + 可学习路由，不用单一全局嵌入 | SVTL 35.3 vs SAMGA`*` 29.6；penultimate **+2.6pp** |
| **对齐结构** | **粗到细**（MMD 粗 + 对比细） | SAMGA 增益的另一半 |
| **单阶段训练** | **不设独立映射网络**（原 Stage B 取消）。编码器直接对齐图像空间；检索侧"映射"由**测试时无标签白化**承担，生成侧由**扩散先验**承担 | 当前 SOTA 全是单阶段；编码器已被 CLIP 教师对齐时，独立映射网络退化为投影（见 §6） |
| **测试时** | SAW + 自适应 CSLS + 坐标恢复 | **单点最大杠杆**：26.22 → **53.23** Top-1 |
| **不做** | 对抗抑制（GRL）、显式被试向量、Stage 2 元训练、**独立映射网络（原 Stage B）** | 实测无益／有害；本方案是**简化**，删的比加的多 |

---

## 1. 任务与评测协议

### 1.1 数据
THINGS-EEG2，10 被试，**63 通道**，250 Hz，刺激后 0–1000 ms（`n_timepoints=250`）。
- 训练：1654 概念 × 10 图 × 4 重复
- 测试：**200 概念 × 1 图 × 80 重复**（共享 200 图 gallery）

### 1.2 协议
严格 **LOSO**（9 训 1 测），**200-way zero-shot 检索**，报 Top-1 / Top-5。

**泄漏约束**（代码中必须显式断言）：
- 训练/测试**概念不重叠** → 概念级监督不构成泄漏；
- 目标被试的**标签不参与**任何训练或超参选择；
- 早期停止只能用源域验证集，**不能**用目标被试测试集。

### 1.3 三档协议必须分开标注（否则数字不可比）
| 档位 | 允许 |
|---|---|
| **朴素** | 只训编码器，无测试时变换 |
| **+ 校准** | 允许**无标签**测试时几何变换（SAW / CSLS / 坐标恢复） |
| **+ 精修** | 允许用测试被试的重复试次做 transductive 精修 |

文献里 ATM 11.9 → SCORE 53.23 的差距**主要就在后两档**，不是编码器。
本方案路线：**先把朴素档做到上限，再叠加无标签校准。**

---

## 2. 为什么不提取被试特征

| 证据 | 数值 | 出处 |
|---|---|---|
| 学习式 `z_s` 跨被试余弦 | **0.9999**，separation **−0.0000** | `scripts/diag_zcollapse.py` |
| per-channel 矩描述子分离度 | **−0.009** | `scripts/diag_subject_statistic.py` |
| 显式条件化 vs ID 表对照 | **无可测收益** | job 637403 |
| anchor 在合成方向中的占比 | 仅 **32%**（`‖net‖=36.1` 压过 `17.0`） | `scripts/diag_anchor_swamp.py` |
| 抑制被试身份 | **不提升检索；对抗式抑制有害** | *Towards Zero-Shot Cross-Subject Generalization* |
| 拉近同刺激跨被试（XS-NCE） | **+2.23pp**，CI **[+1.28, +3.19]** | 同上 |
| 跨被试 EEG→EEG 检索 vs 单被试 EEG→CLIP | **20.05% > 14.85%** | 同上 |

**机理**：`z_s` 塌缩不是 bug，而是**次问题的最优解**——目标函数奖励不变量，却没有任何
一项奖励「条件化携带信息」。所以不必修，应当**绕开**。被试适配改由两个不依赖显式向量的
机制承担：

1. **跨被试一致性损失**（constructive，非抑制）；
2. **测试时无标签几何校准**（SATTC / SCORE 路线）。

---

## 3. SOTA 定位与选型

### 3.1 2026 年真实排名（THINGS-EEG2 inter-subject 200-way）

⚠️ 必须连同"协议"一列读。

| 方法 | Top-1 | Top-5 | 年份 | 协议要点 |
|---|---|---|---|---|
| **SCORE** | **53.23** | **83.55** | 2026 | 无标签坐标恢复 + recovery-aware 训练（用 SAMGA 编码器） |
| SVTL + 精修 | 48.1 | 77.1 | 2026 | 用了测试被试重复试次 |
| **SVTL（无目标适配）** | **35.3** | **65.6** | 2026 | 严格无目标适配 |
| **SAMGA** | **34.4** | **64.8** | 2025/26 | 榜单榜首 |
| ViEEG | 22.9 | 51.4 | 2025 | LOSO |
| Shallow | 22.4 | 50.8 | — | — |
| Neuro-Bridge | 19.0 | 45.9 | — | — |
| PA-NCE | 15.2 | — | 2026 | ATM-S，仅换目标表征 |
| HyFI | 15.1 | 37.2 | AAAI 2026 | **17 通道 + 试次平均**（更宽松） |
| UBP | 12.4 | 33.4 | 2025 | — |
| **ATM / ATM-S** | **11.9** | **33.8** | **2024** | — |
| NICE | 6.2 | 21.4 | 2024 | — |
| BraVL | 1.5 | 5.6 | 2023 | — |

### 3.2 关键纠正：差距**不在**编码器 block

**SAMGA 的 EEG 编码器本身就是 ATM 家族**（开源仓库：`DBComformer.py` + `eeg_encoder/`
下的 ATM-style blocks）。它相对 ATM 的 **+22.5 点几乎全部来自编码器之外**：

| SAMGA 的增益来源 | 层面 |
|---|---|
| 被试感知多粒度视觉目标（InternViT 5 层 + 可学习路由） | **目标/监督侧** |
| 粗到细对齐（MMD 粗 → 对比细） | **损失侧** |
| 被试特定路由残差 + subject dropout | 目标侧 |
| 推理只用全局先验 | 部署侧 |

**SVTL 赢在"结构化视觉目标"，SCORE 赢在"坐标恢复"——两者用的都是 SAMGA 编码器。**
另有独立证据：**视觉编码器的选择比 EEG 编码器的选择更能决定解码性能。**

> **正确杠杆顺序：① 视觉目标/监督结构 → ② 被试感知路由 → ③ 粗到细对齐 → ④ 测试时校准。**

### 3.3 选型结论

| 组件 | 决定 |
|---|---|
| **EEG 主干** | 保持现有 `EEGTrunk`（ATM-S 形态），**不换 block** |
| **升级 1（目标）** | 被试感知多粒度目标（多层 + 可学习路由） |
| **升级 2（损失）** | 粗到细（MMD 粗 + 对比细） |
| **升级 3（校准）** | SAW + CSLS + 坐标恢复 |
| **候选新 block（消融）** | STAMBRIDGE（iTransformer + STAM，2026，最值得试）；ViEEG；CSBrain |
| **对照臂** | EEG-FM（CBraMod / LaBraM）、EEGiT、NICE/shallow |

---

## 4. 架构总览

```
┌─ Stage A：通用 EEG 编码器（被试无关）──────────────────────────────────────┐
│                                                                            │
│  x (B,63,250) ─► EEGTrunk (ATM-S) ─► e (B,512) ─► L2norm ─┐                │
│                                                             │              │
│  结构化多层 CLIP 目标 ─► LayerFusion ─► z_img (B,512) ──────┤              │
│                                                             ▼              │
│   L_img  InfoNCE(e, z_img)           语义对齐            ← 老师            │
│   L_xs   同刺激跨被试 multi-positive  跨被试一致         ← 核心            │
│   L_mmd  粗对齐（分布级）             稳定共享几何       ← 新增            │
│   L_proto 非对称 EEG 原型锚           类级锚定           ← 打开            │
│   L_reg  VICReg                       防方差塌缩                          │
│   (L_rkd 关系蒸馏 → 降为可选消融，见 §5.3(3))                             │
│   ✗ dec 大幅降权  ✗ adv=0  ✗ 无被试向量  ✗ 无支撑集前向                   │
└────────────────────────────────────────────────────────────────────────────┘
                    │ 冻结 e   （无独立映射阶段，见 §6）
                    ▼
┌─ Stage C：LOSO 零样本部署（全无标签）─────────────────────────────────────┐
│   SAW 白化 ─► 自适应 CSLS ─► 坐标恢复 ─► 200-way Top-1/5                    │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Stage A 详解

### 5.1 数据流与 batch 构造

```
每个 step：
  batch_stimuli = 8 个不同刺激
  × 每个刺激抽全部 9 个源被试
  → 72 行 (B=72)，每行 x ∈ R^{63×250}
```

**为什么 9 被试同时出现**：`L_img` 的对角形式在"g 个被试共享同一目标"的布局下与
multi-positive **数学等价**（`losses/contrastive.py` 中有证明）；同时 `L_xs` 需要同刺激
的多被试正样本。**两个损失由同一个布局同时满足**，这是 batch 设计的核心约束。

### 5.2 结构化多层目标（`z_img`）

```
image ─► 冻结视觉编码器（CLIP ViT-H/14）
       ─► 多层 [20,24,28,32,36]
       ─► 每层 LayerNorm + L2norm + 线性投影到 d_embed (512)
       ─► 路由融合（uniform → routed）
       = z_img
```
- **本地已有**：`clip_h14_multilevel`（layers 20/24/28/32/36，train/test 均已预计算）；
  可选增补 `internvit_multilevel_20_24_28_32_36`。
- **新增**：CLIP ViT-L/14 的 **penultimate 1024-d**（PA-NCE 证明 +2.6pp，需一次 GPU 提取）。
- **路由**：`LayerFusion`（`epd/encoders.py` 已有 `uniform` / `routed`）。
  **纪律：先用 `uniform` 回答"多层是否互补"，再上 `routed`。**
- 路由的 subject-specific 残差在**推理时旁路**（只用全局先验）→ 部署无需被试身份。

### 5.3 损失函数（完整定义）

**(1) 语义对齐 `L_img`**
```
L_img = InfoNCE( normalize(e), normalize(z_img) )       τ = 0.07
```

**(2) 跨被试一致性 `L_xs`（核心）**
```
L_xs = MultiPosInfoNCE( e, groups=stimulus_id )          τ = 0.10
```
同刺激、不同被试（含同被试不同重复）互为正样本。**这是"拉近"，与 `dec` 的"推远"方向相反。**

**(3) 关系型几何蒸馏 `L_rkd`（→ 可选消融，不进主线）**
```
L_rkd = ‖ K_e − K_i ‖_F / B² ，  K = normalize_row(Z) normalize_row(Z)ᵀ
```
维度无关（512 vs 768/1024 无需共享坐标系）、尺度不变，用 Gram 而非 CKA
（几何感知蒸馏文献证明 CKA 在 loss=0 时都不保证几何一致）。

**但从主线移除**：原先保留它的唯一理由是"让 Stage B 有意义"。Stage B 已取消（§6），
理由消失；且其证据来自 LLM 蒸馏／人类对齐的**类比**，**不来自 EEG**。保留为消融臂，
用于检验"关系结构"是否比"绝对目标"更鲁棒。

**(4) 非对称原型锚 `L_proto`（打开现有开关）**
维护 EEG 类级 EMA 原型作为稳定神经锚，**图像嵌入向锚对齐**（非对称）。
文献明确：**对称 SupCon 无增益**；非对称锚定才有效。

**(5) 粗对齐 `L_mmd`（新增）**
分布级 MMD 对齐（源域被试间 + EEG/图像间），稳定共享几何、降低 subject-induced shift。
先粗后细是 SAMGA 增益的另一半。

**(6) 防塌缩 `L_reg`**
VICReg variance + covariance。

### 5.4 损失权重表

| 项 | 作用 | 权重 | 与旧配置的差异 |
|---|---|---|---|
| `img` | EEG ↔ 结构化图像目标 | **1.0** | 不变 |
| `cross` | 跨被试同刺激一致性（核心） | **0.5 → 0.7** | **升** |
| `mmd` | 粗对齐（分布级） | **0.1 → 0.2** | **提为正式项** |
| ~~`rkd`~~ | 关系型几何蒸馏 | ~~0.3~~ | **不进主线**（降为消融，§5.3(3)） |
| `proto` | 非对称 EEG 原型锚 | **0.0 → 0.2** | **打开** |
| `reg` | VICReg 防塌缩 | **0.5** | 不变 |
| `dec` | HSIC 被试解耦 | **0.2 → 0.05** | **大幅降** |
| `adv` | 对抗抑制 | **0.0** | **保持关闭** |

> **权重纪律（最重要的一处改动）**：旧配置同时最大化「跨被试一致」和「被试不可分」，
> 二者在梯度上互相抵消——**这正是条件化塌缩的同源病灶**。`dec` 必须降到接近 0。
> 用实验臂 A4 以数值裁决，不靠论证。

### 5.5 正式训练配置

```yaml
# ---------------------------------------------------------------- 骨干
n_channels: 63
n_timepoints: 250
d_model: 200
d_embed: 512
n_heads: 4
n_blocks: 2
dim_ff: 512
dropout: 0.1

# ------------------------------------------------------------ 视觉目标
target:
  feature_sets: [clip_h14_multilevel]     # 可选 + internvit_multilevel_20_24_28_32_36
  layers: [20, 24, 28, 32, 36]
  fusion: uniform                          # uniform -> routed
  l2norm: true
  project_dim: 512

# ---------------------------------------------------------------- 损失
loss_weights:
  img: 1.0
  cross: 0.7
  mmd: 0.2
  proto: 0.2
  reg: 0.5
  dec: 0.05
  adv: 0.0
temp_img: 0.07
temp_cross: 0.10
softplus: true

# ---------------------------------------------------------------- 优化
epochs: 60
lr: 3.0e-4            # ATM 论文设定
weight_decay: 1.0e-4
grad_clip: 1.0
batch_stimuli: 8
subjects_per_stimulus: null   # null = 全部源被试（9）
num_workers: 4
eval_batch: 200
seed: 2025

# ---------------------------------------------------------------- 协议
channel_set: all63
mvnn: train
```

**选择策略**：**末轮选择**（final-epoch selection），与 SOTA 协议一致——
**不用**基于目标被试的早期停止（那是泄漏）。

---

## 6. 为什么不设独立映射阶段（原 Stage B 取消）

**决定：取消独立映射网络。** 训练只有一段（Stage A），部署只有校准（Stage C）。

### 6.1 四条理由
1. **当前 SOTA 全是单阶段**：SAMGA / SVTL / SCORE / PA-NCE 都是"编码器直接对齐图像空间"。
2. **编码器已被 CLIP 教师对齐**（§5.2 的 `L_img`）。此时"EEG 编码 → 图像 CLIP"的最优解
   接近**恒等映射**——训练一个网络去逼近恒等是多余的。
3. **若不落 CLIP 空间，动机就变成循环论证**：唯一理由是"让映射阶段有意义"；且支撑它的
   `L_rkd` 证据来自 LLM 蒸馏／人类对齐的**类比**，**不来自 EEG**（故降为消融）。
4. **文献不支持"映射网络"**：Li et al. 的"两阶段"是 ① 编码器 → CLIP，② **扩散生成**，
   不是"EEG 空间 → CLIP 空间的映射网络"；Perceptogram 支持的是"EEG **一跳**线性回归到 CLIP"。

### 6.2 原诉求由谁承担

| 原诉求 | 承担者 | 证据强度 |
|---|---|---|
| 检索侧的"映射" | **测试时无标签线性校准**（SAW + CSLS + 坐标恢复，Stage C） | **很强**：26.22 → 53.23；且**已有实现** |
| 生成侧的"映射" | **条件扩散先验** `p(z_I \| e)`（后续阶段模块，**不是**映射网络） | 强：Li et al. / Mind's Eye 范式 |
| 编码器塑空间 | Stage A 的 `L_img` + `L_xs` | 强：SAMGA 34.4 / SVTL 35.3 |

关键点：检索侧那个变换本来就是个**线性映射**，而在**测试时无标签地学它，比在源域训一个
网络更好**——这正是 SATTC / SCORE 的核心贡献。

### 6.3 留一个可证伪的出口
若 M0 审计显示 Stage A 空间 → 检索存在**显著且可复现**的余量，允许加一个**极轻量线性头**
（单层，与 Stage C 校准一起学）。
> ⚠️ **这不是恢复 Stage B**：不是"训练一个网络"，而是一个可解释的线性坐标变换。

---

## 7. Stage C 详解（部署校准）

**全项目最大杠杆，编码器无关、全无标签、CPU 可做。**

| 操作 | 作用 | 文献数值 |
|---|---|---|
| **SAW**（被试自适应白化） | 用目标被试**无标签**统计量白化查询嵌入 | ATM 9.2 → 14.8（含 SATTC） |
| **自适应 CSLS** | 消除 hubness | 同上 |
| **坐标恢复**（加权正交 Procrustes + 互近邻伪配对） | 对齐源/目标坐标系 | **26.22 → 53.23** |
| 结构化 PoE | 融合几何专家 + 结构专家 | SATTC 完整版 |
| 可选：跨被试 query 平均 | 共识降噪 | 14.74 → 27.50（+12.76pp） |

**全部已有实现，不要重写**：
- `src/samclip/calibration.py`（`saw_whiten` / `csls_scores`，含**秩亏协方差收缩**与
  **条件数上界**两个保护——文件注释记录了数值踩坑）；
- `/project/peilab/why/eeg-retrieval/scripts/epd/recover.py`（坐标恢复，含**弃权门**）。

---

## 8. 训练协议

| 项 | 设定 |
|---|---|
| LOSO 折 | sub-08 先做单折（既有流水线），通过后跑全 10 折 |
| Stage A 训练数据 | **仅 9 个源被试** |
| Stage C | 目标被试**无标签**数据（无独立映射阶段） |
| 选择 | 末轮 |
| 随机种子 | 3 个（结果取均值，与 SATTC 一致） |

**Slurm 纪律**（依 AGENTS.md）：
- `--partition=normal --account=peilab`，排除 `dgx-09,11,17,30`；
- 缓存变量（`HF_HOME`/`TORCH_HOME`/`PIP_CACHE_DIR` 等）**先于**任何 torch 导入；
- 显式绝对路径解释器 `.venv/bin/python`，不依赖 `activate`；
- **CUDA 硬门**：不可用即 `exit 1`，避免 CPU 空跑；
- 可续跑（每阶段 `[SKIP]` 已有输出）。

---

## 9. 代码改造清单（**已实施**，2026-10-03）

> 本节记录**实际做了什么**，以及与原计划的偏离和原因。偏离项逐条列出——计划与代码不一致
> 时以本节为准，且必须写明理由，否则下一个读代码的人会把偏离当成疏漏。

### 9.1 明确取消（不再是「改写」，是**没有这个机制**）

| 对象 | 处置 | 说明 |
|---|---|---|
| `models/subject_conditioning.py` 主体（`SubjectConditioner` / FiLM / LoRA / hyper 生成器 / `SupportSetEncoder` / `CondParams`） | **保留文件为墓碑**（纯 docstring，无代码、无导入者） | 不直接删除：560 行主假设不该在树里变成一个无声的洞。墓碑里写清了三点实测证据（`z_s` 余弦 0.9999 塌缩；per-channel moments 分离度 -0.01；文献中抑制被试身份无增益、对抗抑制有害），以及转向后的替代物。指向归档目录 |
| `models/backbone.py` 的 FiLM/LoRA 调制缝 | `ConditionedEncoderLayer` → `EncoderLayer`，`CondParams` 参数删除 | trunk 现在是**逐被试完全相同**的共享函数 |
| Stage 2 元训练、`scripts/run_stage2_meta.py`、`configs/meta_s2.yaml`、`slurm/20_stage2.sbatch`、`slurm/validate_stage2*.sbatch` | **归档**至 `scripts/legacy/v1_subject_conditioning/`、`configs/legacy/`、`slurm/legacy/` | 无 hypernetwork 即无此阶段 |
| `configs/ablation_a1_no_conditioning.yaml` | 归档至 `configs/legacy/` | 消融对象已不存在 |
| `SAMCLIP.freeze_shared` / `_sync_frozen_eval` / `frozen_state` / `frozen_drift` / `trainable_parameters` | **删除** | 它们只有一个用途：在 Stage 2 训练 hypernetwork 时保护 Stage A 的几何。Stage 2 没了，保护对象也没了。它编码的 BatchNorm 教训（`requires_grad=False` **不冻结** BN 的 running stats）**保留为 `models/backbone.py` 顶部的一段注释**——代码会过时，教训不会 |
| `evaluate.extract_features(support_x=, z=)` | 签名删除 | 没有任何东西再被「条件化」；被试自适应改由**事后**对 EEG 嵌入做白化实现，更简单也更强（§7） |

### 9.2 保留（**未改动**）
| 对象 | 原因 |
|---|---|
| `models/backbone.py::EEGTrunk` / `TemporalSpatialAggregator` | 即 ATM-S 骨架 |
| `losses/contrastive.py` 全部 | XS-NCE、温度下界 `SCALE_MIN`、等价性证明 |
| `losses/invariance.py` 全部 | `mmd_subject`（粗对齐）、`hsic_subject`（轻解耦）、`SubjectAdversary`（消融臂） |
| `calibration.py`（SAW / CSLS）+ `epd/recover.py` | §7 |
| `data/mvnn.py` / `scripts/build_cache.py` | 预处理管线 |
| `smoke_test.py` | 审计工具（已按 v2 重写，见 9.5） |
| `diag_subject_statistic.py` / `diag_subject_descriptor.py` | **转向证据本身**，且不依赖已删代码，可继续运行 |

### 9.3 新增
| 文件 | 内容 |
|---|---|
| `src/samclip/losses/relational.py` | `gram_matrix` / `gram_distill_loss`（Gram 归一化 ⇒ 尺度不变；**仅消融臂**，默认 0.0） |
| `losses/regularizers.py::PrototypeEMA.image_anchor_contrast` | **非对称原型锚**：梯度流向 **image head**，**不流向** EEG 编码器、更不流向原型缓冲区。方向本身就是机制，`smoke_test` 5 直接断言了这一点 |
| `scripts/probe_target_fusion.py` + `slurm/probe_fusion.sbatch` | 目标融合 A/B/C 探针（`mean`/`routed`/`routed_sr`）。用**日志处理器**读 `dec`，而不是抓 sbatch 的 `.out`——日志目的地由调用方决定，抓文件只在恰好重定向时可读 |
| `configs/ablation_a2_no_cross.yaml` | `cross: 0.0` 对照（**文献唯一单点验证过的项**，+2.23pp，应先跑） |
| `configs/ablation_a7_uniform_target.yaml` / `ablation_a7_routed_sr.yaml` | 目标融合消融臂 |

### 9.4 修改
| 文件 | 改动 |
|---|---|
| `models/samclip.py` | 移除全部条件化 wiring；保留 `LayerRouter`；`forward` 不再接受 `z`/`support_x`；新增 `target_layer_weights()` 供 eval 记录学习到的层权重 |
| `data/things_eeg.py` | 训练集样本**新增 `concept` 字段**（原型库可按 `concept` 或 `stimulus` 建键，见 `prototype_level`） |
| `train.py` | 权重表加 `rkd`；`proto` 改为**非对称**（`proto_contrast` + `image_anchor_contrast`）；`LossWeights.from_cfg` 对**未知键硬报错**（拼错的权重组名会被静默忽略，那正是「报告了一个从未生效的项」的成因）；移除 Stage 2 与支撑集路径 |
| `evaluate.py` | 删除条件化参数；docstring 明确这是**训练侧原始余弦**，部署档在 `calibration.py` |
| `scripts/run_eval.py` | 报告由「conditioning × geometry」二维降为**几何阶梯一维**（raw → SAW → CSLS → +recovery），并输出 `target_layer_weights`；删除 `--k-support`（已无消费者） |
| `scripts/submit_pipeline.py` | DAG 由 `cache→stage1→stage2→eval×2` 降为 `cache→stageA→eval`；新增 `--out-tag`（eval 报告按父目录名建键，两个消融臂共用一个 tag 会互相覆盖） |
| `configs/default.yaml` | 按 §5.5 重写；**整节删除** `conditioning` 与 `geom`/`z_norm`/`support_anchor` 论证；`dec: 0.2 → 0.05`；`loss_weights` 加 `rkd`；新增 `prototype_level` 与 `calibration` 段 |
| `slurm/30_eval.sbatch` | 删除 `K_SUPPORT` 传递 |
| `docs/eeg2image_cross_subject_plan.md` | 顶部标注为 v1 历史文档，指向本文档 |

### 9.5 与计划的偏离（**逐条列出理由**）

| 计划 | 实际 | 理由 |
|---|---|---|
| 新增 `configs/stage_a.yaml` | **未新增**；`configs/default.yaml` 即 Stage A 配置 | 两者内容会 95% 重复，而「一份架构只有一处定义」正是本仓库配置继承存在的原因。重复的架构节正是「run 静默按错误宽度加载 checkpoint」的成因 |
| 新增 `scripts/extract_clip_penultimate.py` | **未写**（仍留在 M3） | 它是 M3 的输入准备，不是架构骨架的一部分；在 M2 结论出来之前提取会浪费 `/project` 空间（仅 ~59 GB 可用） |
| 新增 `procrustes_dist` | **未加** | 坐标恢复已作为**无标签、闭式**拟合存在于 `calibration.py` + `epd/recover.py`；再造一个蒸馏项去做同一件事没有消费者，属于 §9.2 所说的「训练一个网络去做闭式拟合做得更好的事」 |
| `routed_sr` 被 v1 实测否决 | **重新注册为活跃消融臂** | v1 的反对理由是它推高被试依赖项（0.22→0.48），而那个压力来自 `dec: 0.2`。v2 把 `dec` 降到 0.05，**权衡前提已经改变**，不能拿旧结论当新结论。它仍**不是**默认值 |
| `diag_zcollapse.py` / `diag_anchor_swamp.py` / `diag_zscale.py` / `probe_subject_z.py` / `probe_regression.py` 保留为回归基线 | **归档**至 `scripts/legacy/v1_subject_conditioning/` | 它们 `import` 已删除的 `SupportSetEncoder`，留原地只会变成 5 个 import 即崩的脚本。归档保留证据，同时不让活跃目录里全是断脚本。禁用的是**范式**，不是证据 |

> **净删除量 > 净新增量**。这一点是刻意的：v1 的问题不是组件太少，而是有一个从未生效的
> 机制占着代码、配置、smoke 断言和两个流水线阶段。

### 9.6 v3.1：两个「项在跑但没在训练」的 bug（2026-10-03）

**触发原因**：v3 完整实验（`v3-sub-08`）跑通、无报错，但 raw Top-1 只有 14.0（校准后
20.5），距参考基准 26.22 仍差 5.7pp，且训练在第 10 个 epoch 就到达 raw ≈12 然后 40 个
epoch 全是噪声。**loss 值正常、梯度有限、形状正确**——这套组合骗过了整个 smoke 套件。

诊断工具 `scripts/diag_term_influence.py`（新增）对**每个损失项分别求对模型参数的梯度
范数**——权重为 $w$ 的项对表征的实际作用力是 $w\,\partial L/\partial\theta$，而不是
$w L$。在一个已训练 checkpoint 上测得：

| 项 | 值 | 权重 | $w\cdot L$ | $\lVert w\,\partial L/\partial\theta\rVert$ 占比 |
|---|---|---|---|---|
| `img` | 3.57 | 1.0 | 3.57 | 44.8% |
| `cross` | 3.52 | 0.7 | 2.47 | 32.3% |
| `var` | 0.468 | 0.5 | 0.234 | 9.7% |
| `cov` | 0.128 | 0.5 | 0.064 | 9.2% |
| `anchor` | 5.30 | 0.2 | 1.06 | **1.5%** |
| `proto` | 5.32 | 0.2 | 1.06 | **1.4%** |
| `mmd` | 0.016 | 0.2 | 0.003 | **0.9%** |
| `dec` | 0.087 | 0.05 | 0.004 | 0.25% |

`proto`+`anchor` 占了 **26% 的 loss 值，只贡献 2.9% 的梯度**。加上 `mmd`，三个"设计
过的机制"合计贡献 **3.9%**；实际在训练网络的是 `img`+`cross`+VICReg。这就是训练早停
和 Stage 1 / Stage 2 无法区分的原因。

**根因 1：`PrototypeEMA._contrast_to_proto` 没有温度。**
它把 L2 归一化向量之间的**余弦相似度直接当 logits** 送进 `cross_entropy`。余弦值域是
$[-1,1]$，所以 384/1654 类的 softmax 近似均匀，loss 被钉在 $\ln N$：实测 `proto` 在
50 个 epoch 里一直是 5.32–5.50，而 $\ln 384 = 5.95$——**它从未离开初始化**。同步对照：
`img`（有温度，scale 收敛到 5.4）从 5.95 降到 3.51。均匀 softmax 的交叉熵是一个**完全
正常的数字**，这正是它存活下来的原因。
- 修复：加入可学习、有界温度，与 `InfoNCE` **共用同一个钳制函数**
  （`contrastive.effective_logit_scale`，含 `SCALE_MIN` 下界这一"载重"约束），并注册进
  `Trainer.criterion_parameters()` / `criterion_state()`（它挂在 Trainer 上，
  `model.parameters()` 看不见，与 9c 同一失效模式）。
- 实测：同一损失在 scale 1.0（旧路径）与初始 scale 2.727 下梯度比为 2.5×；在收敛尺度
  14 下为 **13.9×**。

**根因 2：`mmd_subject` 的 RBF 核看不见被试偏移。**
带宽中心取自**中位数成对距离**。在 L2 归一化空间里集中效应让任意两点距离都接近
$\sqrt 2$，于是 $\sigma \approx 1.427$，而**被试均值位移只有 0.195**——差 7 倍。更糟的是
`_rbf(a, a)` **包含对角线**（恒为 1，梯度为零），实测报告的 0.016 中约 $2/N = 0.0156$
就是这一常数偏置。全带宽扫描下**无偏** RBF-MMD² 仅 ~1e-4。**没有任何带宽能救它**：核宽
到能覆盖整个被试时分辨不出均值差，窄到能分辨时就把同一被试的样本当成互不相干。
- 修复：默认换成**线性核**。线性核 MMD 恒等于 $\lVert \mu_a - \mu_b\rVert^2$（被试均值
  距离的平方）——无带宽、无对角线偏置，且它瞄准的正是偏移真正存在的尺度。
- 实测梯度占比：**0.9% → 8.1%**（同为权重 1.0），一个数量级。
- RBF 分支保留为消融臂，`configs:mmd_kernel: rbf`。

**附带修正**：既然 MMD 现在是活项（~8% 梯度），Stage 2 把它归零就会**真的**删掉唯一惩罚
被试均值漂移的正则项（`cross` 是同刺激对比，不做这件事）。新增
`schedule.stage2_mmd`（默认 0.05，`0.0` 即精确复现参考配方）。

**回归测试**：`smoke_test.py` 9k / 9l 六个断言——断言的是**性质**（该项必须真的能推动
表征；线性核必须恒等于均值距离；必须随人为偏移单调变化；RBF 形态在同等偏移下报不出可用
量级），而不是"存在一个温度参数"。

#### v3.1 结果（`v31-sub-08`，seed 2025，final-epoch）

修复在**训练侧**效果明确，在**可部署指标上未兑现**：

| 指标 | v3 | v3.1 | Δ |
|---|---|---|---|
| Stage 2（ep20–49）平均 raw Top-1 | 12.63 | **14.63** | **+2.00** |
| 最优 raw Top-1 | 15.50 @ep19 | **16.50 @ep44** | +1.00 |
| final raw Top-1 | 14.00 | 14.50 | +0.50 |
| 最优 whiten+CSLS | **20.50** | 19.00 | −1.50 |

**Stage 2 的单调退化被修好了**：v3 的第二阶段从 ep19 的 15.5 一路退到 12–13，而 v3.1
稳在 14–16.5 并缓慢上升。`proto` 从 5.95 降到 4.56、`anchor` 从 5.95 降到 3.75、
`sc_proto` 从 2.73 学到 5.24——三个数字都证明这两个项**第一次真的在训练**（v3 里它们
50 个 epoch 都钉在 5.30–5.50）。

**可部署档的下降是白化这一格的混淆，不是编码器变差**：
`scripts/probe_calibration_rungs.py`（新增）扫 `saw_whiten` 的 `shrink`——v3 的曲线平坦
（19.5–20.5），而 **v3.1 的曲线随 shrink 单调上升**（shrink 0.0→0.7 时 17.0→19.0）。
白化是"用查询自身协方差去掉被试特异方向"；v3.1 的协方差里**有害成分更少**，激进白化因此
破坏的信号多于去掉的噪声。两个 checkpoint 的白化诊断几乎相同（cond 985 vs 941），所以
这不是数值稳定性问题，而是**`shrink` 是固定常量、却被拿去比较两种协方差结构不同的表征**
——一个真实的混淆项。去掉混淆后差距缩到 19.0 vs 20.5（n=200，1 trial = 0.5pp），
**在单折单种子下二者统计上不可区分**。

**因此当前瓶颈不在损失项的管线**。修复后目标函数已健康（`img` 44.8% / `cross` 32.3% /
VICReg 18.8% / `proto`+`anchor` 恢复为真实项 / `mmd` 8.1%），但 raw Top-1 仍在第 10 个
epoch 就到达 ~12 然后进入平台。

### 9.7 瓶颈定位：泛化差距，不是容量（v3.2 实验的立项依据）

§9.6 结尾列了三个候选（聚合器瓶颈、`batch_stimuli`、评测噪声）。第 1 个已被**测量否定**，
第 2 个优先级低于下面的发现。

**(a) 秩帽子是松的——容量不是限速环节。** `EEGTrunk` 末端 `TemporalSpatialAggregator`
的 `out_dim = agg_width × agg_pool`（默认 16×8 = 128）且**与 `d_model` 无关**，head 再把
它映到 `d_embed` 512 —— 所以 EEG 嵌入的秩上限一直是 128。新增
`scripts/probe_embedding_rank.py` 在 `v31-sub-08/last.pt` 上实测：

```
agg out_dim (RANK CEILING) = 128   trunk out_dim = 200   d_embed = 512
eeg  eff_rank  9.51   dims@90% 12   dims@99% 24   sv1 0.204   nnz_sv 199
img  eff_rank 10.92   dims@90% 12   dims@99% 20   sv1 0.175   nnz_sv 199
-> 秩帽子使用率 24/128 = 18.8%   cap is slack
```

嵌入只用了 128 个可用方向中的 24 个，而且它的**有效秩（9.5）基本等于图像目标的（10.9）**
—— 表征并不比它要对齐的东西更贫瘠。加宽聚合器买的是模型从不索取的方向。

**(b) 训练对比已被解决，测试却差 6 个点 —— 这是泛化差距。** 从
`outputs/slurm/samclip-s1-637907.out` 的 EP49 训练侧读数，`img` 项 = **3.69**。批次是
128 个概念 × 3 被试，若模型把 128 个**概念**完美分开、但把一个概念的 3 个被试放在同一
向量上，该值为 `-log(3e/(3e + 381))` = **3.87**。实测值**低于** 3.87，说明源侧不但类别
分开、类内还有轻微锐度（`sc_img` 已被学到 5.48，即温度从 0.07 收紧到 0.18 的反方向）。
同一个 checkpoint 在留出被试上只有 14.5 raw Top-1，白化后 20.0。

**(c) 缩小差距的手段此前完全缺席：整条流水线没有任何数据增强。**
`LosoTrainDataset.__init__` 一直有 `augment` 形参，而 `train.py` / `run_stage1.py` /
所有 config **零引用** —— 每个已记录的结果都是在原始标准化 epoch 上训练的。

这一点很关键，因为 `cross` 项的语义是"同刺激不同被试不变性"，而它能见到的被试差异
**只有 9 个源被试的自然变异**。白化的 +5.5pp 已经定位了疾病所在：测试时嵌入的**顶部方差
方向仍是被试特异的**。新增 `src/samclip/data/augment.py` 制造的正是留出被试与源被试之间的
那种变异 —— 电极增益漂移、坏电极、潜伏期抖动、噪声底 —— 于是编码器被要求对其不变的那部分
覆盖了幸存到测试时的干扰项。

四个变换都作用在**已标准化的 `(C, T)` epoch** 上，因此下面每个量级都是"每通道 z 单位"，
跨被试、跨缓存重建含义一致：

| 变换 | 默认 | 对应被试差异 |
|---|---|---|
| `shift` | ±8 / 250 = 3.2% | 刺激锁定潜伏期抖动 |
| `noise` | σ 0.15 z | 噪声底 |
| `gain` | 每通道 log-normal σ 0.20 | 阻抗 / 幅度漂移 |
| `channel_dropout` | 每行 10% 通道 | 坏电极 |

**顺序不是随意的**：`channel_dropout` 必须**最后**。第一级是 conv → BatchNorm1d → ELU，
把噪声加在 dropout 之后会把被置零的通道变成**半波整流的噪声底** —— 一个"我这个通道被丢
掉了"的可学习信号，反而毁掉这个变换。放在最后时 `ELU(0) = 0`，被丢通道对 head 的贡献
是**零**而非一个大的归一化常数。`smoke_test.py` 9m 把这条性质固化下来（4 个断言）。

`channel_dropout` 之所以安全，是因为第一级 conv 是 `groups=n_channels`（通道 c 只到达
通道 c 的输出），其后 `BatchNorm1d` 用一个 ~90% 是信号的混合批次统计量归一化：恒零通道
映到 ~0，再过 `ELU(0) = 0`。

**(d) v3.2 实验设计（3 臂 × 3 种子）。** 单折 n=200 下 1 trial = 0.5pp，而 v3/v3.1 的
全部差异都在 2–6 trials 量级 —— 在此之前任何改动都无法被判读。所以本轮同时建立**测量
基线**（`submit_pipeline.py --seeds`，其 help 本身就写着"protocol wants >=3 seeds for
the mean/std table"）：

| 臂 | config | 改动 |
|---|---|---|
| `base` | `loso_sub08.yaml` | 无（当前 best 配置） |
| `aug` | `loso_sub08_aug.yaml` | + 上表四个增强 |
| `cap` | `loso_sub08_cap.yaml` | `agg_width 16 → 64`（秩帽子 128 → 512） |

`cap` 臂**明知会落在噪声内**仍然提交：它是 (a) 的第一个假设，而"用数据驳回假设"比"用
推理驳回假设"便宜些——三个种子的一次测量就把它变成结论，`probe_embedding_rank.py` 的
18.8% 是它的预测值。

#### v3.2 结果（3 臂 × 3 种子，seed 2025/2026/2027，final-epoch，`scripts/summarize_arms.py`）

| 臂 | raw cosine | + SAW whiten | + CSLS | + whiten + CSLS |
|---|---|---|---|---|
| **`base`**（当前配置） | **16.33 ± 2.47** | **21.50 ± 2.78** | **19.67 ± 3.25** | **19.67 ± 1.76** |
| `aug`（+ 四增强） | 15.00 ± 1.50 | 14.67 ± 5.20 | 18.17 ± 5.51 | 14.67 ± 5.25 |
| `cap`（agg 128→512） | 11.50 ± 3.04 | 14.83 ± 3.79 | 14.17 ± 4.04 | 15.33 ± 2.75 |

配对到 `base`（同折、同测试集、同种子，故配对可消掉大部分方差）：

| 臂 | 档 | Δ | 方向 | p | 80% power 所需 n |
|---|---|---|---|---|---|
| `cap` | raw | −4.83pp | **0/3** | 0.205 | ~7 |
| `cap` | whiten | −6.67pp | **0/3** | 0.131 | ~4 |
| `cap` | CSLS | −5.50pp | **0/3** | 0.201 | ~7 |
| `cap` | w+csls | −4.33pp | **0/3** | 0.186 | ~6 |
| `aug` | raw | −1.33pp | 1/3 | 0.524 | ~40 |
| `aug` | whiten | −6.83pp | 0/3 | 0.223 | ~8 |

**结论 1（最重要的收益，来自基线本身）：此前的数字是单个"坏种子"的抽样。**
`base` 的 3 种子均值是 raw **16.33** / 校准档 **21.50**（最优档是 `whiten`，不是 `w+csls`），
而此前报告的单种子值是 14.5 / 20.5。`base_seed2025` 复现了旧值（raw 13.5、w+csls 18.0），
说明 **seed 2025 是三个种子里最差的一个**，而 v3/v3.1 的全部对比都建立在它上面。真实差距到
参考的 26.22 是 **4.7pp**，不是 5.7pp。

**结论 2：`cap` 是**负面**结果，不是中性结果。** 预测是"落在噪声内"，实测是 raw −4.83pp、
**每一个档位都是 0/3 种子为负**。机制清楚：`cap` 的训练侧 `img`（3.67/3.79/3.74）与 `base`
（3.54/3.65/3.78）**基本相同**，即源侧拟合程度一样，但迁移更差 —— 多出来的宽度被花在了
源被试特异、不跨被试的方向上，而源侧损失看不见这件事（它在源被试上计算）。
p ≈ 0.13–0.21，n=3 下不宜称为显著，但 0/3 的方向一致性加上机制，足以把"加宽聚合器"这个
方向关掉。

**结论 3：`aug`（本方案的假设）也失败，而失败方式有信息量。** `aug` 的训练侧 `img` 是
4.07/4.64/4.99，显著**高于** `base` 的 3.54–3.78，也高于 3.87（"概念已分开、类内无锐度"
的点）—— 即增强**强到让源侧对比没被解出来**。而在 `aug` 臂内部，测试表现对"被抑制的程度"
**单调**：抑制最轻的 seed（img 4.07）最好（raw 16.5、w+csls 20.0，与 `base` 齐平），
抑制最重的 seed（img 4.99）最差（raw 13.5、w+csls 9.5）。

所以这是一个真实的负面结果，且它给出了一个明确论断：**源侧对比被解出来是迁移的前提，
不是迁移的症状** —— 增强买来的不变性不足以补偿它阻止的拟合。若要再试，必须把强度调到
"`img` 仍能落到 3.87 以下"的区间，而不是本轮的默认值。

**结论 4：噪声地板已量化，它推翻了此前的比较方式。**

* **同配置同种子重跑**（`v31-sub-08` vs `v32-base-sub-08_seed2025`，config 逐键只有
  `out_dir` 不同）：Stage 2 均值差 **+0.54pp**，逐 epoch 差最大 **3.0pp**（sd 1.28）。
  即 GPU cuDNN 的非确定性本身就在 0.5pp 量级。
* **种子间**：`base` raw 是 13.5/17.5/18.0，sd 2.47pp。
* n=200 时 1 trial = 0.5pp，故 **3 种子只能判读 ~2pp 以上的效应**；`v3` → `v3.1` 的
  raw +0.5pp、部署档 −1.50pp **全部在噪声内**，§9.6 里那些比较不应作为结论引用（
  Stage 2 稳定性 +2.00pp 是唯一越线的，且方向由 `proto`/`anchor`/`sc_proto` 三个损失值
  独立佐证）。

---

## 10. 实验矩阵与验收标准

### 阶段 0：审计（**动手前必做**）
| 检查 | 方法 | 通过标准 |
|---|---|---|
| 冻结瓶颈 | `scripts/probe_regression.py`：Stage A 空间 → CLIP 线性 R² + 检索上限 | 上限显著高于 A0；据此判断是否需要 §6.3 的轻量线性头 |
| 基线可复现 | 复现 ATM-S 朴素档 | 应落在 **11.9 ± 1** Top-1 区间 |

### 阶段 1：损失消融（LOSO sub-08 单折）
| 臂 | 配置 | 回答 |
|---|---|---|
| A0 | `img` only | 单跳基线 ≈ ATM |
| A1 | + `cross` | XS-NCE 增益（预期 **+2~3pp**） |
| A2 | + `mmd`（粗到细） | SAMGA 式增益 |
| A3 | + `proto` | 非对称锚是否有效 |
| A4 | A3 **去掉 `dec`** | 旧解耦项是否在抵消 |
| A5 | A3 + `rkd`（0.3） | 关系蒸馏是否值得进主线（消融） |
| A6 | A3 + 轻量线性头 | 检索侧映射是否还有余量（§6.3） |

### 阶段 2：目标表征消融（固定 A4 编码器）
单层全局 / 多层 `uniform` / 多层 `routed` / + penultimate。
**验收**：结构化目标应显著优于单层全局（文献：29.6 → 35.3）。

### 阶段 3：校准消融（固定编码器）
无 / +SAW / +SAW+CSLS / +SAW+CSLS+坐标恢复 / +全量 SATTC。
**验收**：坐标恢复应带来两位数 Top-1 增益。

### 阶段 4：对照臂（可选）
EEGiT、CBraMod 冻结、ENIGMA 式 per-subject 输入线性层、STAMBRIDGE。

### 目标值（朴素档 → 校准档）
| 里程碑 | 朴素 Top-1 | +校准 Top-1 |
|---|---|---|
| 复现基线 | ~11.9 | ~14.8 |
| 目标侧升级后 | ~30+ | — |
| 全量 | ~35 | **≥ 45** |

---

## 11. 执行顺序（里程碑）

| # | 任务 | 产出 | 依赖 |
|---|---|---|---|
| **M0** | 审计：线性探针 + 基线复现 | 报告；决定是否需要轻量线性头 | — |
| **M1** | 代码清理（§9.1 删除）+ A0 跑通 | 干净基线 | M0 |
| ↳ | ✅ **代码部分已完成（2026-10-03）**：条件化范式全部移除、Stage 2 归档、`smoke_test.py` 全绿、`submit_pipeline.py --dry-run` 走通。**训练部分未跑**（需 GPU，经 Slurm）。 | `docs/eeg2image_v2_plan.md` §9 | — |
| **M2** | `L_xs` / `L_mmd` / `dec` 调整 → A1–A5 | 损失消融结论 | M1 |
| **M3** | 结构化多层目标（含 penultimate 提取）→ 阶段 2 | 目标消融结论 | M2 |
| **M4** | 粗到细（`L_mmd`）→ A3 | — | M2 |
| **M5** | 轻量线性头（§6.3）→ A6 | 检索侧余量裁决 | M3 |
| **M6** | Stage C 校准串联 → 阶段 3 | 最终数字 | M5 |
| **M7** | 全 10 折 LOSO × 3 seeds | 论文级结果 | M6 |

**里程碑门控**：M0 未通过不进 M1；M5 若证明检索侧已无余量则**跳过线性头**继续 M6。

---

## 12. 风险与纪律

| 风险 | 表现 | 对策 |
|---|---|---|
| **冻结瓶颈** | Stage A 空间丢失信息，检索无法补救 | M0 线性探针；不通过则先补 Stage A 训练信号 |
| **误加冗余阶段** | 原 Stage B 是循环论证的产物 | **已取消**（§6）；仅在 M0 显示有显著余量时才加轻量线性头 |
| **目标表征磁盘/GPU** | penultimate 需提取 | `/project` 仅剩 ~59 GB；先提取再决定 |
| **协议不可比** | 把含校准数字与朴素数字并列 | 所有结果表标注档位 |
| **权重互相抵消** | `dec` 与 `cross` 方向相反 | A5 以数值裁决 |
| **概念泄漏** | 同概念当正样本 | 训练 1654 / 测试 200 不重叠；**代码显式断言** |

**必做收尾**：
1. ~~修 `smoke_test.py:525` 置换不变性容差~~ —— **已随条件化删除而消失**：那条断言测的是
   `SupportSetEncoder` 的置换不变性，该模块已移除，断言一并删除，无需重校。v2 的
   `smoke_test.py` 改为断言「模型**没有**任何条件化属性与参数」，见 9.1 / 9.5；
2. `pip freeze > requirements.txt` 冻结；如需 `pyriemann`，安装前
   `export PIP_CACHE_DIR=/project/peilab/why/cache/pip`；
3. 所有写入限定在 `CLIP/`（`third_party/` 只读）。

---

## 13. 参考与可复用资产

### 13.1 检索 SOTA
| 方法 | 出处 | 代码 |
|---|---|---|
| **ATM** | Li et al., NeurIPS 2024 | `ncclab-sustech/EEG_Image_decode`；`MindPilot/model/ATMS_retrieval.py` |
| **SATTC** | CVPR 2026 | `QunjieHuang/SATTC-CVPR2026` |
| **SAMGA** | arXiv 2604.17782 | `LinJiang8/SAMGA` |
| **SVTL** | arXiv 2609.36971 | — |
| **SCORE** | arXiv 2608.19134 | — |
| **PA-NCE / XS-NCE** | *Towards Zero-Shot Cross-Subject Generalization* | — |
| **STAMBRIDGE** | arXiv 2605.23137 | — |
| **ViEEG** | arXiv 2505.12408 | — |
| **CSBrain** | NeurIPS 2025 | — |
| **HyFI** | AAAI 2026, arXiv 2603.22721 | `SangMin316/HyFI` |

### 13.2 范式与组件
- **ENIGMA** — 本地 `/project/peilab/why/ENIGMA/`（per-subject 输入对齐，对照臂）
- **Perceptogram** — `desa-lab/Perceptogram`（线性 EEG→CLIP 足够）
- **EEG-PRIME** — 多级条件化 + GRL（**反面参照**）
- **几何感知蒸馏** — arXiv 2509.25253；RKD arXiv 1904.05068
- **EEG-FM-Bench** — 12 FM × 13 数据集（否决 FM 主线的依据）

### 13.3 本地资产
| 资产 | 位置 |
|---|---|
| XS-NCE + 温度保护（含证明） | `src/samclip/losses/contrastive.py` |
| SAW / CSLS（含秩亏保护） | `src/samclip/calibration.py` |
| MVNN / 坐标恢复 / 多层融合 / 指标 | `/project/peilab/why/eeg-retrieval/scripts/epd/` |
| 预计算多层特征 | `clip_h14_multilevel`、`internvit_multilevel_20_24_28_32_36` |
| 完整重建链路（对照） | `/project/peilab/why/ENIGMA/` |
| 只读第三方仓库 | `third_party/`（EEGiT / CognitionCapturerPro / AVDE） |
