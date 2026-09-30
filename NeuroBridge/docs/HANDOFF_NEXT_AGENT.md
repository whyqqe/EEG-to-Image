# EEG 视觉解码项目 —— 交接文档（给下一个 Agent）

> 写于 2026-09-14。本文档的目的是让一个**没有历史上下文**的 agent 能在不重复已证伪路线的前提下继续推进。
> 请先通读第 4 节（已证伪清单）和第 8 节（实验规范）——这两节能避免你浪费最多时间。

---

## 1. 项目目标与当前定位

**目标**：EEG 解码视觉刺激，做到 SOTA 水平，并产出顶会级论文。

**当前双线状态**：

| 方向 | 我们的最好数字 | 该协议下的公开 SOTA | 差距 |
|---|---|---|---|
| **检索**（200-way Top-1） | 0.5350 CSLS / 0.7450 +Sinkhorn（sub-08 单被试） | 78.1% raw / 86.4% CSLS (NVOL) | **大，且未解释** |
| **生成**（sub-08） | PixCorr 0.1667 / SSIM 0.2300 | PixCorr 0.1668 / SSIM 0.4264 (ENIGMA) | PixCorr 持平，**SSIM 单点差 0.20** |

**10 被试权威基线**（`outputs/cfmsf_all/aggregate.json`，`frozen` 臂，rule=`lvl5+agg`）：

| 指标 | mean | std | min | max |
|---|---|---|---|---|
| `fuse_csls_top1`（**inductive，主指标**） | **0.3815** | 0.0731 | 0.295 | 0.520 |
| `fuse_csls_top5` | 0.7335 | 0.0626 | 0.635 | 0.825 |
| `fuse_csls_sinkhorn_top1`（transductive） | **0.5445** | 0.0865 | 0.420 | 0.680 |
| `best_single_route_top1` | 0.3065 | 0.0585 | — | — |

> `sinkhorn` 列是 **transductive** 的（用了测试集"200 个概念互斥"这一结构）；`csls` 列才是 **inductive**。汇报时要区分，不能混为一谈。
> `joint` 臂对应值：0.3440 / 0.6790 / 0.4715 / 0.2660 —— **全面更低**。

**关键判断**：生成线不是全面落后，只差一个指标；检索线差 2 倍且原因未知。**检索线的 2 倍差距是当前最大的未解问题**，在解释它之前做架构创新是无意义的。

---

## 2. 数据集与评测协议（已逐项核验，可直接信任）

数据集：THINGS-EEG2，`sub-08` 做单被试，`shared_r` 做 10 被试。

**已核验为与文献一致**（`data/things_eeg/preprocessed_eeg/info.json` 是权威来源）：

| 项 | 我们 | 状态 |
|---|---|---|
| 归一化 | `mvnn` | ✅ 与文献一致（曾怀疑缺失，**已证伪**） |
| 采样率 / 时间窗 | 250 Hz / 0–1000 ms | ✅ |
| 训练集 | 1654 概念 × 10 图 × 4 重复，**平均后 16540 对** | ✅ |
| 测试集 | 200 概念 × 1 图 × 80 重复，**平均后 200 行** | ✅ (`shared_r_test.npy` 是 `(200,1024)`，每概念一行) |
| 协议 | 官方 200-way，测试集全部充当 query 和候选 | ✅ |
| **电极数** | **17 / 63** | ❌ **唯一的实质偏差**（见第 5 节） |

**2-way identification 度量自查**（`scripts/nda/eval_standard7.py::two_way`）：
- chance = **0.4942 / 0.5045 ≈ 0.50** ✅（是"对 + 199 个干扰项平均"，不是 200-way 的 0.005）
- 完美重建 = **1.0000** ✅
- ⚠️ **但 `0.5·真 + 0.5·噪声 = 1.0000`** → 顶部极易饱和。**不要用 2-way 的小差异做选择或过度解读**（这解释了历史上 `margin`/`two_way` 作为选择器为何失效）。

**生成评测脚本同源已验证**：我们的 PixCorr 0.1667 与独立复现的 0.166 ± 0.013 **精确一致**，说明评测代码与文献同源，数字可信。

### 公开 SOTA 参考值（用于对齐）

- **ENIGMA** (arXiv 2602.10361)：PixCorr 0.1668、SSIM 0.4264、AlexNet-5 89.12%、Inception 76.54%。多被试 SOTA。
- **同协议独立复现** (arXiv 2605.23996)：PixCorr 0.166±0.013、SSIM 0.409±0.005、Alex2 0.818、Alex5 0.913、Incep 0.831、CLIP-H/14 0.903。**其消融表最关键**：

  | 配置 | 200-way Top-1 |
  |---|---|
  | 无 blur、无 EVNet（**普通编码器基线**） | **61.2%** |
  | 8-blur（多层级扰动目标） | 82.8% |
  | 8-blur + EVNet | 85.3% |

- **NVOL** (arXiv 2609.02582)：78.1% raw / 86.4% +CSLS。**注意：NVOL 已经做了"多层级目标 + CSLS + 检索/生成耦合"，所以架构本身不构成新颖性主张。**
- **CORTIVA** (arXiv 2608.01355)：73.5%，但**不是 LOSO**。
- **HCF** (arXiv 2603.07077)：84.6% zero-shot。
- **SCORE**：LOSO SOTA **53.23%**。
- **SATTC** (CVPR 2026)：跨被试 hubness 校准。

---

## 3. 当前架构：CF-MSF

**Concept-Factorized Multi-level Score Fusion**：多条"概念级路线"各自做 EEG→目标空间的对齐，在**分数级**融合，再上 CSLS + Sinkhorn。

```
EEG → [frozen intra 编码器] → r (1024-d)
                              ├─ route 1: → ViT-H-14 image CLIP        ─┐
                              ├─ route 2: → GaussianBlur CLIP           │
                              ├─ route 3: → LowResolution CLIP          │  13 条
                              ├─ route 4: → Mosaic CLIP                 │  legacy
                              ├─ route 5: → GaussianNoise CLIP          │  路线
                              ├─ ... (depth/edge/text/rn50/mixall/      │
                              │       levels_mean/cat3/cat5)           ─┘
                              ↓
                  每条路线一个 MLP head（gallery NCE 训练）
                              ↓
                  分数矩阵 → 选 top-k 路线 → 求和 → CSLS → Sinkhorn
```

**已验证有效的组件（有统计证据）**：
- **多层级扰动目标路线银行**：`vith_cat5` 单路 0.4050（vs 普通 `img_clip` 0.3050）。**方向被文献独立验证（+21 点）**，但我们执行落后约 44 点（38% vs 82.8%）。
- **CSLS**：0.3815 → 0.4900（+0.11）
- **Sinkhorn**（利用"200 概念互斥"约束）：0.4900 → 0.6650（+0.18）
- **分数级融合**：优于最佳单路，且 20/20 条路线对比为正
- **路线银行的必要性**：10 被试中有 6 种不同目标类型当过"最佳单路"，9/10 是多层级聚合目标 → **单一目标类型不够**

---

## 4. ⚠️ 已证伪清单 —— 不要重做这些

以下每一项都花过真实算力，结论是**否定的**，请勿重试：

| 假设 | 结论 | 证据 |
|---|---|---|
| **端到端联合微调编码器**（`joint` 臂）更好 | ❌ **明确更差** | 10 被试 `joint` **0/10 胜**，`frozen` 10/10 胜，Wilcoxon exact p=**0.00195**，mean_delta −0.0375。`frozen`（编码器冻结）全面更优 |
| `val_top1` 是可靠的选择器 | ❌ Spearman 仅 **+0.379**，动态范围极小 | 用它选 epoch 等于接近随机选（`best_epoch` 分布 min=1 max=71 median=9/80） |
| 选择器分辨率是瓶颈（"提高分辨率能拿回大部分收益"） | ❌ **证伪** | `mini_top1` 只拿回 full→oracle 差距的 **1%**（+0.0006, p=0.50） |
| `margin` 可作为选择器 | ❌ **与泛化反相关**，单调递减 | 最大化它导致选到 epoch 0（几乎未训练的头） |
| `two_way` 作为选择器分辨率高 | ❌ 严重饱和 | mean 0.9185，**86.4% 的 epoch 落在最大值 1% 内** → argmax 不可靠 |
| MLP 是过参数化的，线性模型够用 | ❌ Ridge 差 **3.76×** | EEG→视觉空间强非线性 |
| `Proj` 头在 `frozen` 臂过拟合是瓶颈 | ❌ **假问题** | 该脚本导出的是原始 `r`，**`Proj` 头被丢弃**。`frozen` 臂输出与原始 intra 编码器逐位相同 |
| 编码器是"训练不足" | ❌ 7/10 被试的最佳 `calib_top1` 出现在**前 3 个 epoch** | 是早停而非欠训练 |
| 评测协议不一致（导致 2 倍差距） | ❌ **已核验一致** | MVNN ✓、250Hz ✓、16540 ✓、200 行 ✓、官方 200-way ✓、chance=0.50 ✓ |
| 17 通道（`POSTERIOR_17`）很关键 | ❌ **无显著效果** | 见第 5 节，job 582523 已闭合此问题 |
| 项目里有更强的检索实现可归咎 | ❌ 没有 | 独立实现的 ATM 线在 sub-08 只有 **20.5%**，比 CF-MSF 还差 |

**另外两个曾被发现但已失效/未使用的资产**（别误以为它们无效）：
- `cfmsf_route_probe.py` 里的 expanded route bank（`--banks`）**从未跑过**（`grep --banks` 在所有日志中无命中）。
- `cfmsf_fix_train.py` 的输出目录是空的（该作业崩在设备不匹配，之后被 `cfmsf_sel` 取代）。

---

## 5. 最新实验：通道消融（job 582523，已完成）

**动机**：17 个枕顶电极是从早期原型继承的默认值，**项目里从未与全量 63 通道对比过**，而文献明确保留全部电极（"All electrodes were preserved for analysis"）。这是唯一已知的、未测过的自由度。

**实验设计（严格单变量）**：
- 17 个后部电极正好是 63 通道顺序的**末尾连续块 46–62** → 可以做到**数学上精确的权重膨胀**（`ss_modules.py::dilate_first_linear`）
- 已**三重验证膨胀精确**：
  1. 进程内单元测试：max diff 3e-05
  2. **真导出器端到端**：相对误差 **2.4e-06 (train) / 4.4e-07 (test)**
  3. 负控制：置换两个电极 → 输出变化 2.404（证明测试非空）
- 两条臂：`c63_warm`（膨胀 warm start）、`c63_fresh`（从零），预算与原基线的 30+15 epoch **完全一致**

**结果**：

| 臂 | 最佳单路 | Top-1 | CSLS |
|---|---|---|---|
| baseline_17ch | vith_cat5 | 0.4050 | 0.4950 |
| c63_warm | vith_levels_mean | 0.4150 | 0.4950 |
| c63_fresh | vith_mixall | 0.3850 | 0.5000 |

**逐路配对比较（统计上唯一站得住的比较，13 条路线各自对自己）**：

| 臂 | 指标 | 平均增量 | 胜出 | Wilcoxon p | 符号检验 p |
|---|---|---|---|---|---|
| c63_warm | top1 | **+0.0019** | 7/13 | 0.587 | 1.000 |
| c63_warm | top1_csls | +0.0100 | 7/13 | 0.553 | 1.000 |
| c63_fresh | top1 | −0.0042 | 6/11 | 0.619 | 1.000 |
| c63_fresh | top1_csls | +0.0004 | 5/12 | 0.607 | 0.774 |

**结论：通道数无显著效果 → 23 点差距不在电极数量上。此问题已闭合，改用 17 通道即可（更省）。**

**但这次运行意外发现了另一个真信号 —— 选择器比通道重要得多**：
- 在**同一个探针内**比较不同选路规则选出的路线，其**各自的 test top1 均值**：
  - 用 `val_top1` 选 → 均值 **0.3550**
  - 用 `mini_csls` 选 → 均值 **0.3862**
  - 用 `two_way` 选 → 均值 **0.3838**
  - 用 `mini_top1` 选 → 均值 **0.3900**
- 融合结果随之变化：`val_top1` → csls 0.5150；`two_way` → **csls 0.5350 / +sink 0.7450**（本方向目前最好数字）
- 这是 **val 选择 / test 评测**的干净held-out信号，不是巧合。

（注意：这与第 4 节"`two_way` 饱和"并不矛盾——饱和损害的是**选 epoch**，选**路线组合**时它的排序信息反而有用。）

**发现的一个报告 bug（已修）**：汇总脚本原来查裸键名 `"mlp"`，而基线探针之后的探针写的是 `"mlp|by=..|k=.."`，导致一次成功的运行打印出 `nan`。现已兼容两种键名。

---

## 6. 核心未解问题（按重要性）

### 问题 1（最高优先级）：检索侧 2 倍差距无法解释
- 我们：0.5350 CSLS / 0.7450 +Sinkhorn
- 文献：NVOL 78.1% raw / 86.4% CSLS
- **同协议独立复现的"普通编码器基线"就已 61.2%**，而我们的最佳单路只有 0.4050
- 通道已排除（第 5 节）。协议已核验一致。内部没有更强实现。
- **这意味着差距在编码器架构或读出目标上，不在融合规则上。**

### 问题 2：编码器目标轴错配（r = 0.83 的一阶杠杆）
- 编码器自身的 `calib_top1` 与下游 CSLS+Sinkhorn 强相关：**Pearson +0.834 / Spearman +0.806** → 编码器质量是一阶变量
- 但编码器训练用 1280-dim `sem_image` 目标，而路线银行用 **1024-dim ViT-H-14** → 维度/语义轴不一致
- 编码器**从未**在扰动级目标（GaussianBlur / LowResolution / Mosaic / GaussianNoise）上训练过，而**恰恰是这些目标在下游表现最好**
- 参考：同协议文献显示多层级目标可带来 **+21 点**（61.2% → 82.8%）

### 问题 3：生成侧 SSIM 单点缺口
- 我们 0.2300 vs 0.4264（ENIGMA）。PixCorr 精确持平、Inception 甚至更高
- **机制已知**：语义条件挤压结构信息。多分支 IP-Adapter 在 scale 1.0 时把 UNet 推出训练分布
- 修复方向：结构信息走**空间通道**（ControlNet / init），IP 通道保持语义且**归一化总质量**
- 旁证：arXiv 2510.26391 用"双条件 + ControlNet 空间控制分支"取得显著 PixCorr/SSIM 提升

### 问题 4：新颖性定位
NVOL 已占"多层级目标 + CSLS + 检索/生成耦合"。**架构本身不再是新颖性来源**。可能的差异化：
- 学习式的**路线银行选择**（相对人工固定 top-k）
- 跨被试（LOSO，SCORE 53.23%）
- **更严格的协议本身**（若我们的协议确实更严）

---

## 7. 代码地图

### 核心脚本（都在 `NeuroBridge/scripts/nda/`）

| 脚本 | 作用 |
|---|---|
| `ss_modules.py` | `SharedSpecificEncoder` 架构 + **`POSTERIOR_17` / `CHANNEL_SETS` / `dilate_first_linear` / `channel_indices`** |
| `nda_ss_pretrain.py` | 编码器训练。有 `--channels {posterior,all}`、`--init-ss-checkpoint`（精确膨胀 warm start）、`--warm-start-only` |
| `ocf_export_intra_z.py` | 导出 `shared_r_{train,test}.npy`。有**电极守卫**（拒绝与 checkpoint 不符的 `--channels`） |
| `cfmsf_route_probe.py` | **路线银行探针**（核心实验脚本）。目标空间定义、MLP/Ridge 头、选择器、融合扫描 |
| `cfmsf_train.py` | 单路线头训练 + `SELECTORS` / `SELECTOR_KEY` / 饱和审计 |
| `cfmsf_fuse_eval.py` | 融合 + CSLS + Sinkhorn 评测 |
| `eval_standard7.py` / `eval_official_seven_dir.py` | 七指标 2-way 评测（PixCorr/SSIM/Alex2/Alex5/Incep/CLIP/SwAV/FID）|
| `cfmsf_aggregate.py` | 10 被试汇总 + Wilcoxon / 符号检验 |
| `channel_ablation_audit.py` | 通道膨胀正确性审计（硬门，含负控制） |
| `device_audit.py` | 静态设备审计（防止 CPU 张量被 GPU 索引；曾因 LoRALinear 漏检出过 bug） |

### 编排 / 提交

| 脚本 | 说明 |
|---|---|
| `run_cfmsf_all.sh` + `submit_cfmsf_all.sh` + `slurm/cfmsf_all.sbatch` | 10 被试全流程。**`SUMMARY_ONLY=1` 可无 GPU 重生成汇总** |
| `run_chab_sub08.sh` + `submit_chab_sub08.sh` + `slurm/chab_s08.sbatch` | 通道消融（已完成）。支持 `SUMMARY_ONLY=1` |
| `run_cfmsf_probe_sub08.sh` | 路线探针单被试 |

### 关键产物路径

```
outputs/ocf/intra_enc/sub-XX/checkpoint_ss_calib_best.pth   # 编码器 checkpoint
outputs/ocf/intra_z/sub-XX/shared_r_{train,test}.npy        # 编码器导出 (16540,1024)/(200,1024)
outputs/cfmsf_all/sub-XX/frozen/probe/route_probe.json      # 基线探针（job 581704）
outputs/chab/sub-08/probe_c63_{warm,fresh}/route_probe.json # 通道消融探针（job 582523）
outputs/chab/sub-08/logs/equivalence.log                    # 端到端等价性证据
outputs/chab/sub-08/summary.txt                              # 通道消融裁决
outputs/slurm/*.out                                          # 作业日志
```

---

## 8. 实验规范（必须遵守）

这套规范是这个项目积累下来的最重要的东西，**不要为了快而绕过**：

1. **Leak-free 划分**：`fit`（梯度）/ `val_a` / `val_b`（选择）/ **test 只在最后评测一次**。`outputs/leakfree/split.json` 是权威。
2. **绝不按 test 指标选 checkpoint**。历史上因"test-based selection"污染过一次 VAE head，脚本里现在有硬断言拒绝 `selected_on != "val"`。
3. **预登记判据**：实验脚本头部写明"什么算成功/失败"的**阈值**，在看到数字**之前**。已修复的作业里有阈值断言防止事后改动。
4. **每个断言都要有负控制**。一次真实教训：负控制因切片到空张量而"通过"，实际什么都没测（已在 `channel_ablation_audit.py` 修复并加注释说明）。
5. **静默失败是这个项目的头号风险**。编码器输入是打平的权重矩阵 → 喂错电极顺序**不会报错**，只会输出貌似合理的垃圾。任何跨 montage / 跨维度的操作都要显式守卫。
6. **配对统计而非比较最大值**。"最佳单路"是两个 13 维噪声向量的最大值比较，天然带 winner's curse。要比就**逐路配对**。
7. **作业可续跑**：每个阶段在有产物时 `[SKIP]`。被墙钟杀掉只损失当前阶段。

---

## 9. 环境与运行

```bash
# 解释器（唯一正确的那个）
/project/peilab/why/eeg-brainit/.venv/bin/python

# 缓存必须重定向（home 配额小，历史上出现过 No space left on device）
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/eeg-brainit/open_clip
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
```

- 集群：Slurm，`--partition=normal --account=peilab --gres=gpu:1`，`--exclude=dgx-09,dgx-11,dgx-17,dgx-30`
- **登录节点无 GPU**：不能跑训练/导出。但 `SUMMARY_ONLY=1` 的汇总不需要 GPU。
- 提交方式：`bash scripts/nda/submit_<exp>.sh`（内含预检 + CPU 冒烟 + 自动 `sbatch`）
- 实测速度参考：编码器 30+15 epoch ≈ **6 分钟**（比早期估的 22 分钟快得多）；13 路线 × 80 epoch 探针 ≈ **数分钟**

---

## 10. 下一步建议（按优先级）

### P0：解释检索侧的 2 倍差距（**在它之前做别的都是浪费**）
已有证据指向**编码器**（通道已排除，协议已核验，`calib_top1` 相关性 r=0.83）。具体可选：
- **对齐目标轴**：把编码器训练目标从 1280-dim `sem_image` 换成路线银行用的 **1024-dim ViT-H-14**
- **加入扰动级目标**：把 GaussianBlur / LowResolution / Mosaic / GaussianNoise 显式加进编码器目标（文献显示此方向 **+21 点**）
- 参考"普通编码器基线 61.2%"这一条：我们的 `calib_top1` 只有 25–33%，**编码器本身就比文献基线弱一倍**

### P1：把"选择器 > 通道"这个新发现坐实并变成收益
- `two_way` / `mini_top1` 选路比 `val_top1` 明显好（0.3838/0.3900 vs 0.3550）
- 需要：**跨 10 被试验证**（单被试不够），并确认不是 val_b 过拟合
- 若成立，这是"学习式路线银行选择"这条新颖性主张的实证基础

### P2：生成侧 SSIM 缺口
- 结构走空间通道（ControlNet / init），语义走 IP 通道且归一化总质量
- 只差这一个指标就进入可比 SOTA 表，且机制已知

### P3：跨被试（LOSO）
- 参考 SCORE 53.23%、SATTC（hubness 校准）
- **不要在 P0 解决前开始**

---

## 11. 一句话总结

**方向是对的**（多层级目标被文献独立验证 +21 点，我们的路线银行也独立发现了同一点），**执行落后约 44 点**；通道消融已排除电极数量这个最便宜的解释，下一个必须查的是**编码器目标轴**。同时，融合侧的收益主要来自**选择器**而非通道，这是一个值得坐实的新信号。

**最危险的事**：在解释 2 倍检索差距之前就去做架构创新 —— 那是在一个被削弱的底座上做优化。
