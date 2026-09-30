# Top-1（干净）× 结构（SSIM）专项方案

> 基于 COCA A+B1 实证（2026-09-04）：干净 Top-1=35% vs CPA=73%；  
> 冠军生成=`depth_cn0.5_ip1.0_cpa`（CLIP=0.490, PixCorr=0.175, SSIM_skimage=0.193）。  
> 目标：干净 200-way Top-1↑；paper SSIM→0.28–0.35，且不明显牺牲 CLIP。

---

## 1. 当前架构缺陷（根因，不是调参）

```
EEG ─► NB/NDA-SS 对齐 ─┬─► 检索（RN50）── 图库协议分裂 ──► Top-1 虚高/虚低
                       └─► mem⊕decode ─► IP
邻居图 ─► DepthAnything ─► Depth-CN ──► 结构代理 ≠ GT 几何 ──► SSIM 天花板
CPA text ─────────────────────────────► 语义强、依赖「不干净」检索
```

| 缺陷 | 表现 | 为何卡住指标 |
|------|------|----------------|
| **D1 测时图库域偏移** | 干净 35% / CPA 73% | 主表 Top-1 与训练评测协议不一致 |
| **D2 结构条件外源** | 邻居 depth，非被试所见刺激 | 邻居错 → 强 CN 锁错几何（cn=0.8 全面崩） |
| **D3 无 EEG→结构头** | 只有语义骨干 | 无法学「脑信号里的布局」 |
| **D4 对齐粒度偏高层** | 主对齐最终层 CLIP | 中层纹理/几何弱 → 干净检索与 SSIM 双损 |
| **D5 条件冲突无最优分配** | 固定 cn/ip 或浅 SCR | 高 \(u_{str}\) 才该强 CN；全局强 CN 伤 CLIP |
| **D6 生成用 CPA、主表要干净** | 双协议混用 | 叙事与审稿风险；干净 prompt 消融缺失 |

**结论：**  
- Top-1 问题 ≈ **编码器对齐目标 + 测试协议**（D1/D4）  
- 结构问题 ≈ **结构来源 + 注入策略**（D2/D3/D5）  
- 两者不应再用同一「邻居代理」糊弄。

---

## 2. 总策略（双轨并行，互不阻塞）

```
轨道 T（Top-1）          轨道 S（Structure）
协议洗干净 + 对齐升级      EEG→depth + COCA 注入
        \                    /
         汇合：干净/高置信 text + 预测 depth → 解码器
```

原则：

1. **主表检索只报干净图库**；CPA 仅训练/附录。  
2. **结构条件最终必须来自 EEG（或高置信才用邻居）**。  
3. **冻结 NDA-SS 语义骨干作 IP**；新训模块限于：对齐补丁、depth 头、COCA router。  
4. **禁止**再把 CFM transport 写回默认生成路径。

---

## 3. 轨道 T：解决干净 Top-1

### T0 — 协议固化（立刻）

- 训练：可保留 CPA 增广。  
- 验证/主表：`gallery = clean RN50 image_test`，`image_test_aug=False`。  
- 产出：每 checkpoint 同时 log `top1_clean` / `top1_cpa`。

### T1 — 对齐目标改造（主杠杆）

| 方案 | 做法 | 预期 |
|------|------|------|
| **T1a 双目标** | InfoNCE：干净图 + CPA 图联合，权重 \(\lambda_{clean}\ge\lambda_{cpa}\) | 测时干净 Top-1↑，少掉 CPA |
| **T1b Shallow/中层** | EEG 对齐 RN50/ViT **中间层**（或 concat 中层+最终层） | 文献：granularity mismatch；利于干净匹配 |
| **T1c 温度/原型** | 200 类测试用概念原型（多视图均值）但仍是干净特征 | 稳定 Top-1 |

优先：**T1a + T1b**（在现有 NB/SSP 上继续训或轻量 fine-tune sub-08）。

### T2 — 生成侧诚实消融

- Prompt A：CPA 检索（现冠军）  
- Prompt B：干净检索  
- 同解码器对比 CLIP/SSIM → 论文必须有  

### T 成功线

- 干净 Top-1：**≥50%**（理想 ≥60%，接近论文 NB 量级）  
- 干净 Top-5：≥80%  
- 同时附录 CPA 不作为主 claim  

---

## 4. 轨道 S：解决结构 / SSIM

### S0 — 尺子（已完成）

- 只用 **skimage SSIM**；弃用 `ssim_simple` 作主指标。

### S1 — 结构教师（数据）

- 全体 train/test 刺激：DepthAnything（已有邻居缓存可扩到全库）。  
- 可选：Canny / HED 作第二结构通道。

### S2 — EEG→Depth 头（核心新模块）

```
z_eeg (NDA-SS 或窗特征)
    → DepthHead (轻量 CNN/MLP→低分辨率 depth)
    → 上采样 512
损失: L1/ Huber(depth, DepthAnything(GT图))
     + 可选 depth-CLIP / 梯度一致性
```

- 训练用 **GT 图像的 depth 伪标签**（被试所见刺激），**不是邻居**。  
- 推理：用预测 depth → Depth-ControlNet。  
- 邻居 depth 仅作：低置信 fallback 或消融。

### S3 — COCA 注入（与 Top-1 置信联动）

\[
u_{txt}=\mathrm{margin}_{clean},\quad
u_{str}=\mathrm{cos}(z_{eeg},\,E(\hat{D}))\ \text{或预测置信}
\]

\[
\alpha_{cn}=f(u_{str}),\ 
\alpha_{txt}=f(u_{txt}),\ 
\alpha_{ip}\approx const
\]

- \(u_{str}\) 低 → 小 cn（防锁错几何）  
- \(u_{txt}\) 低 → 空文本（防错概念）  
- 分时：早 CN、晚 IP/text（规则先，可学习后）

### S4 — 可选低层支路

- 高 \(u_{str}\)：MindEye 式模糊/VAE 与输出加权，专刷 SSIM。  
- 低 \(u_{str}\)：关闭，保 CLIP。

### S 成功线

- paper SSIM：**≥0.28**（冲 0.35）  
- PixCorr：≥0.15（已达标则保持）  
- CLIP：不低于 0.46（相对冠军 0.49 允许小跌）

---

## 5. 汇合后的解码配方（目标系统）

```
冻结: NDA-SS mem⊕decode → IP (α_ip≈1.0, cn 温和区)
新训: DepthHead(EEG→D̂)
新训/规则: COCA(u_txt_clean, u_str)

推理:
  D̂ = DepthHead(EEG)
  txt = CleanRetrieve(EEG) if u_txt 高 else (CPA if 附录设置) else ""
  SDXL( IP(z_gen), DepthCN(D̂; α_cn(u_str)), text(txt; α_txt) )
```

相对现状的关键升级：

| 现在 | 目标 |
|------|------|
| 邻居 DepthAnything | **EEG 预测 depth** |
| CPA prompt 为主 | **干净检索 prompt + 门控**；CPA 附录 |
| 固定 cn=0.5 | **按 \(u_{str}\) 路由** |
| 干净 Top-1 不训 | **T1 双目标/中层对齐** |

---

## 6. 执行排期（建议）

| 周 | 轨道 | 任务 | 产出 |
|----|------|------|------|
| W1 | T0+S0 | 协议脚本、双 Top-1 log、七件套基线卡 | 报表 |
| W1–W2 | S2 | 训 DepthHead（sub-08） | 预测 depth 可视化 + SSIM |
| W2 | S3 | COCA 规则接到预测 depth | vs 邻居 depth 消融 |
| W2–W3 | T1 | 干净+CPA 双目标 / 中层对齐 fine-tune | 干净 Top-1 |
| W3 | 汇合 | 干净 prompt + 预测 depth + COCA | 冠军候选 |
| W4 | E | 多被试冻结配方外推 | 主会表 |

---

## 7. 消融矩阵（论文必备）

1. 邻居 depth vs EEG→depth  
2. 固定 cn vs COCA  
3. CPA prompt vs 干净 prompt vs 空文本  
4. 有/无 DepthHead  
5. 有/无中层对齐（Top-1）  
6. 强制 CFM（负面）  

---

## 8. 风险与止损

| 风险 | 止损 |
|------|------|
| DepthHead 过拟合 1654 类 | 强增强、早停、按 SSIM+CLIP 联合选模 |
| 干净 Top-1 涨、生成掉 | prompt 仍可用 CPA 但主表分报；或门控混合 |
| SSIM↑ CLIP↓ 过大 | 降 α_cn 上限；保持 cn∈[0.35,0.6] |
| 多被试 depth 头不通 | 先 shared trunk + subject adapter |

---

## 9. 一句话

**Top-1：把测试协议拉回干净图库，并用双目标/中层对齐把编码器「拉回」该协议。**  
**结构：丢掉「邻居 depth 当真」的幻想，训 EEG→depth，再用 COCA 只在结构可信时注入。**  
两条轨并行，最后在解码器汇合；不改 NDA-SS 语义骨干，不复活 CFM 默认路径。
