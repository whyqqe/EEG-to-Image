# 架构审视：什么破坏了原性能？如何冲 Oracle？

## 结论先说

**是的——后加模块破坏了 NDA-SS 原有生成性能。**  
NDA-SS `mem⊕decode` 原版 **0.439**；v3/v4「同名」`nda_mem` 无文本掉到 **~0.418**，不是随机波动。

## 被破坏的点

| 改动 | 后果 |
|------|------|
| Memory 改用 RGT `z_ret` 检索 | 与 NDA-SS mem 仅 cos≈0.79，邻居图变了 |
| 强制 CFM 进融合（min_cfm） | 稀释原先更强的 NDA decode |
| 弱 EEG–text 头出 prompt（15% Top-1） | 错概念干扰扩散 |
| 多层运输/适配叠在错误骨干上 | 涨点被「换骨干」抵消 |

**没坏、且被 Oracle 证明有用的**：SDXL **文本条件通路**（GT prompt → **0.454**）。

## 应保留 / 应剥离

```
保留（冲上界）：
  NDA-SS 原版 mem_decode_a50 + 原 neighbor_idx
  + 文本 prompt（尽量接近 GT）

暂时剥离出主路径：
  RGT-CFM 强制融合、z_ret memory、弱 text-head prompt
  （可作为旁路消融，不再当默认骨干）
```

## 冲作弊上界的策略

Oracle = GT 概念名写入 prompt。  
可部署近似：用 **最强检索**（RN50/SSP，原 NB ~50%+ Top-1）在 **200-way 测试图库** 取概念 → 写 prompt，挂回 **原版 NDA-SS 条件**。

目标：预测 prompt 生成 CLIP 从 0.428 → 逼近 **0.454**。
