# THINGS-EEG sub-08：SOTA 与本方法指标对照

> **重要**：不同论文的 CLIP / SSIM / Top-1 **协议不完全相同**，表中数字用于定位量级，**不能直接当同一尺子比绝对高低**。  
> 本方法数字来自本地 `outputs/mg_flow/sub-08`、`outputs/top1_structure/sub-08` 等实测。

---

## 1. 检索（200-way zero-shot Top-1 / Top-5）

协议参考：CogCap / ATM / NICE 等 THINGS-EEG **被试内 200-way** 检索。

| Method | Venue | sub-08 Top-1 (%) | sub-08 Top-5 (%) | 备注 |
|--------|-------|------------------|------------------|------|
| BraVL (Du et al.) | TPAMI'23 | 8.8 | 23.7 | CogCap Table 2 |
| NICE (Song et al.) | ICLR'24 | 20.0 | 49.9 | CogCap Table 2 |
| ATM (Li et al.) | NeurIPS'24 | 38.8 | 72.0 | CogCap Table 2 |
| CognitionCapturer (image) | AAAI'25 | 47.6 | 73.5 | CogCap Table 2 |
| CognitionCapturer (all)* | AAAI'25 | 48.1 | 88.6 | *任一模专家对即算对（上界式） |
| NeuroBridge (论文宣称) | — | ~63.2† | — | †全文/多设定均值，非本仓库同协议复现 |
| **Ours NDA-SS（干净 RN50 gallery）** | — | **35.0** | **65.5** | 主表诚实协议 |
| **Ours dual-ft（干净）** | — | **45.0** | — | `nda_clean_dual_finetune` |
| Ours NDA-SS（CPA gallery） | — | 73.0 | 93.5 | **附录 only**；测时增强图库，不可作主 claim |
| Ours dual-ft（CPA） | — | 65.0 | — | 干净↑则 CPA↓ |

---

## 2. 生成 / 重建（低层 + 高层）

### 2.1 与 CogCap / ATM 主表同口径（Brain-Diffuser 七件套风格）

CogCap / ATM 报告的是 **十被试均值或论文表内 CLIP 特征辨识**，不是我们的 OpenCLIP paired cosine。

| Method | Scope | PixCorr↑ | SSIM↑ | CLIP↑ (其论文定义) | 来源 |
|--------|-------|----------|-------|-------------------|------|
| ATM (Li et al.) | EEG 重建（文中表） | ~0.160 | 0.345 | 0.786 | ATM / CogCap 转引 |
| CognitionCapturer (all) | **10subj 均值** | 0.150 | 0.347 | 0.715 | CogCap Table 3 |
| CognitionCapturer (all) | **sub-08** | **0.175** | **0.366** | **0.744** | CogCap supp. Table 6 |
| CognitionCapturer (image) | sub-08 | 0.154 | 0.327 | 0.748 | CogCap supp. Table 7 |
| MindEye (fMRI, NSD) | 参考上界 | 0.309 | 0.323 | 0.941 | CogCap Table 3（跨模态） |

### 2.2 本方法 sub-08（本地实测，生成主结果）

| Method (Ours, sub-08) | PixCorr↑ | SSIM↑ (skimage) | CLIP cos↑ (OpenCLIP paired) | FID↓ | CLIP 2-way↑ (ViT-L/14) | Class Top-1↑‡ |
|------------------------|----------|-----------------|------------------------------|------|------------------------|---------------|
| `pred_coca_ip1.0_cpa` (旧 sem) | 0.164 | 0.199 | 0.539 | 151.5 | 89.8% | 52.0% |
| **`mg_blend_a40_dual` (当前最佳综合)** | **0.154** | **0.213** | **0.593** | **138.2** | **94.3%** | **73.5%** |
| `mg_gated_dual` (2-way 峰值) | 0.132 | 0.190 | 0.585 | 156.0 | **97.1%** | 71.5% |
| `ts_ll_a45_b70` (结构平衡) | 0.212 | 0.258 | — | 161.6 | 89.6% | — |

‡ Class Top-1：用 CLIP ViT-H 文本类库对生成图做 200-类分类；**prompt 含类名（CPA/dual）**，与纯检索 Top-1 不同。

---

## 3. 我们额外 follow 的语义指标（MindEye / Ozcelik）

许多 EEG 重建论文 **不报** 该指标；MindEye / Brain-Diffuser 系常用。

| Method | CLIP 2-way (ViT-L/14) | 协议 |
|--------|----------------------|------|
| Chance | 50% | — |
| Ours `pred_coca` (sem) | 89.8% | Ozcelik/MindEye：corr(GTᵢ,reconᵢ) vs corr(GTᵢ,reconⱼ) |
| **Ours `mg_blend_a40_dual`** | **94.3%** | 同上 |
| Ours `mg_gated_dual` | **97.1%** | 同上 |
| Ours oracle GT-depth + CPA | 95.1% | 上界参考（非可部署） |

---

## 4. 一句话读表

| 维度 | 相对 SOTA 位置（sub-08） |
|------|-------------------------|
| **检索（干净 200-way）** | 弱于 CogCap image(47.6%) / ATM(38.8% 可打平附近仅 dual-ft 45%)；**CPA 73% 不可主报** |
| **SSIM / PixCorr** | 明显低于 CogCap sub-08（0.366 / 0.175） |
| **语义 2-way / 类别一致（生成）** | 我们很强，但需声明 prompt 协议；与 CogCap 的 CLIP 列 **不同定义** |
| **FID** | 我们有报（a40≈138）；CogCap 主表通常不报 FID |

---

## 5. 引用（指标出处）

- **检索表**：Zhang et al., CognitionCapturer, AAAI 2025, Table 2（含 ATM / NICE / BraVL）。  
- **重建七件套**：CogCap Table 3 + supp. Table 6（sub-08）；ATM (Li et al., NeurIPS 2024)。  
- **CLIP 2-way**：MindEye / Ozcelik & VanRullen (Brain-Diffuser) 辨识协议；实现见 `scripts/nda/eval_clip_2way.py`。  
- **本方法**：`outputs/mg_flow/sub-08/summary.json`，`outputs/top1_structure/sub-08/summary.json`。
