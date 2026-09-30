# MG-Flow 实验计划（AAAI 导向）

**方法**：Multi-Granular Dual-Stream Alignment + Gated Hierarchical CFM  
**原则**：CFM **不**强制替换 `mem⊕decode`；仅作门控残差；主指标 = CLIP 2-way + 类别一致率 + FID。

## 本轮范围（sub-08，可提交）

1. 构建粗/细语义文本目标（复用 concept / flat；可选加强细模板）
2. 训练语义双头 + CFM_s^c / CFM_s^f + CFM_{c→f}
3. 门控融合：`z = normalize(mem + g·α·(cfm_fine − mem))`
4. 生成：与 `pred_coca_ip1.0_cpa` 同解码器，换 embed / 双粒度 prompt
5. 评测：paper metrics + CLIP 2-way + 类别一致率 + compare grid

## 资产（本地已有，尽量不新下大模型）

- `nda_ss/.../z_ret` / `z_decode_vith` / `mem_decode_a50`
- `clip_text/.../text_concept_clip.npy`（粗）、`text_flat_clip.npy`（细）
- SDXL + IP + Depth CN + OpenCLIP ViT-H（cache）

## 明确不做

- 无门控强制 RGT-CFM 进默认生成
- 本轮不强制下载 BLIP/LLaVA（磁盘 99%）；细文本用 flat+模板增强
