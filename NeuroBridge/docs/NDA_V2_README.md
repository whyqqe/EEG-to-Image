# NDA-v2 实验说明（sub-08）

## Semantic 对齐（已修正）

```
z_semantic(EEG/SSP-512)
  ├──↔ CLIP-Image  (RN50 → SSP)   λ_img=0.8   # 检索主信号
  └──↔ CLIP-Text   (ViT-H text)   λ_txt=0.2   # 概念描述正则
Decode bridge (separate):
  z_decode ↔ ViT-H image          # IP-Adapter 条件
Perception:
  z_p ↔ HCF(NVOL mid-CLIP) + DINOv2
```

文本模板（每概念 3 条均值）：
- `a photo of a/an {concept}`
- `a photo of {concept}`
- `Describe only what is directly visible in the image of {concept} in one short sentence`

## 当前作业

- 流水线：`scripts/nda/run_nda_v2_semtxt_sub08.sh`
- 输出：`outputs/nda_v2_semtxt/sub-08/`
