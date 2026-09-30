# 文件清单

生成时间：2026-08-23

## docs/ (6)
- ARCHITECTURE.md
- CONVERSATION_LOG.md
- EXPERIMENT_TIMELINE.md
- EXTERNAL_ASSETS.md
- REPRODUCTION.md
- RESULTS_W14.md

## code/scripts/ (ERDC + eval)
- erdc_*.py (12 files)
- eval_atm_pipeline.py
- download_atm_official_assets.py
- download_controlnet_sdxl.py
- activate.sh

## code/configs/ (6)
- atm_distill_s{1,3,4}_*.yaml
- atm_bridge_frozen_s2_sub08.yaml
- literature_baselines.yaml, base.yaml

## code/slurm/ (7)
- erdc_w{10,12,13,14}_*.sbatch
- eval_atm_pipeline_sub08.sbatch
- download_pretrained.sbatch

## code/src/eeg_brainit/models/ (3)
- atm_bridge.py, atm_diffusion_prior.py, atm_backbone.py

## results/
- w12/w13/w14 freeze txt
- metrics/w{12,13,14}_metrics/*.json
- w12_official_bit_bank_metrics.json
- w13_merged_fuse_l0p15_metrics.json
- logs/erdc-w14-527571.out

## artifacts/
- checkpoints/atm_diffusion_prior/sub-08/diffusion_prior.pt
- embeds/sub-XX_test_eeg_1024.npy (10 subjects)
- embeds/sub-08_bit_clip_1024.npy
- embeds/clip_img_train_1024.npy
- LARGE_CHECKPOINT_PATHS.txt

## 同步到本地（示例）

```bash
# 在本地机器执行（约 140MB）
rsync -avz --progress \
  user@cluster:/project/peilab/why/eeg-erdc-repro/ \
  ./eeg-erdc-repro/
```

如需 S3 ckpt（1.7GB）：
```bash
rsync -avz --progress \
  user@cluster:/project/peilab/why/eeg-brainit/outputs/atm_distill_s3_sub08/checkpoints/atm_stage3_best.pt \
  ./eeg-erdc-repro/artifacts/checkpoints/
```
