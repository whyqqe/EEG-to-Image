# 安全下载说明

## 推荐：下载单个压缩包（避免路径错误）

服务器路径：
```
/project/peilab/why/eeg-erdc-repro.tar.gz
```

| 属性 | 值 |
|------|-----|
| 大小 | 约 103 MB |
| SHA256 | `5f6b479f57b917ba5c9678d7d8b65e35229c82c863f4f92dfd4ee2bb8880fee8` |

---

## Windows（PowerShell）

```powershell
# 1. 进入目标目录
cd D:\my_files\FYP

# 2. 下载（把 USER 和 HOST 换成你的集群账号和地址）
scp USER@HOST:/project/peilab/why/eeg-erdc-repro.tar.gz .

# 3. 校验（可选）
Get-FileHash eeg-erdc-repro.tar.gz -Algorithm SHA256

# 4. 解压（需要 tar，Windows 10+ 自带）
tar -xzf eeg-erdc-repro.tar.gz

# 5. 确认 models 目录
dir eeg-erdc-repro\code\src\eeg_brainit\models
# 应看到：atm_bridge.py, atm_diffusion_prior.py, atm_backbone.py
```

## WSL / Git Bash

```bash
cd /d/my_files/FYP
scp USER@HOST:/project/peilab/why/eeg-erdc-repro.tar.gz .
sha256sum eeg-erdc-repro.tar.gz
# 应等于 5f6b479f57b917ba5c9678d7d8b65e35229c82c863f4f92dfd4ee2bb8880fee8
tar -xzf eeg-erdc-repro.tar.gz
ls eeg-erdc-repro/code/src/eeg_brainit/models/
```

## WinSCP（图形界面，最安全）

1. 连接集群 SFTP
2. 进入 `/project/peilab/why/`
3. **只下载** `eeg-erdc-repro.tar.gz`（不要拖整个 eeg-brainit）
4. 本地用 7-Zip 或 `tar -xzf` 解压

---

## 不要这样做

- 不要复制文档里的 `cp -a ...` 命令到 Windows
- 不要同步整个 `eeg-brainit`（>50GB）
- 不要用会把 shell 命令拼进路径的脚本

---

## 可选：S3 编码器 ckpt（+1.7GB）

```powershell
scp USER@HOST:/project/peilab/why/eeg-brainit/outputs/atm_distill_s3_sub08/checkpoints/atm_stage3_best.pt `
  D:\my_files\FYP\eeg-erdc-repro\artifacts\checkpoints\
```
