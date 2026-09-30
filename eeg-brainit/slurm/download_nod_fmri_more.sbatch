#!/usr/bin/env bash
#SBATCH --job-name=nod-fmri-dl
#SBATCH --account=peilab
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
cd "$ROOT"
export NOD_FMRI_SUBJECTS=03,04,05,06
export HOME=/home/sbaiae
export XDG_CONFIG_HOME=/project/peilab/why/cache/eeg-brainit/xdg-config
export XDG_CACHE_HOME=/project/peilab/why/cache/eeg-brainit/xdg
export TMPDIR=/project/peilab/why/cache/eeg-brainit/tmp
bash scripts/download_nod_fmri_s3.sh
