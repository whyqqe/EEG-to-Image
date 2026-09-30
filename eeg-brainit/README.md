# EEG-Brain-IT

**Active plan (2026-08):** supervised **EEG → fMRI → Image** on **NOD** (same-subject visual),
with a parallel **THINGS EEG–fMRI CLIP alignment** baseline. THINGS-EEG2 + frozen NeuroBOLT
is retained only as a **negative ablation** (standalone ≈ chance).

ATM retrieval / SDXL generation remain the strong engineering baselines on THINGS-EEG2.

## Target pipeline

**Scientific goal:** same-subject **EEG → fMRI → Image** on NOD (not EEG→CLIP alone).

**Formal Phase-1 = NeuroBOLT (`glb.pth`) fine-tune** on NOD hemi-ROI fMRI.
LaBraM-only and scratch CNN are ablations only. PCA is not used.

```
NOD EEG (62ch → crop 200 @ ~250Hz)
    ▼
NeuroBOLT (glb.pth init: LaBraM-TS + MSS)  →  fine-tune last blocks + MSS + new ROI head
    ▼
predicted L/R cortex hemi-ROI pools   (Phase-1 primary; no CLIP aux)
    │
    ▼  (Phase-2+, after fMRI gates pass)
Brain-IT / OpenCLIP + SDXL  →  Image
```

| Stage | What | Status |
|-------|------|--------|
| Phase-1 MAIN | **NeuroBOLT FT** → NOD hemi-ROI fMRI | **mainline** |
| Ablation | LaBraM-only FT | optional |
| Ablation | Scratch CNN | optional |
| Phase-2 | GT fMRI → Brain-IT/SDXL | later |
| Phase-3 | Predicted fMRI → Image cascade | later |

Auxiliary (Phase 4): align THINGS-EEG2 and THINGS-fMRI into a shared CLIP space (heterogeneous subjects).

## What we keep vs abandoned

| Keep | Abandoned as mainline |
|------|------------------------|
| ATM features + distill S3 (`outputs/atm_distill_*`) | Spec2Vol virtual volume → NSD BIT without paired visual fMRI |
| `outputs/atm_bridge/` teacher cache | THINGS + frozen NeuroBOLT residual/standalone |
| SDXL + IP-Adapter eval stack | Mega stage1–4 / clip_align / nce epoch dumps (cleaned) |
| Brain-IT weights (optional decoder) | |
| Archived metrics in `outputs/archive_phase0_baselines/` | |

Measured numbers: [`configs/literature_baselines.yaml`](configs/literature_baselines.yaml).

## Layout

```
configs/           YAML (baselines, NOD, THINGS align, legacy ATM)
scripts/           prepare_nod_pairs, train/eval, download, activate
slurm/             peilab SuperPOD jobs
src/eeg_brainit/   models (eeg2fmri, atm_bridge, bit, …)
data/nod/          NOD raw/processed (see data/nod/README.md)
data/processed/    THINGS-EEG2 processed
checkpoints/       Brain-IT, ATM prior, NeuroBOLT/LaBraM links
outputs/           runs; nod_eeg2fmri/; things_align/; eval/; archive_phase0_baselines/
third_party/       Spec2Vol + brainit-fmri clones (reference only)
```

## Cluster notes (peilab)

| Item | Value |
|------|--------|
| Account / partition | `peilab` / `preempt` or `normal` |
| Module | `nvhpc-hpcx-cuda12/23.11` |
| Python | 3.11 venv via `scripts/activate.sh` (**no conda**) |
| Cache | `/project/peilab/why/cache/eeg-brainit` (not `/home`) |
| Writes | only under this repo `outputs/`, `data/`, and project cache |

## Quick start (Phase 0 → Phase 1)

```bash
cd /project/peilab/why/eeg-brainit
source scripts/activate.sh

# 1) Inspect frozen baselines
python -c "import yaml; print(yaml.safe_load(open('configs/literature_baselines.yaml'))['local_measured'].keys())"

# 2) Download NOD-EEG (OpenNeuro ds005811) — see commands in scripts/prepare_nod_pairs.py --help
python scripts/prepare_nod_pairs.py --print-download-commands

# 3) After raw data is in place, build pairs + smoke one subject
# python scripts/prepare_nod_pairs.py --build --max-subjects 1
# sbatch slurm/train_nod_eeg2fmri.sbatch
```

Full command list for operators: see the end of this README section **Operator commands**, or ask the agent for the latest list after cleanup.

## Operator commands (copy-paste)

```bash
cd /project/peilab/why/eeg-brainit
module load slurm nvhpc-hpcx-cuda12/23.11
source scripts/activate.sh
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub

# --- Phase 1 data ---
python scripts/prepare_nod_pairs.py --print-download-commands
# follow printed openneuro / aws / datalad commands into data/nod/raw/

python scripts/prepare_nod_pairs.py --scan-raw   # verify layout
python scripts/prepare_nod_pairs.py --build --max-subjects 1

# --- Phase 1 train (after pairs exist) ---
sbatch slurm/train_nod_eeg2fmri.sbatch

# --- Phase 4 aux (can run once THINGS-fMRI path is configured) ---
# sbatch slurm/train_things_align.sbatch

# --- Queue / logs ---
squeue -u $USER
tail -f outputs/slurm/*nod*.out
```

## License / third_party

Third-party code under `third_party/` retains upstream licenses. Brain-IT / ATM / NeuroBOLT weights are for research use per their respective terms.
