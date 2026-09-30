# EEG-to-Image

Code for EEG-to-image decoding / reconstruction experiments (THINGS-EEG2 and related).

## Layout

| Directory | Role |
|---|---|
| `NeuroBridge/` | Main NeuroBridge / UCK pipeline (alignment, conditioning, generation) |
| `loso_pipeline/` | Leave-one-subject-out multi-modal alignment pipeline |
| `eeg-retrieval/` | Feature extraction and retrieval alignment utilities |
| `eeg-brainit/` | BrainIT / ATM-style EEG decoding baselines and scripts |
| `eeg-erdc-repro/` | ERDC reproduction helpers |
| `ref-samga/` | SAMGA reference implementation (zero-shot EEG-to-image retrieval) |

## Notes

- Large artifacts are **not** included: `data/`, `outputs/`, `checkpoints/`, `*.npy`, `*.pt`, caches, logs, virtualenvs.
- Rebuild features and checkpoints from the scripts in each subdirectory.
