#!/usr/bin/env bash
# Record of Phase-0 cleanup (2026-08-08). Safe to re-run: only removes known husks.
set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
cd "$ROOT"
echo "Already cleaned: stage*/clip_align/nce/siglip/direct debug+checkpoints,"
echo "  atm_bridge_sub08 epoch spam, frozen ATM bridge, NB standalone ckpts, last.pt."
echo "Preserved: atm_distill best, atm_bridge cache, eval/, archive_phase0_baselines/,"
echo "  nb_* train_summary + residual best, checkpoints/brain_it + atm prior."
du -sh outputs checkpoints
