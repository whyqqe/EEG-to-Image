#!/usr/bin/env bash
# Submit the single G2 FINAL job.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/g2}"
LOG_DIR="${NB_ROOT}/outputs/slurm"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
EPOCHS="${EPOCHS:-30}"
mkdir -p "${LOG_DIR}" "${OUT}"

JOBID=$(sbatch --parsable \
  --export=ALL,OUT="${OUT}",SUBJECTS="${SUBJECTS}",EPOCHS="${EPOCHS}" \
  "${NB_ROOT}/slurm/g2_final.sbatch")
echo "${JOBID}" > "${OUT}/final.jobid"

cat > "${LOG_DIR}/g2_final.submit.json" <<EOF
{
  "pipeline": "g2_final",
  "job": "${JOBID}",
  "submitted": "$(date -Iseconds)",
  "subjects": "${SUBJECTS}",
  "epochs": "${EPOCHS}",
  "goal": "One job, one controlled experiment: does an INDEPENDENT parallel perceptual tower give a better structural conditioning signal than the proven serial VAE head, with semantics and generator held fixed?",
  "semantic_condition": "HELD FIXED at HCMA blend_nda_cfm_f_a40 (measured cos-to-target 0.5451, vs 0.4026 for G2's own from-scratch semantic head). This makes a semantic regression structurally impossible, so any delta is attributable to the structural anchor.",
  "varied": ["hcma_ll: sdedit_ll's predicted VAE latent (proven predictor)",
             "g2_ll: our parallel perceptual tower (the innovation)",
             "nc0: no anchor (sub-08 only)",
             "g2sem_ll: our own semantic head as IP condition (sub-08 only)",
             "g2_intra_ll: our perceptual tower trained intra-subject (sub-08 only)"],
  "fairness_fix": "both anchors amplitude-equalised to the same low-band std (0.4641, a train-side constant). Native amplitudes differ 3.4x (ours 29% of target, sdedit_ll's 65.7%), so raw comparison would mostly measure gain.",
  "reference": "sdedit_ll's 10 folds are NOT regenerated; their existing images are evaluated in the same pass, so the reference number comes from the same code and GT cache.",
  "metric_fixes": "val-independent dispersion floors (was a single 0.5 for heads differing 21x in target scale), CFM loss at O(1) instead of 1.5e-3, dropout 0.15 + weight decay 1e-2, and selection weighted 0.3 on structural terms so structure cannot outbid the semantic gates.",
  "outputs": ["outputs/g2/standard7_final/results.json",
              "outputs/g2/fid_pooled_g2_ll.json",
              "outputs/g2/fid_pooled_hcma_ll.json",
              "outputs/g2/fid_pooled_sdedit_ll.json",
              "outputs/g2/final_vs_sota.json"]
}
EOF

echo "[OK] submitted g2-final job ${JOBID}"
cat "${LOG_DIR}/g2_final.submit.json"
