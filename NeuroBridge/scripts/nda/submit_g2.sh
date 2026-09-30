#!/usr/bin/env bash
# Submit the G2 pipeline (sub-08, intra + inter/LOSO, full train->generate->eval).
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
LOG_DIR="${NB_ROOT}/outputs/slurm"
OUT="${OUT:-${NB_ROOT}/outputs/g2}"
mkdir -p "${LOG_DIR}" "${OUT}"

JOBID=$(sbatch "${NB_ROOT}/slurm/g2_sub08.sbatch" | awk '{print $4}')
echo "${JOBID}" > "${OUT}/g2.jobid"

cat > "${LOG_DIR}/g2_sub08.submit.json" <<EOF
{
  "pipeline": "g2",
  "job": "${JOBID}",
  "submitted": "$(date -Iseconds)",
  "script": "${NB_ROOT}/slurm/g2_sub08.sbatch",
  "subject": 8,
  "goal": "Granularity-factorised PARALLEL dual tower with a CFM condition synthesiser, trained and evaluated as BOTH an intra-subject model (fit on sub-08) and an inter-subject LOSO model (fit on the other 9, sub-08 unseen), on identical targets and identical frozen encoder latents.",
  "semantic_tower": "image-level encoding + four VLM-generated description granularities (overall / subject / background / detail), encoded with CLIP ViT-H-14 text",
  "perceptual_tower": "structure = VAE latent restricted to r<0.0625 (92.9% of the recoverable latent variance); texture = log spectral statistics of the VAE latent",
  "condition_synthesiser": "joint dual-tower code -> CFM over 1024-d CLIP image embeddings feeding IP-Adapter; LF latent anchor; depth",
  "ablations": ["cfm0: CFM sample instead of the deterministic mean head",
                "direct_nc0: anchoring disabled (cut=0)",
                "direct_np: empty prompts, removing the oracle-prompt shortcut"],
  "measured_basis": "per-band ceiling probe: LF var_expl 0.00645 vs 0.00049 for everything above r=0.0625 (fine/coarse=0.0765)",
  "note": "structures the perceptual tower as a PARALLEL branch off the frozen latent; the shipped VAE head consumed the semantic head's OUTPUT (z_decode_vith), so the encoder never saw a structural gradient"
}
EOF

echo "[OK] submitted g2 job ${JOBID}"
cat "${LOG_DIR}/g2_sub08.submit.json"
