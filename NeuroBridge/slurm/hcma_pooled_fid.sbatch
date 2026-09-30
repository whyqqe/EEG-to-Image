#!/usr/bin/env bash
#SBATCH --job-name=hcma-pooled-fid
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.err

set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh"
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
OUT="${NB_ROOT}/outputs/hcma_10subj"
python "${NB_ROOT}/scripts/nda/eval_pooled_fid.py" \
  --root "${OUT}" \
  --tag hcma_full_a40 \
  --output-json "${OUT}/metrics_pooled_fid.json" \
  --batch-size 32 \
  --device cuda:0

# merge into summary_all.json + SOTA_COMPARE.md
python - <<'PY'
import json
from pathlib import Path
root = Path("/project/peilab/why/NeuroBridge/outputs/hcma_10subj")
fid = json.loads((root / "metrics_pooled_fid.json").read_text())
agg_path = root / "summary_all.json"
agg = json.loads(agg_path.read_text()) if agg_path.is_file() else {}
agg["pooled_fid"] = {
    "pooled_fid_unique_gt": fid["pooled_fid_unique_gt"],
    "pooled_fid_tiled_gt": fid["pooled_fid_tiled_gt"],
    "mean_per_subject_fid": fid["mean_per_subject_fid"],
    "std_per_subject_fid": fid["std_per_subject_fid"],
    "n_fake_total": fid["n_fake_total"],
    "n_gt_unique": fid["n_gt_unique"],
    "protocol": fid["protocol"]["sota_primary"],
}
if "ours_mean" in agg:
    agg["ours_mean"]["fid_pooled_sota"] = fid["pooled_fid_unique_gt"]
    agg["ours_mean"]["fid_mean_per_subject"] = fid["mean_per_subject_fid"]
agg_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
md = root / "SOTA_COMPARE.md"
text = md.read_text() if md.is_file() else "# HCMA 10-subject vs SOTA\n"
block = (
    "\n## Pooled FID (SOTA distribution-level)\n"
    f"| Variant | Value |\n|---|---|\n"
    f"| **pooled FID (unique GT, primary)** | **{fid['pooled_fid_unique_gt']:.2f}** |\n"
    f"| pooled FID (tiled GT) | {fid['pooled_fid_tiled_gt']:.2f} |\n"
    f"| mean of per-subject FID | {fid['mean_per_subject_fid']:.2f} |\n"
    f"\nProtocol: fake=all {fid['n_fake_total']} gens across 10 subjects; "
    f"real={fid['n_gt_unique']} unique test GT; torchmetrics Inception-v3 FID.\n"
)
if "## Pooled FID" in text:
    pre, _, rest = text.partition("## Pooled FID")
    # drop old section through next ## or EOF
    nxt = rest.find("\n## ")
    text = pre + block[1:] + (rest[nxt+1:] if nxt >= 0 else "")
else:
    text = text.rstrip() + "\n" + block
md.write_text(text, encoding="utf-8")
print("[OK] updated summary_all.json and SOTA_COMPARE.md")
print(f"TOTAL FID (SOTA pooled) = {fid['pooled_fid_unique_gt']:.2f}")
PY
