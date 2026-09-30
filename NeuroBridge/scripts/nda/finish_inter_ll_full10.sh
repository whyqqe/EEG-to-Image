#!/usr/bin/env bash
# Finish inter_ll_full10: pooled FID (GPU) + FT-vs-inter compare table + housekeeping.
# Intended to run inside an interactive GPU allocation (srun/sbatch).
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/inter_ll_full10"
STD7="${NB_ROOT}/outputs/standard7_protocol"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

cd "${NB_ROOT}"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" || true

echo "===== [A] Pooled FID (inter_ll full10, GPU) @ $(date -Iseconds) ====="
"${PYTHON}" scripts/nda/eval_pooled_fid.py \
  --root "${OUT}" \
  --tag inter_ll \
  --images-root /project/peilab/why/data/images_set \
  --output-json "${OUT}/metrics_pooled_fid_inter_ll.json" \
  --device "${DEVICE}" \
  --batch-size 32

echo "===== [B] Compare table (FT vs inter zero-shot) @ $(date -Iseconds) ====="
# FT rows live in shared results.json (35 rows incl. sdedit_ll_subXX); inter rows in results_inter.json.
"${PYTHON}" - <<'PY'
import json
from pathlib import Path
OUT = Path("/project/peilab/why/NeuroBridge/outputs/inter_ll_full10")
STD7 = Path("/project/peilab/why/NeuroBridge/outputs/standard7_protocol")
ft = json.loads((STD7 / "results.json").read_text(encoding="utf-8"))["rows"]
it = json.loads((OUT / "results_inter.json").read_text(encoding="utf-8"))["rows"]
by_ft = {r["tag"]: r for r in ft}
by_it = {r["tag"]: r for r in it}
keys = ["pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "swav", "fid"]

def avg(rows, key):
    vals = [r[key] for r in rows if key in r and r[key] is not None]
    return sum(vals) / len(vals) if vals else None

ft_rows = [by_ft[f"sdedit_ll_sub{s:02d}"] for s in range(1, 11) if f"sdedit_ll_sub{s:02d}" in by_ft]
inter_rows = [by_it[f"inter_ll_sub-{s:02d}"] for s in range(1, 11) if f"inter_ll_sub-{s:02d}" in by_it]
print(f"n_ft={len(ft_rows)} n_inter={len(inter_rows)}")

pfp = OUT / "metrics_pooled_fid_inter_ll.json"
pfv = None
if pfp.is_file():
    pf = json.loads(pfp.read_text(encoding="utf-8"))
    pfv = pf.get("pooled_fid_unique_gt")

lines = [
  "# Zero-shot inter-subject (`inter_ll`) vs per-subject FT (`sdedit_ll`) — 10-subject avg",
  "",
  "Protocol: standard-7 (Ozcelik/ATM/MindEye family) + pooled FID. Decode recipe identical "
  "(LL-RGB init from the same per-subject VAE structure expert + HCMA prompts, s=0.82).",
  "The ONLY difference is the semantic embed source:",
  "- `sdedit_ll` (FT): per-subject FT of the MG-Flow semantic tower on each subject.",
  "- `inter_ll` (ZERO): LOSO-fold MG-Flow ckpt (9-subject pretrain, subject never seen) pure forward, NO FT.",
  "",
  "| Metric | FT 10-subj avg | inter zero-shot avg | delta |",
  "|---|---:|---:|---:|",
]
out_rows = []
for k in keys:
    f, i = avg(ft_rows, k), avg(inter_rows, k)
    d = (i - f) if (f is not None and i is not None) else None
    fs = f"{f:.4f}" if f is not None else "—"
    is_ = f"{i:.4f}" if i is not None else "—"
    ds = f"{d:+.4f}" if d is not None else "—"
    if k == "fid":
        fs = f"{f:.2f}" if f is not None else "—"
        is_ = f"{i:.2f}" if i is not None else "—"
        ds = f"{d:+.2f}" if d is not None else "—"
    lines.append(f"| {k} | {fs} | {is_} | {ds} |")
    out_rows.append({"metric": k, "ft_avg": f, "inter_avg": i, "delta": d})
lines.append(f"| pooled FID | 131.34 | {pfv:.2f} | {pfv-131.34:+.2f} |" if pfv else "| pooled FID | 131.34 | — | — |")
lines += ["", "## 判定标准",
          "- 语义指标(CLIP/Alex2/5/Inception/SwAV): |Δ|<0.01 → 零样本 inter 语义不输 FT，去掉 FT 仍达同一语义层级。",
          "- FID/结构指标(Pix/SSIM): 同 LL init 下主要由语义 embed 影响 → 同样可判定。",
          "",
          "## Per-subject: FT vs inter"]
hdr = "| subject | " + " | ".join(k for k in keys) + " |"
lines += [hdr, "|---|" + "---|" * len(keys)]
for s in range(1, 11):
    stag = f"sub-{s:02d}"
    ftr, itr = by_ft.get(f"sdedit_ll_sub{s:02d}"), by_it.get(f"inter_ll_sub-{s:02d}")
    if not ftr or not itr:
        continue
    cells = []
    for k in keys:
        fv, iv = ftr.get(k), itr.get(k)
        if k == "fid":
            cells.append(f"{fv:.2f}/{iv:.2f}" if fv is not None and iv is not None else "—/—")
        else:
            cells.append(f"{fv:.3f}/{iv:.3f}" if fv is not None and iv is not None else "—/—")
    lines.append(f"| {stag} (FT/inter) | " + " | ".join(cells) + " |")
lines += ["", "*每格格式 `FT值 / inter值`；inter 行 = LOSO fold ckpt 纯前向(该受试完全未见)。*"]
(OUT / "FT_VS_INTER_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote", OUT / "FT_VS_INTER_TABLE.md")
json5 = {"n_ft": len(ft_rows), "n_inter": len(inter_rows),
         "ft_avg": {k: avg(ft_rows, k) for k in keys},
         "inter_avg": {k: avg(inter_rows, k) for k in keys},
         "pooled_fid_inter": pfv}
(OUT / "ft_vs_inter.json").write_text(json.dumps(json5, indent=2), encoding="utf-8")
print("wrote", OUT / "ft_vs_inter.json")
print("FT_VS_INTER table summary:")
for o in out_rows:
    print(f"  {o['metric']:>10s}: FT={o['ft_avg']} inter={o['inter_avg']} delta={o['delta']}")
if pfv:
    print(f"  {'pooledFID':>10s}: FT=131.34 inter={pfv} delta={pfv-131.34:+.2f}")
PY

echo "===== [C] Housekeeping: remove stray 'standard7_protocol}' dir @ $(date -Iseconds) ====="
# The stray dir was created by a typo in the first run; inter results are preserved in
# OUT/results_inter.json and OUT/metrics_pooled_fid_inter_ll.json, so it is safe to remove.
if [[ -d "${NB_ROOT}/outputs/standard7_protocol}" ]]; then
  rm -rf "${NB_ROOT}/outputs/standard7_protocol}"
  echo "[OK] removed stray dir outputs/standard7_protocol}"
else
  echo "[SKIP] no stray dir"
fi

echo "{\"pipeline\":\"inter_ll_full10\",\"finished\":\"$(date -Iseconds)\"}" > "${OUT}/job_done.json"
du -sh "${OUT}" 2>/dev/null || true
echo "===== DONE inter_ll finish @ $(date -Iseconds) ====="
