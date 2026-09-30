#!/usr/bin/env python3
"""NeuroWeave v3 summary: apply pre-registered V1 / domination verdicts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NB_ROOT = Path("/project/peilab/why/NeuroBridge")
sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))
from nw3_arms import (  # noqa: E402
    DOMINATE, INIT_CEILING, RETENTION_TARGET, SOTA_REF, V1_ARMS, V1_BAR,
    dominates, passes_v1, retention,
)


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--init-seven", type=str, default="", help="optional V0 seven-metric of init RGB")
    args = ap.parse_args()
    out = Path(args.out)
    eval_dir = out / "eval"
    cycle_dir = out / "cycle"

    rows = {}
    for p in sorted(eval_dir.glob("*.json")):
        d = _load(p)
        tag = d.get("tag") or p.stem
        rows[tag] = d

    table = []
    v1_pass = []
    for tag, d in rows.items():
        rec = {
            "tag": tag,
            "pixcorr": d.get("pixcorr"), "ssim": d.get("ssim"),
            "alex2": d.get("alex2"), "alex5": d.get("alex5"),
            "inception": d.get("inception"), "clip": d.get("clip"),
            "swav": d.get("swav"), "fid": d.get("fid"),
            "ret_pixcorr": retention("pixcorr", float(d["pixcorr"])) if "pixcorr" in d else None,
            "ret_ssim": retention("ssim", float(d["ssim"])) if "ssim" in d else None,
            "v1_pass": passes_v1(d),
            "dominate": dominates(d),
        }
        if (cycle_dir / f"{tag}.json").is_file():
            c = _load(cycle_dir / f"{tag}.json")
            rec["spatial_cycle_pearson"] = c.get("vae_latent_pearson")
            rec["spatial_cycle_cosine"] = c.get("vae_latent_cosine")
        table.append(rec)
        if rec["v1_pass"]:
            v1_pass.append(tag)

    # pick best by (ssim + pixcorr) among all, and best dominate candidate
    def key_fid(r):
        return (float(r["ssim"] or 0) + float(r["pixcorr"] or 0), -float(r.get("swav") or 9))

    best_fid = max(table, key=key_fid) if table else None
    hard = [r for r in table if r["dominate"]["hard_dominate"]]
    soft = [r for r in table if r["dominate"]["soft_dominate"]]

    init_row = _load(Path(args.init_seven)) if args.init_seven and Path(args.init_seven).is_file() else None

    report = {
        "protocol": {
            "prompts": "generic (prompts_deploy.json) — NO class names",
            "pixcorr_ssim": "official gray@425 gaussian",
            "init_ceiling": INIT_CEILING,
            "retention_target": RETENTION_TARGET,
            "v1_bar": V1_BAR,
            "dominate": DOMINATE,
            "sota_ref": SOTA_REF,
            "v1_arms": {k: {kk: vv for kk, vv in v.items() if kk != "role"} for k, v in V1_ARMS.items()},
        },
        "init_seven": init_row,
        "arms": table,
        "verdict": {
            "v1_any_pass": bool(v1_pass),
            "v1_pass_arms": v1_pass,
            "hard_dominate_arms": [r["tag"] for r in hard],
            "soft_dominate_arms": [r["tag"] for r in soft],
            "best_fidelity_arm": best_fid["tag"] if best_fid else None,
            "best_fidelity": {k: best_fid[k] for k in
                              ["pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "swav", "fid",
                               "ret_pixcorr", "ret_ssim"]} if best_fid else None,
            "action": (
                "V1 PASS — proceed to S2/S3 semantic upgrades" if v1_pass else
                "V1 FAIL — M5 alone insufficient; prioritize M2 factorized retention / lower strength"
            ),
        },
    }
    (out / "nw3_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # human table
    print(f"{'arm':<14}{'Pix':>8}{'SSIM':>8}{'retP':>7}{'retS':>7}{'Incep':>8}{'CLIP':>8}{'SwAV':>8}  V1 Dom")
    print("-" * 90)
    for r in table:
        dom = "H" if r["dominate"]["hard_dominate"] else ("S" if r["dominate"]["soft_dominate"] else "-")
        print(f"{r['tag']:<14}{r['pixcorr']:8.4f}{r['ssim']:8.4f}"
              f"{(r['ret_pixcorr'] or 0):7.3f}{(r['ret_ssim'] or 0):7.3f}"
              f"{r['inception']:8.4f}{r['clip']:8.4f}{r['swav']:8.4f}  "
              f"{'Y' if r['v1_pass'] else 'n'}  {dom}")
    print()
    print("VERDICT:", report["verdict"]["action"])
    print("V1 pass arms:", v1_pass or "(none)")
    print("hard dominate:", report["verdict"]["hard_dominate_arms"] or "(none)")
    print("wrote", out / "nw3_summary.json")


if __name__ == "__main__":
    main()
