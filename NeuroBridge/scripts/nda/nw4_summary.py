"""NW4 summary — applies the CORRECTED 2-D bar from `nw4_arms.py`.

The nw3 bar (pixcorr>=0.211 and ssim>=0.432) is passed by the blurry V0 init
alone (0.2998/0.4997), so it cannot discriminate.  This summary therefore reports:

  fidelity kept?   pixcorr >= 0.211 and ssim >= 0.432
  semantics real?  inception >= 0.65 and clip >= 0.70      <- the new gate
  PASS             both
  soft dominate    beats every reference on every reported metric (nw4_arms.DOMINATE)
  vs NW3           all six nw3 arms' best inception was 0.5236 (= +0.0126 over
                   chance for the entire grid); that is the number to beat
  condition health the generator is only as good as its condition bank, so the
                   S2 report's offdiag/erank are surfaced next to the metrics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nw4_arms import (ARMS, DOMINATE, FID_FLOOR, INIT_CEILING, NW3_REF,  # noqa: E402
                      RETENTION_TARGET, SEM_GATE, SOTA_REF, dominates, passes_2d,
                      retention)

METRICS = ["pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "swav"]


def load_json(p: Path) -> dict | None:
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--json-out", type=str, default="")
    args = ap.parse_args()
    out = Path(args.out)

    init = load_json(out / "v0" / "init_seven.json") or \
        load_json(Path("/project/peilab/why/NeuroBridge/outputs/nw3/sub-08/v0/init_seven.json"))
    s1 = load_json(out / "s1" / "report.json")
    s2 = load_json(out / "s2" / "report.json")

    arms = {}
    for tag in ARMS:
        m = load_json(out / "eval" / f"{tag}.json")
        if m is None:
            continue
        cy = load_json(out / "cycle" / f"{tag}.json")
        rec = {k: m.get(k) for k in METRICS}
        rec["fid"] = m.get("fid")
        rec["pass_2d"] = passes_2d(m)
        rec["fid_ok"] = all(m.get(k, 0.0) >= v for k, v in FID_FLOOR.items())
        rec["sem_ok"] = all(m.get(k, 0.0) >= v for k, v in SEM_GATE.items())
        rec["dominate"] = dominates(m)
        rec["retention"] = {k: round(v, 4) for k, v in retention(m).items()}
        if cy:
            rec["spatial_cycle_pearson"] = cy.get("vae_latent_pearson")
        arms[tag] = rec

    # ---- table ----
    hdr = f"{'arm':<18}" + "".join(f"{h:>9}" for h in METRICS) + \
          f"{'pass':>6}{'sem':>5}{'fid':>5}{'dom':>5}"
    print(hdr)
    print("-" * len(hdr))
    if init:
        print(f"{'V0_init':<18}" + "".join(f"{init.get(h, float('nan')):>9.4f}" for h in METRICS)
              + f"{'--':>6}{'--':>5}{'--':>5}{'--':>5}")
    for tag, r in sorted(arms.items(), key=lambda kv: -(kv[1].get("inception") or 0)):
        print(f"{tag:<18}" + "".join(f"{(r.get(h) or float('nan')):>9.4f}" for h in METRICS)
              + f"{'Y' if r['pass_2d'] else '.':>6}"
              + f"{'Y' if r['sem_ok'] else '.':>5}"
              + f"{'Y' if r['fid_ok'] else '.':>5}"
              + f"{'Y' if r['dominate']['soft'] else '.':>5}")

    print(f"\n[gate] fidelity  pixcorr>={FID_FLOOR['pixcorr']} ssim>={FID_FLOOR['ssim']}")
    print(f"[gate] semantics inception>={SEM_GATE['inception']} clip>={SEM_GATE['clip']} "
          f"(chance 0.50; NW3's best over its whole 6-arm grid was "
          f"{NW3_REF['best_inception']})")
    print(f"[gate] soft dominance: " + ", ".join(
        f"{k}{'<=' if k == 'swav' else '>='}{v}" for k, v in DOMINATE.items()))

    if s2:
        print("\n[condition health] S2 projection (lower offdiag = less collapsed-to-mean; "
              "1.0 = a constant vector)")
        print(f"  nw3 baseline: S2 offdiag={NW3_REF['s2_offdiag']} "
              f"S3(=fed to generator) offdiag={NW3_REF['s3_offdiag']}")
        for name, r in (s2.get("conditions") or {}).items():
            print(f"  {name:<8} raw_offdiag={r.get('raw_offdiag')} -> "
                  f"proj_offdiag={r.get('proj_offdiag')}  erank={r.get('proj_erank')}  "
                  f"anchor_top1={r.get('anchor_test', {}).get('top1')} "
                  f"csls={r.get('anchor_test', {}).get('top1_csls')} "
                  f"gamma={r.get('gamma')}")
        fc = s2.get("fused_control") or {}
        if fc:
            print(f"  fused    offdiag={fc.get('offdiag')} erank={fc.get('erank')} "
                  f"(A3 control: dense mean)")

    # ---- verdicts ----
    print()
    if not arms:
        print("[VERDICT] no arm produced metrics — nothing to judge")
    else:
        sem = {t: (r.get("inception") or 0) for t, r in arms.items()}
        best_sem = max(sem, key=lambda t: sem[t])
        best = arms[best_sem]
        print(f"[VERDICT] best semantics: {best_sem} inception={sem[best_sem]:.4f} "
              f"(clip={best.get('clip'):.4f}, ssim={best.get('ssim'):.4f}, "
              f"pixcorr={best.get('pixcorr'):.4f})")
        d = sem[best_sem] - NW3_REF["init_inception"]
        print(f"[VERDICT] A0/A1+A3 signal vs the V0 init blur: {d:+.4f} Incep "
              f"(nw3's whole grid managed {NW3_REF['best_inception'] - NW3_REF['init_inception']:+.4f})")
        passing = [t for t, r in arms.items() if r["pass_2d"]]
        print(f"[VERDICT] arms passing the 2-D bar: {passing or 'NONE'}")
        soft = [t for t, r in arms.items() if r["dominate"]["soft"]]
        print(f"[VERDICT] arms softly dominating all references: {soft or 'NONE'}")

        # the A4 question: does layering beat the flat control?
        pair = [t for t in ("w4_layered", "w4_flat", "w4_mirror", "w4_allearly",
                            "w4_alllate") if t in arms]
        if len(pair) >= 2:
            print("\n[A4 layer test] inception / ssim by arm (layered must beat flat)")
            for t in sorted(pair, key=lambda x: -(arms[x].get("inception") or 0)):
                print(f"  {t:<18} inception={arms[t].get('inception'):.4f} "
                      f"ssim={arms[t].get('ssim'):.4f} "
                      f"pixcorr={arms[t].get('pixcorr'):.4f}")
            if "w4_layered" in arms and "w4_flat" in arms:
                dl = arms["w4_layered"]["inception"] - arms["w4_flat"]["inception"]
                ds = arms["w4_layered"]["ssim"] - arms["w4_flat"]["ssim"]
                print(f"  layered - flat: dIncep={dl:+.4f} dSSIM={ds:+.4f} -> "
                      f"{'A4 SUPPORTED' if dl > 0.02 and ds > -0.02 else 'A4 NOT SUPPORTED'}")
        # the A3 question: does multi-branch beat the dense fusion?
        if "w4_fused" in arms and "w4_flat" in arms:
            df = arms["w4_flat"]["inception"] - arms["w4_fused"]["inception"]
            print(f"[A3 fusion test] flat - fused: dIncep={df:+.4f} -> "
                  f"{'A3 SUPPORTED' if df > 0.02 else 'A3 NOT SUPPORTED'}")

    if init:
        print(f"\n[ceiling] V0 init pixcorr={INIT_CEILING['pixcorr']} "
              f"ssim={INIT_CEILING['ssim']} inception={INIT_CEILING['inception']}; "
              f"retention target pixcorr>={RETENTION_TARGET['pixcorr']:.4f} "
              f"ssim>={RETENTION_TARGET['ssim']:.4f}")
    print("[refs] " + " | ".join(
        f"{k}: incep {v['inception']} ssim {v['ssim']}" for k, v in SOTA_REF.items()))

    payload = {"protocol": {"prompts": "generic (prompts_deploy.json) — NO class names",
                            "fid_floor": FID_FLOOR, "sem_gate": SEM_GATE,
                            "dominate": DOMINATE, "sota_ref": SOTA_REF,
                            "nw3_ref": NW3_REF},
               "init_seven": init, "s1": s1, "s2": s2, "arms": arms}
    dest = Path(args.json_out) if args.json_out else (out / "nw4_summary.json")
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[json] {dest}")


if __name__ == "__main__":
    main()
