#!/usr/bin/env python3
"""Pre-registered verdicts for the layer-wise injection experiment.

Reads, for each arm, the two metrics both frontiers were measured with:
  * SSIM (+ PixCorr / CLIP twice-2-way / Inception) from eval_official_seven_dir.py
  * cycle_disc_top1 in CLIP space from neuroweave_cycle_score.py

and applies the bars written down in `layered_arms.py` BEFORE the arms were
generated.  Nothing here decides anything after seeing the numbers except which
pre-registered branch to take.

Hard gates (the verdicts are not emitted at all if these fail):
  * the cycle script's GT image-ordering control must have passed, otherwise the
    semantic axis is unmeasured
  * every generated arm must have proven its layer assignment landed, both by
    read-back (`verified` in metrics.json) and by the pixel-diff sanity render;
    a SUSPECT_NOOP arm would look like evidence against layered injection when
    in fact nothing was injected
  * each arm's realised mass must match the plan in layered_arms.py

CAVEAT THAT MUST TRAVEL WITH THESE NUMBERS
------------------------------------------
This is ONE subject.  No significance test is meaningful at n=1, and the
frontier it is compared against is also one subject.  A pass here licenses
running all 10 subjects with paired statistics; it is not itself a result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from layered_arms import (  # noqa: E402
    BAR_CYCLE_PASS, BAR_CYCLE_STRONG, BAR_SSIM, FRONTIER, GENERATE_ARMS,
    LAYERED_ARMS, parse_spec, plan_counts, read_unet_layout,
)

NB = HERE.parents[1]
SDXL = ("/project/peilab/why/cache/eeg-brainit/hf/hub/"
        "models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/"
        "462165984030d82259a11f4367a4eed129e94a7b")


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="e.g. outputs/layered/sub-08")
    args = ap.parse_args()
    root = Path(args.root)
    nl = read_unet_layout(SDXL)["n_levels"]

    # ---- collect -----------------------------------------------------------
    rows: dict[str, dict] = {}
    gates: list[str] = []

    for arm, cfg in LAYERED_ARMS.items():
        ev = Path(cfg["reuse_eval"]) if cfg["reuse_eval"] else root / f"eval/s08_{arm}.json"
        d = load_json(ev)
        if d is None:
            gates.append(f"missing seven-metric eval for {arm}: {ev}")
            continue
        rows[arm] = {"ssim": d.get("ssim"), "pixcorr": d.get("pixcorr"),
                     "clip": d.get("clip"), "inception": d.get("inception"),
                     "alex5": d.get("alex5"), "swid": d.get("swav"), "fid": d.get("fid"),
                     "eval_json": str(ev), "reused": bool(cfg["reuse_eval"])}

    cyc = load_json(root / "cycle/cycle_report.json")
    if cyc is None:
        gates.append(f"missing {root / 'cycle/cycle_report.json'}")
    else:
        if not cyc.get("gt_control", {}).get("pass"):
            gates.append(f"cycle GT-ordering control failed: {cyc.get('gt_control')}")
        spaces = cyc.get("spaces", {})
        for arm in rows:
            for space in ("clip_vith", "rn50", "dino_v2"):
                r = spaces.get(space, {}).get(arm)
                if r is None:
                    gates.append(f"cycle: no {space} row for {arm}")
                    continue
                rows[arm][f"cycle_{space}"] = r.get("cycle_disc_top1")
                rows[arm][f"raw_{space}"] = r.get("cycle_raw_cos")
            if "eeg_gen_agree" in (spaces.get("clip_vith", {}).get(arm) or {}):
                rows[arm]["eeg_gen_agree"] = spaces["clip_vith"][arm]["eeg_gen_agree"]

    # ---- arm-level gates on the assignment actually landing -----------------
    for arm in GENERATE_ARMS:
        m = load_json(root / f"gen/{arm}/metrics.json")
        if m is None:
            gates.append(f"missing gen metrics for {arm} (was it generated?)")
            continue
        ver = m.get("verified")
        if not ver:
            gates.append(f"{arm}: metrics.json has no verified layer assignment "
                         f"(generation was skipped, so the assignment was never proven)")
            continue
        if ver.get("semantic_structure_overlap"):
            print(f"[warn] {arm}: semantic/structure share levels "
                  f"{ver['semantic_structure_overlap']}")
        spec = parse_spec(LAYERED_ARMS[arm]["spec"], 3)
        want = plan_counts(spec, nl)["total_mass"]
        if abs(ver.get("total_mass", -1) - want) > 1e-6:
            gates.append(f"{arm}: realised mass {ver.get('total_mass')} != planned {want}")
        s = m.get("layer_sanity")
        if s is None:
            gates.append(f"{arm}: no layer-sanity render; the assignment is unproven "
                         f"empirically")
        elif s.get("verdict") != "LAYER_SPEC_ACTIVE":
            gates.append(f"{arm}: layer sanity {s.get('verdict')} "
                         f"(pixel_diff={s.get('pixel_diff_mean')})")

    if gates:
        print("[GATE] refusing to emit verdicts:")
        for g in gates:
            print(f"       - {g}")
        sys.exit(2)

    # ---- the pre-registered verdicts ---------------------------------------
    s_s, c_s = rows["single"]["ssim"], rows["single"]["cycle_clip_vith"]
    s_l, c_l = rows["layered"]["ssim"], rows["layered"]["cycle_clip_vith"]
    s_r, c_r = rows["rev"]["ssim"], rows["rev"]["cycle_clip_vith"]
    s_w, c_w = rows["lowall"]["ssim"], rows["lowall"]["cycle_clip_vith"]

    dominates = (s_l >= BAR_SSIM) and (c_l > c_s)
    passes = (s_l >= BAR_SSIM) and (c_l >= BAR_CYCLE_PASS)
    strong = (s_l >= BAR_SSIM) and (c_l >= BAR_CYCLE_STRONG)
    falsified = (s_l <= s_s + 1e-9) and (c_l <= c_s + 1e-9)
    lowall_reproduces = (abs(c_w - c_l) <= 0.05) and (abs(s_w - s_l) <= 0.01)
    any_split_helps = (abs(c_r - c_l) <= 0.05) and (abs(s_r - s_l) <= 0.01)

    if strong:
        verdict = "H-LAYER STRONG PASS"
        action = ("layered injection broke the frontier: matched the best SSIM while "
                  "reaching the semantic regime. Take it to all 10 subjects with the "
                  "paired tests before writing it as a contribution.")
    elif passes:
        verdict = "H-LAYER PASS"
        action = ("layered injection dominates the structural frontier. Run all 10 "
                  "subjects; also try adding ControlNet timing now, since the IP-level "
                  "split is what is carrying the effect.")
    elif falsified:
        verdict = "H-LAYER FALSIFIED"
        action = ("placement does not help on either metric. DROP the layered claim and "
                  "the NeuroWeave generation story; ship the retrieval/fusion + anytime "
                  "result with cycle anticorrelation as the negative result.")
    elif dominates:
        verdict = "H-LAYER PARTIAL"
        action = ("layered keeps SSIM but the semantic gain is below the pre-registered "
                  "bar. Report the exact numbers and decide against the 0.25 bar rather "
                  "than re-tuning the bar.")
    else:
        verdict = "H-LAYER FAIL"
        action = "layered gives up SSIM without buying the semantic gain."

    caveats = []
    if lowall_reproduces:
        caveats.append("MATCHED-MASS CONTROL REPRODUCES LAYERED: the uniform "
                       "half-strength arm performs the same, so the effect is "
                       "conditioning STRENGTH, not layer placement. The placement story "
                       "must be dropped even though the numbers look good.")
    if any_split_helps:
        caveats.append("DIRECTION NOT SUPPORTED: the reversed arm performs the same as "
                       "the layered arm, so any split of the modalities across levels "
                       "works and 'semantic goes late' is not the reason.")

    # ---- print -------------------------------------------------------------
    def fmt(v, d=4):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "  --  "

    print(f"\n{'arm':<12}{'spec':<48}{'SSIM':>9}{'cycle':>9}{'Pix':>9}{'CLIP':>9}{'Incep':>9}")
    print("-" * 105)
    for arm, cfg in LAYERED_ARMS.items():
        r = rows[arm]
        print(f"{arm:<12}{cfg['spec']:<48}{fmt(r.get('ssim')):>9}"
              f"{fmt(r.get('cycle_clip_vith')):>9}{fmt(r.get('pixcorr')):>9}"
              f"{fmt(r.get('clip')):>9}{fmt(r.get('inception')):>9}"
              + ("   <- reused" if cfg["reuse_dir"] else ""))
    print(f"\n{'reference (pre-existing frontier)':<61}{'SSIM':>9}{'cycle':>9}")
    print("-" * 79)
    for k, v in FRONTIER.items():
        print(f"{k:<61}{v['ssim']:>9.4f}{v['cycle_disc_top1']:>9.4f}")

    print(f"\nbars: SSIM >= {BAR_SSIM} AND cycle_disc_top1 >= {BAR_CYCLE_PASS} "
          f"(strong: >= {BAR_CYCLE_STRONG})")
    print(f"cross-space cycle_disc_top1 (a real effect should not be a single-space artefact)")
    print(f"  {'arm':<12}{'clip_vith':>11}{'dino_v2':>10}{'rn50':>10}{'eeg_gen_agree':>15}")
    for arm in rows:
        r = rows[arm]
        print(f"  {arm:<12}{fmt(r.get('cycle_clip_vith')):>11}"
              f"{fmt(r.get('cycle_dino_v2')):>10}{fmt(r.get('cycle_rn50')):>10}"
              f"{fmt(r.get('eeg_gen_agree')):>15}")

    print(f"\n{verdict}")
    print(f"  layered SSIM {fmt(s_l)} vs single {fmt(s_s)}   "
          f"layered cycle {fmt(c_l)} vs single {fmt(c_s)}")
    print(f"  lowall (mass-matched) SSIM {fmt(s_w)} cycle {fmt(c_w)}")
    print(f"  rev (direction ctrl)  SSIM {fmt(s_r)} cycle {fmt(c_r)}")
    print(f"  -> {action}")
    for c in caveats:
        print(f"  !! {c}")

    out = {"verdict": verdict, "action": action, "caveats": caveats,
           "bars": {"ssim": BAR_SSIM, "cycle_pass": BAR_CYCLE_PASS,
                    "cycle_strong": BAR_CYCLE_STRONG},
           "arms": rows, "frontier": FRONTIER,
           "specs": {a: c["spec"] for a, c in LAYERED_ARMS.items()},
           "note": ("single-subject screen; a pass licenses a 10-subject paired "
                    "run, it is not itself a result")}
    (root / "layered_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[summary] wrote {root / 'layered_summary.json'}")


if __name__ == "__main__":
    main()
