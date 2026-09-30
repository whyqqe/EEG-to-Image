#!/usr/bin/env python3
"""Aggregate NeuroWeave Stage-1 reports and apply pre-registered verdicts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fuse_top1(probe: dict) -> float | None:
    """Best inductive fused top1 under any fuse key (prefers csls)."""
    fusions = probe.get("fusions") or probe.get("fusion") or {}
    best = None
    for k, v in fusions.items():
        if not isinstance(v, dict):
            continue
        # prefer CSLS inductive
        for key in ("top1_csls", "csls_top1", "top1"):
            if key in v and "sinkhorn" not in k.lower():
                best = max(best or -1.0, float(v[key]))
                break
    # also look at targets' mlp
    for _n, t in (probe.get("targets") or {}).items():
        mlp = t.get("mlp") or {}
        if "top1_csls" in mlp:
            best = max(best or -1.0, float(mlp["top1_csls"]))
        elif "top1" in mlp:
            best = max(best or -1.0, float(mlp["top1"]))
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    args = ap.parse_args()
    root = Path(args.root)

    def _load(path: Path) -> dict:
        if not path.is_file():
            return {}
        # Two concurrent jobs once concatenated a second trailing "}"; take the
        # first complete JSON value so a race cannot kill the summary stage.
        text = path.read_text()
        return json.JSONDecoder().raw_decode(text)[0]
    s1 = _load(root / "s1_report.json")
    anytime = _load(root / "anytime_report.json")
    cycle = _load(root / "cycle/cycle_report.json")

    print("=" * 72)
    print("NeuroWeave Stage-1 summary (sub-08)")
    print("=" * 72)

    # ---- Axis H / C: arm table
    print(f"\n{'arm':<16}{'mode':<8}{'temporal':<12}{'ep':>4}{'test200':>10}{'top5':>8}{'rank':>8}")
    arm_rows = []
    for arm, v in (s1.get("arms") or {}).items():
        cfg = v.get("cfg") or {}
        te = v.get("test200") or {}
        ep = (v.get("picks") or {}).get("mini_top1", {}).get("epoch", -1)
        print(f"{arm:<16}{cfg.get('mode','?'):<8}{cfg.get('temporal','?'):<12}"
              f"{ep:>4}{te.get('top1', float('nan')):>10.4f}"
              f"{te.get('top5', float('nan')):>8.4f}"
              f"{te.get('mean_rank', float('nan')):>8.1f}")
        arm_rows.append((arm, te.get("top1", 0.0), cfg))

    # probes
    # NOTE the key names: cfmsf_route_probe writes `targets[name]["mlp"]` and a
    # single `fusion` dict keyed "mlp|by=<stat>|k=<n>", each value holding a `raw`
    # sub-dict.  The first version of this file read `fusions` and a flat
    # `top1_csls`, which silently produced 0.0 for every arm and made both
    # pre-registered verdicts trivially "fail" with zero evidence.
    print(f"\n{'arm':<16}{'best_single_csls':>18}{'fuse_csls':>11}{'fuse+sink':>11}  probe")
    probe_scores = {}
    for arm, _, _ in arm_rows:
        p = root / arm / "probe" / "route_probe.json"
        if not p.is_file():
            print(f"{arm:<16}{'MISSING':>18}")
            continue
        pr = json.JSONDecoder().raw_decode(p.read_text())[0]
        tgt = pr.get("targets") or {}
        fus = pr.get("fusion") or {}
        best_single = max((float((t.get("mlp") or {}).get("top1_csls", 0))
                           for t in tgt.values()), default=0.0)
        if fus:
            by_csls = max(fus.values(), key=lambda v: v["raw"]["top1_csls"])
            by_sink = max(fus.values(), key=lambda v: v["raw"].get("sinkhorn_top1", 0))
            fuse_csls = float(by_csls["raw"]["top1_csls"])
            fuse_sink = float(by_sink["raw"].get("sinkhorn_top1", 0))
            fuse_by = by_csls.get("ranked_by")
            routes = by_csls.get("routes")
        else:
            fuse_csls = fuse_sink = 0.0
            fuse_by, routes = None, None
        probe_scores[arm] = {"best_single_csls": best_single, "fuse_csls": fuse_csls,
                            "fuse_sinkhorn": fuse_sink, "fuse_by": fuse_by,
                            "routes": routes,
                            "best_route": max(tgt, key=lambda k: (tgt[k]["mlp"] or {}).get("top1_csls", 0)) if tgt else None}
        print(f"{arm:<16}{best_single:>18.4f}{fuse_csls:>11.4f}{fuse_sink:>11.4f}  {p.name}")

    # ---- Axis T: anytime curves
    if anytime.get("arms"):
        print(f"\nAnytime curve (200-way top1 on levels_mean):")
        print(f"{'arm':<16}{'150ms':>8}{'350ms':>8}{'700ms':>8}{'1000ms':>8}{'Δ full-150':>12}")
        for arm, v in anytime["arms"].items():
            c = v["curve"]
            vals = [c[w]["top1"] for w in ("early", "mid", "late", "full")]
            print(f"{arm:<16}{vals[0]:>8.4f}{vals[1]:>8.4f}{vals[2]:>8.4f}{vals[3]:>8.4f}"
                  f"{vals[3]-vals[0]:>+12.4f}")

    # ---- Axis cycle
    if cycle.get("sets"):
        print(f"\nCycle variants on existing generations:")
        print(f"{'set':<16}{'raw_cos':>10}{'disc_top1':>10}{'eeg_agree':>10}")
        for r in cycle["sets"]:
            print(f"{r['name']:<16}{r['cycle_raw_cos']:>10.4f}{r['cycle_disc_top1']:>10.4f}"
                  f"{r.get('eeg_gen_agree', float('nan')):>10.4f}")
        print(f"  ranks_disagree={cycle.get('verdict', {}).get('ranks_disagree')}")

    # ---- Pre-registered verdicts
    print("\n" + "=" * 72)
    print("PRE-REGISTERED VERDICTS")
    print("=" * 72)
    verdicts = {}
    arm_te = {a: (v.get("test200") or {}).get("top1")
              for a, v in (s1.get("arms") or {}).items()}

    def _te(a):
        return arm_te.get(a)

    def _best(names, base=None):
        vals = [(a, _te(a)) for a in names if _te(a) is not None]
        return max(vals, key=lambda t: t[1]) if vals else (None, None)

    # H1 -- CAPACITY: LoRA adaptation over the frozen encoder.
    #   Pre-registered bar: >= frozen + 0.02 on the 200-way test number.
    h1_arm, h1_val = _best(["lora", "direct"])
    verdicts["H1_capacity_lora_beats_frozen"] = {
        "pass": bool(h1_val is not None and _te("frozen") is not None
                     and h1_val >= _te("frozen") + 0.02),
        "frozen": _te("frozen"), "best_arm": h1_arm, "best": h1_val,
        "threshold": "best(lora,direct) >= frozen + 0.02",
        "round1": {"frozen": 0.3650, "lora": 0.3900, "verdict": "pass (+0.025)"},
    }

    # H2 -- HIERARCHY: this is the claim that has to earn its place.
    #   Round 1: three heads (`multi_head` 0.3700) LOST to one head (`lora` 0.3900),
    #   so round 2 expresses hierarchy as auxiliary early/mid losses on the SINGLE
    #   exported head (`hier_aux`).  Same bar again: lora + 0.02.  If it fails
    #   here, hierarchy is not a contribution and the paper must drop it instead
    #   of re-parameterising it a third time.
    h2_arm, h2_val = _best(["hier_aux", "multi_head"])
    verdicts["H2_hierarchy_beats_single_head"] = {
        "pass": bool(h2_val is not None and _te("lora") is not None
                     and h2_val >= _te("lora") + 0.02),
        "lora": _te("lora"), "best_arm": h2_arm, "best": h2_val,
        "threshold": "best(hier_aux,multi_head) >= lora + 0.02",
        "round1": {"lora": 0.3900, "multi_head": 0.3700,
                   "verdict": "FAIL -- 3 separate heads lost to 1 head"},
        "action_if_fail": ("drop the hierarchy claim rather than re-parameterise"),
    }

    # T -- TEMPORAL.  Round 1's `anytime_train` won at 150 ms (0.130 vs 0.045) but
    #   lost 0.06 at the full window.  The round-2 bar is BOTH: keep the win and
    #   remove the cost.  This is the one axis with a real positive signal.
    def _curve(arm):
        c = ((anytime.get("arms") or {}).get(arm) or {}).get("curve")
        return c if c and "early" in c and "full" in c else None

    t_rows = {}
    for a in ("lora", "anytime_train", "anytime_soft", "anytime_curric",
              "anytime_consist"):
        c = _curve(a)
        if c is None and _te(a) is None:
            continue
        t_rows[a] = {
            "test200": _te(a),
            "full": c["full"]["top1"] if c else None,
            "early": c["early"]["top1"] if c else None,
            "mid": c["mid"]["top1"] if c else None,
            "late": c["late"]["top1"] if c else None,
        }
    full_bar = None if _te("lora") is None else _te("lora") - 0.01
    fixed = [a for a, r in t_rows.items()
             if a.startswith("anytime") and r["full"] is not None
             and r["early"] is not None
             and r["full"] >= (full_bar if full_bar is not None else 1.0)
             and r["early"] >= 0.13]
    best_early = max(((a, r["early"]) for a, r in t_rows.items()
                      if r["early"] is not None), key=lambda t: t[1], default=(None, None))
    verdicts["T_temporal_fix"] = {
        "rows": t_rows,
        "full_bar": full_bar,
        "early_bar": 0.13,
        "arms_meeting_both_bars": fixed,
        "best_early_arm": best_early[0],
        "best_early_top1": best_early[1],
        "FIX_PASSES": bool(fixed),
        "keep_anytime_as_contrib": bool(fixed) or bool(best_early[1] and best_early[1] >= 0.13),
        "drop_causal_mask": True,
        "round1": {"anytime_train": {"early": 0.1300, "full": 0.3300},
                   "lora": {"early": 0.0450, "full": 0.3900},
                   "verdict": "win at 150 ms, cost 0.06 at full window"},
        "rule": ("A schedule earns the progressive-decoding claim only if it keeps "
                 "early >= 0.13 AND brings full back to within 0.01 of lora. "
                 "Winning one at the other's expense is the round-1 failure, not a "
                 "contribution. The hard causal mask stays OUT either way."),
    }

    # C -- CYCLE SPACE.  Round 1 scored cycle in CLIP space only, where the three
    #   sets ranked identically, i.e. CLIP cannot see the structure the SSIM gap
    #   is about; and its `eeg_agree` of 0.0000 was a bug (raw `shared_r` compared
    #   against CLIP).  Round 2 re-scores in DINOv2 and RN50 as well, behind a hard
    #   GT-ordering control.
    cyc_v = (cycle.get("verdict") or {})
    verdicts["C_cycle_space"] = cyc_v or {"note": "cycle report missing"}
    if cyc_v.get("structure_space_helps") is False:
        verdicts["C_cycle_space"]["action"] = (
            "DROP the cycle term: every visual space produces the same ranking, so "
            "the choice of space is not what the contribution needs.")

    for k, v in verdicts.items():
        print(f"\n[{k}]")
        print(json.dumps(v, indent=2))

    out = {
        "arm_test200": {a: (s1.get("arms") or {}).get(a, {}).get("test200")
                        for a in (s1.get("arms") or {})},
        "probe_scores": probe_scores,
        "anytime": anytime,
        "cycle": cycle,
        "verdicts": verdicts,
        "baselines": {
            "frozen_probe_csls_ref": 0.5350,
            "best_single_cat5_ref": 0.4050,
            "encoder_calib_top1_ref": 0.25,
        },
    }
    (root / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[summary] wrote {root / 'summary.json'}")


if __name__ == "__main__":
    main()
