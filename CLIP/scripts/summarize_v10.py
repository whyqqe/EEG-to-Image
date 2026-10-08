#!/usr/bin/env python
"""summarize_v10 -- turn the v10 sweep cells into the three things the pipeline has to decide.

Reads the per-checkpoint JSONs written by `run_eval --fgw-sweep` (30 of them = 10 subjects x 3
seeds on the banked v8 encoder) and answers, in order:

1. FIDELITY. The sweep recomputes the deployed row from scratch. If its (R=full, alpha=0.75,
   tau=0.01) cell does not reproduce the banked `+ T2 reps` Top-1 -- which was measured by a
   different invocation months of edits ago -- then the sweep is computing a different operator
   and nothing below it means anything. Checked per checkpoint; a mismatch aborts.

2. THE VARIANCE-GATING CURVE. gain(R) = Top-1(alpha>0) - Top-1(alpha=0) on the *identical*
   repetition subset, as R falls 80 -> 1. The pre-registered reading:
     * falls  -> the structural term rides on target-side estimation variance; the rep count
                 must be quoted and the baseline given equal test information before any
                 comparison to a repetition-free method is called fair.
     * flat   -> the gain is not bought with test information and the comparison is fair.
   Reported as a PAIRED statistic over runs, with n_pos, because an unpaired difference of two
   noisy means is what this project has already been burned by once.

3. A HONEST CONFIG NUMBER. `alpha` and `tau` were chosen by looking at the same folds that are
   then reported, which is optimistically biased. The nested-LOSO loop removes that: for each
   held-out subject the config is picked on the OTHER nine, and only then scored. If the nested
   number holds up against the fixed-config number, the fixed config was not overfit; if it
   collapses, the headline was a selection artefact and must be replaced by the nested one.

Usage:
    python scripts/summarize_v10.py --sweep-dir outputs/eval/v10_sweep \
        --banked-dir outputs/eval/v8_fgw075 --out outputs/v10_summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

CELL = re.compile(r"^\+ T2 R=(?P<R>\d+),a=(?P<a>[0-9.]+),t=(?P<t>[0-9.]+)"
                  r"(?:,m=(?P<m>[0-9.]+))?(?P<off> \(structural-off\))?$")


def split_tag(tag: str) -> dict:
    """`'R=80,a=0.75,t=0.03,m=0.5'` -> `{'R':'80','a':'0.75','t':'0.03','m':'0.5'}`.

    `m` defaults to 0 so stage-1 tags (which predate it) parse unchanged -- the whole stage-1
    summary has to stay readable by this same code, or re-running it would silently reinterpret
    the earlier grid.
    """
    d = dict(p.split("=") for p in tag.split(","))
    d.setdefault("m", "0")
    return d


def load_cells(d: Path) -> dict:
    """{(subject, seed): {'R=..,a=..,t=..': top1, ...}} plus the raw rows for the gate."""
    out = {}
    for f in sorted(glob.glob(str(d / "sub*_seed*.json"))):
        j = json.load(open(f))
        sub = int(j["target_subject"])
        for key, c in j["checkpoints"].items():
            m = re.search(r"seed(\d+)", key)
            seed = int(m.group(1)) if m else -1
            cells, raw = {}, {}
            for name, row in c.get("rows", {}).items():
                mm = CELL.match(name)
                if mm:
                    # CANONICALISE the tag: `m` is written even when the JSON predates it, so a
                    # stage-1 directory (no mix dimension) and a stage-2.5 one key the SAME cell
                    # identically. Without this, every lookup of the `m=0` cell misses on old
                    # data and returns nan -- which the optimism verdict below then reported as
                    # "small", i.e. a missing cell silently read as a clean bill of health.
                    tag = (f"R={mm['R']},a={mm['a']},t={mm['t']},"
                           f"m={float(mm['m']) if mm['m'] is not None else 0.0:g}")
                    cells.setdefault(tag, {})["off" if mm["off"] else "on"] = float(row["top1"])
                raw[name] = float(row.get("top1", float("nan")))
            out[(sub, seed)] = {"cells": cells, "raw": raw, "file": Path(f).name}
    return out


def paired(deltas: list[float]) -> dict:
    a = np.asarray(deltas, dtype=float)
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    t = float(a.mean() / (sd / np.sqrt(len(a)))) if sd > 0 else float("nan")
    return {"mean": float(a.mean()), "sd": sd, "t": t,
            "n_pos": int((a > 0).sum()), "n": int(len(a))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default="outputs/eval/v10_sweep")
    ap.add_argument("--fidelity", nargs="*", default=[
        "R=80,a=0.75,t=0.01@outputs/eval/v8_fgw075",
        "R=80,a=0.75,t=0.03@outputs/eval/v8scr/a0.75_t0.03"],
        help="TAG@DIR pairs: cells the sweep must reproduce exactly. The tau=0.03 entry is the "
             "one that carries the v10 headline (53.80), so both temperatures are gated -- a "
             "single reference would let a tau mix-up pass.")
    ap.add_argument("--curve-alpha", type=float, default=0.75)
    ap.add_argument("--curve-tau", default="0.03",
                    help="tau the gating curve is read at; 0.03 is the v10 headline's tau, "
                         "0.01 is the v8_fgw075 duplicate. Both are printed.")
    ap.add_argument("--tol", type=float, default=0.51)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sweep = load_cells(Path(args.sweep_dir))
    if not sweep:
        raise SystemExit(f"no sweep JSONs under {args.sweep_dir}")
    print(f"[sum] {len(sweep)} sweep checkpoints")

    # ---- 1. fidelity gates --------------------------------------------------------------
    # Several references, because a single one cannot catch a systematic tau/alpha mix-up:
    # any wrong cell that happens to differ from all references is what we want to see fail.
    gate_report = {}
    if not args.fidelity:
        print("[sum] NOTE: no fidelity references given -- the gate is NOT evaluated in this "
              "run. Expected only when the run is intentionally off the deployed estimator "
              "(e.g. the stage-2 shrink sweep, where every non-0.1 value must differ).")
    for spec in args.fidelity:
        tag, _, dirp = spec.partition("@")
        banked = load_cells(Path(dirp)) if Path(dirp).is_dir() else {}
        diffs, checked = [], 0
        for k, v in sweep.items():
            b = banked.get(k)
            if not b:
                continue
            ref = b["raw"].get("+ T2 reps")
            got = v["cells"].get(tag, {}).get("on")
            if ref is None or got is None:
                continue
            diffs.append(abs(got - ref))
            checked += 1
        gate_report[tag] = {"dir": dirp, "checked": checked,
                            "max_abs_diff": float(max(diffs)) if diffs else None,
                            "mean_abs_diff": float(np.mean(diffs)) if diffs else None}
        if checked:
            print(f"[sum] FIDELITY {tag} vs {dirp} on {checked} ckpts: "
                  f"max|diff|={max(diffs):.3f} mean|diff|={np.mean(diffs):.3f}")
            if max(diffs) > args.tol:
                raise SystemExit(f"[sum] FIDELITY FAILED for {tag} vs {dirp} -- the sweep is "
                                 "not the deployed operator. Nothing below is trustworthy.")
        else:
            print(f"[sum] WARNING: no cells matched reference {tag}@{dirp}; gate NOT evaluated.")

    # ---- 2. variance-gating curve ------------------------------------------------------
    def curve_at(tau_str: str) -> dict:
        out = {}
        print(f"\n[gating curve] alpha={args.curve_alpha:g} tau={tau_str}")
        print(f"{'R':>5} {'on':>7} {'off':>7} {'gain':>8} {'n_pos':>7} {'t':>7}")
        print("-" * 50)
        for R in Rs:
            on, off = [], []
            for v in sweep.values():
                for tag, c in v["cells"].items():
                    parts = split_tag(tag)
                    if (int(parts["R"]) == R and float(parts["a"]) == args.curve_alpha
                            and parts["t"] == tau_str and float(parts["m"]) == 0.0
                            and "on" in c and "off" in c):
                        on.append(c["on"])
                        off.append(c["off"])
            if len(on) < 2:
                continue
            d = np.asarray(on) - np.asarray(off)
            out[str(R)] = {"on_mean": float(np.mean(on)), "off_mean": float(np.mean(off)),
                           "gain_mean": float(d.mean()), **paired(list(d))}
            print(f"{R:>5} {np.mean(on):>7.2f} {np.mean(off):>7.2f} {d.mean():>+8.2f} "
                  f"{int((d > 0).sum()):>3}/{len(d):<3} {out[str(R)]['t']:>7.2f}")
        return out

    Rs = sorted({int(t.split(",")[0][2:]) for v in sweep.values() for t in v["cells"]},
                reverse=True)
    curve = {args.curve_tau: curve_at(args.curve_tau)}
    for other in ("0.01", "0.03", "0.05"):
        if other != args.curve_tau:
            curve[other] = curve_at(other)

    # ---- 2b. M1: the SOURCE-METRIC TEMPLATE (v10 stage 2.5) ------------------------------
    #
    # The object is the PAIRED difference against each cell's own m=0 twin at the identical R,
    # alpha and tau -- the same discipline as the structural-off twin above. Reading a mix cell
    # against the published headline instead would confound the mix with the R it was run at.
    #
    # H1 (interior peak) and H2 (gain larger at low R) are both read off this one table, and
    # both are falsifiable in the direction that matters: an endpoint peak or an R-flat gain
    # means the mechanism is not the variance reduction the docstring claims.
    m1: dict = {}
    mixes = sorted({float(split_tag(t)["m"]) for v in sweep.values() for t in v["cells"]
                    if float(split_tag(t)["m"]) > 0.0})
    if mixes:
        print(f"\n[M1 source-template mix curve] alpha={args.curve_alpha:g} "
              f"tau={args.curve_tau}  (gain vs each cell's own m=0 twin)")
        print(f"{'mix':>6} {'R':>5} {'on':>7} {'off':>7} {'gain':>8} {'n_pos':>7} {'t':>7}")
        print("-" * 52)
        for m in mixes:
            for R in Rs:
                on, ref, off = [], [], []
                for v in sweep.values():
                    cells = v["cells"]
                    tgt = (f"R={R},a={args.curve_alpha:g},t={args.curve_tau},m={m:g}")
                    base = (f"R={R},a={args.curve_alpha:g},t={args.curve_tau},m=0")
                    if tgt in cells and "on" in cells[tgt] and base in cells and "on" in cells[base]:
                        on.append(cells[tgt]["on"])
                        ref.append(cells[base]["on"])
                        if "off" in cells[tgt]:
                            off.append(cells[tgt]["off"])
                if len(on) < 2:
                    continue
                dv = np.asarray(on) - np.asarray(ref)          # mix effect, paired per run
                m1[f"m={m:g},R={R}"] = {
                    "mix": float(m), "R": int(R),
                    "on_mean": float(np.mean(on)), "twin_mean": float(np.mean(ref)),
                    "gain_vs_twin": float(dv.mean()), **paired(list(dv)),
                    "off_mean": float(np.mean(off)) if off else None}
                print(f"{m:>6g} {R:>5} {np.mean(on):>7.2f} {np.mean(ref):>7.2f} "
                      f"{dv.mean():>+8.2f} {int((dv > 0).sum()):>3}/{len(dv):<3} "
                      f"{m1[f'm={m:g},R={R}']['t']:>7.2f}")
        # H1: is the peak interior? Compare each mix's mean gain against the endpoints.
        #
        # Guarded on a NON-EMPTY table: the paired readout needs >= 2 runs, so a single-fold
        # smoke produces `mixes` (non-zero cells exist) but an empty `m1`, and `max()` on that
        # raises -- turning a working smoke run into a crash in the one place that is supposed
        # to be cheap. A smoke that cannot complete is a smoke that cannot be used.
        if m1:
            best = max(m1.values(), key=lambda d: d["gain_vs_twin"])
            peak_at_edge = best["mix"] in (min(mixes), max(mixes)) and len(mixes) > 1
            m1["_verdict"] = {
                "argmax_mix": float(best["mix"]), "argmax_R": int(best["R"]),
                "argmax_gain_pp": float(best["gain_vs_twin"]),
                "H1_interior_peak": (not peak_at_edge),
                "note": ("interior peak: consistent with the bias-variance mechanism "
                         if not peak_at_edge else
                         "PEAK AT AN ENDPOINT -- H1 falsified; the template is not acting as a "
                         "variance-reduced estimate of the target's metric")}
            print(f"[M1] best mix={best['mix']:g} at R={best['R']} "
                  f"gain={best['gain_vs_twin']:+.2f}pp  -> {m1['_verdict']['note']}")
            # H2: is the gain LARGER at the low-R end? That is the mechanism's whole claim --
            # R=20 is where the target's metric is noisiest, so variance reduction should pay
            # off most there. A gain that is flat or inverted in R means the mix is helping for
            # some other reason (or none) and the docstring's story does not hold.
            by_R = {}
            for d in m1.values():
                if isinstance(d, dict) and "R" in d:
                    by_R.setdefault(int(d["R"]), []).append(d["gain_vs_twin"])
            if len(by_R) >= 2:
                lo = min(by_R)
                hi = max(by_R)
                delta = float(np.mean(by_R[lo]) - np.mean(by_R[hi]))
                m1["_H2"] = {"low_R": lo, "high_R": hi, "gain_low_R": float(np.mean(by_R[lo])),
                             "gain_high_R": float(np.mean(by_R[hi])), "low_minus_high_pp": delta,
                             "H2_larger_at_low_R": bool(delta > 0)}
                print(f"[M1] H2: gain at R={lo} is {delta:+.2f}pp vs R={hi} "
                      f"-> {'consistent' if delta > 0 else 'NOT supported'}")
        else:
            print(f"[M1] {len(mixes)} mix value(s) present but no cell had a paired twin on "
                  f">= 2 runs (R swept = {sorted(Rs)}). Nothing to compare; single-fold smoke?")

    # The pre-registered readout, decided before the runs: does the gain survive fewer
    # repetitions? A drop means the gain rides on target-side estimation variance (the rep
    # count must travel with any claim, and a repetition-free baseline is not
    # information-matched); flat means it does not and the comparison is fair as it stands.
    gates = {}
    for tau_s, c in curve.items():
        # A gate needs a curve. tau=0.05 only has R=80 (the R dimension is swept at 0.01/0.03),
        # and comparing R=80 to itself reports a fake "FLAT ... 80 -> 80" -- a fabricated pass
        # on the one gate that is supposed to be able to fail.
        if len(c) < 2:
            gates[tau_s] = {"drop_pp": None, "verdict": "NOT SWEPT over R -- no gate."}
            print(f"\n[gate] tau={tau_s}: R not swept (only {sorted(c)}); no gate computed.")
            continue
        rr = sorted((int(k) for k in c), reverse=True)
        hi_r, lo_r = str(rr[0]), str(rr[-1])
        drop = c[hi_r]["gain_mean"] - c[lo_r]["gain_mean"]
        if drop >= 2.0:
            v = (f"GATED at tau={tau_s}: gain falls {drop:+.2f}pp as R goes {hi_r} -> {lo_r}. "
                 "The structural term rides on target-side estimation variance.")
        elif abs(drop) < 1.0:
            v = (f"FLAT at tau={tau_s}: gain moves {drop:+.2f}pp over R {hi_r} -> {lo_r}. "
                 "Not test-information-dependent; the comparison is fair.")
        else:
            v = (f"INTERMEDIATE at tau={tau_s}: gain moves {drop:+.2f}pp over R {hi_r} -> "
                 f"{lo_r}.")
        gates[tau_s] = {"drop_pp": float(drop), "verdict": v}
        print(f"\n[gate] {v}")

    # ---- 3. alpha / tau grids at full R ------------------------------------------------
    grids = {}
    for label, key in (("alpha", "a"), ("tau", "t")):
        print(f"\n[{label} grid at R=full, "
              f"{'tau=' + args.curve_tau if label == 'alpha' else 'alpha=' + format(args.curve_alpha, 'g')}]")
        print(f"{'value':>7} {'top1':>8} {'off':>8} {'gain':>8}")
        rows = {}
        for v in sweep.values():
            for tag, c in v["cells"].items():
                parts = split_tag(tag)
                if int(parts["R"]) != max(Rs) or float(parts["m"]) != 0.0:
                    continue
                if label == "alpha" and parts["t"] != args.curve_tau:
                    continue
                if label == "tau" and float(parts["a"]) != args.curve_alpha:
                    continue
                rows.setdefault(parts[key], []).append(c)
        for val in sorted(rows):
            on = [c["on"] for c in rows[val] if "on" in c]
            off = [c["off"] for c in rows[val] if "off" in c]
            if not on:
                continue
            grids.setdefault(label, {})[val] = {
                "top1_mean": float(np.mean(on)), "off_mean": float(np.mean(off)),
                "gain_mean": float(np.mean(on) - np.mean(off)) if off else float("nan")}
            print(f"{val:>7} {np.mean(on):>8.2f} {np.mean(off):>8.2f} "
                  f"{grids[label][val]['gain_mean']:>+8.2f}")

    # ---- 4. nested-LOSO over (alpha, tau) ----------------------------------------------
    # Candidates are the cells the sweep actually measured at full R. Selection is per
    # held-out subject on the other nine; ties broken toward the smaller alpha (the shipped
    # default) so the procedure is deterministic.
    full = max(Rs)
    cands = sorted({tag for v in sweep.values() for tag in v["cells"]
                    if int(split_tag(tag)["R"]) == full and float(split_tag(tag)["m"]) == 0.0})
    subs = sorted({k[0] for k in sweep})
    nested = {}
    if len(cands) > 1 and len(subs) > 2:
        print(f"\n[nested-LOSO] {len(cands)} candidate cells x {len(subs)} held-out subjects")
        picks = {}
        for hold in subs:
            train = [k for k in sweep if k[0] != hold]
            best, best_v = None, -1.0
            for tag in cands:
                vals = [sweep[k]["cells"][tag]["on"] for k in train
                        if tag in sweep[k]["cells"] and "on" in sweep[k]["cells"][tag]]
                if len(vals) < max(2, len(train) // 2):
                    continue
                mu = float(np.mean(vals))
                if mu > best_v + 1e-9:
                    best, best_v = tag, mu
            if best is None:
                continue
            held = [sweep[k]["cells"][best]["on"] for k in sweep
                    if k[0] == hold and best in sweep[k]["cells"] and "on" in sweep[k]["cells"][best]]
            picks[str(hold)] = {"cell": best, "train_mean": best_v,
                                "held_top1": float(np.mean(held)) if held else float("nan")}
        if picks:
            held_mean = float(np.nanmean([p["held_top1"] for p in picks.values()]))
            fixed_tag = f"R={full},a={args.curve_alpha:g},t={args.curve_tau},m=0"
            fixed_vals = [sweep[k]["cells"][fixed_tag]["on"] for k in sweep
                          if fixed_tag in sweep[k]["cells"] and "on" in sweep[k]["cells"][fixed_tag]]
            fixed = float(np.mean(fixed_vals)) if fixed_vals else float("nan")
            nested = {"picks": picks, "held_out_mean": held_mean, "fixed_config_mean": fixed,
                      "fixed_cell": fixed_tag, "optimism_pp": fixed - held_mean}
            print(f"{'held-out':>9} {'picked':>22} {'train':>7} {'held':>7}")
            for s, p in picks.items():
                print(f"{s:>9} {p['cell']:>22} {p['train_mean']:>7.2f} {p['held_top1']:>7.2f}")
            if not np.isfinite(fixed):
                # A MISSING cell must never read as a pass. The fixed tag is the headline cell;
                # if it is absent the JSON predates the `m` dimension or the grid is partial, and
                # a nan optimism would silently satisfy the `<= 0.5` branch below and print
                # "small". Name the failure instead.
                note = (f"UNDEFINED: fixed cell '{fixed_tag}' absent from the sweep -- "
                        "optimism not measured, NOT zero.")
            elif fixed - held_mean > 0.5:
                note = ("report the NESTED number as the honest headline; the fixed-config "
                        "mean is what a peeking selection would have claimed.")
            elif held_mean - fixed > 0.5:
                note = ("the fixed config was NOT overfit -- peeking did not inflate it (the "
                        "nested procedure would pick a different, better cell). Keep the fixed "
                        "config for reproducibility, but the grid's ceiling is the nested mean.")
            else:
                note = "selection optimism is small; the fixed config is safe to report."
            print(f"  fixed-config mean {fixed:.2f}  vs  nested-LOSO mean {held_mean:.2f} "
                  f"(optimism {fixed - held_mean:+.2f}pp)")
            print(f"  -> {note}")

    if not nested:
        verdict = "SELECTION OPTIMISM NOT MEASURED"
    elif not np.isfinite(nested.get("optimism_pp", float("nan"))):
        verdict = "SELECTION OPTIMISM NOT MEASURED (fixed cell absent)"
    elif nested["optimism_pp"] > 0.5:
        verdict = "SELECTION OPTIMISM PRESENT"
    else:
        verdict = "SELECTION OPTIMISM SMALL"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"n_checkpoints": len(sweep), "fidelity": gate_report,
             "curve_alpha": args.curve_alpha, "curve_tau": args.curve_tau,
             "curve_by_tau": curve, "curve_gates": gates, "grids": grids,
             "m1_source_template": m1,
             "nested": nested, "verdict": verdict},
            indent=2))
        print(f"\n[sum] wrote {args.out}  ({verdict})")


if __name__ == "__main__":
    main()
