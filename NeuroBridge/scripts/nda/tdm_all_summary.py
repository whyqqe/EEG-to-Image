"""Aggregate the all-subject TDM-DT run.  Runnable at any time; reads only JSON,
so it is safe to re-run while a job is in flight and it cannot alter results.

WHY THIS IS A SEPARATE FILE
    It used to be a heredoc inside `run_tdm_all.sh`, and it crashed there on a
    filename parse (`basename[2:4]` on `s01_tdm_ll_self.json` yields `'1_'`, not
    `'1'`), which turned a fully successful 10-subject night into an exit-1 job with
    no report at all.  A stage that only READS finished results must never be able to
    invalidate the results, so it lives outside the job now and is re-runnable.

WHAT IT ANSWERS, IN ORDER
    1. every row, averaged over the subjects that have it
    2. THE PAIRED CLAIM: full vs ablation, per subject, with wins and a paired SEM.
       This is the headline because both arms come from the same code, the same
       rows, and the same generation settings, so the difference is the mechanisms.
    3. per-subject mechanism read-out: did each arm actually train, what did the
       fused condition's identifiability look like, what did the gates learn
    4. per-subject Gate 0 verdicts: does each mechanism still have a PREMISE, or was
       the premise only ever true on sub-08
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics as st
import sys

OUT = os.environ.get("OUT", "/project/peilab/why/NeuroBridge/outputs/tdm_all")
NB_ROOT = os.environ.get("NB_ROOT", "/project/peilab/why/NeuroBridge")

METS = ("pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "swav", "fid")


def subject_of(path: str) -> int:
    """`s01_tdm_ll_self.json` -> 1, `s00_sdedit_ll.json` -> 0 (a reference row).

    The previous version sliced `[2:4]`, which for `s01_...` is `'1_'` and raises
    `ValueError: invalid literal for int()`.  Match the digits explicitly and take
    the WHOLE run of them, so `s10_...` keeps working too.
    """
    m = re.match(r"s(\d+)_", os.path.basename(path))
    if not m:
        raise ValueError(f"unrecognised eval filename: {path}")
    return int(m.group(1))


def load(tag: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for p in sorted(glob.glob(os.path.join(OUT, "eval", f"s??_{tag}.json"))):
        try:
            s = subject_of(p)
        except ValueError as e:                                    # noqa: PERF203
            print(f"  [warn] {e}")
            continue
        d = json.load(open(p))
        out[s] = {k: d.get(k) for k in METS}
    return out


def main() -> int:
    rows = ["tdm_ll_self", "tdm_abl_self", "tdm_th_star", "tdm_ll_gen",
            "sdedit_ll_095", "frla_off", "frla_uniform", "frla_frla"]
    data = {t: load(t) for t in rows}
    subs = sorted({s for t in rows for s in data[t] if s != 0})

    print("=" * 96)
    print("1. EVERY ROW.  `fid` and `swav` are LOWER-is-better; the rest are higher.")
    print("=" * 96)
    hdr = f"{'row':<16}" + "".join(f"{k:>10}" for k in METS) + f"{'n_subj':>8}"
    print(hdr)
    print("-" * len(hdr))
    for t in rows:
        v = data[t]
        cells = []
        for k in METS:
            xs = [v[s][k] for s in v if v[s].get(k) is not None]
            cells.append(f"{st.mean(xs):>10.4f}" if xs else f"{'-':>10}")
        n = len([s for s in v if v[s].get("pixcorr") is not None])
        print(f"{t:<16}" + "".join(cells) + f"{n:>8}")

    # reference rows are sub-08 only.  Printed separately and NEVER averaged into
    # the 10-subject rows, because their images came from an older pipeline.
    for tag, path in (("REF sdedit_ll", os.path.join(OUT, "eval", "s00_sdedit_ll.json")),
                      ("REF g3f_selfgate",
                       os.path.join(OUT, "eval", "s00_g3f_ll_selfgate.json"))):
        if os.path.isfile(path):
            m = json.load(open(path))
            print(f"{tag:<16}" + "".join(
                f"{m[k]:>10.4f}" if m.get(k) is not None else f"{'-':>10}"
                for k in METS) + f"{1:>8}")

    print()
    print("=" * 96)
    print("2. THE PAIRED CLAIM: full - ablation, per subject.")
    print("   Same code, same rows, same generation settings; only the mechanisms differ.")
    print("   NOTE the sign conventions: for pixcorr/ssim/alex/inception/clip/swav "
          "positive = better,")
    print("   for fid LOWER is better so a NEGATIVE delta favours the full arm.")
    print("=" * 96)
    full, abl = data["tdm_ll_self"], data["tdm_abl_self"]
    paired = sorted(set(full) & set(abl))
    if not paired:
        print("  [FATAL] no subject has both arms; nothing can be claimed")
        return 1
    print(f"  paired subjects: {paired}  (n={len(paired)})")
    print(f"  {'metric':<10}{'mean delta':>13}{'sem':>9}{'sd':>9}"
          f"{'subjects_won':>15}{'better?':>9}")
    for k in METS:
        d = [full[s][k] - abl[s][k] for s in paired
             if full[s].get(k) is not None and abl[s].get(k) is not None]
        if not d:
            continue
        sem = st.stdev(d) / len(d) ** 0.5 if len(d) > 1 else float("nan")
        if k == "fid":
            won = sum(1 for x in d if x < 0)
            verdict = "full" if st.mean(d) < 0 else "abl"
        else:
            won = sum(1 for x in d if x > 0)
            verdict = "full" if st.mean(d) > 0 else "abl"
        print(f"  {k:<10}{st.mean(d):>+13.4f}{sem:>9.4f}{st.stdev(d) if len(d)>1 else 0:>9.4f}"
              f"{f'{won}/{len(d)}':>15}{verdict:>9}")

    # a single composite so "did it help" is one number rather than eight, and one
    # that a reader cannot cherry-pick from.  Only metrics where the full arm wins
    # on the paired mean are counted, and the count is also printed.
    wins = sum(1 for k in METS
               if (lambda d: bool(d) and ((st.mean(d) < 0) if k == "fid" else (st.mean(d) > 0)))(
                   [full[s][k] - abl[s][k] for s in paired
                    if full[s].get(k) is not None and abl[s].get(k) is not None]))
    print(f"  -> full arm is better on {wins}/{len(METS)} metrics on the paired mean")

    print()
    print("=" * 96)
    print("3. MECHANISM READ-OUT (per subject, from the training reports)")
    print("=" * 96)
    for arm, base in (("full", f"{OUT}/tdm"), ("ablation", f"{OUT}/tdm_abl")):
        print(f"  --- {arm} arm")
        for p in sorted(glob.glob(f"{base}/sub-??/tdm_report.json")):
            d = json.load(open(p))
            sd = re.search(r"sub-(\d+)", p).group(1)
            mech = d.get("mechanisms", {})
            g = mech.get("granularity_time_claim", {}) or {}
            mt = g.get("mean_time_ms", {}) or {}
            hub = d.get("hubness", {}) or {}
            disc = d.get("ip_fused_disc", {}) or {}
            best = d.get("best", {}) or {}
            flags = ("trained" if d.get("trained") else "NOT-TRAINED")
            print(f"    sub-{sd} {flags} grad_skips={d.get('grad_skips')} "
                  f"best_score={best.get('score', float('nan')):.4f} "
                  f"fused2way={disc.get('twoway', float('nan')):.4f} "
                  f"hub_skew={hub.get('hub_skew', float('nan')):.2f}")
            if mt:
                order = " < ".join(f"{gname}:{mt[gname]:.0f}ms"
                                   for gname in sorted(mt, key=lambda x: mt[x]))
                print(f"          gate mean time  {order}")
                print(f"          claim {'PASS' if g.get('pass') else 'FAIL'} "
                      f"(detail after overall: "
                      f"{g.get('pass_detail_after_overall')})")
            if isinstance(mech.get("dla"), dict):
                dl = mech["dla"]
                print(f"          DLA mean|tau| ms/band="
                      f"{[round(v, 2) for v in dl.get('per_band_mean_abs_ms', [])]} "
                      f"nondeg={dl.get('pass_non_degenerate')} LF>HF={dl.get('pass_lf_gt_hf')}")
            elif "dla" in mech:
                # the ablation arm disables the mechanism and records WHY as a string
                print(f"          DLA: {mech['dla']}")
            if isinstance(mech.get("rsd"), dict):
                rs = mech["rsd"]
                od = rs.get("per_band_offdiag_norm", [])
                print(f"          RSD offdiag={[round(v, 3) for v in od]}")
            elif "rsd" in mech:
                print(f"          RSD: {mech['rsd']}")

    print()
    print("=" * 96)
    print("4. GATE 0: does each mechanism still have a PREMISE? (per subject)")
    print("   'the premise holds on sub-08' is exactly the single-subject result this")
    print("   project has been burned by, so this is measured for every subject.")
    print("=" * 96)
    for p in sorted(glob.glob(f"{NB_ROOT}/outputs/tdm/gate0_sub??.json")):
        d = json.load(open(p))
        j = d.get("G1_jitter", {}) or {}
        l = d.get("G2_linearity", {}) or {}
        v = d.get("verdict", {}) or {}
        sd = d.get("subject", "?")
        parts = []
        if "sd_ms" in j:
            parts.append(f"jitter sd={j['sd_ms']:.2f}ms pass={j['pass']}")
        if "r2_linear" in l:
            parts.append(f"gamma linR2={l['r2_linear']:.3f} "
                         f"(shuffle {l.get('r2_shuffled', float('nan')):.3f}) "
                         f"RSD={l.get('pass_rsd')}")
            parts.append(f"alpha_global={l.get('globalness_alpha', float('nan')):.3f} vs "
                         f"gamma_global={l.get('globalness_gamma', float('nan')):.3f} "
                         f"DNG={l.get('pass_dng')}")
        g3 = d.get("G3_timecourse", {}) or {}
        if g3.get("skipped"):
            parts.append("G3 skipped")
        elif "ordering" in g3:
            o = g3["ordering"]
            parts.append(f"G3 detail>overall={o.get('pass_detail_after_overall')}")
        print(f"  sub-{sd}: " + " | ".join(parts))
        print(f"          verdict: " + "  ".join(f"{k}={val}" for k, val in v.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
