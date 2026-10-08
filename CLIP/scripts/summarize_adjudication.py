#!/usr/bin/env python
"""ADJUDICATION — is the concept-frame transduction (+15pp) a neural leak, or does it need no EEG?

THE QUESTION. `rep_cloud_scores(src_mix=1)` replaces the query-side EEG metric `de` with the
source subjects' EEG metric `de_ref`. That arm is worth +15pp (30/30 on the v8 grid, 30/30 on the
g3 grid). The open question is WHERE the worth comes from. `de_ref` correlates with the PUBLIC
image-gallery metric `di` at 0.78, while the target's own `de_t` correlates at 0.59, so the source
metric is simply a *better proxy for the gallery metric*. If that is the whole story, the arm needs
no SOURCE EEG at all -- the gallery metric is available at test -- and the "leak" worry is moot.

THE FOUR ARMS (one argument apart, `--fgw-src-ref-mode`). Every arm replaces the same slot at the
same `src_mix=1`; only the reference changes:

| arm | reference | what it tells us |
|---|---|---|
| `eeg` | source subjects' EEG metric | the arm under adjudication (the +15pp) |
| `gallery` | PUBLIC image-gallery metric `di` | if it reproduces the gain, **zero EEG is needed** |
| `self` | the target's own metric `de_t` | a NO-OP; must equal the baseline bit-for-bit |
| `rand` | fixed-seed random symmetric metric | the floor any real geometry must beat |

READING THE OUTCOME BEFORE RUNNING IT (pre-registered):
  * `gallery ≈ eeg` (+15pp) -> the gain is carried by the public gallery geometry. The source EEG
    is an *estimator* of it, not a leak: the arm is legitimate, and no source EEG is even needed.
  * `gallery << eeg` (near 0) -> the source EEG carries structure the public gallery metric does
    not, i.e. subject-shared EEG geometry. This is the reading that keeps the leak concern alive.
  * `self != baseline` -> the plumbing is broken; nothing else can be read.
  * `rand` positive -> the structural term is being driven by something other than the reference.

Run:
    python scripts/summarize_adjudication.py \
        --dir outputs/eval/adjudicate --out outputs/adjudication_summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODES = ["eeg", "gallery", "self", "rand"]


def base_tag(mode: str) -> str:
    # The mode suffix is appended to EVERY m value in the run, so the baseline carries it too
    # (except for `eeg`, whose suffix is the empty string).
    return ("+ T2 R=80,a=0.75,t=0.03,m=0" if mode == "eeg"
            else f"+ T2 R=80,a=0.75,t=0.03,m=0,ref={mode}")


def m1_tag(mode: str) -> str:
    return ("+ T2 R=80,a=0.75,t=0.03,m=1" if mode == "eeg"
            else f"+ T2 R=80,a=0.75,t=0.03,m=1,ref={mode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "outputs/eval/adjudicate"))
    ap.add_argument("--out", default=str(ROOT / "outputs/adjudication_summary.json"))
    args = ap.parse_args()

    print("=" * 88)
    print("ADJUDICATION — where does the concept-frame gain come from?")
    print("=" * 88)

    res: dict[str, dict] = {}
    for mode in MODES:
        files = sorted(glob.glob(str(Path(args.dir) / f"sub*_{mode}.json")))
        t1, t5, b1, b5, cells = [], [], [], [], []
        for f in files:
            rows = list(json.loads(Path(f).read_text())["checkpoints"].values())[0]["rows"]
            if base_tag(mode) not in rows or m1_tag(mode) not in rows:
                continue
            t1.append(rows[m1_tag(mode)]["top1"]); t5.append(rows[m1_tag(mode)]["top5"])
            b1.append(rows[base_tag(mode)]["top1"]); b5.append(rows[base_tag(mode)]["top5"])
            cells.append(Path(f).stem)
        if not t1:
            print(f"  {mode:<8s}: (no cells)")
            res[mode] = {"n": 0}
            continue
        t1a, b1a = np.asarray(t1, float), np.asarray(b1, float)
        t5a, b5a = np.asarray(t5, float), np.asarray(b5, float)
        d1 = t1a - b1a
        sd1 = float(d1.std(ddof=1)) if d1.size > 1 else 0.0
        tt1 = float(d1.mean() / (sd1 / np.sqrt(d1.size))) if sd1 > 0 else float("nan")
        d5 = t5a - b5a
        print(f"  {mode:<8s}: n={t1a.size:2d}  m=1 Top-1 {t1a.mean():6.2f}  (base {b1a.mean():6.2f})"
              f"  Δ {d1.mean():+6.2f}pp  t={tt1:5.2f}  {int((d1>0).sum())}/{d1.size}"
              f"   Top-5 {t5a.mean():6.2f} (Δ {d5.mean():+5.2f})")
        res[mode] = {"n": int(t1a.size), "cells": cells,
                     "m1_top1": float(t1a.mean()), "base_top1": float(b1a.mean()),
                     "delta_top1": float(d1.mean()), "t_top1": tt1,
                     "pos": int((d1 > 0).sum()),
                     "m1_top5": float(t5a.mean()), "delta_top5": float(d5.mean())}

    if res.get("eeg", {}).get("n") and res.get("self", {}).get("n"):
        self_ok = abs(res["self"]["delta_top1"]) < 1e-9
        print(f"\n  plumbing (self == baseline bit-for-bit): "
              f"{'OK' if self_ok else 'BROKEN'}")
        res["plumbing_self_bitflat"] = bool(self_ok)

    if res.get("eeg", {}).get("n") and res.get("gallery", {}).get("n"):
        e, g = res["eeg"]["delta_top1"], res["gallery"]["delta_top1"]
        if g >= 0.6 * e:
            verdict = ("NO-EEG-NEEDED: the public gallery metric reproduces the gain -> the arm is "
                       "an estimator of the gallery geometry, not a neural leak.")
        elif g <= 0.2 * e:
            verdict = ("EEG-SPECIFIC: the public gallery metric does NOT reproduce the gain -> the "
                       "source EEG carries subject-shared structure beyond the gallery metric, and "
                       "the leak concern stands.")
        else:
            verdict = ("SPLIT: the public gallery metric recovers part of the gain -> mixed reading; "
                       "report both numbers.")
        print(f"\n  eeg Δ {e:+.2f}pp vs gallery Δ {g:+.2f}pp  ->  {verdict}")
        res["verdict"] = verdict

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"\n[adjudicate] wrote {args.out}")


if __name__ == "__main__":
    main()
