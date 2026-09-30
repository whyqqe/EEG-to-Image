#!/usr/bin/env python
"""One table for the three things that can be compared and the two that cannot.

Why this is a script and not a paragraph in a report
---------------------------------------------------
The numbers this project now produces come from three different programs with three
different output formats, and two of the pairs between them are NOT comparable. A
written summary drifts: someone reads "our Top-1 was X, SAMGA's was 26.22" out of a
table and subtracts them, and the subtraction was never valid to begin with. So the
caveats are emitted by the same code that emits the numbers, right next to them, and
the script refuses to print a difference it cannot justify.

The three sources
-----------------
  official   `third_party/SAMGA/train.py`, unmodified, via
             `scripts/run_samga_official_baseline.sh`. Its `result.csv` has BOTH
             selection protocols: `top1 acc` (final epoch) and `best top1 acc`
             (test-selected).
  epd arms   `scripts/epd/train.py` via `run_epd_loso_pipeline.sh`. One json per arm,
             `--select-last`, no test-set selection at all.

What is comparable, and what is not
-----------------------------------
  epd arm vs epd arm        YES. One flag apart by construction; that is what the arms
                            exist for.
  official vs epd arms      YES, with the selection caveat applied deliberately --
                            see below. This is the comparison the pipeline was built to
                            make, but it moves FIVE things at once (lr, batch,
                            augmentation, EEG encoder, head width) on top of the
                            intended difference, so a gap between them is a fact about
                            the two systems and not about any one of those five.
  official vs the PUBLISHED 26.22 / SCORE 53.23
                            NO. Those were produced on the Things-EEG InternViT
                            multi-level features (layers 20/24/28/32/36). This machine
                            has no InternViT and no network, so both the official run
                            and our arms are fed five CLIP ViT-H-14 layers (22/24/26/
                            28/30) mapped onto the same five slots. Under a different
                            image backbone the published number is not well-defined as
                            a target, so it is printed as a REFERENCE and never as a
                            difference.

The selection caveat, stated once
---------------------------------
SAMGA's `--early_stop_patience 10` evaluates on the test set every epoch and keeps the
best, so `best top1 acc` is selected ON the test set. Our arms use `--select-last` and
score the test set once. Therefore:
  * official `top1 acc`      vs an epd arm   -- the honest comparison (both last-epoch)
  * official `best top1 acc` vs an epd arm   -- INFLATES the official side by an unknown
                                                amount; shown only to bound it
This script prints both official rows and labels which one is the fair one, because the
unfair one is the number a reader is most likely to quote from their own `result.csv`.

Run:  python scripts/compare_inter_arms.py                 (default paths)
      python scripts/compare_inter_arms.py --target 8 --out-dir outputs/loso/sub08
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def find_official_results(out_root: Path) -> list[Path]:
    """Every non-smoke result.csv under the official root, oldest first.

    This used to return only the newest one, which was right while there could be only
    one kind of official run. There are now two, and they differ in the single variable
    that decides whether the headline number means anything: the image features. CLIP
    ViT-H-14 was the stand-in while InternViT was believed unavailable; InternViT-6B is
    what `inter.sh` actually names and what the paper's 26.22/34.4 are defined under.
    Keeping only the newest would delete the evidence that the substitution existed, so
    all of them are returned and `official_feature_set` labels each one.

    The smoke runs write real `result.csv` files into a `smoke/` subtree, and they are
    complete enough to be picked up by a naive glob -- a 1-epoch, 2-subject run. The
    exclusion is by path because their timestamps are newer than the real run's for as
    long as the real run is still going.
    """
    cands = [p for p in out_root.rglob("result.csv") if "smoke" not in p.parts]
    return sorted(cands, key=lambda p: p.stat().st_mtime)


def official_feature_set(result_dir: Path) -> str:
    """Which image features fed this run, read from the run's own train_config.json.

    Inferred rather than recorded separately: `train.py` already writes the resolved
    `image_feature_dir` into `train_config.json`, so the run carries its own provenance
    and there is no second place for it to drift out of sync.
    """
    cfg = result_dir / "train_config.json"
    if not cfg.is_file():
        return "unknown"
    try:
        d = json.loads(cfg.read_text())
    except Exception:
        return "unknown"
    fd = str(d.get("image_feature_dir", ""))
    if "internvit" in fd.lower():
        return f"internvit-6B ({d.get('image_feature_dim', d.get('feature_dim', '?'))}-d)"
    if "clip" in fd.lower():
        return "clip ViT-H-14 (substitute)"
    return fd or "unknown"


def read_officials(out_root: Path) -> list[dict]:
    rows = []
    for csv_path in find_official_results(out_root):
        row = next(csv.DictReader(csv_path.open()))
        log_path = csv_path.parent / "train.log"
        # The last `top5 acc` line in train.log is the FINAL epoch; `result.csv`'s `top1
        # acc` should agree with it. Where they disagree, one of the two is not the thing
        # this table says it is, so the check is printed rather than asserted away.
        log_final = None
        if log_path.is_file():
            hits = re.findall(r"top5 acc ([\d.]+)%\s+top1 acc ([\d.]+)%", log_path.read_text())
            if hits:
                log_final = (float(hits[-1][1]), float(hits[-1][0]))
        n_epochs = len(re.findall(r"top5 acc [\d.]+%", log_path.read_text())) if log_path.is_file() else None
        rows.append({
            "path": csv_path,
            "feature_set": official_feature_set(csv_path.parent),
            "final_top1": float(row["top1 acc"]),
            "final_top5": float(row["top5 acc"]),
            "best_top1": float(row["best top1 acc"]),
            "best_top5": float(row["best top5 acc"]),
            "best_epoch": int(row["best epoch"]),
            "log_final": log_final,
            "n_epochs": n_epochs,
        })
    return rows


def read_arms(out_dir: Path, target: int, arms: list[str] | None) -> list[dict]:
    rows = []
    pat = f"loso_sub{target:02d}_*_result.json"
    for p in sorted(out_dir.glob(pat)):
        arm = p.name[len(f"loso_sub{target:02d}_"):-len("_result.json")]
        if arms and arm not in arms:
            continue
        r = json.loads(p.read_text())
        t = r["test"]
        rows.append({
            "arm": arm,
            "top1": t["top1"], "top5": t["top5"], "n": t["n"],
            "ci95": t.get("ci95"), "mdd": t.get("min_detectable_diff"),
            "recovery": t.get("recovery") or {},
            "refs": r.get("reference_sota") or {},
            "epochs_run": r.get("epochs_run"),
            "seed": r.get("seed"),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="our arm jsons; defaults to outputs/loso/sub<target>")
    ap.add_argument("--official-root", type=Path,
                    default=ROOT / "outputs/samga_official")
    ap.add_argument("--arms", nargs="*", default=None)
    a = ap.parse_args()

    out_dir = a.out_dir or (ROOT / f"outputs/loso/sub{a.target:02d}")
    w = 92
    print("=" * w)
    print(f"INTER-SUBJECT COMPARISON -- held out sub-{a.target:02d}, 200-way")
    print("=" * w)

    officials = read_officials(a.official_root)
    arms = read_arms(out_dir, a.target, a.arms)
    # The InternViT run is the one that answers to the paper; if it exists, comparisons
    # against "official" use it, and any CLIP-fed run is shown separately as history.
    faithful = next((o for o in reversed(officials) if "internvit" in o["feature_set"].lower()), None)
    official = faithful or (officials[-1] if officials else None)

    if not officials:
        print("\n[official] no non-smoke result.csv found -- the baseline has not finished.")
    if not arms:
        print("\n[arms] no arm result json found -- the pipeline has not finished.")

    # ------------------------------------------------------------------ the main table
    print(f"\n  {'source':<34s} {'Top-1':>7s} {'Top-5':>7s}  {'selection':<26s}")
    for o in officials:
        tag = "official " + o["feature_set"]
        star = " *" if o is official and faithful else "  "
        print(f"  {tag:<32s}{star} {o['final_top1']:7.2f} {o['final_top5']:7.2f}  "
              f"{'last epoch (FAIR)':<26s}")
        print(f"  {'':<34s} {o['best_top1']:7.2f} {o['best_top5']:7.2f}  "
              f"{'test-selected (INFLATED)':<26s}")
    for r in arms:
        print(f"  {('arm ' + r['arm']):<34s} {r['top1']:7.2f} {r['top5']:7.2f}  "
              f"{'last epoch':<26s}")
    if faithful:
        print(f"\n  * = the InternViT-fed run: the only one whose absolute number is "
              f"defined under the paper's backbone.")

    for o in officials:
        print(f"\n  official detail [{o['feature_set']}]: {o['path'].parent}")
        print(f"    best epoch {o['best_epoch']} of {o['n_epochs']} "
              f"({'early stopped' if o['n_epochs'] and o['best_epoch'] < o['n_epochs'] else 'ran to the end'})")
        if o["log_final"] and abs(o["log_final"][0] - o["final_top1"]) > 1e-6:
            print(f"    !! train.log's final-epoch top1 ({o['log_final'][0]}) "
                  f"!= result.csv's 'top1 acc' ({o['final_top1']}); "
                  f"one of the two is not the final epoch.")

    # The feature substitution is a variable, so its effect is measured, not asserted.
    if faithful and len(officials) >= 2:
        prev = [o for o in officials if o is not faithful][-1]
        print(f"\n  FEATURE SUBSTITUTION (the deviation this project carried longest)")
        print(f"    InternViT-6B {faithful['final_top1']:.2f} vs {prev['feature_set']} "
              f"{prev['final_top1']:.2f}  ->  {faithful['final_top1'] - prev['final_top1']:+.2f} Top-1")
        print(f"    Same code, same preprocessing, same seed, same fold; only the frozen "
              f"visual backbone differs.")

    # ------------------------------------------------- the arm-to-arm comparison, guarded
    if len(arms) >= 2:
        print(f"\n  {'=' * (w - 4)}")
        print("  ARM vs ARM (the single-variable comparison -- each arm is one flag from the last)")
        base = arms[0]
        for b in arms[1:]:
            gap = b["top1"] - base["top1"]
            thr = max(base.get("mdd") or 0.0, b.get("mdd") or 0.0)
            verdict = ("ABOVE the resolution" if abs(gap) >= thr
                       else "BELOW the resolution -- call this 'no measured difference'")
            print(f"    {b['arm']} minus {base['arm']}: {gap:+.2f} Top-1  "
                  f"(threshold {thr:.2f})  -> {verdict}")

    # --------------------------------------- official vs arms, with the caveat attached
    if official and arms:
        print(f"\n  {'=' * (w - 4)}")
        print("  OFFICIAL vs OUR ARMS")
        print("    Fair form, both last-epoch:")
        for r in arms:
            print(f"      {r['arm']:<16s} {r['top1'] - official['final_top1']:+7.2f} Top-1 "
                  f"(our {r['top1']:.2f} vs official {official['final_top1']:.2f})")
        print(f"    NOT a valid comparison, shown only as a bound:")
        for r in arms:
            print(f"      {r['arm']:<16s} vs official BEST-epoch "
                  f"({official['best_top1']:.2f}, test-selected): {r['top1'] - official['best_top1']:+7.2f}")
        print(f"\n    This gap moves FIVE differences at once, not one: learning rate")
        print(f"    (1e-4 vs 5e-4), batch (1024 vs 512), augmentation (smooth vs full),")
        print(f"    EEG encoder (SAMGA TSConv vs EEGiT ViT), head width (512 vs 1024).")
        print(f"    It is a fact about the two systems; it is not evidence about any one of them.")

    # --------------------------------------------------------- the deployment ladder
    if any(r["recovery"] for r in arms):
        print(f"\n  {'=' * (w - 4)}")
        print("  SCORE TABLE 4 SHAPE -- frozen features, CPU, per arm")
        print(f"  {'arm':<16s} {'cosine':>7s} {'CSLS':>7s} {'+mom':>7s} "
              f"{'rec rho0':>9s} {'+idreg':>7s} {'lm rate':>8s}")
        for r in arms:
            rec = r["recovery"]
            if not rec:
                continue
            d = rec.get("recovery_diag") or {}
            abst = "  ABSTAINED" if d.get("abstained") else ""
            print(f"  {r['arm']:<16s} {rec['cosine']:7.2f} {rec['csls']:7.2f} "
                  f"{rec['moment_match_csls']:7.2f} {rec['recovery_rho0']:9.2f} "
                  f"{rec['recovery']:7.2f} {d.get('landmark_rate', float('nan')):8.2f}{abst}")
        print("    SCORE's own ladder: cosine 30.01 -> CSLS 39.08 -> +mom 43.80 -> "
              "+rec 50.98 -> +idreg 53.23")
        print("    'lm rate' is the fraction of queries that became landmarks; it is the "
              "observable that")
        print("    predicts whether recovery helps or actively hurts (see recover.py).")

    # ----------------------------------------------------------------- the references
    refs = (arms[0]["refs"] if arms else {}) or {}
    if refs:
        print(f"\n  {'=' * (w - 4)}")
        print("  PUBLISHED REFERENCE -- NOT COMPARABLE TO ANYTHING ABOVE")
        for k in ("ATM_inter", "SATTC_inter", "SAMGA_inter",
                  "SAMGA_published_inter", "SCORE_inter"):
            v = refs.get(k)
            if v:
                print(f"    {k:<14s} {v['top1']:6.2f} / {v['top5']:6.2f}   {v.get('note','')}")
        print(f"    Reason: those runs used Things-EEG InternViT-6B multi-level features, "
              f"so they are only")
        print(f"    addressable by an InternViT-fed run. While the features were missing, "
              f"every number")
        print(f"    here was CLIP-fed and none of them could be read against these rows; "
              f"only the")
        print(f"    DIFFERENCES between our own arms were claims. See the FEATURE "
              f"SUBSTITUTION line")
        print(f"    above for what that substitution was worth, measured rather than "
              f"assumed.")

        # The table above is ten-fold averages with a spread (SAMGA 5 seeds, SCORE 3
        # seeds). A single fold is one draw from that spread, so the fold-matched cell is
        # the only honest target for a one-fold run, and it happens to be published:
        # SAMGA's own Table 2 lists every held-out subject, and sub-08 is 28.7/59.5.
        # Without this line the natural comparison is our one fold against their ten-fold
        # mean, which charges us for a variance we cannot control and would make a
        # perfectly good reproduction look like a failure (or hide a real one).
        if a.target == 8 and faithful:
            print(f"\n  FOLD-MATCHED TARGET (the number this one-fold run answers to)")
            print(f"    SAMGA's own Table 2, held-out sub-08:      28.70 / 59.50   (5 seeds, "
                  f"best TEST-set epoch)")
            print(f"    SCORE's protocol, SAMGA encoder, 10-fold:  26.22 / 57.98   (3 seeds, "
                  f"final epoch)")
            print(f"    our InternViT run, sub-08, seed 2025:      "
                  f"{faithful['final_top1']:6.2f} / {faithful['final_top5']:6.2f}   "
                  f"(1 seed, final epoch)")
            print(f"    Our run is 1 seed of a 5-seed protocol and 1 fold of a 10-fold one, "
                  f"so treat a")
            print(f"    gap of a few points as 'not yet distinguishable', not as a finding. "
                  f"What this run")
            print(f"    CAN settle is whether the pipeline is faithful: with the features "
                  f"now matched, a")
            print(f"    number in this neighbourhood means the remaining machinery "
                  f"(preprocessing, router,")
            print(f"    coarse-to-fine schedule) is doing what the paper's does.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
