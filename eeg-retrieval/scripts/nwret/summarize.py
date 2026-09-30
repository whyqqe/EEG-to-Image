"""Aggregate per-arm results into a leaderboard.

Reads every `*_result.json` under --out-dir and ranks the arms, reporting both
the held-out-validation number (used for selection) and the test number (scored
once). Keeps the published SOTA line in the table so every arm is readable as a
fraction of the target rather than an isolated number.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(out_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(out_dir.glob("*_result.json")):
        if p.name in {"leaderboard.json"}:
            continue
        try:
            d = json.loads(p.read_text())
        except Exception as e:                       # noqa: BLE001
            print(f"[warn] unreadable {p.name}: {e}")
            continue
        if "test" not in d:
            print(f"[warn] {p.name} has no test block; skipped")
            continue
        rows.append(d)
    return rows


def fmt(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"


def align_target(d: dict) -> str:
    """Which image-tower layer this arm aligned to.

    Distinct from `layers`, which is the depth of the EEG encoder's own transformer.
    Conflating the two is what let the earlier sweep run on the wrong axis while its
    report looked like a depth study: every arm printed `layers [12]` and the column
    was read as the layer being tested.
    """
    tl = d.get("target_layer")
    if tl:
        return tl
    return "_pooled (final, cached)" if not d.get("target_features") else "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--sota-top1", type=float, default=94.8)
    ap.add_argument("--sota-top5", type=float, default=99.0)
    # The reference is the SUB-08 column, not the 10-subject average: this is a
    # single-subject study and subject 8 sits above the average for every method
    # in SAMGA Table 2, so the average understated the gap by ~3.5 points.
    ap.add_argument("--sota-name", default="SAMGA intra sub-08")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    rows = load(out_dir)
    rows.sort(key=lambda d: -d["test"].get("top1", -1))

    ridge = None
    rp = out_dir / "baseline_ridge.json"
    if rp.is_file():
        try:
            ridge = json.loads(rp.read_text())["test"]["top1"]
        except Exception:                              # noqa: BLE001
            ridge = None

    top1_sota, top5_sota = a.sota_top1, a.sota_top5

    md: list[str] = []
    md.append("# NW-Retrieval sub-08 leaderboard\n")
    md.append(f"Reference: **{a.sota_name}** = Top-1 {top1_sota} / Top-5 {top5_sota}\n")
    md.append("Protocol: 200-way zero-shot retrieval on THINGS-EEG2. Train repetitions (4)")
    md.append("and test repetitions (80) are averaged, as in `third_party/SAMGA`; the metric is")
    md.append("`module/util.py::retrieve_all` verbatim. Checkpoint selection uses a concept-level")
    md.append("holdout of the **training** concepts (150 of 1654), so the 200 test concepts are")
    md.append("never used for any decision. \n")

    if not rows:
        md.append("_No completed arms yet._\n")
    else:
        md.append("## Ranking by test Top-1\n")
        md.append("| # | arm | backbone | eeg layers | **align target** | aug | val Top-1 | **test Top-1** | test Top-5 | mean rank | % of SOTA | vs ridge |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for i, d in enumerate(rows, 1):
            t, v = d["test"], d.get("best_val", {})
            pct = 100.0 * t.get("top1", 0) / top1_sota
            if ridge is not None:
                delta = t.get("top1", 0) - ridge
                vs = f"{delta:+.2f} {'+' if delta > 0 else '-'}"
            else:
                vs = "-"
            md.append(
                f"| {i} | `{d.get('tag','?')}` | {d.get('backbone','?')} | "
                f"{d.get('layers','?')} | `{align_target(d)}` | {d.get('aug','?')} | "
                f"{fmt(v.get('top1'))} | **{fmt(t.get('top1'))}** | {fmt(t.get('top5'))} | "
                f"{fmt(t.get('mean_rank'),1)} | {pct:.0f}% | {vs} |"
            )
        md.append("")
        md.append("`eeg layers` is the depth of the EEG encoder's own transformer (the axis an "
                  "earlier sweep varied). `align target` is which layer of the frozen image "
                  "tower the EEG is asked to match -- the axis docs section 7 prices as the "
                  "largest single lever. They are different axes and were previously "
                  "conflated in this table.")

        if ridge is not None:
            n_pass = sum(1 for d in rows if d["test"].get("top1", 0) > ridge)
            md.append(f"**Ridge floor: {ridge:.2f} test Top-1** (closed-form linear map, "
                      f"same features and split). {n_pass} of {len(rows)} arms clear it.")
            if n_pass == 0:
                md.append("")
                md.append("> No arm beats a linear baseline. In that regime the ranking above is a "
                          "ranking of overfitting, not of representation quality -- read the `aug` "
                          "and `train_top1_inbatch` fields before drawing any conclusion.")
            md.append("")

        best = rows[0]
        b = best["test"]
        md.append("## Reading the result\n")
        md.append(f"Best arm: `{best.get('tag')}` ({best.get('backbone')}, eeg layers "
                  f"{best.get('layers')}, align target `{align_target(best)}`) "
                  f"-> Top-1 {fmt(b.get('top1'))}, Top-5 {fmt(b.get('top5'))}, mean rank {fmt(b.get('mean_rank'),1)}.")
        md.append("")
        md.append(f"Gap to {a.sota_name}: {top1_sota - b.get('top1',0):.2f} Top-1 points. "
                  f"The mean rank is the more informative of the two when Top-1 is far from "
                  f"saturation: random guessing gives mean rank 100.5 on a 200-way task.")
        md.append("")

        # Group by family so the structural questions stay readable.
        fam = {}
        for d in rows:
            bb = str(d.get("backbone", "?"))
            key = ("CLIP (semantic)" if "openclip" in bb else
                   "DINOv2 (structure)" if "dinov2" in bb else
                   "ViT-B/16 21k (reference)")
            fam.setdefault(key, []).append(d)
        md.append("## By backbone family\n")
        for k, ds in fam.items():
            best_k = max(ds, key=lambda d: d["test"].get("top1", -1))
            md.append(f"- **{k}**: best `{best_k.get('tag')}` layers {best_k.get('layers')} "
                      f"-> Top-1 {fmt(best_k['test'].get('top1'))} "
                      f"(mean rank {fmt(best_k['test'].get('mean_rank'),1)})")
        md.append("")

        md.append("## Generalisation gap (the binding constraint)\n")
        md.append("In-batch training Top-1 on the final epoch vs the validation Top-1 it was "
                  "selected on. A large gap means the arm is memorising, and its test number is "
                  "a measurement of that, not of the EEG-to-image map.\n")
        md.append("| arm | aug | final train Top-1 | best val Top-1 | gap (pts) |")
        md.append("|---|---|---|---|---|")
        for d in rows:
            h = d.get("history") or []
            if not h:
                continue
            tr = h[-1].get("train_top1_inbatch")
            bv = d.get("best_val", {}).get("top1")
            gap = f"{tr - bv:.1f}" if (tr is not None and bv is not None) else "-"
            md.append(f"| `{d.get('tag','?')}` | {d.get('aug','?')} | {fmt(tr)} | {fmt(bv)} | {gap} |")
        md.append("")

        # Alignment-target depth profile. This replaces an earlier "layer-depth" section
        # keyed on the EEG encoder's layer count, which was the wrong axis and printed
        # the same `12` for every arm while reading as if it were the layer under test.
        md.append("## Alignment-target depth profile\n")
        md.append("Which layer of the frozen image tower the EEG is asked to match. The reported "
                  "shape in this literature is an inverted U, and it is the precondition for "
                  "trusting a weighted multi-layer fusion on top.\n")

        probe_p = out_dir / "probe_layers.json"
        if probe_p.is_file():
            try:
                pr = json.loads(probe_p.read_text())
            except Exception:                          # noqa: BLE001
                pr = None
            if pr and pr.get("per_layer"):
                pl = pr["per_layer"]
                pool = pr.get("final_layer_baseline") or {}
                best = pr.get("best_by_val") or {}
                md.append(f"Closed-form ridge probe (val-selected, {pr.get('n_fit_concepts')} fit "
                          f"concepts, {pr.get('n_val_concepts')} val concepts x 10 slots). "
                          f"Ridge is untrained, so this curve is not confounded by overfitting, "
                          f"which is what the deep arms below cannot claim.\n")
                md.append("| align target | dim | lambda* | val Top-1 | test Top-1 | test Top-5 | mean rank |")
                md.append("|---|---|---|---|---|---|---|")
                for r in pl:
                    star = " *" if r["layer"] == "_pooled" else ""
                    md.append(f"| `{r['layer']}`{star} | {r.get('dim','-')} | {r.get('lam',0):.0e} | "
                              f"{fmt(r.get('val_top1'))} | {fmt(r.get('test_top1'))} | "
                              f"{fmt(r.get('test_top5'))} | {fmt(r.get('test_mean_rank'),1)} |")
                md.append("")
                md.append("(*) `_pooled` = the final projected layer, i.e. the cached features every "
                          "earlier arm used. Chance: Top-1 0.50, mean rank 100.5.\n")
                if pool and best:
                    d = best.get("test_top1", 0) - pool.get("test_top1", 0)
                    md.append(f"Peak: `{best.get('layer')}` at {fmt(best.get('val_top1'))} val / "
                              f"{fmt(best.get('test_top1'))} test, i.e. {d:+.2f} test Top-1 vs the "
                              f"final layer (SE on a 200-way Top-1 is ~2.8 points).\n")

        targets = [(align_target(d), d) for d in rows]
        md.append("Deep-model arms by alignment target (this is the A/B that tests whether the "
                  "probe's preference transfers):\n")
        md.append("| arm | align target | val Top-1 | test Top-1 | vs ridge |")
        md.append("|---|---|---|---|---|")
        for tag, d in sorted(targets, key=lambda kv: -kv[1]["test"].get("top1", -1)):
            dt = "-" if ridge is None else f"{d['test'].get('top1',0) - ridge:+.2f}"
            md.append(f"| `{tag}` | `{align_target(d)}` | {fmt(d.get('best_val',{}).get('top1'))} | "
                      f"{fmt(d['test'].get('top1'))} | {dt} |")
        md.append("")

    md.append("## Files\n")
    md.append(f"- `{out_dir}/<arm>_result.json` -- full per-arm record (history, protocol, wall time)")
    md.append(f"- `{out_dir}/leaderboard.json` -- machine-readable version of this table")
    md.append("- `outputs/logs/<arm>.log` -- per-arm training log")

    Path(a.md).write_text("\n".join(md))
    Path(a.json).write_text(json.dumps({
        "reference_sota": {"name": a.sota_name, "top1": top1_sota, "top5": top5_sota},
        "ridge_floor_test_top1": ridge,
        "n_arms": len(rows),
        "n_arms_beating_ridge": sum(1 for d in rows if ridge is not None
                                    and d["test"].get("top1", 0) > ridge),
        "ranking": [
            {"rank": i, "tag": d.get("tag"), "backbone": d.get("backbone"),
             "layers": d.get("layers"), "n_channels": d.get("n_channels"),
             "aug": d.get("aug"), "epochs_run": d.get("epochs_run"),
             "val_top1": d.get("best_val", {}).get("top1"),
             "test_top1": d["test"].get("top1"), "test_top5": d["test"].get("top5"),
             "test_mean_rank": d["test"].get("mean_rank"),
             "pct_of_sota": 100.0 * d["test"].get("top1", 0) / top1_sota,
             "beats_ridge": (None if ridge is None
                             else bool(d["test"].get("top1", 0) > ridge))}
            for i, d in enumerate(rows, 1)
        ],
    }, indent=2))

    print(f"[summary] {len(rows)} arms -> {a.md}")
    if rows:
        print(f"[summary] best: {rows[0].get('tag')} test_top1={rows[0]['test'].get('top1'):.2f} "
              f"(SOTA {top1_sota})")


if __name__ == "__main__":
    main()
