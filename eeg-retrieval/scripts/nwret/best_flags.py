#!/usr/bin/env python
"""Emit the flag set of the best regime so the pipeline can carry it forward.

The sub-08 study has a hard ordering constraint: backbone and layer scans are only
interpretable once *some* configuration beats the closed-form ridge baseline. So
the pipeline runs a small regime search first, then re-runs the wide scans under
whichever regime won.

Selection uses the **validation** Top-1, never the test Top-1. Test is scored once
per arm and printed; if the pipeline also *chose* on it, every downstream scan
would inherit a selection leak, and the final number would be optimistic in a way
no error bar could reveal.

Usage
-----
    python scripts/nwret/best_flags.py --out-dir outputs/sub08 --tags F1 F2 F3 F4 F5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tags", nargs="*", default=[],
                    help="restrict the choice to these tags (default: every result)")
    ap.add_argument("--min-val-top1", type=float, default=0.0,
                    help="refuse to emit flags if the best val Top-1 is below this")
    a = ap.parse_args()

    out = Path(a.out_dir)
    rows = []
    for p in sorted(out.glob("*_result.json")):
        if p.name == "leaderboard.json":
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:                                  # noqa: BLE001
            continue
        if a.tags and d.get("tag") not in a.tags:
            continue
        v = (d.get("best_val") or {}).get("top1")
        if v is None:
            continue
        rows.append((v, d.get("tag"), d))

    if not rows:
        print(f"[flags] no results among {a.tags or 'any'}; emitting defaults")
        print("--aug full")
        return

    v, tag, d = max(rows, key=lambda r: r[0])
    if v < a.min_val_top1:
        print(f"[flags] best val {v:.2f} ({tag}) is below --min-val-top1 "
              f"{a.min_val_top1}; emitting defaults", file=__import__("sys").stderr)
        print("--aug full")
        return

    flags = [f"--aug {d.get('aug', 'full')}"]
    if d.get("freeze_all"):
        flags.append("--freeze-all")
    elif d.get("freeze_blocks"):
        flags.append(f"--freeze-blocks {int(d['freeze_blocks'])}")
    if d.get("stage1_epochs") is not None:
        flags.append(f"--stage1-epochs {int(d['stage1_epochs'])}")
    if d.get("softplus") is False:
        flags.append("--no-softplus")

    print(f"[flags] best VAL regime: {tag} (val {v:.2f}, test "
          f"{d.get('test', {}).get('top1', float('nan')):.2f})", file=__import__("sys").stderr)
    print(" ".join(flags))


if __name__ == "__main__":
    main()
