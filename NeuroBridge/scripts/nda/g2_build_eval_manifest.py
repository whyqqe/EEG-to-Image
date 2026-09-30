#!/usr/bin/env python3
"""Build the G2 evaluation manifest across all folds, from BOTH output layouts.

Two layouts exist, and neither can be changed retroactively:

  canonical  MA/<sub-XX>/generation/<variant>/generated        (multi-fold folds)
  legacy     OUT/generation/g2_<proto>_<variant>/generated     (the first G2 job,
             launched before folds were namespaced by subject)

Rather than rewrite either, this script indexes what is actually on disk and emits
one manifest plus a canonical symlink tree for sub-08's legacy rows. The symlinks
are what let `eval_pooled_fid.py` compute the standard pooled number (fake = all
folds' generations, real = the 200 unique test images) without moving any data.

Each row carries `fold` and `variant` metadata so downstream analysis can group by
either, and rows whose generation directory is incomplete are skipped rather than
evaluated on a partial set.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

VARIANTS = ("direct", "cfm0", "direct_nc0", "direct_np")
COMPLETE_MARKER = "199.png"          # 200 test images, 0..199


def is_complete(gdir: Path) -> bool:
    return (gdir / COMPLETE_MARKER).is_file()


def add_row(rows: list[dict], tag: str, gdir: Path, fold: str, variant: str,
            protocol: str) -> None:
    if not is_complete(gdir):
        return
    rows.append({"tag": tag, "display": f"{fold}/{variant}", "gen_dir": str(gdir),
                 "fold": fold, "variant": variant, "protocol": protocol})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="the G2 output root")
    ap.add_argument("--multi-root", default="", help="defaults to <out>/multi")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--link-canonical", default="1",
                    help="create <multi>/sub-XX/generation/<variant> symlinks for legacy rows")
    args = ap.parse_args()

    out = Path(args.out)
    mroot = Path(args.multi_root) if args.multi_root else (out / "multi")
    rows: list[dict] = []
    linked: list[str] = []

    # ---- canonical (multi-fold) layout ----
    for sub in sorted(mroot.glob("sub-*")):
        if not sub.is_dir():
            continue
        fold = sub.name
        for v in VARIANTS:
            g = sub / "generation" / v / "generated"
            add_row(rows, f"g2_{fold}_{v}", g, fold, v, "inter_loso")

    # ---- legacy layout from the first single-subject job ----
    # Its tags look like g2_intra_direct / g2_inter_cfm0 / g2_inter_direct_np.
    legacy_root = out / "generation"
    legacy_fold = os.environ.get("LEGACY_FOLD", "sub-08")
    for proto in ("intra", "inter"):
        for v in VARIANTS:
            g = legacy_root / f"g2_{proto}_{v}" / "generated"
            if not is_complete(g):
                continue
            tag = f"g2_{legacy_fold}_{proto}_{v}"
            add_row(rows, tag, g, legacy_fold, v, proto)
            if args.link_canonical == "1":
                # The canonical tree expects one variant name per fold. The legacy
                # INTER rows are the fold's cross-subject result, so they become
                # `inter_<variant>`; the intra rows are kept distinct because they
                # answer a different question and must not be pooled with LOSO.
                vname = v if proto == "inter" else f"{proto}_{v}"
                link = mroot / legacy_fold / "generation" / vname
                if not link.exists():
                    link.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        link.symlink_to(g.parent)
                        linked.append(f"{link} -> {g.parent}")
                    except OSError as e:
                        print(f"[WARN] symlink failed {link}: {e}")

    man = {"protocol": "standard7", "rows": rows, "avg_rows": [],
           "note": ("folds are namespaced by subject; inter_* rows are LOSO "
                    "(held-out subject unseen during training)"),
           "symlinks_created": linked}
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(man, indent=2), encoding="utf-8")

    print(f"[manifest] {len(rows)} complete rows -> {args.manifest}")
    by_fold: dict[str, list[str]] = {}
    for r in rows:
        by_fold.setdefault(r["fold"], []).append(r["variant"])
    for k in sorted(by_fold):
        print(f"  {k}: {sorted(by_fold[k])}")
    if linked:
        print(f"[manifest] created {len(linked)} canonical symlinks for pooled FID")
    if not rows:
        raise SystemExit("[FATAL] no complete generation rows found; nothing to evaluate")


if __name__ == "__main__":
    main()
