"""Recover arm results that were written under train.py's default tag.

The bug these recover from
--------------------------
run_arm.sh validates the per-arm tag against `arm_tag()` but never passed it to
train.py, so `--tag` stayed empty and train.py fell back to its default
`{backbone}_L{layers}`. Two consequences, both silent:

  1. The arm's tag exists only in the *log*, not in the result file's name. The
     pipeline's resume check looks for `{tag}_result.json`, so those arms would
     re-run forever -- the file it looks for can never appear.
  2. Every arm with the same `--layers` shares one default tag, so they overwrite
     each other. T22, T28, MTmean and MTrouted all ran `--layers 12`; only the last
     one to finish survived.

This migrates by *content*, never by filename, because the filename is exactly what
cannot be trusted. Each arm's configuration fingerprint (EEG depth, fusion mode,
target layer set, target fusion) identifies it uniquely among the arms that ran.

K49 and K98 are the exception: they differ only in `--n-time-windows`, which the
result record does not store, so they are indistinguishable from their JSON alone.
That is a real gap in the record and is reported as such rather than guessed at.

Usage:  python scripts/migrate_mistagged_results.py [--out-dir DIR] [--apply]
Without --apply it only reports what it would do.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# arm -> the exact fields its result record must show for the mapping to be accepted.
# Copied by hand from run_arm.sh's arms, not derived from them: this is an
# independent check, and deriving it from the same source would defeat the purpose.
FINGERPRINTS: dict[str, dict] = {
    "T22":      dict(layers=[12], fusion_mode="routed",  target_layers=["block22"],
                     target_fusion="single"),
    "T28":      dict(layers=[12], fusion_mode="routed",  target_layers=["block28"],
                     target_fusion="single"),
    "MTmean":   dict(layers=[12], fusion_mode="routed",  target_fusion="mean",
                     target_layers=["block22", "block24", "block26", "block28"]),
    "MTrouted": dict(layers=[12], fusion_mode="routed",  target_fusion="routed",
                     target_layers=["block22", "block24", "block26", "block28"]),
    "E04":      dict(layers=[4],  fusion_mode="routed",  target_layers=["block26"],
                     target_fusion="single"),
    "E08":      dict(layers=[8],  fusion_mode="routed",  target_layers=["block26"],
                     target_fusion="single"),
    "FEU":      dict(layers=[8, 10, 12], fusion_mode="uniform", target_layers=["block26"],
                     target_fusion="single"),
    "FER":      dict(layers=[8, 10, 12], fusion_mode="routed",  target_layers=["block26"],
                     target_fusion="single"),
}
# Deliberately absent: K49 and K98. See the module docstring -- they differ only in a
# field the record does not keep, so any mapping here would be a guess.

TAGS = {
    "T22": "align_block22", "T28": "align_block28",
    "MTmean": "target_fuse_mean", "MTrouted": "target_fuse_routed",
    "E04": "eeg_l04", "E08": "eeg_l08",
    "FEU": "eeg_fuse_uniform", "FER": "eeg_fuse_routed",
}


def matches(rec: dict, fp: dict) -> bool:
    return all(rec.get(k) == v for k, v in fp.items())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/sub08")
    ap.add_argument("--apply", action="store_true",
                    help="actually copy the files; without this, only report")
    args = ap.parse_args()

    d = Path(args.out_dir)
    # Only files carrying train.py's default tag are suspect.
    candidates = [p for p in sorted(d.glob("*_result.json")) if p.name.startswith("timm:")]
    print(f"scanning {len(candidates)} default-tagged result file(s) in {d}")
    print()

    claimed: dict[str, Path] = {}
    for p in candidates:
        try:
            rec = json.loads(p.read_text())
        except Exception as exc:
            print(f"  {p.name}: unreadable ({exc})")
            continue
        hits = [arm for arm, fp in FINGERPRINTS.items() if matches(rec, fp)]
        if len(hits) != 1:
            # Zero hits means the fingerprint matches no known arm; more than one
            # means this file alone does not identify its arm. Either way, guessing
            # would put a wrong name on a real number.
            print(f"  {p.name}: {len(hits)} matching arm(s) {hits} -> NOT migrated")
            continue
        arm = hits[0]
        rec["tag"] = TAGS[arm]
        rec["migrated_from"] = p.name
        rec["migrated_reason"] = ("run_arm.sh did not forward --tag; train.py used its "
                                  "default tag and arms sharing --layers overwrote each other")
        dest = d / f"{TAGS[arm]}_result.json"
        claimed[arm] = dest
        print(f"  {p.name}  ->  {dest.name}")
        print(f"      test_top1={rec['test']['top1']}  layers={rec['layers']}  "
              f"fusion={rec['fusion_mode']}  target_fusion={rec['target_fusion']}")
        if dest.exists():
            print(f"      destination already exists -- refusing to overwrite")
            continue
        if args.apply:
            dest.write_text(json.dumps(rec, indent=2))
            # The checkpoint is only needed to re-score, and re-scoring reads the
            # tag in the filename, so carry it across when it is still around.
            for suffix in ("_best.pt",):
                src_ck = p.with_name(p.name.replace("_result.json", suffix))
                if src_ck.is_file():
                    shutil.copy2(src_ck, d / f"{TAGS[arm]}{suffix}")
                    print(f"      also copied checkpoint -> {TAGS[arm]}{suffix}")

    print()
    missing = [a for a in FINGERPRINTS if a not in claimed]
    if missing:
        print("arms with NO recoverable file (overwritten by another arm sharing "
              "--layers, or not yet run):")
        for a in missing:
            print(f"  {a:9} ({TAGS[a]})")
    print()
    print("K49/K98 cannot be migrated automatically: the record does not store "
          "--n-time-windows, so\n  a default-tagged file whose config is "
          "layers=[12]/single/block26 could be either one.")
    if args.apply:
        print("applied. These arms will now be skipped on the next submission.")
    else:
        print("dry run only -- pass --apply to write the files.")


if __name__ == "__main__":
    main()
