#!/usr/bin/env python
"""Build and validate the THINGS-EEG2 image index.

Writes `data/things/{train,test}_index.json` plus `concepts.json`, and verifies the
EEG array headers against the geometry the index assumes.  This is the pre-flight
for every other stage: if the image order does not match the EEG axes, all
downstream targets are mispaired and nothing else can detect it.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from loso import paths
from loso.data import things


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", default="sub-08",
                    help="subject whose EEG headers are verified (geometry is identical "
                         "across subjects, so any one subject is sufficient)")
    ap.add_argument("--verify-eeg", action="store_true", default=True)
    args = ap.parse_args()

    paths.ensure_dirs()

    for split in ("train", "test"):
        records = things.build_index(split)
        out = paths.THINGS_DIR / f"{split}_index.json"
        things.write_index(records, out)

        concepts = things.concept_records(records)
        counts = Counter(r.concept for r in records)
        dup_keys = [k for k, n in Counter(r.image_key for r in records).items() if n > 1]

        print(f"[{split}] {len(records)} images, {len(concepts)} concepts -> {out}")
        print(f"        images/concept: min={min(counts.values())} max={max(counts.values())}")
        print(f"        duplicate image_keys: {len(dup_keys)}")
        if dup_keys:
            raise SystemExit(f"[FATAL] duplicate image_key(s): {dup_keys[:5]}")
        if len(concepts) != len(set(concepts)):
            raise SystemExit(f"[FATAL] duplicate concept names in {split}")

        # Spot-check the ordering contract: index i must be concept i//K, image i%K.
        k = paths.N_IMAGES_PER_CONCEPT if split == "train" else 1
        for probe in (0, 1, k - 1, k, len(records) - 1):
            r = records[probe]
            expect_c, expect_i = probe // k, probe % k
            if (r.concept_index, r.image_index) != (expect_c, expect_i):
                raise SystemExit(
                    f"[FATAL] index contract violated at {probe}: "
                    f"got concept={r.concept_index} image={r.image_index}, "
                    f"expected concept={expect_c} image={expect_i}"
                )
        print(f"        index contract OK (probed 5 positions)")

        concepts_out = paths.THINGS_DIR / f"{split}_concepts.json"
        prompts = {c: things.concept_prompts(c) for c in concepts}
        concepts_out.write_text(json.dumps(prompts, indent=1))
        print(f"        concept prompts ({len(prompts)}) -> {concepts_out}")

    if args.verify_eeg:
        shapes = things.verify_eeg_geometry(args.subject)
        print(f"[eeg:{args.subject}] headers match: {shapes}")

    train_subjects, test_subject = things.loso_split("sub-08")
    print(f"[loso] test={test_subject}  train={train_subjects}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
