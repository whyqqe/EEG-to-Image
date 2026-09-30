"""Prefight facts NW4 depends on.  All cheap, all must hold or the design changes.

Q1  Are THINGS-EEG2's 10 reps per concept the SAME image or 10 DIFFERENT images?
    -> decides whether instance-level (trial) alignment is meaningful at all.
       Same image  => clip_img rows 0..9 are identical, instance NCE is degenerate.
       Different   => instance NCE gives EEG a real, transferable regression target.

Q2  Do the four attribute banks really vary WITHIN a concept (per-trial)?
    ->>> if they are constant per concept, A0 degenerates to the same bucket problem
        that killed S1, and only the *geometry* of the target space differs.

Q3  Is the test split one image per concept (so 200 rows == 200 concepts)?
    -> the IP-condition and eval protocol assume exactly one row per test concept.

Q4  Do the modality banks share the test concept ORDER?  A2 projects onto a bank via
    top-K nearest neighbours; a permuted bank silently mislabels the prior.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

NB_ROOT = Path(__file__).resolve().parents[2]


def l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def within_vs_across(x: np.ndarray, reps: int, n_concepts: int, label: str) -> None:
    """Mean cos among the reps of a concept vs mean cos across concepts."""
    if x.shape[0] != n_concepts * reps:
        print(f"  {label:<26} rows {x.shape[0]} != {n_concepts}x{reps}; skipped")
        return
    xn = l2n(x.reshape(n_concepts, reps, -1))
    s = xn @ xn.transpose(0, 2, 1)                      # (C, reps, reps)
    within = s.mean((1, 2))
    # across: concept c vs concept c+1 (a cheap, unbiased neighbour pair)
    a, b = l2n(xn.mean(1)), l2n(xn.mean(1))
    across = (a[:-1] * b[1:]).sum(1)
    print(f"  {label:<26} within-concept cos={within.mean():.4f} "
          f"(min {within.min():.4f} max {within.max():.4f})  "
          f"adjacent-concept cos={across.mean():.4f}")
    print(f"  {'':<26} -> reps are "
          f"{'IDENTICAL (same image)' if within.mean() > 0.999 else 'DIFFERENT images'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond-cache", type=str,
                    default=str(NB_ROOT / "outputs/gem/cond_cache"))
    ap.add_argument("--g2-targets", type=str,
                    default=str(NB_ROOT / "outputs/g2/targets"))
    ap.add_argument("--clip-text-dir", type=str,
                    default=str(NB_ROOT / "outputs/nda_ss/sub-08/clip_text"))
    args = ap.parse_args()

    cc = Path(args.cond_cache)
    g2 = Path(args.g2_targets)
    ct = Path(args.clip_text_dir)

    n_flat = np.load(ct / "train" / "text_flat_clip.npy", mmap_mode="r").shape[0]
    n_con = np.load(ct / "train" / "text_concept_clip.npy", mmap_mode="r").shape[0]
    reps = n_flat // n_con
    print(f"[shape] {n_con} concepts x {reps} reps = {n_flat} train rows")

    print("\nQ1/Q2 within- vs across-concept structure:")
    for name, p in (("clip_img1024_train", cc / "clip_img1024_train.npy"),
                    ("clip_depth1024_train", cc / "clip_depth1024_train.npy"),
                    ("clip_edge1024_train", cc / "clip_edge1024_train.npy"),
                    ("attr_overall_train", g2 / "sem_overall_train.npy"),
                    ("attr_subject_train", g2 / "sem_subject_train.npy"),
                    ("attr_background_train", g2 / "sem_background_train.npy"),
                    ("attr_detail_train", g2 / "sem_detail_train.npy"),
                    ("text_flat_clip(concept text)", ct / "train" / "text_flat_clip.npy")):
        if not p.is_file():
            print(f"  {name:<26} MISSING {p}")
            continue
        within_vs_across(np.load(p).astype(np.float32), reps, n_con, name)

    vroot = NB_ROOT / "data/things_eeg/image_feature/ViT-H-14"
    for lv in ("image", "GaussianBlur"):
        p = (vroot / f"{lv}_train.npy") if lv == "image" else (vroot / lv / "train.npy")
        if p.is_file():
            a = np.load(p).astype(np.float32).reshape(n_flat, -1)
            within_vs_across(a, reps, n_con, f"vith_{lv}_train")

    print("\nQ3 test split shape / one-image-per-concept:")
    for name, p in (("clip_img1024_test", cc / "clip_img1024_test.npy"),
                    ("attr_overall_test", g2 / "sem_overall_test.npy"),
                    ("attr_background_test", g2 / "sem_background_test.npy"),
                    ("text_concept_clip_test", ct / "test" / "text_concept_clip.npy")):
        if p.is_file():
            print(f"  {name:<26} {np.load(p, mmap_mode='r').shape}")

    print("\nQ4 do test banks share the concept ORDER? (cos to the test concept-name bank)")
    tn = l2n(np.load(ct / "test" / "text_concept_clip.npy").astype(np.float32))
    for name, p in (("clip_img1024_test", cc / "clip_img1024_test.npy"),
                    ("clip_depth1024_test", cc / "clip_depth1024_test.npy"),
                    ("clip_edge1024_test", cc / "clip_edge1024_test.npy"),
                    ("attr_overall_test", g2 / "sem_overall_test.npy"),
                    ("attr_subject_test", g2 / "sem_subject_test.npy"),
                    ("attr_background_test", g2 / "sem_background_test.npy"),
                    ("attr_detail_test", g2 / "sem_detail_test.npy")):
        if not p.is_file():
            continue
        z = l2n(np.load(p).astype(np.float32))
        diag = (z * tn).sum(1)                      # row i vs concept-name i
        rolled = (z * np.roll(tn, 1, axis=0)).sum(1)
        # if the orders agree, row i must be closest to name i far more often than
        # to a shifted name -- but attributes are weak, so also report the RANK of
        # the true concept name.
        sim = z @ tn.T
        rank = np.argsort(-sim, 1)
        true_rank = np.array([np.where(rank[i] == i)[0][0] + 1 for i in range(len(z))])
        print(f"  {name:<26} cos(own name)={diag.mean():+.4f} "
              f"cos(rolled)={rolled.mean():+.4f}  median rank of own name="
              f"{np.median(true_rank):.0f}/200")

    # the strongest available order check: image bank vs text concept bank
    print("\nQ4b strongest order check (image branch vs concept-name text):")
    zi = l2n(np.load(cc / "clip_img1024_test.npy").astype(np.float32))
    sim = zi @ tn.T
    order = np.argsort(-sim, 1)
    t1 = float((order[:, 0] == np.arange(200)).mean())
    print(f"  clip_img1024_test argmax vs its own concept name: top1={t1:.4f} "
          f"(high => the banks DO share the 200-concept order)")


if __name__ == "__main__":
    main()
