"""Why does 80-trial averaging land on exactly chance while single trials do not?

`scripts/probe_signal.py` produced an internally inconsistent pair of numbers on the
within-subject regime: single-trial top1 = 0.0078 against a chance of 0.0050 (5 sigma
at 16,000 queries, so real), yet 80-trial averaging gave exactly 0.0050 -- one correct
concept out of 200.  Averaging repetitions of a stimulus improves SNR, so the second
number cannot be below the first unless something other than noise is being averaged.

Two candidate causes, and they call for opposite responses:

  A) A property of the *fit*: the probe whitens the PCA components, which amplifies
     the low-variance (noise) directions of a steeply decaying EEG spectrum.  A ridge
     fitted to amplified noise scores above chance on held-out single trials while its
     per-concept average, which is what a real decoder would use, stays at chance.
     -> the 5 sigma is an artefact and the data is *not* shown to be informative.

  B) A property of the *data*: the per-concept average of the EEG carries no more
     CLIP-relevant information than a single trial, which would mean the concept-
     specific component of the response is not consistent across repetitions.

A and B are distinguished by the between-concept versus within-concept spread of the
prediction, which is reported directly here, and by turning whitening off.

Reported per alpha:
  top1/top5                     the two protocols
  between/within ratio          variance of the per-concept mean prediction divided by
                                the mean within-concept variance.  This is the quantity
                                averaging is supposed to improve; if it does not grow
                                with `reps`, averaging is not extracting a consistent
                                component.
  margin                        cos(own target) - mean cos(other targets), for single
                                trials and for the average.  Above zero means the
                                correct target ranks better than an average one.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

from loso import paths
from loso.data import eeg as eeg_mod
from loso.data import things
from loso.data.targets import TargetStore

# `scripts/` is a directory of entry points, not a package, so the shared helpers are
# imported by path rather than by module name.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_signal import apply_ridge, load_test, load_trials, retrieval  # noqa: E402


def margins(pred: np.ndarray, gallery: np.ndarray, slots: np.ndarray,
            n_concepts: int, reps: int) -> dict[str, float]:
    p = pred / np.clip(np.linalg.norm(pred, axis=1, keepdims=True), 1e-8, None)
    g = gallery / np.clip(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-8, None)

    sim = p @ g.T
    own = sim[np.arange(len(slots)), slots]
    # The off-target mean per row: what the correct target has to beat.
    off_sum = sim.sum(axis=1) - own
    off_mean = off_sum / (sim.shape[1] - 1)
    single_margin = float((own - off_mean).mean())

    p_avg = p.reshape(n_concepts, reps, -1).mean(axis=1)
    p_avg = p_avg / np.clip(np.linalg.norm(p_avg, axis=1, keepdims=True), 1e-8, None)
    sim_avg = p_avg @ g.T
    own_avg = np.diag(sim_avg).copy()
    off_sum_avg = sim_avg.sum(axis=1) - own_avg
    off_mean_avg = off_sum_avg / (sim_avg.shape[1] - 1)
    avg_margin = float((own_avg - off_mean_avg).mean())

    # Between- vs within-concept spread of the prediction itself.  If averaging is
    # doing its job, the between-concept component dominates the within-concept one.
    block = p.reshape(n_concepts, reps, -1)
    between = float(block.mean(axis=1).var(axis=0, ddof=0).mean())
    within = float(block.var(axis=1, ddof=0).mean())
    # And the same for the raw similarity, which is what actually decides the ranking.
    sblock = sim.reshape(n_concepts, reps, -1)
    sim_between = float(sblock.mean(axis=1).var(axis=0, ddof=0).mean())
    sim_within = float(sblock.var(axis=1, ddof=0).mean())
    return {
        "single_margin": single_margin,
        "avg_margin": avg_margin,
        "between": between,
        "within": within,
        "between_over_within": between / max(within, 1e-12),
        "sim_between": sim_between,
        "sim_within": sim_within,
        "sim_between_over_within": sim_between / max(sim_within, 1e-12),
        "n_avg_hits": float((sim_avg.argmax(axis=1) == np.arange(n_concepts)).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-subject", default="sub-08")
    ap.add_argument("--alphas", default="1,30,300,3000")
    ap.add_argument("--pool", type=int, default=5)
    ap.add_argument("--n-components", type=int, default=512)
    ap.add_argument("--whiten", default="on,off")
    args = ap.parse_args()

    train_subjects, test_subject = things.loso_split(args.test_subject)
    store_train = TargetStore("train", names=("clip_image",))
    store_test = TargetStore("test", names=("clip_image",))
    n_test_concepts = store_test.shapes.n_images
    n_reps = paths.N_TEST_REPS

    per_subject, global_stats = eeg_mod.build_normalizers(
        train_subjects, "train_subjects",
        cache_path=paths.DATA_ROOT / f"norm_train_subjects_{len(train_subjects)}.json",
    )
    del per_subject
    norm = {s: global_stats for s in list(train_subjects) + [test_subject]}
    norm_keys = {s: s for s in norm}

    x_test, slots_test = load_test(test_subject, norm, norm_keys, args.pool)
    gallery = np.asarray(store_test.arrays["clip_image"], dtype=np.float32)
    x_tr, slots_tr, _ = load_trials([test_subject], "train", norm, norm_keys, args.pool)
    y_tr = np.asarray(store_train.gather(torch.from_numpy(slots_tr))["clip_image"],
                      dtype=np.float32)
    y_tr = y_tr / np.clip(np.linalg.norm(y_tr, axis=1, keepdims=True), 1e-8, None)

    # The gallery's own geometry sets the floor for any margin, so it is reported
    # alongside: a margin has to be read against how similar the targets are to each
    # other, not against zero.
    gu = gallery / np.clip(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-8, None)
    gc = gu @ gu.T
    print(f"[probe] gallery {gu.shape}: off-diag cosine mean="
          f"{float((gc.sum() - np.trace(gc)) / (len(gc) ** 2 - len(gc))):+.4f}  "
          f"chance top1={1 / n_test_concepts:.4f}")

    for whiten in [w.strip() == "on" for w in args.whiten.split(",")]:
        for alpha in [float(a) for a in args.alphas.split(",")]:
            mean = x_tr.mean(axis=0)
            std = x_tr.std(axis=0)
            std[std < 1e-6] = 1.0
            zs = ((x_tr - mean) / std).astype(np.float32)
            pca = PCA(n_components=args.n_components, svd_solver="randomized",
                      random_state=0, whiten=whiten).fit(zs)
            zm, ym = zs @ pca.components_.T, y_tr.mean(axis=0)
            zc = zm - zm.mean(axis=0)
            yc = y_tr - ym
            gram = zc.T @ zc
            # With whitening the PCA scores have unit variance, so the effective ridge
            # strength on the original features differs; alphas are swept rather than
            # translated so the comparison stays honest.
            gram[np.diag_indices_from(gram)] += alpha
            w = np.linalg.solve(gram.astype(np.float64),
                                (zc.T @ yc).astype(np.float64)).astype(np.float32)
            params = (mean, std, pca, (zm.mean(axis=0), ym, w))
            pred = apply_ridge(x_test, params)
            res = retrieval(pred, gallery, slots_test, n_test_concepts, n_reps)
            m = margins(pred, gallery, slots_test, n_test_concepts, n_reps)
            print(f"[probe] whiten={'on ' if whiten else 'off'} alpha={alpha:<7g} "
                  f"single top1={res['single']['top1']:.4f} top5={res['single']['top5']:.4f} | "
                  f"avg top1={res['avg']['top1']:.4f} top5={res['avg']['top5']:.4f} | "
                  f"margin s={m['single_margin']:+.4f} a={m['avg_margin']:+.4f} | "
                  f"b/w pred={m['between_over_within']:.3f} "
                  f"sim={m['sim_between_over_within']:.3f} hits={m['n_avg_hits']:.0f}",
                  flush=True)


if __name__ == "__main__":
    main()
