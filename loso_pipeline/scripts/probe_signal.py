"""Is the CLIP signal linearly reachable from this EEG at all?

Why this exists
---------------
Two Stage-2 runs have now failed the same way: every contrastive term sits exactly on
its uniform plateau (`img` at 6.20 against a constant-encoder baseline of 6.25) while
the anti-collapse terms *decrease*, i.e. the regularisers report success while the
representation collapses.  That is a contradiction, and it has two very different
explanations:

  H1  the architecture/objective cannot extract the signal   -> fix the model
  H2  the signal is not in the inputs as we have assembled them -> fix the data

Guessing between them has already cost two runs, so this script settles it with a
*linear* reference that has none of our architecture in it.  A ridge map from the
normalised EEG to the CLIP image target needs no encoder, no projector, no
temperature and no regulariser; if it retrieves above chance, the EEG/target pairing
is sound and any failure downstream is the model's.

What it reports
---------------
For each regime (within-subject, cross-subject) and each ridge strength, the standard
200-way test protocol of Stage 2: single-trial top1/top5 over 16,000 queries, and the
80-trial-averaged top1/top5 over 200.  `chance` is 1/200 = 0.005 for both.

`shuffled` is the control that makes the rest readable: the targets are permuted
across images before fitting, which destroys any real pairing while leaving every
shape and every scale untouched.  A regime whose real score does not separate from its
shuffled score has found nothing, whatever the absolute number looks like.

This script is a reference, not a component: nothing in the pipeline imports it.  It
exists to answer one question and it is meant to be re-run after any change to the
preprocessing, the target store or the index mapping, because those are exactly the
things whose failure is silent.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from loso import paths
from loso.data import eeg as eeg_mod
from loso.data import things
from loso.data.targets import TargetStore


def pool_time(x: np.ndarray, factor: int) -> np.ndarray:
    """Average consecutive time samples.

    (N, C, T) -> (N, C, T // factor).  Keeps the ERP shape while cutting the feature
    count, which is what makes the cross-subject fit affordable at 595,440 trials.
    """
    if factor <= 1:
        return x
    n, c, t = x.shape
    usable = (t // factor) * factor
    return x[:, :, :usable].reshape(n, c, t // factor, factor).mean(axis=3)


def load_trials(subjects: list[str], split: str, norm: dict, norm_keys: dict,
                pool: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack every trial of every subject as (N, C*T) plus slots and subject ids."""
    xs, slots, sids = [], [], []
    for sid, subj in enumerate(subjects):
        arr = eeg_mod.load_eeg(subj, split)
        flat, flat_slots = eeg_mod.flatten_trials(arr)
        # The normaliser is a two-tensor affine applied per channel; doing it once in
        # numpy over the whole split is far cheaper than per-trial on the GPU path.
        stats = norm[norm_keys[subj]]
        center = stats.center.numpy()[:, None]
        scale = stats.scale.numpy()[:, None]
        x = (np.asarray(flat, dtype=np.float32) - center) / scale
        if pool > 1:
            x = pool_time(x, pool)
        xs.append(x.reshape(len(x), -1))
        slots.append(flat_slots.astype(np.int64))
        sids.append(np.full(len(x), sid, dtype=np.int64))
    return (np.concatenate(xs), np.concatenate(slots), np.concatenate(sids))


def load_test(subject: str, norm: dict, norm_keys: dict, pool: int
              ) -> tuple[np.ndarray, np.ndarray]:
    """The held-out subject's test split as (N, C*T) plus its concept slot per trial."""
    arr = eeg_mod.load_eeg(subject, "test")
    flat, _ = eeg_mod.flatten_trials(arr)
    stats = norm[norm_keys[subject]]
    center = stats.center.numpy()[:, None]
    scale = stats.scale.numpy()[:, None]
    x = (np.asarray(flat, dtype=np.float32) - center) / scale
    if pool > 1:
        x = pool_time(x, pool)
    # (200, 1, 80) -> every trial of concept c carries slot c, which is also the row
    # of the test target arrays it must be retrieved against.
    slots = np.repeat(np.arange(arr.shape[0]), arr.shape[2]).astype(np.int64)
    return x.reshape(len(x), -1), slots


def fit_ridge(x_train: np.ndarray, y_train: np.ndarray, alpha: float,
              n_components: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray,
                                                          np.ndarray, np.ndarray]:
    """Standardise, reduce with PCA, then solve ridge in closed form.

    PCA is unsupervised so it cannot leak the label, and it is what makes the solve a
    `n_components`-sized system instead of a 15,750-sized one.  Returns the pieces
    needed to apply the same map to held-out trials.
    """
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-6] = 1.0
    xs = (x_train - mean) / std
    # Randomised SVD: the full SVD of a (595k, 3150) matrix is not worth the time.
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, svd_solver="randomized",
              random_state=seed, whiten=True)
    zs = pca.fit_transform(xs).astype(np.float32)
    # Ridge with an explicit intercept: the CLIP targets are not centred, and forcing
    # the map through the origin would waste capacity on recovering the offset.
    zm = zs.mean(axis=0)
    ym = y_train.mean(axis=0)
    zc = zs - zm
    yc = y_train - ym
    d = zc.shape[1]
    # (Z^T Z + alpha I)^-1 Z^T Y, solved via Cholesky on the d x d system.
    gram = zc.T @ zc
    gram[np.diag_indices_from(gram)] += alpha
    rhs = zc.T @ yc
    w = np.linalg.solve(gram.astype(np.float64), rhs.astype(np.float64)).astype(np.float32)
    return mean, std, pca, (zm, ym, w)


def apply_ridge(x: np.ndarray, params) -> np.ndarray:
    mean, std, pca, (zm, ym, w) = params
    zs = pca.transform(((x - mean) / std).astype(np.float32)).astype(np.float32)
    return (zs - zm) @ w + ym


def retrieval(pred: np.ndarray, gallery: np.ndarray, slots: np.ndarray,
              n_concepts: int, reps: int) -> dict[str, float]:
    """200-way cosine retrieval, single-trial and `reps`-trial averaged."""
    p = pred / np.clip(np.linalg.norm(pred, axis=1, keepdims=True), 1e-8, None)
    g = gallery / np.clip(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-8, None)

    sim = p @ g.T                                   # (N, n_concepts)
    rank = np.argsort(-sim, axis=1)
    hit1 = rank[:, 0] == slots
    hit5 = (rank[:, :5] == slots[:, None]).any(axis=1)
    single = {"top1": float(hit1.mean()), "top5": float(hit5.mean())}

    # Averaging the repetitions of a concept *before* retrieval: the high-SNR protocol
    # the literature usually quotes, and a different number from the single-trial one.
    order = np.argsort(slots, kind="stable")
    p_avg = p[order].reshape(n_concepts, reps, -1).mean(axis=1)
    sim_avg = p_avg @ g.T
    rank_avg = np.argsort(-sim_avg, axis=1)
    avg = {"top1": float((rank_avg[:, 0] == np.arange(n_concepts)).mean()),
           "top5": float((rank_avg[:, :5] == np.arange(n_concepts)[:, None]).any(axis=1).mean())}
    return {"single": single, "avg": avg}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-subject", default="sub-08")
    ap.add_argument("--alphas", default="1,10,100,1000")
    ap.add_argument("--pool", type=int, default=5,
                    help="time pooling factor; 5 gives 63*50=3150 features")
    ap.add_argument("--n-components", type=int, default=1024)
    ap.add_argument("--max-train-trials", type=int, default=0)
    ap.add_argument("--control", action="store_true",
                    help="also fit a target-shuffled control")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    paths.ensure_dirs()
    train_subjects, test_subject = things.loso_split(args.test_subject)
    n_subjects = len(train_subjects)
    print(f"[probe] test={test_subject} train={n_subjects} subjects "
          f"pool={args.pool} pca={args.n_components}", flush=True)

    store_train = TargetStore("train", names=("clip_image",))
    store_test = TargetStore("test", names=("clip_image",))
    dim = store_train.dims["clip_image"][-1]
    n_test_concepts = store_test.shapes.n_images
    n_test_reps = paths.N_TEST_REPS
    print(f"[probe] CLIP dim={dim} test gallery={n_test_concepts} "
          f"reps={n_test_reps} chance={1 / n_test_concepts:.4f}", flush=True)

    per_subject, global_stats = eeg_mod.build_normalizers(
        train_subjects, "train_subjects",
        cache_path=paths.DATA_ROOT / f"norm_train_subjects_{n_subjects}.json",
    )
    del per_subject
    norm = {s: global_stats for s in list(train_subjects) + [test_subject]}
    norm_keys = {s: s for s in norm}

    x_test, slots_test = load_test(test_subject, norm, norm_keys, args.pool)
    gallery = np.asarray(store_test.arrays["clip_image"], dtype=np.float32)
    print(f"[probe] test trials {x_test.shape}  gallery {gallery.shape}", flush=True)

    # --- regime A: within-subject -------------------------------------------------
    # Trains on the held-out subject's own training trials.  This is the sensitive
    # test for H2: if the EEG<->image pairing or the normalisation were wrong, this
    # is the regime where it would show, because it has the most signal available.
    x_tr, slots_tr, _ = load_trials([test_subject], "train", norm, norm_keys, args.pool)
    print(f"[probe] within-subject train {x_tr.shape}", flush=True)
    y_tr = np.asarray(store_train.gather(torch.from_numpy(slots_tr))["clip_image"],
                      dtype=np.float32)
    # The objective is cosine-based, so the ridge regresses onto unit-norm targets.
    y_tr = y_tr / np.clip(np.linalg.norm(y_tr, axis=1, keepdims=True), 1e-8, None)

    # --- regime B: cross-subject --------------------------------------------------
    x_cs, slots_cs, _ = load_trials(train_subjects, "train", norm, norm_keys, args.pool)
    if args.max_train_trials and len(x_cs) > args.max_train_trials:
        pick = np.random.default_rng(0).choice(len(x_cs), args.max_train_trials,
                                               replace=False)
        x_cs, slots_cs = x_cs[pick], slots_cs[pick]
    print(f"[probe] cross-subject train {x_cs.shape}", flush=True)
    y_cs = np.asarray(store_train.gather(torch.from_numpy(slots_cs))["clip_image"],
                      dtype=np.float32)
    y_cs = y_cs / np.clip(np.linalg.norm(y_cs, axis=1, keepdims=True), 1e-8, None)

    results: dict = {"config": vars(args), "chance": 1 / n_test_concepts,
                     "regimes": {}}
    for regime, (x_trn, y_trn) in (("within_subject", (x_tr, y_tr)),
                                   ("cross_subject", (x_cs, y_cs))):
        entry: dict = {}
        for alpha in [float(a) for a in args.alphas.split(",") if a.strip()]:
            params = fit_ridge(x_trn, y_trn, alpha, args.n_components)
            pred = apply_ridge(x_test, params)
            res = retrieval(pred, gallery, slots_test, n_test_concepts, n_test_reps)
            entry[f"alpha={alpha:g}"] = res
            # The similarity margin is what distinguishes "retrieving a little" from
            # "every query maps to the same place and the ranking is arbitrary".
            p = pred / np.clip(np.linalg.norm(pred, axis=1, keepdims=True), 1e-8, None)
            g = gallery / np.clip(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-8, None)
            own = (p * g[slots_test]).sum(axis=1).mean()
            off = p.mean(axis=0) @ g.T
            print(f"[probe] {regime:15s} alpha={alpha:<7g} "
                  f"single top1={res['single']['top1']:.4f} "
                  f"top5={res['single']['top5']:.4f} | "
                  f"avg{args.pool and n_test_reps} top1={res['avg']['top1']:.4f} "
                  f"top5={res['avg']['top5']:.4f} | "
                  f"cos(own)={own:.4f} off_diag_max={off.max():.4f}", flush=True)
            entry["cos_own"] = float(own)
        if args.control:
            # Permuting the rows destroys the pairing but not the shapes or scales, so
            # any score that survives this is an artefact of the protocol.
            rng = np.random.default_rng(0)
            perm = rng.permutation(len(y_trn))
            params = fit_ridge(x_trn, y_trn[perm], 100.0, args.n_components)
            pred = apply_ridge(x_test, params)
            entry["shuffled_control"] = retrieval(pred, gallery, slots_test,
                                                  n_test_concepts, n_test_reps)
            sc = entry["shuffled_control"]
            print(f"[probe] {regime:15s} shuffled ctrl  "
                  f"single top1={sc['single']['top1']:.4f} | "
                  f"avg top1={sc['avg']['top1']:.4f}", flush=True)
        results["regimes"][regime] = entry

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"[probe] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
