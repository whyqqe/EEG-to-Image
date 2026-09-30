"""Inter-subject target-layer probe: fit on 9 subjects, score the held-out one.

Why this is not `probe_targets.py`
----------------------------------
`probe_targets.py` fits and scores ONE subject (subject 08, 17 channels, a
150-concept holdout). Every number this project has about "which CLIP layer should
the EEG align to" therefore comes from that setting, and section 8 of
`PROTOCOL_INTER.md` records the contradiction it produced: Shallow Alignment puts
ViT-H-14's best inter-subject layer at L11 (32.3% relative depth, 19.0 Top-1) while
our own probe peaks at block24/26 (val 17.07/17.33, test 28.50/27.50). Two readings
were left open -- their sweep sampled ~10 evenly spaced layers, and ridge may prefer a
different depth than a trained encoder -- and both can only be settled by measuring
under the protocol the claim belongs to.

That protocol is inter-subject, so this fits the ridge on NINE subjects' training
trials and scores the tenth subject's test trials. Three things change:

  * the fit matrix is 9x larger. Nine pooled subjects have a much fatter spectrum than
    one subject's 4x1504 rows, so the optimal ridge strength moves, and a λ tuned on
    one subject would be wrong here.
  * the EEG must be per-subject standardised AND MVNN-whitened, or nine subjects at
    native amplitude make the pooled covariance a statement about electrode impedance
    rather than about vision.
  * the held-out subject is scored by a model that has never seen it, which is the
    whole point: the intra-subject 28.50 Top-1 is the number that fails to transfer.

λ selection without touching the target
---------------------------------------
No-validation-split is a *training* protocol; a ridge still needs a λ. Picking it on
the held-out subject's test set would be the leak SCORE and Shallow Alignment avoid,
so λ comes from a source-side holdout: 10% of the training concepts are dropped from
the fit and the nine SOURCE subjects are scored on them as an n_val-way retrieval
task. The target subject contributes to neither side of that choice.

Holding out concepts rather than a subject is the cheaper option and the right one
here, because λ only has to be good enough to rank layers. A subject-level fold would
generalise the *choice* slightly better but costs a factorisation per fold, and at
these dimensions the factorisation is the entire cost.

Two factorisations for the whole sweep, not 32 x len(lams)
----------------------------------------------------------
The expensive part of a ridge is the factorisation and it does not depend on λ. With
the Gram eigendecomposition ``X^T X = V diag(s^2) V^T`` the ridge weight is
``diag(s / (s^2 + lam))``, so every λ is a diagonal rescale of one decomposition, and
the only per-target product is ``V^T X^T Y``:

    prediction = (X_test V) diag(s/(s^2+lam)) (V^T X_fit^T Y_fit)

So every layer, every λ and both fits (selection and report) share two eigen-
decompositions. That is what makes a full sweep minutes instead of hours, and it is
why the Gram matrix is formed explicitly instead of letting a least-squares solver
re-factorise per λ. The Gram is accumulated in row chunks so the 4 GiB fit matrix is
never resident on the GPU at once.

Read it as a ceiling, not a score
---------------------------------
This is a closed-form linear probe: no encoder, no training, no nonlinearity. Its
absolute Top-1 is far below a trained model's, and comparing it to SAMGA's 26.22 would
be comparing a linear map to a deep network. What it answers is *relative*: whether
block11 carries more linearly-decodable inter-subject signal than block26, and whether
fusion beats either. Those are rank questions, and rank is what a probe is for.

Run (needs a GPU and the full caches):
    python scripts/epd/probe_inter.py --target-subject 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epd import config                                          # noqa: E402
from epd.data import load_subject_std                           # noqa: E402


def l2n(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def two_way(sim: np.ndarray) -> float:
    """Two-way identification accuracy from a square (n, n) similarity matrix.

    The rule the rest of the project uses: a query counts only if it is the argmax of
    its own row AND of its own column. Reimplemented in three lines rather than
    imported so the probe carries no dependency on the training-side metrics module.
    """
    n = sim.shape[0]
    return float(((sim.argmax(1) == np.arange(n)) & (sim.argmax(0) == np.arange(n))).mean())


def row_acc(sim: np.ndarray) -> float:
    """Row-argmax accuracy. The same as `two_way` for a square matrix, but defined for
    the rectangular selection case, where the gallery is the val concepts only."""
    return float((sim.argmax(1) == np.arange(sim.shape[0])).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-subject", type=int, default=8)
    ap.add_argument("--source-subjects", type=int, nargs="*", default=None,
                    help="default: every subject 1..10 except the target")
    ap.add_argument("--layers", type=int, nargs="+",
                    default=[9, 11, 13, 15, 17, 21, 24, 26, 31],
                    help="CLIP ViT-H-14 block indices. The default samples the range "
                         "rather than listing candidates, because the two papers "
                         "disagree about WHERE the peak is and a curve is what "
                         "distinguishes 'L11 is best' from 'the curve is still rising "
                         "through L11'.")
    ap.add_argument("--features",
                    default=str(config.OUTPUTS / "features" / "clip_h14_layers"))
    ap.add_argument("--channels", default="all")
    ap.add_argument("--mvnn", default="train", choices=["off", "train", "test"])
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="fraction of SOURCE concepts held out to select lambda")
    ap.add_argument("--split-seed", type=int, default=2025)
    ap.add_argument("--fusion", nargs="*",
                    default=["17+24", "24+26", "17+24+26", "11+17+24+26"],
                    help="'+'-joined layer groups, also probed as concatenated "
                         "L2-normalised features. Fusion is the one place Shallow "
                         "Alignment and SAMGA agree -- both report a combination "
                         "beating any single layer -- so a single layer is not the "
                         "only hypothesis the sweep should spend its budget on.")
    ap.add_argument("--fit-limit", type=int, default=0,
                    help="cap the number of fit rows (smoke runs only)")
    ap.add_argument("--chunk", type=int, default=4096,
                    help="rows per Gram accumulation step; bounds GPU memory")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import torch

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    tgt = a.target_subject
    srcs = a.source_subjects or [s for s in range(1, 11) if s != tgt]
    if tgt in srcs:
        raise SystemExit(f"--target-subject {tgt} appears in --source-subjects {srcs}")
    if not 0.0 <= a.val_frac < 0.5:
        raise SystemExit(f"--val-frac must be in [0, 0.5), got {a.val_frac}")

    print(f"[inter] holds out sub-{tgt:02d}; fits on {len(srcs)} subjects {srcs}")
    print(f"[inter] device={dev}  mvnn={a.mvnn}  channels={a.channels}")

    chan = None if a.channels == "all" else config.CHANNELS_OCCIPITO_PARIETAL
    n_time = config.N_TIMEPOINTS
    n_test = config.N_TEST_CONCEPTS
    n_conc = config.N_TRAIN_CONCEPTS
    n_slots = config.N_IMAGES_PER_CONCEPT
    n_rows_per_subj = n_conc * n_slots

    # ---------------- EEG: nine fit subjects, one held out
    # Sources are whitened from their own TRAINING residuals and the held-out subject
    # from its TEST residuals. That asymmetry is `load_loso`'s and it is the protocol:
    # the target subject's training split is what the fold excludes.
    t0 = time.time()
    blocks = []
    for s in srcs:
        tr_s, _ = load_subject_std(s, chan, mvnn=a.mvnn)
        if tr_s.shape[1] != n_slots:
            raise SystemExit(f"sub-{s:02d} train has {tr_s.shape[1]} images/concept, "
                             f"expected {n_slots}")
        blocks.append(tr_s)
        print(f"  sub-{s:02d} {tr_s.shape} cached", flush=True)
    _, te_eeg = load_subject_std(tgt, chan,
                                 mvnn=("test" if a.mvnn != "off" else "off"), verbose=True)
    n_ch = te_eeg.shape[-2]
    print(f"[inter] EEG loaded in {time.time() - t0:.0f}s ({n_ch} channels)")

    X = np.empty((len(srcs) * n_rows_per_subj, n_ch * n_time), dtype=np.float32)
    for i, blk in enumerate(blocks):
        X[i * n_rows_per_subj:(i + 1) * n_rows_per_subj] = blk.reshape(n_rows_per_subj, -1)
    del blocks
    n_rows = X.shape[0]
    if a.fit_limit:
        X = np.ascontiguousarray(X[:a.fit_limit])
        n_rows = X.shape[0]
        print(f"[inter] --fit-limit -> {n_rows} fit rows")
    print(f"[inter] fit matrix {X.shape} ({X.nbytes / 2**30:.2f} GiB)")

    # Row r of X is (subject, concept, image) -- subject-major, then concept-major.
    # Only valid without --fit-limit, which is why that flag prints a warning below.
    concept_of_row = np.tile(np.repeat(np.arange(n_conc), n_slots), len(srcs))[:n_rows]
    slot_of_row = np.tile(np.arange(n_slots), len(srcs) * n_conc)[:n_rows]

    # ---------------- the lambda split: concepts, inside the sources only
    rng = np.random.default_rng(a.split_seed)
    perm = rng.permutation(n_conc)
    n_val = max(2, int(round(a.val_frac * n_conc)))
    val_c = np.sort(perm[:n_val])
    fit_c = np.sort(perm[n_val:])
    is_val_c = np.zeros(n_conc, dtype=bool)
    is_val_c[val_c] = True
    row_in_sel = is_val_c[concept_of_row] & (slot_of_row == 0)
    row_in_fit = ~is_val_c[concept_of_row]
    if a.fit_limit:
        print("[warn ] --fit-limit breaks the concept/slot row layout the lambda split "
              "relies on; the selection numbers below are not meaningful. Smoke only.")
    print(f"[inter] lambda selection: fit on {len(fit_c)} concepts "
          f"({int(row_in_fit.sum())} rows), score {len(val_c)}-way on {int(row_in_sel.sum())}")

    if row_in_sel.sum() == 0:
        raise SystemExit("the lambda split selected no rows: --fit-limit is too small "
                         "to reach the val concepts")

    # ---------------- image targets
    def target_conc(k: int, split: str) -> np.ndarray:
        """(n_conc, n_slots, D) L2-normalised, the same treatment the project's
        alignment target gets. Normalising per layer before concatenating is what
        makes a fusion group a fusion rather than a majority vote by scale."""
        p = Path(a.features) / split / f"block{k:02d}.npy"
        if not p.is_file():
            raise SystemExit(f"missing layer cache {p}")
        return l2n(np.load(p).astype(np.float32))

    def build_Y(ks: list[int]) -> tuple[np.ndarray, np.ndarray]:
        tr = np.concatenate([target_conc(k, "train").reshape(n_rows_per_subj, -1)
                             for k in ks], axis=1)
        te = np.concatenate([target_conc(k, "test").reshape(n_test, -1)
                             for k in ks], axis=1)
        return tr, te

    def expand(tr_conc: np.ndarray) -> np.ndarray:
        """One image feature per (concept, image), repeated for every subject.

        Image j of concept c is ONE stimulus, and all nine subjects responded to it.
        Tiling is what makes cross-subject structure real: rows (s, c, j) and (s', c, j)
        share a target, so the ridge is forced to put them in the same place. An
        `np.repeat` here would pair each subject with a different concept's image --
        training converges and the number is noise.
        """
        reps = (n_rows + tr_conc.shape[0] - 1) // tr_conc.shape[0]
        return np.tile(tr_conc, (reps, 1))[:n_rows]

    # ---------------- standardise: fit stats, applied to both sides
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    X -= mu
    X /= sd
    mu_1d, sd_1d = mu.reshape(-1), sd.reshape(-1)
    # The held-out subject is standardised with the FIT statistics, never its own test
    # moments. A subject calibrated by its own test statistics is the thing SCORE shows
    # is worth 27 Top-1 points, and it must not be smuggled into a comparison whose
    # entire purpose is to measure the uncalibrated geometry.
    te_flat = te_eeg[:, 0].reshape(n_test, -1)
    te_flat = ((te_flat - mu_1d) / sd_1d).astype(np.float32)
    del te_eeg

    # ---------------- the linear algebra
    def cross(rows: np.ndarray | None, Y: np.ndarray) -> "torch.Tensor":
        """X^T Y (or X[rows]^T Y[rows]) accumulated in chunks."""
        out = None
        n = n_rows if rows is None else len(rows)
        for i in range(0, n, a.chunk):
            sl = slice(i, min(i + a.chunk, n))
            xc = torch.from_numpy(np.ascontiguousarray(X[sl] if rows is None else X[rows[sl]])).to(dev)
            yc = torch.from_numpy(np.ascontiguousarray(Y[sl] if rows is None else Y[rows[sl]])).to(dev)
            g = xc.T @ yc
            out = g if out is None else out + g
            del xc, yc
        return out

    def factorise(rows: np.ndarray | None) -> tuple["torch.Tensor", "torch.Tensor"]:
        """eigh of X^T X -> (singular values, eigenvectors), both descending.

        Only the Gram matrix is needed. `eigh` rather than `svd` because the Gram is
        symmetric positive semi-definite by construction, so its eigenvectors are
        orthogonal and the singular values are exactly their square roots -- which is
        the identity the ridge weight below depends on.

        Accumulated in row chunks so the 4 GiB fit matrix never has to be resident on
        the GPU: each chunk contributes a (D, D) product and is then dropped.
        """
        t0 = time.time()
        n = n_rows if rows is None else len(rows)
        G = None
        for i in range(0, n, a.chunk):
            sl = slice(i, min(i + a.chunk, n))
            src = X[sl] if rows is None else X[rows[sl]]
            xc = torch.from_numpy(np.ascontiguousarray(src)).to(dev)
            g = xc.T @ xc
            G = g if G is None else G + g
            del xc
        ev, vec = torch.linalg.eigh(G)
        ev = ev.flip(0).clamp_min(0.0)
        vec = vec.flip(1)
        print(f"[inter]   gram+eigh {tuple(G.shape)} in {time.time() - t0:.0f}s "
              f"(top eig {float(ev[0]):.3e}, bottom {float(ev[-1]):.3e})", flush=True)
        del G
        return ev.sqrt(), vec

    # ---------------- specs: one per layer, plus the fusion groups
    specs: list[tuple[str, list[int]]] = [(f"block{k:02d}", [k]) for k in a.layers]
    seen = {name for name, _ in specs}
    for g in a.fusion:
        ks = [int(x) for x in g.split("+")]
        name = "fuse_" + "+".join(f"{k:02d}" for k in ks)
        if name in seen:
            continue
        specs.append((name, ks))
        seen.add(name)

    missing = [k for _, ks in specs for k in ks
               if not (Path(a.features) / "train" / f"block{k:02d}.npy").is_file()]
    if missing:
        raise SystemExit(f"layers with no cached features: {sorted(set(missing))}")

    # ============ stage 1: pick lambda per target ============
    t0 = time.time()
    s_sel, V_sel = factorise(np.flatnonzero(row_in_fit))
    sel_rows = np.flatnonzero(row_in_sel)
    A_sel = torch.from_numpy(np.ascontiguousarray(X[sel_rows])).to(dev) @ V_sel

    chosen: dict[str, float] = {}
    select_report: dict[str, dict] = {}
    for name, ks in specs:
        tr_conc, _ = build_Y(ks)
        Y = expand(tr_conc)
        Y_fit = torch.from_numpy(np.ascontiguousarray(Y[row_in_fit])).to(dev)
        Xv_sel = cross(np.flatnonzero(row_in_fit), Y)
        # V^T X^T Y: the only per-target product. Cheap (D x D_eig @ D_eig x D_target),
        # and reused by every lambda in the grid below.
        VtB = V_sel.T @ Xv_sel
        del Xv_sel, Y_fit
        # Gallery = slot 0 of each val concept, in val_c order. `sel_rows` walks the
        # fit matrix in row order, which is subject-major then val_c ascending, so the
        # expected column for row i is i % n_val -- not i, which would be true only if
        # there were a single subject.
        gallery = l2n(tr_conc.reshape(n_conc, n_slots, -1)[val_c, 0].astype(np.float64))
        expect = np.arange(len(sel_rows)) % len(val_c)
        scores = {}
        for lam in a.lams:
            w = s_sel / (s_sel.pow(2) + lam)
            pred = ((A_sel * w.unsqueeze(0)) @ VtB).cpu().numpy()
            sim = l2n(pred.astype(np.float64)) @ gallery.T
            scores[float(lam)] = float((sim.argmax(1) == expect).mean())
        best = max(scores, key=scores.get)
        chosen[name] = best
        select_report[name] = scores
        print(f"[inter] λ {name:>22s}: {best:g} "
              f"{' '.join(f'{k:g}:{v:.3f}' for k, v in scores.items())}", flush=True)
    del A_sel
    print(f"[inter] lambda selection done in {time.time() - t0:.0f}s")

    # ============ stage 2: report, at the selected lambda ============
    t0 = time.time()
    s_full, V_full = factorise(None)
    A_te = torch.from_numpy(np.ascontiguousarray(te_flat)).to(dev) @ V_full

    report: dict[str, dict] = {}
    for name, ks in specs:
        tr_conc, te_conc = build_Y(ks)
        Y = expand(tr_conc)
        VtB = V_full.T @ cross(None, Y)
        lam = chosen[name]
        w = s_full / (s_full.pow(2) + lam)
        pred = ((A_te * w.unsqueeze(0)) @ VtB).cpu().numpy()
        sim = l2n(pred.astype(np.float64)) @ l2n(te_conc.astype(np.float64)).T
        report[name] = {
            "layers": ks, "lam": lam, "two_way": two_way(sim),
            "row_top1": row_acc(sim), "dim": int(te_conc.shape[1]),
            "select": select_report.get(name, {}),
        }
        print(f"[inter] {name:>22s} dim {te_conc.shape[1]:>4d} λ {lam:<8g} "
              f"two-way {report[name]['two_way']:.4f}  row-top1 "
              f"{report[name]['row_top1']:.4f}", flush=True)

    order = sorted(report, key=lambda k: -report[k]["two_way"])
    print(f"\n[inter] ranked by two-way on held-out sub-{tgt:02d}:")
    for k in order:
        print(f"  {k:>22s} {report[k]['two_way']:.4f}   row {report[k]['row_top1']:.4f}")

    best_single = max((k for k in report if not k.startswith("fuse_")),
                      key=lambda k: report[k]["two_way"])
    best_fuse = [k for k in order if k.startswith("fuse_")]
    print(f"\n[inter] best single layer: {best_single} "
          f"({report[best_single]['two_way']:.4f})")
    if best_fuse:
        bf = best_fuse[0]
        print(f"[inter] best fusion:       {bf} ({report[bf]['two_way']:.4f})")
        if report[bf]["two_way"] > report[best_single]["two_way"]:
            print("[inter] VERDICT: fusion beats the best single layer")
        else:
            print("[inter] VERDICT: fusion does NOT beat the best single layer here")

    out = Path(a.out) if a.out else (
        config.OUTPUTS / f"sub{tgt:02d}" / f"probe_inter_tgt{tgt:02d}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "target_subject": tgt, "source_subjects": srcs, "mvnn": a.mvnn,
        "channels": a.channels, "val_concepts": val_c.tolist(),
        "n_fit_rows": int(n_rows), "layers": a.layers, "lams": list(a.lams),
        "ranking": order, "results": report,
    }, indent=2))
    print(f"\n[inter] wrote {out}  ({time.time() - t0:.0f}s for the report)")


if __name__ == "__main__":
    main()
