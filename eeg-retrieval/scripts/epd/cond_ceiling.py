#!/usr/bin/env python3
"""How much of the EEG->condition mapping is left on the table?

The measurement that motivates this
-----------------------------------
Across the 16 scored arms of the semantic ladder, the FINAL generated CLIP two-way
correlates with

    cos(exported IP condition, oracle condition)   r = +0.916
    the 200-way retrieval Top-1 of the EEG          r = +0.639

The oracle condition is the ground-truth CLIP embedding of the test image, and the
decoder turns it into CLIP 0.998 through the identical pipeline. So the quantity the
pipeline is actually limited by is how close the shipped 1024-d condition is to that
oracle -- NOT how well the EEG identifies the concept among the 1654 training concepts.
`A4` is the proof: it retrieved 36.5 (vs A0's 50.8) yet shipped a condition of the same
quality (cos 0.664 vs 0.664) and generated the same images.

That makes cos(cond, oracle) a cheap proxy: it needs one export and no Diffusion at all.

What this script answers
------------------------
A closed-form RIDGE from the flattened EEG to the SAME oracle target gives the linear
ceiling for that proxy. Ridge is closed form, so it has no training to overfit and no
schedule to tune; if the trained tower does not beat it, then the gap to the oracle is
not a training problem and no amount of capacity, schedule, or regularisation will
close it -- the EEG itself does not carry more of that direction.

  tower > ridge   the frontend is leaving linear structure on the table -> train better
  tower ~ ridge   the frontend is AT the linear ceiling -> the target or the EEG
                  preprocessing is the constraint, not the optimiser

It also reports the same comparison for the 200-way retrieval metric, so a reader can
see the two proxies diverge rather than take the r=0.916 on faith.

Usage
-----
  python scripts/epd/cond_ceiling.py --subject 8
  python scripts/epd/cond_ceiling.py --subject 8 --channels occipito_parietal
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from epd import config                                    # noqa: E402
from epd.data import concept_split, load_subject           # noqa: E402
from epd.metrics import mean_rank, retrieval_report        # noqa: E402

LAMS = [1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def cos_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return (p * q).sum(1) / (np.linalg.norm(p, axis=1) * np.linalg.norm(q, axis=1) + 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--channels", default="all", choices=["all", "occipito_parietal"])
    ap.add_argument("--val-concepts", type=int, default=150)
    ap.add_argument("--split-seed", type=int, default=2025)
    ap.add_argument("--slots", type=str, default="0",
                    help="which image slots of each training concept to fit on: '0' "
                         "(the same single slot the test set has, so the ridge sees a "
                         "matched distribution) or 'all' (all 10, which is what train.py "
                         "does and is the tower's actual advantage)")
    ap.add_argument("--out", default="outputs/sub08/cond_ceiling.json")
    a = ap.parse_args()

    dev = torch.device("cpu")
    ch = None if a.channels == "all" else config.CHANNELS_OCCIPITO_PARIETAL
    tr_eeg, te_eeg = load_subject(a.subject, ch)
    n_conc, n_slots, n_test = (config.N_TRAIN_CONCEPTS, config.N_IMAGES_PER_CONCEPT, te_eeg.shape[0])
    print(f"[data ] train {tuple(tr_eeg.shape)}  test {tuple(te_eeg.shape)}  "
          f"channels {tr_eeg.shape[2]}")

    # ---- targets: the SAME space the exported condition lives in ------------
    atoms_p = ROOT / "data/image_feature/ViT-H-14/image_train.npy"
    atoms = np.load(atoms_p).astype(np.float32)
    if atoms.ndim == 2:
        atoms = atoms[:, None]
    oracle_p = (ROOT / f"outputs/sub{a.subject:02d}"
                / "epd_da2_depth8_export/conds/ip_oracle_test.npy")
    oracle = np.load(oracle_p).astype(np.float32)
    if oracle.ndim == 3:
        oracle = oracle[:, 0]
    if atoms.shape[0] != n_conc or oracle.shape[0] != n_test:
        raise SystemExit(f"[target] atoms {atoms.shape} vs {n_conc} concepts; "
                         f"oracle {oracle.shape} vs {n_test} test concepts")
    print(f"[tgt  ] atoms {atoms.shape} (train concepts x slots), oracle {oracle.shape} "
          f"(test, GT CLIP)")

    split = concept_split(a.val_concepts, a.split_seed)
    fit_c, val_c = split.fit_concepts, split.val_concepts
    fit_slots = [0] if a.slots == "0" else list(range(n_slots))
    print(f"[split] fit {len(fit_c)} val {len(val_c)} concepts, slots {fit_slots}")

    X_fit = torch.from_numpy(
        tr_eeg[fit_c][:, fit_slots].reshape(-1, tr_eeg.shape[2] * tr_eeg.shape[3])).float()
    X_val = torch.from_numpy(
        tr_eeg[val_c].reshape(len(val_c), n_slots, -1)).float()
    X_te = torch.from_numpy(te_eeg[:, 0].reshape(n_test, -1)).float()
    print(f"[ridge] X_fit {tuple(X_fit.shape)}  X_val {tuple(X_val.shape)}  "
          f"X_te {tuple(X_te.shape)}")

    t0 = time.time()
    # The SVD of X_fit is 112 s at 15040x15750 and 12 s at 1504x15750, and every lambda
    # reuses it. Cached because the two `--slots` settings are meant to be compared and
    # paying the factorisation twice makes that comparison feel expensive enough to skip.
    cache = ROOT / f"outputs/sub{a.subject:02d}/_cond_ceiling_svd_slots{a.slots}.pt"
    if cache.is_file():
        U, S, V = torch.load(cache, map_location="cpu", weights_only=False)
        print(f"[ridge] SVD loaded from {cache} ({S.numel()} components)")
    else:
        U, S, Vh = torch.linalg.svd(X_fit, full_matrices=False)
        V = Vh.T
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save((U, S, V), cache)
        print(f"[ridge] SVD {time.time() - t0:.0f}s  rank {S.numel()}  "
              f"s[-1]={float(S[-1]):.2e} -> cached {cache}")
    A_val, A_te = X_val @ V, X_te @ V

    Y_fit = torch.from_numpy(
        l2n(atoms[fit_c][:, fit_slots].reshape(-1, atoms.shape[-1]))).float()
    Y_val = torch.from_numpy(l2n(atoms[val_c].reshape(len(val_c), n_slots, -1))).float()
    Y_te = torch.from_numpy(l2n(oracle)).float()

    # ---- lambda selection on the val split, on the SAME proxy -------------
    # The ridge is closed form: with A = X @ V, the weights in the V basis are
    #     W = diag(S / (S^2 + lam)) @ Y_fit        shape (rank, D)
    # and a prediction is A @ W. The weight matrix comes from the FIT targets -- an
    # earlier revision of this script multiplied by `Y_val` here, which is not a ridge
    # at all and is why the shapes did not line up: `Y_val` has one row per VAL concept
    # while the SVD basis has one column per FIT sample.
    #
    # Selection uses the val concepts' own centroids, not the test oracle: selecting on
    # the test proxy would make the "ceiling" an in-sample number and it would beat the
    # tower for the wrong reason.
    Y_fit_np = Y_fit.numpy()
    g_val = l2n(np.asarray(atoms[val_c]).reshape(len(val_c), n_slots, -1).mean(1))
    A_val_c = A_val.mean(1)                                   # (V, rank)
    best = None
    for lam in LAMS:
        W = (S / (S * S + lam)).unsqueeze(-1).numpy() * Y_fit_np     # (rank, D)
        pc = l2n(A_val_c.numpy() @ W)                                # (V, D)
        c = float(np.mean(cos_rows(pc, g_val)))
        print(f"[lam  ] {lam:8.0e}  val cos-to-centroid {c:+.4f}")
        if best is None or c > best[1]:
            best = (lam, c)
    lam = best[0]
    print(f"[lam  ] selected {lam:g} (val cos-to-centroid {best[1]:+.4f})\n")

    W = (S / (S * S + lam)).unsqueeze(-1).numpy() * Y_fit_np
    pred_te = l2n(A_te.numpy() @ W)
    ridge_cos = cos_rows(pred_te, oracle)

    # ---- the trained tower's shipped condition, for the same subject -------
    tower_cos = None
    tower_p = ROOT / f"outputs/sub{a.subject:02d}/epd_sem_A0_s2025_export/conds/ip_deploy_test.npy"
    if tower_p.is_file():
        z = l2n(np.load(tower_p).astype(np.float32))
        tower_cos = cos_rows(z, oracle)
    # and the archived joint arm, whose condition limiter predates the ladder
    joint_cos = None
    jp = ROOT / f"outputs/sub{a.subject:02d}/epd_da2_depth8_export/conds/ip_deploy_test.npy"
    if jp.is_file():
        joint_cos = cos_rows(l2n(np.load(jp).astype(np.float32)), oracle)

    # ---- and the retrieval metric on the same prediction -------------------
    # Ridge predicts a 1024-d vector; the gallery the tower retrieves against is the
    # projected training keys. Building that gallery needs the image encoder, which is
    # a GPUs-loadable model this script deliberately avoids -- so retrieval is reported
    # only if the cached gallery is present.
    ret = None
    gall_p = ROOT / "outputs/sub08/epd_sem_A0_s2025_export/export_report.json"
    if gall_p.is_file():
        rep = json.loads(gall_p.read_text())
        ret = {"gallery_size": rep.get("gallery_size")}

    print("=" * 78)
    print(f"cos(condition, oracle) on the {n_test} test concepts, subject {a.subject}")
    print("=" * 78)
    print(f"  oracle (GT CLIP)                1.0000  <- decoder turns this into CLIP 0.998")
    print(f"  ridge, closed form              {float(ridge_cos.mean()):+.4f}")
    if tower_cos is not None:
        print(f"  A0 tower (this ladder)          {float(tower_cos.mean()):+.4f}")
    if joint_cos is not None:
        print(f"  epd_da2_depth8 (archived)       {float(joint_cos.mean()):+.4f}")
    print(f"\n  paired, ridge minus A0 tower: ", end="")
    if tower_cos is not None:
        d = ridge_cos - tower_cos
        from epd.stats import sign_test
        st = sign_test(d)
        print(f"{float(d.mean()):+.4f}  (sign p {st['p']:.4f}, "
              f"{float((d > 0).mean()):.0%} of concepts better)")
    else:
        print("(no tower condition found)")
    print(f"""
If ridge beats the tower here, the optimiser is the constraint and the next experiment
is a training one. If they match, the next experiment has to change the TARGET or the
EEG representation, because a linear map of the flattened EEG has already extracted
everything that direction contains.
""")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "subject": a.subject, "channels": a.channels, "slots": a.slots,
        "lambda": lam, "ridge_cos_per_concept": [float(v) for v in ridge_cos],
        "ridge_cos_mean": float(ridge_cos.mean()),
        "tower_cos_mean": None if tower_cos is None else float(tower_cos.mean()),
        "joint_cos_mean": None if joint_cos is None else float(joint_cos.mean()),
        "oracle_path": str(oracle_p), "atoms_path": str(atoms_p),
    }, indent=2), encoding="utf-8")
    print(f"[done] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
