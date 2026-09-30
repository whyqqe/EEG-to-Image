"""Decide whether A0 (open-set attribute alignment) can beat the concept-label space.

The whole NW4 plan rests on one falsifiable claim:

    "z_sem aligned to *attribute* text embeddings transfers to unseen concepts
     better than z_sem aligned to *concept-identity* embeddings."

If that is false, A0 is dead and we should not spend a GPU day on it.

Method (no GPU, closed-form, selection on val_b concepts only):

    W  = ridge_fit( z_train[fit],  target_train[fit] )
    q  = l2n( z_test @ W )
    sim = q @ l2n( target_test ).T
    score = 200-way top-1 (sim[i].argmax() == i)

Every bank must be in the SAME test concept order (the protocol's alphabetical
200).  That is asserted, not assumed: `--assert-order` cross-checks the concept
name text bank against the phrase list.

Baselines that matter:
  concept_name   the label space itself        (what a classifier would use)
  clip_img       trial-level CLIP image space  (what IP-Adapter expects)
  vith_cat5      the multi-level target that scored 40.5% in the route probe
  attr_*         the four Qwen caption fields  (A0's candidates)
  attr_all       concat of the four            (A0's upper bound)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NB_ROOT = Path(__file__).resolve().parents[2]


def l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def metrics_200(sim: np.ndarray) -> dict:
    n = sim.shape[0]
    order = np.argsort(-sim, 1)
    return {
        "top1": float((order[:, 0] == np.arange(n)).mean()),
        "top5": float(np.mean([i in order[i, :5] for i in range(n)])),
        "mean_rank": float(np.mean([np.where(order[i] == i)[0][0] + 1 for i in range(n)])),
    }


def make_ridge(tgt_tr: np.ndarray, fit_i: np.ndarray, val_i: np.ndarray,
               ztr: np.ndarray, zte: np.ndarray, tgt_te: np.ndarray,
               cid_val: np.ndarray, gal_val: np.ndarray,
               alphas: list[float]) -> dict:
    """Fit on `fit_i`, pick alpha on val_b, report 200-way test.

    Selection is by val **mean rank**, not val top-1: val has only 82 concepts /
    820 rows, so a top-1 difference of 0.005 is pure noise.  Mean rank is smooth,
    monotone in quality, and does not saturate at the extremes the way top-1 does.
    Every alpha's test score is also returned so the choice can be audited.
    """
    X = ztr[fit_i].astype(np.float64)
    Y = tgt_tr[fit_i].astype(np.float64)
    XtX = X.T @ X
    XtY = X.T @ Y
    g_val = l2n(gal_val)
    best = {"alpha": None, "val_rank": float("inf"), "val_top1": -1.0}
    Ws: dict[float, np.ndarray] = {}
    per_alpha: dict[str, float] = {}
    for a in alphas:
        W = np.linalg.solve(XtX + a * np.eye(X.shape[1]), XtY)
        Ws[a] = W
        qv = l2n(ztr[val_i] @ W)
        sv = qv @ g_val.T
        order = np.argsort(-sv, 1)
        rank = float(np.mean([np.where(order[i] == cid_val[i])[0][0] + 1
                              for i in range(len(val_i))]))
        top1 = float((order[:, 0] == cid_val).mean())
        per_alpha[f"{a:g}"] = round(top1, 4)
        if rank < best["val_rank"]:
            best = {"alpha": a, "val_rank": rank, "val_top1": top1}
    W = Ws[best["alpha"]]
    qt = l2n(zte @ W)
    sim = qt @ l2n(tgt_te).T
    test_per_alpha = {}
    for a, Wa in Ws.items():
        qa = l2n(zte @ Wa)
        test_per_alpha[f"{a:g}"] = round(metrics_200(qa @ l2n(tgt_te).T)["top1"], 4)
    return {"alpha": best["alpha"], "val_rank": best["val_rank"],
            "val_top1": best["val_top1"],
            "test_per_alpha": test_per_alpha, **metrics_200(sim)}


def concept_snr(bank: np.ndarray, cid: np.ndarray) -> float:
    """How separable are the concepts inside this bank, before any EEG is involved.

    For each row: cos to its own concept mean minus the best cos to another concept
    mean.  Positive => the space itself groups the 10 reps of a concept together.
    This is a property of the TARGET SPACE, so it bounds what any encoder can read.
    """
    nb = l2n(bank)
    n_cls = int(cid.max()) + 1
    cnt = np.bincount(cid, minlength=n_cls).astype(np.float64)
    mean = np.zeros((n_cls, nb.shape[1]), dtype=np.float64)
    # sum_i x_i / cnt[c_i]  ==  (1/cnt_c) * sum_{i in c} x_i  == concept mean
    np.add.at(mean, cid, nb / cnt[cid][:, None])
    mean = l2n(mean)
    sim = nb @ mean.T
    sim[np.arange(len(nb)), cid] = -np.inf          # exclude own concept
    own = (nb * mean[cid]).sum(1)
    return float(np.mean(own - sim.max(1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--z-root", type=str, default=str(NB_ROOT / "outputs/ocf/intra_z"))
    ap.add_argument("--cond-cache", type=str, default=str(NB_ROOT / "outputs/gem/cond_cache"))
    ap.add_argument("--g2-targets", type=str, default=str(NB_ROOT / "outputs/g2/targets"))
    ap.add_argument("--clip-text-dir", type=str,
                    default=str(NB_ROOT / "outputs/nda_ss/sub-08/clip_text"))
    ap.add_argument("--split-json", type=str, default=str(NB_ROOT / "outputs/leakfree/split.json"))
    ap.add_argument("--alphas", type=str, default="0.1,1,10,100,1000,1e4")
    ap.add_argument("--json-out", type=str, default="")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(NB_ROOT / "scripts/nda"))
    import leakfree as LF  # noqa: E402

    sid = f"{args.subject:02d}"
    cc = Path(args.cond_cache)
    g2 = Path(args.g2_targets)
    ct = Path(args.clip_text_dir)

    ztr = l2n(np.load(Path(args.z_root) / f"sub-{sid}" / "shared_r_train.npy").astype(np.float32))
    zte = l2n(np.load(Path(args.z_root) / f"sub-{sid}" / "shared_r_test.npy").astype(np.float32))
    print(f"[eeg] train {ztr.shape} test {zte.shape}")

    split = LF.load(args.split_json)
    fit_i = LF.rows_for(split, "fit", len(ztr))
    val_i = LF.rows_for(split, "val_b", len(ztr))

    # ---- concept ids ----
    # THINGS-EEG2 train is concept-major: 1654 concepts x 10 reps, verified on
    # text_flat_clip (cos(row0,row1..9)==1.0, cos(row0,row10)==0.405).
    import json as _json
    meta = _json.loads((ct / "train" / "meta.json").read_text(encoding="utf-8"))
    reps = int(meta["n_flat"]) // int(meta["n_concepts"])
    n_cls = int(meta["n_concepts"])
    if len(ztr) != int(meta["n_flat"]):
        raise SystemExit(f"[FATAL] eeg rows {len(ztr)} != meta n_flat {meta['n_flat']}")
    cid_tr = np.arange(len(ztr), dtype=np.int64) // reps
    print(f"[cid] {n_cls} train concepts x {reps} reps = {len(cid_tr)} rows")

    # val_b concept ids must be a subset of the train gallery
    cid_val = cid_tr[val_i]

    def load(p: Path) -> np.ndarray:
        if not p.is_file():
            raise SystemExit(f"[FATAL] missing {p}")
        return np.load(p).astype(np.float32)

    # ---- candidate banks: name -> (train_rows, test_rows) ----
    banks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    banks["concept_name"] = (load(ct / "train" / "text_concept_clip.npy"),
                             load(ct / "test" / "text_concept_clip.npy"))
    for tag, fn in (("clip_img", "clip_img1024"),
                    ("clip_depth", "clip_depth1024"),
                    ("clip_edge", "clip_edge1024")):
        banks[tag] = (load(cc / f"{fn}_train.npy"), load(cc / f"{fn}_test.npy"))
    for f in ("overall", "subject", "background", "detail"):
        banks[f"attr_{f}"] = (load(g2 / f"sem_{f}_train.npy"), load(g2 / f"sem_{f}_test.npy"))
    banks["attr_all"] = (np.concatenate([banks[f"attr_{f}"][0] for f in
                                         ("overall", "subject", "background", "detail")], 1),
                         np.concatenate([banks[f"attr_{f}"][1] for f in
                                         ("overall", "subject", "background", "detail")], 1))
    # multi-level ViT-H bank (the 40.5% reference) if present
    vroot = NB_ROOT / "data/things_eeg/image_feature/ViT-H-14"
    levels = ["image", "GaussianBlur", "LowResolution", "Mosaic", "GaussianNoise"]
    lp = []
    for lv in levels:
        if lv == "image":
            tr_p, te_p = vroot / "image_train.npy", vroot / "image_test.npy"
        else:
            tr_p, te_p = vroot / lv / "train.npy", vroot / lv / "test.npy"
        if tr_p.is_file() and te_p.is_file():
            d = np.load(tr_p, mmap_mode="r").shape[-1]
            lp.append((np.load(tr_p).astype(np.float32).reshape(-1, d),
                       np.load(te_p).astype(np.float32).reshape(200, -1)))
    if len(lp) == len(levels):
        banks["vith_cat5"] = (np.concatenate([l2n(a) for a, _ in lp], 1),
                              np.concatenate([l2n(b) for _, b in lp], 1))
        banks["vith_mean5"] = (l2n(np.mean([l2n(a) for a, _ in lp], 0)),
                               l2n(np.mean([l2n(b) for _, b in lp], 0)))
        # the two combinations that actually matter for the design decision:
        #   vith_cat5 + attrs  -> is the strongest bank IMPROVED by open-set text?
        #   clip_img + attrs   -> 1024-d entry point the IP-Adapter consumes, made
        #                         open-set by conjugating it with the caption text
        banks["vith_cat5+attr_all"] = (
            np.concatenate([banks["vith_cat5"][0], banks["attr_all"][0]], 1),
            np.concatenate([banks["vith_cat5"][1], banks["attr_all"][1]], 1))
        banks["clip_img+attr_all"] = (
            np.concatenate([banks["clip_img"][0], banks["attr_all"][0]], 1),
            np.concatenate([banks["clip_img"][1], banks["attr_all"][1]], 1))
        banks["clip_img+attr_all+vith"] = (
            np.concatenate([banks["clip_img"][0], banks["attr_all"][0],
                            banks["vith_cat5"][0]], 1),
            np.concatenate([banks["clip_img"][1], banks["attr_all"][1],
                            banks["vith_cat5"][1]], 1))
    else:
        print(f"[warn] vith bank incomplete ({len(lp)}/{len(levels)} levels); skipped")

    # ---- assert the test concept order is shared ----
    phrases = _json.loads((ct / "test" / "concept_phrases.json").read_text(encoding="utf-8"))
    if len(phrases) != 200:
        raise SystemExit(f"[FATAL] test phrases {len(phrases)} != 200")

    alphas = [float(x) for x in args.alphas.split(",")]
    # gallery for val selection = the bank's OWN train rows (concept-level would be
    # cleaner, but the ridge q lives in the bank's space, so use the same space)
    rows = []
    n_flat = len(ztr)
    for name, (tr, te) in banks.items():
        if te.shape[0] != 200:
            print(f"[skip] {name}: test rows {te.shape[0]} != 200")
            continue
        # Banks come in two shapes: per-ROW (n_flat, D, e.g. clip_img1024_train) and
        # per-CONCEPT (n_cls, D, e.g. text_concept_clip).  Normalise both into
        # (per-row for fitting, per-concept for the val gallery).
        if tr.shape[0] == n_flat:
            tr_rows = tr
            gal_con = np.zeros((n_cls, tr.shape[1]), dtype=np.float32)
            np.add.at(gal_con, cid_tr, tr / float(reps))
            snr = concept_snr(tr, cid_tr)
        elif tr.shape[0] == n_cls:
            tr_rows = tr[cid_tr]
            gal_con = tr
            snr = concept_snr(tr[cid_tr], cid_tr)
        else:
            print(f"[skip] {name}: train rows {tr.shape[0]} matches neither "
                  f"n_flat={n_flat} nor n_cls={n_cls}")
            continue
        gal_con = l2n(gal_con)
        r = make_ridge(tr_rows, fit_i, val_i, ztr, zte, te, cid_val, gal_con, alphas)
        r["name"] = name
        r["dim"] = int(tr.shape[1])
        r["concept_snr"] = round(snr, 4)
        rows.append(r)
        print(f"  {name:<24} dim={r['dim']:<5} snr={snr:+.4f} "
              f"val_rank={r['val_rank']:5.1f} alpha={r['alpha']:<7g} TEST={r['top1']:.4f} "
              f"per-alpha={r['test_per_alpha']}")

    rows.sort(key=lambda r: -r["top1"])
    print(f"\n{'bank':<24}{'dim':>6}{'snr':>8}{'valrank':>9}{'test200':>9}{'top5':>8}{'rank':>7}")
    print("-" * 73)
    for r in rows:
        print(f"{r['name']:<24}{r['dim']:>6}{r['concept_snr']:>8.4f}{r['val_rank']:>9.1f}"
              f"{r['top1']:>9.4f}{r['top5']:>8.4f}{r['mean_rank']:>7.1f}")

    best = rows[0]
    by_name = {r["name"]: r for r in rows}
    print(f"\n[verdict] best bank = {best['name']} ({best['top1']:.4f})")
    if "concept_name" in by_name and "attr_all" in by_name:
        d = by_name["attr_all"]["top1"] - by_name["concept_name"]["top1"]
        print(f"[verdict] attr_all {by_name['attr_all']['top1']:.4f} vs concept_name "
              f"{by_name['concept_name']['top1']:.4f} -> delta {d:+.4f}")
        print("[verdict] " + ("A0 SUPPORTED: open-set attributes transfer to unseen "
                              "concepts better than concept identity"
                              if d > 0.02 else
                              "A0 NOT SUPPORTED: attributes do not beat the label space"))
    for base, combo in (("clip_img", "clip_img+attr_all"),
                        ("vith_cat5", "vith_cat5+attr_all")):
        if base in by_name and combo in by_name:
            d = by_name[combo]["top1"] - by_name[base]["top1"]
            print(f"[verdict] {combo} {by_name[combo]['top1']:.4f} vs {base} "
                  f"{by_name[base]['top1']:.4f} -> delta {d:+.4f} "
                  f"({'CONCAT HELPS' if d > 0.01 else 'concat does not help'})")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"[json] {args.json_out}")


if __name__ == "__main__":
    main()
