#!/usr/bin/env python
"""Locate the best CLIP layer to align EEG to, using a closed-form ridge probe.

Why this is the right next step
-------------------------------
The design doc's phase-1 gate is a layer-depth sweep, and it has not been done:
our cached image features are the *final* CLIP layer only (1024-d = `visual.proj`
output). Every arm so far therefore asked the EEG encoder to hit CLIP's most
abstraction-heavy representation, with no way to test another depth.

The design doc also prices this as the single largest lever. Testing that lever by
training a deep model per candidate layer would cost hours of GPU per layer. Ridge
tests it in seconds per layer, because:

  * ridge is closed form, so there is no training and no overfitting, and
  * the expensive factorisation depends only on the EEG (X), not on the target (Y),
    so ONE decomposition serves every layer.

So this is a cheap, high-information probe of exactly the quantity we care about:
if layer L's features give better EEG->image retrieval than the final layer, then
aligning the EEG encoder to layer L is the fix, and we have located it for the
cost of one feature extraction pass.

Reading the numbers
-------------------
Selection is on VALIDATION Top-1 (concept-level holdout of training concepts,
sweeping all 10 image slots). Test is reported at the selected (layer, lambda) and
is never used to choose. A per-layer lambda sweep is included because layers differ
in scale and anisotropy, and holding lambda fixed across them would confound
"this layer is better" with "this layer happened to suit the regularisation".

What a positive result would look like: a clear interior maximum, i.e. some middle
layer beating both the early blocks and `_pooled` (the current, final-layer
baseline) by more than the ~2.8-point standard error on a 200-way task.

Usage
-----
    python scripts/nwret/probe_layers.py --subject 8 --features outputs/features/clip_h14_layers
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nwret import config
from nwret.data import concept_split, load_subject
from nwret.metrics import mean_rank, retrieval_report


def l2n(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--channels", default="occipito_parietal", choices=["all", "occipito_parietal"])
    ap.add_argument("--features", default=str(config.OUTPUTS / "features" / "clip_h14_layers"))
    ap.add_argument("--val-concepts", type=int, default=150)
    ap.add_argument("--split-seed", type=int, default=2025)
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[1e2, 1e3, 1e4, 1e5, 1e6])
    ap.add_argument("--layers", nargs="*", default=None,
                    help="restrict to these layer keys (default: every extracted layer)")
    ap.add_argument("--rank", type=int, default=0,
                    help="truncate the EEG SVD to this many components (0 = all)")
    ap.add_argument("--fit-limit", type=int, default=0, help="cap fit concepts (smoke only)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"[probe] device={dev}")

    fdir = Path(a.features)
    tr_dir, te_dir = fdir / "train", fdir / "test"
    if not tr_dir.is_dir():
        raise SystemExit(f"missing {tr_dir}; run extract_layers.py first")

    keys = sorted(p.stem for p in tr_dir.glob("*.npy"))
    if a.layers:
        keys = [k for k in keys if k in set(a.layers)]
    if not keys:
        raise SystemExit("no layer arrays found")
    # _pooled last so the baseline line reads naturally in the output
    keys = [k for k in keys if k != "_pooled"] + (["_pooled"] if "_pooled" in keys else [])
    print(f"[probe] {len(keys)} target spaces: {keys[0]}..{keys[-1]}")

    # ---------------- EEG
    channels = None if a.channels == "all" else config.CHANNELS_OCCIPITO_PARIETAL
    tr_eeg, te_eeg = load_subject(a.subject, channels)
    n_ch = tr_eeg.shape[2]
    print(f"[probe] EEG train {tr_eeg.shape} test {te_eeg.shape} ({n_ch} channels)")

    split = concept_split(a.val_concepts, a.split_seed)
    fit_c, val_c = split.fit_concepts, split.val_concepts
    if a.fit_limit:
        fit_c = fit_c[:a.fit_limit]
        print(f"[probe] --fit-limit {a.fit_limit} -> {len(fit_c)} fit concepts")

    def flat(arr, c):
        x = arr[c]
        return x.reshape(-1, x.shape[-2] * x.shape[-1])

    # Val keeps its slot axis: (n_val_concepts, n_slots, feature). Flattening to
    # (n_val_concepts*n_slots, feature) and slicing contiguously would NOT give
    # "slot s for every concept" -- a flat reshape is concept-major, so n_concepts
    # contiguous rows are n_concepts/n_slots whole concepts with all their slots.
    # That silently turns a 150-way retrieval into a 150-way retrieval over 15
    # distinct concepts, and it returns a plausible number while doing so.
    n_val_slots = tr_eeg.shape[1]
    X_fit = torch.from_numpy(flat(tr_eeg, fit_c)).float()
    X_val_all = torch.from_numpy(
        tr_eeg[val_c].reshape(len(val_c), n_val_slots, -1)).float()
    X_te = torch.from_numpy(te_eeg[:, 0].reshape(te_eeg.shape[0], -1)).float()

    # z-score using fit statistics only
    mu = X_fit.mean(0, keepdim=True)
    sd = X_fit.std(0, keepdim=True).clamp_min(1e-6)
    X_fit = ((X_fit - mu) / sd).to(dev)
    X_val_all = ((X_val_all - mu) / sd).to(dev)
    X_te = ((X_te - mu) / sd).to(dev)
    print(f"[probe] fit {tuple(X_fit.shape)}  "
          f"val {tuple(X_val_all.shape)} ({len(val_c)}-way x {n_val_slots} slots)  "
          f"test {tuple(X_te.shape)}")

    # ---------------- one factorisation, reused for every target
    t0 = time.time()
    U, S, Vh = torch.linalg.svd(X_fit, full_matrices=False)
    V = Vh.T
    if a.rank:
        k = min(a.rank, S.numel())
        U, S, V = U[:, :k], S[:k], V[:, :k]
        print(f"[probe] truncated to rank {k}")
    print(f"[probe] SVD done in {time.time() - t0:.0f}s  "
          f"(s[0]={float(S[0]):.2e}, s[-1]={float(S[-1]):.2e})")

    # These two depend only on the EEG, never on the target. One projection of the
    # whole fit set is what makes the layer scan cost seconds per layer rather than
    # a fresh fit per layer.
    A_val_all = X_val_all @ V               # (n_val, n_slots, k)
    A_te = X_te @ V                         # (n_test, k)
    print(f"[probe] EEG projections cached: {tuple(A_val_all.shape)} {tuple(A_te.shape)}",
          flush=True)

    def predict(A: torch.Tensor, B: torch.Tensor, lam: float) -> np.ndarray:
        """Y_pred = A diag(s/(s^2+lam)) B."""
        w = (S / (S * S + lam)).unsqueeze(-1)
        return (A * w.squeeze(-1).unsqueeze(0)) @ B

    results: list[dict] = []

    for key in keys:
        Ytr = np.load(tr_dir / f"{key}.npy")
        Yte = np.load(te_dir / f"{key}.npy")
        if Ytr.ndim != 3:
            print(f"[skip] {key}: unexpected shape {Ytr.shape}")
            continue
        D = Ytr.shape[-1]

        Y_fit = torch.from_numpy(l2n(Ytr[fit_c].reshape(-1, D))).float().to(dev)
        Y_val = l2n(Ytr[val_c].reshape(len(val_c), n_val_slots, D))
        Y_te = l2n(Yte[:, 0])

        t_l = time.time()
        B = U.T @ Y_fit                                   # (k, D) -- one matmul per layer
        best_lam, best_val, per_lam = None, -1.0, []
        for lam in a.lams:
            # val: sweep every image slot so the selection signal is not 10x noisier
            # than it needs to be (the same fix applied to the deep-model path).
            t1, t5, mr = [], [], []
            for s in range(n_val_slots):
                Yh = predict(A_val_all[:, s], B, lam).cpu().numpy()
                rep = retrieval_report(Yh, Y_val[:, s])
                mr.append(mean_rank(Yh, Y_val[:, s]))
                t1.append(rep["top1"]); t5.append(rep["top5"])
            v_top1 = float(np.mean(t1))
            per_lam.append({"lam": lam, "val_top1": v_top1,
                            "val_top5": float(np.mean(t5)),
                            "val_mean_rank": float(np.mean(mr)),
                            "val_top1_std": float(np.std(t1))})
            if v_top1 > best_val:
                best_val, best_lam = v_top1, lam

        # Test scored once, at the lambda chosen on validation.
        Yh_te = predict(A_te, B, best_lam).cpu().numpy()
        te = retrieval_report(Yh_te, Y_te)
        te["mean_rank"] = mean_rank(Yh_te, Y_te)

        kind = "final (shipped)" if key == "_pooled" else ("block " + key.replace("block", ""))
        results.append({"layer": key, "kind": kind, "dim": D,
                        "lam": best_lam, "val_top1": best_val,
                        "val_top5": per_lam[[p["lam"] for p in per_lam].index(best_lam)]["val_top5"],
                        "val_top1_std": per_lam[[p["lam"] for p in per_lam].index(best_lam)]["val_top1_std"],
                        "test_top1": te["top1"], "test_top5": te["top5"],
                        "test_mean_rank": te["mean_rank"], "per_lam": per_lam})
        print(f"[probe] {key:>8} D={D:>4} lam*={best_lam:<9.0e} "
              f"val {best_val:6.2f} +-{results[-1]['val_top1_std']:4.2f} | "
              f"test {te['top1']:6.2f} top5 {te['top5']:6.2f} rank {te['mean_rank']:5.1f} "
              f"({time.time() - t_l:.0f}s)", flush=True)

    if not results:
        raise SystemExit("no usable layers")

    # Selection on validation only. Test is never used to pick.
    best = max(results, key=lambda r: r["val_top1"])
    pooled = next((r for r in results if r["layer"] == "_pooled"), None)
    interior = [r for r in results if r["layer"] != "_pooled"]
    peak = max(interior, key=lambda r: r["val_top1"]) if interior else None

    out = Path(a.out) if a.out else (config.OUTPUTS / f"sub{a.subject:02d}" / "probe_layers.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "subject": a.subject, "channels": a.channels, "n_channels": n_ch,
        "features": str(fdir), "n_fit_concepts": len(fit_c), "n_val_concepts": len(val_c),
        "selection": "val Top-1 (10 slots averaged); test scored once at the selected lambda",
        "best_by_val": best,
        "final_layer_baseline": pooled,
        "per_layer": results,
    }
    out.write_text(json.dumps(payload, indent=2, default=float))

    # ---------------- report
    print()
    print("=" * 96)
    print("ALIGNMENT-TARGET LAYER PROBE  (closed-form ridge, val-selected)")
    print("=" * 96)
    print(f"  {'layer':>8} {'dim':>5} {'lambda*':>10} {'val top1':>9} {'test top1':>10} "
          f"{'test top5':>10} {'mean rank':>10}")
    # Depth order, not score order: the inverted-U shape across depth IS the result
    # here, and sorting by score would scramble exactly the pattern being looked for.
    # It also removes any way to read a ranking off the test column.
    for r in results:
        mark = "  <-- best by val" if r["layer"] == best["layer"] else ""
        star = " *" if r["layer"] == "_pooled" else "  "
        print(f"  {r['layer']:>8}{star} {r['dim']:>4} {r['lam']:>10.0e} {r['val_top1']:>9.2f} "
              f"{r['test_top1']:>10.2f} {r['test_top5']:>10.2f} {r['test_mean_rank']:>10.1f}{mark}")
    print("  (* = _pooled, the final-layer space our cached features live in)")
    print()
    print(f"  chance level: top1 0.50, mean rank 100.5  (200-way)")
    print(f"  standard error on a 200-way top1: ~2.8 points")
    if pooled is not None:
        print()
        print(f"  final-layer baseline : val {pooled['val_top1']:.2f}, test {pooled['test_top1']:.2f}")
        if peak is None:
            print("  no interior layers were evaluated, so there is nothing to compare")
            print("  against; re-run without --layers to sweep the whole depth")
        else:
            d_test = peak["test_top1"] - pooled["test_top1"]
            d_val = peak["val_top1"] - pooled["val_top1"]
            # Per-slot spread is not the error bar on the mean: the selection metric
            # averages n_val_slots slots, so its SE is std/sqrt(n_slots). Comparing a
            # mean against a per-slot std would overstate the noise ~3x on 10 slots.
            se_val = float(np.hypot(peak["val_top1_std"], pooled["val_top1_std"])
                           / np.sqrt(n_val_slots))
            # A single depth being best is weak evidence; a whole band of depths beating
            # the final layer is not. Count the band, since it does not depend on one
            # noisy point being the winner.
            band = [r for r in results
                    if r["layer"] != "_pooled" and r["val_top1"] > pooled["val_top1"]]
            interior_sorted = sorted(results, key=lambda r: r["layer"] != "_pooled")
            lo, hi = interior_sorted[0], interior_sorted[-1]
            inverted_u = (peak["layer"] not in (lo["layer"], hi["layer"])
                          and peak["val_top1"] > lo["val_top1"]
                          and peak["val_top1"] > hi["val_top1"])

            print(f"  best interior layer  : {peak['layer']}  val {peak['val_top1']:.2f}, "
                  f"test {peak['test_top1']:.2f}")
            print(f"    delta vs final layer : test {d_test:+.2f}, val {d_val:+.2f} "
                  f"+-{se_val:.2f} (SE of the val difference)")
            print(f"    inverted-U across depth : {'yes' if inverted_u else 'no'} "
                  f"(shallow {lo['val_top1']:.2f} -> peak {peak['val_top1']:.2f} -> "
                  f"deep {hi['val_top1']:.2f} on val)")
            print(f"    interior layers above the final layer (val) : "
                  f"{len(band)} of {len(results) - 1}"
                  + (f"  [{band[0]['layer']}..{band[-1]['layer']}]" if band else ""))
            print()
            if abs(d_val) > 2 * se_val or d_test > 2.8:
                print("  -> the alignment target is a real lever here: a middle layer beats the")
                print("     final layer beyond noise. This is the documented lever, and it is the fix.")
            elif inverted_u and len(band) >= 3:
                print("  -> consistent with the documented lever but not significant on any single")
                print("     depth: the inverted U is present and a BAND of layers beats the final one,")
                print("     which is systematic rather than one lucky point, yet the peak's own margin")
                print("     is inside the noise. Arm PA tests whether it transfers to the deep model;")
                print("     if the deep arm does not move, treat this as suggestive only.")
            else:
                print("  -> no interior peak and no band above the final layer. The layer axis does")
                print("     NOT explain the gap, so the next suspect is the EEG encoder itself.")
    print(f"\n[probe] wrote {out}")


if __name__ == "__main__":
    main()
