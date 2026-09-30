"""Per-band ceiling probe: is there recoverable EEG signal ABOVE r=0.0625?

Why this exists
---------------
The previous per-band budget (LF 0.1467 vs everything-else 0.0018) was measured
through `pred_vae_test.npy`, i.e. through a head trained with L1 to regress the
FULL latent.  That head is *taught* to push every band it cannot predict toward
the conditional mean, and the non-LF bands are exactly those bands.  Measuring
their correlation in that head's output is circular: it cannot distinguish

    (a) the band carries no EEG information, from
    (b) the band carries information the L1 objective chose to discard.

So the earlier "non-LF is only 1.2% of LF" number is a LOWER BOUND on the
achievable per-band correlation, not an upper bound.  This script replaces it
with a fair ceiling estimate: one ridge regression PER BAND, fit on train,
selected on a held-in train split, scored on test.  Ridge with the optimal
lambda is the strong linear baseline; if a band is unrecoverable even to ridge,
it is genuinely unrecoverable (not an artifact of a collapse-prone objective).

Leak-free
---------
lambda is selected on a held-in 15% train split.  Test is touched once at the
end, for reporting only.

Reported quantity
-----------------
`var_expl = E_frac * corr^2` where E_frac is the band's share of GT latent
energy.  This is the band's contribution to the fraction of latent variance the
EEG could in principle explain, and it is directly comparable across bands.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BANDS = [(0.0, 0.0625), (0.0625, 0.125), (0.125, 0.25), (0.25, 0.5), (0.5, 2.0)]


def radial_grid(h: int, w: int) -> np.ndarray:
    # Normalise so that r=1.0 is the Nyquist rate along an axis.
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    return np.sqrt(fy ** 2 + fx ** 2) / 0.5


def band_masks(r: np.ndarray) -> list[np.ndarray]:
    return [((r >= lo) & (r < hi)) for lo, hi in BANDS]


def load_y(path: str) -> np.memmap:
    a = np.load(path, mmap_mode="r")
    return a


def band_coeffs(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """x: (n,C,H,W) float32 -> (n, C*n_band*2) real coefficient vector."""
    F = np.fft.fft2(x, axes=(-2, -1))           # complex128
    sel = F[:, :, mask]                          # (n, C, nb)
    out = np.concatenate([sel.real, sel.imag], axis=-1)
    return out.reshape(x.shape[0], -1).astype(np.float32)


def per_image_corr(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    Fp = np.fft.fft2(pred, axes=(-2, -1))[:, :, mask]
    Fg = np.fft.fft2(gt, axes=(-2, -1))[:, :, mask]
    a = np.concatenate([Fp.real, Fp.imag], axis=-1).reshape(pred.shape[0], -1)
    b = np.concatenate([Fg.real, Fg.imag], axis=-1).reshape(gt.shape[0], -1)
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    cc = (a * b).sum(1) / (np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1)) + 1e-12)
    return float(cc.mean()), float(cc.std(ddof=1) / np.sqrt(len(cc)))


def ridge_fit(X, Y, lam):
    """Primal ridge via normal equations. X:(n,d) Y:(n,m) -> W:(d,m)."""
    A = X.T @ X
    A[np.diag_indices_from(A)] += lam * X.shape[0]
    return np.linalg.solve(A, X.T @ Y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--z-train", required=True)
    ap.add_argument("--z-test", required=True)
    ap.add_argument("--vae-train", required=True)
    ap.add_argument("--vae-test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1e3])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    Xtr = np.load(args.z_train).astype(np.float32)
    Xte = np.load(args.z_test).astype(np.float32)
    Xtr /= np.linalg.norm(Xtr, axis=1, keepdims=True).clip(1e-8)
    Xte /= np.linalg.norm(Xte, axis=1, keepdims=True).clip(1e-8)
    n, d = Xtr.shape
    print(f"[probe] z train {Xtr.shape}  test {Xte.shape}")

    Gte = np.load(args.vae_test).astype(np.float32)
    C, H, W = Gte.shape[1], Gte.shape[2], Gte.shape[3]
    r = radial_grid(H, W)
    masks = band_masks(r)
    for (lo, hi), m in zip(BANDS, masks):
        print(f"[probe] band {lo:.4f}-{hi:<7.4f} n_coef/ch={int(m.sum())}")

    Gtr = load_y(args.vae_train)
    if Gtr.shape[0] != n:
        raise SystemExit(f"row mismatch: z {n} vs vae {Gtr.shape[0]}")

    # ---- energy fraction, from a subset (variance is stationary enough) ----
    sub = np.asarray(Gtr[: min(2048, n)], dtype=np.float32)
    Fs = np.abs(np.fft.fft2(sub, axes=(-2, -1))) ** 2
    tot = float(Fs.sum())
    e_frac = [float((Fs[:, :, m]).sum() / tot) for m in masks]
    del Fs, sub

    nval = int(n * args.val_frac)
    perm = np.random.default_rng(0).permutation(n)
    ival, itr = perm[:nval], perm[nval:]

    results = {}
    names = [f"{lo}-{hi}" for lo, hi in BANDS]
    for bi, (bname, mask) in enumerate(zip(names, masks)):
        # ---- band-limited targets, accumulated without materialising all Y ----
        Ytr = np.empty((n, C * int(mask.sum()) * 2), dtype=np.float32)
        for s in range(0, n, args.chunk):
            e = min(s + args.chunk, n)
            Ytr[s:e] = band_coeffs(np.asarray(Gtr[s:e], dtype=np.float32), mask)
        print(f"[probe] {bname}: target dim {Ytr.shape[1]}")

        # ---- lambda on held-in split ----
        best, best_lam = -9.9, None
        for lam in args.lams:
            Wr = ridge_fit(Xtr[itr], Ytr[itr], lam)
            p = Xtr[ival] @ Wr
            a = p - p.mean(1, keepdims=True)
            b = Ytr[ival] - Ytr[ival].mean(1, keepdims=True)
            cc = float(((a * b).sum(1) / (np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1)) + 1e-12)).mean())
            if cc > best:
                best, best_lam = cc, lam
        print(f"[probe] {bname}: val corr {best:.4f} @ lambda={best_lam:g}")

        # ---- refit on all train, score on test (touched once) ----
        # NB: the ridge weight matrix must NOT be called W -- W is the image width.
        Wt = ridge_fit(Xtr, Ytr, best_lam)
        Pte = (Xte @ Wt).astype(np.float32)
        # reconstruct the spatial field so we can score it the same way the
        # shipped pipeline is scored (per-image correlation on the real field)
        nb = int(mask.sum())
        Fr = np.zeros((Pte.shape[0], C, H, W), dtype=np.complex64)
        Fr[:, :, mask] = (Pte[:, : nb * C] + 1j * Pte[:, nb * C:]).reshape(-1, C, nb)
        pfield = np.real(np.fft.ifft2(Fr, axes=(-2, -1)))
        mu, se = per_image_corr(pfield, Gte, mask)
        results[bname] = {
            "n_coef_per_ch": int(mask.sum()),
            "energy_frac": e_frac[bi],
            "val_corr": best,
            "lam": best_lam,
            "test_corr": mu,
            "test_se": se,
        }
        results[bname]["var_expl"] = e_frac[bi] * mu * mu
        del Ytr, Fr, pfield

    tot_ve = sum(v["var_expl"] for v in results.values())
    print("\n=== PER-BAND CEILING (ridge, lambda on held-in split, test scored once) ===")
    print(f"{'band':<16}{'E_frac':>9}{'val_corr':>10}{'test_corr':>11}{'var_expl':>10}{'share':>8}")
    for k, v in results.items():
        sh = v["var_expl"] / tot_ve if tot_ve > 0 else 0.0
        print(f"{k:<16}{v['energy_frac']:>9.4f}{v['val_corr']:>10.4f}"
              f"{v['test_corr']:>11.4f}{v['var_expl']:>10.5f}{sh:>8.3f}")
    print(f"{'TOTAL':<16}{'':>9}{'':>10}{'':>11}{tot_ve:>10.5f}")
    lo_key, hi_key = "0.0-0.0625", None
    coarse = results["0.0-0.0625"]["var_expl"]
    fine = tot_ve - coarse
    print(f"\ncoarse (r<0.0625) var_expl = {coarse:.5f}")
    print(f"fine   (r>=0.0625) var_expl = {fine:.5f}   -> fine/coarse = {fine/max(coarse,1e-12):.4f}")
    print("verdict: fine band is worth a tower iff fine/coarse is not negligible")
    print("         (a tower needs >= ~0.10 of coarse to move any image metric)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"bands": results, "total_var_expl": tot_ve,
         "coarse_var_expl": coarse, "fine_var_expl": fine,
         "fine_over_coarse": fine / max(coarse, 1e-12),
         "note": ("ridge ceiling per band; lambda chosen on held-in train split; "
                  "test scored once. Replaces the earlier circular estimate that "
                  "used the collapsed L1 head's own output.")},
        indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
