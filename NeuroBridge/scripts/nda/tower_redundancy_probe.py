"""Does the THIRD tower (image CLIP) carry information the prompt-text tower does not?

WHY THIS MEASUREMENT DECIDES AN ARCHITECTURE QUESTION
    The proposed design has three condition sources:
        tower-1  prompt text embeddings   (1024-d, CLIP text space)
        tower-2  SDXL VAE latent          (4x64x64)
        tower-3  image CLIP embedding     (1280-d, CLIP image space)
    Tower-3 is only worth a tower if image-CLIP carries something the prompt text
    does not.  CLIP was TRAINED so that a good caption's text embedding lands near
    its image embedding, so the two are collinear BY CONSTRUCTION and the question
    is empirical, not rhetorical:

        R^2( concat(4 prompt fields) -> image CLIP )   vs   R^2( class template -> image CLIP )

    If the prompt fields explain the image embedding nearly as well as the class
    template does AND almost fully, tower-3 is a restatement of tower-1.  If they
    explain a class template's amount but far from all of it, tower-3 carries
    instance-level information that no text field expresses.

    Precedent for expecting value: MindEye2 (ICML 2024) keeps a CLIP-image space
    reconstruction arm AND a predicted-caption arm, and uses the caption for a final
    img2img refinement -- i.e. the field treats them as complementary.  This script
    measures whether that holds on OUR captions.

Everything is train-fit / test-evaluated; no test information enters the fit.
"""
from __future__ import annotations

import os

import numpy as np

T = os.environ.get("NB_ROOT", "/project/peilab/why/NeuroBridge") + "/outputs/g2/targets"
FIELDS = ("overall", "subject", "background", "detail")


def load(name: str, split: str) -> np.ndarray:
    a = np.load(f"{T}/sem_{name}_{split}.npy").astype(np.float32)
    return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-8, None)


IMG_TR, IMG_TE = load("image", "train"), load("image", "test")
CON_TR, CON_TE = load("concept_tmpl", "train"), load("concept_tmpl", "test")
TXT_TR = np.concatenate([load(f, "train") for f in FIELDS], 1)
TXT_TE = np.concatenate([load(f, "test") for f in FIELDS], 1)

rng = np.random.default_rng(0)
sub = rng.permutation(len(TXT_TR))[:6000]          # 6000 train rows is plenty for a ridge


def ridge_r2(Xtr, Ytr, Xte, Yte, lams=(1e-2, 1e-1, 1.0, 10.0, 100.0)) -> tuple[float, float]:
    """mean per-dimension R^2 and mean cosine between prediction and target.

    BOTH sides are standardised on the FIT split only, and the residual is measured
    against the same fit-split mean, so this is an honest cross-validated R^2 and not
    an in-sample fit dressed up.

    Lambda is CHOSEN ON A HELD-IN SPLIT OF THE TRAIN ROWS.  That matters here: with
    4x1024 = 4096 predictors and 6000 rows, an untuned ridge scores R^2 = -0.31 --
    worse than predicting the mean -- which would have been reported as "text cannot
    explain the image embedding at all".  The ceiling is set by the penalised fit, not
    by the fixed lambda.
    """
    n = len(Xtr)
    idx = rng.permutation(n)
    fit, val = idx[: int(0.8 * n)], idx[int(0.8 * n):]
    Xm, Xs = Xtr[fit].mean(0), Xtr[fit].std(0).clip(1e-6)
    Ym, Ys = Ytr[fit].mean(0), Ytr[fit].std(0).clip(1e-6)
    Xf, Yf = (Xtr[fit] - Xm) / Xs, (Ytr[fit] - Ym) / Ys
    Xv, Yv = (Xtr[val] - Xm) / Xs, Ytr[val]
    G = Xf.T @ Xf
    eye = np.eye(Xf.shape[1])
    best_lam, best = lams[0], -1e18
    for lam in lams:
        W = np.linalg.solve(G + lam * eye, Xf.T @ Yf)
        P = Xv @ W * Ys + Ym
        r = 1.0 - ((P - Yv) ** 2).sum(-1) / ((Yv - Ym) ** 2).sum(-1)
        if float(np.mean(r)) > best:
            best, best_lam = float(np.mean(r)), lam
    W = np.linalg.solve(G + best_lam * eye, Xf.T @ Yf)
    P = ((Xte - Xm) / Xs) @ W * Ys + Ym
    r2 = 1.0 - ((P - Yte) ** 2).sum(-1) / ((Yte - Ym) ** 2).sum(-1)
    pu = P / np.clip(np.linalg.norm(P, axis=1, keepdims=True), 1e-8, None)
    cos = (pu * Yte).sum(-1)
    return float(np.mean(r2)), float(np.mean(cos))


print("=" * 88)
print("Can the prompt text explain the image CLIP embedding?   (fit on train, scored on test)")
print("=" * 88)
print(f"  {'predictor':<44}{'mean R^2':>12}{'mean cos':>12}")
r_conc = ridge_r2(TXT_TR[sub], IMG_TR[sub], TXT_TE, IMG_TE)
print(f"  {'concat(overall,subject,background,detail)':<44}{r_conc[0]:>12.4f}{r_conc[1]:>12.4f}")
for f in FIELDS:
    X = load(f, "train")[sub], load(f, "test")
    r = ridge_r2(X[0], IMG_TR[sub], X[1], IMG_TE)
    print(f"  {'only ' + f:<44}{r[0]:>12.4f}{r[1]:>12.4f}")
r_cls = ridge_r2(CON_TR[sub], IMG_TR[sub], CON_TE, IMG_TE)
print(f"  {'class template only (the class prior)':<44}{r_cls[0]:>12.4f}{r_cls[1]:>12.4f}")

print()
print("  Per-row alignment between the image embedding and each text field.")
print("  NOTE the two spaces have DIFFERENT widths (text 1024, image 1280 = ViT-H/14),")
print("  so a raw cosine is not defined; the alignment below is the FITTED prediction's")
print("  cosine from the table above.  A raw cosine across the whole gallery is reported")
print("  as the anisotropy check instead:")
print(f"    mean pairwise cosine among the 200 test image embeddings = "
      f"{float((IMG_TE @ IMG_TE.T)[~np.eye(len(IMG_TE), dtype=bool)].mean()):+.4f}")
print(f"    mean pairwise cosine among the 200 test class templates    = "
      f"{float((CON_TE @ CON_TE.T)[~np.eye(len(CON_TE), dtype=bool)].mean()):+.4f}")
print("    -> if those are near +0.9, CLIP/caption spaces are strongly anisotropic and")
print("       a high cosine means almost nothing; the discriminative read-out is R^2.")

print()
print("  READING:")
print("    If concat(text) already explains most of the image embedding's variance, the")
print("    third tower cannot add much BEYOND the prompt tower, and the honest design is")
print("    two towers.  If it leaves a large unexplained part, image-CLIP carries")
print("    instance-level content that no text field states, and a third tower is")
print("    justified -- which is what MindEye2's separate CLIP arm + caption arm implies.")
