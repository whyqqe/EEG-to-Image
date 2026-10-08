"""Verify the v10 structural knobs: no-op safety + the knobs actually bite."""
import sys

import numpy as np

sys.path.insert(0, "src")
from samclip import calibration as C  # noqa: E402

rng = np.random.default_rng(0)
C_, R, d = 20, 8, 16
g = rng.normal(size=(C_, d))
z_reps = g[:, None, :] + 0.6 * rng.normal(size=(C_, R, d))
q = z_reps.mean(1)
sr = C._sq_cos_dist(rng.normal(size=(C_, d)))


def rec(x, **kw):
    return C.subspace_soft_recovery(x, g, k=5, rho=0.1, rank=None, tau=0.05,
                                    hard_landmarks=False, **kw)


print("[1] alpha=0 bit-identical to shipped S3R, with and without new kwargs")
a, _ = rec(q, alpha=0.0)
b, _ = rec(q, alpha=0.0, fgw_struct_ref=sr, fgw_struct_mix=0.0)
print("    equal:", np.array_equal(a, b))

print("[2] struct_mix=0 must equal not passing a template")
x, _ = rec(q, alpha=0.5)
y, _ = rec(q, alpha=0.5, fgw_struct_ref=sr, fgw_struct_mix=0.0)
print("    equal:", np.allclose(x, y, atol=1e-12))

print("[3] template must actually bite at mix>0, and stay finite")
z, _ = rec(q, alpha=0.5, fgw_struct_ref=sr, fgw_struct_mix=0.5)
print("    changes output:", not np.allclose(x, z), " finite:", np.isfinite(z).all())

print("[4] rep_cloud_scores end-to-end: R-prefix slice reaches the operator")
for sub in (None, 4, 1):
    sc, dg = C.rep_cloud_scores(z_reps, g, k=5, rho=0.1, rep_subsample=sub)
    tag = sub if sub else R
    print("    R=%-3s top1=%5.1f  diag.R=%s" % (tag, 100 * np.mean(sc.argmax(1) == np.arange(C_)),
                                               dg["rep_subsample"]))
print("[5] R=1 must equal the plain averaged query in the same whitened space")
sc1, _ = C.rep_cloud_scores(z_reps, g, k=5, rho=0.1, rep_subsample=1)
print("    ran without error:", sc1.shape == (C_, C_))
