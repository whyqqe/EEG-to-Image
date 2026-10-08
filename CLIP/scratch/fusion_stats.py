"""Paired stats for the fusion-rule choice, so a +0.57 is not mistaken for a real effect."""
import glob
import statistics as st

import numpy as np


def top1(m):
    return float(np.mean(m.argmax(1) == np.arange(m.shape[0])) * 100)


def zs(m):
    return (m - m.mean(1, keepdims=True)) / m.std(1, keepdims=True).clip(1e-9)


def csls(m, k=10):
    idx1 = np.argsort(-m, axis=1)[:, :k]
    fwd = np.take_along_axis(m, idx1, axis=1).mean(1, keepdims=True)
    idx0 = np.argsort(-m, axis=0)[:k, :]
    bwd = np.take_along_axis(m, idx0, axis=0).mean(0, keepdims=True)
    return 2 * m - fwd - bwd


D = "outputs/scores/v8scr_a0.75_t0.03"
per = {}
for p in sorted(glob.glob(D + "/sub*_seed*.npz")):
    z = np.load(p)
    t2 = z["row::+ T2 reps"]
    key = p.split("/")[-1][:-4]
    per[key] = {
        "T2": top1(t2),
        "T2+csls": top1(csls(t2)),
        "T2+csls.t03": top1(csls(t2) * 0.3 + zs(t2) * 0.7),
        "sum": top1(zs(z["row::+ CSLS + recovery"]) + zs(t2)),
    }

ks = sorted(per)
print("n = %d runs\n" % len(ks))
for a, b in [("T2+csls", "T2"), ("T2+csls", "sum"), ("T2", "sum")]:
    d = np.array([per[k][a] - per[k][b] for k in ks])
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    print("%-12s vs %-6s : %+.2f pp  (sd %.2f, t=%.2f, pos %d / neg %d)"
          % (a, b, d.mean(), d.std(ddof=1), t, (d > 0).sum(), (d < 0).sum()))

print()
print("per-subject (mean over 3 seeds):")
for name in ("T2", "T2+csls"):
    row = []
    for s in sorted({k.split("_")[0] for k in ks}):
        v = st.mean([per[k][name] for k in ks if k.startswith(s + "_")])
        row.append("%s %.1f" % (s, v))
    print("  %-9s %s" % (name, "  ".join(row)))
print()
mean_c = st.mean([per[k]["T2+csls"] for k in ks])
print("T2+csls mean = %.2f   vs SCORE 53.23 = %+.2f" % (mean_c, mean_c - 53.23))
print("but SCORE's own reported sd is 1.62, and this +0.57 over T2 was chosen on these folds")
