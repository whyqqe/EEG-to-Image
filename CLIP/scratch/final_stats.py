"""Single-model stats for the best FGW config, per-fold spread and paired tests."""
import glob
import json
import statistics as st

import numpy as np

ROW2 = "+ T2 reps"
ROWF = "+ T1(CSLS + recovery) + T2 reps"


def find(o):
    if isinstance(o, dict):
        if isinstance(o.get("rows"), dict):
            return o["rows"]
        for v in o.values():
            g = find(v)
            if g is not None:
                return g
    return None


def load(d, row):
    return {p.split("/")[-1][:-5]: find(json.load(open(p)))[row]["top1"]
            for p in glob.glob(d + "/sub*_seed*.json")}


B = "outputs/eval/v8scr/a0.75_t0.03"
S = "outputs/eval/v8_s3r"
G3S = "outputs/eval/s3r_tau0.01"

for row in (ROW2, ROWF):
    a = load(S, row) if row in find(json.load(open(S + "/sub01_seed2025.json"))) else None
    b = load(B, row)
    if a is None:
        a = load(G3S, row)
        base = "G3+S3R"
    else:
        base = "v8+S3R"
    k = sorted(set(a) & set(b))
    d = np.array([b[x] - a[x] for x in k])
    print("=== %s : FGW(a0.75,t0.03) vs %s, n=%d ===" % (row, base, len(k)))
    print("    base  = %.2f" % np.mean([a[x] for x in k]))
    print("    FGW   = %.2f" % np.mean([b[x] for x in k]))
    print("    delta = %+.2f pp   sd=%.2f  t=%.2f   正/负=%d/%d"
          % (d.mean(), d.std(ddof=1), d.mean() / (d.std(ddof=1) / np.sqrt(len(d))),
             (d > 0).sum(), (d < 0).sum()))
    per = {}
    for x in k:
        per.setdefault(x.split("_")[0], []).append(b[x] - a[x])
    print("    逐被试: " + "  ".join("%s%+.1f" % (s, st.mean(v)) for s, v in sorted(per.items())))
    print("    FGW per-seed sd across the 30 runs = %.2f" % st.pstdev(list(b.values())))
    print()

# where does SCORE sit?
print("SCORE = 53.23 +- 1.62 (single model, per the paper's protocol)")
print("我们单模型最佳行 = %.2f  -> %+.2f" % (st.mean(list(load(B, ROW2).values())),
                                                st.mean(list(load(B, ROW2).values())) - 53.23))
