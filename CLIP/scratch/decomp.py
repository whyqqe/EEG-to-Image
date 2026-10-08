import json, glob, statistics as st
import numpy as np


def find(o):
    if isinstance(o, dict):
        if isinstance(o.get("rows"), dict):
            return o["rows"]
        for v in o.values():
            g = find(v)
            if g is not None:
                return g
    return None


H = "+ T1(CSLS + recovery) + T2 reps"


def load(d):
    return {p.split('/')[-1][:-5]: find(json.load(open(p)))[H]["top1"]
            for p in glob.glob(d + "/sub*_seed*.json")}


V9H = load("outputs/eval/v9"); V9S = load("outputs/eval/v9_s3r"); V9F = load("outputs/eval/v9_fgw")
G3H = load("outputs/eval/g3"); G3S = load("outputs/eval/s3r_tau0.01"); G3F = load("outputs/eval/fgw_a0.75")
ks = sorted(set(V9F) & set(G3F) & set(G3S) & set(V9S) & set(G3H) & set(V9H))
print("完整分解, 同一批 %d 个已完成 run: %s" % (len(ks), ks))
print()
print("%-22s %10s %10s %10s   %s" % ("operator", "G3 enc", "v9 enc", "deploy gain", "train gain (v9-G3)"))
for lab, a, b in [("hard (SCORE-style)", G3H, V9H), ("S3R (alpha=0)", G3S, V9S),
                  ("FGW (alpha=0.75)", G3F, V9F)]:
    ka = np.array([a[k] for k in ks]); kb = np.array([b[k] for k in ks]); d = kb - ka
    print("%-22s %10.2f %10.2f %+10.2f   %+6.2f (pos %d / neg %d)"
          % (lab, ka.mean(), kb.mean(), kb.mean() - ka.mean(), d.mean(), (d > 0).sum(), (d < 0).sum()))
print()
v9h = np.mean([V9H[k] for k in ks]); v9s = np.mean([V9S[k] for k in ks]); v9f = np.mean([V9F[k] for k in ks])
g3h = np.mean([G3H[k] for k in ks]); g3s = np.mean([G3S[k] for k in ks]); g3f = np.mean([G3F[k] for k in ks])
print("v9 encoder operator gains:  S3R-hard=%+.2f   FGW-hard=%+.2f" % (v9s - v9h, v9f - v9h))
print("G3 encoder operator gains:  S3R-hard=%+.2f   FGW-hard=%+.2f" % (g3s - g3h, g3f - g3h))
print()
for k in ks:
    print("   %-18s G3: hard %5.1f s3r %5.1f fgw %5.1f  |  v9: hard %5.1f s3r %5.1f fgw %5.1f"
          % (k, G3H[k], G3S[k], G3F[k], V9H[k], V9S[k], V9F[k]))
