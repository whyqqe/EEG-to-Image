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


G3H = load("outputs/eval/g3"); G3S = load("outputs/eval/s3r_tau0.01"); G3F = load("outputs/eval/fgw_a0.75")
V8F = load("outputs/eval/v8_fgw075")
V9F = load("outputs/eval/v9_fgw")

# (1) the combination that matters: same folds, v8-trained encoder + FGW operator
ks = sorted(set(G3F) & set(V8F))
print("=== A. v8 训练 + FGW(0.75) 部署, vs G3 + FGW(0.75), 同折 n=%d ===" % len(ks))
a = np.array([G3F[k] for k in ks]); b = np.array([V8F[k] for k in ks])
print("   G3 enc + FGW : %.2f" % a.mean())
print("   v8 enc + FGW : %.2f   delta %+.2f (pos %d / neg %d)"
      % (b.mean(), (b - a).mean(), (b > a).sum(), (b < a).sum()))
print()

# (2) three-way at the FGW operator, restricted to folds where all three encoders exist
ks3 = sorted(set(V8F) & set(V9F) & set(G3F))
print("=== B. 三种编码器 x 同一 FGW(0.75) 算子, 同折 n=%d ===" % len(ks3))
for lab, D in [("G3 (baseline)", G3F), ("v8 (alpha=0 train)", V8F), ("v9 (alpha=0.75 train)", V9F)]:
    v = np.array([D[k] for k in ks3])
    print("   %-22s %.2f" % (lab, v.mean()))
print()

# (3) is alpha=0.75 also the deployment optimum on the v8 encoder?
try:
    V8S = load("outputs/eval/v8_s3r")
    kks = sorted(set(V8F) & set(V8S) & set(G3H))
    print("=== C. v8 编码器下的算子收益, 同折 n=%d ===" % len(kks))
    print("   v8 + hard-ish(S3R) : %.2f" % np.mean([V8S[k] for k in kks]))
    print("   v8 + FGW(0.75)     : %.2f   -> operator gain %+.2f"
          % (np.mean([V8F[k] for k in kks]),
             np.mean([V8F[k] for k in kks]) - np.mean([V8S[k] for k in kks])))
except Exception as e:
    print("=== C. skipped: %s" % e)
print()
print("基准: G3+hard=45.33  G3+S3R=48.13  G3+FGW(0.75) all30=50.53  v8+S3R=50.62  SCORE=53.23")
