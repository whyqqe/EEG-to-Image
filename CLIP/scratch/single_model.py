"""Single-model numbers across the screened configs. NO ensembling -- fair comparison."""
import glob
import json
import statistics as st

ROWS = ["+ T2 reps", "+ T1(CSLS + recovery) + T2 reps", "+ CSLS + recovery"]


def find(o):
    if isinstance(o, dict):
        if isinstance(o.get("rows"), dict):
            return o["rows"]
        for v in o.values():
            g = find(v)
            if g is not None:
                return g
    return None


CONFIGS = ["a0.5_t0.01", "a0.625_t0.01", "a0.75_t0.01", "a0.875_t0.01", "a0.75_t0.03"]
print("单模型 (30 run = 10 折 x 3 seed, 每个 run 独立报 top-1, 不集成)")
print("%-16s %14s %26s %18s" % ("config", "+ T2 reps", "+T1(CSLS+rec)+T2", "+ CSLS + recovery"))
best = (None, -1)
for c in CONFIGS:
    vals = {}
    for r in ROWS:
        acc = []
        for p in glob.glob(f"outputs/eval/v8scr/{c}/sub*_seed*.json"):
            rr = find(json.load(open(p)))
            if r in rr:
                acc.append(rr[r]["top1"])
        vals[r] = st.mean(acc) if acc else float("nan")
    print("%-16s %14.2f %26.2f %18.2f" % (c, vals[ROWS[0]], vals[ROWS[1]], vals[ROWS[2]]))
    for r in ROWS[:2]:
        if vals[r] > best[1]:
            best = ((c, r), vals[r])
print()
print("最佳单模型行: %s / %s = %.2f" % (best[0][0], best[0][1], best[1]))
print("SCORE = 53.23   -> 差距 %+.2f" % (best[1] - 53.23))
