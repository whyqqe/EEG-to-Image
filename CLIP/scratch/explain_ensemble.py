"""Explain 3-seed ensembling with real numbers: mean-of-accuracy vs mean-of-score-matrix."""
import glob
import numpy as np

ROW = "fused::+ T1(CSLS + recovery) + T2 reps"
D = "outputs/scores/v8scr_a0.5_t0.01"


def top1(m):
    return float(np.mean(m.argmax(1) == np.arange(m.shape[0])) * 100)


for fold in ("sub01", "sub05", "sub08"):
    fs = sorted(glob.glob(f"{D}/{fold}_seed*.npz"))
    if len(fs) < 3:
        continue
    seeds = [np.load(f)[ROW] for f in fs]
    names = [f.split("seed")[-1].split(".")[0] for f in fs]
    singles = [top1(m) for m in seeds]
    mean_of_acc = float(np.mean(singles))
    ens = top1(np.mean(seeds, 0))
    best = max(singles)
    # how many concepts does EACH seed get right, and how many does the ensemble get?
    n = seeds[0].shape[0]
    truth = np.arange(n)
    per_seed_correct = [(m.argmax(1) == truth) for m in seeds]
    union = np.logical_or.reduce(per_seed_correct)
    inter = np.logical_and.reduce(per_seed_correct)
    print("=" * 70)
    print(f"{fold}   (200-way retrieval, {len(fs)} seeds)")
    for nm, s, c in zip(names, singles, per_seed_correct):
        print(f"   seed{nm}: top1 = {s:5.1f}   (gets {c.sum():3d}/200 concepts right)")
    print(f"   -> mean of the three ACCURACIES = {mean_of_acc:5.1f}   <- never beats the best seed")
    print(f"   -> mean of the three SCORE MATRICES, then argmax = {ens:5.1f}")
    print(f"      best single seed = {best:5.1f}   gain over mean-of-acc = {ens - mean_of_acc:+.1f}")
    print(f"      concepts right by at least one seed = {union.sum()}/200")
    print(f"      concepts right by ALL three seeds   = {inter.sum()}/200")
    only_one = int((per_seed_correct[0] | per_seed_correct[1] | per_seed_correct[2]).sum()
                   - int(per_seed_correct[0].sum()))
    print(f"      concepts that seed0 misses but another seed catches = "
          f"{int((~per_seed_correct[0] & union).sum())}")
