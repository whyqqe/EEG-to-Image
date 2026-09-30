#!/usr/bin/env python3
"""Aggregate the semantic ladder and price every arm against the control.

Why a script rather than a block in the runner
----------------------------------------------
Each task of `run_epd_sem.sh` scores ONE (arm, seed) and writes its own line. The
claims this experiment exists to make are all CROSS-task:

  * is `A1..A4` better than `A0`, on the retrieval axis and on the generated images;
  * does the structural tower's PRESENCE in training change the semantic tower,
    which is `A0@2025` against the archived `sem_only` arm;
  * how much of each arm's effect survives averaging over three seeds.

None of those is available to a single task, and each needs the per-concept vectors
rather than the aggregate scores: the unpaired threshold for a 200-way score is ~10
points, which is larger than most of the effects a ladder of single-flag changes
produces. Averaging the per-concept vectors over seeds before pairing keeps the
pairing valid (every seed is scored on the identical 200 test concepts and the
identical 150 val concepts, because `--split-seed` is fixed) and shrinks the
seed-to-seed term as well.

Missing arms are reported, never imputed. An arm with one seed out of three still
gets a row and its `seeds` column says `1/3`, because the alternative -- silently
reporting a single-seed number next to three-seed numbers -- is how a replication
turns into a decoration.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epd.stats import verdict                                    # noqa: E402

# The ladder, in the order the header of `run_epd_sem.sh` argues for. `A0` is the
# reference every other arm is paired against, and the comparison is stated rather
# than implied because a ladder without a fixed reference is a set of numbers.
ARMS: list[tuple[str, str]] = [
    ("A0", "control: the shipped semantic block, no structural tower"),
    ("A1", "epochs 100 -> 25 (selection bias vs schedule length)"),
    ("A2", "channels all63 -> SAMGA's 17 occipito-parietal (also 70 -> 28 tokens)"),
    ("A3", "learned temperature -> fixed + softplus (EEGiT's loss, our schedule)"),
    ("A4", "target block26 (1280-d residual) -> _pooled (1024-d joint space)"),
]
SEEDS = [2025, 2026, 2027]
REF = "A0"

# The archived arm whose semantic tower was trained JOINTLY with the DA2 structural
# tower at seed 2025, with the same semantic flags as `A0`. This is the only
# same-seed partner available for the architecture question, which is why the
# comparison is stated in terms of `A0@2025` and not of `A0`'s three-seed mean.
JOINT_ARM = "sem_only"
JOINT_DIR = "epd_da2_depth8_metrics"
JOINT_CKPT_TAG = "epd_da2_depth8"

TWKEYS = ["clip", "inception", "alex5", "alex2"]
LOWKEYS = ["pixcorr", "ssim"]
TABLE_KEYS = ["pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "swav", "fid"]


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def retrieval_vector(out: Path, tag: str) -> np.ndarray | None:
    """The per-concept test Top-1 indicators for a run, from either writer.

    `train.py` puts it in `<tag>_result.json` under `test.per_concept`; runs that
    predate that are backfilled into `<tag>_perconcept.json` by
    `per_concept_eval.py`. Both are accepted because half the arms in this project
    were trained before the vector existed, and refusing the backfilled form would
    make exactly the most important comparison unpaired.
    """
    r = load_json(out / f"{tag}_result.json")
    if r is not None:
        pc = (r.get("test") or {}).get("per_concept")
        if pc is not None:
            return np.asarray(pc["top1"], dtype=np.float64)
    b = load_json(out / f"{tag}_perconcept.json")
    if b is not None:
        return np.asarray(b["per_concept"]["top1"], dtype=np.float64)
    return None


def metric_vector(met: Path, tag: str, key: str) -> np.ndarray | None:
    """The per-concept vector for one metric, from whichever decomposition it has."""
    d = load_json(met / f"{tag}_seven_persample.json")
    if d is None:
        return None
    v = (d.get("per_image_lowlevel", {}).get(key) if key in LOWKEYS
         else d.get("q", {}).get(key))
    return None if v is None else np.asarray(v, dtype=np.float64)


def stack(vecs: list[np.ndarray | None], expect: int, what: str) -> np.ndarray | None:
    """Seed-average a list of per-concept vectors, refusing mismatched lengths.

    A silent broadcast here would pair concept i of one arm against concept j of
    another, which is the one failure mode that produces a confident wrong answer
    instead of an error.
    """
    got = [v for v in vecs if v is not None]
    if not got:
        return None
    for v in got:
        if v.size != expect:
            raise SystemExit(f"{what}: a vector has {v.size} entries, expected {expect}; "
                             f"the concept orders do not match")
    return np.mean(np.stack(got, axis=0), axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/project/peilab/why/eeg-retrieval")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--met-dir", default="")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--out-file", default="",
                    help="also write the report here (e.g. "
                         "outputs/sub08/epd_sem_ladder.md). The report goes to stdout "
                         "either way, so a submitted job needs this to leave anything "
                         "behind for a reader that did not tail the job log.")
    args = ap.parse_args()

    # Tee rather than redirect: the same text lands in the job log and in the file, so
    # a failure part-way through still leaves the part that was computed.
    if args.out_file:
        f = Path(args.out_file)
        f.parent.mkdir(parents=True, exist_ok=True)

        class _Tee:
            def __init__(self, *streams: object) -> None:
                self._s = streams

            def write(self, s: str) -> int:
                for t in self._s:
                    t.write(s)  # type: ignore[attr-defined]
                return len(s)

            def flush(self) -> None:
                for t in self._s:
                    t.flush()  # type: ignore[attr-defined]

        sys.stdout = _Tee(sys.stdout, f.open("w", encoding="utf-8"))

    root = Path(args.root).resolve()
    out = Path(args.out_dir) if args.out_dir else root / "outputs/sub08"
    met = Path(args.met_dir) if args.met_dir else out / "epd_sem_metrics"

    # ---- gather, per arm, per seed ------------------------------------------
    ret: dict[str, list[np.ndarray | None]] = {}
    agg: dict[str, list[dict]] = {}
    gen: dict[str, dict[str, list[np.ndarray | None]]] = {}
    avail: dict[str, list[int]] = {}

    for arm, _ in ARMS:
        ret[arm], agg[arm] = [], []
        gen[arm] = {k: [] for k in TWKEYS + LOWKEYS}
        avail[arm] = []
        for seed in SEEDS:
            tag = f"epd_sem_{arm}_s{seed}"
            r = load_json(out / f"{tag}_result.json")
            m = load_json(met / f"{tag}_seven.json")
            if r is None and m is None:
                ret[arm].append(None)
                for k in gen[arm]:
                    gen[arm][k].append(None)
                continue
            avail[arm].append(seed)
            ret[arm].append(retrieval_vector(out, tag))
            agg[arm].append(r or {})
            for k in gen[arm]:
                gen[arm][k].append(metric_vector(met, tag, k))

    # ---- the retrieval ladder ------------------------------------------------
    print("## Semantic ladder, sub-08: retrieval")
    print()
    print("`--split-seed` is fixed at 2025 for every arm, so all arms are scored on the "
          "identical 150 val and 200 test concepts and the per-concept vectors pair "
          "directly. The unpaired threshold for a 200-way score is "
          "`min_detectable_diff` ~9.8 points; every claim below is paired.")
    print()
    print("| arm | seeds | test Top-1 (mean of seeds) | per-seed | val Top-1 of the "
          "selected epoch | selected epoch | what changed |")
    print("|---|---:|---:|---|---:|---:|---|")
    for arm, desc in ARMS:
        a = [x for x in agg[arm] if x]
        if not a:
            print(f"| `{arm}` | 0/3 | MISSING | | | | {desc} |")
            continue
        tops = [x["test"]["top1"] for x in a]
        vals = [x.get("best_val", {}).get("top1") for x in a]
        eps = [x.get("best_val", {}).get("epoch") for x in a]
        vs = "/".join("?" if v is None else f"{v:.1f}" for v in vals)
        es = "/".join("?" if e is None else str(e) for e in eps)
        print(f"| `{arm}` | {len(a)}/3 | {np.mean(tops):.2f} | "
              f"{'/'.join(f'{t:.1f}' for t in tops)} | {vs} | {es} | {desc} |")

    # The unpaired spread, printed so the paired deltas below can be judged against
    # what a per-seed draw of this metric looks like.
    print()
    spreads = [np.ptp([x["test"]["top1"] for x in agg[a] if x])
               for a, _ in ARMS if len([x for x in agg[a] if x]) > 1]
    for arm, _ in ARMS:
        a = [x for x in agg[arm] if x]
        if len(a) > 1:
            tops = [x["test"]["top1"] for x in a]
            print(f"- `{arm}` seed-to-seed spread: {np.ptp(tops):.1f} points "
                  f"(range {min(tops):.1f}..{max(tops):.1f})")
    if spreads:
        print(f"\nThe largest spread above ({max(spreads):.1f} points) is the reason the "
              f"per-arm mean of three seeds is reported rather than a single run.")

    # ---- paired retrieval tests ---------------------------------------------
    print()
    print(f"### Paired retrieval tests vs `{REF}` (per-concept Top-1, seed-averaged)")
    print()
    n = next((v.size for v in ret[REF] if v is not None), None)
    if n is None:
        print(f"`{REF}` has no retrieval vector yet; comparison skipped.")
    else:
        bv = stack(ret[REF], n, REF)
        print("| arm | delta Top-1 | 95% CI | sign p | frac shifted up | verdict |")
        print("|---|---:|---|---:|---:|---|")
        for arm, _ in ARMS:
            if arm == REF:
                continue
            av = stack(ret[arm], n, arm)
            if av is None:
                print(f"| `{arm}` | - | - | - | - | not scored |")
                continue
            v = verdict(av, bv, n_resamples=args.resamples)
            print(f"| `{arm}` | {100 * v['delta']:+.2f} pts | "
                  f"[{100 * v['lo']:+.2f}, {100 * v['hi']:+.2f}] | {v['sign_p']:.3f} | "
                  f"{v['frac_positive']:.2f} | {v['verdict']} |")
        print()
        print("The CI is in POINTS of the 200-way Top-1, so it is directly comparable "
              "to the ~9.8-point unpaired threshold above.")

    # ---- the generation ladder ----------------------------------------------
    print()
    print("### Generated images, seed-averaged")
    print()
    print("| arm | seeds | " + " | ".join(TABLE_KEYS) + " |")
    print("|---|---:|" + "---:|" * len(TABLE_KEYS))
    for arm, _ in ARMS:
        got = [load_json(met / f"epd_sem_{arm}_s{s}_seven.json") for s in SEEDS]
        got = [g for g in got if g]
        if not got:
            print(f"| `{arm}` | 0/3 | " + " | ".join(["MISSING"] * len(TABLE_KEYS)) + " |")
            continue
        cells = []
        for k in TABLE_KEYS:
            vals = [g[k] for g in got if k in g]
            cells.append(f"{np.mean(vals):.3f}" if vals else "-")
        print(f"| `{arm}` | {len(got)}/3 | " + " | ".join(cells) + " |")

    print()
    print(f"### Paired tests on the generated images vs `{REF}`")
    print()
    print("PixCorr/SSIM pair on their per-IMAGE vectors; the two-way metrics on the "
          "official Pearson per-concept `q_i`. Both are indexed by test concept, so "
          "both pair the same way.")
    print()
    print("| arm | metric | delta | 95% CI | sign p | verdict |")
    print("|---|---|---:|---|---:|---|")
    m = next((v.size for v in gen[REF]["clip"] if v is not None), None)
    for arm, _ in ARMS:
        if arm == REF or m is None:
            continue
        for k in LOWKEYS + TWKEYS:
            av, bv = stack(gen[arm][k], m, f"{arm}/{k}"), stack(gen[REF][k], m, f"{REF}/{k}")
            if av is None or bv is None:
                continue
            v = verdict(av, bv, n_resamples=args.resamples)
            print(f"| `{arm}` | {k} | {v['delta']:+.3f} | "
                  f"[{v['lo']:+.3f}, {v['hi']:+.3f}] | {v['sign_p']:.3f} | "
                  f"{v['verdict']} |")

    # ---- the architecture question ------------------------------------------
    print()
    print("### Does the structural tower's PRESENCE change the semantic tower?")
    print()
    print(f"`{REF}@2025` (trained alone) against the archived `{JOINT_ARM}` arm, which "
          f"used the same semantic flags at the same seed but was trained JOINTLY with "
          f"the DA2 depth tower. Same-seed, same-generation-settings, same generator "
          f"seed -- the only difference is the second tower. The two towers have "
          f"disjoint parameter sets, so the couplings this prices are the global "
          f"gradient clip and the shared schedule.")
    print()
    joint_ret = retrieval_vector(out, JOINT_CKPT_TAG)
    a0_2025 = retrieval_vector(out, f"epd_sem_{REF}_s2025")
    if joint_ret is None:
        print(f"- retrieval: `{JOINT_ARM}` has no per-concept vector, and this arm was "
              f"trained before `train.py` wrote one. Run `scripts/epd/per_concept_eval.py "
              f"--ckpt {out}/{JOINT_CKPT_TAG}_best.pt` to backfill it and re-run this "
              f"summary; the comparison is left UNPAIRED until then rather than "
              f"reported against the 9.8-point threshold as if it were informative.")
    elif a0_2025 is None:
        print(f"- retrieval: `{REF}@2025` is not scored yet")
    else:
        a0v = stack([a0_2025], joint_ret.size, f"{REF}@2025")
        v = verdict(a0v, joint_ret, n_resamples=args.resamples)
        print(f"- **retrieval** (`{REF}@2025` minus `{JOINT_ARM}`): "
              f"{100 * v['delta']:+.2f} pts, CI [{100 * v['lo']:+.2f}, "
              f"{100 * v['hi']:+.2f}], sign p {v['sign_p']:.3f} -> {v['verdict']}")

    jmet = met / f"{JOINT_ARM}_seven_persample.json"
    if not jmet.is_file():
        jmet = out / JOINT_DIR / f"{JOINT_ARM}_seven_persample.json"
    jd = load_json(jmet)
    if jd is None:
        print(f"- generated images: `{JOINT_ARM}` per-sample file not found at {jmet}")
    else:
        for k in LOWKEYS + TWKEYS:
            jv = (jd.get("per_image_lowlevel", {}).get(k) if k in LOWKEYS
                  else jd.get("q", {}).get(k))
            if jv is None:
                continue
            jv = np.asarray(jv, dtype=np.float64)
            av = stack(gen[REF][k], jv.size, f"{REF}/{k}")
            if av is None:
                print(f"- **{k}**: `{REF}` has no vector, so this one is skipped")
                continue
            v = verdict(av, jv, n_resamples=args.resamples)
            print(f"- **{k}** (`{REF}@2025` minus `{JOINT_ARM}`): {v['delta']:+.3f}, "
                  f"CI [{v['lo']:+.3f}, {v['hi']:+.3f}], sign p {v['sign_p']:.3f} -> "
                  f"{v['verdict']}")

    print()
    print("### Reproduce the A0 baseline before reading any of the above as an advance")
    print()
    print("`A0` is the same flag list as the archived joint arm's semantic block. If "
          "`A0@2025`'s retrieval does not land near that arm's 50.00, something other "
          "than the structural tower changed and the deltas above are not the arm "
          "effects they claim to be. That check is the reason `A0` exists as an arm "
          "rather than being read off the archive.")


if __name__ == "__main__":
    main()
