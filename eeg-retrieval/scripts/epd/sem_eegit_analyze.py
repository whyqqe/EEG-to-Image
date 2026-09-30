#!/usr/bin/env python3
"""Price one EEGiT-objective run against the recorded A0 control.

The comparison, and why it is paired
------------------------------------
Two runs of the PURE semantic tower -- no structural branch, no ControlNet
contribution -- differing only in the loss flags:

    `A0`      recorded control: `--lr 5e-4`, learned temperature, both sides normalised
    `TAU_L2`  this run:         the same flags plus `--fixed-temp --no-eeg-l2norm`

`--split-seed` is pinned at 2025 for both, and both use seed 2025, so they are scored
on the identical 200 test concepts and the identical 150 val concepts. Every claim here
is therefore a same-seed paired bootstrap CI plus a sign test on per-concept vectors.
That matters because a single 200-way score has a ~9.8-point unpaired threshold, larger
than the effect a two-flag change plausibly produces.

Three readouts, and they are not redundant
------------------------------------------
  * `test Top-1` -- the retrieval axis. The A4 arm is why this alone is not enough: it
    retrieved WORSE while generating just as well.
  * `cos(cond, oracle)` -- the cosine between the exported IP-Adapter condition and the
    ground-truth CLIP joint embedding. This is the condition-quality number, and among
    the readouts here it is the one that tracks generated CLIP.
  * the seven generated-image metrics -- the actual deliverable.

A missing run is reported, never imputed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epd.stats import verdict                                    # noqa: E402

# The A0 control was trained by the earlier run, so its artefacts carry an unprefixed
# tag and live in that run's metrics directory. Keeping the lookup in one table is what
# stops a reader from having to know that when re-running this.
CONTROL = dict(name="A0", tag="epd_sem_A0_s{s}", met="epd_sem_metrics",
               desc="recorded control: learned tau, both sides normalised")
CANDIDATE = dict(name="TAU_L2", tag="epd_sem_eegit_TAU_L2_s{s}",
                 met="epd_sem_eegit_metrics",
                 desc="`--fixed-temp --no-eeg-l2norm`: tau pinned at 0.07 and only the "
                      "image side normalised, i.e. EEGiT's released objective")
ARMS = [CONTROL, CANDIDATE]
SEEDS = [2025, 2026, 2027]
REF = CONTROL["name"]

TWKEYS = ["clip", "inception", "alex5", "alex2"]
LOWKEYS = ["pixcorr", "ssim"]
TABLE_KEYS = ["pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "swav", "fid"]


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def tag_of(arm: dict, seed: int) -> str:
    return arm["tag"].format(s=seed)


def retrieval_vector(out: Path, tag: str) -> np.ndarray | None:
    """The per-concept test Top-1 indicators, from either writer."""
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
    d = load_json(met / f"{tag}_seven_persample.json")
    if d is None:
        return None
    v = (d.get("per_image_lowlevel", {}).get(key) if key in LOWKEYS
         else d.get("q", {}).get(key))
    return None if v is None else np.asarray(v, dtype=np.float64)


def cond_cos_vector(out: Path, tag: str) -> np.ndarray | None:
    """Per-concept cosine between the arm's exported condition and the oracle.

    Recomputed from the .npy files rather than cached anywhere, because it is two
    array reads and a normalise -- and because a cached copy would be one more thing
    that can silently belong to a different checkpoint than the one being reported.
    """
    oracle = out / "epd_da2_depth8_export/conds/ip_oracle_test.npy"
    deploy = out / f"{tag}_export/conds/ip_deploy_test.npy"
    if not (oracle.is_file() and deploy.is_file()):
        return None
    o = np.load(oracle).astype(np.float64)
    c = np.load(deploy).astype(np.float64)
    if o.shape != c.shape:
        raise SystemExit(f"{tag}: condition {c.shape} does not match oracle {o.shape}")
    o /= np.maximum(np.linalg.norm(o, axis=1, keepdims=True), 1e-8)
    c /= np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-8)
    return (o * c).sum(1)


def stack(vecs: list[np.ndarray | None], expect: int, what: str) -> np.ndarray | None:
    """Seed-average a list of per-concept vectors, refusing mismatched lengths.

    A silent broadcast here would pair concept i of one arm against concept j of
    another -- the one failure that yields a confident wrong answer instead of an error.
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
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--out-file", default="")
    args = ap.parse_args()

    # Tee rather than redirect: the same text lands in the job log and in the file, so
    # a failure part-way through still leaves whatever was computed.
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

    # ---- gather, per arm, per seed ------------------------------------------
    ret: dict[str, list] = {a["name"]: [] for a in ARMS}
    agg: dict[str, list] = {a["name"]: [] for a in ARMS}
    gen: dict[str, dict[str, list]] = {a["name"]: {k: [] for k in TWKEYS + LOWKEYS}
                                       for a in ARMS}
    ccos: dict[str, list] = {a["name"]: [] for a in ARMS}
    avail: dict[str, list[int]] = {a["name"]: [] for a in ARMS}

    for arm in ARMS:
        met = out / arm["met"]
        for seed in SEEDS:
            t = tag_of(arm, seed)
            r = load_json(out / f"{t}_result.json")
            m = load_json(met / f"{t}_seven.json")
            if r is None and m is None:
                ret[arm["name"]].append(None)
                for k in gen[arm["name"]]:
                    gen[arm["name"]][k].append(None)
                ccos[arm["name"]].append(None)
                continue
            avail[arm["name"]].append(seed)
            ret[arm["name"]].append(retrieval_vector(out, t))
            agg[arm["name"]].append(r or {})
            for k in gen[arm["name"]]:
                gen[arm["name"]][k].append(metric_vector(met, t, k))
            ccos[arm["name"]].append(cond_cos_vector(out, t))

    n = next((v.size for v in ret[REF] if v is not None), None)

    print("## EEGiT-objective vs the recorded control, sub-08")
    print()
    print("Two runs of the PURE semantic tower -- no structural branch, no ControlNet "
          "contribution -- differing only in the loss flags. Both use seed 2025 and "
          "`--split-seed 2025`, so they are scored on the identical 150 val / 200 test "
          "concepts and every comparison below is paired. The unpaired threshold for a "
          "200-way score is ~9.8 points; no claim here relies on it.")
    print()
    print("| arm | seeds | test Top-1 (mean) | per-seed | cos(cond,oracle) | what changed |")
    print("|---|---:|---:|---|---:|---|")
    for a in ARMS:
        nm = a["name"]
        got = [x for x in agg[nm] if x]
        cv = stack(ccos[nm], n, nm) if n else None
        if not got:
            print(f"| `{nm}` | 0/3 | MISSING | | | {a['desc']} |")
            continue
        tops = [x["test"]["top1"] for x in got]
        print(f"| `{nm}` | {len(got)}/3 | {np.mean(tops):.2f} | "
              f"{'/'.join(f'{t:.1f}' for t in tops)} | "
              f"{'-' if cv is None else f'{cv.mean():.4f}'} | {a['desc']} |")

    if n is None:
        print()
        print(f"`{REF}` has no per-concept retrieval vector; nothing below can be paired.")
        return

    # ---- paired, readout by readout -----------------------------------------
    nm = CANDIDATE["name"]
    print()
    print(f"### `{nm}` against `{REF}`, paired per concept")
    print()
    print("| readout | delta | 95% CI | sign p | verdict |")
    print("|---|---:|---|---:|---|")
    rv = stack(ret[REF], n, REF)
    av = stack(ret[nm], n, nm)
    if av is None or rv is None:
        print(f"| test Top-1 | - | - | - | `{nm}` is not scored yet |")
    else:
        v = verdict(av - rv, np.zeros_like(av), n_resamples=args.resamples)
        print(f"| test Top-1 | {100 * v['delta']:+.2f} pts | "
              f"[{100 * v['lo']:+.2f}, {100 * v['hi']:+.2f}] | {v['sign_p']:.3f} | "
              f"{v['verdict']} |")
    for k in LOWKEYS + TWKEYS:
        bv = stack(gen[REF][k], n, f"{REF}/{k}")
        gv = stack(gen[nm][k], n, f"{nm}/{k}")
        if bv is None or gv is None:
            continue
        v = verdict(gv - bv, np.zeros_like(gv), n_resamples=args.resamples)
        print(f"| {k} | {v['delta']:+.3f} | [{v['lo']:+.3f}, {v['hi']:+.3f}] | "
              f"{v['sign_p']:.3f} | {v['verdict']} |")
    cvr = stack(ccos[REF], n, REF)
    cvn = stack(ccos[nm], n, nm)
    if cvr is not None and cvn is not None:
        v = verdict(cvn - cvr, np.zeros_like(cvn), n_resamples=args.resamples)
        print(f"| cos(cond,oracle) | {v['delta']:+.4f} | "
              f"[{v['lo']:+.4f}, {v['hi']:+.4f}] | {v['sign_p']:.3f} | {v['verdict']} |")

    # ---- the generation table -----------------------------------------------
    print()
    print("### Generated images, seed-averaged (pure semantic condition injection)")
    print()
    print("The deliverable: 200 images per (arm, seed) generated by txt2img with the "
          "ControlNet at scale 0, so the only input that differs between arms is whose "
          "EEG built the IP-Adapter condition.")
    print()
    print("| arm | " + " | ".join(TABLE_KEYS) + " |")
    print("|---|" + "---:|" * len(TABLE_KEYS))
    for a in ARMS:
        nm = a["name"]
        met = out / a["met"]
        got = [load_json(met / f"{tag_of(a, s)}_seven.json") for s in SEEDS]
        got = [g for g in got if g]
        if not got:
            print(f"| `{nm}` | " + " | ".join(["MISSING"] * len(TABLE_KEYS)) + " |")
            continue
        cells = []
        for k in TABLE_KEYS:
            vals = [g[k] for g in got if k in g]
            cells.append(f"{np.mean(vals):.3f}" if vals else "-")
        print(f"| `{nm}` | " + " | ".join(cells) + " |")

    # ---- read this before believing any of it -------------------------------
    print()
    print("### How to read the result")
    print()
    print("- A POSITIVE delta on `cos(cond,oracle)` with a Top-1 CI that straddles zero "
          "is the expected shape if the sharper objective sharpens the condition "
          "without changing its rank order: retrieval is invariant to a global "
          "rescaling of the loss, the condition's direction is not.")
    print("- `A0`'s `cos(cond,oracle)` at seed 2025 is 0.6647. If this run does not beat "
          "it by more than its CI, the honest conclusion is that EEGiT's objective is "
          "not the binding term here -- the same shape of answer the interface probe "
          "gave for the tokenizer.")
    print("- The generation numbers are the ones to actually use. A condition can trade "
          "CLIP for PixCorr (the `depth_cn070` arm did), so no single column is the "
          "score; read the seven together, and read `fid` as the outlier detector.")


if __name__ == "__main__":
    main()
