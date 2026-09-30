#!/usr/bin/env python3
"""Backfill the per-concept test retrieval vector for an existing checkpoint.

Why this exists
---------------
`train.py` now writes `test.per_concept` (the per-concept Top-1/Top-5 indicators and
ranks), so that two arms can be PAIRED over the same 200 test concepts instead of
compared as two independent 200-way accuracies. Every checkpoint trained before that
change has no such vector -- and the most load-bearing comparison in this project,
`sem_only` (whose semantic tower was trained jointly with the DA2 structural tower)
against the same semantic configuration trained ALONE, is between one arm that has the
vector and one that does not.

The alternative was to declare that comparison unpaired, which would leave the
architecture claim resting on the ~10-point threshold of a 200-way score while the
generation-side version of the same comparison is resolved to ~0.02. Re-running the
forward pass costs seconds on a GPU and removes the asymmetry, so this does that.

It is read-only with respect to the run it scores: it writes a SEPARATE
`<tag>_perconcept.json` rather than editing `_result.json`. `_result.json` is the
record of what training produced, and rewriting it afterwards would make the record
depend on when it was read.

Everything below mirrors `train.py:main`'s final-evaluation path deliberately, including
the definition of `channels` and the way the target is assembled, because the pairing
is POSITIONAL: a row order that differs from training's by one permutation would
produce a plausible-looking vector that pairs nothing.

Usage
-----
    python scripts/epd/per_concept_eval.py \
        --ckpt outputs/sub08/epd_da2_depth8_best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epd import config  # noqa: E402
from epd.data import load_subject  # noqa: E402
from epd.model import build_from_args  # noqa: E402
from epd.train import evaluate  # noqa: E402


def load_target(tdir: Path, keys: list[str]) -> np.ndarray:
    """Assemble the test-side alignment target the same way `train.py` does.

    One key -> a (N, M, D) array; several -> an (N, M, k, D) stack, because the
    downstream batch layout treats the last axis as the feature and the `k` axis is
    carried through. Reproduced rather than simplified: a single-key run and a
    multi-key run are different tensors, and only the former would survive a naive
    `np.load`.
    """
    arrs = []
    for key in keys:
        f = tdir / "test" / f"{key}.npy"
        if not f.is_file():
            raise SystemExit(f"target layer {key!r} has no test split under {tdir}")
        arrs.append(np.load(f))
    if len(arrs) == 1:
        return arrs[0]
    return np.stack(arrs, axis=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise SystemExit(f"--ckpt {ckpt_path} does not exist")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "args" not in ck:
        raise SystemExit(f"{ckpt_path} does not carry an `args` block; cannot rebuild")
    a = ck["args"]
    tag = str(a.get("tag", ckpt_path.stem))

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        if not args.allow_cpu:
            raise SystemExit("CUDA is not available; submit this via sbatch, or pass "
                             "--allow-cpu for a deliberate smoke test")
        print("WARNING: running on CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    # ---- channels, exactly as `train.py` derives them --------------------------
    channels = None if a.get("channels") == "all" else config.CHANNELS_OCCIPITO_PARIETAL
    ch_names = list(channels) if channels else None
    _tr_eeg, te_eeg = load_subject(int(a.get("subject", 8)), channels)
    if ch_names is None:
        ch_names = json.loads((config.EEG_DIR / "info.json").read_text())["ch_names"]

    # ---- the alignment target, exactly as `train.py` assembles it --------------
    keys = list(a.get("target_layers") or [a.get("target_layer", "_pooled")])
    if a.get("target_features"):
        tdir = Path(a["target_features"])
        img_te = load_target(tdir, keys)
        width = int(np.load(tdir / "train" / f"{keys[0]}.npy", mmap_mode="r").shape[-1])
    else:
        img_te = np.load(config.IMAGE_FEATURE_DIR / "image_test.npy")
        width = int(np.load(config.IMAGE_FEATURE_DIR / "image_train.npy",
                            mmap_mode="r").shape[-1])

    model = build_from_args(a, ch_names, width)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint does not match this model: "
                         f"missing={list(missing)[:6]} unexpected={list(unexpected)[:6]}")
    model.to(device).eval()

    # The z-score statistics are buffers and came back with the state dict. Checked
    # rather than assumed: a model would run with `stats_set` False and normalise by
    # 0/1, producing a score that looks like a poor arm instead of a broken script.
    tok = getattr(model.encoder, "tokenizer", None)
    if tok is not None and hasattr(tok, "stats_set") and not bool(tok.stats_set):
        raise SystemExit("checkpoint does not carry the tokenizer's z-score statistics, "
                         "so the EEG would be normalised by 0/1")

    n_te = int(te_eeg.shape[0])
    print(f"[eval ] {tag}: {n_te} test concepts, {len(ch_names)} channels, "
          f"target {list(keys)}")

    rep = evaluate(model, te_eeg, img_te, device,
                   batch=args.batch_size, l2norm=not bool(a.get("no_img_l2norm", False)))
    pc = rep["per_concept"]
    mean_pc = 100.0 * float(np.mean(pc["top1"]))
    if abs(mean_pc - rep["top1"]) > 1e-9:
        raise SystemExit(f"per-concept mean {mean_pc} != aggregate {rep['top1']}")
    # The pairing is positional, so the length is part of the contract.
    if int(pc["n"]) != n_te:
        raise SystemExit(f"per-concept vector has {pc['n']} entries, expected {n_te}")

    out = Path(args.out) if args.out else ckpt_path.with_name(f"{tag}_perconcept.json")
    out.write_text(json.dumps({
        "tag": tag, "ckpt": str(ckpt_path), "n": int(pc["n"]),
        "order": "index i is test concept i, the same order as list_test_images() and "
                 "as the EEG test rows",
        "note": "backfilled from the checkpoint by scripts/epd/per_concept_eval.py; the "
                "run predates train.py writing this vector",
        "top1": rep["top1"], "top5": rep["top5"], "mean_rank": rep["mean_rank"],
        "per_concept": pc,
    }, indent=2), encoding="utf-8")
    print(f"[done] {tag}: test Top-1 {rep['top1']:.2f} (top5 {rep['top5']:.2f}, "
          f"mean_rank {rep['mean_rank']:.2f}) -> {out}")


if __name__ == "__main__":
    main()
