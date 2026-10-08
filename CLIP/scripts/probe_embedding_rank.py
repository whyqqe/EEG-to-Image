#!/usr/bin/env python
"""Is the trunk's 128-dim aggregator actually binding?

`EEGTrunk` ends in `TemporalSpatialAggregator` whose `out_dim = agg_width * agg_pool`
(= 128 at the defaults) and the head then maps that to `d_embed` 512. So the embedding's
RANK CEILING is 128 regardless of `d_model`: whatever widening happens upstream, only
128 directions can survive the pooled vector.

Whether that ceiling is the limiter is a measurement, not an argument. If the live
embedding already occupies far fewer than 128 effective dimensions, the cap is slack and
raising it buys nothing; if the spectrum is crowded up against 128, the cap is the
constraint and every other lever is downstream of it.

Reports, per checkpoint, the spectrum of the (N, d_embed) EEG embeddings and of the image
targets, plus the same for the target so the EEG's rank can be read as a fraction of the
geometry it is trying to match.

Run:  python scripts/probe_embedding_rank.py --ckpts a/last.pt b/last.pt \
          --target-subject 8 --mvnn test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import config, evaluate  # noqa: E402
from samclip.data import things_eeg  # noqa: E402
from samclip.data.targets import load_target_stack  # noqa: E402
from samclip.models import build_model  # noqa: E402


def spectrum_stats(x: np.ndarray) -> dict:
    """Rank diagnostics of a (N, D) matrix, on the CENTRED data (variance, not spread)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    xc = x - x.mean(axis=0, keepdims=True)
    # Economy SVD: at N=200 << D=512 the Gram matrix is the cheap side.
    s = np.linalg.svd(xc, compute_uv=False)
    s2 = s ** 2
    s2 = s2 / max(s2.sum(), 1e-30)          # variance fractions
    nz = s2[s2 > 1e-12]
    # Participation ratio / "effective rank": (sum l)^2 / sum l^2 -- the standard
    # soft count of dimensions carrying variance.
    eff = float((s2.sum() ** 2) / max((s2 ** 2).sum(), 1e-30))
    entropy_rank = float(np.exp(-(nz * np.log(nz)).sum()))
    cum = np.cumsum(s2)
    return {
        "n": n,
        "d": x.shape[1],
        "eff_rank": eff,
        "entropy_rank": entropy_rank,
        "dims_90": int(np.searchsorted(cum, 0.90) + 1),
        "dims_99": int(np.searchsorted(cum, 0.99) + 1),
        "sv1_frac": float(s2[0]),
        "nonzero_sv": int((s > 1e-8 * max(s[0], 1e-30)).sum()),
    }


def extract(ckpt_path: Path, target_subject: int, mvnn: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt["cfg"]
    channel_set = model_cfg.get("channel_set", "all63")
    channels = (config.CHANNELS_OCCIPITO_PARIETAL
                if channel_set == "occipital17" else None)
    img = model_cfg.get("image", {}) or {}
    _, test = things_eeg.load_subject_std(target_subject, channels, mvnn=mvnn)
    targets_te = load_target_stack(img.get("feature_set", "clip_h14_multilevel"),
                                   img.get("layers"), "test")
    model = build_model(model_cfg, targets_te.shape[2], targets_te.shape[-1])
    model.load_state_dict(ckpt["model"])
    model.eval()
    loader = DataLoader(things_eeg.TestDataset(np.asarray(test), targets_te),
                        batch_size=200, shuffle=False, collate_fn=things_eeg.collate)
    with torch.no_grad():
        feats = evaluate.extract_features(model, loader, torch.device("cpu"))
    return feats, model, model_cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--target-subject", type=int, required=True)
    ap.add_argument("--mvnn", default="test")
    args = ap.parse_args()

    for c in args.ckpts:
        p = Path(c)
        tag = f"{p.parent.parent.name}/{p.parent.name}"
        feats, model, cfg = extract(p, args.target_subject, args.mvnn)

        # The cap we are testing: the aggregator's pooled width, straight from the model.
        agg_cap = getattr(model.trunk.agg, "out_dim", None)
        trunk_out = getattr(model.trunk, "out_dim", None)

        print(f"\n{'=' * 78}\n[{tag}]\n{'=' * 78}")
        print(f"  agg out_dim (RANK CEILING) = {agg_cap}   trunk out_dim = {trunk_out}   "
              f"d_embed = {feats['eeg'].shape[1]}")
        for label, x in (("eeg ", feats["eeg"]), ("img ", feats["img"])):
            st = spectrum_stats(x)
            print(f"  {label} eff_rank {st['eff_rank']:7.2f}  entropy {st['entropy_rank']:7.2f}  "
                  f"dims@90% {st['dims_90']:4d}  dims@99% {st['dims_99']:4d}  "
                  f"sv1 {st['sv1_frac']:.3f}  nnz_sv {st['nonzero_sv']}")
        st_e = spectrum_stats(feats["eeg"])
        if agg_cap:
            usage = 100.0 * st_e["dims_99"] / agg_cap
            print(f"  -> aggregator cap usage: {st_e['dims_99']}/{agg_cap} dims at 99% "
                  f"variance = {usage:.1f}%   "
                  f"{'CAP IS BINDING' if usage > 85 else 'cap is slack'}")


if __name__ == "__main__":
    main()
