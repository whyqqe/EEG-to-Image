#!/usr/bin/env python
"""Cache the trunk features of a trained (or untrained) dual tower, for a closed-form probe.

Why this exists
---------------
The structural branch has now failed three times, and every one of those failures was
diagnosed from a *trained* model's output. That is the wrong instrument for the
question, because a bad end-to-end score conflates two very different causes:

  (a) the interface (topography tokenizer -> pretrained patch_embed -> frozen blocks)
      never let the information through -- nothing downstream could have recovered it;
  (b) the information IS in the features and the decoder / objective threw it away.

A trained checkpoint cannot distinguish those: whatever the loss discarded is gone
from the features too, so "the trained encoder is at chance" is equally consistent
with both. This script caches the features so the SAME ridge probe that already
measured the raw-EEG ceiling (5.07% instance Top-1 on sub-08, 63ch) can be re-run
with those features as its input. Three rows then settle it:

    raw EEG        5.07%   the closed-form ceiling from the signal itself
    pretrained     ?       (a) -- does the interface carry it at all, before any training?
    trained        ?       (b) -- did training keep it?

  pretrained ~ raw, trained ~ chance  -> the objective/optimiser destroyed it
  pretrained << raw                   -> the interface destroyed it, and no loss fixes that

The `--pretrained-only` mode is what makes the first of those rows possible: it builds
the identical architecture from the run's own saved args but does NOT load the trained
weights, so the features are exactly what the released pretrained checkpoints plus the
untrained decoder would produce at initialisation.

Usage
-----
    # both variants, into one cache dir (needs a GPU: see slurm/epd_struct_probe.sbatch)
    python scripts/epd/extract_probe_features.py --subject 8 \
        --ckpt outputs/sub08/epd_dual_dino3_best.pt \
        --sources struct_grid struct_pooled sem_embed --out-dir outputs/probe_feats/dino3

The output layout is (n_concepts, n_slots, D) per source, which is what
`probe_targets.py --feature-cache` expects; the probe then fits on the fit concepts
and reports on the held-out ones, exactly as it does for raw EEG.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epd import config                      # noqa: E402  (sets HF cache paths)
from epd.data import concept_split, load_subject   # noqa: E402
from epd.model import build_from_args       # noqa: E402

# What each source is, and why it is in the list. `struct_grid` is the one the
# structural decoder actually consumes; `struct_pooled` is after the tower's fusion
# and LayerNorm, so it is a different tensor from the grid's mean and the pair
# separates "the decoder was handed a bad summary" from "the grid itself is empty".
SOURCES = ("struct_grid", "struct_gridmean", "struct_pooled", "sem_embed", "sem_fused")


def head_input_width(sd: dict) -> int:
    """The width `img_head` was built at, read off the checkpoint's own weights.

    Copied from `export_conds.py` rather than imported, because that module runs a
    full export at import time of its `main` only -- but the lookup is subtle enough
    (three head kinds, three different key names) that a second, slightly different
    implementation would be worse than a duplicated one. See the long comment there.
    """
    for key in ("img_head.weight", "img_head.0.weight", "img_head.projection.weight"):
        w = sd.get(key)
        if w is not None and w.ndim == 2:
            return int(w.shape[1])
    cands = {k: v for k, v in sd.items()
             if k.startswith("img_head.") and v.ndim == 2}
    if not cands:
        raise SystemExit("checkpoint has no 2-D `img_head.*` weight to read the "
                         f"alignment width from: {sorted(k for k in sd if 'img_head' in k)}")
    return int(cands[max(cands, key=lambda k: cands[k].shape[1])].shape[1])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--ckpt", required=True,
                    help="the run's checkpoint; its `args` define the architecture and "
                         "its `model` the trained weights")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    ap.add_argument("--tag", default="", help="label baked into the manifest")
    ap.add_argument("--val-concepts", type=int, default=150)
    ap.add_argument("--split-seed", type=int, default=2025)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-concepts", type=int, default=0,
                    help="smoke only: truncate the train concepts")
    a = ap.parse_args()
    if not a.tag:
        a.tag = Path(a.ckpt).stem
    return a


def main() -> None:
    a = parse_args()
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available())
                       else "cpu")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[feat] device={dev} subject=sub-{a.subject:02d} -> {out}")

    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["args"]
    if not cfg.get("struct_backbone"):
        raise SystemExit("this checkpoint has no structural tower (args.struct_backbone "
                         "is empty), so there is no structural feature to probe")
    channels = (None if cfg.get("channels", "all") == "all"
                else config.CHANNELS_OCCIPITO_PARIETAL)
    ch_names = (list(channels) if channels
                else json.loads((config.EEG_DIR / "info.json").read_text())["ch_names"])
    image_dim = head_input_width(ckpt["model"])
    print(f"[feat] ckpt {a.ckpt} (epoch {ckpt.get('epoch')}, tag {cfg.get('tag')}) "
          f"struct_backbone={cfg['struct_backbone']} channels={len(ch_names)}")

    tr_eeg, te_eeg = load_subject(a.subject, channels)     # (C,10,Ch,T), (200,1,Ch,T)
    print(f"[feat] EEG train {tr_eeg.shape} test {te_eeg.shape}")

    split = concept_split(a.val_concepts, a.split_seed)
    fit_c = split.fit_concepts
    if a.limit_concepts:
        # Smoke only. The fit list has to be filtered alongside the array, or the
        # truncation turns into an index error rather than a smaller run -- and the
        # cache it writes is deliberately NOT usable by `probe_targets.py` (which
        # asserts the full concept count), because a probe over 12 concepts would
        # still print itself as a 200-way result.
        tr_eeg = tr_eeg[:a.limit_concepts]
        fit_c = [c for c in fit_c if c < a.limit_concepts]
        print(f"[feat] --limit-concepts {a.limit_concepts} -> train EEG {tr_eeg.shape}, "
              f"{len(fit_c)} fit concepts (smoke cache; not probeable)")

    # ---------------- the two variants, in one process
    #
    # `trained` loads the checkpoint. `pretrained` builds the identical architecture
    # from the same args and stops there, so the ONLY difference between the two
    # feature sets is the training that happened in between -- which is the whole
    # point of the comparison.
    variants: dict[str, dict] = {}
    for name, load in (("trained", True), ("pretrained", False)):
        model = build_from_args(cfg, ch_names, image_dim).eval()
        if load:
            missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
            if missing or unexpected:
                raise SystemExit(
                    f"checkpoint does not match the model built from its own args: "
                    f"missing={list(missing)[:6]} unexpected={list(unexpected)[:6]}")
        # The z-score buffers arrive with the state dict on the trained variant. On
        # the pretrained one they are unset, and an unset tokenizer raises rather
        # than passing raw values through -- so fit them from the same fit split
        # `train.py` used. Both branches are then guaranteed to see identical inputs.
        for tok in (model.encoder.tokenizer,
                    (model.struct.encoder.tokenizer if model.struct is not None else None)):
            if tok is None:
                continue
            if not bool(tok.stats_set):
                tok.set_norm_stats(tr_eeg[fit_c])
                print(f"[feat] {name}: fitted z-score stats for "
                      f"{type(tok).__name__} from {len(fit_c)} fit concepts")
        variants[name] = {"model": model.to(dev), "loaded": load}

    # The two tokenizers must agree on the z-score, or the `pretrained` variant would
    # differ from `trained` by its input scaling as well as by its weights, and the
    # comparison would be measuring the wrong thing.
    t_sem = variants["trained"]["model"].encoder.tokenizer
    t_st = variants["trained"]["model"].struct.encoder.tokenizer
    d_mean = float((t_sem.eeg_mean - t_st.eeg_mean).abs().max())
    d_std = float((t_sem.eeg_std - t_st.eeg_std).abs().max())
    if max(d_mean, d_std) > 1e-4:
        raise SystemExit(f"the two tokenizers carry different z-score statistics "
                         f"(max |dmean| {d_mean:.2e}, max |dstd| {d_std:.2e}); the "
                         f"structural branch is seeing differently scaled EEG than "
                         f"the semantic one, which is a bug in training, not a probe "
                         f"result")
    p_sem = variants["pretrained"]["model"].encoder.tokenizer
    dz = float((p_sem.eeg_mean - t_sem.eeg_mean).abs().max())
    if dz > 1e-4:
        # Only enforced on a real run. `--limit-concepts` deliberately fits the
        # refitted statistics on a handful of concepts, so a mismatch there is the
        # flag doing its job, not a counterfactual that has gone dirty -- and the
        # cache such a run writes is already refused by the probe.
        msg = (f"the refitted z-score statistics do not reproduce the checkpoint's "
               f"(max |dmean| {dz:.2e}); the pretrained variant would differ from the "
               f"trained one by its input scaling as well as by its weights, so the "
               f"comparison would not be clean")
        if a.limit_concepts:
            print(f"[feat] NOTE (smoke): {msg}; expected under --limit-concepts")
        else:
            raise SystemExit(msg)

    # ---------------- forward everything once per variant
    #
    # Train and test are accumulated in SEPARATE lists. They are not interchangeable
    # and cannot share a buffer: the train split carries 10 image slots per concept
    # and the test split carries 1, so concatenating them raises on a shape mismatch
    # -- which is what the first smoke run did. Keeping them apart is also what the
    # probe needs: it slices `arr[:n_train]` against `arr[n_train:]` to recover the
    # two splits from a single file, so the layout has to be train-major and complete.
    collected: dict[str, dict[str, dict[str, list]]] = {
        v: {"train": {s: [] for s in a.sources}, "test": {s: [] for s in a.sources}}
        for v in variants}
    t0 = time.time()
    for vname, v in variants.items():
        model = v["model"]
        for split_name, eeg in (("train", tr_eeg), ("test", te_eeg)):
            n_c, n_s = eeg.shape[0], eeg.shape[1]
            for i in range(0, n_c, a.batch_size):
                j = min(i + a.batch_size, n_c)
                x = torch.from_numpy(
                    np.ascontiguousarray(eeg[i:j].reshape((j - i) * n_s, *eeg.shape[2:]))
                ).to(dev)
                with torch.no_grad():
                    sd = model.forward_all(x, None, training=False)
                    z, fused, _ = model.encode_eeg(x, None, training=False)
                    st = sd["struct"]
                    feats = {
                        "struct_grid": st["grid"].flatten(1),
                        "struct_gridmean": st["grid"].mean(1),
                        "struct_pooled": st["fused"],
                        "sem_embed": z,
                        "sem_fused": fused,
                    }
                for s in a.sources:
                    collected[vname][split_name][s].append(
                        feats[s].float().cpu().numpy().reshape(j - i, n_s, -1))
            print(f"[feat] {vname}/{split_name}: {n_c} concepts x {n_s} slots done "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # ---------------- write
    manifest = {"ckpt": a.ckpt, "tag": a.tag, "subject": a.subject,
                "struct_backbone": cfg["struct_backbone"],
                "n_train_concepts": int(tr_eeg.shape[0]),
                "n_train_slots": int(tr_eeg.shape[1]),
                "n_test_concepts": int(te_eeg.shape[0]),
                "n_test_slots": int(te_eeg.shape[1]),
                "sources": {}, "variants": {}}
    for vname in variants:
        manifest["variants"][vname] = {"loaded_from_ckpt": variants[vname]["loaded"]}
    for s in a.sources:
        for vname in variants:
            tr_arr = np.concatenate(collected[vname]["train"][s], axis=0)
            te_arr = np.concatenate(collected[vname]["test"][s], axis=0)
            # Two files, not one array with a shared concept axis. The train split has
            # 10 image slots per concept and the test split has 1, so there is no single
            # (concepts, slots, D) layout that holds both -- an earlier version tried
            # and the guard below is what caught it. Splitting also mirrors the target
            # caches this probe is scored against (`train_vae_latents_f16.npy` /
            # `test_vae_latents_f16.npy`), so the two sides of the comparison are
            # stored the same way.
            name = f"{vname}_{s}"
            paths = {}
            for split_name, arr in (("train", tr_arr), ("test", te_arr)):
                p = out / f"X_{name}_{split_name}.npy"
                np.save(p, arr)
                paths[split_name] = str(p)
                print(f"[feat] wrote {p} {arr.shape} {arr.dtype}")
            manifest["sources"][name] = {
                "variant": vname, "feature": s, **paths,
                "train_shape": list(tr_arr.shape), "test_shape": list(te_arr.shape)}

    # A cross-variant difference of exactly zero would mean the checkpoint never
    # got loaded -- the failure mode that would make the whole comparison vacuous.
    for s in a.sources:
        d = float(np.abs(collected["trained"]["train"][s][0]
                         - collected["pretrained"]["train"][s][0]).max())
        print(f"[feat] trained-vs-pretrained max |delta| on {s}: {d:.4e}")
        if d == 0.0:
            raise SystemExit(f"the trained and pretrained features are IDENTICAL for "
                             f"{s}; the checkpoint weights did not load, and every row "
                             f"of the probe would be the same run reported twice")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[feat] wrote {out / 'manifest.json'}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
