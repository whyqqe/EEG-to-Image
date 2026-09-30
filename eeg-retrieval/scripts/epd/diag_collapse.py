"""Is the structural head predicting a constant, or predicting badly?

The two have opposite fixes -- one is an architecture/optimisation problem, the
other is a data problem -- and the epoch log alone cannot tell them apart, because
both show a variance ratio near 0 with a healthy-looking regression loss.

Measurements, all on the SAVED checkpoint and the real val EEG:

  * the across-sample std of the prediction and of the target, in the same
    normalised space the loss is defined in (that ratio is `vae_var`);
  * the per-sample correlation with the sample's OWN target. A head that has
    collapsed to the conditional mean scores ~0 here while its loss is finite, and
    a head that is predicting badly-but-not-constantly scores above 0;
  * the same two numbers for a freshly-constructed tower, so "the architecture
    cannot pass EEG to its output" is separated from "training destroyed it";
  * the input's own across-sample std, so a dead input path is ruled out.

Usage: diag_collapse.py <ckpt.pt>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epd import config                                      # noqa: E402
from epd.data import concept_split, load_subject             # noqa: E402
from epd.model import build_from_args                        # noqa: E402


def spatial_stats(p: np.ndarray, T: np.ndarray, tag: str) -> None:
    """Across-sample spread and per-sample correlation, in one place so the
    trained / untrained numbers are produced by identical code.

    The variance decomposition is the part that matters when the ratio comes out
    healthy. A total std can be met by two very different predictions, and they
    have opposite fixes:

      WITHIN-sample  (spatial)  the field varies across positions, so the tensor
                                holds a depth MAP;
      ACROSS-sample  (level)    each sample is flat and the samples differ in
                                level, so the tensor holds one number per sample.

    `variance_floor` is computed as `pred.std(dim=(0, 2, 3))`, which sums these two
    into one number -- so a flat-with-varying-level prediction passes the floor
    while carrying no spatial structure at all, and `vae_top1` (centred per
    sample) collapses to chance because centring a flat field gives zero. Both
    halves are printed so that combination is visible rather than inferred from a
    healthy-looking ratio.
    """
    across = float(p.std(axis=0).mean())
    tgt_across = float(T.std(axis=0).mean())
    level = float(p.mean(axis=(2, 3)).std(axis=0).mean())
    spatial = float(p.std(axis=(2, 3)).mean())
    tgt_level = float(T.mean(axis=(2, 3)).std(axis=0).mean())
    tgt_spatial = float(T.std(axis=(2, 3)).mean())
    pf = p.reshape(len(p), -1)
    tf = T.reshape(len(T), -1)
    pc = pf - pf.mean(1, keepdims=True)
    tc = tf - tf.mean(1, keepdims=True)
    pc = pc / (np.linalg.norm(pc, axis=1, keepdims=True) + 1e-9)
    tc = tc / (np.linalg.norm(tc, axis=1, keepdims=True) + 1e-9)
    r_own = float((pc * tc).sum(1).mean())
    # The floor: how well does the SAMPLE-INDEPENDENT mean field do on each target?
    # If the prediction beats this, it is carrying per-concept information; if it is
    # below it, it is worse than a constant at the only thing that matters.
    mean_map = tf.mean(0, keepdims=True)
    mc = mean_map - mean_map.mean(1, keepdims=True)
    mc = mc / (np.linalg.norm(mc, axis=1, keepdims=True) + 1e-9)
    r_floor = float((mc * tc).sum(1).mean())
    return {
        "mean": float(p.mean()), "min": float(p.min()), "max": float(p.max()),
        "var_ratio": across / max(tgt_across, 1e-12),
        "spatial": spatial, "tgt_spatial": tgt_spatial,
        "level": level, "tgt_level": tgt_level,
        "r_own": r_own, "r_floor": r_floor, "margin": r_own - r_floor,
    }


def report(m: dict, tag: str, n_slots: int) -> None:
    print(f"[diag] {tag}  ({n_slots} slot{'s' if n_slots > 1 else ''}, "
          f"averaged exactly as `evaluate_selection` does)")
    print(f"        range mean {m['mean']:+.4f} min {m['min']:+.4f} max {m['max']:+.4f}")
    print(f"        var ratio {m['var_ratio']:.6f}")
    print(f"        decomposition   spatial (within) {m['spatial']:.6f} vs target "
          f"{m['tgt_spatial']:.6f}  |  level (across) {m['level']:.6f} vs target "
          f"{m['tgt_level']:.6f}")
    print(f"        per-sample r(own target)      {m['r_own']:+.4f}")
    print(f"        same for the mean-field pred  {m['r_floor']:+.4f}")
    print(f"        margin over that floor        {m['margin']:+.4f}")


def main() -> int:
    ckpt_path = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["args"]
    print(f"[diag] {ckpt_path}: epoch {ck.get('epoch')}, tag {cfg.get('tag')}")
    print(f"[diag] arch={cfg.get('struct_arch')} target={cfg.get('struct_target')}"
          f"@{cfg.get('struct_scale')} centred={cfg.get('struct_center')}"
          f" loss={cfg.get('vae_loss')} w_var={cfg.get('w_var')}")
    channels = (None if cfg.get("channels", "all") == "all"
                else config.CHANNELS_OCCIPITO_PARIETAL)
    ch_names = (list(channels) if channels else
                json.loads((config.EEG_DIR / "info.json").read_text())["ch_names"])
    tr_eeg, _ = load_subject(cfg["subject"], channels)
    split = concept_split(cfg["val_concepts"], cfg["split_seed"])
    val_eeg = tr_eeg[split.val_concepts]
    print(f"[diag] val EEG {val_eeg.shape}, {len(ch_names)} channels")

    # ---- the target, rebuilt exactly as training built it -----------------
    root = Path(cfg["_coarse_root"]) if cfg.get("_coarse_root") else (
        config.OUTPUTS / "struct_targets" / "coarse")
    scale = int(cfg.get("struct_scale", 64))
    if scale == 64:
        tpath = Path(cfg["depth_cache"]) / f"train_{cfg['struct_target']}_64.npy"
    else:
        tpath = root / f"train_{cfg['struct_target']}_{scale}.npy"
    raw = np.load(tpath)
    if raw.ndim == 3:
        raw = raw[:, None]
    rows = np.sort((split.val_concepts[:, None] * tr_eeg.shape[1]
                    + np.arange(tr_eeg.shape[1])[None, :]).ravel())
    T = np.asarray(raw[rows], dtype=np.float32)
    if cfg.get("_struct_field_mean"):
        f = np.asarray(cfg["_struct_field_mean"], dtype=np.float32).reshape(1, *T.shape[1:])
        T = T - f
    T = T / np.asarray(cfg["_vae_std"], dtype=np.float32).reshape(1, -1, 1, 1)
    print(f"[diag] target {T.shape} from {tpath.name}, mean "
          f"{T.mean():+.4f} std {T.std():.4f}")

    # The input's own spread: if this is ~0 the EEG never reached the tower and
    # nothing downstream is interpretable.
    xin = val_eeg[:, 0]
    print(f"[diag] input EEG {xin.shape}: std {xin.std():.4f}, "
          f"per-channel mean std {xin.std(axis=(0, 2)).mean():.4f}")

    def predict(model, tag):
        """Per-slot metrics, then the average `evaluate_selection` prints.

        The slot axis is not cosmetic. The validation sweep scores all 10
        augmentation slots of every concept and averages, while a single-slot
        evaluation uses one; on a head whose output is still growing the two
        differ by more than an order of magnitude, and reporting one while
        reading the other from the epoch log makes the checkpoint look like it
        collapsed when it did not. Both are produced here so the diag's number can
        be matched against the log's line for the same epoch.
        """
        n_aug = tr_eeg.shape[1]
        model.to(device).eval()
        per_slot = {}
        preds, gts = [], []
        with torch.no_grad():
            for s in range(n_aug):
                acc = []
                for i in range(0, len(val_eeg), 64):
                    x = torch.from_numpy(
                        np.ascontiguousarray(val_eeg[i:i + 64, s])).float().to(device)
                    acc.append(model.forward_all(x, None, training=False)["struct"]["vae"]
                               .float().cpu().numpy())
                p = np.concatenate(acc)
                Ts = T[s::n_aug]
                assert len(p) == len(Ts), f"{len(p)} predictions vs {len(Ts)} targets"
                per_slot[s] = spatial_stats(p, Ts, f"{tag} slot {s}")
                preds.append(p)
                gts.append(Ts)
        keys = per_slot[0].keys()
        avg = {k: float(np.mean([per_slot[s][k] for s in per_slot])) for k in keys}
        report(per_slot[0], f"{tag} [slot 0 only]", 1)
        report(avg, f"{tag} [all {n_aug} slots]", n_aug)
        return avg, np.concatenate(preds), np.concatenate(gts)

    # ---- the trained model ------------------------------------------------
    # The width `img_head` consumes. Read off the checkpoint's own weights rather
    # than from a config key, because the key's name depends on the head kind:
    # `mlp` heads are `img_head.weight` (Linear -> d_embed, input width in dim 1)
    # and EEGiT's `ProjectionHead` is `img_head.projection.weight`, also
    # (d_embed, input_width). Both put the input width in dim 1, but the module
    # name differs, so the lookup tries both instead of assuming one.
    sd = ck["model"]
    w = None
    for cand in ("img_head.weight", "img_head.projection.weight"):
        if cand in sd:
            w = sd[cand]
            break
    if w is None:
        raise SystemExit(f"cannot find img_head's input layer in the checkpoint; "
                         f"keys matching 'img_head': "
                         f"{[k for k in sd if k.startswith('img_head')][:6]}")
    image_dim = int(w.shape[1])
    print(f"[diag] img_head input width {image_dim} (read from {w.shape})")
    model = build_from_args(cfg, ch_names, image_dim).to(device)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    print(f"[diag] load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
    if missing:
        print(f"[diag]   missing[:4] {missing[:4]}")
    _, pred_arr, gt_arr = predict(model, "TRAINED (this checkpoint)")

    # ---- the export's own gate, on this checkpoint ------------------------
    # The gate is what actually decides whether the pipeline proceeds or dies, and
    # it uses a DIFFERENT variance statistic from the one printed above
    # (`x.var(axis=0).mean() / x.var()`, the share of variance that is
    # across-sample, not a ratio of stds). Predicting a pass from the wrong
    # statistic is how a gate failure is discovered after the export has run, so
    # the real function is imported and called rather than reimplemented.
    from epd.export_conds import collapse_gate  # noqa: E402
    g = collapse_gate(pred_arr, gt_arr, "depth")
    print(f"[diag] export collapse gate: pred var ratio {g['pred_var_ratio']:.4f} "
          f"(floor {g['var_ratio_floor']:.2f}), target's own ratio "
          f"{g['target_var_ratio']:.4f}")
    print(f"[diag]   r(pred,gt) {g['r_pred_to_gt']:+.4f} vs r(constant,gt) "
          f"{g['r_constant_to_gt']:+.4f} -> margin {g['margin_over_constant']:+.4f} "
          f"(floor {g['margin_floor']:.2f})")
    print(f"[diag]   -> {'PASS' if g['passed'] else 'FAIL'}; the export would "
          f"{'proceed' if g['passed'] else 'ABORT without --allow-collapse'}")

    # ---- a fresh tower, for comparison ------------------------------------
    # Built from the same config so the only difference is the weights. The depth
    # head is PRETRAINED in this architecture, so this isolates the EEG interface's
    # own pass-through rather than measuring a random projection.
    #
    # Its tokenizer must be given the fit-split statistics before it can run at all
    # (`EEGPatchTokenizer` refuses an unfitted z-score), and the checkpoint's own
    # buffers are the ones the trained model was scored with -- so they are copied
    # across rather than refitted, which keeps the two models' inputs identical and
    # leaves the weights as the only difference.
    fresh = build_from_args(cfg, ch_names, image_dim).to(device)
    fresh.load_state_dict(
        {k: v for k, v in ck["model"].items() if k.endswith("tokenizer.stats_set")
         or k.endswith("tokenizer.eeg_mean") or k.endswith("tokenizer.eeg_std")},
        strict=False)
    predict(fresh, "FRESH init (same config, untrained)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
