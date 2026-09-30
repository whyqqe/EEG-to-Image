"""Channel ablation: correctness audit for a montage change on a trained encoder.

WHY THIS EXISTS
---------------
Every historical run in this project fed the encoder 17 posterior electrodes
(`POSTERIOR_17`), while the published THINGS-EEG2 baselines keep all 63 ("all
electrodes were preserved for analysis").  Nothing in the tree records a
comparison, so `--channels all` is a genuinely untested degree of freedom rather
than a settled default.  Changing it, though, is the one edit that can silently
destroy a result instead of failing: the encoder's input is a FLAT weight of
shape ``(feature_dim, channels * samples)``, so a model fed the wrong montage --
a permutation, a subset of the same length, the right names in the wrong order --
runs to completion and emits plausible-looking features.  Nothing downstream can
detect it.

So the montage change is only trustworthy if two properties hold, and this script
measures both:

  1. EXACT WARM START.  Widening 17 -> 63 channels places the old per-channel
     weight blocks at their positions in the new montage and zeroes the other 46,
     so the widened model must equal the original model BIT-FOR-BIT (to float32
     matmul tolerance) on the same trials.  If it does not, the "channels" arm is
     contaminated by re-initialisation and its result means nothing.

  2. THE MONTAGE IS ACTUALLY 63 AND THE COLUMNS ARE WHERE WE THINK.  The gain has
     to come from the 46 extra electrodes actually carrying signal, and their
     weight columns must start at zero, otherwise "adding channels" was really
     "adding noise".  We check both the data width and the zero-block structure.

Exit code is non-zero if any property fails, so the caller can gate a job on it.

Usage:
  python channel_ablation_audit.py --subject 8 \
      --checkpoint outputs/ocf/intra_enc/sub-08/checkpoint_ss_calib_best.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT))
sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))

from nda_ss_pretrain import warm_start_with_dilation  # noqa: E402
from ss_modules import POSTERIOR_17, SharedSpecificEncoder, resolve_channels  # noqa: E402

FAILS: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label}{('  -- ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(label)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--checkpoint", type=str,
                    default="outputs/ocf/intra_enc/sub-08/checkpoint_ss_calib_best.pth")
    ap.add_argument("--eeg-dir", type=str,
                    default=str(NB_ROOT / "data/things_eeg/preprocessed_eeg"))
    ap.add_argument("--info", type=str,
                    default=str(NB_ROOT / "data/things_eeg/preprocessed_eeg/info.json"))
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    ck_path = Path(args.checkpoint)
    if not ck_path.is_absolute():
        ck_path = NB_ROOT / ck_path
    if not ck_path.is_file():
        sys.exit(f"[FATAL] missing checkpoint {ck_path}")

    all_channels = list(json.loads(Path(args.info).read_text(encoding="utf-8"))["ch_names"])
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    n_ch_ck = int(ck["channels_num"])
    eeg_len = int(ck["eeg_sample_points"])
    img_dim = int(ck["img_dim"])
    n_extra = int(ck.get("n_extra_blocks", 1))
    subjects = [int(s) for s in ck["subjects"]]

    print("=" * 78)
    print("CHANNEL ABLATION AUDIT")
    print(f"  checkpoint      {ck_path}")
    print(f"  trained on      {n_ch_ck} channels, {subjects}, "
          f"channel_set={ck.get('channel_set', '(pre-names: posterior)')}")
    print(f"  released montage {len(all_channels)} channels  ({args.info})")
    print("=" * 78)

    # ---------------------------------------------------------------- montage
    print("\n[1] montage sanity")
    check(len(all_channels) == 63, f"released montage is 63 channels",
          f"got {len(all_channels)}")
    check(n_ch_ck == len(POSTERIOR_17),
          f"checkpoint is the {len(POSTERIOR_17)}-channel posterior run",
          f"channels_num={n_ch_ck}")
    ck_names = ck.get("channel_names")
    if ck_names is None:
        ck_names = list(POSTERIOR_17)
    ck_names = [str(c) for c in ck_names]
    check(ck_names == POSTERIOR_17, "checkpoint montage == POSTERIOR_17, in order",
          "so the dilated columns are the 46 non-posterior electrodes")

    # ------------------------------------------------------------ model pair
    print("\n[2] build 17ch and dilated 63ch models from one checkpoint")
    torch.manual_seed(0)
    m17 = SharedSpecificEncoder(subject_ids=subjects, feature_dim=img_dim,
                               eeg_sample_points=eeg_len, channels_num=n_ch_ck,
                               n_extra_blocks=n_extra, use_adapter=True)
    m17.load_state_dict(ck["model_state_dict"])
    m17.eval()

    m63 = SharedSpecificEncoder(subject_ids=subjects, feature_dim=img_dim,
                               eeg_sample_points=eeg_len, channels_num=len(all_channels),
                               n_extra_blocks=n_extra, use_adapter=True)
    rep = warm_start_with_dilation(m63, ck, resolve_channels("all"), all_channels, eeg_len)
    m63.eval()
    print(f"       loaded={rep['loaded']} dilated={rep['dilated']} skipped={rep['skipped']}")
    check(not rep["skipped"], "every tensor transferred (no silent partial load)",
          f"skipped={rep['skipped']}")
    check(len(rep["dilated"]) == 2,
          "exactly the two flattened-EEG input layers were dilated",
          f"{rep['dilated']}")

    # ---------------------------------------------------- zero outside columns
    print("\n[3] the 46 added electrodes start at exactly zero weight")
    idx = [all_channels.index(c) for c in POSTERIOR_17]
    sd63 = m63.state_dict()
    for key in rep["dilated"]:
        w = sd63[key]
        cols = torch.ones(w.shape[1], dtype=torch.bool)
        for c in idx:
            cols[c * eeg_len:(c + 1) * eeg_len] = False
        block = w[:, cols]
        check(float(block.abs().sum()) == 0.0,
              f"{key}: non-posterior columns are zero",
              f"sum|w|={float(block.abs().sum()):.3e}")
    check(idx == list(range(46, 63)),
          "posterior electrodes are the contiguous tail block 46..62",
          f"idx={idx}")

    # ------------------------------------------------------------- exactness
    print("\n[4] EXACT WARM START: dilated 63ch == original 17ch on real EEG")
    from module.dataset import EEGPreImageDataset  # noqa: E402

    common = dict(eeg_data_dir=args.eeg_dir, time_window=[0, 250],
                  rn50_dir=str(NB_ROOT / "data/things_eeg/image_feature/RN50"))
    ds17 = EEGPreImageDataset([args.subject], args.eeg_dir, POSTERIOR_17, [0, 250],
                              common["rn50_dir"], "", False, [], True, False, None,
                              False, False, False, False)
    ds63 = EEGPreImageDataset([args.subject], args.eeg_dir, [], [0, 250],
                              common["rn50_dir"], "", False, [], True, False, None,
                              False, False, False, False)
    check(int(ds17.channels_num) == 17, "17ch dataset yields 17 channels",
          f"got {ds17.channels_num}")
    check(int(ds63.channels_num) == 63, "all-channel dataset yields 63 channels",
          f"got {ds63.channels_num}")
    check(int(ds17.num_sample_points) == eeg_len == 250,
          "sample window matches the checkpoint", f"{eeg_len}")

    eeg17, *rest = next(iter(DataLoader(ds17, batch_size=args.batch, shuffle=False)))
    eeg63 = next(iter(DataLoader(ds63, batch_size=args.batch, shuffle=False)))[0]
    sid = rest[2]
    # the 17 channels must appear as the tail block of the 63-channel tensor,
    # otherwise the montage order assumption behind the dilation is wrong
    tail = eeg63[:, 46:, :]
    check(bool(torch.allclose(eeg17, tail, atol=1e-5)),
          "posterior trials sit at the tail of the 63-channel tensor",
          f"max|diff|={float((eeg17 - tail).abs().max()):.3e}")

    with torch.no_grad():
        o17 = m17(eeg17, sid)
        o63 = m63(eeg63, sid)
    dmax = float((o17 - o63).abs().max())
    rel = dmax / max(float(o17.abs().max()), 1e-9)
    check(dmax < 1e-4, "dilated 63ch output == 17ch output on the same trials",
          f"max|diff|={dmax:.3e} (rel {rel:.2e})")

    # a NEGATIVE control: the property must be falsifiable. Permuting two
    # posterior ELECTRODES in the 63ch tensor has to move the output, otherwise
    # the tail-block check above is vacuous because the extra channels were never
    # read. (Index the channel axis, not the flattened axis: the dataset hands
    # back (batch, channels, samples), and a flattened-style slice here would be
    # empty and would silently pass.)
    perm = eeg63.clone()
    a, b = 46, 47
    perm[:, a, :] = eeg63[:, b, :]
    perm[:, b, :] = eeg63[:, a, :]
    check(not torch.allclose(perm, eeg63), "NEGATIVE CONTROL setup actually permuted data",
          f"max|diff|={float((perm - eeg63).abs().max()):.3e}")
    with torch.no_grad():
        o_bad = m63(perm, sid)
    moved = float((o_bad - o63).abs().max())
    check(moved > 1e-4, "NEGATIVE CONTROL: permuting two posterior electrodes changes output",
          f"max|diff|={moved:.3e} (must be > 1e-4 or the test is vacuous)")

    # and the mirror control: perturbing a channel that the dilation left at zero
    # weight must NOT matter for the 17ch-consistent part of the model. This pins
    # down which electrodes the warm-start arm can possibly be using.
    zeros = eeg63.clone()
    zeros[:, 0, :] = 10.0
    with torch.no_grad():
        o_zero = m63(zeros, sid)
    check(float((o_zero - o63).abs().max()) <= 1e-5,
          "added electrodes (idx<46) are inert at init, as the zero-block implies",
          f"max|diff|={float((o_zero - o63).abs().max()):.3e}")

    print("\n" + "=" * 78)
    if FAILS:
        print(f"AUDIT FAILED ({len(FAILS)}): " + "; ".join(FAILS))
        sys.exit(1)
    print("AUDIT PASSED: the 63-channel arm is an exact, single-variable extension "
          "of the 17-channel baseline.")
    print("=" * 78)


if __name__ == "__main__":
    main()
