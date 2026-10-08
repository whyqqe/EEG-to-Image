#!/usr/bin/env python
"""Accuracy vs SUBJECT, with the concepts and the model held fixed.

WHY THIS PROBE EXISTS
---------------------
The fold is LOSO on sub-08, so the only number the project has ever recorded is
sub-08's 200-way Top-1 (`raw cosine` 20.33 on the best v4 arm). That number confounds
two very different statements:

  "the encoder decodes these 200 concepts at ~20% from EEG"      (a representation fact)
  "and it does so no worse for a subject it never saw"           (a transfer fact)

They are separable for free, because ALL TEN subjects were shown the same 200 test
concepts. Scoring the same trained model on the 200 test concepts of a subject that WAS
in the training set and on sub-08 that was NOT changes only one variable: whether the
encoder has ever seen that subject's observation statistics. No retraining, no labels, no
fitting -- it is the same `extract_features` path the fold already uses, run nine more
times.

MVNN follows the protocol's own rule: source subjects get `mvnn="train"` (their train
residuals are legitimately available) and the held-out subject gets `mvnn="test"`.

READ: if the nine trained subjects also land near 20%, then sub-08's score is mostly a
statement about the REPRESENTATION, and subject transfer costs only a few points. If they
land near 50-60% while sub-08 sits at 20%, the subject axis is the dominant error and the
`subject == modality` framing is addressing the right problem after all. Either way the
sub-08 cell here should reproduce the recorded 20.33, which is the built-in sanity check.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import evaluate                                   # noqa: E402
from samclip.data.things_eeg import load_subject_std           # noqa: E402
from samclip.models import build_model                         # noqa: E402
from samclip.utils import load_config                          # noqa: E402


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--held-out", type=int, default=8)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg") or load_config("configs/loso_sub08_v4.yaml")
    from samclip import train as train_mod
    _, tte = train_mod.build_targets(cfg)
    model = build_model(cfg, tte.shape[2], tte.shape[-1]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[per-subj] ckpt={args.ckpt} epoch={ck.get('epoch')} arch={cfg.get('arch')}")

    gallery = torch.as_tensor(np.ascontiguousarray(tte[:, 0]), dtype=torch.float32)
    with torch.no_grad():
        z_img = torch.nn.functional.normalize(
            model.encode_target(gallery.to(device), training=False), dim=-1).cpu().numpy()

    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(f"\n  {'sub':<6}{'role':<12}{'mvnn':<7}{'top1':>7}{'top5':>7}{'meanrank':>10}"
          f"{'offset':>9}")
    rows = {}
    for s in subjects:
        mvnn = "test" if s == args.held_out else "train"
        _, te = load_subject_std(s, None, None, mvnn=mvnn)
        x = np.asarray(te[:, 0], dtype=np.float32)              # (200, 63, 250)
        chunks = []
        for i in range(0, len(x), args.batch):
            t = torch.as_tensor(np.ascontiguousarray(x[i:i + args.batch])).to(device)
            chunks.append(model.embed_eeg(t).cpu())
        raw = torch.cat(chunks)
        z = torch.nn.functional.normalize(model.apply_smn(raw.to(device), None), dim=-1)
        r = evaluate.retrieval_report(z.cpu().numpy(), z_img)
        off = model.subject_offset_ratio(raw)
        role = "HELD OUT" if s == args.held_out else "trained"
        rows[s] = r["top1"]
        print(f"  {s:<6}{role:<12}{mvnn:<7}{r['top1']:>7.2f}{r['top5']:>7.2f}"
              f"{r['mean_rank']:>10.1f}{off:>9.4f}")

    trained = [v for k, v in rows.items() if k != args.held_out]
    print(f"\n  trained subjects  mean {np.mean(trained):.2f}  "
          f"sd {np.std(trained, ddof=1):.2f}  range [{min(trained):.2f}, {max(trained):.2f}]")
    print(f"  held-out sub-{args.held_out:02d}      {rows[args.held_out]:.2f}")
    print(f"  SUBJECT GAP (trained mean - held out) = "
          f"{np.mean(trained) - rows[args.held_out]:+.2f} pp")


if __name__ == "__main__":
    main()
