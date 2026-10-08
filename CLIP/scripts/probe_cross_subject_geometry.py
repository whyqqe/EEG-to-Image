#!/usr/bin/env python
"""Does the shared encoder put an UNSEEN subject's concept geometry where a SEEN subject's is?

WHY THIS PROBE EXISTS
---------------------
The v4 redesign's central claim is "subjects are modalities, and one shared map aligns
them". The project has measured that claim only at 200-way retrieval on 200 *test*
concepts of the held-out subject (`raw cosine` = the deployment number), which confounds
three different failures into one score:

  (a) the encoder does not know the concepts      -- an Axis-1 / capacity failure
  (b) the encoder knows them but places the unseen subject's cloud elsewhere
  (c) the per-concept responses are simply noisy

Every subject in THINGS-EEG2 saw the SAME 1654 TRAIN concepts. So (b) can be measured
directly and with no training: encode a SEEN subject (sub-01) and the UNSEEN held-out
subject (sub-08) on the same train concepts, average each concept's 10 image trials into
a prototype, and ask whether sub-08's prototype for concept c lands nearest to sub-01's
prototype for concept c. Cross-subject concept identification like this isolates (b): the
concepts are shared by construction and the subject is the only thing varying.

The controls that make it interpretable:

  * within-08 and within-01 split-half -- the SAME measurement with the subject held
    fixed, using two disjoint halves of that concept's 10 trials. This is the ceiling
    the cross-subject number has to be compared against. If cross ~= within, the unseen
    subject is aligned as well as the noise allows and the problem is elsewhere.
  * EEG -> image on the train concepts -- the deployed task, but with concepts the
    source subjects trained on, i.e. "seen concepts, unseen subject".

No target labels are used anywhere and nothing is fitted. Every number is a retrieval
accuracy over `--concepts` alternative concepts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import config, evaluate                     # noqa: E402
from samclip.data.things_eeg import load_subject_std     # noqa: E402
from samclip.models import build_model                   # noqa: E402
from samclip.utils import load_config                    # noqa: E402


@torch.no_grad()
def encode_eeg(model, x: np.ndarray, device, batch: int = 200) -> np.ndarray:
    """``(N, Ch, T)`` -> ``(N, d)`` through the model's OWN deployment path."""
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        t = torch.as_tensor(np.ascontiguousarray(x[i:i + batch]), dtype=torch.float32)
        t = t.to(device)
        raw = model.embed_eeg(t)
        z = model.apply_smn(raw, None)      # None == "the batch IS one subject"
        out.append(torch.nn.functional.normalize(z, dim=-1).cpu())
    return torch.cat(out).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seen", type=int, default=1, help="a source (trained-on) subject")
    ap.add_argument("--unseen", type=int, default=8, help="the held-out subject")
    ap.add_argument("--concepts", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg") or load_config("configs/loso_sub08_v4.yaml")
    from samclip import train as train_mod
    ttr, _ = train_mod.build_targets(cfg)
    K, D = ttr.shape[2], ttr.shape[-1]
    model = build_model(cfg, K, D).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[probe] ckpt={args.ckpt} epoch={ck.get('epoch')} arch={cfg.get('arch')} "
          f"d_align={cfg.get('d_align')} hidden={cfg.get('share_head_hidden')}")

    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.permutation(ttr.shape[0])[:args.concepts])
    n = len(idx)

    # --- the two subjects' EEG on the SAME concepts -------------------------
    tr_seen, _ = load_subject_std(args.seen, None, None, mvnn="train")
    tr_unseen, _ = load_subject_std(args.unseen, None, None, mvnn="train")
    seen = np.asarray(tr_seen[idx], dtype=np.float32)        # (n, 10, 63, 250)
    unseen = np.asarray(tr_unseen[idx], dtype=np.float32)
    print(f"[probe] {n} concepts x {seen.shape[1]} images | seen=sub-{args.seen:02d} "
          f"unseen=sub-{args.unseen:02d}")

    def protos(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        b, i, c, t = arr.shape
        flat = encode_eeg(model, arr.reshape(b * i, c, t), device).reshape(b, i, -1)
        half = i // 2
        return flat.mean(1), flat[:, :half].mean(1), flat[:, half:].mean(1)

    f_seen, h1_seen, h2_seen = protos(seen)
    f_uns, h1_uns, h2_uns = protos(unseen)

    def rep(name: str, q: np.ndarray, g: np.ndarray, centre: bool = False) -> float:
        qq, gg = (q - q.mean(0, keepdims=True), g - g.mean(0, keepdims=True)) \
            if centre else (q, g)
        r = evaluate.retrieval_report(qq, gg)
        print(f"  {name:<46} top1 {r['top1']:6.2f}   top5 {r['top5']:6.2f}   "
              f"meanrank {r['mean_rank']:6.1f}")
        return r["top1"]

    print(f"\n[probe] {n}-way concept identification "
          f"({'as encoded' if True else ''}, SMN already inside the forward):")
    w_seen = rep(f"within sub-{args.seen:02d}  (half1 -> half2)", h1_seen, h2_seen)
    w_uns = rep(f"within sub-{args.unseen:02d} (half1 -> half2)", h1_uns, h2_uns)
    x_8to1 = rep(f"CROSS  sub-{args.unseen:02d} -> sub-{args.seen:02d}",
                 f_uns, f_seen)
    x_1to8 = rep(f"CROSS  sub-{args.seen:02d} -> sub-{args.unseen:02d}",
                 f_seen, f_uns)
    print(f"\n[probe] with an explicit re-centring of both clouds (deployment-legal):")
    x_8to1c = rep(f"CROSS  sub-{args.unseen:02d} -> sub-{args.seen:02d}  +centre",
                  f_uns, f_seen, centre=True)
    x_1to8c = rep(f"CROSS  sub-{args.seen:02d} -> sub-{args.unseen:02d}  +centre",
                  f_seen, f_uns, centre=True)

    # --- the actual deployed task, on TRAIN concepts ("seen concepts, unseen subject")
    gal = torch.as_tensor(ttr[idx].mean(axis=1), dtype=torch.float32).to(device)  # (n,K,D)
    with torch.no_grad():
        z_img = torch.nn.functional.normalize(
            model.encode_target(gal, training=False), dim=-1).cpu().numpy()
    print(f"\n[probe] EEG -> IMAGE on the same train concepts (seen concepts, "
          f"unseen subject):")
    rep(f"sub-{args.unseen:02d} -> image", f_uns, z_img)
    rep(f"sub-{args.unseen:02d} -> image  +centre", f_uns, z_img, centre=True)
    rep(f"sub-{args.seen:02d} -> image", f_seen, z_img)
    rep(f"sub-{args.seen:02d} -> image  +centre", f_seen, z_img, centre=True)

    # --- offsets, for the D1 record
    def off(z: np.ndarray) -> float:
        z = np.asarray(z, np.float64)
        return float(np.linalg.norm(z.mean(0)) / np.linalg.norm(z, axis=1).mean())
    print(f"\n[probe] offset_ratio  sub-{args.seen:02d}={off(f_seen):.4f}  "
          f"sub-{args.unseen:02d}={off(f_uns):.4f}")

    print(f"\n[probe] READ: if CROSS is close to the within-subject ceilings "
          f"({w_seen:.1f}/{w_uns:.1f}), the unseen subject's concept geometry IS being "
          f"aligned and the deployment gap is not a subject-placement failure. "
          f"If CROSS collapses toward chance ({100.0 / n:.2f}) the shared map is not "
          f"transferring to the unseen subject at all.")


if __name__ == "__main__":
    main()
