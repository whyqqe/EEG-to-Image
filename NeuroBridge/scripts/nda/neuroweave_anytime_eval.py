#!/usr/bin/env python3
"""NeuroWeave anytime / progressive-decoding curve on exported encoders.

For each arm under --root that has enc/sub-XX/shared_r_*.npy OR a checkpoint,
re-encode the raw EEG with temporal masks at 150/350/700/1000 ms and score
200-way Top-1 against ViT-H-14 levels_mean.  This is the eval-only half of the
§5.4 choice: it asks whether progressive decoding has a useful delay-accuracy
tradeoff WITHOUT requiring a causal mask at train time.

Does NOT retrain anything.  Reads raw EEG from the same dataset the trainer used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT))
sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))

from module.dataset import EEGPreImageDataset  # noqa: E402
from ss_modules import SharedSpecificEncoder  # noqa: E402
from cfmsf_joint_train import DEFAULT_CHANNELS, build_target, l2n  # noqa: E402
from cfmsf_fix_train import test200_metrics  # noqa: E402
from neuroweave_s1_train import WIN, encode_r, mask_time, SUBJECT as _SUB  # noqa: E402

SUBJECT = 8


def load_encoder(arm_dir: Path, eeg_len: int, channels_num: int, n_extra: int, dev):
    """Rebuild (encoder, proj) from the arm's chosen checkpoint.

    THE PROJ HEAD IS NOT OPTIONAL.  Our exported `shared_r` is the RAW encoder
    output `r`, which is NOT in any CLIP space -- the route probe trains its own
    head on it.  The first version of this script scored `r` directly against the
    target bank, so every arm with a projection read ~cos 0.009 / top1 0.000 and
    only `direct` (which by construction aligns `r` itself) produced a real curve.
    That made the whole anytime table meaningless, so the head is now applied.
    """
    ck_path = arm_dir / "checkpoint_by_mini_top1.pth"
    if not ck_path.is_file():
        cands = sorted(arm_dir.glob("checkpoint_by_*.pth"))
        if not cands:
            return None, None, "no checkpoint"
        ck_path = cands[0]
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    model = SharedSpecificEncoder(
        subject_ids=[SUBJECT], feature_dim=1024, eeg_sample_points=eeg_len,
        channels_num=channels_num, n_extra_blocks=n_extra, use_adapter=True,
    ).to(dev)
    try:
        model.load_state_dict(ck["state_dict"], strict=True)
    except RuntimeError:
        from cfmsf_fix_train import inject_lora
        inject_lora(model, rank=8, alpha=16.0)
        model.load_state_dict(ck["state_dict"], strict=True)

    proj = None
    if ck.get("proj") is not None:
        from neuroweave_s1_train import MultiHead
        from cfmsf_joint_train import Proj
        if ck.get("multi_head"):
            proj = MultiHead(1024, 1024)
        else:
            proj = Proj(1024, 1024)
        proj.load_state_dict(ck["proj"])
        proj = proj.to(dev)
    return model, proj, str(ck_path)


@torch.no_grad()
def encode_q_masked(model, proj, eeg: np.ndarray, end, dev, bs: int = 512) -> np.ndarray:
    """Mask the input, run the encoder, then apply the arm's own head.

    `multi_head` arms are scored with the LATE head, because that is the head the
    arm's own `test200` number came from, so the curve stays in one space.
    """
    model.eval()
    if proj is not None:
        proj.eval()
    out = []
    for s in range(0, len(eeg), bs):
        z = torch.from_numpy(eeg[s:s + bs]).to(dev)
        if end is not None:
            z = mask_time(z, end)
        r = model(z, SUBJECT, return_parts=True)[2]
        if proj is None:
            q = r
        elif hasattr(proj, "late"):          # MultiHead
            q = proj.late(r)
        else:
            q = proj(r)
        out.append(q.cpu().numpy())
    return l2n(np.concatenate(out).astype(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--arms", type=str, default="")
    ap.add_argument("--test-subject", type=int, default=8)
    ap.add_argument("--eeg-dir", type=str,
                    default=str(NB_ROOT / "data/things_eeg/preprocessed_eeg"))
    ap.add_argument("--rn50-dir", type=str,
                    default=str(NB_ROOT / "data/things_eeg/image_feature/RN50"))
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--ridge", action="store_true",
                    help="also fit a tiny ridge on fit rows for each window "
                         "(uses leakfree fit); default is raw shared_r cosine")
    args = ap.parse_args()

    global SUBJECT
    SUBJECT = args.test_subject
    root = Path(args.root)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds_te = EEGPreImageDataset(
        [SUBJECT], args.eeg_dir, DEFAULT_CHANNELS, [0, 250],
        args.rn50_dir, "", False, [], True, False, None, False, False, False, False)
    eeg_len = int(ds_te.num_sample_points)
    channels_num = int(ds_te.channels_num)
    raw_te = np.stack([ds_te[i][0].numpy() for i in range(len(ds_te))]).astype(np.float32)
    _tgt_tr, tgt_te, _ = build_target("levels_mean")

    arms = ([a.strip() for a in args.arms.split(",") if a.strip()]
            if args.arms else sorted(p.name for p in root.iterdir()
                                     if p.is_dir() and (p / "arm_result.json").is_file()))

    report = {"subject": f"sub-{SUBJECT:02d}", "windows_ms":
              {k: int(v * 1000 / 250) for k, v in WIN.items()},
              "arms": {}}
    for arm in arms:
        arm_dir = root / arm
        model, proj, note = load_encoder(arm_dir, eeg_len, channels_num, 1, dev)
        if model is None:
            print(f"[skip] {arm}: {note}", flush=True)
            continue
        curve = {}
        for name, end in WIN.items():
            q = encode_q_masked(model, proj, raw_te, end if name != "full" else None, dev)
            m = test200_metrics(q, tgt_te)
            curve[name] = {"end_samples": end, "ms": int(end * 1000 / 250), **m}
            print(f"[{arm} {name:>5} @{end:3d}] top1={m['top1']:.4f} "
                  f"top5={m['top5']:.4f} cos={m['paired_cos']:.4f}", flush=True)
        # Control: the full-window number MUST reproduce the arm's own test200,
        # otherwise the curve is in a different space than the table.
        # raw_decode: a duplicate submission appended a stray "}" to two of these.
        ref = json.JSONDecoder().raw_decode(
            (arm_dir / "arm_result.json").read_text())[0].get("test200", {})
        full = curve["full"]["top1"]
        agree = abs(full - ref.get("top1", -1)) < 1e-9
        curve["_control_full_matches_test200"] = bool(agree)
        print(f"[{arm}] control full={full:.4f} vs test200={ref.get('top1')} -> "
              f"{'OK' if agree else 'MISMATCH'}", flush=True)
        report["arms"][arm] = {"ckpt": note, "has_proj": proj is not None, "curve": curve}

    out = root / "anytime_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[anytime] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
