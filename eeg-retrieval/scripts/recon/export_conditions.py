#!/usr/bin/env python3
"""Emit the (200, 1024) conditioning array the reconstruction stack consumes.

WHAT DOWNSTREAM EXPECTS
-----------------------
The generator and every metric script in the sibling reconstruction project read a single
`(N, 1024)` L2-normalised OpenCLIP ViT-H-14 array and nothing else -- they never touch EEG.
That decoupling is what lets SAMGA be swapped in by producing one file, and it also means
the file's *row order* is the entire interface contract. Row i must be the condition for
the i-th test stimulus in `image_metadata.npy`'s canonical test order, because the ground
truth images, the neighbour gallery and the 2-way identification pool are all indexed that
way. A permuted array would still generate 200 plausible images and score plausibly while
being wrong, so the order is asserted below rather than assumed.

ARMS
----
    --mode head      SAMGA encoder -> trained generation head -> CLIP-1024.  The method.
    --mode identity  centred, L2-normalised encoder output, no head.          Control.

The identity arm exists to answer "did the head do anything?" with the same downstream
pipeline rather than with a loss number. If `head` and `identity` produce the same
reconstruction metrics, the head is a no-op and the honest report says so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "recon"))

from samga_recon import GenHead  # noqa: E402

N_TEST_STIMULI = 200


def load_head(path: Path, device: torch.device) -> GenHead:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = blob.get("config", {})
    head = GenHead(dim_in=cfg.get("dim_in", 1024), dim_out=cfg.get("dim_out", 1024),
                   hidden=cfg.get("hidden", 2048), dropout=cfg.get("dropout", 0.1),
                   residual=cfg.get("residual", True),
                   standardize=cfg.get("standardize", True))
    head.load_state_dict(blob["gen_head"], strict=True)
    head.to(device).eval()
    return head


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-feats", type=Path, required=True,
                    help="npz from export_eeg_feats.py --split test")
    ap.add_argument("--gen-head", type=Path, default=None, help="required for --mode head")
    ap.add_argument("--mode", default="head", choices=["head", "identity"])
    ap.add_argument("--clip-test", type=Path, default=None,
                    help="optional (200,1,1024) or (200,1024) target, for a paired report")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--subject", default="")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        # Do not fall through to a CUDA tensor that will only fail at the first operation
        # with "Found no NVIDIA driver". Fail here, where the message can say what to do.
        raise SystemExit(
            "[FATAL] no CUDA device visible. Run on a GPU node, or pass --allow-cpu to "
            "run on CPU deliberately (fine for the 200-row test split, slow for export).")
    feats = np.load(args.test_feats if args.test_feats.is_absolute() else REPO / args.test_feats)
    x = torch.from_numpy(feats["hidden"].astype(np.float32))
    oi = feats["object_idx"]

    if x.shape[0] != N_TEST_STIMULI:
        raise SystemExit(f"[FATAL] expected {N_TEST_STIMULI} test rows, got {x.shape[0]}")
    # The order assertion. `test.npy` is [200 concepts, 1 image, 80 reps, 63, 250], so an
    # averaged pass must yield object_idx = 0,1,...,199 in that exact sequence. Anything
    # else means the array would be silently permuted relative to the ground truth.
    if not np.array_equal(oi, np.arange(N_TEST_STIMULI)):
        first_bad = int(np.argmax(oi != np.arange(N_TEST_STIMULI)))
        raise SystemExit(
            f"[FATAL] test rows are not in canonical stimulus order: object_idx[0:8]={oi[:8]} "
            f"first mismatch at row {first_bad} (got {int(oi[first_bad])}, expected "
            f"{first_bad}). The conditions would be attributed to the wrong ground-truth "
            f"images."
        )

    with torch.no_grad():
        if args.mode == "head":
            if args.gen_head is None:
                raise SystemExit("[FATAL] --mode head requires --gen-head")
            head = load_head(args.gen_head if args.gen_head.is_absolute() else REPO / args.gen_head,
                             device)
            cond = head(x.to(device)).float().cpu().numpy()
            source = f"gen_head:{args.gen_head}"
        else:
            cond = F.normalize(x.to(device).float() - x.to(device).float().mean(0, keepdim=True),
                               dim=-1).float().cpu().numpy()
            source = "identity_centered_normalized"

    cond = cond.astype(np.float32)
    cond /= np.maximum(np.linalg.norm(cond, axis=1, keepdims=True), 1e-8)

    report = {"subject": args.subject, "mode": args.mode, "source": source,
              "shape": list(cond.shape), "row_order": "canonical image_metadata test order",
              "l2_normalized": True,
              "norm_mean": float(np.linalg.norm(cond, axis=1).mean()),
              "test_feats": str(args.test_feats)}

    if args.clip_test is not None:
        cp = args.clip_test if args.clip_test.is_absolute() else REPO / args.clip_test
        if cp.is_file():
            gt = np.load(cp).reshape(N_TEST_STIMULI, -1).astype(np.float32)
            gt /= np.maximum(np.linalg.norm(gt, axis=1, keepdims=True), 1e-8)
            paired = (cond * gt).sum(1)
            # Concept-level 200-way Top-1/5 on the conditioning array itself. This is the
            # upper bound on what the decoder can be given, before any pixels exist, and
            # it should track the retrieval numbers from the training log.
            sim = cond @ gt.T
            order = np.argsort(-sim, axis=1)
            report["paired_cos"] = float(paired.mean())
            report["cond_top1"] = float((order[:, 0] == np.arange(N_TEST_STIMULI)).mean())
            report["cond_top5"] = float(
                (order[:, :5] == np.arange(N_TEST_STIMULI)[:, None]).any(1).mean())
            print(f"[DIAG] paired_cos={report['paired_cos']:.4f} "
                  f"cond_top1={report['cond_top1']:.4f} cond_top5={report['cond_top5']:.4f}")

    dest = args.out if args.out.is_absolute() else REPO / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.save(dest, cond)
    dest.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] wrote {dest} shape={cond.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
