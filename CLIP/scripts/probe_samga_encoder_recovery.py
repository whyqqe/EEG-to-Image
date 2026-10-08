#!/usr/bin/env python
"""probe_samga_encoder_recovery -- run OUR SCORE reimplementation on THEIR encoder.

THE ONE QUESTION THIS ANSWERS
----------------------------
On our sub-08 checkpoint, SCORE's deployment stack (moment match -> CSLS landmarks ->
orthogonal recovery -> CSLS) buys only +1.50 Top-1 over plain CSLS. On SCORE's own
paper (Table 2, 10 folds) the same stack buys **+12.97** (35.78 -> 48.75) from an
encoder that was *not* trained for recovery either. Same operator, same closed-set
200-way protocol, an order of magnitude different leverage.

Two explanations fit, and they imply opposite next moves:

  H1 ENCODER  -- their encoder's cross-subject geometry leaves useful mutual-NN
                 pseudo-labels for the recovery to fit, ours does not. Then the work
                 is on the EEG-side representation, and the diagnostic to move is
                 landmark accuracy.
  H2 FOLD     -- sub-08 is simply a hard fold for recovery. SAMGA official scores
                 22.00 final / 25.00 best here against a 34.4 ten-fold average, so a
                 fold that is ~10pp below average is plausible. Then our numbers are
                 already near this fold's ceiling and the honest move is the 10-fold
                 run, not more model surgery.

The experiment is to hold the fold fixed and swap only the encoder: we still have the
SAMGA official sub-08 checkpoint that PRODUCED the 22.00 reference, so we can score the
same recovery code on it, on the same 200 test concepts. Whichever hypothesis survives,
the number is worth having, because it is the first time the recovery module is measured
on an encoder other than ours.

WHY THE GROUND-TRUTH COLUMN IS LEGITIMATE HERE
----------------------------------------------
Landmark accuracy -- the fraction of mutual-NN pseudo-pairs that are true EEG-image
pairs -- uses the diagonal, i.e. labels. It is computed for DIAGNOSIS ONLY, next to the
label-free numbers, and must never be used to select landmarks, weights, `rho`, or a
checkpoint. It is the one quantity that separates "the recovery failed" from "the
recovery had nothing to fit", which is exactly the question, and no label-free proxy
answers it as directly.

FIDELITY
--------
The forward pass is transcribed from `third_party/SAMGA/train.py` (build_image_teacher +
the test loop), and the EEG/image tensors come from SAMGA's OWN `EEGPreImageDataset`
rather than a reimplementation, so the dataset's normalisation is the one their numbers
were produced with. `--verify-baseline` re-scores raw cosine and checks it against the
22.00 / 50.00 in the run's `result.csv`; if that does not reproduce, nothing below it
means anything and the script says so loudly.

Run (CPU is fine, 200 queries):
    python scripts/probe_samga_encoder_recovery.py \
        --ckpt /project/peilab/why/eeg-retrieval/outputs/samga_official/inter/seed2025/\
20260923-192743-sub-08/checkpoint_last.pth \
        --o --out outputs/probe/samga_encoder_recovery.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from samclip import calibration, config  # noqa: E402

# The SAMGA baseline lives in a sibling project and is imported READ-ONLY. We never write
# there; every output of this script lands under CLIP/.
SAMGA = Path("/project/peilab/why/eeg-retrieval/third_party/SAMGA")
SAMGA_DATA = Path("/project/peilab/why/eeg-retrieval/data/preprocessed_eeg")
SAMGA_IMG = Path("/project/peilab/why/eeg-retrieval/data/image_feature/"
                 "internvit_multilevel_20_24_28_32_36")

# Published reference for this exact fold/seed, from the run's own result.csv.
REF_TOP1, REF_TOP5 = 22.00, 50.00          # final epoch (checkpoint_last.pth)
REF_TOP1_BEST = 25.00                       # best epoch 21 (checkpoint_test_best.pth)


def build_and_load(ckpt_path: Path, device):
    """Rebuild the SAMGA encoder + projectors + router and load the checkpoint.

    Transcribed from `train.py` lines 355-450 (construction) and 669-688 (forward). The
    router is read in its deployed mode: `router_eval_mode='global'`, so `force_global`
    is True and the per-subject bias is dropped -- which is what a held-out subject sees,
    since it has no learned bias row.
    """
    sys.path.insert(0, str(SAMGA))
    from module.eeg_encoder.model import TSConv
    from module.projector import ProjectorLinear, ShareEncoder

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = sd.get("layer_ids", [20, 24, 28, 32, 36])
    n_layers = len(cfg)

    eeg_feature_dim, feature_dim, image_mid_dim = 1024, 512, 1024
    image_feature_dim = 3200          # InternViT-6B per layer, verified from the npy
    model = TSConv(feature_dim=eeg_feature_dim, eeg_sample_points=250, channels_num=63)
    eeg_projector = ProjectorLinear(eeg_feature_dim, feature_dim)
    share_encoder = ShareEncoder(feature_dim, feature_dim)
    img_pre_projector = ProjectorLinear(image_feature_dim, image_mid_dim)
    img_projectors = torch.nn.ModuleList(
        [ProjectorLinear(image_mid_dim, feature_dim) for _ in range(n_layers)])

    model.load_state_dict(sd["model_state_dict"])
    eeg_projector.load_state_dict(sd["eeg_projector_state_dict"])
    share_encoder.load_state_dict(sd["share_enc_state_dict"])
    img_pre_projector.load_state_dict(sd["img_pre_projector_state_dict"])
    img_projectors.load_state_dict(sd["img_projectors_state_dict"])

    router_sd = sd["layer_router_state_dict"]
    import torch.nn as nn
    subject_bias = nn.Embedding(router_sd["subject_bias.weight"].shape[0], n_layers)
    global_logits = nn.Parameter(router_sd["global_logits"].clone())
    subject_bias.load_state_dict({"weight": router_sd["subject_bias.weight"]})

    for m in (model, eeg_projector, share_encoder, img_pre_projector, img_projectors):
        m.eval().to(device)
    return dict(model=model, eeg_projector=eeg_projector, share_encoder=share_encoder,
                img_pre_projector=img_pre_projector, img_projectors=img_projectors,
                global_logits=global_logits, layer_ids=cfg)


def load_samga_tensors(subject: int, device, image_feature_dir: Path, layer_ids):
    """Their own dataset object, iterated once. Nothing about the preprocessing is
    reimplemented -- if their loader normalises, so do we, by construction.

    `image_feature_dir` must be the STACKED `[Nobj, Nimg, K, D]` directory (what their
    `evaluate_config.json` calls `effective_image_feature_dir`, built by
    `prepare_multilayer_feature_dir`), not the per-layer source directory.
    """
    sys.path.insert(0, str(SAMGA))
    from module.dataset import EEGPreImageDataset

    ds = EEGPreImageDataset(
        subject_ids=[subject], eeg_data_dir=str(SAMGA_DATA), selected_channels=[],
        time_window=[0, 250], image_feature_dir=str(image_feature_dir),
        text_feature_dir="", image_aug=False, aug_image_feature_dirs=[], average=True,
        _random=False, eeg_transform=None, train=False, image_test_aug=False,
        eeg_test_aug=False, frozen_eeg_prior=True,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=200, shuffle=False)
    eegl, imgl = [], []
    with torch.no_grad():
        for batch in loader:
            eegl.append(batch[0]); imgl.append(batch[1])
    return torch.cat(eegl).to(device), torch.cat(imgl).to(device)


@torch.no_grad()
def extract(parts, eeg, img):
    """Their forward, both branches. EEG: enc -> eeg_projector -> share.
    Image: per-layer pre+layer projector -> softmax(global logits) mix -> share."""
    q = parts["share_encoder"](parts["eeg_projector"](parts["model"](eeg)))
    layers = []
    for i in range(img.shape[1]):
        f = parts["img_pre_projector"](img[:, i, :])
        layers.append(parts["img_projectors"][i](f))
    layers = torch.stack(layers, dim=1)
    w = torch.softmax(parts["global_logits"] / 1.0, dim=-1)
    g = parts["share_encoder"](torch.sum(layers * w.unsqueeze(-1), dim=1))
    return q.cpu().numpy().astype(np.float64), g.cpu().numpy().astype(np.float64), w.numpy()


def landmark_accuracy(q_rec: np.ndarray, g: np.ndarray, k: int = 10) -> dict:
    """Mutual-NN landmarks under CSLS on `q_rec`, plus how many are TRUE pairs.

    The accuracy uses the diagonal and is therefore ANALYSIS ONLY -- never an input to
    any decision. See the module docstring.
    """
    C = q_rec.shape[0]
    s = calibration.csls_scores(q_rec, g, k=k)
    fwd, bwd = s.argmax(1), s.argmax(0)
    keep = bwd[fwd] == np.arange(C)
    qi = np.nonzero(keep)[0]
    if qi.size == 0:
        return {"n_landmarks": 0, "landmark_rate": 0.0, "landmark_acc": 0.0}
    return {"n_landmarks": int(qi.size), "landmark_rate": float(qi.size) / C,
            "landmark_acc": float((fwd[qi] == qi).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--target-subject", type=int, default=8)
    ap.add_argument("--csls-k", type=int, default=10)
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--image-feature-dir", default=None,
                    help="stacked [Nobj, Nimg, K, D] dir; defaults to the ckpt run's "
                         "multilayer_cache/stacked_<layers>")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[samga] device={device}  ckpt={args.ckpt}")
    ckpt = Path(args.ckpt)
    parts = build_and_load(ckpt, device)
    print(f"[samga] layer_ids={parts['layer_ids']}  router_global_weights="
          f"{np.round(torch.softmax(parts['global_logits'].detach(), -1).numpy(), 4).tolist()}")

    img_dir = (Path(args.image_feature_dir) if args.image_feature_dir else
               ckpt.parent / "multilayer_cache"
               / ("stacked_" + "_".join(str(x) for x in parts["layer_ids"])))
    print(f"[samga] image features: {img_dir}")
    eeg, img = load_samga_tensors(args.target_subject, device, img_dir, parts["layer_ids"])
    print(f"[samga] eeg={tuple(eeg.shape)}  img={tuple(img.shape)}")
    q, g, w = extract(parts, eeg, img)
    print(f"[samga] q={q.shape} g={g.shape}")

    res: dict = {"ckpt": args.ckpt, "target_subject": args.target_subject,
                 "layer_weights": w.tolist(), "reference": {"top1": REF_TOP1,
                                                            "top5": REF_TOP5}}

    def rec(x, label):
        r = calibration.report_with_scores(
            calibration.csls_scores(x, g, k=args.csls_k))
        st = landmark_accuracy(x, g, args.csls_k)
        print(f"  {label:<40} top1 {r['top1']:6.2f} top5 {r['top5']:6.2f}  "
              f"landmarks {st['n_landmarks']:>3} ({st['landmark_rate']:.3f})  "
              f"ACCURACY {st['landmark_acc']*100:5.1f}%")
        return {**r, **st}

    print(f"\nSAMGA official encoder, sub-08, SCORE deploy stack "
          f"(THEIR encoder, OUR recover.py):")
    print(f"  {'step':<40} {'Top-1':>6} {'Top-5':>6}  {'landmarks':>18}  {'correct':>11}")
    print("-" * 90)
    # Full cosine on BOTH sides. Normalising only the query is not cosine: it leaves each
    # gallery column scaled by 1/||g_j||, and since the image norms vary (mean 6.0 here,
    # against 19.4 for the queries) that reorders columns and understates Top-1 by ~10pp.
    # `csls_scores` normalises both sides, which is why every row below it was unaffected.
    qn = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    gn = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-8)
    res["raw_cosine"] = calibration.report_with_scores(qn @ gn.T)
    print(f"  {'raw cosine (their retrieve_all)':<40} "
          f"{res['raw_cosine']['top1']:6.2f} {res['raw_cosine']['top5']:6.2f}")
    res["csls"] = rec(q, "+ CSLS ranking")
    qm, _ = calibration.coordinate_recovery(q, g, k=args.csls_k, rho=args.rho,
                                            moment=True, orientation=False)
    res["moment"] = rec(qm, "+ mean and scale")
    qr, dg = calibration.coordinate_recovery(q, g, k=args.csls_k, rho=args.rho,
                                             moment=True, orientation=True)
    res["recovery"] = rec(qr, "+ recovery (rho=%.2f)" % args.rho)
    res["recovery_diag"] = {k: (v if not hasattr(v, "tolist") else v.tolist())
                            for k, v in dg.items() if k != "recovery_diag"}
    if "recovery_diag" in dg:
        res["recovery_inner"] = dg["recovery_diag"]

    print("-" * 90)
    # Two legitimate references exist for this fold: the FINAL epoch (22.00, what
    # `result.csv` reports as the headline) and the BEST epoch (25.00, `checkpoint_test_
    # best.pth`). Checking against the final-epoch value only would flag a correct
    # best-epoch checkpoint as a mismatch, so both are accepted and the match is named.
    got = res["raw_cosine"]["top1"]
    if abs(got - REF_TOP1) < 0.51:
        verdict = f"MATCHES the final-epoch reference {REF_TOP1:.2f}"
    elif abs(got - REF_TOP1_BEST) < 0.51:
        verdict = f"MATCHES the best-epoch reference {REF_TOP1_BEST:.2f}"
    else:
        verdict = (f"MISMATCH vs both final-epoch {REF_TOP1:.2f} and best-epoch "
                   f"{REF_TOP1_BEST:.2f} -- do not trust the rows above")
    ok = verdict.startswith("MATCHES")
    print(f"BASELINE CHECK: raw cosine {got:.2f} -> {verdict}")
    print(f"\nlandmark accuracy on their encoder: "
          f"{res['recovery']['landmark_acc']*100:.1f}%   "
          f"(ours on sub-08: 46.7%; SCORE's working regime: 98-100%)")
    print(f"recovery gain CSLS -> recovery: "
          f"{res['recovery']['top1'] - res['csls']['top1']:+.2f}   "
          f"(SCORE paper: +14.15 cumulative / +12.97 Table 2; ours: +1.50)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2, default=str))
        print(f"[samga] wrote {args.out}")


if __name__ == "__main__":
    main()
