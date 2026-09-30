#!/usr/bin/env python3
"""Multi-subject shared+specific pretrain + per-subject calibration (MindCross/MindBridge).

Phase A: train shared wide backbone + all subject paths jointly.
Phase B: freeze shared, calibrate target subject adapter/embedder (reset-tuning).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT))
sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))

from module.dataset import EEGPreImageDataset  # noqa: E402
from module.loss import ContrastiveLoss  # noqa: E402
from module.projector import ProjectorLinear  # noqa: E402
from module.util import retrieve_all  # noqa: E402
from ss_modules import (  # noqa: E402
    CHANNEL_SETS,
    POSTERIOR_17,
    SharedSpecificEncoder,
    channel_indices,
    diff_loss,
    dilate_first_linear,
    load_shared_from_eegproject,
    resolve_channels,
)

# Kept as an alias so nothing else that imports this module changes meaning: the
# historical 17-channel montage is still what `--channels posterior` resolves to.
DEFAULT_CHANNELS = POSTERIOR_17


def warm_start_with_dilation(
    model: SharedSpecificEncoder,
    ckpt: dict,
    channels: list[str],
    all_channels: list[str],
    n_samples: int,
) -> dict:
    """Copy a previous SharedSpecificEncoder run into `model`, widening its input.

    A montage change alters the width of the FLATTENED EEG input, so exactly two
    tensors change shape and everything else transfers verbatim:

      * `shared.model.0.*`      -- the wide backbone's input projection
      * `specific.<sid>.net.0.*` -- the per-subject embedder's input projection

    Both are widened with `dilate_first_linear`, which keeps the old per-channel
    weight blocks at their positions in the NEW montage and zeroes the rest.  The
    returned state is therefore EXACTLY the old model on the old channels, so a
    17->63 run starts from a model that has already learned everything the
    17-channel run learned, and any change in the result is attributable to the
    extra electrodes rather than to re-initialisation.
    """
    src = ckpt["model_state_dict"]
    ck_ch = int(ckpt.get("channels_num", 0) or 0)
    if not ck_ch:
        return {"loaded": 0, "dilated": [], "note": "checkpoint has no channels_num"}
    ck_names = ckpt.get("channel_names")
    if ck_names is None:
        # A checkpoint written before `channel_names` existed can only have been
        # the 17-channel montage, since nothing else was ever trained.
        ck_names = POSTERIOR_17 if ck_ch == len(POSTERIOR_17) else None
    if ck_names is None:
        raise ValueError(
            f"checkpoint records channels_num={ck_ch} but no channel_names, so its "
            f"montage cannot be mapped onto {len(channels) or 'all'} channels.  "
            f"Warm-starting across unknown montages would place weights in the "
            f"wrong columns and is refused."
        )
    idx = channel_indices(list(ck_names), all_channels)
    tgt = model.state_dict()
    dilated, loaded, skipped = [], 0, []
    for k, v in src.items():
        if k not in tgt:
            skipped.append(k)
            continue
        if tgt[k].shape == v.shape:
            tgt[k] = v
            loaded += 1
            continue
        # the only legitimate shape change is a widened flattened-EEG input
        if (k.endswith(".weight") and v.dim() == 2
                and v.shape[1] == ck_ch * n_samples
                and tgt[k].shape[1] == (len(channels) or len(all_channels)) * n_samples
                and k.rsplit(".", 1)[0] + ".bias" in src):
            base = k.rsplit(".", 1)[0]
            w, b = dilate_first_linear(
                v, src[base + ".bias"], idx,
                len(channels) or len(all_channels), n_samples,
            )
            tgt[k], tgt[base + ".bias"] = w, b
            dilated.append(k)
            loaded += 1
            continue
        skipped.append(f"{k} ({tuple(v.shape)} != {tuple(tgt[k].shape)})")
    model.load_state_dict(tgt, strict=True)
    return {"loaded": loaded, "dilated": dilated, "skipped": skipped[:10],
            "src_channels": ck_ch, "dst_channels": len(channels) or len(all_channels)}


def evaluate(model, eeg_proj, img_proj, loader, device, subject_filter=None):
    model.eval()
    eeg_proj.eval()
    img_proj.eval()
    eeg_list, img_list = [], []
    with torch.no_grad():
        for batch in loader:
            eeg, img, _txt, sid, *_ = batch
            if subject_filter is not None:
                mask = sid == subject_filter
                if not mask.any():
                    continue
                eeg, img, sid = eeg[mask], img[mask], sid[mask]
            eeg = eeg.to(device)
            img = img.to(device)
            sid = sid.to(device)
            raw = model(eeg, sid)
            z = eeg_proj(raw)
            zi = img_proj(img)
            eeg_list.append(z.cpu().numpy())
            img_list.append(zi.cpu().numpy())
    if not eeg_list:
        return 0.0, 0.0
    eeg_np = np.concatenate(eeg_list)
    img_np = np.concatenate(img_list)
    img_n = img_np / np.linalg.norm(img_np, axis=1, keepdims=True).clip(1e-8)
    top5, top1, total = retrieve_all(eeg_np, img_n, True)
    return 100.0 * top1 / total, 100.0 * top5 / total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb-root", type=str, default=str(NB_ROOT))
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--init-checkpoint", type=str, default="", help="EEGProject ckpt to warm-start shared")
    ap.add_argument("--init-ss-checkpoint", type=str, default="",
                    help=("warm-start the WHOLE SharedSpecificEncoder from a previous "
                          "run's checkpoint_ss_calib_best.pth. If that run used a "
                          "different --channels, the flattened-EEG input layers are "
                          "channel-dilated so the new model starts out exactly equal to "
                          "the old one on the old electrodes (see "
                          "warm_start_with_dilation). This is what makes "
                          "'17 -> 63 channels' a one-variable experiment."))
    ap.add_argument("--channels", type=str, default="posterior", choices=sorted(CHANNEL_SETS),
                    help=("electrode set. 'posterior' is the 17-channel montage every "
                          "historical result used, and stays the DEFAULT so old commands "
                          "reproduce bit-for-bit. 'all' keeps the full 63-channel montage, "
                          "which is what the published THINGS-EEG2 baselines use ('all "
                          "electrodes were preserved'). No run in this project has ever "
                          "compared the two, which is why this flag exists."))
    ap.add_argument("--warm-start-only", action="store_true",
                    help=("write the warm-started checkpoint and exit WITHOUT training. "
                          "Used to materialise the channel-dilated encoder as a real "
                          "artifact, so the actual exporter can be run on it and its "
                          "shared_r compared against the pre-dilation run. That turns "
                          "'the dilation is exact in-process' into 'the dilation is exact "
                          "through the whole downstream path'. Requires "
                          "--init-ss-checkpoint."))
    ap.add_argument("--train-subjects", type=str, default="1,2,4,5,6,7,8,9,10", help="skip weak sub-03 by default")
    ap.add_argument("--calib-subject", type=int, default=8)
    ap.add_argument("--pretrain-epochs", type=int, default=30)
    ap.add_argument("--calib-epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--calib-lr", type=float, default=3e-4)
    ap.add_argument("--lambda-diff", type=float, default=0.1, help="MindCross orthogonality weight")
    ap.add_argument("--feature-dim", type=int, default=512, help="SSP projector out dim")
    ap.add_argument("--n-extra-blocks", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    root = Path(args.nb_root)
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    subjects = [int(x) for x in args.train_subjects.split(",") if x.strip()]
    if args.calib_subject not in subjects:
        subjects.append(args.calib_subject)
    subjects = sorted(set(subjects))
    print(f"[INFO] subjects={subjects} calib={args.calib_subject}")

    eeg_dir = str(root / "data/things_eeg/preprocessed_eeg")
    rn50_dir = str(root / "data/things_eeg/image_feature/RN50")
    aug_dir = str(root / "data/things_eeg/image_feature/RN50/GaussianBlur-GaussianNoise-LowResolution-Mosaic")

    # The electrode set is now a variable, not a constant. `selected_channels=[]`
    # means "the whole montage", and the dataset tells us which montage it read,
    # so the names recorded in the checkpoint always describe the tensor layout
    # that was actually trained.
    selected_channels = resolve_channels(args.channels)
    import json as _json
    with open(Path(eeg_dir) / "info.json", encoding="utf-8") as fh:
        all_channels = list(_json.load(fh)["ch_names"])
    train_ds = EEGPreImageDataset(
        subjects, eeg_dir, selected_channels, [0, 250],
        rn50_dir, "", True, [aug_dir], True, True, None, True, True, False, True,
    )
    # Per-subject test loaders
    test_loaders = {}
    for sid in subjects:
        tds = EEGPreImageDataset(
            [sid], eeg_dir, selected_channels, [0, 250],
            rn50_dir, "", False, [], True, False, None, False, False, False, False,
        )
        test_loaders[sid] = DataLoader(tds, batch_size=200, shuffle=False)

    img_dim = int(train_ds.image_features.shape[-1])
    channels_num = int(train_ds.channels_num)
    eeg_len = int(train_ds.num_sample_points)
    # The names, in the exact order the dataset stacked them -- the flattened
    # layout the first Linear layer sees, and therefore the layout any later
    # warm start has to reproduce.
    channel_names = list(all_channels) if not selected_channels else list(selected_channels)
    assert len(channel_names) == channels_num, (
        f"channel bookkeeping mismatch: {len(channel_names)} names vs channels_num="
        f"{channels_num}. A wrong list here would silently mis-place weights in "
        f"every dilation, so refusing to continue."
    )
    print(f"[INFO] channels={args.channels} n={channels_num} "
          f"({channel_names[0]}..{channel_names[-1]}) samples={eeg_len}")

    model = SharedSpecificEncoder(
        subject_ids=subjects,
        feature_dim=img_dim,
        eeg_sample_points=eeg_len,
        channels_num=channels_num,
        n_extra_blocks=args.n_extra_blocks,
        use_adapter=True,
    ).to(device)
    eeg_projector = ProjectorLinear(img_dim, args.feature_dim).to(device)
    img_projector = ProjectorLinear(img_dim, args.feature_dim).to(device)

    if args.init_checkpoint:
        ckpt_path = Path(args.init_checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = root / ckpt_path
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        n = load_shared_from_eegproject(model, ckpt["model_state_dict"])
        print(f"[OK] warm-start shared from {ckpt_path.name}: {n} tensors")
        if ckpt.get("eeg_projector_state_dict"):
            try:
                eeg_projector.load_state_dict(ckpt["eeg_projector_state_dict"])
            except Exception as e:
                print(f"[WARN] eeg_projector load skipped: {e}")
        if ckpt.get("img_projector_state_dict"):
            try:
                img_projector.load_state_dict(ckpt["img_projector_state_dict"])
            except Exception as e:
                print(f"[WARN] img_projector load skipped: {e}")

    warm_report: dict = {}
    if args.init_ss_checkpoint:
        ss_path = Path(args.init_ss_checkpoint)
        if not ss_path.is_absolute():
            ss_path = root / ss_path
        if not ss_path.is_file():
            raise SystemExit(f"[FATAL] --init-ss-checkpoint not found: {ss_path}")
        ss_ckpt = torch.load(ss_path, map_location="cpu", weights_only=False)
        if "model_state_dict" not in ss_ckpt:
            raise SystemExit(
                f"[FATAL] {ss_path.name} is not a SharedSpecificEncoder checkpoint "
                f"(no model_state_dict). --init-checkpoint is for EEGProject ckpts; "
                f"this flag wants a checkpoint_ss_calib_best.pth."
            )
        prev = [int(s) for s in ss_ckpt.get("subjects", [])]
        if prev and prev != subjects:
            raise SystemExit(
                f"[FATAL] --init-ss-checkpoint was trained on subjects={prev} but this "
                f"run trains subjects={subjects}. Warm-starting across a different "
                f"subject list silently transfers another subject's embedder, so this "
                f"is refused."
            )
        warm_report = warm_start_with_dilation(
            model, ss_ckpt, selected_channels, all_channels, eeg_len
        )
        print(f"[OK] warm-start SharedSpecificEncoder from {ss_path.name}: "
              f"{warm_report['loaded']} tensors, dilated={warm_report['dilated']}, "
              f"{warm_report['src_channels']}ch -> {warm_report['dst_channels']}ch")
        if warm_report["skipped"]:
            raise SystemExit(
                f"[FATAL] {len(warm_report['skipped'])} tensors did not transfer and are "
                f"NOT an input-width change: {warm_report['skipped']}. A silent partial "
                f"load would make this arm unattributable."
            )
        for name, prj in (("eeg_projector", eeg_projector), ("img_projector", img_projector)):
            sd = ss_ckpt.get(f"{name}_state_dict")
            if sd:
                prj.load_state_dict(sd)

    if args.warm_start_only:
        if not args.init_ss_checkpoint:
            raise SystemExit(
                "[FATAL] --warm-start-only needs --init-ss-checkpoint; without it there "
                "is nothing to warm-start from and the artifact would be a fresh init "
                "mislabelled as a dilated one."
            )
        torch.save(
            {
                "phase": "warm-start-only",
                "epoch": 0,
                "model_state_dict": model.state_dict(),
                "eeg_projector_state_dict": eeg_projector.state_dict(),
                "img_projector_state_dict": img_projector.state_dict(),
                "subjects": subjects,
                "calib_subject": args.calib_subject,
                "img_dim": img_dim,
                "feature_dim": args.feature_dim,
                "eeg_sample_points": eeg_len,
                "channels_num": channels_num,
                "channel_set": args.channels,
                "channel_names": channel_names,
                "n_extra_blocks": args.n_extra_blocks,
                "calib_top1": None,
                "design": "channel-dilated warm start, no training",
                "warm_start": warm_report,
            },
            out / "checkpoint_ss_calib_best.pth",
        )
        report = {
            "pipeline": "nda_ss_pretrain_calib",
            "mode": "warm-start-only",
            "subjects": subjects,
            "calib_subject": args.calib_subject,
            "trained_epochs": 0,
            "channels": {
                "channel_set": args.channels,
                "channels_num": channels_num,
                "channel_names": channel_names,
                "warm_start": warm_report,
            },
            "checkpoint": str(out / "checkpoint_ss_calib_best.pth"),
        }
        (out / "ss_pretrain_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
        return

    # beta=1.0 → image-only contrastive (NeuroBridge default)
    criterion = ContrastiveLoss(0.07, 1.0, 1.0, True, True, False, False, True).to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    history = []

    # -------- Phase A: joint multi-subject pretrain --------
    params = list(model.parameters()) + list(eeg_projector.parameters()) + list(img_projector.parameters())
    opt = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    best_score, best_epoch = -1.0, 0

    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        eeg_projector.train()
        img_projector.train()
        ep_loss = 0.0
        for batch in tqdm(train_loader, desc=f"pretrain-{epoch}"):
            eeg, img, _txt, sid, *_ = batch
            eeg, img, sid = eeg.to(device), img.to(device), sid.to(device)
            opt.zero_grad()
            raw, s, r = model(eeg, sid, return_parts=True)
            z_e = eeg_projector(raw)
            z_i = img_projector(img)
            loss = criterion(z_e, z_i, torch.zeros_like(z_e))
            loss = loss + args.lambda_diff * diff_loss(s, r)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            ep_loss += float(loss.item())

        # eval on calib subject primarily
        t1, t5 = evaluate(model, eeg_projector, img_projector, test_loaders[args.calib_subject], device)
        row = {
            "phase": "pretrain",
            "epoch": epoch,
            "loss": ep_loss / len(train_loader),
            "calib_top1": t1,
            "calib_top5": t5,
        }
        # quick multi-subj mean top1 every 5 epochs
        if epoch % 5 == 0 or epoch == args.pretrain_epochs:
            tops = []
            for sid, ld in test_loaders.items():
                tops.append(evaluate(model, eeg_projector, img_projector, ld, device)[0])
            row["mean_top1"] = float(np.mean(tops))
            row["per_subj_top1"] = {str(s): float(t) for s, t in zip(subjects, tops)}
        history.append(row)
        print(f"[pretrain {epoch}] loss={row['loss']:.4f} sub{args.calib_subject}_top1={t1:.1f}%")
        score = t1
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save(
                {
                    "phase": "pretrain",
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "eeg_projector_state_dict": eeg_projector.state_dict(),
                    "img_projector_state_dict": img_projector.state_dict(),
                    "subjects": subjects,
                    "calib_subject": args.calib_subject,
                    "img_dim": img_dim,
                    "feature_dim": args.feature_dim,
                    "eeg_sample_points": eeg_len,
                    "channels_num": channels_num,
                    "n_extra_blocks": args.n_extra_blocks,
                    "design": "MindCross-shared+specific + MindBridge-adapter",
                },
                out / "checkpoint_ss_pretrain_best.pth",
            )

    # -------- Phase B: calibrate target subject --------
    ckpt = torch.load(out / "checkpoint_ss_pretrain_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    eeg_projector.load_state_dict(ckpt["eeg_projector_state_dict"])
    img_projector.load_state_dict(ckpt["img_projector_state_dict"])

    model.train_only_subject(args.calib_subject)
    # lightly tune projector for calib subject
    for p in eeg_projector.parameters():
        p.requires_grad = True
    for p in img_projector.parameters():
        p.requires_grad = False

    calib_ds = EEGPreImageDataset(
        [args.calib_subject], eeg_dir, selected_channels, [0, 250],
        rn50_dir, "", True, [aug_dir], True, False, None, True, True, False, True,
    )
    calib_loader = DataLoader(calib_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    calib_params = [p for p in model.parameters() if p.requires_grad] + list(eeg_projector.parameters())
    opt_c = optim.AdamW(calib_params, lr=args.calib_lr, weight_decay=1e-4)
    best_c, best_c_ep = -1.0, 0

    for epoch in range(1, args.calib_epochs + 1):
        model.train()
        eeg_projector.train()
        # keep shared frozen even if train() flips BN — no BN here
        model.freeze_shared()
        ep_loss = 0.0
        for batch in tqdm(calib_loader, desc=f"calib-{epoch}"):
            eeg, img, _txt, sid, *_ = batch
            eeg, img, sid = eeg.to(device), img.to(device), sid.to(device)
            opt_c.zero_grad()
            raw, s, r = model(eeg, sid, return_parts=True)
            z_e = eeg_projector(raw)
            z_i = img_projector(img)
            loss = criterion(z_e, z_i, torch.zeros_like(z_e))
            loss = loss + args.lambda_diff * diff_loss(s, r)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(calib_params, 1.0)
            opt_c.step()
            ep_loss += float(loss.item())

        t1, t5 = evaluate(model, eeg_projector, img_projector, test_loaders[args.calib_subject], device)
        row = {
            "phase": "calib",
            "epoch": epoch,
            "loss": ep_loss / max(len(calib_loader), 1),
            "calib_top1": t1,
            "calib_top5": t5,
        }
        history.append(row)
        print(f"[calib {epoch}] loss={row['loss']:.4f} top1={t1:.1f}% top5={t5:.1f}%")
        if t1 > best_c:
            best_c, best_c_ep = t1, epoch
            torch.save(
                {
                    "phase": "calib",
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "eeg_projector_state_dict": eeg_projector.state_dict(),
                    "img_projector_state_dict": img_projector.state_dict(),
                    "subjects": subjects,
                    "calib_subject": args.calib_subject,
                    "img_dim": img_dim,
                    "feature_dim": args.feature_dim,
                    "eeg_sample_points": eeg_len,
                    "channels_num": channels_num,
                    # Recorded so a later run can dilate EXACTLY: without the names,
                    # a warm start across montages cannot know which columns mean
                    # which electrode and is refused (see warm_start_with_dilation).
                    "channel_set": args.channels,
                    "channel_names": channel_names,
                    "n_extra_blocks": args.n_extra_blocks,
                    "calib_top1": t1,
                    "calib_top5": t5,
                    "design": "MindCross-shared+specific + MindBridge-calibrate",
                },
                out / "checkpoint_ss_calib_best.pth",
            )

    report = {
        "pipeline": "nda_ss_pretrain_calib",
        "subjects": subjects,
        "calib_subject": args.calib_subject,
        "pretrain_best_epoch": best_epoch,
        "pretrain_best_calib_top1": best_score,
        "calib_best_epoch": best_c_ep,
        "calib_best_top1": best_c,
        "lambda_diff": args.lambda_diff,
        "channels": {
            "channel_set": args.channels,
            "channels_num": channels_num,
            "channel_names": channel_names,
            "warm_start": warm_report,
        },
        "refs": ["MindCross", "MindBridge", "ShaSpec"],
        "checkpoint": str(out / "checkpoint_ss_calib_best.pth"),
    }
    (out / "ss_pretrain_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(out / "ss_history.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
