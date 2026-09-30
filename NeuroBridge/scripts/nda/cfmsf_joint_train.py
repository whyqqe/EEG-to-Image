#!/usr/bin/env python3
"""CF-MSF Stage 0: train the EEG ENCODER against the multi-level target (sub-08).

WHY THIS STAGE EXISTS
---------------------
Job 581602 measured, on the official 200-way test split, that a 2-layer MLP reading
out of a FROZEN encoder goes from 30.5% to 40.5% Top-1 purely by changing WHICH
target it aligns to (plain ViT-H-14 image -> concat of five blur/level views), and
that fusing thirteen such routes reaches 50.0% (CSLS) / 71.5% (+Sinkhorn) against
40.5% / 50.0% for the previous four-route system.

But that encoder was NEVER trained with that target: `outputs/ocf/intra_enc` was
trained with an image-only contrastive loss against RN50 features
(`nda_ss_pretrain.py` passes `rn50_dir`), and RN50 is the WEAKEST arm in the probe
(26.0% vs 40.5%).  So the current 40.5% is a read-out of a representation optimised
for something else.  The 86-91%-Top-1 systems do not do that: they train the encoder
and the multi-level target together.

This script does exactly that one change and nothing else: same split, same
gallery-NCE, same leak-free discipline.  The encoder is now in the gradient path.

ARMS (both in ONE job, so the comparison cannot drift)
-----------------------------------------------------
  joint   : encoder + projection trained together on the multi-level target
  frozen  : SAME loss, SAME data, SAME epochs, encoder weights frozen
            => this is the probe's setting, and the control that decides whether
               any gain is joint training or just the extra optimisation steps.

LEAK-FREE
---------
* Gradients only on `fit` rows (leakfree/split.json, 1489 concepts).
* Checkpoint selection on `val_a` -- the CANONICAL root-stage set (leakfree.py
  reserves valA for "NB encoder / root-stage selection", valB for downstream).
  val_b is reported as an independent check and never selects anything.
* The 200 test concepts are touched exactly once, after training.
* Selection uses 2-WAY identification accuracy, deliberately, not top-1: the
  probe measured Spearman(val_top1, test_top1) = 0.379, so top-1 on ~820 rows is
  a noise-dominated selector (it even cost 3.5 points when used to pick a fusion
  subset).  2-way contrasts the correct concept against distractors ON THE SAME
  ROW, cancelling per-row nuisance.  Both are reported so the claim is checkable.
* Every in-loop number is computed by running the CURRENT encoder on the CURRENT
  val rows.  Frozen pre-training features are never substituted for a measurement
  of a model that has since been updated (that would silently report the old
  encoder under the new arm's name).

OUTPUT
------
    <out>/<arm>/checkpoint_best.pth
    <out>/<arm>/enc/sub-08/shared_r_{train,test}.npy   <- drops straight into
                                                          cfmsf_route_probe --z-root
    <out>/joint_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT))
sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))

from module.dataset import EEGPreImageDataset  # noqa: E402
from ss_modules import SharedSpecificEncoder, diff_loss  # noqa: E402
import leakfree as LF  # noqa: E402

DEFAULT_CHANNELS = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]
VITH = "data/things_eeg/image_feature/ViT-H-14"
LEVELS = ["image", "GaussianBlur", "LowResolution", "Mosaic", "GaussianNoise"]
SUBJECT = 8          # rebound in main(); module-level so helpers can read it


def l2t(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def load_level(backbone: str, level: str, tag: str) -> np.ndarray:
    """(N, D) L2-normalised rows; `tag` is 'train' or 'test'.

    train.npy is (concepts, images, D), test.npy is (200, 1, D); both flatten to
    one row per (concept, image), which is the row order the EEG dataset uses.
    """
    p = (NB_ROOT / f"{backbone}/image_{tag}.npy" if level == "image"
         else NB_ROOT / f"{backbone}/{level}/{tag}.npy")
    a = np.load(p).astype(np.float32)
    return l2n(a.reshape(-1, a.shape[-1]))


def build_target(target: str) -> tuple[np.ndarray, np.ndarray, int]:
    """(train_rows, test_rows, dim) for the requested multi-level target."""
    def stack(tag: str, levels: list[str]) -> np.ndarray:
        mats = [load_level(VITH, lv, tag) for lv in levels]
        if target.endswith("_mean"):
            return l2n(np.mean(mats, axis=0))
        return np.concatenate(mats, 1)      # each block already unit-norm

    if target == "image":
        return load_level(VITH, "image", "train"), load_level(VITH, "image", "test"), 1024
    if target == "lowresolution":
        return (load_level(VITH, "LowResolution", "train"),
                load_level(VITH, "LowResolution", "test"), 1024)
    if target == "levels_mean":
        return stack("train", LEVELS), stack("test", LEVELS), 1024
    if target == "cat5":
        return stack("train", LEVELS), stack("test", LEVELS), 1024 * len(LEVELS)
    if target == "cat3":
        lv = ["image", "GaussianBlur", "LowResolution"]
        return stack("train", lv), stack("test", lv), 1024 * len(lv)
    raise SystemExit(f"[FATAL] unknown --target {target}")


class Proj(nn.Module):
    """EEG feature -> target space.

    Kept SEPARATE from the encoder so the exported `shared_r` keeps its own
    (1024-d) geometry regardless of the target dim: the CF-MSF routes are trained
    on `shared_r`, so the export contract must not move with the target.
    """

    def __init__(self, dim: int, out_dim: int, hidden: int = 1024, drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def two_way(q: np.ndarray, gal: np.ndarray, labels: np.ndarray, n_neg: int,
            seed: int) -> dict:
    """2-way identification: correct column vs `n_neg` distractors, same row.

    The low-variance selection statistic.  Top-1 over 1654 concepts at a ~2-5% hit
    rate is close to a per-row coin flip and therefore barely informative about the
    model; 2-way is 0.5 at chance for a random representation and only saturates as
    the representation becomes genuinely discriminative, so the same rows measure
    much more.  `labels` are column indices into `gal`.
    """
    sim = q @ gal.T
    rng = np.random.default_rng(seed)
    n_rows = len(q)
    neg = rng.integers(0, gal.shape[0], size=(n_rows, n_neg))
    correct = sim[np.arange(n_rows), labels][:, None]
    neg_s = np.take_along_axis(sim, neg, axis=1)
    wins = (correct > neg_s).sum() + 0.5 * (correct == neg_s).sum()
    # mean rank is reported alongside because it is bounded, continuous and uses
    # the whole ranking rather than the argmax, so it is the second low-variance
    # read on the same rows (agreement between the two is the stability check).
    rank = (np.argsort(np.argsort(-sim, axis=1), axis=1)[np.arange(n_rows), labels] + 1)
    return {"two_way": float(wins.sum() / (n_rows * n_neg)),
            "paired_cos": float(correct.mean()),
            "top1": float((sim.argmax(1) == labels).mean()),
            "mean_rank": float(rank.mean())}


@torch.no_grad()
def encode(model, proj, eeg: np.ndarray, dev, bs: int = 512) -> np.ndarray:
    model.eval()
    proj.eval()
    out = []
    for s in range(0, len(eeg), bs):
        z = torch.from_numpy(eeg[s:s + bs]).to(dev)
        out.append(proj(model(z, SUBJECT, return_parts=True)[2]).cpu().numpy())
    return l2n(np.concatenate(out).astype(np.float32))


def eval_rows(model, proj, eeg_all: np.ndarray, rows: np.ndarray, gal: np.ndarray,
              labels_all: np.ndarray, dev, n_neg: int, seed: int) -> dict:
    """Encode ONLY `rows` (the encoder is the thing being measured, so the numbers
    must come from the current weights, not from a cached feature bank)."""
    return two_way(encode(model, proj, eeg_all[rows], dev), gal, labels_all[rows],
                   n_neg, seed)


def train_arm(arm: str, args, dev, data: dict) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tgt_tr, tgt_te, tgt_dim = data["tgt_tr"], data["tgt_te"], data["tgt_dim"]
    cid, fit_rows, val_a, val_b = data["cid"], data["fit"], data["val_a"], data["val_b"]

    # concept gallery = mean multi-level target per TRAIN concept (denoises the 10
    # images/concept); test concepts are not in this gallery.
    gal = np.zeros((data["n_cls"], tgt_dim), dtype=np.float64)
    np.add.at(gal, cid, tgt_tr.astype(np.float64))
    cnt = np.bincount(cid, minlength=data["n_cls"])
    gal /= np.clip(cnt, 1, None)[:, None]
    gal = l2n(gal.astype(np.float32))
    gal_t = torch.from_numpy(gal).to(dev)

    model = SharedSpecificEncoder(
        subject_ids=[SUBJECT], feature_dim=1024, eeg_sample_points=args.eeg_len,
        channels_num=args.channels_num, n_extra_blocks=args.n_extra_blocks,
        use_adapter=True,
    ).to(dev)
    proj = Proj(1024, tgt_dim, hidden=args.proj_hidden, drop=args.drop).to(dev)

    init_note = "scratch"
    if not args.scratch:
        ck = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if [int(s) for s in ck["subjects"]] != [SUBJECT]:
            raise SystemExit(
                f"[FATAL] init checkpoint {Path(args.init_checkpoint).name} declares "
                f"subjects={[int(s) for s in ck['subjects']]} but this run is for "
                f"sub-{SUBJECT:02d}. Training from another subject's encoder would be a "
                f"cross-subject experiment wearing an intra-subject label, so it is "
                f"refused rather than warned about.")
        model.load_state_dict(ck["model_state_dict"])
        init_note = f"{Path(args.init_checkpoint).name}:{ck.get('phase')}@ep{ck.get('epoch')}"

    if arm == "frozen":
        for p in model.parameters():
            p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad] + list(proj.parameters())
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    loader = DataLoader(Subset(data["ds"], fit_rows.tolist()),
                        batch_size=args.batch_size, shuffle=True, drop_last=True)
    # The target bank stays CPU-RESIDENT and is indexed on the CPU, then moved.
    # Indexing a CPU tensor with a GPU index raises at run time, and a CPU-only
    # smoke test cannot catch that, so the invariant is made explicit here: gather
    # on CPU, transfer per batch.  At 1024-d the transfer is 512*1024*4 = 2 MB per
    # step, i.e. free next to the encoder's own activations, and it removes the
    # whole class of device bugs from this path.
    tgt_all_t = torch.from_numpy(tgt_tr)

    best = {"two_way": -1.0, "epoch": -1, "val_a": None, "val_b": None}
    hist: list[dict] = []
    ck_path = Path(args.out) / arm / "checkpoint_best.pth"
    ck_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(args.epochs):
        model.train()
        proj.train()
        if arm == "frozen":
            model.eval()          # dropout off in the frozen branch's encoder
        tot, n = 0.0, 0
        for eeg, _img, _txt, sid, obj, _im, _rep in loader:
            eeg, sid, obj = eeg.to(dev), sid.to(dev), obj.to(dev)
            opt.zero_grad(set_to_none=True)
            out, s, r = model(eeg, sid, return_parts=True)
            q = proj(r)
            loss = F.cross_entropy((l2t(q) @ gal_t.T) / args.tau, obj)
            if args.inst_weight > 0:
                tgt = tgt_all_t[obj.cpu()].to(dev, non_blocking=True)
                loss = loss + args.inst_weight * (1.0 - (l2t(q) * l2t(tgt)).sum(-1)).mean()
            if arm == "joint" and args.lambda_diff > 0:
                loss = loss + args.lambda_diff * diff_loss(s, r)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.clip)
            opt.step()
            tot += float(loss.detach())
            n += 1
        sch.step()

        m_a = eval_rows(model, proj, data["raw_tr"], val_a, gal, cid, dev,
                        args.n_neg, args.seed)
        m_b = eval_rows(model, proj, data["raw_tr"], val_b, gal, cid, dev,
                        args.n_neg, args.seed + 1)
        hist.append({"epoch": ep, "loss": tot / max(n, 1),
                     **{f"val_a_{k}": v for k, v in m_a.items()},
                     **{f"val_b_{k}": v for k, v in m_b.items()}})
        if m_a["two_way"] > best["two_way"]:
            best = {"two_way": m_a["two_way"], "epoch": ep, "val_a": m_a, "val_b": m_b}
            torch.save({"state_dict": model.state_dict(), "proj": proj.state_dict(),
                        "epoch": ep, "arm": arm, "target": args.target,
                        "val_a": m_a, "val_b": m_b, "init": init_note,
                        "subjects": [SUBJECT], "feature_dim": 1024,
                        "eeg_sample_points": args.eeg_len,
                        "channels_num": args.channels_num,
                        "n_extra_blocks": args.n_extra_blocks}, ck_path)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"[{arm} ep{ep:03d}] loss={hist[-1]['loss']:.3f} "
                  f"valA 2way={m_a['two_way']:.4f} cos={m_a['paired_cos']:.4f} "
                  f"top1={m_a['top1']:.4f} | valB 2way={m_b['two_way']:.4f} "
                  f"top1={m_b['top1']:.4f}", flush=True)

    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    proj.load_state_dict(ck["proj"])

    # ---- honest final row: the 200 test concepts, bank col j == query row j ----
    q_te = encode(model, proj, data["raw_te"], dev)
    sim = q_te @ tgt_te.T
    order = np.argsort(-sim, 1)
    test_row = {
        **two_way(q_te, tgt_te, np.arange(len(tgt_te)), args.n_neg, args.seed + 2),
        "test200_top1": float((order[:, 0] == np.arange(len(order))).mean()),
        "test200_top5": float(np.mean([i in order[i, :5] for i in range(len(order))])),
        "test200_mean_rank": float(np.mean([np.where(order[i] == i)[0][0] + 1
                                            for i in range(len(order))])),
    }

    # ---- export `shared_r` so cfmsf_route_probe can re-fit its own heads on it ----
    d = Path(args.out) / arm / "enc" / f"sub-{SUBJECT:02d}"
    d.mkdir(parents=True, exist_ok=True)
    for tag, eeg_all in (("train", data["raw_tr"]), ("test", data["raw_te"])):
        with torch.no_grad():
            parts = []
            for s in range(0, len(eeg_all), 512):
                z = torch.from_numpy(eeg_all[s:s + 512]).to(dev)
                parts.append(model(z, SUBJECT, return_parts=True)[2].cpu().numpy())
        np.save(d / f"shared_r_{tag}.npy", np.concatenate(parts).astype(np.float32))

    return {"arm": arm, "target": args.target, "init": init_note,
            "freeze_encoder": arm == "frozen", "best": best,
            "history_tail": hist[-6:], **test_row, "ckpt": str(ck_path),
            "enc_export": str(d)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--arms", type=str, default="joint,frozen")
    ap.add_argument("--target", type=str, default="levels_mean",
                    choices=["image", "lowresolution", "levels_mean", "cat3", "cat5"])
    ap.add_argument("--test-subject", type=int, default=8)
    ap.add_argument("--init-checkpoint", type=str, default="",
                    help=("warm-start encoder. Default resolves to "
                          "outputs/ocf/intra_enc/sub-<test-subject>/checkpoint_ss_calib_best.pth "
                          "-- i.e. the SAME subject's own intra encoder. It must not be "
                          "a fixed subject: a hardcoded default silently warm-started "
                          "every subject from sub-08's encoder, which the subject guard "
                          "below now rejects instead of training nine subjects from the "
                          "wrong initialisation."))
    ap.add_argument("--scratch", action="store_true", help="no warm start (control)")
    ap.add_argument("--split-json", type=str,
                    default=str(NB_ROOT / "outputs/leakfree/split.json"))
    ap.add_argument("--eeg-dir", type=str,
                    default=str(NB_ROOT / "data/things_eeg/preprocessed_eeg"))
    ap.add_argument("--rn50-dir", type=str,
                    default=str(NB_ROOT / "data/things_eeg/image_feature/RN50"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--drop", type=float, default=0.15)
    ap.add_argument("--proj-hidden", type=int, default=1024)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--inst-weight", type=float, default=0.2)
    ap.add_argument("--lambda-diff", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--n-neg", type=int, default=64)
    ap.add_argument("--n-extra-blocks", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    global SUBJECT
    SUBJECT = args.test_subject
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Resolve the warm start from the SUBJECT, not from a fixed path.  The guard in
    # train_arm() cross-checks the checkpoint's own `subjects` field, so a mismatch is
    # a hard error rather than a silently mis-initialised run.
    if args.scratch:
        args.init_checkpoint = ""
    elif not args.init_checkpoint:
        cand = (NB_ROOT / f"outputs/ocf/intra_enc/sub-{SUBJECT:02d}"
                / "checkpoint_ss_calib_best.pth")
        if not cand.is_file():
            raise SystemExit(f"[FATAL] no per-subject init encoder at {cand}; "
                             f"pass --init-checkpoint or --scratch")
        args.init_checkpoint = str(cand)
        print(f"[joint] init encoder for sub-{SUBJECT:02d}: {cand.name}", flush=True)
    elif f"sub-{SUBJECT:02d}" not in args.init_checkpoint:
        # An explicitly passed path that does not even mention this subject is almost
        # certainly a copy-paste error; make the caller confirm rather than guess.
        raise SystemExit(
            f"[FATAL] --init-checkpoint {args.init_checkpoint} does not reference "
            f"sub-{SUBJECT:02d}. If this is intentional (e.g. a cross-subject "
            f"transfer study) the subject guard in train_arm() will still reject a "
            f"checkpoint whose own `subjects` field disagrees.")

    ds_tr = EEGPreImageDataset([SUBJECT], args.eeg_dir, DEFAULT_CHANNELS, [0, 250],
                               args.rn50_dir, "", False, [], True, False, None, True,
                               False, False, False)
    ds_te = EEGPreImageDataset([SUBJECT], args.eeg_dir, DEFAULT_CHANNELS, [0, 250],
                               args.rn50_dir, "", False, [], True, False, None, False,
                               False, False, False)
    args.eeg_len = int(ds_tr.num_sample_points)
    args.channels_num = int(ds_tr.channels_num)

    n_tr = len(ds_tr)
    raw_tr = np.stack([ds_tr[i][0].numpy() for i in range(n_tr)]).astype(np.float32)
    raw_te = np.stack([ds_te[i][0].numpy() for i in range(len(ds_te))]).astype(np.float32)
    cid = np.array([ds_tr[i][4] for i in range(n_tr)], dtype=np.int64)
    n_cls = int(cid.max()) + 1

    tgt_tr, tgt_te, tgt_dim = build_target(args.target)

    split = LF.load(args.split_json)
    fit_rows = LF.rows_for(split, "fit", n_tr)
    val_a = LF.rows_for(split, "val_a", n_tr)
    val_b = LF.rows_for(split, "val_b", n_tr)
    if set(fit_rows.tolist()) & (set(val_a.tolist()) | set(val_b.tolist())):
        raise SystemExit("[FATAL] fit/val overlap")

    print(f"[joint] sub-{SUBJECT:02d} raw={raw_tr.shape} concepts={n_cls} "
          f"instances={raw_te.shape} target={args.target} dim={tgt_dim} "
          f"fit={len(fit_rows)} valA={len(val_a)} valB={len(val_b)}", flush=True)

    data = {"ds": ds_tr, "raw_tr": raw_tr, "raw_te": raw_te, "cid": cid, "n_cls": n_cls,
            "fit": fit_rows, "val_a": val_a, "val_b": val_b,
            "tgt_tr": tgt_tr, "tgt_te": tgt_te, "tgt_dim": tgt_dim}

    report: dict = {
        "subject": f"sub-{SUBJECT:02d}", "target": args.target, "target_dim": tgt_dim,
        "params": {k: getattr(args, k) for k in
                   ("epochs", "lr", "tau", "inst_weight", "lambda_diff", "batch_size",
                    "weight_decay", "drop", "proj_hidden", "seed", "scratch")},
        "selection": {"set": "val_a", "statistic": "two_way",
                      "why": ("val_top1 is noise-dominated here: Spearman(val_top1, "
                              "test_top1)=0.379 measured in job 581602; 2-way cancels "
                              "per-row nuisance and is stable at ~830 rows")},
        "leakfree": {"fit_concepts": int(split["counts"]["fit"]),
                     "val_a_concepts": int(split["counts"]["val_a"]),
                     "val_b_concepts": int(split["counts"]["val_b"]),
                     "test_concepts": 200, "note": "test read once, after training"},
        "arms": {},
    }
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if arm not in ("joint", "frozen"):
            raise SystemExit(f"[FATAL] unknown arm {arm}")
        report["arms"][arm] = train_arm(arm, args, dev, data)

    report["reference"] = {
        "probe_581602": {"single_route_best": 0.405, "fuse13_csls": 0.50,
                         "fuse13_csls_sinkhorn": 0.715,
                         "server": "cat5 target on FROZEN encoder"},
        "cfmsf_581546": {"csls": 0.40, "csls_sinkhorn": 0.50},
        "NOTE": ("same 200 test concepts; this stage moves the ENCODER only, so any "
                 "change here is attributable to joint training"),
    }
    (out / "joint_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[joint] wrote {out}")


if __name__ == "__main__":
    main()
