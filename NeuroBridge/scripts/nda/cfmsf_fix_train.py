#!/usr/bin/env python3
"""CF-MSF FIX ROUND: why did joint training LOSE, and what actually fixes it?

THE MEASURED FAILURE (sub-08, job 581652, logs/cfmsf_all/sub-08/logs/joint_train.log)
-------------------------------------------------------------------------------
    arm      train loss        valA 2-way        valB 2-way      paired cos
    joint    6.85 -> 2.05      0.866 -> 0.849    0.870 -> 0.844  0.294 -> 0.168
    frozen   6.85 -> 4.19      0.888 -> 0.934    0.887 -> 0.931  0.273 -> 0.214

`joint` crushed the TRAIN loss by 70% while its VALIDATION numbers fell, i.e. the
encoder memorised the 1489 fit concepts (14890 rows) and lost generalisation.  This
note names three separate defects and gives each its own arm, because they have
different fixes and lumping them together would leave the cause unknown.

DEFECT 1 -- OBJECTIVE MISMATCH (the train loss falling is not evidence of progress)
    `joint` optimises `proj(r)` and then EXPORTS `r`.  The thing being optimised and
    the thing being measured were never the same object, so a falling train loss was
    never evidence of anything about the export.  That is also why the collapse looks
    surprising: it was invisible to the objective.
    ARM `direct`: no projection head at all -- align `r` ITSELF (1024-d) to the
    1024-d multi-level target.  Then the exported feature IS the optimised feature.
    (`--target` must be 1024-d; the script asserts this rather than truncating.)

DEFECT 2 -- THE ENCODER IS TOO FREE FOR THE DATA (the actual overfitting)
    ~0.7M trainable parameters chasing a 1489-concept contrastive objective.
    TWO fixes, tested separately because they fail differently:
    ARM `disc`  : keep full capacity but make the encoder learn 30x slower
                  (encoder 1e-5 vs head 3e-4) -- classic discriminative fine-tuning.
    ARM `lora`  : freeze the encoder and adapt it through rank-8 low-rank updates
                  (B initialised to zero, so the arm starts as EXACTLY `frozen` and
                  can only move as far as the evidence justifies).  This is the
                  structural fix: it makes the hypothesis space small instead of
                  hoping a small LR keeps it there.

DEFECT 3 -- THE SELECTOR HAS NO RESOLUTION  (a measurement defect, not a model one)
    `frozen`'s 2-way moved only 0.916 -> 0.934 across 40 epochs (range 0.018), and
    2-way and top-1 DISAGREED about the best epoch (2-way said ep5, top-1 peaked at
    ep20).  Selecting the checkpoint is therefore partly arbitrary, which contaminates
    every arm comparison.
    FIRST ATTEMPT AT THIS FIX, AND WHY IT WAS WRONG (job 581657): `margin` =
    mean(cos to the correct concept - cos to the best WRONG concept) was introduced
    here as the continuous replacement.  It is ANTI-correlated with generalisation in
    this regime: it fell monotonically over training (frozen -0.0949 @ep0 -> -0.1144
    @ep39) while 2-way rose.  Maximising it therefore selected EPOCH 0 -- the untrained
    head -- for `frozen`, `disc` and `joint`, so all of their headline test numbers
    were computed on an essentially untrained projection and cannot be compared with
    the probe's 0.380-0.405 band.  A selector bug silently became a results bug, with
    nothing in the outputs to reveal it.
    REPLACEMENT: `mini_top1` -- retrieval Top-1 on the held-out CONCEPT set against
    concept-mean columns (val_a: 83 concepts/830 rows; val_b: 82/820), i.e. a small
    replica of the 200-way task the paper reports.  `two_way` is near-saturated,
    full-gallery top-1 is a 2-5% hit rate, and `margin` is retained only as a recorded
    column.  Additionally EVERY selector now keeps its own checkpoint and the round
    reports an (arm x selector) test table, so the selector's influence is bounded and
    visible rather than hidden behind a single number.

CONTROL
    ARM `frozen`: required again.  It is the probe's setting and the reference the fix
    arms have to beat; keeping it in the same job means the comparison cannot drift.

ISOLATION, STATED UP FRONT
    `joint` vs `direct` isolates the objective mismatch (both fully trainable).
    `frozen` vs `lora`/`disc` isolates the capacity fix (all three export `r`).
    `lora` vs `disc` asks whether the fix has to be structural or can be a learning rate.

LEAK-FREE: same discipline as the parent script.  Gradients on `fit` only; checkpoint
selection on `val_a` only; `val_b` reported as an independent check and never selects;
the 200 test concepts are read once per selector, after training, purely to build the
diagnostic table -- the selector itself is fixed before any test number exists.

RESUMABLE: each arm writes `<out>/<arm>/arm_result.json` plus one checkpoint per
selector.  An arm with a result file is skipped; an arm with only checkpoints is
RE-SCORED without retraining.  Both paths matter here: job 581657 crashed in its fifth
arm after finishing four, so the 50 minutes already spent are recoverable without
re-running the training or altering any number that was already produced.

NOTE ON SCOPE: sub-08 only, on purpose.  The question here is "which fix works", and
answering it on one subject first costs minutes instead of hours; the winner then goes
into the 10-subject chain that is already running.

This file adds NO changes to any script the running 10-subject job invokes: it imports
its helpers read-only and writes to its own output tree.
"""
from __future__ import annotations

import argparse
import json
import math
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
# read-only imports of the parent stage's pure helpers (no side effects at import)
from cfmsf_joint_train import (DEFAULT_CHANNELS, Proj, build_target,  # noqa: E402
                               l2n, l2t)

SUBJECT = 8          # rebound in main()


# --------------------------------------------------------------------------- LoRA
class LoRALinear(nn.Module):
    """Frozen base weight + trainable rank-`rank` update, `B` initialised to ZERO.

    The zero-init is what makes `lora` worth running as a controlled comparison: at
    step 0 the module is numerically identical to `frozen`, so any difference after
    training is attributable to the adaptation and not to a changed starting point.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        # The device/dtype MUST be inherited from the wrapped weight.
        # Job 581657 died on the line below in its original form
        # (`torch.empty(rank, in_features)` with no device): this class is built
        # AFTER the encoder has been moved to the GPU and is then grafted into it by
        # `inject_lora`, and `setattr` does not move a submodule's parameters.  The
        # new tensors therefore stayed on the CPU while `self.base` was on cuda:0,
        # and the first forward pass raised
        #     RuntimeError: Expected all tensors to be on the same device, ...
        # Deriving from `base.weight` makes the class device-correct by construction
        # for any device, including the `meta` device the smoke test uses.
        self.A = nn.Parameter(torch.empty(rank, base.in_features,
                                          device=base.weight.device,
                                          dtype=base.weight.dtype))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank,
                                         device=base.weight.device,
                                         dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / max(rank, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * F.linear(F.linear(x, self.A), self.B)


def inject_lora(module: nn.Module, rank: int, alpha: float) -> int:
    """Replace every nn.Linear in `module` (recursively) with a LoRALinear.

    `setattr` here is a deliberate, audited graft: it does NOT move the new module's
    tensors, so `LoRALinear.__init__` is responsible for creating A/B on the wrapped
    weight's device/dtype.  The device audit flags every graft by default; this one is
    marked rather than silenced, and the invariant is independently enforced by the
    `meta`-device propagation test run in the submit script's smoke stage.
    """
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank, alpha))  # device-audit: ok
            n += 1
        else:
            n += inject_lora(child, rank, alpha)
    return n


# ----------------------------------------------------------------- metrics
def metrics(q: np.ndarray, gal: np.ndarray, labels: np.ndarray, n_neg: int,
            seed: int) -> dict:
    """Two-way accuracy, per-row MARGIN, and the full-gallery ranking stats.

    History of the `margin` statistic, kept honest here because it was proposed and
    then DISPROVED in this very round (job 581657): `margin` = mean(correct minus
    best-wrong cosine) was meant to give the epoch selector usable resolution, but
    measured over training it fell monotonically (frozen: -0.0949 at epoch 0 to
    -0.1144 at epoch 39) while 2-way ROSE (0.888 -> 0.934). Maximising it therefore
    selected epoch 0 for `frozen`, `disc` and `joint`, i.e. it selected the
    essentially-untrained head and destroyed the round's headline numbers. It is
    retained as a recorded statistic, and `mini_top1` (below) is the selector.
    """
    sim = q @ gal.T
    n_rows = len(q)
    ar = np.arange(n_rows)
    correct = sim[ar, labels]
    # margin: correct concept vs the best INCORRECT concept, per row
    masked = sim.copy()
    masked[ar, labels] = -np.inf
    margin = float((correct - masked.max(1)).mean())

    rng = np.random.default_rng(seed)
    neg = rng.integers(0, gal.shape[0], size=(n_rows, n_neg))
    neg_s = np.take_along_axis(sim, neg, axis=1)
    wins = (correct[:, None] > neg_s).sum() + 0.5 * (correct[:, None] == neg_s).sum()

    order = np.argsort(-sim, axis=1)
    rank = np.argsort(order, axis=1)[ar, labels] + 1
    return {"two_way": float(wins.sum() / (n_rows * n_neg)),
            "margin": margin,
            "paired_cos": float(correct.mean()),
            "top1": float((order[:, 0] == labels).mean()),
            "top5": float(np.mean([labels[i] in order[i, :5] for i in range(n_rows)])),
            "mean_rank": float(rank.mean())}


def mini_gallery(gal: np.ndarray, cid: np.ndarray, rows: np.ndarray):
    """Restrict a concept gallery to the concepts present in `rows`.

    WHY: the reported metric is retrieval over a 200-concept bank, and every selector
    available was the wrong SHAPE for it -- full-gallery `top1` is a 2-5% hit rate
    (noise), `two_way` is near-saturated, and `margin` proved anti-correlated.  This
    turns the held-out concept set into a small replica of the real task: 83 concepts
    / 830 rows for val_a, 82 / 820 for val_b, against concept-mean columns built from
    training-side targets.  It is a different quantity from the 200-way test number,
    so selecting on it does not read the test set.
    """
    concepts = np.unique(cid[rows])
    return gal[concepts], np.searchsorted(concepts, cid[rows]), concepts


def mini_metrics(q: np.ndarray, gal_mini: np.ndarray, labels: np.ndarray) -> dict:
    """Retrieval over the restricted gallery: the selection statistic."""
    sim = q @ gal_mini.T
    order = np.argsort(-sim, axis=1)
    ar = np.arange(len(q))
    rank = np.argsort(order, axis=1)[ar, labels] + 1
    return {"mini_top1": float((order[:, 0] == labels).mean()),
            "mini_top5": float(np.mean([labels[i] in order[i, :5]
                                        for i in range(len(q))])),
            "mini_rank": float(rank.mean())}


def test200_metrics(q: np.ndarray, tgt: np.ndarray) -> dict:
    """The reported metric: 200 test rows, one target per concept."""
    sim = q @ tgt.T
    order = np.argsort(-sim, axis=1)
    n = len(order)
    ar = np.arange(n)
    rank = np.argsort(order, axis=1)[ar, ar] + 1
    return {"top1": float((order[:, 0] == ar).mean()),
            "top5": float(np.mean([i in order[i, :5] for i in range(n)])),
            "mean_rank": float(rank.mean()),
            "paired_cos": float((q * tgt).sum(1).mean())}



@torch.no_grad()
def encode(model, proj, eeg: np.ndarray, dev, bs: int = 512) -> np.ndarray:
    model.eval()
    if proj is not None:
        proj.eval()
    out = []
    for s in range(0, len(eeg), bs):
        z = torch.from_numpy(eeg[s:s + bs]).to(dev)
        r = model(z, SUBJECT, return_parts=True)[2]
        out.append((r if proj is None else proj(r)).cpu().numpy())
    return l2n(np.concatenate(out).astype(np.float32))


def measure(model, proj, eeg_all, rows, gal, cid, dev, n_neg, seed) -> dict:
    """Full-gallery statistics for one row set (kept for ad-hoc probing)."""
    return metrics(encode(model, proj, eeg_all[rows], dev), gal, cid[rows], n_neg, seed)


# ----------------------------------------------------------------- arms
#   mode: how much of the encoder is trainable
#   proj: whether a projection head exists (False => align `r` itself)
#   lr:   "single" (one LR) or "disc" (encoder gets --enc-lr, head gets --lr)
ARM_CFG = {
    "frozen": dict(mode="frozen", proj=True, lr="single"),
    "joint":  dict(mode="full",   proj=True, lr="single"),
    "direct": dict(mode="full",   proj=False, lr="single"),
    "disc":   dict(mode="full",   proj=True, lr="disc"),
    "lora":   dict(mode="lora",   proj=True, lr="single"),
}

# Every statistic that gets its own checkpoint, so the (arm x selector) table can be
# reported instead of a single bet.  `mini_top1` is primary (see metrics/mini_gallery).
SELECTORS = ("mini_top1", "two_way", "margin", "top1")
CHOSEN = "mini_top1"


def build_gallery(tgt_tr: np.ndarray, cid: np.ndarray, n_cls: int) -> np.ndarray:
    gal = np.zeros((n_cls, tgt_tr.shape[1]), dtype=np.float64)
    np.add.at(gal, cid, tgt_tr.astype(np.float64))
    cnt = np.bincount(cid, minlength=n_cls)
    gal /= np.clip(cnt, 1, None)[:, None]
    return l2n(gal.astype(np.float32))


def train_arm(arm: str, args, dev, data: dict) -> dict:
    model, proj, gal, gal_va, lab_va, gal_vb, lab_vb, init_note, n_lora, cfg = build_arm(
        arm, args, dev, data)
    gal_t = torch.from_numpy(gal).to(dev)
    tgt_dim = data["tgt_dim"]

    enc_params = [p for p in model.parameters() if p.requires_grad]
    proj_params = [] if proj is None else list(proj.parameters())
    if cfg["lr"] == "disc" and enc_params:
        opt = torch.optim.AdamW(
            [{"params": enc_params, "lr": args.enc_lr},
             {"params": proj_params, "lr": args.lr}],
            weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(enc_params + proj_params, lr=args.lr,
                                weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    trainable = enc_params + proj_params
    if not trainable:
        raise SystemExit(f"[FATAL] arm '{arm}' has nothing trainable")

    loader = DataLoader(Subset(data["ds"], data["fit"].tolist()),
                        batch_size=args.batch_size, shuffle=True, drop_last=True)
    tgt_all_t = torch.from_numpy(data["tgt_tr"])   # CPU-resident; indexed on the CPU

    d = Path(args.out) / arm
    d.mkdir(parents=True, exist_ok=True)

    # Every selector keeps its own checkpoint, so the round reports a
    # (arm x selector) table instead of betting the whole comparison on one
    # statistic -- which is precisely how job 581657 lost its headline numbers.
    hist: list[dict] = []
    picks = {s: (-1e9, -1) for s in SELECTORS}

    for ep in range(args.epochs):
        model.train()
        if proj is not None:
            proj.train()
        if cfg["mode"] == "frozen":
            model.eval()             # dropout off: this arm must not move at all
        tot, n = 0.0, 0
        for eeg, _img, _txt, sid, obj, _im, _rep in loader:
            eeg, sid, obj = eeg.to(dev), sid.to(dev), obj.to(dev)
            opt.zero_grad(set_to_none=True)
            out, s, r = model(eeg, sid, return_parts=True)
            # DEFECT 1: when a projection exists the loss is on proj(r); when it does
            # not, the loss is on `r` -- the very tensor that gets exported.
            q = r if proj is None else proj(r)
            loss = F.cross_entropy((l2t(q) @ gal_t.T) / args.tau, obj)
            if args.inst_weight > 0:
                loss = loss + args.inst_weight * (
                    1.0 - (l2t(q) * l2t(tgt_all_t[obj.cpu()].to(dev, non_blocking=True))).sum(-1)).mean()
            if cfg["mode"] == "full" and args.lambda_diff > 0:
                loss = loss + args.lambda_diff * diff_loss(s, r)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.clip)
            opt.step()
            tot += float(loss.detach())
            n += 1
        sch.step()

        q_a = encode(model, proj, data["raw_tr"][data["val_a"]], dev)
        q_b = encode(model, proj, data["raw_tr"][data["val_b"]], dev)
        ma = {**metrics(q_a, gal, data["cid"][data["val_a"]], args.n_neg, args.seed),
              **mini_metrics(q_a, gal_va, lab_va)}
        mb = {**metrics(q_b, gal, data["cid"][data["val_b"]], args.n_neg, args.seed + 1),
              **mini_metrics(q_b, gal_vb, lab_vb)}
        row = {"epoch": ep, "loss": tot / max(n, 1),
               **{f"val_a_{k}": v for k, v in ma.items()},
               **{f"val_b_{k}": v for k, v in mb.items()}}
        hist.append(row)

        for sel in SELECTORS:
            v = ma[sel]
            if v > picks[sel][0]:
                picks[sel] = (v, ep)
                torch.save({"state_dict": model.state_dict(),
                            "proj": None if proj is None else proj.state_dict(),
                            "epoch": ep, "arm": arm, "selector": sel, "target": args.target,
                            "val_a": ma, "val_b": mb, "loss": row["loss"],
                            "init": init_note, "n_lora": n_lora,
                            "encode_r_directly": proj is None,
                            "subjects": [SUBJECT], "feature_dim": 1024,
                            "eeg_sample_points": args.eeg_len,
                            "channels_num": args.channels_num,
                            "n_extra_blocks": args.n_extra_blocks},
                           d / f"checkpoint_by_{sel}.pth")
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"[{arm} ep{ep:03d}] loss={row['loss']:.3f} "
                  f"valA 2way={ma['two_way']:.4f} marg={ma['margin']:.4f} "
                  f"mini={ma['mini_top1']:.4f} top1={ma['top1']:.4f} | "
                  f"valB 2way={mb['two_way']:.4f} mini={mb['mini_top1']:.4f}", flush=True)

    return finalize_arm(arm, args, dev, data, model, proj, hist, init_note, n_lora,
                        d, gal)

def build_arm(arm: str, args, dev, data: dict):
    """Construct the arm's model/projection/galleries WITHOUT training it.

    Split out of `train_arm` so that re-scoring an already-trained arm (which is what
    recovers the 4 arms job 581657 finished before crashing) goes through byte-identical
    construction: the same init checkpoint, the same LoRA injection, the same galleries.
    If re-scoring built its own model, the re-scored numbers would come from a differently
    initialised object and the round would not be internally comparable.
    """
    cfg = ARM_CFG[arm]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tgt_dim = data["tgt_dim"]
    if not cfg["proj"] and tgt_dim != 1024:
        raise SystemExit(
            f"[FATAL] arm '{arm}' aligns `r` directly, so the target must be 1024-d "
            f"(the encoder's own width); got {tgt_dim}. Choose a 1024-d --target "
            f"(image / lowresolution / levels_mean) or use an arm with a projection.")

    gal = build_gallery(data["tgt_tr"], data["cid"], data["n_cls"])
    gal_va, lab_va, cva = mini_gallery(gal, data["cid"], data["val_a"])
    gal_vb, lab_vb, cvb = mini_gallery(gal, data["cid"], data["val_b"])

    model = SharedSpecificEncoder(
        subject_ids=[SUBJECT], feature_dim=1024, eeg_sample_points=args.eeg_len,
        channels_num=args.channels_num, n_extra_blocks=args.n_extra_blocks,
        use_adapter=True,
    ).to(dev)

    init_note = "scratch"
    if not args.scratch:
        ck = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if [int(s) for s in ck["subjects"]] != [SUBJECT]:
            raise SystemExit(f"[FATAL] init checkpoint subjects={ck['subjects']} "
                             f"but this run is sub-{SUBJECT:02d}")
        model.load_state_dict(ck["model_state_dict"])
        init_note = f"{Path(args.init_checkpoint).name}:{ck.get('phase')}@ep{ck.get('epoch')}"

    n_lora = 0
    if cfg["mode"] == "frozen":
        for p in model.parameters():
            p.requires_grad = False
    elif cfg["mode"] == "lora":
        for p in model.parameters():
            p.requires_grad = False
        n_lora = inject_lora(model, args.lora_rank, args.lora_alpha)
        if n_lora == 0:
            raise SystemExit("[FATAL] LoRA injection matched no Linear layers")

    proj = None if not cfg["proj"] else Proj(1024, tgt_dim, hidden=args.proj_hidden,
                                            drop=args.drop).to(dev)

    # Runtime backstop for the job-581657 bug class.  On a GPU node this aborts with a
    # message naming the arm and the offending parameter instead of dying inside a
    # matmul with a device error that says nothing about where the tensor came from.
    # It cannot fire on a CPU-only smoke (there is only one device there), which is
    # exactly why the submit script also runs a `meta`-device propagation unit test.
    all_params = list(model.named_parameters())
    if proj is not None:
        all_params += [("proj." + n, p) for n, p in proj.named_parameters()]
    off = [(n, str(p.device)) for n, p in all_params if p.device != dev]
    if off:
        raise SystemExit(
            f"[FATAL] arm '{arm}': {len(off)} of {len(all_params)} parameter tensors "
            f"are not on {dev} -- e.g. {off[0][0]} on {off[0][1]}. A module was built "
            f"on the CPU and grafted onto the device-resident model (setattr does not "
            f"move parameters); construct it with the target's device/dtype instead.")

    return model, proj, gal, gal_va, lab_va, gal_vb, lab_vb, init_note, n_lora, cfg


def finalize_arm(arm: str, args, dev, data, model, proj, hist, init_note, n_lora,
                 d: Path, gal: np.ndarray) -> dict:
    """Score EVERY selector's saved checkpoint on the 200-way test set, then export.

    Scoring all selectors is the direct answer to defect 3.  Job 581657 shipped one
    headline number per arm taken from a single selector (`margin`), and that selector
    turned out to pick epoch 0 -- so a selector bug silently became a results bug with
    nothing in the outputs to reveal it.  Producing the whole (arm x selector) table
    makes the selector's influence visible and bounded.

    The headline is the `mini_top1` checkpoint, chosen on val_a.  Reading test for the
    other selectors is diagnostic only: a selector must never be picked because it
    happens to score well on test, and the report says so explicitly.
    """
    sels: dict[str, dict] = {}
    for sel in SELECTORS:
        f = d / f"checkpoint_by_{sel}.pth"
        if not f.is_file():
            continue
        ck = torch.load(f, map_location=dev, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        if proj is not None and ck["proj"] is not None:
            proj.load_state_dict(ck["proj"])
        q_te = encode(model, proj, data["raw_te"], dev)
        sels[sel] = {"epoch": int(ck["epoch"]),
                     "val_a": ck["val_a"], "val_b": ck["val_b"],
                     **test200_metrics(q_te, data["tgt_te"])}
    if CHOSEN not in sels:
        raise SystemExit(f"[FATAL] arm '{arm}': no checkpoint for the primary "
                         f"selector '{CHOSEN}'; have {sorted(sels)}")

    # export from the PRIMARY selector's weights
    ck = torch.load(d / f"checkpoint_by_{CHOSEN}.pth", map_location=dev,
                    weights_only=False)
    model.load_state_dict(ck["state_dict"])
    if proj is not None and ck["proj"] is not None:
        proj.load_state_dict(ck["proj"])

    # `enc/` holds `r` (what the probe and the routes consume).  `enc_aligned/` holds
    # proj(r) when a projection exists, so the probe can measure the space the loss
    # actually optimised.  Comparing the two for the SAME arm is what tests defect 1.
    spaces: dict[str, str] = {}
    for space, want_proj in (("enc", False), ("enc_aligned", True)):
        if want_proj and proj is None:
            continue
        sd = d / space / f"sub-{SUBJECT:02d}"
        sd.mkdir(parents=True, exist_ok=True)
        for tag, eeg_all in (("train", data["raw_tr"]), ("test", data["raw_te"])):
            with torch.no_grad():
                parts = []
                for s in range(0, len(eeg_all), 512):
                    z = torch.from_numpy(eeg_all[s:s + 512]).to(dev)
                    r = model(z, SUBJECT, return_parts=True)[2]
                    parts.append((proj(r) if want_proj else r).cpu().numpy())
            np.save(sd / f"shared_r_{tag}.npy", np.concatenate(parts).astype(np.float32))
        spaces[space] = str(sd)

    keys = ("two_way", "margin", "paired_cos", "mini_top1", "mini_top5", "mini_rank",
            "top1", "top5", "mean_rank")
    selector_note = {
        "chosen": CHOSEN,
        "why": ("mini_top1 on val_a is the only statistic with the SHAPE of the reported "
                "metric: retrieval over a held-out concept bank. two_way is near-saturated, "
                "full-gallery top1 is a 2-5% hit rate, and margin was measured to fall "
                "monotonically through training in job 581657 and is retained only as a "
                "recorded column."),
        "test_use": ("test is read once per selector for the diagnostic table; the "
                     "selector itself was fixed on val_a BEFORE any of these numbers "
                     "existed, and is never chosen from its test column"),
        "picks": {s: {"epoch": v["epoch"], "val_a": v["val_a"], "val_b": v["val_b"]}
                  for s, v in sels.items()},
        "chosen_val_a": sels[CHOSEN]["val_a"],
        "chosen_val_b": sels[CHOSEN]["val_b"],
        "disagreement": sorted({v["epoch"] for v in sels.values()}),
        "selector_matrix": {s: {"epoch": v["epoch"], "test200_top1": v["top1"],
                                "test200_top5": v["top5"],
                                "test200_mean_rank": v["mean_rank"],
                                "val_a_mini_top1": v["val_a"]["mini_top1"],
                                "val_b_mini_top1": v["val_b"]["mini_top1"]}
                            for s, v in sels.items()},
    }
    if hist:
        selector_note["last_epoch_val_a"] = {k: hist[-1][f"val_a_{k}"] for k in keys}

    headline = sels[CHOSEN]
    result = {"arm": arm, "cfg": cfg_of(arm), "init": init_note, "n_lora": n_lora,
              "encode_r_directly": proj is None, "selectors": selector_note,
              "history": hist, "spaces": spaces,
              "ckpt": str(d / f"checkpoint_by_{CHOSEN}.pth"),
              "test200_top1": headline["top1"],
              "test200_top5": headline["top5"],
              "test200_mean_rank": headline["mean_rank"],
              "test200_paired_cos": headline["paired_cos"],
              # Aliases so the UNCHANGED cfmsf_joint_summary.py can print this round's
              # numbers in its comparison table instead of `nan`.  `best.val_a` is the
              # dict the summary means by it; `two_way` here is val_b's number (the
              # independent held-out check), labelled as such in fix_report.json.
              "best": {"val_a": headline["val_a"]},
              "two_way": headline["val_b"]["two_way"]}
    return result


def cfg_of(arm: str) -> dict:
    return ARM_CFG[arm]


def rescore_arm(arm: str, args, dev, data: dict) -> dict:
    """Rebuild an arm's model and score its EXISTING per-selector checkpoints.

    This is the resume path.  Job 581657 crashed in its fifth arm AFTER writing three
    checkpoints for each of the first four, so `frozen`/`joint`/`direct`/`disc` can be
    re-scored without retraining -- which matters because the crash fix and the
    selector fix both landed after ~50 minutes of GPU time had already been spent.
    """
    model, proj, gal, _gva, _lva, _gvb, _lvb, init_note, n_lora, _cfg = build_arm(
        arm, args, dev, data)
    print(f"[{arm}] re-scoring existing checkpoints (no training)", flush=True)
    return finalize_arm(arm, args, dev, data, model, proj, None, init_note, n_lora,
                        Path(args.out) / arm, gal)




def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--arms", type=str, default="frozen,joint,direct,disc,lora")
    ap.add_argument("--target", type=str, default="levels_mean",
                    choices=["image", "lowresolution", "levels_mean", "cat3", "cat5"])
    ap.add_argument("--test-subject", type=int, default=8)
    ap.add_argument("--init-checkpoint", type=str, default="")
    ap.add_argument("--scratch", action="store_true")
    ap.add_argument("--split-json", type=str,
                    default=str(NB_ROOT / "outputs/leakfree/split.json"))
    ap.add_argument("--eeg-dir", type=str,
                    default=str(NB_ROOT / "data/things_eeg/preprocessed_eeg"))
    ap.add_argument("--rn50-dir", type=str,
                    default=str(NB_ROOT / "data/things_eeg/image_feature/RN50"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4, help="head / projection LR")
    ap.add_argument("--enc-lr", type=float, default=1e-5,
                    help="encoder LR for the `disc` arm (30x below the head LR)")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
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
    ap.add_argument("--force", action="store_true",
                    help=("retrain arms even when their result/checkpoints exist. "
                          "Default is to resume: an arm with arm_result.json is "
                          "skipped, and an arm with only its per-selector checkpoints "
                          "is re-SCORED without retraining (the path that recovers the "
                          "four arms job 581657 finished before crashing)."))
    args = ap.parse_args()

    global SUBJECT
    SUBJECT = args.test_subject
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.scratch:
        args.init_checkpoint = ""
    elif not args.init_checkpoint:
        cand = (NB_ROOT / f"outputs/ocf/intra_enc/sub-{SUBJECT:02d}"
                / "checkpoint_ss_calib_best.pth")
        if not cand.is_file():
            raise SystemExit(f"[FATAL] no per-subject init encoder at {cand}")
        args.init_checkpoint = str(cand)
        print(f"[fix] init encoder: {cand.name}", flush=True)

    ds = EEGPreImageDataset([SUBJECT], args.eeg_dir, DEFAULT_CHANNELS, [0, 250],
                            args.rn50_dir, "", False, [], True, False, None, True,
                            False, False, False)
    ds_te = EEGPreImageDataset([SUBJECT], args.eeg_dir, DEFAULT_CHANNELS, [0, 250],
                               args.rn50_dir, "", False, [], True, False, None, False,
                               False, False, False)
    args.eeg_len = int(ds.num_sample_points)
    args.channels_num = int(ds.channels_num)
    raw_tr = np.stack([ds[i][0].numpy() for i in range(len(ds))]).astype(np.float32)
    raw_te = np.stack([ds_te[i][0].numpy() for i in range(len(ds_te))]).astype(np.float32)
    cid = np.array([ds[i][4] for i in range(len(ds))], dtype=np.int64)
    n_cls = int(cid.max()) + 1

    tgt_tr, tgt_te, tgt_dim = build_target(args.target)
    split = LF.load(args.split_json)
    fit = LF.rows_for(split, "fit", len(cid))
    val_a = LF.rows_for(split, "val_a", len(cid))
    val_b = LF.rows_for(split, "val_b", len(cid))
    if set(fit.tolist()) & (set(val_a.tolist()) | set(val_b.tolist())):
        raise SystemExit("[FATAL] fit/val overlap")

    data = {"ds": ds, "raw_tr": raw_tr, "raw_te": raw_te, "cid": cid, "n_cls": n_cls,
            "fit": fit, "val_a": val_a, "val_b": val_b,
            "tgt_tr": tgt_tr, "tgt_te": tgt_te, "tgt_dim": tgt_dim}
    print(f"[fix] sub-{SUBJECT:02d} raw={raw_tr.shape} concepts={n_cls} "
          f"target={args.target} dim={tgt_dim} fit={len(fit)} "
          f"valA={len(val_a)} valB={len(val_b)}", flush=True)

    report: dict = {
        "pipeline": "cfmsf_fix", "subject": f"sub-{SUBJECT:02d}", "target": args.target,
        "target_dim": tgt_dim,
        "why": ("job 581652 measured joint training degrading val while train loss fell; "
                "this round separates the objective mismatch, the encoder capacity, and "
                "the selector resolution"),
        "params": {k: getattr(args, k) for k in
                   ("epochs", "lr", "enc_lr", "lora_rank", "lora_alpha", "tau",
                    "inst_weight", "lambda_diff", "batch_size", "weight_decay",
                    "drop", "proj_hidden", "seed")},
        "selection": {"set": "val_a", "primary_statistic": CHOSEN,
                      "checkpointed_alternatives": list(SELECTORS),
                      "rule": ("the selector is fixed on val_a before any test number "
                               "exists; the per-selector test columns are diagnostic "
                               "and are never used to choose a selector"),
                      "test_concepts_read": "once per selector, after training"},
        "arms": {},
    }
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if arm not in ARM_CFG:
            raise SystemExit(f"[FATAL] unknown arm {arm}; have {sorted(ARM_CFG)}")
        res_file = out / arm / "arm_result.json"
        cks = [out / arm / f"checkpoint_by_{s}.pth" for s in SELECTORS]

        if res_file.is_file() and not args.force:
            # Re-read the stored result rather than re-deriving it: a resumed run must
            # not silently recompute an arm on a different code path than the one that
            # produced the number being carried forward.
            report["arms"][arm] = json.loads(res_file.read_text())
            print(f"[{arm}] [skip] arm_result.json present (use --force to retrain)",
                  flush=True)
        elif all(c.is_file() for c in cks) and not args.force:
            report["arms"][arm] = rescore_arm(arm, args, dev, data)
            res_file.write_text(json.dumps(report["arms"][arm], indent=2),
                                encoding="utf-8")
        else:
            report["arms"][arm] = train_arm(arm, args, dev, data)
            res_file.write_text(json.dumps(report["arms"][arm], indent=2),
                                encoding="utf-8")

        r = report["arms"][arm]
        sm = r["selectors"]["selector_matrix"]
        picks = " ".join(f"{s}->ep{sm[s]['epoch']}" for s in SELECTORS if s in sm)
        print(f"[{arm}] test200 top1={r['test200_top1']:.4f} "
              f"top5={r['test200_top5']:.4f} | {picks}", flush=True)

    # The selector question, answered from the table rather than assumed: for each arm,
    # which selectors' checkpoints actually scored best on the 200-way test set.
    matrix = {a: r["selectors"]["selector_matrix"] for a, r in report["arms"].items()
              if r.get("selectors", {}).get("selector_matrix")}
    report["selector_analysis"] = {
        "per_arm": {a: {s: v["test200_top1"] for s, v in m.items()}
                    for a, m in matrix.items()},
        "caveat": ("reading the best cell here is selection-on-test and is NOT how the "
                   "headline number is produced; this table exists to bound how much a "
                   "selector choice can move a result, and to show that the primary "
                   "selector was chosen on val_a alone"),
    }
    report["reference"] = {
        "job_581652_frozen_on_10subj": "in flight; frozen is the arm every fix must beat",
        "probe_581602_frozen_single_route": 0.405,
        "probe_581602_fuse13_csls": 0.50,
        "probe_581602_fuse13_csls_sinkhorn": 0.715,
    }
    (out / "fix_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # The existing cfmsf_joint_summary.py reads `<root>/<arm>/probe/...` and looks for
    # `joint_report.json`; writing the same payload under that name lets the unchanged
    # summary tool consume this run.  The `pipeline` field keeps the provenance honest.
    (out / "joint_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[fix] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
