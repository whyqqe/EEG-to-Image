#!/usr/bin/env python3
"""NeuroWeave Stage-1 (sub-08): corrected hierarchical EEG representation.

WHY THIS ROUND EXISTS
---------------------
NeuroWeave.md proposed hierarchical + causal + retrieval-augmented diffusion.
Measured defects in that design (and in our prior stack) force three *choice*
axes that the user asked to test separately rather than pick a priori:

  AXIS H -- hierarchical objective (§5.2 corrected)
    The encoder was trained on RN50 image contrastive; the route bank's best
    targets are ViT-H-14 multi-level aggregates.  Arms below change ONLY that.

  AXIS T -- temporal / "Causal" (§5.4 disputed)
    Static images + full-window visibility make a hard causal mask unmotivated.
    Three operationalisations are tested in the SAME job so the paper can pick
    based on evidence, not taste:
      * drop         : full window, no temporal policy (rename without Causal)
      * anytime_train: random temporal truncation during training
      * causal_stage : stage-matched masks (early head sees early EEG only, ...)
    Anytime *eval* (mask later samples at test time) is a separate script and
    runs on every arm, so "drop vs progressive decoding" is measured for free.

  AXIS C -- capacity (known joint failure)
    Full joint fine-tune lost 0/10 (p=0.002).  This round therefore uses
    frozen / LoRA / direct (objective-matched) only -- never a free joint arm.

ARMS (one job, leak-free, resumable)
------------------------------------
  frozen         : encoder frozen, Proj -> levels_mean           [control]
  lora           : LoRA rank-8 on encoder, Proj -> levels_mean   [capacity fix]
  direct         : LoRA, align `r` itself to levels_mean         [obj match]
  multi_head     : LoRA + 3 heads (early/mid/late visual levels) [true hierarchy]
  anytime_train  : like lora, but random temporal truncation     [T: anytime]
  causal_stage   : multi_head + stage-matched temporal masks     [T: causal]

Selection uses `mini_top1` on val_a (margin was anti-correlated in job 581657).
Exports `shared_r_{train,test}.npy` under <out>/<arm>/enc/sub-08/ for the probe.
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
from ss_modules import SharedSpecificEncoder  # noqa: E402
import leakfree as LF  # noqa: E402
from cfmsf_joint_train import (DEFAULT_CHANNELS, Proj, build_target,  # noqa: E402
                               l2n, l2t, load_level, VITH)
from cfmsf_fix_train import (  # noqa: E402
    LoRALinear, inject_lora, metrics, mini_gallery, mini_metrics, test200_metrics,
)

SUBJECT = 8
SELECTORS = ("mini_top1", "two_way", "top1")
CHOSEN = "mini_top1"

# 250 Hz -> sample indices for NeuroWeave's early/mid/late windows.
WIN = {"early": 38, "mid": 88, "late": 175, "full": 250}  # 150/350/700/1000 ms
WINDOW_LIST = ("early", "mid", "late", "full")
# Arms that train on truncated windows; their anytime curve is a reported output.
ANYTIME_ARMS = ("anytime_train", "anytime_soft", "anytime_curric", "anytime_consist")


def sample_end(mode: str, ep: int, args, rng) -> int:
    """Which temporal truncation to train on this step.

    ROUND 1 RESULT (job 586588/586589) -- the motivation for this function.
    `anytime_train` sampled the four windows UNIFORMLY, so only 25% of steps saw a
    complete trial.  It produced the round's one clear win -- 200-way top1 0.130 at
    150 ms vs 0.045 for the next best arm, ~3x -- but its FULL-window number fell
    from 0.390 to 0.330.  The model was trained mostly on incomplete input and was
    therefore out of distribution exactly where the paper reports (and where the
    route bank is fitted).  Three schedules attack that, separated because they can
    fail differently:

      anytime : round-1 rule (uniform over 4), kept ONLY for comparability.
      soft    : keep the full window with probability `--anytime-full-p`, else
                sample uniformly from the three truncated windows, so the full
                window is always in distribution.
      curric  : start with NO truncation and ramp the truncation probability to
                `--anytime-max-p` over `--curric-ramp` epochs -- tests whether
                seeing complete trials first is what the full window needs.
    """
    if mode == "anytime":
        return int(rng.choice([WIN[w] for w in WINDOW_LIST]))
    if mode == "soft":
        if rng.random() < args.anytime_full_p:
            return WIN["full"]
        return int(rng.choice([WIN["early"], WIN["mid"], WIN["late"]]))
    if mode == "curric":
        # EXACTLY zero truncation at epoch 0 (`ep / (ramp-1)`, not `(ep+1)/ramp`):
        # the point of this schedule is that the model sees ONLY complete trials
        # first, so the full window is the in-distribution case before any
        # truncation is introduced.  The `(ep+1)/ramp` form starts at max_p/ramp
        # (3% at ramp=20), which a smoke check flagged as not matching the intent.
        ramp = max(2, args.curric_ramp)
        p = min(args.anytime_max_p, args.anytime_max_p * ep / (ramp - 1))
        if rng.random() >= p:
            return WIN["full"]
        return int(rng.choice([WIN["early"], WIN["mid"], WIN["late"]]))
    raise KeyError(f"no temporal schedule for mode={mode!r}")



def mask_time(eeg: torch.Tensor, end: int) -> torch.Tensor:
    """Zero samples at and after `end`. Keeps input width fixed (no re-init)."""
    if end >= eeg.shape[-1]:
        return eeg
    out = eeg.clone()
    out[..., end:] = 0
    return out


def build_level_bank() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-stream targets for multi_head / causal_stage."""
    early_tr = l2n(0.5 * (load_level(VITH, "GaussianBlur", "train")
                          + load_level(VITH, "LowResolution", "train")))
    early_te = l2n(0.5 * (load_level(VITH, "GaussianBlur", "test")
                          + load_level(VITH, "LowResolution", "test")))
    mid_tr = load_level(VITH, "image", "train")
    mid_te = load_level(VITH, "image", "test")
    late_tr, late_te, _ = build_target("levels_mean")
    return {
        "early": (early_tr, early_te),
        "mid": (mid_tr, mid_te),
        "late": (late_tr, late_te),
    }


def concept_gallery(tgt: np.ndarray, cid: np.ndarray, n_cls: int) -> np.ndarray:
    gal = np.zeros((n_cls, tgt.shape[1]), dtype=np.float64)
    np.add.at(gal, cid, tgt.astype(np.float64))
    cnt = np.bincount(cid, minlength=n_cls)
    gal /= np.clip(cnt, 1, None)[:, None]
    return l2n(gal.astype(np.float32))


@torch.no_grad()
def encode_r(model, eeg: np.ndarray, end: int | None, dev, bs: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(eeg), bs):
        z = torch.from_numpy(eeg[s:s + bs]).to(dev)
        if end is not None:
            z = mask_time(z, end)
        out.append(model(z, SUBJECT, return_parts=True)[2].cpu().numpy())
    return l2n(np.concatenate(out).astype(np.float32))


@torch.no_grad()
def encode_q(model, proj, eeg: np.ndarray, end: int | None, dev, bs: int = 512) -> np.ndarray:
    model.eval()
    if proj is not None:
        proj.eval()
    out = []
    for s in range(0, len(eeg), bs):
        z = torch.from_numpy(eeg[s:s + bs]).to(dev)
        if end is not None:
            z = mask_time(z, end)
        r = model(z, SUBJECT, return_parts=True)[2]
        out.append((r if proj is None else proj(r)).cpu().numpy())
    return l2n(np.concatenate(out).astype(np.float32))


class MultiHead(nn.Module):
    """Three independent projections: early / mid / late visual levels."""

    def __init__(self, dim: int = 1024, out_dim: int = 1024, drop: float = 0.15):
        super().__init__()
        self.early = Proj(dim, out_dim, drop=drop)
        self.mid = Proj(dim, out_dim, drop=drop)
        self.late = Proj(dim, out_dim, drop=drop)

    def forward(self, r: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"early": self.early(r), "mid": self.mid(r), "late": self.late(r)}


ARM_CFG = {
    "frozen":          dict(mode="frozen", proj="single", temporal="none"),
    "lora":            dict(mode="lora",   proj="single", temporal="none"),
    "direct":          dict(mode="lora",   proj="none",   temporal="none"),
    "multi_head":      dict(mode="lora",   proj="multi",  temporal="none"),
    "anytime_train":   dict(mode="lora",   proj="single", temporal="anytime"),
    "causal_stage":    dict(mode="lora",   proj="multi",  temporal="causal"),
    # --- round 2: fix the anytime full-window regression ---------------------
    # `anytime_train` won at 150 ms (0.130 vs 0.045) but LOST at 1000 ms
    # (0.330 vs 0.390).  These three keep the 150 ms win and try to remove the
    # cost, and they are separated because the mechanism differs.
    "anytime_soft":    dict(mode="lora",   proj="single", temporal="soft"),
    "anytime_curric":  dict(mode="lora",   proj="single", temporal="curric"),
    "anytime_consist": dict(mode="lora",   proj="single", temporal="soft",
                            consist=True),
    # --- round 2: replace the 3-head hierarchy that did not win ---------------
    # `multi_head` (0.3700) did not beat `lora` (0.3900), so separate heads are
    # not the mechanism.  This keeps ONE exported head (the late one) and adds
    # early/mid supervision as auxiliary losses, which is the weaker claim the
    # evidence can actually support.
    "hier_aux":        dict(mode="lora",   proj="single", temporal="none",
                            aux=True),
}
ROUND1_ARMS = ("frozen", "lora", "direct", "multi_head", "anytime_train",
               "causal_stage")



def build_model(arm: str, args, dev, data: dict):
    cfg = ARM_CFG[arm]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = SharedSpecificEncoder(
        subject_ids=[SUBJECT], feature_dim=1024, eeg_sample_points=args.eeg_len,
        channels_num=args.channels_num, n_extra_blocks=args.n_extra_blocks,
        use_adapter=True,
    ).to(dev)

    ck = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    if [int(s) for s in ck["subjects"]] != [SUBJECT]:
        raise SystemExit(
            f"[FATAL] init subjects={[int(s) for s in ck['subjects']]} != [{SUBJECT}]")
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
        # device-audit: ok -- LoRALinear inherits device from base.weight
        bad = [n for n, p in model.named_parameters()
               if p.requires_grad and p.device != dev]
        if bad:
            raise SystemExit(f"[FATAL] LoRA params on wrong device: {bad[:5]}")

    proj = None
    if cfg["proj"] == "single":
        proj = Proj(1024, data["tgt_dim"], drop=args.drop).to(dev)
    elif cfg["proj"] == "multi":
        proj = MultiHead(1024, 1024, drop=args.drop).to(dev)
    # proj == "none" => align r itself; requires tgt_dim == 1024
    if cfg["proj"] == "none" and data["tgt_dim"] != 1024:
        raise SystemExit("[FATAL] direct arm requires 1024-d target (levels_mean)")

    return model, proj, init_note, n_lora, cfg


def train_arm(arm: str, args, dev, data: dict) -> dict:
    out_arm = Path(args.out) / arm
    result_path = out_arm / "arm_result.json"
    if result_path.is_file():
        print(f"[{arm}] resume: {result_path}", flush=True)
        return json.loads(result_path.read_text())

    model, proj, init_note, n_lora, cfg = build_model(arm, args, dev, data)
    out_arm.mkdir(parents=True, exist_ok=True)

    banks = data["banks"]  # early/mid/late (tr, te)
    gal_late = concept_gallery(data["tgt_tr"], data["cid"], data["n_cls"])
    gal_va, lab_va, _ = mini_gallery(gal_late, data["cid"], data["val_a"])
    gal_late_t = torch.from_numpy(gal_late).to(dev)

    gals = {}
    gals_t = {}
    for name, (tr, _te) in banks.items():
        g = concept_gallery(tr, data["cid"], data["n_cls"])
        gals[name] = g
        gals_t[name] = torch.from_numpy(g).to(dev)

    enc_params = [p for p in model.parameters() if p.requires_grad]
    if isinstance(proj, MultiHead):
        head_params = list(proj.parameters())
    elif proj is not None:
        head_params = list(proj.parameters())
    else:
        head_params = []
    trainable = enc_params + head_params
    if not trainable:
        raise SystemExit(f"[FATAL] arm {arm} has nothing trainable")
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    loader = DataLoader(Subset(data["ds"], data["fit"].tolist()),
                        batch_size=args.batch_size, shuffle=True, drop_last=True)
    tgt_late_t = torch.from_numpy(data["tgt_tr"])
    tgt_early_t = torch.from_numpy(banks["early"][0])
    tgt_mid_t = torch.from_numpy(banks["mid"][0])

    hist: list[dict] = []
    picks = {s: (-1e9, -1) for s in SELECTORS}
    rng = np.random.default_rng(args.seed)

    for ep in range(args.epochs):
        model.train()
        if proj is not None:
            proj.train()
        if cfg["mode"] == "frozen":
            model.eval()
        tot, n = 0.0, 0
        for eeg, _img, _txt, sid, obj, _im, _rep in loader:
            eeg, sid, obj = eeg.to(dev), sid.to(dev), obj.to(dev)
            opt.zero_grad(set_to_none=True)

            if cfg["temporal"] in ("anytime", "soft", "curric"):
                end = sample_end(cfg["temporal"], ep, args, rng)
                eeg_in = mask_time(eeg, end)
            else:
                eeg_in = eeg

            if cfg["temporal"] == "causal":
                # Stage-matched: each head sees only its temporal window.
                r_e = model(mask_time(eeg, WIN["early"]), sid, return_parts=True)[2]
                r_m = model(mask_time(eeg, WIN["mid"]), sid, return_parts=True)[2]
                r_l = model(eeg, sid, return_parts=True)[2]
                assert isinstance(proj, MultiHead)
                qe, qm, ql = proj.early(r_e), proj.mid(r_m), proj.late(r_l)
                loss = (
                    F.cross_entropy((l2t(qe) @ gals_t["early"].T) / args.tau, obj)
                    + F.cross_entropy((l2t(qm) @ gals_t["mid"].T) / args.tau, obj)
                    + F.cross_entropy((l2t(ql) @ gals_t["late"].T) / args.tau, obj)
                ) / 3.0
                if args.inst_weight > 0:
                    loss = loss + args.inst_weight * (
                        (1 - (l2t(qe) * l2t(tgt_early_t[obj.cpu()].to(dev))).sum(-1)).mean()
                        + (1 - (l2t(qm) * l2t(tgt_mid_t[obj.cpu()].to(dev))).sum(-1)).mean()
                        + (1 - (l2t(ql) * l2t(tgt_late_t[obj.cpu()].to(dev))).sum(-1)).mean()
                    ) / 3.0
            elif cfg["proj"] == "multi":
                r = model(eeg_in, sid, return_parts=True)[2]
                qs = proj(r)
                loss = (
                    F.cross_entropy((l2t(qs["early"]) @ gals_t["early"].T) / args.tau, obj)
                    + F.cross_entropy((l2t(qs["mid"]) @ gals_t["mid"].T) / args.tau, obj)
                    + F.cross_entropy((l2t(qs["late"]) @ gals_t["late"].T) / args.tau, obj)
                ) / 3.0
                if args.inst_weight > 0:
                    loss = loss + args.inst_weight * (
                        (1 - (l2t(qs["early"]) * l2t(tgt_early_t[obj.cpu()].to(dev))).sum(-1)).mean()
                        + (1 - (l2t(qs["mid"]) * l2t(tgt_mid_t[obj.cpu()].to(dev))).sum(-1)).mean()
                        + (1 - (l2t(qs["late"]) * l2t(tgt_late_t[obj.cpu()].to(dev))).sum(-1)).mean()
                    ) / 3.0
            else:
                r = model(eeg_in, sid, return_parts=True)[2]
                q = r if proj is None else proj(r)
                loss = F.cross_entropy((l2t(q) @ gal_late_t.T) / args.tau, obj)
                if args.inst_weight > 0:
                    loss = loss + args.inst_weight * (
                        1 - (l2t(q) * l2t(tgt_late_t[obj.cpu()].to(dev))).sum(-1)).mean()
                if cfg.get("aux"):
                    # HIERARCHICAL SUPERVISION, ONE HEAD.  Round 1 showed three
                    # separate heads do not beat one (`multi_head` 0.3700 vs
                    # `lora` 0.3900), so the hierarchy is expressed as auxiliary
                    # losses on the SAME head that gets exported.  If this does
                    # not beat `lora` either, hierarchy as a contribution is dead
                    # and the paper should say so rather than re-parameterise it.
                    ra = model(mask_time(eeg, WIN["early"]), sid,
                               return_parts=True)[2]
                    rm = model(mask_time(eeg, WIN["mid"]), sid,
                               return_parts=True)[2]
                    loss = loss + args.aux_weight * (
                        F.cross_entropy((l2t(proj(ra)) @ gals_t["early"].T) / args.tau, obj)
                        + F.cross_entropy((l2t(proj(rm)) @ gals_t["mid"].T) / args.tau, obj))
                if cfg.get("consist"):
                    # PROGRESSIVE-DECODING OBJECTIVE.  The same trial read from a
                    # truncated window must land near its OWN full-window
                    # representation.  The target is detached, so only the
                    # partial-window path moves -- this is what should make the
                    # anytime curve FLAT instead of rising with the window, and
                    # it is the mechanism that could remove the round-1
                    # full-window cost instead of trading against it.
                    with torch.no_grad():
                        q_full = proj(model(eeg, sid, return_parts=True)[2])
                    loss = loss + args.tcons_weight * (
                        1 - (l2t(q) * l2t(q_full)).sum(-1)).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.clip)
            opt.step()
            tot += float(loss.detach())
            n += 1
        sch.step()

        # Selection always uses FULL-window late space (comparable across arms).
        if cfg["proj"] == "multi":
            # score with late head on full-window r
            model.eval()
            proj.eval()
            with torch.no_grad():
                parts = []
                for s in range(0, len(data["val_a"]), 512):
                    rows = data["val_a"][s:s + 512]
                    z = torch.from_numpy(data["raw_tr"][rows]).to(dev)
                    r = model(z, SUBJECT, return_parts=True)[2]
                    parts.append(proj.late(r).cpu().numpy())
                q_a = l2n(np.concatenate(parts))
        else:
            q_a = encode_q(model, proj, data["raw_tr"][data["val_a"]], None, dev)

        ma = {**metrics(q_a, gal_late, data["cid"][data["val_a"]], args.n_neg, args.seed),
              **mini_metrics(q_a, gal_va, lab_va)}
        # Diagnostic only (never selects): how good is the SAME checkpoint when the
        # trial is truncated to 150/350 ms?  Checkpoint selection stays on the full
        # window so every arm is picked by one rule, but recording these makes the
        # anytime curve auditable per epoch instead of only after the fact.
        if arm in ANYTIME_ARMS and proj is not None and not isinstance(proj, MultiHead):
            for _wn, _ws in (("early", WIN["early"]), ("mid", WIN["mid"])):
                q_w = encode_q(model, proj, data["raw_tr"][data["val_a"]], _ws, dev)
                ma[f"{_wn}_mini_top1"] = mini_metrics(q_w, gal_va, lab_va)["mini_top1"]
        row = {"epoch": ep, "loss": tot / max(n, 1),
               **{f"val_a_{k}": v for k, v in ma.items()}}
        hist.append(row)

        for sel in SELECTORS:
            v = ma[sel]
            if v > picks[sel][0]:
                picks[sel] = (v, ep)
                torch.save({
                    "state_dict": model.state_dict(),
                    "proj": None if proj is None else proj.state_dict(),
                    "epoch": ep, "arm": arm, "selector": sel,
                    "cfg": cfg, "init": init_note, "n_lora": n_lora,
                    "val_a": ma, "subjects": [SUBJECT],
                    "feature_dim": 1024, "eeg_sample_points": args.eeg_len,
                    "channels_num": args.channels_num,
                    "n_extra_blocks": args.n_extra_blocks,
                    "encode_r_directly": proj is None,
                    "multi_head": isinstance(proj, MultiHead),
                }, out_arm / f"checkpoint_by_{sel}.pth")

        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"[{arm} ep{ep:03d}] loss={row['loss']:.3f} "
                  f"valA 2way={ma['two_way']:.4f} mini={ma['mini_top1']:.4f} "
                  f"top1={ma['top1']:.4f}", flush=True)

    # Load chosen checkpoint and export shared_r + test200.
    ck_path = out_arm / f"checkpoint_by_{CHOSEN}.pth"
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    if proj is not None and ck["proj"] is not None:
        proj.load_state_dict(ck["proj"])

    # test200 in late space
    if cfg["proj"] == "multi":
        with torch.no_grad():
            parts = []
            for s in range(0, len(data["raw_te"]), 512):
                z = torch.from_numpy(data["raw_te"][s:s + 512]).to(dev)
                r = model(z, SUBJECT, return_parts=True)[2]
                parts.append(proj.late(r).cpu().numpy())
            q_te = l2n(np.concatenate(parts))
        te = test200_metrics(q_te, banks["late"][1])
    else:
        q_te = encode_q(model, proj, data["raw_te"], None, dev)
        te = test200_metrics(q_te, data["tgt_te"])

    enc_dir = out_arm / "enc" / f"sub-{SUBJECT:02d}"
    enc_dir.mkdir(parents=True, exist_ok=True)
    for tag, eeg_all in (("train", data["raw_tr"]), ("test", data["raw_te"])):
        np.save(enc_dir / f"shared_r_{tag}.npy",
                encode_r(model, eeg_all, None, dev).astype(np.float32))

    result = {
        "arm": arm, "cfg": cfg, "init": init_note, "n_lora": n_lora,
        "chosen_selector": CHOSEN,
        "picks": {s: {"value": float(picks[s][0]), "epoch": int(picks[s][1])}
                  for s in SELECTORS},
        "history_tail": hist[-6:],
        "val_a_chosen": hist[picks[CHOSEN][1]] if picks[CHOSEN][1] >= 0 else None,
        "test200": te,
        "enc_export": str(enc_dir),
        "ckpt": str(ck_path),
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[{arm}] test200 top1={te['top1']:.4f} top5={te['top5']:.4f} "
          f"rank={te['mean_rank']:.1f}  wrote {enc_dir}", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--arms", type=str,
                    default="frozen,lora,anytime_train,anytime_soft,anytime_curric,"
                            "anytime_consist,hier_aux")
    ap.add_argument("--target", type=str, default="levels_mean",
                    choices=["image", "levels_mean", "cat3", "cat5"])
    ap.add_argument("--test-subject", type=int, default=8)
    ap.add_argument("--init-checkpoint", type=str, default="")
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
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--inst-weight", type=float, default=0.2)
    # ---- round-2 temporal schedule knobs -----------------------------------
    ap.add_argument("--anytime-full-p", type=float, default=0.5,
                    help=("P(full window) for the 'soft'/'consist' schedules.  Round 1 "
                          "used the uniform 'anytime' rule (P(full)=0.25), which won at "
                          "150 ms but cost 0.06 at the full window; this knob is the "
                          "direct control over that tradeoff."))
    ap.add_argument("--anytime-max-p", type=float, default=0.6,
                    help="final truncation probability for the 'curric' ramp")
    ap.add_argument("--curric-ramp", type=int, default=20,
                    help="epochs over which 'curric' ramps truncation 0 -> max-p")
    ap.add_argument("--aux-weight", type=float, default=0.3,
                    help="weight on the early/mid auxiliary CE terms for arm 'hier_aux'")
    ap.add_argument("--tcons-weight", type=float, default=1.0,
                    help=("weight on the truncated-vs-full representation consistency "
                          "term for arm 'anytime_consist' (the progressive-decoding "
                          "objective)"))
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--n-neg", type=int, default=64)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--n-extra-blocks", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    global SUBJECT
    SUBJECT = args.test_subject
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.init_checkpoint:
        args.init_checkpoint = str(
            NB_ROOT / f"outputs/ocf/intra_enc/sub-{SUBJECT:02d}"
            / "checkpoint_ss_calib_best.pth")
    if not Path(args.init_checkpoint).is_file():
        raise SystemExit(f"[FATAL] missing init {args.init_checkpoint}")

    # direct requires 1024-d; force levels_mean when arms include it
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARM_CFG:
            raise SystemExit(f"[FATAL] unknown arm {a}; have {sorted(ARM_CFG)}")
    if "direct" in arms and args.target != "levels_mean":
        print("[warn] forcing --target levels_mean because direct is in arms",
              flush=True)
        args.target = "levels_mean"

    ds_tr = EEGPreImageDataset(
        [SUBJECT], args.eeg_dir, DEFAULT_CHANNELS, [0, 250],
        args.rn50_dir, "", False, [], True, False, None, True, False, False, False)
    ds_te = EEGPreImageDataset(
        [SUBJECT], args.eeg_dir, DEFAULT_CHANNELS, [0, 250],
        args.rn50_dir, "", False, [], True, False, None, False, False, False, False)
    args.eeg_len = int(ds_tr.num_sample_points)
    args.channels_num = int(ds_tr.channels_num)

    raw_tr = np.stack([ds_tr[i][0].numpy() for i in range(len(ds_tr))]).astype(np.float32)
    raw_te = np.stack([ds_te[i][0].numpy() for i in range(len(ds_te))]).astype(np.float32)
    cid = np.array([ds_tr[i][4] for i in range(len(ds_tr))], dtype=np.int64)
    n_cls = int(cid.max()) + 1
    tgt_tr, tgt_te, tgt_dim = build_target(args.target)
    banks = build_level_bank()

    split = LF.load(args.split_json)
    fit = LF.rows_for(split, "fit", len(ds_tr))
    val_a = LF.rows_for(split, "val_a", len(ds_tr))
    val_b = LF.rows_for(split, "val_b", len(ds_tr))
    if set(fit.tolist()) & (set(val_a.tolist()) | set(val_b.tolist())):
        raise SystemExit("[FATAL] fit/val overlap")

    print(f"[nweave-s1] sub-{SUBJECT:02d} raw={raw_tr.shape} target={args.target} "
          f"dim={tgt_dim} arms={arms}", flush=True)

    data = {
        "ds": ds_tr, "raw_tr": raw_tr, "raw_te": raw_te, "cid": cid, "n_cls": n_cls,
        "fit": fit, "val_a": val_a, "val_b": val_b,
        "tgt_tr": tgt_tr, "tgt_te": tgt_te, "tgt_dim": tgt_dim, "banks": banks,
    }

    report = {
        "subject": f"sub-{SUBJECT:02d}",
        "target": args.target,
        "target_dim": tgt_dim,
        "windows_samples": WIN,
        "axes": {
            "H": "hierarchical objective (levels_mean / multi_head)",
            "T": "temporal policy (none / anytime_train / causal_stage)",
            "C": "capacity (frozen / lora / direct) -- joint excluded (0/10 fail)",
        },
        "selection": {"set": "val_a", "statistic": CHOSEN},
        "leakfree": {k: int(split["counts"][k]) for k in ("fit", "val_a", "val_b")},
        "arms": {},
        "params": {k: getattr(args, k) for k in
                   ("epochs", "lr", "tau", "inst_weight", "lora_rank", "lora_alpha",
                    "batch_size", "seed")},
    }
    for arm in arms:
        report["arms"][arm] = train_arm(arm, args, dev, data)

    (out / "s1_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[nweave-s1] wrote {out / 's1_report.json'}", flush=True)


if __name__ == "__main__":
    main()
