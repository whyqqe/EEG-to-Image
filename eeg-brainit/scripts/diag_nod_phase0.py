#!/usr/bin/env python3
"""Phase-0 information diagnosis for NOD NeuroBOLT → fMRI → CLIP cascade.

Quantifies whether Phase-1 predicted ROIs contain CLIP-decodable visual
semantics. Produces go / no-go for further cascade (Stage B) training.

Metrics:
  1) pred vs GT geometry (sample/dim corr, diversity, category structure)
  2) fMRI retrieval (pred → GT)
  3) ridge probe ROI → CLIP (GT vs pred vs moment-matched pred)
  4) frozen Stage-A NodRoiBitDecoder retrieval (GT / pred / moment-match)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("HF_HOME", "/project/peilab/why/cache/eeg-brainit/hf")
os.environ.setdefault("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub")

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eeg_brainit.models.eeg2fmri import retrieval_metrics
from eeg_brainit.utils.config import ensure_dirs
from train_nod_cascade import (
    _load_decoder,
    _match_moments,
    _split_by_image_id,
    build_neurobolt,
    load_multiclip_subjects,
)


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / n


def _predict_fmri(model, eeg: np.ndarray, device, bs: int = 128) -> np.ndarray:
    outs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(eeg), bs):
            x = torch.from_numpy(eeg[i : i + bs]).to(device)
            outs.append(model(x)["fmri_pred"].float().cpu().numpy())
    return np.concatenate(outs, 0).astype(np.float32)


def _sample_dim_corr(pred: np.ndarray, gt: np.ndarray) -> dict:
    sc = np.array([np.corrcoef(pred[i], gt[i])[0, 1] for i in range(len(pred))], dtype=np.float64)
    dc = np.array([np.corrcoef(pred[:, d], gt[:, d])[0, 1] for d in range(gt.shape[1])], dtype=np.float64)
    return {
        "sample_corr_mean": float(np.nanmean(sc)),
        "sample_corr_median": float(np.nanmedian(sc)),
        "sample_corr_p10": float(np.nanpercentile(sc, 10)),
        "sample_corr_p90": float(np.nanpercentile(sc, 90)),
        "dim_corr_mean": float(np.nanmean(dc)),
        "dim_corr_median": float(np.nanmedian(dc)),
        "pred_std_mean": float(pred.std(axis=0).mean()),
        "gt_std_mean": float(gt.std(axis=0).mean()),
    }


def _offdiag_mean_cos(x: np.ndarray, rng: np.random.Generator, max_n: int = 800) -> float:
    n = min(len(x), max_n)
    idx = rng.choice(len(x), size=n, replace=False)
    xn = _l2(x[idx])
    sim = xn @ xn.T
    off = ~np.eye(n, dtype=bool)
    return float(sim[off].mean())


def _category_structure(pred: np.ndarray, gt: np.ndarray, ids: list[str], seed: int = 0) -> dict:
    """ImageNet synset = first token of image_id (nXXXXXXXX)."""
    cats = np.array([i.split(":", 1)[-1].split("_", 1)[0] for i in ids])
    by: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(cats):
        by[c].append(i)
    keys = [k for k, v in by.items() if len(v) >= 2]
    gn = _l2(gt)
    pn = _l2(pred)
    within_gt, within_pr = [], []
    for c in keys:
        ix = by[c]
        for a, b in zip(ix[:-1], ix[1:]):
            within_gt.append(float(gn[a] @ gn[b]))
            within_pr.append(float(pn[a] @ pn[b]))
    rng = np.random.default_rng(seed)
    between_gt, between_pr = [], []
    n_pair = max(len(within_gt), 1)
    for _ in range(n_pair):
        a, b = rng.integers(0, len(ids), size=2)
        while cats[a] == cats[b]:
            a, b = rng.integers(0, len(ids), size=2)
        between_gt.append(float(gn[a] @ gn[b]))
        between_pr.append(float(pn[a] @ pn[b]))
    return {
        "n_cats_with_ge2": len(keys),
        "n_within_pairs": len(within_gt),
        "within_cos_gt": float(np.mean(within_gt)) if within_gt else float("nan"),
        "between_cos_gt": float(np.mean(between_gt)) if between_gt else float("nan"),
        "cat_gap_gt": float(np.mean(within_gt) - np.mean(between_gt)) if within_gt else float("nan"),
        "within_cos_pred": float(np.mean(within_pr)) if within_pr else float("nan"),
        "between_cos_pred": float(np.mean(between_pr)) if between_pr else float("nan"),
        "cat_gap_pred": float(np.mean(within_pr) - np.mean(between_pr)) if within_pr else float("nan"),
    }


def _ridge_probe(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    lam: float = 1.0,
) -> dict:
    Xtr = torch.from_numpy(x_tr).float()
    Ytr = torch.from_numpy(y_tr).float()
    Xva = torch.from_numpy(x_va).float()
    Yva = torch.from_numpy(y_va).float()
    mx, my = Xtr.mean(0), Ytr.mean(0)
    Xtr, Ytr = Xtr - mx, Ytr - my
    Xva, Yva = Xva - mx, Yva - my
    d = Xtr.shape[1]
    a = Xtr.T @ Xtr + lam * torch.eye(d)
    w = torch.linalg.solve(a, Xtr.T @ Ytr)
    pred = Xva @ w
    ss_res = ((pred - Yva) ** 2).sum(0)
    ss_tot = ((Yva - Yva.mean(0)) ** 2).sum(0).clamp_min(1e-8)
    r2 = float((1.0 - ss_res / ss_tot).mean())
    p = F.normalize(pred + my, dim=-1)
    t = F.normalize(Yva + my, dim=-1)
    sim = p @ t.T
    n = sim.size(0)
    top1 = (sim.argmax(1) == torch.arange(n)).float().mean().item()
    top5 = (sim.topk(min(5, n), dim=1).indices == torch.arange(n).unsqueeze(1)).any(1).float().mean().item()
    return {
        "r2_mean": r2,
        "top1": float(top1),
        "top5": float(top5),
        "chance_top1": 1.0 / max(n, 1),
        "n": int(n),
    }


def _moment_match_np(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    mu_p, sd_p = pred.mean(0), pred.std(0).clip(min=1e-4)
    mu_r, sd_r = ref.mean(0), ref.std(0).clip(min=1e-4)
    return ((pred - mu_p) / sd_p * sd_r + mu_r).astype(np.float32)


def _decoder_retrieval(decoder, fmri: np.ndarray, clip: np.ndarray, device, bs: int = 64) -> dict:
    emb = []
    decoder.eval()
    with torch.no_grad():
        for i in range(0, len(fmri), bs):
            x = torch.from_numpy(fmri[i : i + bs]).to(device)
            emb.append(decoder(x)["clip_emb"].float().cpu())
    pred_c = torch.cat(emb, 0)
    tgt = torch.from_numpy(clip)
    return retrieval_metrics(pred_c, tgt)


def _verdict(report: dict) -> dict:
    """Go/no-go for cascade Stage B based on Phase-0 gates."""
    chance = float(report["chance_top1"])
    dec = report["stage_a_decoder"]
    ridge = report["ridge_probe"]
    cat = report["category_structure"]

    pred_t1 = float(dec["pred"]["top1"])
    mm_t1 = float(dec["pred_moment_match"]["top1"])
    gt_t1 = float(dec["gt"]["top1"])
    ridge_pred = float(ridge["pred"]["top1"])
    ridge_gt = float(ridge["gt"]["top1"])
    cat_gap_pred = float(cat["cat_gap_pred"])
    cat_gap_gt = float(cat["cat_gap_gt"])

    # Soft absolute floors (sub-01 style n≈400 → chance 0.25%; multi-subj n≈2400 → 0.04%)
    abs_floor = max(5.0 * chance, 0.02)  # ≥5×chance or 2%
    strong_floor = max(20.0 * chance, 0.05)

    best_pred_t1 = max(pred_t1, mm_t1, ridge_pred)
    reasons = []
    if gt_t1 < max(10.0 * chance, 0.05):
        reasons.append("GT decoder ceiling too weak; fix targets/decoder before cascade")
    if best_pred_t1 < abs_floor:
        reasons.append(
            f"pred CLIP top1={best_pred_t1*100:.2f}% < floor {abs_floor*100:.2f}% "
            "(no decodable semantics in Phase-1 pred)"
        )
    if ridge_gt > 5.0 * chance and ridge_pred < 2.0 * chance:
        reasons.append("ridge probe: GT→CLIP works but pred→CLIP ≈ chance")
    if cat_gap_gt > 0.02 and cat_gap_pred < 0.005:
        reasons.append("category geometry present in GT but absent in pred")

    if not reasons and best_pred_t1 >= strong_floor:
        decision = "GO"
        summary = "Phase-1 pred shows usable CLIP-decodable signal; cascade Stage B allowed"
    elif not reasons and best_pred_t1 >= abs_floor:
        decision = "CONDITIONAL"
        summary = "Weak but above-chance pred signal; cascade only with light bridge loss, not hard CLIP"
    else:
        decision = "NO-GO"
        summary = "Do not run cascade Stage B; rebuild Phase-1 with CLIP-informed targets / EEG→CLIP baseline"

    return {
        "decision": decision,
        "summary": summary,
        "best_pred_clip_top1": best_pred_t1,
        "gt_clip_top1": gt_t1,
        "abs_floor": abs_floor,
        "strong_floor": strong_floor,
        "reasons": reasons,
    }


def run_one(
    *,
    label: str,
    eeg: np.ndarray,
    fmri: np.ndarray,
    clip: np.ndarray,
    ids: list[str],
    ch_names: list[str],
    phase1_ckpt: Path,
    decoder_ckpt: Path,
    device: torch.device,
    val_frac: float,
    seed: int,
    ridge_train_cap: int,
) -> dict:
    train_idx, val_idx = _split_by_image_id(ids, val_frac, seed)
    print(f"[{label}] n={len(ids)} train={len(train_idx)} val={len(val_idx)} rois={fmri.shape[1]}")

    ck = torch.load(phase1_ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    model = build_neurobolt(ch_names, fmri.shape[1], cfg.get("eeg2fmri", {}), device, heads_only=True)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    print(f"[{label}] Phase-1 load missing={len(missing)} unexpected={len(unexpected)}")

    print(f"[{label}] predicting fMRI...")
    pred_all = _predict_fmri(model, eeg, device)
    pred_va = pred_all[val_idx]
    gt_va = fmri[val_idx]
    clip_va = clip[val_idx]
    ids_va = [ids[i] for i in val_idx]

    rng = np.random.default_rng(seed)
    geom = _sample_dim_corr(pred_va, gt_va)
    geom["pred_offdiag_cos"] = _offdiag_mean_cos(pred_va, rng)
    geom["gt_offdiag_cos"] = _offdiag_mean_cos(gt_va, rng)
    cat = _category_structure(pred_va, gt_va, ids_va, seed=seed)

    ret_f = retrieval_metrics(torch.from_numpy(pred_va), torch.from_numpy(gt_va))

    # Ridge on a capped train subset for speed
    tr = list(train_idx)
    rng.shuffle(tr)
    tr = tr[: min(ridge_train_cap, len(tr))]
    pred_tr, gt_tr, clip_tr = pred_all[tr], fmri[tr], clip[tr]
    pred_mm_tr = _moment_match_np(pred_tr, gt_tr)
    pred_mm_va = _moment_match_np(pred_va, gt_va)
    ridge = {
        "gt": _ridge_probe(gt_tr, clip_tr, gt_va, clip_va),
        "pred": _ridge_probe(pred_tr, clip_tr, pred_va, clip_va),
        "pred_moment_match": _ridge_probe(pred_mm_tr, clip_tr, pred_mm_va, clip_va),
        "train_n": len(tr),
        "lam": 1.0,
    }

    decoder = _load_decoder(decoder_ckpt, device)
    # Match moments using val GT batch stats (same helper as cascade)
    with torch.no_grad():
        pred_t = torch.from_numpy(pred_va).to(device)
        gt_t = torch.from_numpy(gt_va).to(device)
        pred_mm_t = _match_moments(pred_t, gt_t).cpu().numpy()
    dec = {
        "gt": _decoder_retrieval(decoder, gt_va, clip_va, device),
        "pred": _decoder_retrieval(decoder, pred_va, clip_va, device),
        "pred_moment_match": _decoder_retrieval(decoder, pred_mm_t, clip_va, device),
        "decoder_ckpt": str(decoder_ckpt),
    }

    report = {
        "label": label,
        "n_total": len(ids),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_rois": int(fmri.shape[1]),
        "chance_top1": 1.0 / max(len(val_idx), 1),
        "phase1_ckpt": str(phase1_ckpt),
        "geometry": geom,
        "category_structure": cat,
        "fmri_retrieval_pred_to_gt": {k: float(v) for k, v in ret_f.items()},
        "ridge_probe": ridge,
        "stage_a_decoder": dec,
    }
    report["verdict"] = _verdict(report)
    return report


def _print_report(report: dict) -> None:
    v = report["verdict"]
    g = report["geometry"]
    c = report["category_structure"]
    r = report["ridge_probe"]
    d = report["stage_a_decoder"]
    print("\n" + "=" * 72)
    print(f"PHASE-0 [{report['label']}] n_val={report['n_val']} chance_top1={report['chance_top1']*100:.3f}%")
    print("-" * 72)
    print(
        f"geometry: sample_corr={g['sample_corr_mean']:.4f} dim_corr={g['dim_corr_mean']:.4f} "
        f"std pred/gt={g['pred_std_mean']:.4f}/{g['gt_std_mean']:.4f}"
    )
    print(
        f"category: gap_gt={c['cat_gap_gt']:.4f} gap_pred={c['cat_gap_pred']:.4f} "
        f"(within-between cos)"
    )
    fr = report["fmri_retrieval_pred_to_gt"]
    print(f"fmri retrieval pred→gt: top1={fr['top1']*100:.2f}% top5={fr['top5']*100:.2f}%")
    print(
        f"ridge→CLIP: gt_t1={r['gt']['top1']*100:.2f}% pred_t1={r['pred']['top1']*100:.2f}% "
        f"mm_t1={r['pred_moment_match']['top1']*100:.2f}% "
        f"| r2 gt/pred={r['gt']['r2_mean']:.4f}/{r['pred']['r2_mean']:.4f}"
    )
    print(
        f"Stage-A decoder: gt_t1={d['gt']['top1']*100:.2f}% pred_t1={d['pred']['top1']*100:.2f}% "
        f"mm_t1={d['pred_moment_match']['top1']*100:.2f}%"
    )
    print(f"VERDICT: {v['decision']} — {v['summary']}")
    for reason in v["reasons"]:
        print(f"  - {reason}")
    print("=" * 72 + "\n")


def main():
    parser = argparse.ArgumentParser(description="NOD Phase-0 cascade go/no-go diagnosis")
    parser.add_argument("--pairs-root", default="data/nod/processed/classmean")
    parser.add_argument("--clip-dir", default="data/nod/processed/clip_vit_h14_all")
    parser.add_argument("--fallback-clip-dir", default="data/nod/processed/clip_vit_h14")
    parser.add_argument("--phase1-ckpt", default="outputs/nod_eeg2fmri/neurobolt_classmean_v3/checkpoints/best.pt")
    parser.add_argument("--decoder-ckpt", default="outputs/nod_cascade/neurobolt_bit_v1/decoder_best.pt")
    parser.add_argument("--output-dir", default="outputs/nod_cascade/phase0_diag")
    parser.add_argument(
        "--subjects",
        default="sub-01",
        help="Comma-separated subject list, or 'all' for every subject under pairs-root",
    )
    parser.add_argument("--also-all-subjects", action="store_true", help="Also run pooled 6-subject diagnosis")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge-train-cap", type=int, default=4000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs_root = ROOT / args.pairs_root
    clip_dir = ROOT / args.clip_dir
    if not (clip_dir / "index.json").is_file():
        clip_dir = ROOT / args.fallback_clip_dir
        print(f"[WARN] full CLIP missing, fallback {clip_dir}")
    out_dir = ROOT / args.output_dir
    ensure_dirs(out_dir)

    phase1 = ROOT / args.phase1_ckpt
    decoder_ck = ROOT / args.decoder_ckpt
    for p in (phase1, decoder_ck):
        if not p.is_file():
            raise FileNotFoundError(p)

    # Load all subjects once, then slice.
    all_names, eeg, fmri, clip, ids, ch_names = load_multiclip_subjects(pairs_root, clip_dir, max_subjects=0)
    print(f"[INFO] loaded subjects={all_names} device={device}")

    reports = []

    def select(subjects: list[str] | None, label: str):
        if subjects is None:
            return eeg, fmri, clip, ids, label
        want = set(subjects)
        mask = np.array([i.split(":", 1)[0] in want for i in ids], dtype=bool)
        if not mask.any():
            raise RuntimeError(f"no samples for subjects={subjects}")
        return eeg[mask], fmri[mask], clip[mask], [x for x, m in zip(ids, mask) if m], label

    if args.subjects.strip().lower() == "all":
        runs = [(None, "all_subjects")]
    else:
        subs = [s.strip() for s in args.subjects.split(",") if s.strip()]
        runs = [(subs, "+".join(subs))]
        if args.also_all_subjects:
            runs.append((None, "all_subjects"))

    for subjects, label in runs:
        e, f, c, i, lab = select(subjects, label)
        report = run_one(
            label=lab,
            eeg=e,
            fmri=f,
            clip=c,
            ids=i,
            ch_names=ch_names,
            phase1_ckpt=phase1,
            decoder_ckpt=decoder_ck,
            device=device,
            val_frac=args.val_frac,
            seed=args.seed,
            ridge_train_cap=args.ridge_train_cap,
        )
        _print_report(report)
        reports.append(report)
        (out_dir / f"phase0_{lab}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = {
        "runs": [
            {
                "label": r["label"],
                "decision": r["verdict"]["decision"],
                "summary": r["verdict"]["summary"],
                "best_pred_clip_top1": r["verdict"]["best_pred_clip_top1"],
                "gt_clip_top1": r["verdict"]["gt_clip_top1"],
                "sample_corr": r["geometry"]["sample_corr_mean"],
                "ridge_pred_top1": r["ridge_probe"]["pred"]["top1"],
                "decoder_pred_top1": r["stage_a_decoder"]["pred"]["top1"],
                "decoder_gt_top1": r["stage_a_decoder"]["gt"]["top1"],
            }
            for r in reports
        ],
        "overall_decision": (
            "NO-GO"
            if any(r["verdict"]["decision"] == "NO-GO" for r in reports)
            else (
                "CONDITIONAL"
                if any(r["verdict"]["decision"] == "CONDITIONAL" for r in reports)
                else "GO"
            )
        ),
    }
    (out_dir / "phase0_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[OK] wrote", out_dir / "phase0_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
