#!/usr/bin/env python3
"""Rebuild ACK predicted prompts using CLIP-space q (Phase-0 fix).

Bug in v1: names came from bare shared_r · CLIP-image (top1=0), which hurts
generation. Correct naming lives on the semantic readout:

    name_i = argmax_j  l2(q_i) · l2(G_test_img_j)

Optional gate uses q's top1-top2 margin. Sinkhorn also runs on q·G_test.
Does not retrain. Archives any z-based pred prompts if present.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocf_train import l2n  # noqa: E402
from ack_heads_train import (  # noqa: E402
    concept_of,
    hcma_prompt,
    retrieval_bank,
    sinkhorn,
    write_prompts,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads-dir", type=str, required=True,
                    help="existing ack heads export (has conds/ip_q_test.npy)")
    ap.add_argument("--out-prompts", type=str, required=True)
    ap.add_argument("--captions-dir", type=str, default=str(NB_ROOT / "outputs/g2/captions"))
    ap.add_argument("--clip-img-dir", type=str, default=str(NB_ROOT / "outputs/gem/cond_cache"))
    ap.add_argument("--z-root", type=str, default=str(NB_ROOT / "outputs/ocf/intra_z"))
    ap.add_argument("--test-subject", type=int, default=8)
    ap.add_argument("--uck-q", type=str, default="",
                    help="optional UCK ip_q_test.npy for naming ablation")
    ap.add_argument("--oracle-prompts", type=str,
                    default=str(NB_ROOT / "outputs/g2f/prompts/prompts_oracle.json"))
    ap.add_argument("--phrases-json", type=str,
                    default=str(NB_ROOT / "outputs/nda_ss/sub-08/clip_text/train/concept_phrases.json"))
    ap.add_argument("--mu-npy", type=str, default="",
                    help="optional neural book for neural_nn prompts")
    ap.add_argument("--gate-margin", type=float, default=0.02)
    ap.add_argument("--gate-margin-hi", type=float, default=0.03)
    ap.add_argument("--sinkhorn-tau", type=float, default=0.07)
    ap.add_argument("--report", type=str, default="")
    args = ap.parse_args()

    heads = Path(args.heads_dir)
    q_path = heads / "conds" / "ip_q_test.npy"
    if not q_path.is_file():
        raise SystemExit(f"[FATAL] missing {q_path}")
    q = np.load(q_path).astype(np.float32)

    sid = f"{args.test_subject:02d}"
    sub = f"sub-{sid}"
    zte = l2n(np.load(Path(args.z_root) / sub / "shared_r_test.npy").astype(np.float32))
    caps_te = [json.loads(l) for l in (Path(args.captions_dir) / "captions_test.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    con_te = [concept_of(c["path"]) for c in caps_te]
    if len(con_te) != len(q) or len(zte) != len(q):
        raise SystemExit(f"[FATAL] rows q={len(q)} z={len(zte)} cap={len(con_te)}")

    clip_te = l2n(np.load(Path(args.clip_img_dir) / "clip_img1024_test.npy").astype(np.float32))
    if len(clip_te) != len(q):
        raise SystemExit(f"[FATAL] clip test {len(clip_te)} vs q {len(q)}")

    out_p = Path(args.out_prompts)
    out_p.mkdir(parents=True, exist_ok=True)

    # archive previous z-based preds if they sit beside us / in heads
    for src_dir in (heads / "prompts", out_p):
        bad = src_dir / "prompts_pred.json"
        if bad.is_file() and not (src_dir / "prompts_pred_z_bad.json").is_file():
            # only archive if it looks like the broken z run (optional marker)
            shutil.copy2(bad, src_dir / "prompts_pred_z_bad.json")

    ret_z = retrieval_bank(zte, clip_te, con_te)   # diagnostic only
    ret_q = retrieval_bank(q, clip_te, con_te)     # PRIMARY naming

    sim_q = l2n(q) @ clip_te.T
    sink_idx = sinkhorn(sim_q, iters=60, tau=args.sinkhorn_tau)
    sink_names = [con_te[int(i)] for i in sink_idx]
    sink_top1 = float(np.mean([a == b for a, b in zip(sink_names, con_te)]))

    pred = ret_q["pred_names"]
    margins = ret_q["margins"]

    def gate(names, margs, thr: float):
        return [(n if float(m) >= thr else "object") for n, m in zip(names, margs)]

    gated = gate(pred, margins, args.gate_margin)
    gated_hi = gate(pred, margins, args.gate_margin_hi)
    n_gate = int(sum(1 for n in gated if n == "object"))
    n_gate_hi = int(sum(1 for n in gated_hi if n == "object"))

    # neural NN from mu if available (train names; cannot hit test classes)
    neural_names = ["object"] * len(pred)
    mu_p = Path(args.mu_npy) if args.mu_npy else heads / "proto" / "mu_all.npy"
    phr_p = Path(args.phrases_json)
    if mu_p.is_file() and phr_p.is_file():
        mu = l2n(np.load(mu_p).astype(np.float32))
        phrases = json.loads(phr_p.read_text(encoding="utf-8"))
        topi = (zte @ mu.T).argmax(1)
        neural_names = [phrases[int(i)] for i in topi]

    write_prompts(pred, out_p / "prompts_pred.json")
    write_prompts(gated, out_p / "prompts_pred_gate.json")
    write_prompts(gated_hi, out_p / "prompts_pred_gate_hi.json")
    write_prompts(sink_names, out_p / "prompts_sinkhorn.json")
    write_prompts(neural_names, out_p / "prompts_neural_nn.json")
    write_prompts(pred, out_p / "prompts_empty.json", empty=True)
    write_prompts(pred, out_p / "prompts_deploy.json", use_generic=True)

    oracle_src = Path(args.oracle_prompts)
    if oracle_src.is_file():
        shutil.copy2(oracle_src, out_p / "prompts_oracle.json")

    # UCK-q naming ablation
    uck_block = {}
    uck_q_path = Path(args.uck_q) if args.uck_q else Path()
    if uck_q_path.is_file():
        uq = np.load(uck_q_path).astype(np.float32)
        if len(uq) == len(q):
            ret_uq = retrieval_bank(uq, clip_te, con_te)
            write_prompts(ret_uq["pred_names"], out_p / "prompts_pred_uckq.json")
            g_uq = gate(ret_uq["pred_names"], ret_uq["margins"], args.gate_margin)
            write_prompts(g_uq, out_p / "prompts_pred_uckq_gate.json")
            uck_block = {
                "test200_top1_uck_q": ret_uq["top1"],
                "mean_margin_uck_q": ret_uq["mean_margin"],
                "n_gated_uck_q": int(sum(1 for n in g_uq if n == "object")),
            }

    # also keep ip naming diagnostic if present
    ip_block = {}
    ip_p = heads / "conds" / "ip_ack_test.npy"
    if ip_p.is_file():
        ret_ip = retrieval_bank(np.load(ip_p).astype(np.float32), clip_te, con_te)
        ip_block = {"test200_top1_ip": ret_ip["top1"], "mean_margin_ip": ret_ip["mean_margin"]}

    cond_dir = out_p.parent / "conds"
    cond_dir.mkdir(parents=True, exist_ok=True)
    np.save(cond_dir / "pred_q_idx.npy", ret_q["top1_idx"])
    np.save(cond_dir / "sinkhorn_q_idx.npy", sink_idx)
    np.save(cond_dir / "pred_q_margins.npy", margins)
    (cond_dir / "pred_names_q.json").write_text(json.dumps({
        "query": "ack_q",
        "pred": pred,
        "true": con_te,
        "gated": gated,
        "gated_hi": gated_hi,
        "sinkhorn": sink_names,
        "neural_nn": neural_names,
        "gate_margin": args.gate_margin,
        "gate_margin_hi": args.gate_margin_hi,
    }, indent=1), encoding="utf-8")

    rep = {
        "pipeline": "ack_rebuild_prompts_q",
        "subject": sub,
        "fix": "naming uses l2(q)@l2(G_test_img); z retrieval kept as diagnostic only",
        "test200_top1_z_diagnostic": ret_z["top1"],
        "test200_top1_q": ret_q["top1"],
        "test200_top1_sinkhorn_q": sink_top1,
        "test200_chance": ret_q["chance"],
        "mean_margin_q": ret_q["mean_margin"],
        "frac_margin_ge_gate": float((margins >= args.gate_margin).mean()),
        "frac_margin_ge_gate_hi": float((margins >= args.gate_margin_hi).mean()),
        "n_gated_to_object": n_gate,
        "n_gated_to_object_hi": n_gate_hi,
        "gate_margin": args.gate_margin,
        "gate_margin_hi": args.gate_margin_hi,
        "n_correct_pred": int(sum(a == b for a, b in zip(pred, con_te))),
        "n_correct_gated_among_kept": int(sum(
            a == b for a, b, g in zip(pred, con_te, gated) if g != "object")),
        "prompts_dir": str(out_p),
        **uck_block,
        **ip_block,
    }
    rep_path = Path(args.report) if args.report else out_p / "rebuild_report.json"
    rep_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    print(f"[ack] wrote prompts -> {out_p}")
    print(f"[ack] sample: true={con_te[0]!r} pred_q={pred[0]!r} "
          f"correct={pred[0]==con_te[0]} margin={float(margins[0]):.4f}")


if __name__ == "__main__":
    main()
