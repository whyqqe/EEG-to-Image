#!/usr/bin/env python3
"""Zero-shot inter-subject encode: load a cross-subject MG-Flow ckpt and run
pure inference (NO per-subject FT) on one subject's z_ret to produce semantic
embeds identical in format to mg_flow_train.py exports.

Used to answer: "if we drop per-subject fine-tuning, does the pure inter model
still meet SOTA?" — we decode these embeds and run standard-7 eval.

Outputs (mirrors mg_flow_train save block):
  embeds/blend_nda_cfm_f_{a25,a40,a55}_test.npy
  embeds/z_mg_gated_test.npy, embeds/gate_test.npy
  embeds/z_s_c/z_s_f/z_cfm_c/z_cfm_f/z_c2f_test.npy
  inter_encode_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))

from mg_flow_modules import MGFlowModel, l2_t  # noqa: E402


def l2(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)).astype(np.float32)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="cross-subject (LOSO) best.pt")
    ap.add_argument("--z-ret-test", type=str, required=True, help="holdout subject z_ret (200,D)")
    ap.add_argument("--nda-decode-test", type=str, required=True, help="holdout subject rag memory (200,1024)")
    ap.add_argument("--clip-img-test", type=str, required=True, help="(200,D) image CLIP gallery")
    ap.add_argument("--t-coarse-test", type=str, required=True)
    ap.add_argument("--t-fine-test", type=str, required=True)
    ap.add_argument("--text-concept-test", type=str, required=True, help="(200,D) class text")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--ode-steps", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    out = Path(args.output_dir)
    (out / "embeds").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    z_te = l2(np.load(args.z_ret_test))
    img_te = l2(np.load(args.clip_img_test))
    tc_te = l2(np.load(args.t_coarse_test))
    tf_te = l2(np.load(args.t_fine_test))
    nda_te = l2(np.load(args.nda_decode_test))
    text_cls_te = l2(np.load(args.text_concept_test))
    assert z_te.shape[0] == nda_te.shape[0] == img_te.shape[0], (z_te.shape, nda_te.shape, img_te.shape)
    n = z_te.shape[0]
    dim = z_te.shape[1]

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = MGFlowModel(ret_dim=dim, clip_dim=img_te.shape[1]).to(device)
    if isinstance(ck, dict) and "model_delta" in ck:
        # subject-FT delta: merge onto base_ckpt (shared full ckpt)
        base = ck.get("base_ckpt") or ""
        if not base:
            raise ValueError(f"delta ckpt {args.ckpt} missing base_ckpt")
        base_ck = torch.load(base, map_location="cpu", weights_only=False)
        base_state = base_ck["model"] if isinstance(base_ck, dict) and "model" in base_ck else base_ck
        model.load_state_dict(base_state, strict=True)
        cur = model.state_dict()
        cur.update(ck["model_delta"])
        model.load_state_dict(cur, strict=True)
        print(f"[OK] loaded delta {args.ckpt} on base {base}")
    else:
        state = ck.get("model", ck.get("model_state_dict", ck.get("state_dict", ck)))
        try:
            model.load_state_dict(state, strict=True)
        except Exception:
            # tolerate partial / prefix mismatch — strip prefixes
            cleaned = {}
            for k, v in state.items():
                nk = k
                for pre in ["model.", "module."]:
                    if nk.startswith(pre):
                        nk = nk[len(pre):]
                cleaned[nk] = v
            missing, unexpected = model.load_state_dict(cleaned, strict=False)
            print(f"[WARN] non-strict load; missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    # ---- coarse/fine bundle (same as decode_bundle)
    bundles = {k: [] for k in ["z_s_c", "z_s_f", "z_cfm_c", "z_cfm_f", "z_c2f"]}
    gates, gated, cfm_f = [], [], []
    for i in range(0, n, 256):
        zr = torch.from_numpy(z_te[i : i + 256]).to(device)
        c, f = model.encode(zr)
        bundles["z_s_c"].append(c.cpu().numpy())
        bundles["z_s_f"].append(f.cpu().numpy())
        zcc = model.cfm_c.decode(c, steps=args.ode_steps)
        zcf = model.cfm_f.decode(f, steps=args.ode_steps)
        zc2f = model.cfm_c2f.decode(c, steps=args.ode_steps)
        bundles["z_cfm_c"].append(zcc.cpu().numpy())
        bundles["z_cfm_f"].append(zcf.cpu().numpy())
        bundles["z_c2f"].append(zc2f.cpu().numpy())
        # gated gen (same as train save)
        tc = torch.from_numpy(tc_te[i : i + 256]).to(device)
        nda = torch.from_numpy(nda_te[i : i + 256]).to(device)
        conf = (l2_t(c) * l2_t(tc)).sum(-1, keepdim=True)
        z_ref = l2_t(0.5 * zcf + 0.5 * zc2f)
        delta = model.to_gen(z_ref)
        g = torch.sigmoid(model.gate_mlp(torch.cat([l2_t(z_ref), l2_t(nda), conf], dim=-1)))
        z_hat = l2_t(nda + g * delta)
        gates.append(g.cpu().numpy())
        gated.append(z_hat.cpu().numpy())
        cfm_f.append(zcf.cpu().numpy())

    for k, v in bundles.items():
        np.save(out / "embeds" / f"{k}_test.npy", l2(np.concatenate(v, 0)))
    np.save(out / "embeds" / "z_mg_gated_test.npy", l2(np.concatenate(gated, 0)))
    np.save(out / "embeds" / "gate_test.npy", np.concatenate(gates, 0).astype(np.float32))
    z_cfm = l2(np.concatenate(cfm_f, 0))
    for alpha, name in [(0.25, "a25"), (0.40, "a40"), (0.55, "a55")]:
        blend = l2((1 - alpha) * nda_te + alpha * z_cfm)
        np.save(out / "embeds" / f"blend_nda_cfm_f_{name}_test.npy", blend)

    # sanity: gated CLIP cosine to GT images
    sim = l2(np.concatenate(gated, 0)) @ img_te.T
    diag = np.mean([sim[i, i] for i in range(n)])
    report = {
        "pipeline": "mg_flow_inter_encode",
        "ckpt": args.ckpt,
        "mode": "zero-shot NO per-subject FT",
        "n": n,
        "gated_self_clip_cosine": float(diag),
        "ode_steps": args.ode_steps,
    }
    (out / "inter_encode_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
