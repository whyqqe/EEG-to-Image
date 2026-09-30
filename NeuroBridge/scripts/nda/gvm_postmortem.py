#!/usr/bin/env python3
"""Post-mortem analysis of the GVM run: which condition path actually carries the
image information, and are the three towers separable at all?

Written because the generation stage never ran (the job died in the shell diagnostic
before it), so the report JSONs are the ONLY evidence available, and the question
they answer -- "is the fused condition earning its keep?" -- can be answered from
them directly by scoring every exported condition against the true image embedding
in the space the IP-Adapter actually consumes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

NB = Path("/project/peilab/why/NeuroBridge")
OUT = NB / "outputs/gvm/sub-08"
CC = NB / "outputs/gem/cond_cache"


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def rowcos(x):
    z = l2(x); g = z @ z.T; n = len(z)
    return float((g.sum() - np.trace(g)) / (n * (n - 1)))


def rowid(p, t):
    s = l2(p) @ l2(t).T
    np.fill_diagonal(s, -np.inf)
    return float((s.argmax(1) == np.arange(len(p))).mean())


T = np.load(CC / "clip_img1024_test.npy").astype(np.float32)
print("=" * 78)
print("A. EVERY EXPORTED CONDITION, SCORED IN THE IP-ADAPTER'S OWN SPACE")
print("=" * 78)
print(f"  reference: TRUE image embedding row-cos {rowcos(T):.4f} "
      f"(the diversity the generator expects)")
print()
names = ["ip_clip_test", "ip_sem_test", "ip_fused_auto_test", "ip_clip_loser_test",
         "ip_oracle_test", "ip_static_test"]
rows = []
for arm in ("full", "nofront", "noise"):
    print(f"  --- arm {arm} ---")
    for nm in names:
        p = OUT / arm / "conds" / f"{nm}.npy"
        if not p.is_file():
            continue
        P = np.load(p).astype(np.float32)
        c = float((l2(P) * l2(T)).sum(1).mean())
        r = rowcos(P)
        print(f"    {nm:<22} cos-to-true-image {c:+.4f} | row-cos {r:.4f} | "
              f"row-id {rowid(P, T):.4f}")
        if arm == "full" and nm != "ip_static_test" and nm != "ip_clip_loser_test":
            rows.append((nm, c, r))

print()
print("=" * 78)
print("B. TOWER REDUNDANCY (from each arm's own report)")
print("=" * 78)
for arm in ("full", "nofront", "noise"):
    rp = OUT / arm / "gem_report.json"
    if not rp.is_file():
        continue
    d = json.loads(rp.read_text(encoding="utf-8"))
    xm = d.get("tower_attribution", {})
    if not xm:
        continue
    print(f"  --- arm {arm} ---")
    for tgt in ("txt", "img", "vae", "nvol"):
        cells = {f"{s}->{tgt}": xm[f"{s}->{tgt}"]["test_cos"]
                 for s in ("sem", "clip", "vae") if f"{s}->{tgt}" in xm}
        if not cells:
            continue
        vals = list(cells.values())
        print(f"    target {tgt:<5} " + "  ".join(f"{k.split('->')[0]}:{v:.4f}"
                                                  for k, v in cells.items())
              + f"   spread {max(vals) - min(vals):.4f}")
    print(f"    ridge val R2: t5pool->cliptext "
          f"{d.get('ridges', {}).get('t5pool_to_cliptext_val_r2', float('nan')):.4f} | "
          f"cliptext->clipimg "
          f"{d.get('ridges', {}).get('cliptext_to_clipimg_val_r2', float('nan')):.4f}")

print()
print("=" * 78)
print("C. DOES THE FROZEN FRONT END HELP?  (paired arms, same code/targets)")
print("=" * 78)
for arm in ("full", "nofront", "noise"):
    rp = OUT / arm / "gem_report.json"
    if not rp.is_file():
        continue
    d = json.loads(rp.read_text(encoding="utf-8"))
    f = d["frozen_condition_check"]
    h = d.get("history_tail", [])
    best = h[-1] if h else {}
    print(f"  {arm:<8} pool->text {f['pool_to_clip_text_cos']:.4f} | "
          f"clip {f['clip_tower_cos']:.4f} | best ep {d['best']['epoch']} "
          f"score {d['best']['score']:.4f}")
    cc = d.get("condition_concentration", {})
    print(f"           conditions: fused row-cos {cc.get('fused_rowcos', float('nan')):.4f} "
          f"| clip {cc.get('clip_rowcos', float('nan')):.4f} "
          f"| vae {cc.get('vae_rowcos', float('nan')):.4f} "
          f"| s_code {cc.get('s_code_rowcos', float('nan')):.4f}")

print()
print("=" * 78)
print("D. VERDICT INPUTS")
print("=" * 78)
if rows:
    best_nm, best_c, _ = max(rows, key=lambda t: t[1])
    fused = [r for r in rows if "fused" in r[0]]
    print(f"  best condition in full arm : {best_nm} ({best_c:+.4f})")
    if fused:
        print(f"  the MAIN generation condition (fused) : {fused[0][1]:+.4f}")
        print(f"  -> the fused condition is {best_c - fused[0][1]:+.4f} WORSE than the "
              f"best single path")
