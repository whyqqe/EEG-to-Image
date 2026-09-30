#!/usr/bin/env python3
"""LG-SELECT: validation of per-sample gating by composing EXISTING grid images.

Key idea (cheap upper-bound test for LG-Gate):
  A real "gated generation" would set CN_i per sample from the router u_i. We
  cannot afford to re-run the diffuser for many policies, but we ALREADY have
  fixed-CN images for every sample (CN=0 sdedit-LL, and c025/c032/c040 at
  strength 0.82 / 0.86). So for a policy "u_i >= thresh => strong CN, else
  CN off", the per-sample output is exactly one of those pre-generated images.
  Composing them (index-aligned symlinks) gives the set-level standard-7 result
  the policy WOULD have produced -- no re-generation needed.

Rows produced:
  lgsel_r2_s082 / lgsel_o2_s082 : 2-level router / oracle at strength 0.82
         u>=med -> intra_hs_c040_s082,  u<med -> sdedit_ll_intra (CN=0)
  lgsel_r2_s086 / lgsel_o2_s086 : same, strength 0.86
         u>=med -> intra_hs_c040_s086,  u<med -> intra_hs_c025_s086
  lgsel_r4_s082 / lgsel_o4_s082 : 4-level (sdedit/c025/c032/c040 by u quartile)

  r* = router u_hat (deployable). o* = oracle u_true (GT-depth upper bound).

Writes a std7 manifest + symlink trees under <out>/generation/<tag>/generated.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def build_row(out: Path, tag: str, u: np.ndarray, levels: list[tuple[float, Path]]) -> str:
    """levels: [(bucket_upper_exclusive, src_dir), ...] sorted ascending; the last
    level's threshold is ignored (upper bound = +inf). bucket(u) = digitize over the
    first len(levels)-1 thresholds -> 0..len(levels)-1, mapping to srcs in order."""
    srcs = [lv[1] for lv in levels]
    edges = np.asarray([lv[0] for lv in levels[:-1]], dtype=np.float64)  # k-1 cut points
    sel = np.clip(np.digitize(u, edges), 0, len(srcs) - 1).astype(np.int32)  # (N,)
    dst = out / "generation" / tag / "generated"
    dst.mkdir(parents=True, exist_ok=True)
    n = len(u)
    for i in range(n):
        src_p = Path(srcs[int(sel[i])]) / f"{i:03d}.png"
        if not src_p.is_file():
            raise FileNotFoundError(f"{src_p} (tag={tag} i={i})")
        link = dst / f"{i:03d}.png"
        if link.is_symlink() or link.exists():
            link.unlink()
        try:
            link.symlink_to(src_p)
        except OSError:
            shutil.copyfile(src_p, link)
    counts = {Path(s).name: int((sel == k).sum()) for k, s in enumerate(srcs)}
    print(f"[row {tag}] n={n} sel={counts}")
    return tag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--u-hat", type=str, required=True)
    ap.add_argument("--u-true", type=str, required=True)
    ap.add_argument("--grid-root", type=str, required=True, help="intra_hcma_s/sub-08/generation")
    ap.add_argument("--output-dir", type=str, required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    grid = Path(args.grid_root).resolve()
    # candidate image dirs (index-aligned 0..199)
    D = {
        "sdedit082": grid / "sdedit_ll_intra/generated",
        "c025_082": grid / "intra_hs_c025_s082/generated",
        "c032_082": grid / "intra_hs_c032_s082/generated",
        "c040_082": grid / "intra_hs_c040_s082/generated",
        "c025_086": grid / "intra_hs_c025_s086/generated",
        "c040_086": grid / "intra_hs_c040_s086/generated",
    }
    for k, v in D.items():
        if not (v / "199.png").is_file():
            raise FileNotFoundError(f"candidate {k}: {v}/199.png")

    u_hat = np.load(args.u_hat).astype(np.float32).reshape(-1)
    u_true = np.load(args.u_true).astype(np.float32).reshape(-1)
    n = len(u_hat)
    assert n == 200 and len(u_true) == n, (n, len(u_true))

    man = {"protocol": "standard7", "rows": [], "avg_rows": []}
    q25, q50, q75 = np.percentile(u_hat, [25, 50, 75])
    qt25, qt50, qt75 = np.percentile(u_true, [25, 50, 75])

    def add(tag, display):
        man["rows"].append({"tag": tag, "display": display,
                            "gen_dir": str(out / "generation" / tag / "generated")})

    # ---- 2-level router / oracle at strength 0.82 ----
    build_row(out, "lgsel_r2_s082", u_hat, [(q50, D["sdedit082"]), (np.inf, D["c040_082"])])
    add("lgsel_r2_s082", "lgsel router 2lvl s082 (u>=med->c040 else CN=0)")
    build_row(out, "lgsel_o2_s082", u_true, [(qt50, D["sdedit082"]), (np.inf, D["c040_082"])])
    add("lgsel_o2_s082", "lgsel oracle 2lvl s082 (upper bound)")

    # ---- 4-level router / oracle at strength 0.82 ----
    build_row(out, "lgsel_r4_s082", u_hat,
              [(q25, D["sdedit082"]), (q50, D["c025_082"]), (q75, D["c032_082"]), (np.inf, D["c040_082"])])
    add("lgsel_r4_s082", "lgsel router 4lvl s082 (u quartile -> CN 0/.25/.32/.40)")
    build_row(out, "lgsel_o4_s082", u_true,
              [(qt25, D["sdedit082"]), (qt50, D["c025_082"]), (qt75, D["c032_082"]), (np.inf, D["c040_082"])])
    add("lgsel_o4_s082", "lgsel oracle 4lvl s082 (upper bound)")

    # ---- 2-level at strength 0.86 (strongest strength family) ----
    build_row(out, "lgsel_r2_s086", u_hat, [(q50, D["c025_086"]), (np.inf, D["c040_086"])])
    add("lgsel_r2_s086", "lgsel router 2lvl s086 (u>=med->c040 else c025)")
    build_row(out, "lgsel_o2_s086", u_true, [(qt50, D["c025_086"]), (np.inf, D["c040_086"])])
    add("lgsel_o2_s086", "lgsel oracle 2lvl s086 (upper bound)")

    # ---- sanity: verify symlink trees ----
    for row in man["rows"]:
        gd = Path(row["gen_dir"])
        ok = all((gd / f"{i:03d}.png").is_file() for i in range(n))
        if not ok:
            raise RuntimeError(f"incomplete tree {gd}")
    (out / "manifest_lgsel.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"[OK] {len(man['rows'])} rows -> {out / 'manifest_lgsel.json'}")


if __name__ == "__main__":
    main()
