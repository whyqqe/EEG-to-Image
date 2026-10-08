#!/usr/bin/env python
"""Is the moments anchor erased by the normalisation?

`anchor='stats'` computes `z = stat_proj(moments(S)) + net(S)` and then `normalize_z` maps
the SUM to a constant norm. If `||net(S)||` grows much larger than `||stat_proj(moments)||`
during training, the sum's DIRECTION is almost entirely the learned part, so normalising
the sum discards the anchor -- the anchor is present in the code, absent in the function.
That would explain why the anchored arm measured no better than the unanchored one.

If so it is an implementation bug with a known fix, not a dead end: normalise the two
channels separately and combine the UNIT directions with learned weights, so neither can
swamp the other.

  python scripts/diag_anchor_swamp.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import config  # noqa: E402
from samclip.data import things_eeg  # noqa: E402
from samclip.models.subject_conditioning import SupportSetEncoder  # noqa: E402
from samclip.utils import load_config  # noqa: E402

CKPTS = {
    "C__support__stats (anchored)": "outputs/probe_subject/C__support__stats__seed2025/last.pt",
    "B__support__noanchor (control)": "outputs/probe_subject/B__support__noanchor__seed2025/last.pt",
}


def load_encoder(ckpt: str, cfg: dict, n_ch: int, n_t: int) -> SupportSetEncoder:
    anchor = cfg.get("conditioning", {}).get("support_anchor", "none")
    if "stats" in ckpt and "noanchor" not in ckpt:
        anchor = "stats"
    elif "noanchor" in ckpt:
        anchor = "none"
    enc = SupportSetEncoder(n_ch, n_t,
                            d_trial=int(cfg.get("conditioning", {}).get("support_d_trial", 128)),
                            d_z=int(cfg.get("conditioning", {}).get("d_z", 64)),
                            anchor=anchor)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    mine = {k.split("support_encoder.", 1)[1]: v for k, v in sd.items()
            if "support_encoder." in k}
    enc.load_state_dict(mine, strict=False)
    enc.eval()
    return enc


def report(tag: str, enc: SupportSetEncoder, draws: np.ndarray, n_sub: int) -> None:
    """`draws`: (n_sub * draws_per_sub, K, C, T) with subject blocks in order."""
    with torch.no_grad():
        x = torch.from_numpy(draws).float()
        b, k, c, t = x.shape
        # reproduce `forward` up to the two channels
        h = x.reshape(b * k, 1, c, t)
        h = enc.temporal(h)
        h = enc.pool(h).flatten(1)
        h = enc.proj(h).reshape(b, k, -1)
        pooled, _ = enc.attn(enc.query.expand(b, -1, -1), h, h, need_weights=False)
        net = enc.out(enc.norm(pooled.squeeze(1)))
        stat = (enc.stat_proj(enc.moments(x)) if enc.stat_proj is not None
                else torch.zeros_like(net))

    n_net, n_stat = net.norm(dim=-1), stat.norm(dim=-1)
    print(f"\n{'=' * 74}\n{tag}\n{'=' * 74}")
    if enc.stat_proj is None:
        print("  no anchor channel installed (anchor='none')")
    print(f"  ||net(S)||            mean {n_net.mean():8.4f}")
    print(f"  ||stat_proj(S)||      mean {n_stat.mean():8.4f}")
    if n_stat.mean() > 0:
        print(f"  anchor share of norm  {float(n_stat.mean() / (n_net.mean() + n_stat.mean())):.4f}"
              f"   (the learned channel is {float(n_net.mean() / max(n_stat.mean(), 1e-9)):.1f}x larger)")

    # Directional separation across subjects, per channel and for the SUM.
    def sep(v: torch.Tensor) -> float:
        un = F.normalize(v, dim=-1)
        cos = un @ un.T
        n = v.shape[0]
        blocks = [[i * (n // n_sub) + j for j in range(n // n_sub)] for i in range(n_sub)]
        same, diff = [], []
        for i in range(n):
            for j in range(i + 1, n):
                bi = [k for k, bl in enumerate(blocks) if i in bl][0]
                bj = [k for k, bl in enumerate(blocks) if j in bl][0]
                (same if bi == bj else diff).append(float(cos[i, j]))
        return float(np.mean(same) - np.mean(diff))

    print(f"  separation: net channel      {sep(net):+.6f}")
    if enc.stat_proj is not None:
        print(f"  separation: stat channel     {sep(stat):+.6f}")
    total = net + stat
    print(f"  separation: SUM (= what normalize_z then rescales) {sep(total):+.6f}"
          f"   <-- 0.0 means z_s is subject-agnostic")
    # what normalize_z actually emits
    print(f"  after normalize_z (norm pinned to 0.02*sqrt(d_z)): "
          f"{sep(F.normalize(total, dim=-1) * 0.16):+.6f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(config.CONFIGS / "loso_sub08.yaml"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    src = cfg.get("source_subjects") or [s for s in config.all_subjects()
                                         if s != int(cfg["target_subject"])]
    channels = (config.CHANNELS_OCCIPITO_PARIETAL
                if cfg.get("channel_set", "all63") == "occipital17" else None)
    data = things_eeg.load_loso(src, int(cfg["target_subject"]), channels,
                               mvnn=cfg.get("mvnn", "off"))
    arr = [np.asarray(data.tr_eeg[s]).reshape(-1, *np.asarray(data.tr_eeg[s]).shape[-2:])
           for s in range(data.n_subjects)]
    n_ch, n_t = arr[0].shape[-2:]

    rng = np.random.default_rng(args.seed)
    stacks, subs = [], []
    for s in range(data.n_subjects):
        for _ in range(args.draws):
            stacks.append(arr[s][rng.choice(len(arr[s]), size=args.k, replace=False)])
            subs.append(s)
    draws = np.stack(stacks)
    print(f"[data] {data.n_subjects} subjects x {args.draws} draws x K={args.k}, "
          f"trial ({n_ch}, {n_t}); subject blocks in order")

    for tag, ckpt in CKPTS.items():
        p = Path(ckpt)
        if not p.exists():
            print(f"[skip] {tag}: {ckpt} missing")
            continue
        report(tag, load_encoder(str(p), cfg, n_ch, n_t), draws, data.n_subjects)

    print("\nVERDICT: if the anchored arm's 'stat channel' separation is clearly positive "
          "while its 'SUM' separation is ~0, the anchor is informative but is erased by "
          "normalising the sum -- fix by normalising the two channels separately.")


if __name__ == "__main__":
    main()
