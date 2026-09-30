#!/usr/bin/env python3
"""NeuroWeave v3 — single source of truth for arms, bars, and kill criteria.

See docs/NEUROWEAVE_V3_ARCHITECTURE.md. Pure Python, no torch.
"""

from __future__ import annotations

# Dominate thresholds (Pareto): not-worse than every column-best, and
# strictly better on >= 3 metrics. SwAV is lower-is-better.
DOMINATE = {
    "pixcorr": 0.211,
    "ssim": 0.432,
    "alex2": 0.818,
    "alex5": 0.913,
    "inception": 0.831,
    "clip": 0.903,
    "swav": 0.489,
}

# Spatial-pathway ceiling measured on pred_lowlevel_rgb_512 (official protocol).
INIT_CEILING = {"pixcorr": 0.2998, "ssim": 0.4997}

# Retention needed to beat BrainAE from the init ceiling.
RETENTION_TARGET = {
    "pixcorr": 0.211 / 0.2998,  # ~0.704
    "ssim": 0.432 / 0.4997,    # ~0.864
}

# V1 fidelity-preserving generation grid: strength × CN timing.
# Low strength = keep more of the init (fidelity). Early CN end = structure lock then semantics.
V1_ARMS = {
    "s20_cn40": {"strength": 0.20, "cn_scale": 0.45, "cn_end": 0.40, "ip_scale": 0.80,
                 "role": "high fidelity, timed CN"},
    "s30_cn40": {"strength": 0.30, "cn_scale": 0.45, "cn_end": 0.40, "ip_scale": 0.90,
                 "role": "default fidelity band"},
    "s40_cn50": {"strength": 0.40, "cn_scale": 0.50, "cn_end": 0.50, "ip_scale": 1.00,
                 "role": "balanced"},
    "s50_cn30": {"strength": 0.50, "cn_scale": 0.40, "cn_end": 0.30, "ip_scale": 1.00,
                 "role": "more semantic, early CN only"},
    "s60_cn40": {"strength": 0.60, "cn_scale": 0.45, "cn_end": 0.40, "ip_scale": 1.00,
                 "role": "semantic-leaning"},
    "s30_cn00": {"strength": 0.30, "cn_scale": 0.00, "cn_end": 0.00, "ip_scale": 0.90,
                 "role": "ablation: no CN (init+IP only)"},
}

# Pre-registered V1 pass: any arm with PixCorr>=BrainAE AND SSIM>=BrainAE.
V1_BAR = {"pixcorr": 0.211, "ssim": 0.432}

# Protocol: generic prompts only for the main table.
PROMPTS_GENERIC = "outputs/g2f/prompts/prompts_deploy.json"

SOTA_REF = {
    "ATM": {"pixcorr": 0.160, "ssim": 0.345, "alex2": 0.776, "alex5": 0.866,
            "inception": 0.734, "clip": 0.786, "swav": 0.582},
    "BrainAE": {"pixcorr": 0.211, "ssim": 0.432, "alex2": 0.768, "alex5": 0.869,
                "inception": 0.753, "clip": 0.816, "swav": 0.541},
    "CogCapPro": {"pixcorr": 0.166, "ssim": 0.409, "alex2": 0.818, "alex5": 0.913,
                  "inception": 0.831, "clip": 0.903, "swav": 0.489},
}


def retention(metric: str, value: float) -> float:
    ceil = INIT_CEILING[metric]
    return float(value / ceil) if ceil > 0 else 0.0


def passes_v1(row: dict) -> bool:
    return float(row.get("pixcorr", -1)) >= V1_BAR["pixcorr"] and float(row.get("ssim", -1)) >= V1_BAR["ssim"]


def dominates(row: dict) -> dict:
    """Return per-metric comparison vs column-best and overall verdict."""
    lower_better = {"swav"}
    wins, ties, losses = [], [], []
    for k, thr in DOMINATE.items():
        v = float(row.get(k, float("nan")))
        if k in lower_better:
            if v < thr:
                wins.append(k)
            elif v <= thr * 1.001:
                ties.append(k)
            else:
                losses.append(k)
        else:
            if v > thr:
                wins.append(k)
            elif v >= thr * 0.999:
                ties.append(k)
            else:
                losses.append(k)
    ok = (len(wins) + len(ties) >= 5) and (len(wins) >= 3) and (len(losses) == 0)
    # Soft dominate: not worse on >=5, strictly better on >=3, allow at most 1 loss within 2%.
    soft = (len(wins) >= 3) and (len(losses) <= 1) and (len(wins) + len(ties) >= 5)
    return {
        "wins": wins, "ties": ties, "losses": losses,
        "n_wins": len(wins), "n_ties": len(ties), "n_losses": len(losses),
        "hard_dominate": ok, "soft_dominate": soft,
    }
