#!/usr/bin/env python3
"""Single source of truth for the layer-wise injection experiment.

Everything here is PURE PYTHON: no torch, no diffusers, no PIL.  That is
deliberate -- the CPU preflight must be able to unit-test the layer plan, the
spec parser and the assignment verifier without a GPU or a 5 GB model, and the
bash orchestrator must be able to read the exact arm specs from one place
instead of repeating them as string literals that can drift.

PRE-REGISTERED DESIGN (sub-08, N=200)
-------------------------------------
Modalities (branch order is fixed):  0 = semantic CLIP (ip_mem), 1 = depth, 2 = edge
All arms use the SAME conditions, prompts, seed, steps, ControlNet and low-level
init as the existing single-channel arm, so the ONLY thing that varies is where
each modality is injected.

SDXL has 11 cross-attention levels and they are not symmetric:

    early = down_blocks.{1,2}                        -> 4 levels  (2 + 2)
    late  = mid_block + up_blocks.{0,1}              -> 7 levels  (1 + 3 + 3)

`down_blocks.0` and `up_blocks.2` carry no cross-attention and are not levels.

    arm        spec                            semantic     structure      mass
    single     all:1.0, all:1.0, all:1.0       all 11       all 11         33
    layered    late:1.0, early:1.0, early:1.0  late 7       early 4        15
    rev        early:1.0, late:1.0, late:1.0   early 4      late 7         18
    lowall     all:5/11 x3                     all 11       all 11         15

`single` is REUSED, not regenerated: it is exactly `outputs/mb_s08/gen/mb_p3_i3_cn`
(steps 28, guidance 5, seed 42, cn_scale 0.40, strength 0.82, 3 branches at 1.0),
already scored at SSIM 0.3696 and cycle_disc_top1 0.080.

WHY `lowall` EXISTS
-------------------
`layered` carries LESS total conditioning than `single` (15 vs 33), so "layered
wins" could be explained by "less conditioning wins" rather than by placement.
`lowall` is a uniform-strength arm with total mass 15 -- EXACTLY `layered`'s mass
-- so it isolates placement from strength:

    layered > single  AND  lowall approx single   ->  placement is the active ingredient
    layered approx lowall                         ->  it was only ever the strength

`rev` is the direction control: same modality split, opposite assignment.  If
`rev` approx `layered`, then any split helps and the "semantic goes late" story is
not supported.

PRE-REGISTERED VERDICTS (dual metric, both must be read together)
----------------------------------------------------------------
The frontier these arms have to beat, measured with the SAME scorers:

    arm            SSIM      cycle_disc_top1
    atm_aligned    0.2300    0.565
    mb_p1_i1       0.3500    0.130
    single         0.3696    0.080

    H-LAYER PASS          SSIM >= 0.3696  AND  cycle_disc_top1 >= 0.25
                          (matches the best SSIM on the frontier while at least
                           tripling the best structural arm's semantic score)
    H-LAYER STRONG PASS   SSIM >= 0.3696  AND  cycle_disc_top1 >= 0.40
                          (genuinely breaks toward the atm_aligned regime)
    H-LAYER FALSIFIED     layered <= single on BOTH, or lowall reproduces
                          layered on both
    DIRECTION SUPPORTED   layered ranks above single ranks above rev on the
                          combined rank; if rev approx layered, direction is not
                          supported and the placement story must be dropped.
"""
from __future__ import annotations

import json
from pathlib import Path

MODES = ("all", "early", "late", "none")
SIDES = ("early", "late")

# The 11 SDXL cross-attention levels, in forward-pass order.
LEVEL_ORDER = (
    "down_blocks.1.attentions.0", "down_blocks.1.attentions.1",
    "down_blocks.2.attentions.0", "down_blocks.2.attentions.1",
    "mid_block.attentions.0",
    "up_blocks.0.attentions.0", "up_blocks.0.attentions.1", "up_blocks.0.attentions.2",
    "up_blocks.1.attentions.0", "up_blocks.1.attentions.1", "up_blocks.1.attentions.2",
)

# ---- the arms -------------------------------------------------------------
# `lowall`'s scale is 5/11 because it must land on exactly `layered`'s mass:
#   3 branches x 11 levels x s = 15  ->  s = 15/33 = 5/11
LOWALL_SCALE = 5.0 / 11.0

LAYERED_ARMS = {
    "single": {
        "spec": "all:1.0,all:1.0,all:1.0",
        "reuse_dir": "outputs/mb_s08/gen/mb_p3_i3_cn/generated",
        "reuse_eval": "outputs/mb_s08/eval/s08_mb_p3_i3_cn.json",
        "role": "control: one conditioning channel, every modality at every level",
    },
    "layered": {
        "spec": f"late:1.0,early:1.0,early:1.0",
        "reuse_dir": None, "reuse_eval": None,
        "role": "candidate: semantic late, structure early (the NeuroWeave claim)",
    },
    "rev": {
        "spec": "early:1.0,late:1.0,late:1.0",
        "reuse_dir": None, "reuse_eval": None,
        "role": "direction control: same split, opposite assignment",
    },
    "lowall": {
        "spec": f"all:{LOWALL_SCALE:.17g},all:{LOWALL_SCALE:.17g},all:{LOWALL_SCALE:.17g}",
        "reuse_dir": None, "reuse_eval": None,
        "role": "matched-mass strength control: uniform, total mass == layered",
    },
}
GENERATE_ARMS = ("layered", "rev", "lowall")

# Frontier to beat, from artifacts measured with the same scorers.
FRONTIER = {
    "atm_aligned": {"ssim": 0.2300, "cycle_disc_top1": 0.565},
    "mb_p1_i1": {"ssim": 0.3500, "cycle_disc_top1": 0.130},
    "single": {"ssim": 0.3696, "cycle_disc_top1": 0.080},
}
BAR_SSIM = 0.3696
BAR_CYCLE_PASS = 0.25
BAR_CYCLE_STRONG = 0.40


# ---- pure logic ----------------------------------------------------------
def level_side(attn_name: str) -> str:
    """'early' for down blocks, 'late' for mid+up blocks, 'other' if unexpected."""
    if attn_name.startswith("down_blocks"):
        return "early"
    if attn_name.startswith("mid_block") or attn_name.startswith("up_blocks"):
        return "late"
    return "other"


def level_key(attn_name: str) -> str:
    """Collapse a processor name to its level.

    `down_blocks.1.attentions.0.transformer_blocks.2.attn2.processor`
        -> `down_blocks.1.attentions.0`
    """
    i = attn_name.find(".transformer_blocks")
    return attn_name[:i] if i > 0 else attn_name


def parse_spec(s: str, n: int) -> list[tuple[str, float]]:
    """'late:1.0,early:1.0,early:1.0' -> [('late',1.0), ('early',1.0), ('early',1.0)]."""
    items = [x.strip() for x in s.split(",") if x.strip()]
    if len(items) != n:
        raise ValueError(f"spec has {len(items)} entries but there are {n} branches")
    out: list[tuple[str, float]] = []
    for it in items:
        if ":" not in it:
            raise ValueError(f"spec entry '{it}' is not MODE:SCALE")
        mode, val = it.split(":", 1)
        mode = mode.strip().lower()
        if mode not in MODES:
            raise ValueError(f"spec entry '{it}': mode must be one of {MODES}")
        try:
            out.append((mode, float(val)))
        except ValueError:
            raise ValueError(f"spec entry '{it}': scale '{val}' is not a float")
    return out


def scale_dict(mode: str, s: float) -> dict:
    """One branch's dict for `set_ip_adapter_scale`.

    Unspecified groups fall back to `default_scale=0.0` inside diffusers, but we
    always name all three groups explicitly so the intent survives a diffusers
    version that changes that default.
    """
    if mode == "all":
        return {"down": s, "mid": s, "up": s}
    if mode == "early":
        return {"down": s, "mid": 0.0, "up": 0.0}
    if mode == "late":
        return {"down": 0.0, "mid": s, "up": s}
    return {"down": 0.0, "mid": 0.0, "up": 0.0}


def expected_levels(mode: str, levels_early: list[str], levels_late: list[str]) -> set[str]:
    """Which levels a branch with this mode must end up active on."""
    if mode == "all":
        return set(levels_early) | set(levels_late)
    if mode == "early":
        return set(levels_early)
    if mode == "late":
        return set(levels_late)
    return set()


def plan_counts(spec: list[tuple[str, float]], n_levels: dict[str, int]) -> dict:
    """Per-branch active level counts and the total conditioning mass.

    mass = sum over (branch, level) of the applied scale -- the same currency the
    single-channel baseline is measured in, so arms can be mass-matched.
    """
    total_levels = sum(n_levels.values())
    per_branch, mass = [], 0.0
    for mode, s in spec:
        n = total_levels if mode == "all" else n_levels.get(mode, 0) if mode in SIDES else 0
        per_branch.append(int(n))
        mass += float(s) * int(n)
    return {"active_levels_per_branch": per_branch,
            "n_branches": len(spec),
            "total_mass": mass,
            "uniform_equivalent_scale": (mass / (len(spec) * total_levels)
                                         if spec and total_levels else 0.0)}


def read_unet_layout(sdxl_path: str) -> dict:
    """Derive the level counts from the real UNet config on disk (loads no weights)."""
    cfg = json.loads((Path(sdxl_path) / "unet" / "config.json").read_text())
    lpb = int(cfg["layers_per_block"])
    down = [i for i, t in enumerate(cfg["down_block_types"]) if "CrossAttn" in t]
    up = [i for i, t in enumerate(cfg["up_block_types"]) if "CrossAttn" in t]
    return {"layers_per_block": lpb,
            "down_blocks_with_attn": down, "up_blocks_with_attn": up,
            "n_levels": {"early": len(down) * lpb, "late": 1 + len(up) * (lpb + 1)}}


def verify_assignment(level_map: dict[str, list[float]], spec: list[tuple[str, float]],
                      n_levels: dict[str, int]) -> dict:
    """Assert the REALISED per-level assignment is the planned one, or raise.

    A silently wrong assignment renders plausible images and looks like a clean
    null result, which is the single most dangerous failure mode of this
    experiment, so every mismatch is fatal:
      * a level this code cannot classify
      * a level count that disagrees with the UNet config
      * a branch active on the wrong levels
      * a 'none' branch that is non-zero somewhere
      * realised total mass != planned mass
    """
    for lvl in level_map:
        if level_side(lvl) == "other":
            raise ValueError(f"unclassifiable attention level '{lvl}'; the "
                             f"early/late split is undefined for it")
    levels = {s: sorted(k for k in level_map if level_side(k) == s) for s in SIDES}
    for s, n in n_levels.items():
        if len(levels[s]) != n:
            raise ValueError(f"found {len(levels[s])} {s} levels but the UNet config "
                             f"implies {n}: {levels[s]}")

    branches = []
    for i, (mode, sc) in enumerate(spec):
        active = {k for k, scales in level_map.items() if scales[i] > 0}
        want = expected_levels(mode, levels["early"], levels["late"])
        if active != want:
            raise ValueError(
                f"branch {i} (mode={mode}) active on {len(active)} levels, planned "
                f"{len(want)}; extra={sorted(active - want)} missing={sorted(want - active)}")
        if mode == "none" and any(scales[i] != 0.0 for scales in level_map.values()):
            raise ValueError(f"branch {i} declared 'none' but is non-zero somewhere")
        branches.append({"branch": i, "mode": mode, "scale": sc,
                         "n_levels_active": len(active)})

    real_mass = float(sum(sum(scales) for scales in level_map.values()))
    plan_mass = plan_counts(spec, n_levels)["total_mass"]
    if abs(real_mass - plan_mass) > 1e-6:
        raise ValueError(f"realised mass {real_mass} != planned {plan_mass}")

    overlap = []
    if len(spec) > 1:
        sem = {k for k, sc in level_map.items() if sc[0] > 0}
        for i in range(1, len(spec)):
            st = {k for k, sc in level_map.items() if sc[i] > 0}
            inter = sorted(sem & st)
            if inter:
                overlap.append({"branch": i, "levels": inter})
    return {"branches": branches, "total_mass": real_mass,
            "n_levels": {"early": len(levels["early"]), "late": len(levels["late"])},
            "levels_early": levels["early"], "levels_late": levels["late"],
            "semantic_structure_overlap": overlap}


def synthetic_level_map(spec: list[tuple[str, float]]) -> dict[str, list[float]]:
    """A level_map as it WOULD look after a correct assignment, for tests."""
    return {lvl: [float(s) if (m == "all" or m == level_side(lvl)) else 0.0
                  for m, s in spec]
            for lvl in LEVEL_ORDER}
