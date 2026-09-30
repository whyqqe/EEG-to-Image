#!/usr/bin/env python3
"""CPU preflight for the layer-wise injection experiment.

This runs on the login node with no GPU and no model weights.  It exists because
the failure mode that would ruin this experiment is SILENT: a layer spec that
does not land still renders perfectly plausible images, and the resulting null
would be reported as evidence against layered injection when nothing was ever
injected.  So the guards are tested here, with negative controls that must raise,
instead of being trusted.

CHECKS
  1. real UNet layout: 11 levels, early=4, late=7, read from the config on disk
  2. spec parser: valid specs parse; five malformed forms each raise
  3. scale_dict: every mode zeroes exactly the groups it should
  4. the four arms: level counts and total conditioning mass, by hand
  5. `lowall` mass == `layered` mass (the load-bearing matched-mass control)
  6. verify_assignment ACCEPTS a correctly assigned map for all four arms
  7. verify_assignment REJECTS four distinct corruptions (negative controls)
  8. the layered arm really is semantic-late / structure-early, and its
     semantic and structural level sets are disjoint while `single`'s overlap
  9. the specs in run_layered_sub08.sh match layered_arms.LAYERED_ARMS, so the
     bash orchestrator cannot drift away from the source of truth
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from layered_arms import (  # noqa: E402
    FRONTIER, GENERATE_ARMS, LAYERED_ARMS, LOWALL_SCALE, LEVEL_ORDER, SIDES,
    level_key, level_side, parse_spec, plan_counts, read_unet_layout,
    scale_dict, synthetic_level_map, verify_assignment,
)

NB = HERE.parents[1]
SDXL = ("/project/peilab/why/cache/eeg-brainit/hf/hub/"
        "models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/"
        "462165984030d82259a11f4367a4eed129e94a7b")

fails: list[str] = []


def ok(msg: str) -> None:
    print(f"[ok]   {msg}")


def check(cond: bool, msg: str) -> None:
    if cond:
        ok(msg)
    else:
        fails.append(msg)
        print(f"[FAIL] {msg}")


def must_raise(fn, label: str) -> None:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        ok(f"{label} rejected ({type(exc).__name__}: {str(exc)[:70]})")
        return
    fails.append(label)
    print(f"[FAIL] {label} was ACCEPTED but must be rejected")


print("=" * 78)
print("1. real UNet layout")
layout = read_unet_layout(SDXL)
nl = layout["n_levels"]
print(f"     {json.dumps(layout)}")
check(nl == {"early": 4, "late": 7}, f"layout early=4 late=7 (got {nl})")
check(sum(nl.values()) == len(LEVEL_ORDER) == 11,
      f"11 levels in both the config and LEVEL_ORDER (got {sum(nl.values())}/{len(LEVEL_ORDER)})")
check(layout["down_blocks_with_attn"] == [1, 2] and layout["up_blocks_with_attn"] == [0, 1],
      "down_blocks.0 and up_blocks.2 correctly excluded (they have no cross-attention)")

print("\n2. spec parser")
real_names = [
    "down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor",
    "mid_block.attentions.0.transformer_blocks.1.attn2.processor",
    "up_blocks.0.attentions.2.transformer_blocks.7.attn2.processor",
]
check(level_key(real_names[0]) == "down_blocks.1.attentions.0", "level_key strips down suffix")
check(level_key(real_names[1]) == "mid_block.attentions.0", "level_key strips mid suffix")
check(level_key(real_names[2]) == "up_blocks.0.attentions.2", "level_key strips up suffix")
check([level_side(n) for n in real_names] == ["early", "late", "late"],
      "side: down=early, mid=late, up=late")
check(level_side("weird_block.0") == "other", "unknown block classified as 'other'")
check(parse_spec("late:1.0,early:1.0,early:1.0", 3)
      == [("late", 1.0), ("early", 1.0), ("early", 1.0)], "valid spec parses")
for bad, label in [
    ("late:1.0,early:1.0", "wrong entry count"),
    ("late,early,early", "missing scale"),
    ("sideways:1.0,early:1.0,early:1.0", "unknown mode"),
    ("late:abc,early:1.0,early:1.0", "non-numeric scale"),
]:
    must_raise(lambda b=bad: parse_spec(b, 3), f"parse_spec {label}")

print("\n3. scale_dict zeroes the right groups")
check(scale_dict("all", 1.0) == {"down": 1.0, "mid": 1.0, "up": 1.0}, "all -> every group on")
check(scale_dict("early", 1.0) == {"down": 1.0, "mid": 0.0, "up": 0.0}, "early -> only down")
check(scale_dict("late", 1.0) == {"down": 0.0, "mid": 1.0, "up": 1.0}, "late -> mid+up only")
check(scale_dict("none", 1.0) == {"down": 0.0, "mid": 0.0, "up": 0.0}, "none -> all off")

print("\n4. the four arms: level counts and mass, by hand")
specs = {a: parse_spec(cfg["spec"], 3) for a, cfg in LAYERED_ARMS.items()}
plans = {a: plan_counts(s, nl) for a, s in specs.items()}
for a in LAYERED_ARMS:
    print(f"     {a:<8} spec={LAYERED_ARMS[a]['spec']:<50} "
          f"levels={plans[a]['active_levels_per_branch']} mass={plans[a]['total_mass']:.6f}")
check(plans["single"]["total_mass"] == 33.0 and
      plans["single"]["active_levels_per_branch"] == [11, 11, 11], "single: 3x11 @1.0 = 33")
check(plans["layered"]["active_levels_per_branch"] == [7, 4, 4] and
      plans["layered"]["total_mass"] == 15.0, "layered: semantic 7 late, structure 4 early, mass 15")
check(plans["rev"]["active_levels_per_branch"] == [4, 7, 7] and
      plans["rev"]["total_mass"] == 18.0, "rev: mirrored assignment, mass 18")
check(plans["lowall"]["active_levels_per_branch"] == [11, 11, 11], "lowall: uniform on all levels")

print("\n5. matched-mass control (this is what makes 'layered > single' interpretable)")
check(abs(plans["lowall"]["total_mass"] - plans["layered"]["total_mass"]) < 1e-9,
      f"lowall mass {plans['lowall']['total_mass']:.9f} == layered mass "
      f"{plans['layered']['total_mass']:.9f}")
check(abs(LOWALL_SCALE - 5.0 / 11.0) < 1e-15, f"lowall scale is exactly 5/11 ({LOWALL_SCALE!r})")
check(plans["layered"]["total_mass"] < plans["single"]["total_mass"],
      "layered carries LESS conditioning than single (so a win is not 'more conditions')")

print("\n6. verify_assignment ACCEPTS correctly assigned maps")
for a, spec in specs.items():
    try:
        v = verify_assignment(synthetic_level_map(spec), spec, nl)
        ok(f"{a}: accepted, mass={v['total_mass']:.6f}, "
           f"overlap={len(v['semantic_structure_overlap'])} branch(es)")
    except Exception as exc:  # noqa: BLE001
        fails.append(f"{a}: valid map rejected")
        print(f"[FAIL] {a}: valid map rejected -> {exc}")

print("\n7. verify_assignment REJECTS corruptions (negative controls)")
good = synthetic_level_map(specs["layered"])

# (a) a branch sitting on the wrong side of the split
perm = {k: list(v) for k, v in good.items()}
early0 = next(k for k in perm if level_side(k) == "early")
late0 = next(k for k in perm if level_side(k) == "late")
perm[early0][0], perm[late0][0] = perm[late0][0], perm[early0][0]
must_raise(lambda: verify_assignment(perm, specs["layered"], nl),
           "semantic branch on an early level")

# (b) a level silently missing (would mean a group was never matched)
missing = {k: v for k, v in good.items() if k != LEVEL_ORDER[-1]}
must_raise(lambda: verify_assignment(missing, specs["layered"], nl), "a missing level")

# (c) a 'none' branch that is non-zero somewhere
noneb = {k: list(v) for k, v in good.items()}
noneb[LEVEL_ORDER[-1]][2] = 1.0
must_raise(lambda: verify_assignment(noneb, parse_spec("late:1.0,early:1.0,none:1.0", 3), nl),
           "'none' branch active somewhere")

# (d) right active sets, wrong strength -> plan mass disagreement
off = {k: list(v) for k, v in good.items()}
late0 = next(k for k in off if level_side(k) == "late")
off[late0][0] = 0.99
must_raise(lambda: verify_assignment(off, specs["layered"], nl), "mass off by 0.01")

# (e) an unclassifiable block must not be ignored
weird = {k: list(v) for k, v in good.items()}
weird["mystery_block.0.attentions.0"] = [1.0, 1.0, 1.0]
must_raise(lambda: verify_assignment(weird, specs["layered"], nl), "unclassifiable level")

print("\n8. the hypothesis is encoded in the spec, and the split is disjoint")
sem, d, e = specs["layered"]
check(sem[0] == "late", "layered: semantic branch (0) is LATE")
check([d[0], e[0]] == ["early", "early"], "layered: structure branches (1,2) are EARLY")
v_lay = verify_assignment(synthetic_level_map(specs["layered"]), specs["layered"], nl)
check(v_lay["semantic_structure_overlap"] == [],
      "layered: semantic and structural level sets are disjoint")
v_single = verify_assignment(synthetic_level_map(specs["single"]), specs["single"], nl)
check(len(v_single["semantic_structure_overlap"]) == 2,
      "single: both structural branches co-active with semantic on all 11 levels")
check([m for m, _ in specs["rev"]] == ["early", "late", "late"],
      "rev: mirrored, so direction is testable")

print("\n9. bash orchestrator agrees with the source of truth")
runsh = NB / "scripts/nda/run_layered_sub08.sh"
if runsh.is_file():
    txt = runsh.read_text()
    # the run script must NOT carry spec literals; it must ask layered_arms for them,
    # otherwise the two can drift and the job would render a different experiment
    # than the audit and the summary reason about.
    for a in GENERATE_ARMS:
        check(f"spec_of {a}" in txt, f"run script pulls {a}'s spec from layered_arms")
    check(LOWALL_SCALE.__repr__() not in txt,
          "run script contains no hard-coded lowall scale (it must come from the module)")
    check("--layer-report" in txt and "--layer-sanity" in txt,
          "run script wires the assignment proof and the pixel-diff guard")
else:
    print("     (run script not written yet; skipped)")

print("\n10. the pre-registered bars are consistent with the frontier")
check(FRONTIER["single"]["ssim"] == 0.3696 and FRONTIER["single"]["cycle_disc_top1"] == 0.080,
      "single frontier point matches the measured mb_p3_i3_cn artifact")
check(FRONTIER["atm_aligned"]["cycle_disc_top1"] > FRONTIER["single"]["cycle_disc_top1"],
      "the frontier really is inverse: high SSIM -> low semantic identifiability")

print("\n" + "=" * 78)
if fails:
    print(f"[FATAL] {len(fails)} check(s) failed:")
    for f in fails:
        print(f"        - {f}")
    sys.exit(1)
print("[ok] layer plan audit: all checks passed")
