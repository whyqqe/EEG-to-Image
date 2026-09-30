"""NW4 single source of truth: generation arms, layer specs, and pre-registered bars.

Kept dependency-free (pure Python) so the CPU preflight can import it without
torch/diffusers, and so the orchestrator, the auditor and the summariser cannot
disagree about what an arm means.

WHY THE OLD V1 BAR WAS WRONG
---------------------------
nw3's bar was (pixcorr >= 0.211 and ssim >= 0.432).  The V0 init image -- the
blurry EEG->VAE decode, with Incep 0.5110 i.e. essentially chance -- ALREADY
scores 0.2998 / 0.4997.  So that bar passes by "not breaking the blur": it has no
discriminating power at all.  Measured endpoints make the trap explicit:

    init      SSIM 0.4997  Incep 0.5110   (perfect fidelity bottom, no semantics)
    rev       SSIM 0.3880  Incep 0.6918
    mb_p2_i3  SSIM 0.2716  Incep 0.7404   (best semantics)

Slope dIncep/dSSIM ~= -1.0 along that front, so reaching Incep 0.831 from the init
would cost ~0.32 SSIM (down to ~0.18).  Sliding the existing front CANNOT reach
the dominance point; only breaking the front can.  Hence the bar has to be 2-D
with a SEMANTIC gate, not just a fidelity floor.
"""

from __future__ import annotations

# ---------------------------------------------------------------- fidelity
# V0 measured ceiling: the EEG->VAE spatial path alone.
INIT_CEILING = {"pixcorr": 0.2998, "ssim": 0.4997, "inception": 0.5110}
# retention target relative to the ceiling (nw3 used these ratios)
RETENTION_TARGET = {"pixcorr": 0.7038, "ssim": 0.8645}

# ---------------------------------------------------------------- bars
# fidelity must be kept AND semantics must actually appear (> chance by a margin)
FID_FLOOR = {"pixcorr": 0.211, "ssim": 0.432}          # = BrainAE
SEM_GATE = {"inception": 0.65, "clip": 0.70}           # +0.15 / +0.20 over chance

# ---------------------------------------------------------------- condition gate
# THE gate that matters, established by nw4_diag_vstrue.py.
#
# Generation quality tracks `vs_true` = mean cos(condition_i, that trial's TRUE
# clip_img_i), and specifically its excess over a CONSTANT condition vector, which
# already scores ~0.62 because CLIP embeddings share a large common component:
#
#     vs_cl = vs_true - cos(constant, true)
#
# Every (condition, generation score) pair on disk obeys it, and the ordering is
# NOT the retrieval ordering:
#
#     condition      vs_cl    retrieval top1   inception   source
#     ip_blend      +0.0450        0.100         0.7053    hybrid_s08
#     ip_uck        +0.0367        0.140         0.7280    hybrid_s08
#     ip_hard       -0.0910        0.090         0.6324    hybrid_s08
#     nw3 z_fused   -0.7560        0.000         0.5236    nw3 (what it generated)
#     nw4 raw head  -0.3450        0.350         (never run)
#
# The nw4 raw head is the BEST retriever here and the WORST condition, which is why
# `offdiag` was the wrong gate: it rewarded discriminability, the property that is
# anti-correlated with what the adapter reads.  offdiag is retained for diagnostics
# only and must not gate anything.
CONST_FLOOR = {"vs_true": 0.6201}          # measured, mean of the TRAIN clip_img bank
COND_GATE = {"vs_cl": 0.0450}              # must beat hybrid's best condition

# ---------------------------------------------------------------- honest bar
# The bar to beat with GENERIC prompts.  Verified clean: prompts_deploy.json has
# 0/200 class-name hits, eval protocol is identical (`eval_official_seven_dir.py`,
# gray@425), and nw4_diag_initleak.py measured the init images at 0.0050 top-1
# (= chance), so the shared sdedit_ll init is NOT what produced these numbers.
HONEST_REF = {
    "hs_uck_uckF":  {"inception": 0.7280, "clip": 0.7982, "pixcorr": 0.1646,
                     "ssim": 0.3674, "alex2": 0.7391, "alex5": 0.8447,
                     "swav": 0.5920, "fid": 177.33, "vs_cl": 0.0367},
    "hs_blend_uckF": {"inception": 0.7053, "clip": 0.7983, "vs_cl": 0.0450},
    "hs_hard_uckF": {"inception": 0.6324, "clip": 0.7123, "vs_cl": -0.0910},
}

# soft dominance = beat every reference on EVERY reported metric.
DOMINATE = {"pixcorr": 0.211, "ssim": 0.432, "alex2": 0.818,
            "alex5": 0.913, "inception": 0.831, "clip": 0.903, "swav": 0.489}

SOTA_REF = {
    "ATM":       {"pixcorr": 0.160, "ssim": 0.345, "alex2": 0.776, "alex5": 0.866,
                  "inception": 0.734, "clip": 0.786, "swav": 0.582},
    "BrainAE":   {"pixcorr": 0.211, "ssim": 0.432, "alex2": 0.768, "alex5": 0.869,
                  "inception": 0.753, "clip": 0.816, "swav": 0.541},
    "CogCapPro": {"pixcorr": 0.166, "ssim": 0.409, "alex2": 0.818, "alex5": 0.913,
                  "inception": 0.831, "clip": 0.903, "swav": 0.489},
}

# NW3's measured 6-arm grid.  All six arms shared ONE condition vector
# (`s3/conds/ip_primary_test.npy`, identical to z_fused), so the grid only swept
# strength/ControlNet timing and never tested the condition design at all.
NW3_REF = {"best_inception": 0.5236, "best_arm": "s30_cn40",
           "init_inception": 0.5110, "init_pixcorr": 0.2998, "init_ssim": 0.4997,
           "s3_offdiag": 0.6994, "s2_offdiag": 0.5668, "vs_cl": -0.7560}

# ---------------------------------------------------------------- layers
# mode -> {"down","mid","up"} assignment, via set_ip_adapter_scale dicts.
#   late  : mid + up   (the layers that carry semantics / global layout)
#   early : down only  (the layers that carry local structure and edges)
#   all   : everywhere (nw3's behaviour, no layering)
MODES = ("all", "early", "late")

# ---------------------------------------------------------------- arms
# Each arm: branch order (must match --cond-npys order), per-branch layer spec,
# plus the generation knobs.  `branch-spec` uses layered_arms' MODE:SCALE grammar.
# ---------------------------------------------------------------- generation params
# MATCHED to the honest reference (hybrid_s08 / uck_nat_s08) so a win is a real win.
# The earlier nw3-style values (strength 0.30) left the init image dominant and the
# semantic condition nearly inert -- which is also why nw3's grid could not move.
GEN_STRENGTH = 0.82
GEN_CN = 0.40
GEN_CN_END = 0.40
GEN_STEPS = 28
GEN_GUIDANCE = 5.0
GEN_IP_SCALE = 1.0

BRANCHES = ("img", "attr", "depth", "edge")   # order is the contract
SEMANTIC_BRANCHES = ("img", "attr")
STRUCT_BRANCHES = ("depth", "edge")

ARMS: dict[str, dict] = {
    # --- A0/A1 claim: the new open-set trial-level condition, single branch -------
    "w4_img_only": {
        "branches": ("img",), "spec": "late:0.9",
        "strength": GEN_STRENGTH, "cn_scale": GEN_CN, "cn_end": GEN_CN_END,
        "note": "single A0/A1 condition projected onto the clip_img manifold"},
    # --- A3 control: nw3's dense fusion (the thing that collapsed to the mean) ---
    "w4_fused": {
        "branches": ("fused",), "spec": "all:0.9",
        "strength": GEN_STRENGTH, "cn_scale": GEN_CN, "cn_end": GEN_CN_END,
        "note": "dense mean of all branches = nw3 S3 behaviour (offdiag 0.6994)"},
    # --- A3: multi-branch, no layering (all levels everywhere) -------------------
    "w4_flat": {
        "branches": BRANCHES, "spec": "all:0.85,all:0.85,all:0.85,all:0.85",
        "strength": GEN_STRENGTH, "cn_scale": GEN_CN, "cn_end": GEN_CN_END,
        "note": "4 real branches, each at every UNet level"},
    # --- A4: the actual hypothesis ----------------------------------------------
    "w4_layered": {
        "branches": BRANCHES, "spec": "late:0.9,late:0.9,early:0.8,early:0.8",
        "strength": GEN_STRENGTH, "cn_scale": GEN_CN, "cn_end": GEN_CN_END,
        "note": "semantic -> mid+up, structure -> down  (A4's claim)"},
    # --- A4 control: the MIRROR of `layered` ------------------------------------
    "w4_mirror": {
        "branches": BRANCHES, "spec": "early:0.9,early:0.9,late:0.8,late:0.8",
        "strength": GEN_STRENGTH, "cn_scale": GEN_CN, "cn_end": GEN_CN_END,
        "note": "negative control: swapping the two groups must NOT help"},
    # --- A4 control: all early / all late (pure level effects, no semantics split)
    "w4_allearly": {
        "branches": BRANCHES, "spec": "early:0.85,early:0.85,early:0.85,early:0.85",
        "strength": GEN_STRENGTH, "cn_scale": GEN_CN, "cn_end": GEN_CN_END,
        "note": "single-level control: everything into down blocks"},
    "w4_alllate": {
        "branches": BRANCHES, "spec": "late:0.85,late:0.85,late:0.85,late:0.85",
        "strength": GEN_STRENGTH, "cn_scale": GEN_CN, "cn_end": GEN_CN_END,
        "note": "single-level control: everything into mid+up blocks"},
    # --- ablate the structural branches entirely --------------------------------
    "w4_sem_only": {
        "branches": BRANCHES, "spec": "late:0.9,late:0.9,all:0.0,all:0.0",
        "strength": GEN_STRENGTH, "cn_scale": GEN_CN, "cn_end": GEN_CN_END,
        "note": "semantics only; structure branches zeroed (ControlNet depth stays)"},
    # --- ablate the semantic branches (expect a fidelity/semantics tradeoff) -----
    "w4_struct_only": {
        "branches": BRANCHES, "spec": "all:0.0,all:0.0,early:0.85,early:0.85",
        "strength": GEN_STRENGTH, "cn_scale": GEN_CN, "cn_end": GEN_CN_END,
        "note": "structure only; semantic branches zeroed"},
    # --- strength sweep on the winner shape (fidelity knob) ---------------------
    "w4_layered_s20": {
        "branches": BRANCHES, "spec": "late:0.9,late:0.9,early:0.8,early:0.8",
        "strength": 0.55, "cn_scale": GEN_CN, "cn_end": GEN_CN_END,
        "note": "lower SDEdit strength: more fidelity, less semantics"},
    "w4_layered_s40": {
        "branches": BRANCHES, "spec": "late:0.9,late:0.9,early:0.8,early:0.8",
        "strength": 0.92, "cn_scale": GEN_CN, "cn_end": GEN_CN_END,
        "note": "higher SDEdit strength: more semantics, less fidelity"},
}

# arms whose branches are the plain 4 (everything else is 1-branch)
MULTI = tuple(a for a, v in ARMS.items() if len(v["branches"]) == len(BRANCHES))


def spec_counts(spec: str) -> dict:
    """Sanity: how many entries and which modes a spec names."""
    items = [x.strip() for x in spec.split(",") if x.strip()]
    return {"n": len(items),
            "modes": [i.split(":", 1)[0] for i in items],
            "scales": [float(i.split(":", 1)[1]) for i in items]}


def passes_2d(m: dict) -> bool:
    """Keep fidelity AND actually show semantics."""
    return all(m.get(k, 0.0) >= v for k, v in FID_FLOOR.items()) and \
           all(m.get(k, 0.0) >= v for k, v in SEM_GATE.items())


def cond_gate(vs_true: float) -> dict:
    """Can the adapter read this condition at all?

    `nw4_diag_vstrue.py` showed generation tracks vs_cl = vs_true - cos(constant,
    true), NOT retrieval accuracy and NOT offdiag.  A condition that fails this gate
    cannot beat the reference no matter how the branches are wired, so the arms must
    not be run on one -- that is what this function exists to refuse.
    """
    vs_cl = float(vs_true) - CONST_FLOOR["vs_true"]
    return {"vs_true": round(float(vs_true), 4), "vs_cl": round(vs_cl, 4),
            "floor": CONST_FLOOR["vs_true"], "gate": COND_GATE["vs_cl"],
            "pass": vs_cl >= COND_GATE["vs_cl"],
            "beats_reference": vs_cl >= HONEST_REF["hs_uck_uckF"]["vs_cl"]}


def dominates(m: dict) -> dict:
    w, l, t = [], [], []
    for k, v in DOMINATE.items():
        got = m.get(k)
        if got is None:
            continue
        if k == "swav":  # lower is better
            (w if got <= v else l).append(k)
        elif got > v + 1e-9:
            w.append(k)
        elif abs(got - v) <= 1e-9:
            t.append(k)
        else:
            l.append(k)
    return {"wins": w, "ties": t, "losses": l, "n_wins": len(w),
            "n_losses": len(l), "hard": len(l) == 0 and len(t) == 0,
            "soft": len(l) == 0}


def retention(m: dict) -> dict:
    return {k: m.get(k, 0.0) / INIT_CEILING[k] for k in ("pixcorr", "ssim")}
