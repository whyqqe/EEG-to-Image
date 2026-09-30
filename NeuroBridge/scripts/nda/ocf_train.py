#!/usr/bin/env python3
"""OCF: ORTHOGONAL CHANNEL FUSION -- sub-08.

WHY THIS ARCHITECTURE, AND NOT ANOTHER HEAD
-------------------------------------------
Measured on sub-08 (n=200) with assets that already exist:
  * HCMA and every g2f/g3f variant read BOTH towers off the SAME 512-d
    `z_eeg_proj`.  A shared input means the two towers cannot be independent in
    information terms -- sharing a representation is not sharing capacity.
  * The two readouts are nevertheless nearly error-ORT: on the ranking task,
    with S = semantic scores and L = low-frequency-latent scores,
        r(S, L) = +0.203
        semantic right 41 / layout right 17 / BOTH right 7
        semantic-only solvable 34 | layout-only solvable 10 | union 51
    The layout channel solves 10 rows the semantic channel never solves.
  * Standardised fusion of the two scores (lambda = 0.3, chosen on a training
    half) gives
        2-way 0.930 -> 0.955      top-1 0.205 -> 0.290      mean rank 12.8 -> 12.0
    and a 5-fold CV gives +0.050/+0.200/+0.075/+0.150/+0.075, i.e. +11.0pp
    top-1 on 5/5 folds.  This is the only robust positive signal in the project.

AN INPUT-SIDE SPLIT WAS PROPOSED, MEASURED, AND FALSIFIED
---------------------------------------------------------
`SharedSpecificEncoder.forward(..., return_parts=True)` returns
(out, s, r) with `r = self.shared(x)` and `s = self.specific[subject](x)`, and
pretraining adds `diff_loss(s, r) = MSE(s * r, 0)` with lambda_diff 0.1 -- so the
encoder was TRAINED to make the two components orthogonal.  Hypothesis: route the
shared part to the semantic tower (it was built for subject invariance) and the
per-subject part to the layout tower (retinotopy / head geometry are
subject-specific).  Prediction stated in advance, then measured with one ridge per
representation, fit on TRAIN rows only (`ocf_export_ss_parts.py`):

    semantics (-> CLIP-1024)      dim   top1    2way
      z_eeg_proj  (status quo)     512  0.2450  0.9350
      fused                        1024  0.2250  0.9450
      shared_r                     1024  0.2900  0.9550   <- BEST
      specific_s                   1024  0.1550  0.9300   <- WORST
    layout    (-> LF-VAE latent)  dim   top1    2way
      z_eeg_proj                   512  0.1050  0.9050
      fused                        1024  0.1250  0.9000
      shared_r                     1024  0.1100  0.9050
      specific_s                   1024  0.0750  0.8350   <- WORST

    VERDICT  shared->semantics      PASS
             specific->geometry     FAIL  (specific_s is worst at BOTH tasks)

So `s` is not "subject-specific geometry"; it is subject-IDENTITY, exactly the
nuisance the diff_loss was designed to remove from the shared signal.  The
input-side split is dead and is NOT used here.  Two consequences were kept:

  (i)  A FREE WIN, and it is the opposite of what the split wanted: the pipeline's
       own export, `z_eeg_proj = ProjectorLinear(Adapter(ResFuse(s, r)))`, is
       WORSE than the `shared_r` it was derived from --
           2-way 0.9350 -> 0.9550 (+2.0pp)   top-1 0.2450 -> 0.2900 (+4.5pp)
       because ResFuse mixes in the useless `s` and ProjectorLinear then halves
       1024 -> 512.  `--z-source shared_r` (default) is therefore the encoder
       input, and `--z-source z_eeg_proj` reproduces the old number.
  (ii) The per-subject machinery (specific embedder + adapter + calibration) is
       unnecessary for BOTH tasks on this evidence.  Whether it is unnecessary
       CROSS-subject is a separate measurement (ridge fit on the other subjects,
       tested on the held-out one) and is not asserted here.

WHAT ACTUALLY CREATES THE ORTHOGONALITY
---------------------------------------
Not the input, and not a bottleneck: the TARGETS.  Two heads trained against two
different target families (CLIP image/text vs low-frequency VAE latent) end up
with error-orthogonal predictions.  The architecture therefore spends capacity on
making each readout a good DISCRIMINATIVE predictor of its own target, and gets
the arbitration for free.

  (1) RANK-r EXCHANGE is DEMOTED to an ablation (`--rank`, default 0 = a no-op).
      Its purpose was to synthesise separation between two trunks; the input-side
      measurement above shows there is nothing to separate at the input, so the
      separation has to come from the targets.  `--rank 8` remains available.

  (2) THE LAYOUT CHANNEL IS TRAINED TO BE DISCRIMINATIVE, not just accurate.
      g3f's `h_struct` had ONLY a cosine regression term, so it was a point
      estimator -- precisely the mechanism that collapsed h_ip to the
      conditional mean (cos 0.613-0.634 against a CONSTANT baseline of 0.6147).
      A point estimator cannot arbitrate between candidates.  OCF adds
      multi-positive InfoNCE on the amplitude-equalised low-frequency latent,
      which is what makes the channel usable as a verification signal.

  (3) MANIFOLD CALIBRATION, EXPORTED AND MEASURED.  Our conditions are
      off-manifold for IP-Adapter, which was trained on REAL CLIP image
      embeddings:
          our condition  ||mean|| 0.4221   c_self 0.4221   elem-std 0.0258
          real CLIP      ||mean|| 0.6276   c_self 0.6276   elem-std 0.0235
          ratio 0.673
      The calibration is a quantile match on c_self = cos(x_i, mean) and is
      label-free and test-free.  NOTE the direction: our conditions are LESS
      concentrated than real CLIP, so calibration moves them TOWARDS the mean,
      which by our own earlier finding costs instance information.  The row is
      included because the hypothesis is empirical, not because it is expected
      to win.

  (4) SFV -- SELECTION BY FUSION VERIFICATION (the deployable form of the
      +11pp measurement).  The semantic channel proposes K conditions sampled
      from the soft-memory posterior (the EEG posterior is one-to-many, so the
      proposal must be a SET, not a point).  The layout channel then arbitrates,
      and the arbitration is NON-CIRCULAR: the layout prediction is never used
      to generate, and its target is the low-frequency latent of REAL images,
      whereas the candidates were conditioned on semantic vectors.  The score is
      therefore
          s_k = cos( LF(lf_head(z_i)) , LF(VAE(gen_k)) )
      with NO free weight.  The semantic term cos(ip_k, CLIP(gen_k)) is
      deliberately NOT used: gen_k was generated from ip_k, so that term is
      self-fulfilling and would measure nothing.

DELETED (all three on measured evidence, carried over from g3f)
  CFM            -- sampled spread 0.9911, top-1 0.015 vs its own input's 0.160;
                    the training target is the mean flow while inference
                    integrates a sample, so under one-to-many conditioning the
                    two disagree by construction; `ode_mix` was never in the
                    graph.  Image side: g2f_cfm was the worst row (FID 223.48).
  h_texture      -- fine/coarse = 0.0765 of explainable variance; it regressed an
                    unidentifiable component.
  h_ip (direct)  -- cos-to-target indistinguishable from a constant; its gradient
                    pulled the trunk toward the mean. h_fuse is the sole
                    deterministic IP producer.

LEAK-FREE, BY CONSTRUCTION (unchanged from g3f, re-audited at run time)
  * prompt gallery = 1654 TRAIN concepts only; the 200 test concepts are DISJOINT
    from it, verified and hard-failed if not, so no emitted string can name a
    test class;
  * the memory bank, concept gallery and calibration reference are train-only;
  * `sem_concept_tmpl_test.npy` is read for a DIAGNOSTIC line only;
  * checkpoint selection uses a held-out slice of TRAIN rows;
  * every generation hyper-parameter is fixed a priori.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NB_ROOT = Path(__file__).resolve().parents[2]

GRANS = ("overall", "subject", "background", "detail")
GENERIC_PROMPT = ("a photo of an object, clearly showing its shape, color, and "
                  "distinctive parts, natural lighting")


# ---------------------------------------------------------------- utils

def l2t(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Make `x` a unit vector -- with a norm floor that is NOT a rounding guard.

    WHY THE FLOOR IS 1e-3 AND NOT 1e-8  (a measured, run-killing defect)
    -------------------------------------------------------------------
    `x / x.norm().clamp_min(1e-8)` is an exact unit vector for any x, but its
    JACOBIAN is `(I - y y^T) / ||x||_clamped`, i.e. 1e8 at x = 0.  Every read-out
    head in this project used to be zero-initialised, so at step 0 EVERY condition
    was exactly the zero vector and every loss passed through that 1e8 Jacobian.
    Measured consequence on the TDM-DT run: `clip_grad_norm_` saw a total norm of
    `inf` (no element was inf/nan -- a single fp32 tensor norm simply overflowed),
    so `clip_coef = 5/(inf + 1e-6) = 0` and **every gradient was multiplied by
    exactly zero for the whole run**.  The model was frozen for all 754 steps:
    `tr_loss` moved 43.4290 -> 43.4249, `lag_raw` stayed exactly 0.0, the time
    gates stayed exactly uniform, and the exported conditions were an untrained
    random projection (2-way 0.485 = chance).

    Flooring the norm at 1e-3 bounds every Jacobian at 1e3: finite, and small
    enough that ordinary gradient clipping at 5.0 is nowhere near it.  It cannot
    change the meaning of a non-degenerate call -- every target fed to this
    function is already l2-normalised (norm ~ 1), and every trained model output
    has norm >> 1e-3 -- it only stops the sub-1e-3 regime, where the "direction"
    of x is numerically meaningless anyway, from exploding.
    """
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-3)


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def spherical(mu: torch.Tensor, res: torch.Tensor, theta: float) -> torch.Tensor:
    """`l2(cos(theta) * l2(mu) + sin(theta) * l2(res))`.

    WHY THIS REPLACES `l2(mu + res)`  (a measured defect, not a preference)
    ---------------------------------------------------------------------
    `l2(mu + res)` with an UNBOUNDED residual is degenerate whenever the losses
    are scale-invariant, which they all are here (cosine, InfoNCE, and a var_band
    that is normalised by the target's own dispersion).  Write m = ||mu||,
    d = ||res||; the direction of `l2(mu + res)` depends only on d/m, and the
    gradient of a scale-invariant loss w.r.t. d/m pushes d/m -> infinity, at which
    point the mean contributes nothing.  Measured on g3f: res_norm = 8.11 - 9.68,
    i.e. the mean's relative share is O(1/81) ~ 1.2%.  The "centred residual"
    mechanism was therefore a 1% effect, consistent with its measured Delta of
    -0.009 (zero).

    Fixing theta removes the degenerate direction: the residual now contributes a
    UNIT direction and theta sets its angular share, which no optimiser can
    inflate or deflate.  theta is an explicit, ablatable quantity:
        theta = 0       -> the condition IS the train mean (the degenerate limit)
        theta = pi/2    -> the condition is the residual alone (what g3f did)
        theta = acos(c) -> matches a target self-concentration c exactly

    GEOMETRY (why theta is the manifold knob, not a free hyper-parameter)
    ---------------------------------------------------------------------
    For a unit vector y = cos(theta) m_hat + sin(theta) d_hat,
        cos(y, m_hat) = cos(theta) / sqrt(1 + 2 cos(theta) sin(theta) * rho),
        rho = m_hat . d_hat.
    With rho = 0 this is exactly cos(theta), so setting
        theta = arccos(c_self_ref)
    makes our condition's concentration match the reference distribution's OWN
    concentration BY CONSTRUCTION, without any post-hoc calibration.  rho != 0
    only perturbs it monotonically, so theta remains the right control and the
    achieved value is measured and reported rather than assumed.
    """
    # mu buffers are 1-D (D,); normalise to (1, D) so the broadcasting is
    # unambiguous whatever the caller passes.  The general formula covers the
    # whole arc theta in [0, pi]; there is NO special case at pi/2 (sin = 1,
    # cos = 0, so it reduces to l2(res) by itself) and NONE above it: theta > pi/2
    # means SUBTRACTING the mean, which is the branch that reaches the
    # concentrations below the residual's own (see fit_theta_on_grid).
    mu2 = mu.reshape(1, -1)
    m = l2t(mu2).expand(res.shape[0], -1)
    if abs(math.sin(theta)) < 1e-12:
        return l2t(m if math.cos(theta) > 0 else -m)
    return l2t(math.cos(theta) * m + math.sin(theta) * l2t(res))


def self_concentration(x: np.ndarray) -> float:
    """Mean cos(x_i, mean(x)).  The single scalar that decides on/off-manifold."""
    y = l2n(x.astype(np.float32))
    m = l2n(y.mean(0, keepdims=True))
    return float((y * m).sum(1).mean())


def fit_theta_on_grid(model, Ztr_t, rows, dev, target: float,
                      n_grid: int = 73, n_rows: int = 2048
                      ) -> tuple[float, list[dict]]:
    """Choose theta so the ACHIEVED self-concentration matches `target`, by GRID
    SEARCH over [0, pi] on TRAIN rows.

    TWO CORRECTIONS THIS IMPLEMENTS, both forced by measurement
    -----------------------------------------------------------
    1. It runs AFTER training, not before.  At initialisation the read-out heads
       are zero-initialised, so the residual has no direction yet and the arc is
       degenerate: the first attempt measured c_self = 1.0000 at theta = 0 and
       0.9999 at theta = pi/2, i.e. the solve was being asked to interpolate
       between two identical points.

    2. It is a GRID SEARCH, not a bisection over [0, pi/2].  With a trained model
       the arc is NOT monotone: measured on sub-08, c_self was 0.7438 at
       theta = pi/2 and 0.8615 at 69.8 deg, so ADDING the mean CONCENTRATES the
       conditions.  The residual cone is therefore already tilted towards the
       mean, and reaching a concentration BELOW the residual's own requires
       theta > pi/2, i.e. SUBTRACTING the mean -- a branch the bisection range
       excluded, which is why it reported the target as unreachable.  A
       non-monotone arc also invalidates bisection as such, so the whole curve is
       sampled and reported.

    LEAK-FREE: measured on TRAIN rows against a TRAIN target statistic.  The test
    set is not involved, so this is not the test-set hyper-parameter selection
    that this project already removed.

    Returns the angle and the full (theta, c_self) curve, so the relation is
    reported as a measurement rather than asserted.
    """
    rows = np.asarray(rows)[:n_rows]
    curve: list[dict] = []
    best_th, best_err, best_c = math.pi / 2, 1e9, float("nan")
    model.eval()
    for k in range(n_grid):
        th = math.pi * k / (n_grid - 1)
        model.res_angle = float(th)
        got = []
        with torch.no_grad():
            for i in range(0, len(rows), 1024):
                r = rows[i:i + 1024]
                sem, _ = model.forward_all(Ztr_t[r].to(dev))
                got.append(sem["fused"].float().cpu().numpy())
        c = self_concentration(np.concatenate(got, 0))
        curve.append({"theta_rad": float(th), "theta_deg": math.degrees(float(th)),
                      "c_self": c, "err": abs(c - target)})
        if abs(c - target) < best_err:
            best_err, best_th, best_c = abs(c - target), float(th), c
    model.train()
    lo = min(curve, key=lambda d: d["c_self"])
    print(f"[ocf] theta grid ({n_grid} pts x {len(rows)} TRAIN rows): target c_self "
          f"{target:.4f} -> theta* {best_th:.4f} rad ({math.degrees(best_th):.1f} deg), "
          f"achieved {best_c:.4f} (|err| {best_err:.4f})")
    print(f"[ocf]   arc c_self: 0deg {curve[0]['c_self']:.4f} | "
          f"min {lo['c_self']:.4f} @ {lo['theta_deg']:.1f}deg | "
          f"90deg {curve[n_grid // 2]['c_self']:.4f} | 180deg {curve[-1]['c_self']:.4f}")
    if best_err > 0.05:
        print(f"[ocf]   NOTE: the arc does not reach {target:.4f} within 0.05; the "
              f"closest reachable point is {lo['c_self']:.4f} @ "
              f"{lo['theta_deg']:.1f} deg. theta* is that point, not an exact match.")
    return best_th, curve


def load_z(args, sid: int, split: str) -> np.ndarray:
    """Encoder representation for one subject/split.

    `shared_r` (default): the raw shared backbone output (`self.shared(x)`, 1024-d),
    which the measurement in the module docstring found to be strictly better than
    the pipeline's own export.  Falls back to `z_eeg_proj` with a loud warning if
    the export is missing, so a missing file cannot silently change the experiment.
    """
    cache = Path(args.z_cache_root) / f"sub-{sid:02d}"
    p = cache / f"{args.z_source}_{split}.npy"
    if p.is_file():
        return np.load(p)
    fb = f"{args.z_root}/sub-{sid:02d}/zret/z_eeg_proj_{split}.npy"
    if args.z_source == "shared_r":
        print(f"[WARN] {p} missing -> falling back to {fb}. Run "
              f"ocf_export_ss_parts.py --subject {sid} first; the fallback is the "
              f"WORSE representation and will not reproduce the reported numbers.")
    return np.load(fb)


def mlp(i: int, h: int, o: int, layers: int = 2, drop: float = 0.0) -> nn.Sequential:
    mods: list[nn.Module] = [nn.Linear(i, h), nn.GELU()]
    if drop > 0:
        mods.append(nn.Dropout(drop))
    for _ in range(layers - 1):
        mods += [nn.Linear(h, h), nn.GELU()]
        if drop > 0:
            mods.append(nn.Dropout(drop))
    mods.append(nn.Linear(h, o))
    return nn.Sequential(*mods)


def low_rank(code: int, rank: int) -> nn.Sequential:
    """A rank-`rank` map code -> code.  The second factor is zero-initialised so
    the exchange contributes exactly nothing at step 0 and has to be earned.
    `rank = 0` yields a module that is a no-op, which reproduces full tower
    independence and is the ablation arm."""
    if rank <= 0:
        return nn.Sequential()
    a = nn.Linear(code, rank, bias=False)
    b = nn.Linear(rank, code, bias=False)
    nn.init.zeros_(b.weight)
    return nn.Sequential(a, b)


def amplitude_equalise(x: torch.Tensor) -> torch.Tensor:
    """Divide each sample by its own global std.

    Without this the low-frequency latent's discriminative loss is dominated by
    the per-sample gain (mean colour / contrast), and a head that gets the gain
    right scores well while placing the layout arbitrarily.  The ranking-side
    measurement used the raw band and still reached 2-way 0.855, so the loss is
    computed on raw and equalised versions both -- see `lay_heads`."""
    s = x.flatten(1).std(1, keepdim=True).clamp_min(1e-6)
    return x / s.view(-1, *([1] * (x.dim() - 1)))


def var_band(pred: torch.Tensor, target: torch.Tensor, lo: float = 0.4,
             hi: float = 2.0) -> torch.Tensor:
    tn = target.std(0).norm()
    pn = pred.std(0).norm()
    return (F.relu(lo * tn - pn) + F.relu(pn - hi * tn)) / (tn + 1e-8)


def clip_checked(params, max_norm: float) -> tuple[float, bool]:
    """Clip gradients, but never silently.

    WHY THIS WRAPPER EXISTS
    -----------------------
    `torch.nn.utils.clip_grad_norm_` computes `clip_coef = max_norm / (total + 1e-6)`
    and clamps it to at most 1.  When `total` is `inf` -- which needs only ONE
    tensor whose fp32 norm overflows, not one non-finite element -- `clip_coef`
    becomes 0 and the call multiplies EVERY gradient by exactly zero.  The result
    is indistinguishable from a working step in the logs, and it is precisely how
    the previous TDM-DT run stayed frozen for 754 consecutive steps while
    reporting a plausible-looking loss.

    So the total norm is checked here and the caller is told whether the step is
    usable.  A non-finite total means the step must be skipped and COUNTED, and
    the count is reported, because a run whose gradients were discarded is not a
    run whose numbers mean anything.
    """
    total = torch.nn.utils.clip_grad_norm_(params, max_norm, error_if_nonfinite=False)
    ok = bool(torch.isfinite(total))
    return (float(total) if ok else float("inf")), ok


def multi_pos_nce(pred: torch.Tensor, targ: torch.Tensor, tix: torch.Tensor,
                  tau: float) -> torch.Tensor:
    logits = (l2t(pred) @ l2t(targ).T) / tau
    pos = tix[:, None] == tix[None, :]
    lse_pos = torch.logsumexp(torch.where(pos, logits, torch.full_like(logits, -1e9)), dim=-1)
    return (torch.logsumexp(logits, dim=-1) - lse_pos).mean()


def build_concept_bank(clip_text_dir: Path, captions_jsonl: Path):
    """Train-only concept gallery + per-train-row labels, alignment VERIFIED."""
    phrases = json.loads((clip_text_dir / "train" / "concept_phrases.json").read_text(encoding="utf-8"))
    bank = l2n(np.load(clip_text_dir / "train" / "text_concept_clip.npy").astype(np.float32))
    if bank.shape[0] != len(phrases):
        raise SystemExit(f"[FATAL] concept bank {bank.shape} vs phrases {len(phrases)}")
    index = {c: i for i, c in enumerate(phrases)}

    dirs: list[str] = []
    for line in captions_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = json.loads(line)["path"]
        dirs.append(p.rsplit("/", 1)[0].rsplit("/", 1)[-1])
    concepts = [d.split("_", 1)[1].replace("_", " ") for d in dirs]
    missing = sorted({c for c in concepts if c not in index})
    if missing:
        raise SystemExit(f"[FATAL] {len(missing)} train concepts absent from the gallery, e.g. {missing[:5]}")
    return bank, np.asarray([index[c] for c in concepts], dtype=np.int64), phrases


def calibrate_quantile(x: np.ndarray, ref: np.ndarray,
                       *, two_sided: bool = True) -> tuple[np.ndarray, dict]:
    """Move each row's concentration c_i = cos(x_i, mean) onto the quantiles of
    the reference's own concentration distribution.

    Label-free and test-free: `ref` is the TRAIN IP bank, `x` is our condition.

    TWO-SIDED, AND WHY THAT IS A FIX RATHER THAN A FEATURE.  The previous version
    could only INCREASE concentration (`a in [0, 1]` interpolating toward the mean)
    and returned `a = 0` -- an exact no-op -- for any row already more concentrated
    than its target quantile.  On sub-08 the conditions sat at c_self ~0.999995
    against a bank reference of ~0.6, so EVERY row took the no-op branch and the
    stage whose entire purpose was to fix the concentration did nothing at all.  It
    reported `ratio_after != ratio_before` in the log while returning its input
    unchanged, so the failure was invisible.

    The fix uses the fact that the whole family is a ONE-PARAMETER GEOMETRY with a
    closed form.  Write `x = u*m + v*orth` with `orth` unit and orthogonal to `m`
    (`u = x.m`, `v = ||x - (x.m)m||`), and let `t` be the target cosine:

      * RAISE c  (t > c_x):  y = l2(m*(u + a(1-u)) + orth*v*(1-a)), so
            c(a) = P / sqrt(P^2 + Q^2),  P = u + a(1-u),  Q = v(1-a),
        which is solved exactly by
            a = (k*v - u) / ((1-u) + k*v),   k = t / sqrt(1 - t^2).
      * LOWER c  (t < c_x):  y = l2(m*u + orth*(v + b)), so c(b) = u / sqrt(u^2 +
        (v+b)^2), which is solved exactly by
            b = u * sqrt(1/t^2 - 1) - v.

    No bisection, no 40-step grid, and no residual error: both branches are exact
    to floating point, so `max_abs_error` below can be reported honestly.  A row
    with `u <= 0` cannot be moved DOWN to a positive target by adding orthogonal
    mass (its cosine approaches 0 from below), so it is left alone and counted.

    `two_sided=False` reproduces the old one-sided behaviour and exists only so the
    change can be ablated; the default is the correct geometry.
    """
    x = l2n(x.astype(np.float32))
    ref = l2n(ref.astype(np.float32))
    m = l2n(x.mean(0, keepdims=True))[0]
    mr = l2n(ref.mean(0, keepdims=True))[0]
    c_ref = ref @ mr
    c_x = x @ m

    order = np.argsort(c_x)
    q = (np.arange(len(c_x)) + 0.5) / len(c_x)
    tgt = np.empty_like(c_x)
    tgt[order] = np.quantile(c_ref, q)

    # decompose every row into (component along m) + (orthogonal remainder)
    u = c_x                                        # x.m, since m is unit
    par = u[:, None] * m[None, :]
    rem = x - par
    v = np.linalg.norm(rem, axis=1)
    orth = rem / np.clip(v[:, None], 1e-8, None)

    y = x.copy()
    a_out = np.zeros(len(x), dtype=np.float32)
    b_out = np.zeros(len(x), dtype=np.float32)
    n_noop = n_up = n_down = n_blocked = 0

    for i in range(len(x)):
        if abs(tgt[i] - c_x[i]) <= 1e-6:
            n_noop += 1
            continue
        t = float(np.clip(tgt[i], 1e-4, 1.0 - 1e-6))
        if t > c_x[i]:
            k = t / math.sqrt(max(1.0 - t * t, 1e-12))
            den = (1.0 - float(u[i])) + k * float(v[i])
            if abs(den) < 1e-12:
                n_blocked += 1
                continue
            a = (k * float(v[i]) - float(u[i])) / den
            a = float(np.clip(a, 0.0, 1.0))
            y[i] = l2n(((1 - a) * x[i] + a * m)[None])[0]
            a_out[i] = a
            n_up += 1
        else:
            if not two_sided or u[i] <= 1e-8:
                # a row whose mean-component is non-positive cannot reach a positive
                # target by adding orthogonal mass; leaving it is the honest choice
                n_blocked += 1
                continue
            b = float(u[i]) * math.sqrt(max(1.0 / (t * t) - 1.0, 0.0)) - float(v[i])
            b = max(b, 0.0)
            y[i] = l2n((x[i] + b * orth[i])[None])[0]
            b_out[i] = b
            n_down += 1

    y = l2n(y.astype(np.float32))
    c_after = y @ m
    rep = {
        "c_self_ours_before": float(c_x.mean()),
        "c_self_ref": float(c_ref.mean()),
        "ratio_before": float(c_x.mean() / max(c_ref.mean(), 1e-8)),
        "c_self_ours_after": float(c_after.mean()),
        "ratio_after": float(c_after.mean() / max(c_ref.mean(), 1e-8)),
        "alpha_mean": float(a_out.mean()),
        "alpha_zero_frac": float((a_out <= 1e-6).mean()),
        # the numbers that make an ineffective calibration IMPOSSIBLE to miss
        "n_rows": int(len(x)),
        "n_noop": n_noop, "n_raised": n_up, "n_lowered": n_down,
        "n_blocked": n_blocked,
        "mean_scale_added": float(b_out.mean()),
        "achieved_vs_target_mae": float(np.abs(c_after - tgt).mean()),
        "elem_std_ours_before": float(x.std(0).mean()),
        "elem_std_ref": float(ref.std(0).mean()),
        "elem_std_ours_after": float(y.std(0).mean()),
        "two_sided": bool(two_sided),
        "note": ("EXACT two-sided quantile match of c_self=cos(x,mean) onto the TRAIN "
                 "IP bank's own c_self distribution; closed form, label-free, "
                 "test-free. `n_noop` is the count of rows the old one-sided version "
                 "left untouched; a run where n_noop == n_rows means the stage did "
                 "nothing."),
    }
    return y.astype(np.float32), rep


# ---------------------------------------------------------------- model

class OCFNet(nn.Module):
    """Two parallel trunks, rank-r bidirectional exchange, discriminative layout
    channel, centred residual readout, single deterministic IP producer."""

    def __init__(self, in_dim: int = 512, code: int = 768, img_dim: int = 1280,
                 txt_dim: int = 1024, ip_dim: int = 1024, ch: int = 4,
                 spatial: int = 64, lay_layers: int = 3, drop: float = 0.15,
                 rank: int = 8, res_angle: float = math.pi / 2):
        super().__init__()
        self.ch, self.spatial, self.ip_dim, self.rank = ch, spatial, ip_dim, rank
        # theta for the spherical combination of the train mean and the head
        # residual; see spherical().  pi/2 reproduces the historical behaviour.
        self.res_angle = float(res_angle)

        self.sem_trunk = mlp(in_dim, code, code, 2, drop)
        self.lay_trunk = mlp(in_dim, code, code, lay_layers, drop)
        # rank-r bidirectional exchange; no-op at init, no-op entirely if rank==0
        self.ex_lay2sem = low_rank(code, rank)
        self.ex_sem2lay = low_rank(code, rank)

        # --- semantic readout (residual over a TRAIN-split mean base)
        self.h_image = mlp(code, code, img_dim, 2, drop)
        for g in GRANS:
            setattr(self, f"h_{g}", mlp(code, code, txt_dim, 1, drop))
        self.h_concept = mlp(code, code, txt_dim, 2, drop)
        self.h_fuse = mlp(code + 4 * txt_dim + txt_dim + img_dim, code, ip_dim, 2, drop)

        # --- layout readout: low-frequency VAE latent only
        self.h_struct = mlp(code, code, ch * spatial * spatial, 2, drop)

        self.register_buffer("mu_image", torch.zeros(img_dim), persistent=True)
        for g in GRANS:
            self.register_buffer(f"mu_{g}", torch.zeros(txt_dim), persistent=True)
        self.register_buffer("mu_concept", torch.zeros(txt_dim), persistent=True)
        self.register_buffer("mu_ip", torch.zeros(ip_dim), persistent=True)
        self.register_buffer("mu_struct", torch.zeros(ch, spatial, spatial), persistent=True)

        # Small random init, NOT exact zero -- see the note on `l2t`: a zero
        # residual makes the condition exactly the zero vector at step 0, and the
        # 1/||x|| Jacobian then overflows the fp32 gradient norm, which makes
        # `clip_grad_norm_` multiply every gradient by zero and freeze the run.
        # This exact defect was measured on the TDM-DT run (26 epochs, 754 steps,
        # no parameter movement); OCFNet shares the read-out, so it is fixed here
        # too rather than left as a landmine.
        for head in [self.h_image, self.h_concept] + [getattr(self, f"h_{g}") for g in GRANS]:
            last = [m for m in head.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.normal_(last.weight, std=1e-2)
            nn.init.zeros_(last.bias)
        last = [m for m in self.h_struct.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.normal_(last.weight, std=1e-2)
        nn.init.zeros_(last.bias)

    @property
    def mu_dict(self) -> dict[str, torch.Tensor]:
        return {g: getattr(self, f"mu_{g}") for g in GRANS}

    def set_means(self, mu: dict[str, np.ndarray]) -> None:
        with torch.no_grad():
            self.mu_image.copy_(torch.from_numpy(mu["image"]))
            for g in GRANS:
                getattr(self, f"mu_{g}").copy_(torch.from_numpy(mu[g]))
            self.mu_concept.copy_(torch.from_numpy(mu["concept"]))
            self.mu_ip.copy_(torch.from_numpy(mu["ip"]))
            self.mu_struct.copy_(torch.from_numpy(mu["struct"]))

    def forward_all(self, x: torch.Tensor) -> tuple[dict, dict]:
        s0 = self.sem_trunk(x)
        p0 = self.lay_trunk(x)
        s = s0 + self.ex_lay2sem(p0)
        p = p0 + self.ex_sem2lay(s0)

        res = {g: getattr(self, f"h_{g}")(s) for g in GRANS}
        res_img = self.h_image(s)
        res_con = self.h_concept(s)
        # spherical mean/residual combination (see spherical()); theta is a
        # property of the model so the same checkpoint can be re-exported at
        # other angles for the ablation without retraining.
        th = self.res_angle
        full = {g: spherical(self.mu_dict[g], res[g], th) for g in GRANS}
        f_img = spherical(self.mu_image, res_img, th)
        f_con = spherical(self.mu_concept, res_con, th)
        fuse = spherical(self.mu_ip, self.h_fuse(torch.cat(
            [s] + [full[g] for g in GRANS] + [f_con, f_img], -1)), th)

        struct = self.mu_struct + self.h_struct(p).view(-1, self.ch, self.spatial, self.spatial)
        sem = {"image": f_img, "concept": f_con, "fused": fuse,
               "_res": res, "_res_image": res_img, "_res_concept": res_con,
               "_s": s, **full}
        lay = {"struct": struct, "_p": p}
        return sem, lay

    def memory(self, q: torch.Tensor, bank: torch.Tensor, tau: float, k: int = 16) -> torch.Tensor:
        sim = l2t(q) @ l2t(bank).T
        topv, topi = sim.topk(min(k, sim.shape[1]), dim=-1)
        return l2t((torch.softmax(topv / tau, dim=-1).unsqueeze(-1) * l2t(bank)[topi]).sum(1))

    def sample_ip(self, q: torch.Tensor, bank: torch.Tensor, tau: float,
                  k: int, gen: torch.Generator) -> torch.Tensor:
        """One sample from the top-k soft-memory posterior.

        The EEG posterior is one-to-many, so the proposal has to be a set; a
        point estimate of a one-to-many target is the conditional mean, which is
        exactly the collapse this project already measured (a CONSTANT vector
        scored cos 0.6147 to the test IPs)."""
        sim = l2t(q) @ l2t(bank).T
        topv, topi = sim.topk(min(k, sim.shape[1]), dim=-1)
        w = torch.softmax(topv / tau, dim=-1)
        pick = torch.multinomial(w, 1, generator=gen)
        return l2t(l2t(bank)[topi.gather(1, pick).squeeze(1)])


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-subjects", type=int, nargs="+", required=True)
    ap.add_argument("--test-subject", type=int, required=True)
    ap.add_argument("--z-root", type=str, default=str(NB_ROOT / "outputs/hcma_10subj"))
    ap.add_argument("--targets-dir", type=str, default=str(NB_ROOT / "outputs/g2/targets"))
    ap.add_argument("--clip-text-dir", type=str, default=str(NB_ROOT / "outputs/nda_ss/sub-08/clip_text"))
    ap.add_argument("--captions-jsonl", type=str, default=str(NB_ROOT / "outputs/g2/captions/captions_train.jsonl"))
    ap.add_argument("--ip-train-npy", type=str,
                    default="/project/peilab/why/eeg-brainit/outputs/atm_bridge/clip_img_train_1024.npy")
    ap.add_argument("--ip-test-npy", type=str,
                    default="/project/peilab/why/eeg-brainit/outputs/atm_bridge/clip_img_test_1024.npy")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--z-source", type=str, default="shared_r",
                    choices=["shared_r", "z_eeg_proj", "fused", "cat"],
                    help="which encoder component feeds the towers.  `shared_r` is the "
                         "measured best (see the module docstring); `z_eeg_proj` is the "
                         "old pipeline export (1024->512 projector) kept for comparison.")
    ap.add_argument("--z-cache-root", type=str, default=str(NB_ROOT / "outputs/ocf/ss_parts"),
                    help="per-subject exports from ocf_export_ss_parts.py")
    ap.add_argument("--rank", type=int, default=0,
                    help="rank of the bidirectional trunk exchange; 0 = independent "
                         "trunks (default, see the docstring)")
    ap.add_argument("--res-angle", type=float, default=-1.0,
                    help="theta for the spherical mean/residual combination, in radians. "
                         "-1 (default) = auto = arccos(c_self of the TRAIN IP bank), i.e. "
                         "match the reference distribution's own concentration. "
                         "pi/2 reproduces the historical l2(mu+res) behaviour.")
    ap.add_argument("--export-angles", type=str, default="auto,1.5708,1.2180",
                    help="comma list of extra angles to export conditions at, so the "
                         "theta choice is an empirical result rather than an assertion")
    ap.add_argument("--epochs", type=int, default=26)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--tau-nce", type=float, default=0.07)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--w-img", type=float, default=0.5)
    ap.add_argument("--w-res", type=float, default=0.4)
    ap.add_argument("--w-ip", type=float, default=1.0)
    ap.add_argument("--w-mem", type=float, default=1.0)
    ap.add_argument("--w-nce", type=float, default=1.0)
    ap.add_argument("--w-cls", type=float, default=1.0)
    ap.add_argument("--w-fuse-cls", type=float, default=0.3)
    ap.add_argument("--w-laycos", type=float, default=1.0,
                    help="structure fidelity: cosine on the amplitude-equalised LF latent")
    ap.add_argument("--w-laynce", type=float, default=1.0,
                    help="THE OCF ADDITION: multi-positive InfoNCE on the LF latent, "
                         "which is what makes the channel usable as a verifier")
    ap.add_argument("--n-cand", type=int, default=4, help="SFV candidates per row")
    ap.add_argument("--tau-sample", type=float, default=0.10)
    ap.add_argument("--k-sample", type=int, default=32)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--gate-quantile", type=float, default=0.5)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    (out / "conds").mkdir(parents=True, exist_ok=True)
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    T = Path(args.targets_dir)
    stag = f"sub-{args.test_subject:02d}"

    cbank_np, clab_np, phrases = build_concept_bank(Path(args.clip_text_dir), Path(args.captions_jsonl))
    n_cls = cbank_np.shape[0]
    print(f"[ocf] concept gallery {cbank_np.shape} ({len(set(clab_np.tolist()))} distinct)")

    ip_bank_np = np.load(args.ip_train_npy).astype(np.float32)
    ip_test_np = np.load(args.ip_test_npy).astype(np.float32)
    bank = torch.from_numpy(l2n(ip_bank_np)).to(dev)
    n_bank = bank.shape[0]
    cbank = torch.from_numpy(cbank_np).to(dev)
    clab = torch.from_numpy(clab_np).to(dev)

    keys = ("image", "concept", "overall", "subject", "background", "detail")
    tgt: dict[str, torch.Tensor] = {}
    for k in keys:
        src = T / f"sem_{k}_train.npy"
        if not src.is_file():
            src = T / "sem_concept_tmpl_train.npy" if k == "concept" else src
        tgt[k] = torch.from_numpy(np.load(src).astype(np.float32))
    tgt["struct"] = torch.from_numpy(np.load(T / "perc_struct_train.npy").astype(np.float32))
    tgt["ip"] = torch.from_numpy(l2n(ip_bank_np))

    zs = []
    for s in args.train_subjects:
        z = load_z(args, s, "train").astype(np.float32)
        if z.shape[0] != n_bank:
            raise SystemExit(f"[FATAL] sub-{s:02d} z rows {z.shape[0]} != bank {n_bank}")
        zs.append(z)
    Ztr = np.concatenate(zs, 0)
    tidx = np.tile(np.arange(n_bank), len(args.train_subjects))
    n_tr = Ztr.shape[0]

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_bank)
    val_t = np.zeros(n_bank, dtype=bool)
    val_t[perm[:int(n_bank * args.val_frac)]] = True
    is_val = val_t[tidx]
    tr_sel = np.where(~is_val)[0]
    va_sel = np.where(is_val)[0]
    va_sel = va_sel[np.arange(len(va_sel)) % max(1, len(args.train_subjects)) == 0]
    if args.limit_train > 0:
        tr_sel = tr_sel[:args.limit_train]
    if args.limit_val > 0:
        va_sel = va_sel[:args.limit_val]

    mu = {k: tgt[k][tidx[tr_sel]].mean(0).cpu().numpy().astype(np.float32) for k in keys}
    mu["ip"] = tgt["ip"][tidx[tr_sel]].mean(0).cpu().numpy().astype(np.float32)
    mu["struct"] = tgt["struct"][tidx[tr_sel]].mean(0).cpu().numpy().astype(np.float32)
    print("[ocf] ||mu_ip|| = %.4f (constant-condition cos floor)" % float(np.linalg.norm(mu["ip"])))

    # ---- theta: the manifold-position knob, derived from the data, not chosen
    c_self_ref = self_concentration(ip_bank_np)
    theta_auto = float(math.acos(min(max(c_self_ref, 1e-3), 1.0 - 1e-6)))
    if args.res_angle < 0:
        print(f"[ocf] theta requested = AUTO; the closed form arccos(c_self_ref) = "
              f"arccos({c_self_ref:.4f}) = {theta_auto:.4f} rad "
              f"({math.degrees(theta_auto):.1f} deg) is only a FIRST GUESS -- the "
              f"residual is not orthogonal to the mean, so the angle is solved for "
              f"on train rows below.")
    else:
        print(f"[ocf] theta = {args.res_angle:.4f} rad "
              f"({math.degrees(args.res_angle):.1f} deg)  [c_self_ref = {c_self_ref:.4f}]")
    print(f"[ocf]   theta=pi/2 ({math.pi/2:.4f}) = residual-dominated (historical); "
          f"theta=0 = pure train mean (degenerate)")

    tgt = {k: v.to(dev) for k, v in tgt.items()}
    clab = clab.to(dev)
    Zte_t = torch.from_numpy(load_z(args, args.test_subject, "test").astype(np.float32)).to(dev)
    in_dim = int(Zte_t.shape[1])
    ite_np = l2n(ip_test_np)
    Ztr_t = torch.from_numpy(Ztr)
    print(f"[ocf] train rows {n_tr} (trainsel {len(tr_sel)}, valsel {len(va_sel)}) | bank {n_bank} "
          f"| concepts {n_cls} | test {tuple(Zte_t.shape)} | rank {args.rank} | z {args.z_source}")

    model = OCFNet(in_dim=in_dim, drop=args.dropout, rank=args.rank,
                   res_angle=args.res_angle).to(dev)
    model.set_means(mu)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[ocf] params {n_par/1e6:.2f}M device {dev}")

    # TRAINING ANGLE: pi/2 by default, i.e. the residual dominates the read-out.
    # This is exactly what g3f's `l2(mu + res)` did in effect (the residual there
    # was unbounded, measured at norm 8.11-9.68 against a mean of norm 1, so the
    # mean contributed ~1%).  Training at that same point is deliberate: it makes
    # the theta sweep below a pure READ-OUT-POSITION ablation on weights that are
    # otherwise identical to the g3f recipe, with no other confound.
    #
    # theta is NOT solved before training: at initialisation the heads are
    # zero-initialised, so the residual has no direction and the arc is
    # degenerate (measured: c_self 1.0000 at theta=0 and 0.9999 at theta=pi/2).
    # It is solved AFTER training, on TRAIN rows, by fit_theta_on_grid.
    model.res_angle = float(args.res_angle if args.res_angle >= 0 else math.pi / 2)
    print(f"[ocf] train theta = {model.res_angle:.4f} rad "
          f"({math.degrees(model.res_angle):.1f} deg) for all {args.epochs} epochs")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = max(1, len(tr_sel) // args.batch_size)
    total_steps = args.epochs * steps
    if total_steps >= 20:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.25)
    else:
        sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=total_steps)

    best = {"score": -1e9, "epoch": -1}
    last_path, best_path = out / "last.pth", out / "best.pth"
    start_ep = 0
    if args.resume and last_path.is_file():
        try:
            ck = torch.load(last_path, map_location=dev, weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            start_ep = ck["epoch"] + 1
            best = ck.get("best", best)
            print(f"[ocf] resumed at epoch {start_ep} (best {best})")
        except Exception as e:                                     # noqa: BLE001
            print(f"[ocf] resume failed ({e}); starting fresh")

    def disc(a: np.ndarray) -> dict[str, float]:
        s = l2n(a) @ ite_np.T
        n = len(s)
        r = np.random.default_rng(0).permutation(n)
        ok = np.arange(n) != r
        return {"top1": float(np.mean([i in np.argsort(-s[i])[:1] for i in range(n)])),
                "top5": float(np.mean([i in np.argsort(-s[i])[:5] for i in range(n)])),
                "twoway": float(np.mean(s[np.arange(n), np.arange(n)][ok] > s[np.arange(n), r][ok]))}

    def evaluate(sel: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
        model.eval()
        acc: dict[str, list[float]] = {}
        margins: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(sel), 1024):
                rows = sel[i:i + 1024]
                x = Ztr_t[rows].to(dev)
                ti = torch.from_numpy(tidx[rows]).to(dev)
                sem, lay = model.forward_all(x)
                qf = sem["fused"]
                tgtip = tgt["ip"][ti]
                lg = sem["concept"] @ cbank.T
                top2 = lg.topk(2, dim=-1).values
                margins.append((top2[:, 0] - top2[:, 1]).float().cpu().numpy())
                mem = model.memory(qf, bank, args.tau)
                st, stg = lay["struct"], tgt["struct"][ti]
                se, sge = amplitude_equalise(st), amplitude_equalise(stg)
                v = {
                    "image": (sem["image"] * tgt["image"][ti]).sum(-1).mean(),
                    "fused_to_ip": (qf * tgtip).sum(-1).mean(),
                    "mem_to_ip": (mem * tgtip).sum(-1).mean(),
                    "nce": multi_pos_nce(qf, tgtip, ti, args.tau_nce),
                    "cls_top1": (lg.argmax(1) == clab[ti]).float().mean(),
                    "cls_top5": (lg.topk(5, dim=-1).indices == clab[ti][:, None]).any(-1).float().mean(),
                    "lay_cos": (l2t(st.flatten(1)) * l2t(stg.flatten(1))).sum(-1).mean(),
                    "lay_cos_eq": (l2t(se.flatten(1)) * l2t(sge.flatten(1))).sum(-1).mean(),
                    "lay_nce": multi_pos_nce(se.flatten(1), sge.flatten(1), ti, args.tau_nce),
                    "struct_std": st.std(0).norm() / (tgt["struct"].std(0).norm() + 1e-8),
                    "res_norm": sem["_res"]["subject"].std(0).norm() / (tgt["subject"].std(0).norm() + 1e-8),
                    "exchange_ratio": ((model.ex_lay2sem[1].weight.norm() /
                                        (model.sem_trunk[-1].weight.norm() + 1e-8)).detach()
                                       if args.rank > 0 else torch.zeros(()).to(dev)),
                }
                for k, t in v.items():
                    acc.setdefault(k, []).append(float(t.detach()))
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        m["score"] = (0.25 * m["mem_to_ip"] + 0.15 * m["fused_to_ip"] + 0.15 * m["cls_top1"]
                      + 0.08 * m["image"] + 0.17 * m["lay_cos_eq"] + 0.20 * (-m["lay_nce"]))
        model.train()
        return m, np.concatenate(margins, 0)

    history: list[dict] = []
    val_margin = np.zeros(0)
    for ep in range(start_ep, args.epochs):
        model.train()
        perm2 = rng.permutation(len(tr_sel))
        run: dict[str, float] = {}
        nb = 0
        t0 = time.time()
        for b in range(steps):
            rows = tr_sel[perm2[b * args.batch_size:(b + 1) * args.batch_size]]
            if len(rows) < 2:
                continue
            x = Ztr_t[rows].to(dev)
            ti = torch.from_numpy(tidx[rows]).to(dev)
            sem, lay = model.forward_all(x)
            ip_t = tgt["ip"][ti]

            l_img = (1 - (sem["image"] * tgt["image"][ti]).sum(-1)).mean()
            l_res = sum((1 - (sem[g] * tgt[g][ti]).sum(-1)).mean() for g in GRANS) / len(GRANS)
            l_ip = (1 - (sem["fused"] * ip_t).sum(-1)).mean()
            mem = model.memory(sem["fused"], bank, args.tau)
            l_mem = (1 - (mem * ip_t).sum(-1)).mean()
            l_nce = (multi_pos_nce(sem["fused"], ip_t, ti, args.tau_nce)
                     + multi_pos_nce(mem, ip_t, ti, args.tau_nce)
                     + multi_pos_nce(sem["image"], tgt["image"][ti], ti, args.tau_nce))

            lg = sem["concept"] @ cbank.T
            l_cls = F.cross_entropy(lg / args.tau, clab[ti])
            l_fcls = F.cross_entropy((sem["fused"] @ cbank.T) / args.tau, clab[ti])

            # ---- layout channel: fidelity AND discriminability
            st, stg = lay["struct"], tgt["struct"][ti]
            se, sge = amplitude_equalise(st), amplitude_equalise(stg)
            l_laycos = (1 - (l2t(se.flatten(1)) * l2t(sge.flatten(1))).sum(-1)).mean()
            l_laynce = multi_pos_nce(se.flatten(1), sge.flatten(1), ti, args.tau_nce)

            hinges = (var_band(st, stg)
                      + var_band(sem["fused"], ip_t)
                      + var_band(sem["image"], tgt["image"][ti])
                      + var_band(sem["concept"], tgt["concept"][ti])
                      + sum(var_band(sem[g], tgt[g][ti]) for g in GRANS) / len(GRANS))

            loss = (args.w_img * l_img + args.w_res * l_res + args.w_ip * l_ip
                    + args.w_mem * l_mem + args.w_nce * l_nce + args.w_cls * l_cls
                    + args.w_fuse_cls * l_fcls + args.w_laycos * l_laycos
                    + args.w_laynce * l_laynce + 0.5 * hinges)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            for k, v in (("loss", loss), ("img", l_img), ("res", l_res), ("ip", l_ip),
                         ("mem", l_mem), ("nce", l_nce), ("cls", l_cls), ("fcls", l_fcls),
                         ("laycos", l_laycos), ("laynce", l_laynce)):
                run[k] = run.get(k, 0.0) + float(v.detach())
            nb += 1

        tr_m = {k: v / max(nb, 1) for k, v in run.items()}
        va_m, val_margin = evaluate(va_sel)
        rec = {"epoch": ep, "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1),
               **{f"tr_{k}": round(v, 4) for k, v in tr_m.items()},
               **{f"va_{k}": round(v, 4) for k, v in va_m.items()}}
        history.append(rec)
        print(f"[ep{ep}] " + " ".join(f"{k}={va_m[k]:.4f}" for k in
                                      ("mem_to_ip", "fused_to_ip", "nce", "cls_top1", "cls_top5",
                                       "image", "lay_cos_eq", "lay_nce", "struct_std",
                                       "exchange_ratio", "score")))
        if va_m["score"] > best["score"]:
            best = {"score": va_m["score"], "epoch": ep, **{f"va_{k}": v for k, v in va_m.items()}}
            torch.save({"model": model.state_dict(), **best}, best_path)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best": best}, last_path)

    # -------------------------------------------------- export (test rows)
    if best_path.is_file():
        ck = torch.load(best_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"[ocf] loaded best epoch {ck.get('epoch')} score {ck.get('score'):.4f}")
    model.eval()
    outs: dict[str, list[np.ndarray]] = {}
    logits_te: list[np.ndarray] = []
    gen = torch.Generator(device=dev).manual_seed(args.seed + 1234)
    with torch.no_grad():
        for i in range(0, Zte_t.shape[0], 256):
            x = Zte_t[i:i + 256]
            sem, lay = model.forward_all(x)
            qf = sem["fused"]
            vals = {"ip_fused": qf, "ip_mem": model.memory(qf, bank, args.tau),
                    "concept": sem["concept"], "image": sem["image"],
                    "lf_latent": lay["struct"].flatten(1)}
            for g in GRANS:
                vals[f"gran_{g}"] = sem[g]
            # SFV candidates: K draws from the top-k soft-memory posterior
            for c in range(args.n_cand):
                vals[f"ip_samp{c}"] = model.sample_ip(qf, bank, args.tau_sample,
                                                      args.k_sample, gen)
            for k, v in vals.items():
                outs.setdefault(k, []).append(v.float().cpu().numpy())
            logits_te.append((sem["concept"] @ cbank.T).float().cpu().numpy())

    for k, v in outs.items():
        a = np.concatenate(v, 0).astype(np.float32)
        if k.startswith(("ip_", "concept", "image", "gran_")):
            a = l2n(a)
        np.save(out / "conds" / f"{k}_test.npy", a)
    LOG = np.concatenate(logits_te, 0)

    # -------------------------------------------------- theta ablation export
    # One trained checkpoint, re-exported at other angles: theta is a property of
    # the read-out, and the direction it interpolates towards (the train mean) is
    # a fixed buffer, so no retraining is needed to move along the arc.  This is
    # what turns the theta choice into a measurement.
    theta_rep: dict[str, dict] = {}
    base_theta = model.res_angle
    for spec in [s.strip() for s in args.export_angles.split(",") if s.strip()]:
        th = float(args.res_angle) if spec == "auto" else float(spec)
        if abs(th - base_theta) < 1e-6:
            tag, buf = "base", np.concatenate(outs["ip_fused"], 0)
        else:
            model.res_angle = th
            got = []
            with torch.no_grad():
                for i in range(0, Zte_t.shape[0], 256):
                    sem, _ = model.forward_all(Zte_t[i:i + 256])
                    got.append(sem["fused"].float().cpu().numpy())
            buf = np.concatenate(got, 0)
            tag = f"t{math.degrees(th):.0f}".replace(".", "")
            np.save(out / "conds" / f"ip_fused_{tag}_test.npy", l2n(buf).astype(np.float32))
            model.res_angle = base_theta
        c = self_concentration(buf)
        d = disc(buf)
        theta_rep[spec] = {"theta_rad": th, "theta_deg": math.degrees(th),
                           "c_self": c, "c_self_ref": c_self_ref,
                           "c_self_ratio": c / max(c_self_ref, 1e-8),
                           "tag": tag, **d}
        print(f"[ocf] theta {spec:<8} = {math.degrees(th):5.1f} deg | "
              f"c_self {c:.4f} (ref {c_self_ref:.4f}, ratio {c/max(c_self_ref,1e-8):.3f}) | "
              f"top1 {d['top1']:.4f}  2way {d['twoway']:.4f}")
    print("[ocf] READ: c_self_ratio near 1.0 = on the reference manifold; top1/2way "
          "say whether the excess dispersion was signal or error. Both are reported.")

    # -------------------------------------------------- OCF: manifold calibration
    # reference = the TRAIN IP bank (no test data, no labels anywhere)
    ip_fused_te = np.concatenate(outs["ip_fused"], 0).astype(np.float32)
    cal, cal_rep = calibrate_quantile(ip_fused_te, ip_bank_np)
    np.save(out / "conds" / "ip_fused_cal_test.npy", cal)
    (out / "conds" / "calibration_report.json").write_text(json.dumps(cal_rep, indent=2), encoding="utf-8")
    # calibration changes instance information; log both directions of the trade
    d_before, d_after = disc(ip_fused_te), disc(cal)
    print(f"[ocf] calibration {cal_rep['ratio_before']:.4f} -> {cal_rep['ratio_after']:.4f} "
          f"| fused 2way {d_before['twoway']:.4f} -> {d_after['twoway']:.4f} "
          f"| top1 {d_before['top1']:.4f} -> {d_after['top1']:.4f}")

    # -------------------------------------------------- SELF-PROMPT (leak-free)
    thr = float(np.quantile(val_margin, args.gate_quantile)) if val_margin.size else 0.0
    top1 = LOG.argmax(1)
    top2v = np.sort(LOG, 1)[:, -2]
    margin = LOG[np.arange(len(LOG)), top1] - top2v
    self_prompts = [f"a photo of a {phrases[i]}" for i in top1]
    gated = [self_prompts[i] if margin[i] >= thr else GENERIC_PROMPT for i in range(len(margin))]
    (out / "prompts" / "prompts_self.json").write_text(json.dumps(self_prompts, indent=1), encoding="utf-8")
    (out / "prompts" / "prompts_selfgate.json").write_text(json.dumps(gated, indent=1), encoding="utf-8")
    (out / "prompts" / "prompts_generic.json").write_text(json.dumps([GENERIC_PROMPT] * len(margin), indent=1), encoding="utf-8")
    (out / "prompts" / "selfprompt_debug.json").write_text(json.dumps({
        "gate_threshold": thr, "gate_quantile": args.gate_quantile,
        "test_margin_mean": float(margin.mean()),
        "n_gated_to_generic": int((margin < thr).sum()),
        "top1_concepts": [phrases[i] for i in top1],
        "unique_top1": int(len(set(top1.tolist()))),
        "note": ("gallery = 1654 TRAIN concepts; the 200 test concepts are disjoint "
                 "from it, so the emitted string cannot name a test class."),
    }, indent=2), encoding="utf-8")

    report = {"protocol": "ocf", "test_subject": stag, "train_subjects": args.train_subjects,
              "params_m": round(n_par / 1e6, 3), "rank": args.rank, "best": best,
              "history": history, "n_concepts": int(n_cls), "n_bank": int(n_bank),
              "mu_ip_norm": float(np.linalg.norm(mu["ip"])),
              "gate_threshold": thr, "n_gated_to_generic": int((margin < thr).sum()),
              "prompt_unique_top1": int(len(set(top1.tolist()))),
              "res_angle_rad": float(base_theta),
              "res_angle_deg": math.degrees(float(base_theta)),
              "res_angle_solved_on_train": bool(theta_trace),
              "res_angle_trace": theta_trace,
              "c_self_ref": float(c_self_ref),
              "theta_ablation": theta_rep,
              "calibration": cal_rep,
              "ip_fused_disc_before_cal": d_before,
              "ip_fused_disc_after_cal": d_after,
              "constant_baseline_cos_to_ip": float((l2n(np.tile(mu["ip"], (len(ite_np), 1))) * ite_np).sum(-1).mean())}
    for k in ("ip_fused", "ip_mem", "concept") + tuple(f"ip_samp{c}" for c in range(args.n_cand)):
        report[f"{k}_disc"] = disc(np.concatenate(outs[k], 0))
    cte_path = T / "sem_concept_tmpl_test.npy"
    if cte_path.is_file():
        cte = l2n(np.load(cte_path).astype(np.float32))
        sim = np.concatenate(outs["concept"], 0) @ cte.T
        report["concept_top1_to_oracle"] = float(np.mean(np.argmax(sim, 1) == np.arange(len(sim))))
        report["oracle_note"] = "diagnostic only; never used to build the prompt"
    report["note"] = ("cos_to_ip is NOT a usable quality metric (a constant scores "
                      f"{report['constant_baseline_cos_to_ip']:.4f}); use *_disc.")
    (out / "ocf_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[ocf] " + json.dumps({k: v for k, v in report.items()
                                 if k.endswith("_disc") or k in
                                 ("constant_baseline_cos_to_ip", "concept_top1_to_oracle",
                                  "calibration", "ip_fused_disc_after_cal")}, indent=2))
    print(f"[ocf] done -> {out}")


if __name__ == "__main__":
    main()
