#!/usr/bin/env python3
"""CPU smoke test for the T3 towers, the Spectra operator and the target builder.

Catches the failure classes a real run would hide until hour 3:
  * shape/stride errors in Dec / Vel / T3Net
  * the torch Spectra operator disagreeing with the numpy target builder
    (if these drift, training optimises a target that does not exist)
  * the low/high split not being orthogonal
  * THE MECHANISM ITSELF: in the regime where the condition carries no information
    about the high band -- which is exactly where EEG is blind (r>0.5: corr 0.0355)
    -- does the HF head still emit a field with the right texture dispersion, or
    does it collapse to zero the way the shipped L1 target does?

Tolerances: numpy's FFT is complex128, torch's is complex64, so after squaring and
summing 4096 bins the statistics agree to ~1e-6 relative, not 1e-7.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
SB = Path("/tmp/t3_smoke")
SB.mkdir(parents=True, exist_ok=True)
sys.argv = [sys.argv[0]]  # keep our own args out of t3_build_targets

import t3_build_targets as BT  # noqa: E402
import t3_train_towers as TT  # noqa: E402

FAIL = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print(f"[{'OK ' if cond else 'FAIL'}] {name} {extra}")
    if not cond:
        FAIL.append(name)


H = W = 64
C = 4
rng = np.random.default_rng(0)
yy, xx = np.mgrid[0:H, 0:W]
# Genuinely smooth content: r ~ 0.01-0.02, well inside cut=0.0625. A near-Nyquist
# test sinusoid would put ~99% of the energy in the HF band (measured) and make
# every "smooth signal" assertion meaningless.
low = (np.sin(2 * np.pi * 0.005 * xx) + np.cos(2 * np.pi * 0.008 * yy)).astype(np.float32)
x = (rng.standard_normal((6, C, H, W)).astype(np.float32) * 0.3 + low[None, None] * 2.0)

# --------------------------------------------------------------------------- #
# 1. target builder: orthogonality, band energy, statistics layout
# --------------------------------------------------------------------------- #
np.save(SB / "vae_train.npy", x.astype(np.float32))
np.save(SB / "vae_test.npy", x.astype(np.float32))
sys.argv = ["t3_build_targets.py", "--vae-train-npy", str(SB / "vae_train.npy"),
            "--vae-test-npy", str(SB / "vae_test.npy"), "--output-dir", str(SB / "tgt")]
BT.main()

lf = np.load(SB / "tgt/lf_latent_train.npy").astype(np.float32)
hf = np.load(SB / "tgt/hf_latent_train.npy").astype(np.float32)
st_np = np.load(SB / "tgt/hf_stats_train.npy")
check("stats builder returns float32", st_np.dtype == np.float32, str(st_np.dtype))
# Two separate checks, because they answer different questions:
#  (a) is the SPLIT orthogonal in exact arithmetic? (the builder's own invariant)
lo_c, hi_c = BT.band_split(x, BT.radial(H, W), 0.0625)
check("split is orthogonal in float32", float(np.abs(lo_c + hi_c - x).max()) < 1e-4,
      f"max|err|={float(np.abs(lo_c + hi_c - x).max()):.2e}")
#  (b) do the STORED files match the split, up to the deliberate float16 storage?
#      (the builder stores fp16 to match the shipped VAE latent cache, so ~1e-3
#      absolute is the quantisation floor for values of order a few)
x_rt = x.astype(np.float16).astype(np.float32)
err = float(np.abs((lf + hf) - x_rt).max())
check("stored lo+hi matches input to fp16 precision", err < 0.01, f"max|err|={err:.2e}")
check("LF dominates for smooth content",
      float((lf**2).mean()) > float((hf**2).mean()),
      f"lf={float((lf**2).mean()):.4f} hf={float((hf**2).mean()):.4f}")

z = np.zeros_like(x)
st_zero = BT.hf_stats(z, BT.radial(H, W), BT.angle(H, W), 0.0625, 16, 8, 8)
d_zero = float(np.abs(st_zero - st_np).mean())
check("zero field is a POOR fit to the HF stats target", d_zero > 0.1,
      f"mean|stats(0)-target|={d_zero:.4f} (the collapse solution must be penalised)")

# --------------------------------------------------------------------------- #
# 2. torch Spectra must reproduce the numpy statistics
# --------------------------------------------------------------------------- #
sp = TT.Spectra(H, W, 0.0625, 16, 8, 8).eval()
with torch.no_grad():
    st_t = sp(torch.from_numpy(x)).numpy()
check("Spectra layout matches builder", st_t.shape == st_np.shape,
      f"torch={st_t.shape} numpy={st_np.shape}")
d = float(np.abs(st_t - st_np).max())
rel = d / max(float(np.abs(st_np).mean()), 1e-9)
check("Spectra values match builder (fp32-FFT tolerance)", rel < 1e-5,
      f"max|diff|={d:.2e} rel={rel:.2e}")

# --------------------------------------------------------------------------- #
# 3. module shapes for both hf_head variants and both struct_input values
# --------------------------------------------------------------------------- #
for hf_head in ("pix", "cfm"):
    for struct_input in ("sem", "both"):
        m = TT.T3Net(in_dim=1024, spatial=64, ch=C, n_cells=36, loc_dim=1024,
                     use_local=True, use_hf=True, use_depth=True,
                     struct_extra=(1024 if struct_input == "both" else 0),
                     hf_head=hf_head, stats_dim=st_np.shape[1]).eval()
        o = m(torch.randn(3, 1024), torch.randn(3, 1024), torch.randn(3, 512))
        ok = (o["lf"].shape == (3, C, 64, 64) and o["z_loc"].shape == (3, 36, 1024)
              and o["depth"].shape == (3, 1, 64, 64) and o["z_hcf"].shape == (3, 1280))
        ok = ok and (("hf" in o) if hf_head == "pix" else (m.vel is not None))
        check(f"T3Net shapes hf={hf_head} struct={struct_input}", ok)
        # the texture head must not START inside the collapse state
        if hf_head == "pix":
            check(f"HF field is non-degenerate at init hf={hf_head} struct={struct_input}",
                  float(o["hf"].std()) > 1e-6, f"std={float(o['hf'].std()):.2e}")

m = TT.T3Net(in_dim=1024, spatial=64, ch=C, use_hf=False, hf_head="pix")
o = m(torch.randn(2, 1024), torch.randn(2, 1024), torch.randn(2, 512))
check("mono variant has no HF head", "hf" not in o and "tx_cond" not in o)

# --------------------------------------------------------------------------- #
# 4. THE MECHANISM TEST.
#
# Regime: the condition carries NO information about the high band -- the measured
# situation for EEG (corr 0.0355 above r>0.5). Under L1 the optimal predictor there
# is the conditional MEDIAN, and for a zero-mean HF field that is the zero field.
# The distributional head has no such incentive: only the field's statistics are
# scored, so it can satisfy the target by emitting a non-degenerate texture.
# --------------------------------------------------------------------------- #
N = 64
tgt_full = rng.standard_normal((N, C, H, W)).astype(np.float32)
r_, a_ = BT.radial(H, W), BT.angle(H, W)
_, tgt_hf = BT.band_split(tgt_full, r_, 0.0625)
tgt_stats = torch.from_numpy(BT.hf_stats(tgt_full, r_, a_, 0.0625, 16, 8, 8))
tgt_hf_t = torch.from_numpy(tgt_hf)
ref_std = float(tgt_hf_t.flatten(1).std(1).mean())
cond = torch.randn(N, 768)          # uninformative by construction
STEPS = 300
print(f"       [mechanism] target HF std={ref_std:.4f}, cond is uninformative")


def run(kind: str) -> dict:
    torch.manual_seed(0)
    if kind == "l1":
        # the shipped recipe: zero-init regressor + pixel L1 (its own collapse mode)
        head = TT.Dec(768, C, 64, hidden=256, zero_init=True)
        opt = torch.optim.AdamW(head.parameters(), lr=3e-3)
        for _ in range(STEPS):
            pred = head(cond)
            sc = tgt_hf_t.flatten(1).pow(2).mean(1).sqrt().clamp(min=1e-3).view(-1, 1, 1, 1)
            loss = torch.nn.functional.l1_loss(pred / sc, tgt_hf_t / sc)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            out = head(cond)
    elif kind == "stats":
        head = TT.Dec(768, C, 64, hidden=256, zero_init=False)
        opt = torch.optim.AdamW(head.parameters(), lr=3e-3)
        for _ in range(STEPS):
            loss = torch.nn.functional.smooth_l1_loss(sp(head(cond)), tgt_stats)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            out = head(cond)
    elif kind == "cfm":
        vel = TT.Vel(768, C, 64, hidden=256)
        opt = torch.optim.AdamW(vel.parameters(), lr=3e-3)
        sc = tgt_hf_t.flatten(1).pow(2).mean(1).sqrt().clamp(min=1e-3).view(-1, 1, 1, 1)
        x1_all = tgt_hf_t / sc
        for _ in range(STEPS):
            idx = torch.randint(0, N, (16,))
            x1 = x1_all[idx]
            c = cond[idx]
            t = torch.rand(16)
            eps = torch.randn_like(x1)
            xt = (1 - t.view(-1, 1, 1, 1)) * eps + t.view(-1, 1, 1, 1) * x1
            loss = torch.nn.functional.mse_loss(vel(xt, t, c), x1 - eps)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            xi = torch.zeros(N, C, 64, 64)
            for k in range(16):
                xi = xi + (1 / 16) * vel(xi, torch.full((N,), k / 16), cond)
            out = xi * sc     # back to target scale for a like-for-like std
    else:
        raise ValueError(kind)
    with torch.no_grad():
        return {"ratio": float(out.flatten(1).std(1).mean()) / max(ref_std, 1e-9),
                "stats_l1": float(torch.nn.functional.l1_loss(sp(out), tgt_stats))}


res = {k: run(k) for k in ("l1", "stats", "cfm")}
for k, v in res.items():
    print(f"       [{k:>5}] out/ref std = {v['ratio']:.3f}   stats_l1={v['stats_l1']:.4f}")
check("control (shipped L1 recipe) collapses toward zero", res["l1"]["ratio"] < 0.5,
      f"ratio={res['l1']['ratio']:.3f}")
check("stats-supervised HF head does NOT collapse", res["stats"]["ratio"] > 0.9,
      f"ratio={res['stats']['ratio']:.3f}")
check("CFM mean-flow does NOT collapse", res["cfm"]["ratio"] > 0.9,
      f"ratio={res['cfm']['ratio']:.3f}")

# --------------------------------------------------------------------------- #
# 5. helpers
# --------------------------------------------------------------------------- #
a, b = torch.randn(2, C, 64, 64), torch.randn(2, C, 64, 64)
check("gd_loss finite", bool(torch.isfinite(TT.gd_loss(a, b))))
check("unit gives unit RMS",
      float((TT.unit(a).flatten(1).pow(2).mean(1) - 1).abs().max()) < 1e-5)

print()
if FAIL:
    print(f"[SMOKE] FAILED: {FAIL}")
    sys.exit(1)
print("[SMOKE] all checks passed")
