"""Fast post-fix verification. Seconds, not minutes. No writes outside CLIP/."""
import sys
import numpy as np
import torch
sys.path.insert(0, 'src')
import torch.nn.functional as F
from samclip import calibration as C
from samclip.losses.soft_plan import sinkhorn_plan, fgw_plan, soft_plan_loss
from samclip.losses.contrastive import csls_correct

torch.manual_seed(0)
n_sub, per = 9, 22
a = F.normalize(torch.randn(n_sub * per, 64), dim=-1)
b = F.normalize(torch.randn(n_sub * per, 64), dim=-1)
subj = torch.repeat_interleave(torch.arange(n_sub), per)
sim = csls_correct(a @ b.t(), k=20)
same = (subj[:, None] == subj[None, :]).float()

print('[1] cross-subject plan mass must be ~0 for ALL alpha (the >0.5 sign-flip trap)')
ok = True
for al in (0.0, 0.1, 0.25, 0.5, 0.6, 0.75, 0.9, 1.0):
    if al == 0.0:
        P = sinkhorn_plan(sim.masked_fill(subj[:, None] != subj[None, :], -1e4), tau=0.1, iters=20)
    else:
        P = fgw_plan(sim, a, b, al, tau=0.1, iters=20, outer=10, block=subj)
    off = ((P * (1 - same)).sum() / P.sum()).item()
    ok &= off < 1e-2
    print('    alpha=%.2f  cross=%.3e  diag=%.4f' % (al, off, (P * same).sum().item() / per))
print('    PASS' if ok else '    *** FAIL ***')

print('[2] alpha=0 bit-identical to shipped S3R; alpha>0 is not a no-op')
rng = np.random.default_rng(0)
q = rng.normal(size=(200, 64)); g = rng.normal(size=(200, 64))
o0, _ = C.subspace_soft_recovery(q, g, k=10, rho=0.1, rank=None, tau=0.05, hard_landmarks=False, alpha=0.0)
o0b, _ = C.subspace_soft_recovery(q, g, k=10, rho=0.1, rank=None, tau=0.05, hard_landmarks=False)
oa, da = C.subspace_soft_recovery(q, g, k=10, rho=0.1, rank=None, tau=0.05, hard_landmarks=False, alpha=0.25)
print('    bit-identical=%s  alpha0.25 differs=%s  plan_acc=%.4f'
      % (np.array_equal(o0, o0b), not np.allclose(o0, oa), da['plan_acc']))

print('[3] deployed GW gradient vs finite difference (small N, cheap)')
M = 8
C1 = C._sq_cos_dist(rng.normal(size=(M, 5)))
C2 = C._sq_cos_dist(rng.normal(size=(M, 5)))
pi = rng.random((M, M)); pi /= pi.sum()
L = (C1[:, None, :, None] - C2[None, :, None, :]) ** 2
gwf = lambda p: float((L * p[None, None, :, :] * p[:, :, None, None]).sum())
r = pi.sum(1, keepdims=True); c = pi.sum(0, keepdims=True)
new = (C1 * C1) @ r + ((C2 * C2) @ c.T).T - 2.0 * (C1 @ pi @ C2.T)
old = C1 @ r + (C2 @ c.T) - 2.0 * (C1 @ pi @ C2.T)
h = 1e-6
G = np.zeros((M, M))
for i in range(M):
    for j in range(M):
        Pp = pi.copy(); Pp[i, j] += h
        Pm = pi.copy(); Pm[i, j] -= h
        G[i, j] = (gwf(Pp) - gwf(Pm)) / (2 * h)
print('    corr(fixed, true) = %+.6f   corr(old, true) = %+.6f'
      % (np.corrcoef(new.ravel(), G.ravel())[0, 1], np.corrcoef(old.ravel(), G.ravel())[0, 1]))

print('[4] training loss finite/differentiable, per-alpha')
for al in (0.1, 0.25, 0.5, 0.75, 1.0):
    aa = a.clone().requires_grad_(True)
    l, d = soft_plan_loss(aa, b, block=subj, tau=0.1, csls_k=20, alpha=al, return_diag=True)
    l.backward()
    print('    alpha=%.2f loss=%7.4f diag_mass=%.4f |grad|=%.2e finite=%s'
          % (al, l.item(), d['soft_plan_diag_mass'], aa.grad.abs().sum(), bool(torch.isfinite(l))))
