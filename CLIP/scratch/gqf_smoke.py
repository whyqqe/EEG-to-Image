import time
import numpy as np
import torch
from samclip.models.samclip import group_quotient

torch.manual_seed(0)
B, C, T = 128, 63, 250


def draw_M(c, mag=3.15, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(c, c))
    return torch.tensor(np.eye(c, dtype=np.float32)
                        + (mag * g / (np.linalg.norm(g) + 1e-12)).astype(np.float32))


outs = []
x = torch.randn(B, C, T)
M = draw_M(C)

with torch.no_grad():
    W = group_quotient(x, "whiten")
    W2 = group_quotient(M @ x, "whiten")
    whiteness = (W @ W.transpose(-1, -2) - torch.eye(C)).abs().max().item()
    gram_inv = (W.transpose(-1, -2) @ W - W2.transpose(-1, -2) @ W2).abs().max().item()
outs.append(f"whiteness max|WW^T-I|      = {whiteness:.2e}  (want ~0)")
outs.append(f"Gram invariance max|d|     = {gram_inv:.2e}  (want ~0)")

# timing fwd+bwd of the full forward the trainer runs (both solvers)
xg = torch.randn(B, C, T, requires_grad=True)


def bench(fn, name, n=15):
    for _ in range(2):
        y = fn(xg); y.sum().backward(); xg.grad = None
    t = time.time()
    for _ in range(n):
        y = fn(xg); y.sum().backward(); xg.grad = None
    return f"  {name:22s} {(time.time()-t)/n*1000:7.1f} ms/step (fwd+bwd, B=128, CPU)"


outs.append("")
outs.append("== per-training-step cost ==")
outs.append(bench(lambda z: z, "identity (baseline)"))
outs.append(bench(lambda z: group_quotient(z, "whiten"), "gqf whiten (cholesky)"))


def _eigh(z):
    c = z @ z.transpose(-1, -2)
    sc = c.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True)
    c = c + 1e-4 * sc.unsqueeze(-1) * torch.eye(C)
    w, q = torch.linalg.eigh(c)
    w = w.clamp_min(1e-4 * sc.abs().clamp_min(1e-12))
    return ((q * w.rsqrt().unsqueeze(-2)) @ q.transpose(-1, -2)) @ z


outs.append(bench(_eigh, "old eigh (reference)"))

with open("scratch/gqf_smoke.txt", "w") as f:
    f.write("\n".join(outs) + "\n")
print("\n".join(outs))
