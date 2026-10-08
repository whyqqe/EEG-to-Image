import time, torch
torch.manual_seed(0)
B, C, T = 128, 63, 250
x = torch.randn(B, C, T, requires_grad=True)


def _reg(x, eps=1e-4):
    c = x @ x.transpose(-1, -2)
    scale = c.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True)
    return c + eps * scale.unsqueeze(-1) * torch.eye(C), scale


def eigh_w(x, eps=1e-4):
    c, scale = _reg(x, eps)
    w, q = torch.linalg.eigh(c)
    w = w.clamp_min(eps * scale.abs().clamp_min(1e-12))
    return ((q * w.rsqrt().unsqueeze(-2)) @ q.transpose(-1, -2)) @ x


def chol_w(x, eps=1e-4):
    c, _ = _reg(x, eps)
    L = torch.linalg.cholesky(c)
    return torch.linalg.solve_triangular(L, x, upper=False)


def ns_w(x, eps=1e-4, iters=6):
    c, _ = _reg(x, eps)
    nrm = c.flatten(1).norm(dim=1).view(-1, 1, 1) + 1e-12
    Y = c / nrm
    Z = torch.eye(C).expand_as(c).clone()
    for _ in range(iters):
        T_ = 3 * torch.eye(C) - Z @ Y
        Y = 0.5 * (Y @ T_)
        Z = 0.5 * (T_ @ Z)
    return (Z * nrm.rsqrt().unsqueeze(-1)) @ x


def bench(fn, name, n=15):
    for _ in range(2):
        y = fn(x); y.sum().backward(); x.grad = None
    t = time.time()
    for _ in range(n):
        y = fn(x); y.sum().backward(); x.grad = None
    return f"  {name:12s} {(time.time()-t)/n*1000:7.1f} ms/step (fwd+bwd)"


out = ["== fwd+bwd per training step, B=128 =="]
out.append(bench(lambda z: z, "identity"))
out.append(bench(chol_w, "chol"))
out.append(bench(ns_w, "newton-schulz"))
out.append(bench(eigh_w, "eigh"))
out.append("")
out.append("== whiteness + O-invariance ==")
for nm, fn in (("chol", chol_w), ("ns", ns_w), ("eigh", eigh_w)):
    with torch.no_grad():
        W = fn(x.detach())
        err = (W @ W.transpose(-1, -2) - torch.eye(C)).abs().max().item()
        g = torch.Generator().manual_seed(1)
        Gm = torch.eye(C) + 3.15 * torch.randn(C, C, generator=g) / torch.randn(C, C, generator=g).norm()
        W2 = fn(Gm @ x.detach())
        ge = (W.transpose(-1, -2) @ W - W2.transpose(-1, -2) @ W2).abs().max().item()
    out.append(f"  {nm:5s} max|WW^T-I|={err:.2e}   Gram-invariance max|d|={ge:.2e}")

print("\n".join(out))
