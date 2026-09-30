"""Label-free cross-subject test-time calibration for EEG→image retrieval.

Implements a SATTC-style stack (standardized cosine + candidate whitening +
subject-adaptive whitening + CSLS/Ada-CSLS + mutual-NN structural expert + PoE),
plus a physiology-aware query prior from channel×band energies (our differentiator).

References:
  - Lample et al., CSLS (EMNLP/ACL bilingual lexicon)
  - SATTC (CVPR 2026): SAW + Ada-CSLS + structural PoE for THINGS-EEG LOSO
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True).clip(min=eps)
    return x / n


def zca_whiten(x: np.ndarray, eps: float = 1e-5, ref: np.ndarray | None = None) -> np.ndarray:
    """Whiten x using covariance of ref (or x itself)."""
    ref = x if ref is None else ref
    mu = ref.mean(axis=0, keepdims=True)
    xc = ref - mu
    cov = (xc.T @ xc) / max(len(ref) - 1, 1)
    # eigendecomposition for ZCA
    w, v = np.linalg.eigh(cov + eps * np.eye(cov.shape[0], dtype=np.float64))
    w = np.maximum(w, eps)
    w_inv_sqrt = 1.0 / np.sqrt(w)
    p = (v * w_inv_sqrt) @ v.T
    return ((x - mu) @ p.T).astype(np.float32)


def candidate_whiten(gallery: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    return l2_normalize(zca_whiten(gallery, eps=eps))


def subject_adaptive_whiten(queries: np.ndarray, calib: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """SAW: whiten queries with unlabeled calibration stats from same subject."""
    return l2_normalize(zca_whiten(queries, eps=eps, ref=calib))


def cosine_sim(q: np.ndarray, g: np.ndarray) -> np.ndarray:
    return l2_normalize(q) @ l2_normalize(g).T


def csls(sim: np.ndarray, k: int = 10) -> np.ndarray:
    """Classic CSLS: S_ij - 0.5 r_i - 0.5 c_j."""
    k = max(1, min(k, sim.shape[1] - 1, sim.shape[0] - 1))
    # row mean of top-k
    part = np.partition(sim, -k, axis=1)[:, -k:]
    r = part.mean(axis=1, keepdims=True)
    partc = np.partition(sim, -k, axis=0)[-k:, :]
    c = partc.mean(axis=0, keepdims=True)
    return sim - 0.5 * r - 0.5 * c


def adaptive_csls(sim: np.ndarray, k_min: int = 5, k_max: int = 30) -> np.ndarray:
    """Density-aware CSLS with per-row / per-col neighborhood sizes."""
    n_q, n_g = sim.shape
    # local density ~ gap between top1 and mean of top-k_max
    k_ref = min(k_max, n_g - 1, n_q - 1)
    top = np.partition(sim, -k_ref, axis=1)[:, -k_ref:]
    dens_q = (top.max(axis=1) - top.mean(axis=1))  # sharper → smaller neighborhood
    dens_q = (dens_q - dens_q.min()) / (dens_q.ptp() + 1e-8)
    k_row = (k_min + (1.0 - dens_q) * (k_max - k_min)).astype(int).clip(k_min, k_ref)

    topc = np.partition(sim, -k_ref, axis=0)[-k_ref:, :]
    dens_c = topc.max(axis=0) - topc.mean(axis=0)
    dens_c = (dens_c - dens_c.min()) / (dens_c.ptp() + 1e-8)
    k_col = (k_min + (1.0 - dens_c) * (k_max - k_min)).astype(int).clip(k_min, k_ref)

    r = np.zeros((n_q, 1), dtype=np.float32)
    for i in range(n_q):
        ki = int(k_row[i])
        r[i, 0] = np.partition(sim[i], -ki)[-ki:].mean()
    c = np.zeros((1, n_g), dtype=np.float32)
    for j in range(n_g):
        kj = int(k_col[j])
        c[0, j] = np.partition(sim[:, j], -kj)[-kj:].mean()
    return sim - 0.5 * r - 0.5 * c


def structural_expert(sim: np.ndarray, top_l: int = 5) -> np.ndarray:
    """Mutual NN / bidirectional rank / popularity prior on pre-CSLS similarities."""
    n_q, n_g = sim.shape
    # query→gallery ranks
    order_qg = np.argsort(-sim, axis=1)
    rank_qg = np.empty_like(order_qg)
    rows = np.arange(n_q)[:, None]
    rank_qg[rows, order_qg] = np.arange(n_g)[None, :]

    order_gq = np.argsort(-sim, axis=0)
    rank_gq = np.empty_like(order_gq)
    cols = np.arange(n_g)[None, :]
    rank_gq[order_gq, cols] = np.arange(n_q)[:, None]

    mutual_top1 = ((rank_qg == 0) & (rank_gq == 0)).astype(np.float32)
    bidir = ((rank_qg < top_l) & (rank_gq < top_l)).astype(np.float32)
    # popularity: how often a gallery item is in query top-1
    pop = (rank_qg == 0).sum(axis=0).astype(np.float32)
    pop = pop / (pop.max() + 1e-8)
    hub_pen = pop[None, :]

    # positive bias for mutual/bidirectional, negative for hubs
    s = 1.0 * mutual_top1 + 0.5 * bidir - 0.35 * hub_pen
    return s


def poe_fuse(s_geom: np.ndarray, s_struct: np.ndarray, beta: float = 0.5) -> np.ndarray:
    """Product-of-experts in log space: geom + beta * struct."""
    # standardize each expert
    def _std(x):
        return (x - x.mean()) / (x.std() + 1e-8)

    return _std(s_geom) + float(beta) * _std(s_struct)


def physiology_query_prior(subspace_energy: np.ndarray, sim: np.ndarray, strength: float = 0.25) -> np.ndarray:
    """Modulate rows by occipital/visual-band energy (label-free physiology prior).

    subspace_energy: (N_q, M) non-negative energies from channel×band banks.
    Prefer queries with higher early-visual energy to keep sharper CSLS rows;
    low-energy (noisy) queries get stronger hubness penalty via row shrinkage.
    """
    # visual-ish columns: occipital_* and parietal_* alphas/betas if present order unknown —
    # use top-energy concentration as proxy for "structured" neural response
    e = subspace_energy.astype(np.float32)
    e = e / (e.sum(axis=1, keepdims=True) + 1e-8)
    # entropy: lower → more focused subspace → trust more
    ent = -(e * np.log(e + 1e-8)).sum(axis=1)
    ent = (ent - ent.min()) / (ent.ptp() + 1e-8)
    trust = 1.0 - ent  # high trust for focused patterns
    trust = trust[:, None]
    # shrink ambiguous rows toward column mean (reduces false hubs for noisy queries)
    col_mean = sim.mean(axis=0, keepdims=True)
    return (1.0 - strength * (1.0 - trust)) * sim + strength * (1.0 - trust) * col_mean


def retrieval_from_sim(sim: np.ndarray) -> dict[str, float]:
    n = sim.shape[0]
    pred = sim.argmax(axis=1)
    gt = np.arange(n)
    top1 = float((pred == gt).mean())
    top5 = float((np.argsort(-sim, axis=1)[:, :5] == gt[:, None]).any(axis=1).mean())
    # hubness: skewness of N_k occurrence (k=10)
    k = min(10, sim.shape[1])
    topk = np.argsort(-sim, axis=1)[:, :k]
    counts = np.bincount(topk.ravel(), minlength=sim.shape[1]).astype(np.float64)
    # standardized skewness
    mu, sd = counts.mean(), counts.std() + 1e-8
    hub = float((((counts - mu) / sd) ** 3).mean())
    return {"top1": top1, "top5": top5, "hubness_skew": hub, "chance_top1": 1.0 / max(n, 1)}


def run_calibration_suite(
    queries: np.ndarray,
    gallery: np.ndarray,
    calib_queries: np.ndarray | None = None,
    subspace_energy: np.ndarray | None = None,
    csls_k: int = 10,
    beta: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Evaluate a ladder of label-free calibrations."""
    q0 = l2_normalize(queries.astype(np.float32))
    g0 = l2_normalize(gallery.astype(np.float32))
    calib = l2_normalize(calib_queries.astype(np.float32)) if calib_queries is not None else q0

    out: dict[str, dict[str, float]] = {}

    # 1) raw cosine
    s = cosine_sim(q0, g0)
    out["cosine"] = retrieval_from_sim(s)

    # 2) candidate whitening
    gw = candidate_whiten(g0)
    s = cosine_sim(q0, gw)
    out["cosine_cw"] = retrieval_from_sim(s)

    # 3) SAW + CW
    qw = subject_adaptive_whiten(q0, calib)
    s_base = cosine_sim(qw, gw)
    out["saw_cw"] = retrieval_from_sim(s_base)

    # 4) + CSLS
    s_csls = csls(s_base, k=csls_k)
    out["saw_cw_csls"] = retrieval_from_sim(s_csls)

    # 5) + Ada-CSLS
    s_ada = adaptive_csls(s_base)
    out["saw_cw_adacsls"] = retrieval_from_sim(s_ada)

    # 6) structural + PoE on Ada-CSLS
    s_struct = structural_expert(s_base)
    s_poe = poe_fuse(s_ada, s_struct, beta=beta)
    out["sattc_like_poe"] = retrieval_from_sim(s_poe)

    # 7) physiology-aware row prior then PoE (our addition)
    if subspace_energy is not None:
        s_phys = physiology_query_prior(subspace_energy, s_base, strength=0.25)
        s_phys_ada = adaptive_csls(s_phys)
        s_phys_poe = poe_fuse(s_phys_ada, structural_expert(s_phys), beta=beta)
        out["phys_poe"] = retrieval_from_sim(s_phys_poe)

    return out
