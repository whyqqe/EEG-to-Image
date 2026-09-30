#!/usr/bin/env python3
"""Render the UCK-NAT architecture figure (Address -> K hypotheses -> Render-Verify).

Produces docs/figures/uck_nat_architecture.{png,pdf,svg}
Pure matplotlib, no external deps beyond numpy/matplotlib.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- palette
C_NEUTRAL = "#f2f4f7"
E_NEUTRAL = "#5b6472"
C_ADDR = "#e3ecfb"
E_ADDR = "#2f5fa8"
C_RES = "#f0e7fa"
E_RES = "#6b3fa0"
C_BACK = "#e4f4ea"
E_BACK = "#2f7d4f"
C_GEN = "#fdf0dd"
E_GEN = "#b3721a"
C_SEL = "#fbe4e4"
E_SEL = "#a33a3a"
C_SIDE = "#f7f7f4"
E_SIDE = "#8a8a7a"
C_CLAIM = "#fff9d6"
E_CLAIM = "#9a7b0a"

FS = 8.0
FSS = 7.0


def box(ax, x, y, w, h, text, fc, ec, fs=FS, bold=False, lw=1.2, ls="solid", z=3):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.35,rounding_size=1.1",
            linewidth=lw, facecolor=fc, edgecolor=ec,
            linestyle=ls, zorder=z, mutation_aspect=0.28,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=z + 1, linespacing=1.45,
            fontweight="bold" if bold else "normal", color="#12161c")


def container(ax, x, y, w, h, title, ec, z=1, ls=(0, (5, 3))):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.4,rounding_size=1.4",
            linewidth=1.1, facecolor="white", edgecolor=ec,
            linestyle=ls, zorder=z, alpha=0.55,
        )
    )
    ax.text(x + 1.6, y + h - 0.6, title, ha="left", va="top",
            fontsize=FS + 0.4, fontweight="bold", color=ec, zorder=z + 2)


def arrow(ax, p, q, color="#3a4250", lw=1.5, ls="solid", rad=0.0, z=6, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            p, q, arrowstyle=style, mutation_scale=11,
            linewidth=lw, color=color, linestyle=ls, zorder=z,
            connectionstyle=f"arc3,rad={rad}", shrinkA=1.5, shrinkB=1.5,
        )
    )


fig, ax = plt.subplots(figsize=(13.4, 14.2))
ax.set_xlim(0, 132)
ax.set_ylim(-1, 141)
ax.set_aspect("equal")
ax.axis("off")

# ---------------------------------------------------------------- title
ax.text(66, 138.4,
        "UCK-NAT:  Neural Address  →  K-Hypothesis Generation  →  Render-Verify Selection",
        ha="center", va="center", fontsize=12.5, fontweight="bold", color="#12161c")
ax.text(66, 134.6,
        "simultaneous semantic + structural alignment  |  subject-invariant identity code  |  "
        "candidate diversity by construction, not by loss",
        ha="center", va="center", fontsize=8.4, color="#4a5260", style="italic")

# ---------------------------------------------------------------- stage 0
box(ax, 30, 124, 42, 7,
    "EEG trial  ·  one viewed image  ·  17 subjects protocol, leak-free split",
    C_NEUTRAL, E_NEUTRAL, fs=FS + 0.5)
arrow(ax, (51, 124), (51, 118.4))

box(ax, 20, 110, 62, 8.4,
    "STAGE 0  ·  IDENTITY CODE\n"
    r"$z=\mathrm{shared\_r}$   (1024-d, single subject-invariant carrier $\to$ drives BOTH towers)",
    C_NEUTRAL, E_NEUTRAL, fs=FS + 0.5, bold=False)

arrow(ax, (51, 110), (51, 101.4), color=E_ADDR)

# ---------------------------------------------------------------- stage A
container(ax, 3, 73, 94, 28.4, "STAGE A   NEURAL ADDRESSING   (discrete anchors in brain coordinates)", E_ADDR)

box(ax, 6, 84, 30, 11,
    "Concept prototype bank  μ\n"
    "1654 concepts × 1024-d\n"
    "train concepts only (no leak)",
    C_ADDR, E_ADDR, fs=FSS + 0.4)
arrow(ax, (36, 89.5), (42, 89.5), color=E_ADDR, lw=1.3)

box(ax, 42, 86, 18, 7,
    r"$\alpha=\mathrm{softmax}(z\cdot\mu/\tau)$"
    "\nSinkhorn balanced",
    C_ADDR, E_ADDR, fs=FSS + 0.4)
arrow(ax, (60, 89.5), (66, 89.5), color=E_ADDR, lw=1.3)

box(ax, 66, 84, 28, 11,
    "TOP-K ANCHORS\n"
    r"$\mathrm{addr}_1 \dots \mathrm{addr}_K$  (disjoint concept IDs)" "\n"
    "⇒ diversity GUARANTEED,\nno mode collapse possible",
    C_ADDR, E_ADDR, fs=FSS + 0.4, lw=1.6)

box(ax, 42, 76, 52, 6.6,
    "Loss: top-K NCE  —  only the negative concepts ranked above the positive get gradient"
    "\nobjective is  recall@K  (not top-1)   +  Sinkhorn balance,  temperature annealed",
    C_ADDR, E_ADDR, fs=FSS, lw=1.0)

box(ax, 6, 76, 33, 6.6,
    "DIAGNOSTICS\n"
    "recall@K · margin · 55× vs chance (CLIP route ≈ chance)",
    "#fbfcfe", E_ADDR, fs=FSS, lw=1.0, ls="dashed")

# ---------------------------------------------------------------- stage A'
arrow(ax, (51, 73), (51, 70.4), color=E_RES)
container(ax, 3, 54, 94, 16.4, "STAGE A′   RESIDUAL  →  INSTANCE OFFSET HEADS   (the part both prior families discard)",
          E_RES, ls=(0, (5, 3)))

box(ax, 6, 57, 32, 8,
    r"Residual   $r=z-\sum_k \alpha_k \mu_k$" "\n"
    r"projected orthogonal to $\mathrm{span}\{\mu_k\}$",
    C_RES, E_RES, fs=FSS + 0.4)
arrow(ax, (38, 61), (42, 61), color=E_RES, lw=1.3)

box(ax, 42, 57, 20, 8,
    "K residual heads\n"
    r"$\tau_1(r)\dots\tau_K(r)$"
    "\n(small MLPs, only these train)",
    C_RES, E_RES, fs=FSS + 0.4)
arrow(ax, (62, 61), (66, 61), color=E_RES, lw=1.3)

box(ax, 66, 57, 28, 8,
    "MONITOR (kill switch)\n"
    "concept probe on r must be ≈ chance\n"
    "else the anchors leak ⇒ K branches collapse",
    "#fbfcfe", E_RES, fs=FSS, lw=1.0, ls="dashed")

# ---------------------------------------------------------------- stage B
arrow(ax, (51, 54), (51, 51.4), color=E_BACK)
container(ax, 3, 29, 94, 22.4, "STAGE B   DUAL-TOWER CONDITIONING  +  K-HYPOTHESIS ASSEMBLY   (UCK backbone, frozen)",
          E_BACK)

box(ax, 6, 38, 30, 9,
    "SEMANTIC TOWER  (UCK)\n"
    r"$q=\mathrm{MLP}(z)$" "\n"
    r"IP $=\mathrm{mem}(q,G^{\mathrm{img}})$   soft retrieval",
    C_BACK, E_BACK, fs=FSS + 0.4)

box(ax, 6, 30.6, 30, 6.6,
    "STRUCTURE TOWER  (UCK)\n"
    r"$F=\mathrm{Up}(z)$  →  depth map + low-level RGB init",
    C_BACK, E_BACK, fs=FSS + 0.4)

box(ax, 41, 33, 31, 12,
    "HYPOTHESIS ASSEMBLY\n"
    r"$c_k=\mathrm{l2}\!\left(G^{\mathrm{img}}_{\mathrm{addr}_k}+\tau_k(r)\right)$"
    "\n"
    r"$\lambda=0 \Rightarrow K{=}1$ reproduces UCK bit-wise",
    "#eefaf1", E_BACK, fs=FSS + 0.5, lw=1.6)

box(ax, 75, 36, 19, 9,
    "FROZEN  vs  TRAINED\n"
    "frozen: UCK IP, UCK F\n"
    r"trained: $\{\tau_k\}$ + address"
    "\nSOTA cannot regress",
    "#fbfcfe", E_BACK, fs=FSS, lw=1.0, ls="dashed")

# ---------------------------------------------------------------- stage C
arrow(ax, (56, 29), (56, 27.4), color=E_GEN)
arrow(ax, (21, 29), (21, 27.4), color=E_GEN)

box(ax, 3, 20, 94, 7.4,
    "STAGE C  ·  K-CANDIDATE GENERATION\n"
    "SDXL + ControlNet-Img2Img (strength 0.84) + IP-Adapter   —   depth & low-level init SHARED across K\n"
    r"only the IP condition differs   $\Rightarrow$   $x_1 \dots x_K$   (adaptive K from address margin)",
    C_GEN, E_GEN, fs=FSS + 0.4, lw=1.3)

# ---------------------------------------------------------------- stage D
arrow(ax, (51, 20), (51, 18.4), color=E_SEL)

box(ax, 3, 6, 94, 12.4,
    "STAGE D  ·  RENDER-VERIFY SELECTION     (the measurement that is NOT the generation posterior)\n"
    r"re-encode the rendered image:   $\phi(x_k)=\mathrm{CLIP}(x_k)$   $\to$   "
    r"$s_k=\mathrm{sim}\left(\mathrm{cond}(z),\,\phi(x_k)\right)$   $\to$   "
    r"$k^\ast=\arg\max_k s_k$"
    "\n"
    r"verified: 2-way $0.995$  ·  own-generation top-1 $0.735$  ·  "
    r"rank-corr(cond, render) $0.71$   $\Rightarrow$   $\mathrm{top1}(N)\approx\mathrm{recall}@N\cdot p^{\,N-1}$"
    "\n"
    "optional 2nd pass:  $x_{k^\\ast}$ re-conditioned and refined once  (two-stage > one-stage)",
    C_SEL, E_SEL, fs=FSS + 0.4, lw=1.3)

# feedback loop D -> C
arrow(ax, (97.6, 12), (97.6, 23.7), color=E_SEL, lw=1.2, ls=(0, (4, 2)), rad=-0.45)

# ---------------------------------------------------------------- outputs
arrow(ax, (51, 6), (51, 4.6), color=E_NEUTRAL)
box(ax, 3, 0.2, 94, 4.4,
    "REPORTED   ·   identification: top-1 · 2-way  ↑↑      image quality: PixCorr · SSIM · CLIP (measured, not assumed)\n"
    "ablation:  K=1 (bit-wise UCK) · random-K (noise arm) · oracle-anchor upper bound · recall@K reported separately",
    C_NEUTRAL, E_NEUTRAL, fs=FSS + 0.2)

# ---------------------------------------------------------------- right column
box(ax, 100, 74, 30, 27,
    "BORROWED  —  NOT CLAIMED\n"
    "\n"
    "CORTIVA: score-level fusion\n"
    "(not embedding-level), +10.3 top-1\n"
    "\n"
    "SATTC: label-free whitening\n"
    "+ CSLS, fixes hubness, shortlists\n"
    "\n"
    "SCORE: cross-subject coordinate\n"
    "recovery without target labels\n"
    "\n"
    "NVOL / BReAD / SeeEEG:\n"
    "two-stage retrieval→diffusion",
    C_SIDE, E_SIDE, fs=FSS, lw=1.0)

box(ax, 100, 44, 30, 27,
    "SCORE-LEVEL FUSION (4 routes)\n"
    "temperature-scaled, then fused\n"
    "BEFORE ranking:\n"
    "\n"
    r"1. neural address   $z\cdot\mu$   $\leftarrow$ non-CLIP route" "\n"
    r"2. semantic         $q\cdot G^{\mathrm{img}}$" "\n"
    r"3. text             $G^{\mathrm{txt}}$" "\n"
    r"4. structure        depth / low-level" "\n"
    "\n"
    "whitening + CSLS  →  shortlist top-N\n"
    "feeds Stage A anchors & Stage D",
    "#eefaf1", E_BACK, fs=FSS, lw=1.0)

box(ax, 100, 22, 30, 19,
    "GUARDS  (why SOTA is safe)\n"
    "\n"
    "·  K=1 ⇒ bit-wise UCK row\n"
    "·  verifier never trains\n"
    "   the conditioning path\n"
    "·  innovation is additive only\n"
    "·  adaptive K saves compute",
    "#fbfcfe", E_SIDE, fs=FSS, lw=1.0, ls="dashed")

box(ax, 100, 0.2, 30, 18.8,
    "CLAIMED NOVELTY  (3)\n"
    "\n"
    "1. neural address as an\n"
    "   independent non-CLIP\n"
    "   scoring route\n"
    "\n"
    "2. best-of-N verifier on\n"
    "   RENDERED images for\n"
    "   reconstruction\n"
    "\n"
    "3. structure tower inside\n"
    "   score fusion",
    C_CLAIM, E_CLAIM, fs=FSS, lw=1.3)

# connectors to the right column
arrow(ax, (94, 89.5), (100, 69), color=E_ADDR, lw=1.1, rad=-0.25)
arrow(ax, (100, 50), (97.6, 24.5), color=E_BACK, lw=1.1, rad=0.25)
arrow(ax, (94, 62), (100, 57), color=E_RES, lw=1.1, rad=-0.2)

fig.tight_layout(pad=0.4)
for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"uck_nat_architecture.{ext}", dpi=210, bbox_inches="tight",
                facecolor="white")
print("wrote", OUT / "uck_nat_architecture.png")
