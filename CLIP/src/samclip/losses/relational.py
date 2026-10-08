"""Relational knowledge distillation (RKD): match the *geometry*, not the vectors.

Every other term in the objective is pointwise. `clip_alignment_loss` asks "is this
row's EEG embedding near its own image target?"; `cross_subject_loss` asks "are the
rows that share a stimulus near each other?". Both can, in principle, be satisfied by
a mapping that preserves the right *pairs* while distorting the global shape of the
representation -- the EEG side can be a systematically squeezed or sheared version of
the image space and every pairwise ranking that matters still comes out right, until
it does not.

RKD closes that gap by comparing second-order structure: the matrix of pairwise
similarities among the batch's rows. Its special value for this project is that it is
**subject-agnostic by construction**. A Gram matrix does not know which row belongs to
which subject, so a batch whose rows are the same stimuli seen by different subjects
has an image-side Gram matrix with visible block structure, and distilling it forces the
EEG side to reproduce that structure -- which is exactly "different subjects, same
stimulus, land together" expressed geometrically rather than as a positive pair. It is
therefore the natural *complement* to ``cross_subject_loss`` rather than a duplicate of
it: the contrastive term supplies the hard positives, this one supplies the shape.

Cost: `O(B^2 D)` to form each Gram matrix and `O(B^2)` to compare, with no parameters.
On this pipeline's batch size (72 rows) it is negligible next to the encoder pass.

Off by default (`weights.rkd = 0.0` in the config). It is registered as an ablation arm
because the evidence for it is indirect: the published comparisons bundle relational
terms with visual-target changes, so it cannot be credited on its own from the
literature, and this project's own audit found no arm where it was load-bearing.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def gram_matrix(z: torch.Tensor, normalize_rows: bool = True) -> torch.Tensor:
    """``(B, D)`` -> ``(B, B)`` Gram matrix of L2-normalised rows.

    Normalising the rows is what makes the comparison meaningful across the two arms:
    the EEG side has no reason to reach the image side's vector norm, and without
    normalisation the loss would be dominated by whichever side has the larger scale,
    i.e. it would be optimised by shrinking the EEG embeddings -- a collapse dressed up
    as geometry matching.
    """
    if z.dim() != 2:
        raise ValueError(f"expected (B, D), got {tuple(z.shape)}")
    if normalize_rows:
        z = F.normalize(z, dim=-1)
    return z @ z.t()


def gram_distill_loss(z_eeg: torch.Tensor, z_target: torch.Tensor,
                      normalize_gram: bool = True, eps: float = 1e-8) -> torch.Tensor:
    """Squared Frobenius distance between the two Gram matrices.

    `normalize_gram` divides each Gram matrix by its Frobenius norm before subtracting.
    It matters more than it looks: the image-side Gram matrix is *frozen* (the targets
    are precomputed, not learned), so an unnormalised distance would be a fixed-scale
    regression target and the EEG side would be pushed toward the image embeddings'
    absolute scale rather than their structure. The normalisation leaves the loss
    scale-invariant in both arms, so its weight does not silently interact with the
    encoder's output scale (which the VICReg term is separately shaping).
    """
    g_e = gram_matrix(z_eeg)
    g_t = gram_matrix(z_target)
    if normalize_gram:
        g_e = g_e / g_e.norm().clamp_min(eps)
        g_t = g_t / g_t.norm().clamp_min(eps)
    return (g_e - g_t).pow(2).mean()
