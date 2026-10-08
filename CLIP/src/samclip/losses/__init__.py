"""Losses: contrastive alignment, distribution matching, geometry, anti-collapse.

The **v4** objective is assembled from four of these pieces in ``samclip.train.Trainer``:

    L = w_mmd * L_mmd   (crossmodal.mmd_crossmodal, EEG cloud vs image cloud)
      + w_fine * ( w_img * L_clip   (contrastive.clip_alignment_loss)
                 + w_xs  * L_xs )   (contrastive.cross_subject_loss, multi-positive)
      + w_spec * L_spec   (regularizers.spectral_concentration)
      + w_reg  * L_cov    (regularizers.vicreg_terms, covariance half only)

with ``w_mmd: 0.9 -> 0.5`` across Stage 1 and ``w_fine = 1 - w_mmd``, reproducing the
reference implementation's coarse-to-fine schedule (`third_party/SAMGA/inter.sh`), then
``w_mmd = 0`` for Stage 2. Four terms were REMOVED and the reasons are measurements, not
preferences -- see ``docs/eeg2image_v4_architecture.md`` §3.7:

  * ``VICReg-var`` -> replaced by ``L_spec``. The per-dimension hinge pushes all 512 axes
    outward while the signal lives in ~16 directions; it cannot be satisfied (0/512 dims
    reached std 1.0) yet held 18.8% of the encoder's gradient.
  * ``mmd_subject`` -> replaced by the cross-MODAL ``L_mmd``. Matching subjects to each
    other is a different quantity from the reference's term, and can be satisfied inside
    a subject-specific subspace the image cloud never occupies.
  * ``HSIC dec`` (0.05 -> 0) -> suppressing subject identity on its own does not improve
    retrieval, and doing it hard removes signal.
  * ``proto`` / ``rkd`` / ``adv`` -> no measured gain, or evidence that does not transfer.

``mmd_subject``, ``hsic_subject``, ``PrototypeEMA``, ``gram_distill_loss`` and
``SubjectAdversary`` all remain importable: they are ablation arms, and keeping them
callable is what makes "removal did not cost anything" a testable claim rather than an
assertion.
"""
from .contrastive import (
    SCALE_MAX,
    SCALE_MIN,
    InfoNCE,
    clip_alignment_loss,
    cross_subject_loss,
    csls_correct,
    recovery_aware_alignment,
    repetition_collapse_loss,
    score_fusion_loss,
    subject_supervised_contrast,
)
from .episode import deploy_stack_episode, whiten_map
from .recovery import orthogonal_procrustes, recovery_episode
from .crossmodal import LADDER, mmd_crossmodal
from .invariance import (
    SubjectAdversary,
    grad_reverse,
    hsic,
    hsic_subject,
    mmd_subject,
    subject_onehot,
)
from .regularizers import (
    PrototypeEMA,
    spectral_concentration,
    spectral_rank_target,
    spectrum_report,
    vicreg_terms,
)
from .relational import gram_distill_loss, gram_matrix
from .soft_plan import sinkhorn_plan, soft_plan_loss
from .metric_distill import metric_self_distill, metric_subject_consistency
from .bias_anchor import (gallery_metric, metric_anchor_loss, metric_anchor_loss_by_subject,
                          subject_consensus_anchor_loss, _whiten_cloud)

__all__ = [
    "SCALE_MIN", "SCALE_MAX",
    "InfoNCE", "clip_alignment_loss", "cross_subject_loss", "subject_supervised_contrast",
    "csls_correct", "recovery_aware_alignment", "score_fusion_loss",
    "repetition_collapse_loss",
    "deploy_stack_episode", "whiten_map",
    "sinkhorn_plan", "soft_plan_loss",
    "SubjectAdversary", "grad_reverse", "hsic", "hsic_subject", "mmd_subject",
    "subject_onehot", "PrototypeEMA", "vicreg_terms",
    "spectral_concentration", "spectral_rank_target", "spectrum_report",
    "mmd_crossmodal", "LADDER",
    "gram_distill_loss", "gram_matrix",
    "metric_self_distill", "metric_subject_consistency",
    "gallery_metric", "metric_anchor_loss", "metric_anchor_loss_by_subject",
    "subject_consensus_anchor_loss", "_whiten_cloud",
]
