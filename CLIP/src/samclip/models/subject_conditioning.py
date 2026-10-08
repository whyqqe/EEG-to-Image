"""REMOVED in v2 -- subject-as-modality conditioning (``z_s`` + FiLM + LoRA).

This module used to hold the project's central idea:

    ``SupportSetEncoder``   -- K unlabelled support trials -> ``z_s``
    ``SubjectConditioner``  -- ``z_s`` -> FiLM affines + low-rank (LoRA) corrections
    ``CondParams``          -- the per-batch conditioning object threaded into the trunk

It is deliberately NOT deleted outright, so that the tree records *why* it is gone
rather than leaving a silent hole where 560 lines of the project's main hypothesis
used to be. Nothing imports this module any more; it is documentation, not code.

--------------------------------------------------------------------------------
WHY IT WAS REMOVED
--------------------------------------------------------------------------------
The paradigm was "a subject is an observation modality: a different ``z_s`` defines a
different observation operator on a shared trunk". It failed for a specific,
measurable reason, and the reason is not "we tuned it badly".

1. THE CONDITIONING COLLAPSED TO A CONSTANT.
   ``scripts/diag_zcollapse.py`` on a trained Stage-1 checkpoint measured pairwise
   cosine **0.9999** between the ``z_s`` of DIFFERENT subjects and a subject
   separation of **-0.00000**.

   This is structural, not an accident: the objective rewarded subject invariance
   (HSIC ``dec``, MMD) and NOTHING rewarded ``z_s`` carrying subject information. A
   constant ``z_s`` is therefore the *optimum of the sub-problem as written*. There is
   no bug to fix -- the objective has to change, or the mechanism has to go.

2. THE OBVIOUS CARRIER FOR ``z_s`` HAD ALREADY BEEN REMOVED BY PREPROCESSING.
   ``scripts/diag_subject_statistic.py`` measured per-channel ``[mean_c, std_c]`` -- 
   which is what the moments anchor was built from -- at separation **-0.009**, i.e.
   subject-agnostic. The cause is that every subject is z-scored (and optionally
   MVNN-whitened) with its own statistics, so those two numbers are gone before the
   encoder sees them.

   The one descriptor that DID survive was the log eigenvalue spectrum of the spatial
   covariance (**+0.60** separation), which is why the ``geom`` anchor existed. But
   even a working descriptor inside a collapsing objective does not make the
   conditioning load-bearing.

3. THE MEASURED PAYOFF WAS ZERO, AND THE LITERATURE AGREES.
   * On the real sub-08 fold, 1 epoch: support-set conditioning vs the id-table
     control showed no measurable difference (job 637403).
   * An independent LOSO study on THINGS-EEG2 finds that *suppressing* subject
     identity does not improve retrieval -- and that adversarial suppression actively
     HURTS -- while pulling same-stimulus responses from different subjects together
     helps (+2.23pp Top-1, 95% CI [+1.28, +3.19]).
   * The 2026 leaders in this task (SAMGA, SVTL, SCORE) get their gains from the
     target/supervision side, the alignment structure and test-time geometry -- not
     from conditioning the EEG encoder on a subject vector.

--------------------------------------------------------------------------------
WHAT REPLACED IT
--------------------------------------------------------------------------------
  * ``models/backbone.py``     -- a plain shared trunk, identical for every subject.
  * ``losses.contrastive``     -- cross-subject same-stimulus consistency (multi-
                                  positive InfoNCE): the "pull together" term that
                                  the evidence says is the productive one.
  * ``losses.regularizers``    -- VICReg anti-collapse + an asymmetric EEG-prototype
                                  anchor (image embeddings pulled to neural anchors).
  * ``calibration``            -- label-free test-time geometry (SAW + adaptive CSLS
                                  + SCORE coordinate recovery), which is where the
                                  largest single lever actually lives.

Full reasoning and the supporting measurements: ``docs/eeg2image_v2_plan.md`` §2.

--------------------------------------------------------------------------------
FOR PROVENANCE
--------------------------------------------------------------------------------
The removed implementation, and the diagnostic scripts that measure its failure
(``diag_zcollapse``, ``diag_subject_statistic``, ``diag_anchor_swamp``, ``diag_zscale``,
``probe_subject_z``), are kept under ``scripts/legacy/v1_subject_conditioning/``.
They no longer run against the v2 model but they are the evidence this decision rests
on, so they are archived rather than discarded.

Two engineering lessons from that code are worth carrying forward even though the code
is gone:

  * MAGNITUDE PROPORTIONALITY. The FiLM/LoRA generators were zero-initialised on their
    last layer, so their effective modulation rate scaled with ``|z|``: a step of ~lr
    in the final Linear moved ``dgamma`` by ~lr*|z|. At ``|z| ~ 8-12`` the modulation
    ran away and the image contrast froze at ``ln(72) = 4.2767`` (the no-information
    value) for 12k steps. Any hypernetwork that multiplies a generator by an input
    vector inherits this: keep the input's RMS at the generator's own init scale.
  * A FROZEN BATCHNORM IS NOT FROZEN by ``requires_grad=False`` -- with ``momentum > 0``
    it keeps writing running statistics under ``no_grad``. See the note at the top of
    ``models/backbone.py``.
"""
from __future__ import annotations

__all__: list[str] = []
