"""SAM-CLIP v2: a shared cross-subject EEG encoder aligned to a structured image target.

See `docs/eeg2image_v2_plan.md` for the full design.

One training stage, no subject conditioning, and no mapping network. Subject
adaptation lives in the objective (cross-subject same-stimulus consistency) and in
label-free test-time geometry -- not in per-subject parameters. The evidence for that
choice, including the measurements that killed the conditioning paradigm, is in
`models/subject_conditioning.py` and the plan doc §2.

Module map
----------
    config.py                  paths, geometry, cache redirection
    utils.py                   seeding, logging, config IO, checkpointing
    data/                      THINGS-EEG2 loading, MVNN, targets, batch sampler
    models/                    shared EEG trunk + SAMCLIP assembly + target router
    losses/                    contrastive, invariance, relational, regularizers
    evaluate.py                Top-k / mean-rank retrieval metrics (raw cosine)
    calibration.py             deployment-time geometry (SAW / CSLS / recovery) -- Stage C
    train.py                   the single training stage (Stage A)

The measurement ladder, so a number can be attributed to the right layer:
raw cosine (encoder only) -> SAW whitening -> CSLS -> coordinate recovery. The last
group is label-free CPU post-processing on frozen features and is worth more than every
training change combined (26.22 -> 53.23 Top-1, PROTOCOL_INTER.md §7).
"""

__version__ = "0.2.0"
