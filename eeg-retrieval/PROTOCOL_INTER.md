# Inter-subject (LOSO) protocol spec

Every number below is transcribed from the source paper, not inferred. The point of
this file is that our inter-subject runs must be **directly subtractable** from the
published tables, and today they are not: our pipeline trains on one subject, holds
out 150 of the 1654 training concepts for validation, and selects a checkpoint on
that holdout. None of the three matches any SOTA paper.

## 1. The task

Two regimes appear in every paper, and they differ in more than the data split:

| | intra-subject | **inter-subject (ours)** |
|---|---|---|
| train | one subject | **9 subjects** |
| test | same subject | **1 held-out subject** |
| channels | 17 occipito-parietal | **all 63** |
| reported | per subject + average | per subject + average |

The channel asymmetry is not a detail -- it is the single most reproducible finding
in the inter-subject literature, and two independent papers quantify it:

| channels | UBP | HyFI | NeuroBridge | SIMON |
|---|---:|---:|---:|---:|
| OP (17) | 12.9 | 14.6 | 14.9 | 17.2 |
| ALL (63) | **13.7** | **16.4** | **19.0** | **19.6** |

SIMON's explanation: anterior channels carry "anchoring" features that are too weak
to force a Top-1 hit but sufficient to pull the correct target into the Top-5 -- the
Top-5 gain (44.1 -> 49.9) is larger than the Top-1 gain (17.2 -> 19.6).

Note the opposite direction for intra-subject, where adding anterior channels *hurts*:
Shallow Alignment measures a 12-18 point drop (InternViT 82.6 -> 70.9). So the
channel count must not be carried over from our intra-subject runs.

## 2. Data split

- train: **all 1654 concepts**, each 10 images, each 4 repetitions (`--val-concepts 0`)
- test: the 200 held-out concepts, 1 image each, 80 repetitions, **averaged to one
  trial per concept**
- repetitions are averaged in **both** splits
- retrieval is **200-way**, correct pairing on the diagonal

The 1654/200 concept split is disjoint by construction (`C_tr ∩ C_te = ∅`), which is
what makes the task zero-shot.

Shallow Alignment states the no-validation rule explicitly: *"For fair comparison
with prior baselines, the main experiments follow the standard protocol and train on
the full training set without a validation split."* They also check the alternative
(74 held-out concepts, ~5% of training, selected by lowest validation loss) and find
only minor fluctuations that preserve every conclusion -- so our 150-concept holdout
is not *wrong*, it is just not comparable, and it costs us 150 of 1504 concepts.

## 3. Checkpoint selection

This is where the field is deliberately careful, because the obvious choice is a leak.

| policy | who | quote |
|---|---|---|
| **final epoch** | SCORE | *"train each model for 50 epochs, and report the final epoch"* |
| **final epoch** | Shallow Alignment (primary) | *"We adopt the final-epoch checkpoint as our primary submitted result. Best-epoch accuracy is listed for reference only: selecting the checkpoint by test accuracy would constitute test-set over-selection."* |
| test Top-1 | SAMGA | selects on **test** -- a leak; our pipeline correctly refuses to copy it |
| val loss | Shallow Alignment (appendix only) | sensitivity check, not the reported number |

**Decision: select the last epoch.** It is the only policy that is simultaneously
leak-free, comparable, and does not consume 150 training concepts. It also removes
the entire reason `--val-concepts` exists, which is what makes "no validation split"
and "final epoch" the *same* decision rather than two.

## 4. Preprocessing

Transcribed identically by NICE, ATM, UBP, NeuroBridge, SAMGA, SCORE and Shallow
Alignment, so there is no ambiguity to resolve:

- band-pass 0.1-100 Hz
- epoch 0-1000 ms post-stimulus
- baseline correction from the 200 ms pre-stimulus mean
- downsample to 250 Hz
- **MVNN** (multivariate noise normalization) on the training data
- average repetitions

Our cached arrays already implement 0-1000 ms / 200 ms baseline / 250 Hz /
repetition-averaging, and the shapes confirm it: `(1654, 10, 4, 63, 250)` averages to
`(1654, 10, 63, 250)`, and `(200, 1, 80, 63, 250)` to `(200, 1, 63, 250)`.

**MVNN: implemented, `--mvnn`.** See `epd/mvnn.py` for the method and section 6 for
which split each role fits from. The field's phrasing is "MVNN is applied to the
training data", and IDES is explicit that this means *only* the training data. Two
things about it that are easy to get wrong and are therefore worth stating:

  * It must be fitted on **un-averaged** trials. The residual about each condition's
    mean is the only thing that estimates noise, and averaging repetitions is
    precisely the operation that destroys it. The averaged caches cannot feed a
    whitener, so `_raw_blocks` reads the raw file and the fitted `W` is applied to
    the averaged one. That order is legal because `W` is linear and therefore
    commutes with averaging -- asserted in `test_epd_mvnn.py`.
  * Shrinkage must be applied in **correlation** space, not covariance space. EEG
    channel variances span two orders of magnitude on this montage, so the obvious
    "shrink the covariance towards `tr(S)/p · I`" target is set by the loud
    electrodes and injects variance *into* the quiet ones -- reproducing, at small
    `lam`, exactly the imbalance MVNN exists to remove. Measured on a fixture with a
    64x spread at `lam = 0.008`, the quietest channel's variance was inflated by 36%
    and whitened only 5x less than it should have been. The fix is to normalise each
    channel to unit variance first (which is UNN), shrink the correlation matrix
    towards `I`, and unscale. `diag(sigma)` is then the sample variance exactly, and
    a test asserts it.

## 5. Normalization across subjects

SAMGA: *"Channel-wise z-score normalization is performed using the mean and standard
deviation computed from the training split, and the same statistics are applied to
the test data."*

Read this carefully, because it is the one place a single-subject pipeline silently
breaks: the statistics are **per subject**. Nine subjects concatenated under one
global mean/std would let the subject with the largest amplitude own the scale.

SCORE adds a second, complementary mechanism: **Euclidean Alignment** (He & Wu 2020),
which whitens each subject's trials against a reference covariance before training.
It is label-free, unsupervised, and cheap. But do not over-rate it -- measured on a
*strong* encoder in SCORE's own controlled table it adds only **+1.23 Top-1** and is
**not significant** (sign test p=0.585), even though the classical literature reports
+4.33% on attention-decoding. The gain shrinks as the encoder improves.

## 6. Known gaps in our pipeline

Ordered by cost, cheapest first. Items 1-3 and 6 are now closed; 4-5 are wired but
unexercised.

1. ~~**No validation split (-150 concepts).**~~ **Closed.** `--val-concepts 0` plus
   `--select-last`, and the two are derived from each other rather than checked by
   hand: with no holdout there is no selection signal, so `select_last` follows.
2. ~~**Single subject only.**~~ **Closed.** `load_loso` returns nine concatenated
   source subjects, `build_from_args` takes `n_subjects`, and every row carries its
   subject index.
3. ~~**No per-subject normalization.**~~ **Closed.** `load_subject_std` z-scores each
   subject with its own training statistics before concatenation, and
   `assert_standardised` verifies it reached the array.
4. **`subject_ids` are plumbed but never supplied in a real run.** The path exists end
   to end (`encode_eeg(eeg, subject_ids, training)` -> `LayerFusion.forward`) and the
   LOSO unit tests exercise it, but no training script has yet passed a real
   `subject_of_row` tensor through a full run. Until then `subject_residual` is a
   zero-initialised `nn.Embedding`. *First smoke run closes this.*
5. **Target-side routing is not subject-conditioned by default.** `--target-fusion
   routed_sr` implements SAMGA's `b_s`, including the "subject-aware training /
   subject-agnostic inference" split the ablation credits for the inter-subject gain,
   and `encode_image` takes `subject_ids`. Off by default; needs a run.
6. ~~**No MVNN.**~~ **Closed.** `--mvnn {off,train,test}`; every inter-subject paper
   on this benchmark applies it.

## 7. Targets to beat

THINGS-EEG inter-subject, LOSO, 63 channels, 200-way:

| method | Top-1 | Top-5 | note |
|---|---:|---:|---|
| ATM (NeurIPS 2024) | 5.5 | 20.0 | the original cross-subject baseline |
| SATTC (CVPR 2026) | 14.8 | 38.4 | + label-free test-time calibration |
| SAMGA (ESWA 2026) | **34.4** | **64.8** | SAMGA's OWN Table 2, 5 seeds, best-epoch |
| SAMGA encoder, SCORE's protocol | **26.22 ± 1.08** | 57.98 ± 0.88 | SCORE's Table 2 "Original \| None", 3 seeds, final epoch |
| SCORE (2026) = SAMGA encoder + recovery | **53.23 ± 1.62** | **83.55 ± 1.13** | complete SCORE, 3 seeds |

**SUB-08 SPECIFICALLY**, which is the only published cell we can compare a one-fold run to:
SAMGA's Table 2 gives held-out sub-08 = **28.7 Top-1 / 59.5 Top-5**.

**Why there are two SAMGA rows, and which one is the target.**
This file previously carried a single row attributing 26.22 to "SAMGA (ESWA 2026)". That
attribution was wrong in a way that matters. 26.22 is SCORE's *re-measurement* of the
SAMGA encoder under SCORE's own protocol (50 epochs, final epoch, "Original | None" row
of SCORE's Table 2). SAMGA's own paper reports **34.4 / 64.8**, averaged over five
seeds. The 8.2-point spread between the two is not a detail: it is the difference
between "report the final epoch" and "keep the best epoch", and it decides what a
faithful reproduction should be aiming at.

That distinction is load-bearing here because SAMGA's released code
(`third_party/SAMGA/train.py`, `--early_stop_patience 10` by default in `inter.sh`)
evaluates on the test set every epoch and keeps the best, so **SAMGA's published
inter-subject numbers are test-set selected**; SCORE re-ran the same encoder with
final-epoch reporting and got 26.22. Our own runs use `--select-last`, which means:
  * our final-epoch number is comparable to SCORE's 26.22 row, and
  * comparing it to SAMGA's 34.4 charges us for an advantage they did not earn.
Both are reported by `scripts/compare_inter_arms.py`, labelled.

One more discrepancy worth recording: SAMGA's *paper* says 60 epochs (Table 1) while
its *released* `inter.sh` sets `NUM_EPOCHS=50`, and SCORE also uses 50. We follow the
released code, which is the reproducible object.

**Our data is the same benchmark.** `N_TRAIN_CONCEPTS = 1654`, `N_TEST_CONCEPTS = 200`,
10 images/concept, 4 training reps, 80 test reps, 63 channels -- identical to
SAMGA's, SATTC's and SCORE's description. So the DATASET is directly comparable.

**But the numbers in the table above are not subtractable from a single-fold run, and
this file used to say they were.** Two independent invalidations, both measured, both
of which must be cleared before a difference in that table means anything:

  1. **Every row above is a multi-fold, multi-seed average** (SAMGA 5 seeds x 10 folds;
     SCORE 3 seeds x 10 folds; hence the ±1.08 and ±1.62). A single fold is one draw
     from that distribution. Our one-fold official-baseline run scored **19.00 final /
     22.00 best** against SAMGA's sub-08 cell of 28.7; that 6.7-to-9.7 point gap is
     NOT evidence about anything until (2) is cleared and the other nine folds exist.
     The repo now records sub-08's published value precisely so this comparison is at
     least fold-matched rather than average-matched.
  2. **The image features differ.** Those rows used `InternViT-6B-448px-V2.5`,
     layers 20/24/28/32/36. Every run recorded before 2026-09-23 19:26 was fed five
     CLIP ViT-H-14 layers (blocks 22/24/26/28/30), because InternViT was believed
     unavailable. **That belief was wrong on both counts it rested on**, and the
     substitution has since been removed: all 16,540 training images and all 200 test
     images are on disk under `data/images_set/`, the login node has outbound network,
     and `OpenGVLab/InternViT-6B-448px-V2_5` is 11.1 GB. `slurm/extract_internvit.sbatch`
     produced the five per-layer arrays in SAMGA's own
     `internvit_multilevel_20_24_28_32_36` layout, with the stacked cache verified at
     `(1654, 10, 5, 3200)` / `(200, 1, 5, 3200)` -- the shape SAMGA's README specifies.

     The extractor gates that had to pass first, because a mis-pathed hook or a
     transposed axis yields correctly-shaped arrays carrying the wrong information:
       * the image listing reproduces `image_metadata.npy`'s `{train,test}_img_files`
         sequence exactly (16540/16540 and 200/200, complete) -- this is the row-alignment
         proof and it is independent of the visual backbone;
       * within-concept image similarity exceeds cross-concept similarity at the middle
         layer (0.316 vs 0.003 after global centering).

     Note that raw cosines are ~0.994 for *any* two images, which is why that gate centers
     globally and why comparing these features by raw cosine is not meaningful.

     **Removing that substitution is what makes the current run answerable to the
     paper.** The baseline is being rerun (sub-08, seed 2025, one fold) on the InternViT
     features; `scripts/run_samga_official_baseline.sh` now fails hard if the model does
     not report `image raw feature dimension: 3200`, so this cannot silently regress.

What that leaves: differences BETWEEN our own arms remain claims, because both sides
share every one of the above. Differences between us and the table require all ten
folds before they can be read at all -- though the feature half of that gap is now
closed, so a one-fold sub-08 run can at least be checked against SAMGA's own published
sub-08 cell, 28.7/59.5 (their Table 2).

Two corrections to what this file claimed before, both because the sources were
read rather than recalled:

  * **The ATM row.** "ATM (NeurIPS 2024) 11.9 / 33.8" is an INTRA-subject number.
    SATTC's Table 1 reports ATM's own cross-subject baseline as **20.0 Top-5 / 5.5
    Top-1**, and shows that merely standardising inference -- cosine on
    L2-normalised features plus candidate whitening -- lifts the same frozen
    features to **30.5 / 9.2**. Planning against 11.9 means planning against a
    number from a different task.
  * **SCORE is the same benchmark, not a different collection.** This file
    previously called it incomparable as "THINGS-EEG2, a different collection".
    SCORE's own setup: *"We evaluate SCORE under LOSO protocols on THINGS-EEG2 and
    Alljoined-1.6M ... on THINGS-EEG2 we use all 63 channels, segment epochs from 0
    to 1000 ms after onset, baseline-correct with the 200 ms pre-stimulus mean,
    resample to 250 Hz, apply multivariate noise normalization (MVNN), and average
    repetitions of the same image."* That is our protocol, transcribed. So
    **53.23 / 83.55 is the number to beat**, and it is SAMGA plus a deployment-time
    orthogonal map -- not a different dataset.

What this implies for the plan: **the gap at the top is not mainly an encoder
gap.** SCORE runs on top of SAMGA and adds 27 points of Top-1 with both encoders
frozen, no target labels and no encoder updates, by aligning the subject's
coordinate frame to the image space. A plan that spends its budget on the
alignment target while neglecting deployment-time geometry is optimising the
smaller term.

## 8. The target-layer question is open

Shallow Alignment's Table 4 reports, for **ViT-H-14** (the backbone we use):

| | relative depth | intra Top-1 | inter Top-1 | inter Top-5 |
|---|---|---:|---:|---:|
| final output | 100% | 33.5 | 7.8 | 23.5 |
| best intermediate | **32.3% (L11)** | 76.8 | **19.0** | 44.2 |

If that transferred, we would be running the wrong layer: we align to `block26`
(relative depth 80.6%). But **our own val-selected ridge probe on our own cached
features does not reproduce it** (subject 08, 17 ch, 150-concept holdout):

| layer | rel. depth | val Top-1 | test Top-1 |
|---|---:|---:|---:|
| block11 | 32.3% | 12.27 | 16.50 |
| block17 | 51.6% | 14.53 | 18.00 |
| **block24** | 74.2% | 17.07 | **28.50** |
| block26 | 80.6% | 17.33 | 27.50 |
| `_pooled` | 100% | 15.40 | 25.00 |

Two candidate explanations, both testable: (a) their sweep probed only ~10 evenly
spaced layers ("*we adopt a uniform sampling strategy, probing approximately ten
evenly spaced intermediate layers*"), so L11 may be the best *among probed* layers
rather than the true peak; (b) ridge and a trained encoder can prefer different
depths. Either way the question is **open, and it is an inter-subject question** --
our probe is intra, and Shallow Alignment traces the full curve for intra only. It
must be re-measured with 9 training subjects before the architecture is frozen.

Worth noting in the same breath: the same paper finds that *fusion* beats any single
layer -- InternViT `{L28}` 82.6 -> `{L24,L28}` 87.8 -> +more -> 90.8 -> +adaptive
pooling 91.3 -- and SAMGA's ablation agrees (fixed 87.8 / uniform 89.1 / learned 91.3).
We already have `--target-layers` and `--target-fusion {single,mean,routed}` wired.
