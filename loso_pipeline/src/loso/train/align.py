"""Stage 2: multi-modal alignment with subject invariance and disentanglement.

One training step and its rationale
-----------------------------------
`x` (B, 63, 250) goes through the encoder, which returns `z_inv`, `z_sub` and the
Transformer tokens.  Five projections of `z_inv` are matched to the frozen teachers,
and three regularisers act on the split itself:

  L_img   soft-label InfoNCE, P_img(z_inv) vs CLIP image embedding   (primary)
  L_text  soft-label InfoNCE, P_text(z_inv) vs CLIP caption embedding (auxiliary)
  L_dino  Huber,             P_dino(z_inv) vs DINOv2 feature          (structure)
  L_vae   cosine+SmoothL1,   P_vae(z_inv)  vs pooled VAE latent       (coarse appearance)
  L_time  set alignment,     P_time(tokens) vs VAE latent patches     (time-resolved)
  L_trial repeated-trial contrastive over z_inv
  L_var   VICReg variance hinge on z_inv                              (anti-collapse)
  L_cov   VICReg off-diagonal covariance of z_inv                     (anti-collapse)
  L_adv   GRL subject classifier on z_inv                             (capped)
  L_dist  CORAL + MMD between subjects' z_inv                         (capped)
  L_orth  ||z_inv^T z_sub||_F                                         (small)

Three details worth stating because they are easy to get wrong:

* The soft-label targets come from the *frozen teacher* similarity, not from the
  encoder's own outputs.  Computing them from the encoder would let the model
  minimise the loss by moving the targets rather than the features.  The
  implementation is already inside `no_grad`.

* `L_var`/`L_cov` are not optional extras.  Every other term here is satisfied by a
  near-constant `z_inv`, and the previous run demonstrated it: it reached a total
  loss of 31.90 against 32.57 for an encoder that emits one fixed vector.  See
  `loso.diagnostics` for what that looked like and why two statistics are needed.

* The adversarial and distribution terms are capped by design.  Left at parity they
  dominate, and the encoder then discards subject-discriminative *and* task-relevant
  signal together -- the standard failure of an unbounded domain-adversarial loss.

Batches are drawn by `GroupedBatchSampler`, not uniformly: the soft label needs
same-image peers in the batch to be anything other than a one-hot diagonal, and the
previous run's batches gave 75.8% of rows no peer at all.

Held-out subject isolation
--------------------------
The held-out subject appears in no loss, no normalisation statistic (under the
default `norm_source="train_subjects"`), no early-stopping decision and no
hyper-parameter choice.  It is touched only by `evaluate`, which is read-only, and
its retrieval accuracy is what selects the checkpoint.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from loso import diagnostics as D
from loso import paths
from loso.data import eeg as eeg_mod
from loso.data import things
from loso.data.targets import TargetStore
from loso.losses import align as L
from loso.models.eeg_encoder import EEGEncoder, EncoderConfig
from loso.models.heads import AlignmentHeads, HeadConfig, vae_patched, vae_pooled
from loso.train.sampler import GroupedBatchSampler


@dataclass
class AlignConfig:
    test_subject: str = "sub-08"
    epochs: int = 60
    batch_size: int = 512
    lr: float = 3e-4
    head_lr_mult: float = 3.0     # projectors are fresh; the trunk starts pretrained
    weight_decay: float = 0.05
    warmup_epochs: int = 3
    grad_clip: float = 1.0
    topk: int = 10
    norm_source: str = "train_subjects"
    amp_dtype: str = "bf16"
    num_workers: int = 8
    eval_every: int = 5
    eval_trials: int = 0          # 0 = all 16,000 test trials
    log_every: int = 50
    seed: int = 0
    max_steps_per_epoch: int = 0  # 0 = no cap; set for smoke tests
    adv_warmup_epochs: int = 5    # ramp the GRL so alignment establishes first
    #: "grouped" draws whole same-image groups into each batch (see
    #: `GroupedBatchSampler`); "shuffled" is uniform sampling.  Grouped is the
    #: default because the soft label needs same-image peers to be informative: the
    #: previous run's uniform batches left 75.8% of rows with no positive peer, so
    #: their soft target collapsed to a one-hot and the term degenerated to plain
    #: InfoNCE.  "shuffled" is kept so the effect of the sampler can be measured
    #: rather than assumed.
    sampler: str = "grouped"
    #: Sub-directory under `ALIGN_DIR/<subject>/`.  Empty means the canonical
    #: location that Stage 3 reads.  Set for previews and ablations so a short
    #: diagnostic run cannot overwrite the real run's `best.pt` and `history.json` --
    #: which it otherwise would, since the path is derived only from the subject.
    run_tag: str = ""
    #: How many steps of the first epoch run the per-term gradient-budget probe
    #: (`measure_gradient_budget`).  Each measured step costs one backward per term,
    #: so 0 keeps it off; 1-2 is enough to see which term owns the gradient.
    gradient_budget_steps: int = 0
    #: Retrieval protocols reported at each evaluation.  `1` is the headline
    #: single-trial number; `80` averages all 80 repetitions of each test concept,
    #: which is the high-SNR setting the literature usually quotes.  Reporting both
    #: matters because they are not interchangeable and a checkpoint can be better at
    #: one than the other -- the previous run reported a single number and it was
    #: ambiguous which regime it described.
    eval_avg_reps: tuple[int, ...] = (1, 80)
    #: Average the 4 train repetitions of each image before the encoder.  UCK /
    #: SAMGA both train on the averaged trial; it cuts the epoch length by 4x and
    #: raises the SNR of every supervision signal.  Single-trial evaluation is
    #: unchanged -- only the *training* distribution is averaged.
    avg_trials: bool = True
    #: Top-k for the UCK memory retrieve term.  Kept small so the soft prototype
    #: stays local; UCK measured k=16.
    mem_k: int = 16

    # --- architecture ---
    enc: EncoderConfig = field(default_factory=EncoderConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    weights: L.LossWeights = field(default_factory=L.LossWeights)


def build_optimizer(model: torch.nn.Module, heads: torch.nn.Module,
                    cfg: AlignConfig) -> torch.optim.Optimizer:
    """AdamW with no weight decay on norms, biases or the learned temperature.

    Decaying a LayerNorm gain or the logit scale biases them toward zero for no
    statistical reason; the temperature in particular would drift downward and
    quietly sharpen every contrastive term.

    Heads use a higher LR (`head_lr_mult`) because they are freshly initialised
    while the trunk starts from a larger parameter set; they must NOT also appear
    in the trunk groups -- AdamW rejects duplicated parameters.
    """
    enc_decay: list[torch.nn.Parameter] = []
    enc_no_decay: list[torch.nn.Parameter] = []
    head_decay: list[torch.nn.Parameter] = []
    head_no_decay: list[torch.nn.Parameter] = []
    for module, decay_g, no_decay_g in (
        (model, enc_decay, enc_no_decay),
        (heads, head_decay, head_no_decay),
    ):
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name.endswith("logit_scale") or "region_embed" in name:
                no_decay_g.append(p)
            else:
                decay_g.append(p)
    head_lr = cfg.lr * cfg.head_lr_mult
    return torch.optim.AdamW([
        {"params": enc_decay, "weight_decay": cfg.weight_decay, "lr": cfg.lr},
        {"params": enc_no_decay, "weight_decay": 0.0, "lr": cfg.lr},
        {"params": head_decay, "weight_decay": cfg.weight_decay, "lr": head_lr},
        {"params": head_no_decay, "weight_decay": 0.0, "lr": head_lr},
    ], betas=(0.9, 0.98), eps=1e-8)


def lr_at(step: int, total: int, warmup: int, base: float) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def build_concept_gallery(store: TargetStore, device: torch.device) -> torch.Tensor:
    """Mean CLIP-image embedding per train concept -- UCK's G_img.

    Shape (n_concepts, d).  Built once and kept on device; every training step
    only reads it.  The 1,654 concepts are disjoint from the 200 test concepts by
    construction of the THINGS-EEG2 split.
    """
    ipc = store.shapes.images_per_concept
    n_concepts = store.shapes.n_images // ipc
    rows = torch.arange(store.shapes.n_images, device="cpu")
    img = store.as_normalized(rows)["clip_image"].float()
    img = img.view(n_concepts, ipc, -1).mean(dim=1)
    return F.normalize(img, dim=-1).to(device)


def compute_losses(model: EEGEncoder, heads: AlignmentHeads, store: TargetStore,
                   batch: dict[str, torch.Tensor], cfg: AlignConfig,
                   grl_lambda: float,
                   concept_gallery: torch.Tensor | None = None,
                   ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One forward pass and all loss terms.

    Terms whose weight is 0.0 are not computed.  That is load-bearing for
    wall-time under the UCK recipe: `set_alignment` alone materialises a ~4 GB
    similarity at batch 512, and skipping it when `weights.time == 0` is what
    makes the step affordable.
    """
    x = batch["x"]
    subject_id = batch["subject_id"]
    slots = batch["target_slot"]
    w = cfg.weights

    need_tokens = w.time != 0.0
    out = model(x, subject_id, return_tokens=need_tokens)
    z_inv, z_sub = out["z_inv"], out["z_sub"]
    tokens = out.get("tokens")
    teacher = store.as_normalized(slots, device=x.device)
    scale = heads.scaled_logit_scale()

    terms: dict[str, torch.Tensor] = {}
    concept = slots // store.shapes.images_per_concept
    same_concept = L.same_concept_mask(concept)

    # --- cross-modal alignment (in-batch) -------------------------------------
    if w.img != 0.0:
        sim_img = L.teacher_similarity(L.center_normalize(teacher["clip_image"]))
        p_img = heads.img(z_inv)
        terms["img"] = L.soft_label_contrastive(
            p_img, teacher["clip_image"], scale, sim_img, cfg.topk, same_concept,
            center=True,
        )
    else:
        p_img = heads.img(z_inv)

    if w.text != 0.0:
        sim_text = L.teacher_similarity(L.center_normalize(teacher["clip_text_caption"]))
        p_text = heads.text(z_inv)
        terms["text"] = L.soft_label_contrastive(
            p_text, teacher["clip_text_caption"], scale, sim_text, cfg.topk,
            same_concept, center=True,
        )

    if w.dino != 0.0:
        terms["dino"] = L.huber_alignment(heads.dino(z_inv), teacher["dino"])

    if w.vae != 0.0:
        latent = teacher["vae_latent"]
        p_vae = heads.vae(z_inv)
        terms["vae"] = L.cosine_mse(
            p_vae, F.normalize(vae_pooled(latent, cfg.head.vae_pool), dim=-1))

    # --- UCK concept-gallery terms -------------------------------------------
    # Against the fixed 1,654-row train-concept gallery, not the in-batch keys.
    # This is the measured UCK primary; the in-batch soft-label term alone sat on
    # its constant-encoder plateau for the previous run.
    if concept_gallery is not None and (w.gallery != 0.0 or w.mem != 0.0):
        if w.gallery != 0.0:
            terms["gallery"] = L.gallery_nce(p_img, concept_gallery, concept, scale,
                                             center=True)
        if w.mem != 0.0:
            terms["mem"] = L.memory_alignment(p_img, concept_gallery, scale,
                                              k=cfg.mem_k)

    # --- time-resolved alignment (expensive; skipped when weight is 0) --------
    if w.time != 0.0:
        assert tokens is not None
        latent = teacher["vae_latent"]
        n_patch = cfg.head.n_time_patches
        tok = tokens.transpose(1, 2)
        if tok.shape[-1] != n_patch:
            tok = F.adaptive_avg_pool1d(tok, n_patch)
        p_time = heads.time(tok.transpose(1, 2))
        patches = vae_patched(latent, cfg.head.vae_pool, n_patch)
        terms["time"] = L.set_alignment(p_time, patches, scale)

    # --- anti-collapse --------------------------------------------------------
    z_centred = z_inv - z_inv.mean(dim=0, keepdim=True)
    if w.var != 0.0:
        terms["var"] = L.vicreg_variance(z_centred, gamma=cfg.head.vicreg_gamma)
    if w.cov != 0.0:
        terms["cov"] = L.vicreg_covariance(z_centred)

    # --- invariance and disentanglement ---------------------------------------
    if w.trial != 0.0:
        terms["trial"] = L.repeated_trial_contrastive(z_inv, slots, scale)

    if w.adv != 0.0:
        logits_subject = heads.subject(z_inv, grl_lambda)
        terms["adv"] = L.subject_adversarial_loss(logits_subject, subject_id)

    if w.dist != 0.0:
        n_subj = int(subject_id.max().item()) + 1
        terms["dist"] = _cross_subject_distance(z_inv, subject_id, n_subj)

    if w.orth != 0.0:
        terms["orth"] = L.orthogonality_loss(z_inv, z_sub)

    # Only terms that were actually produced are required to be present.  Zero-
    # weight terms are deliberately absent so `weighted_total`'s `active` check
    # does not force us to spend the compute we just skipped.
    active = set(terms)
    total = L.weighted_total(terms, cfg.weights, active=active)
    return total, terms


def _cross_subject_distance(z: torch.Tensor, subject_id: torch.Tensor,
                            n_subjects: int) -> torch.Tensor:
    """CORAL + MMD averaged over subject pairs present in the batch.

    Pairwise rather than one-vs-rest: a global mean-matching term is satisfied by
    aligning everything to the batch centre, which is exactly the collapse the
    design warns about.  Pairwise requires each subject to agree with each other,
    which fixes all of them relative to one another.
    """
    groups = [z[subject_id == s] for s in range(n_subjects)]
    groups = [g for g in groups if g.shape[0] >= 2]
    if len(groups) < 2:
        return z.new_zeros(())
    coral_terms, mmd_terms = [], []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            coral_terms.append(L.coral_loss(groups[i], groups[j]))
            mmd_terms.append(L.mmd_loss(groups[i], groups[j]))
    return torch.stack(coral_terms).mean() + torch.stack(mmd_terms).mean()


def measure_gradient_budget(model: EEGEncoder, heads: AlignmentHeads,
                            store: TargetStore, batch: dict[str, torch.Tensor],
                            cfg: AlignConfig, grl_lambda: float,
                            concept_gallery: torch.Tensor | None = None,
                            ) -> dict[str, float]:
    """Share of the encoder's gradient produced by each loss term.

    Requires one backward per term, so it is a diagnostic and not part of the
    training loop; `AlignConfig.gradient_budget_steps` controls how many steps run it.

    This exists because the weights are not the budget.  The previous run gave
    `L_time` weight 3.0 and it produced ~58% of the gradient, which is how a term
    built on a false correspondence came to dominate the objective -- and since the
    weights looked balanced on paper, nothing in the logs showed it.  A term's share
    depends on its weight *and* on the gradient magnitude it produces, and only the
    latter is a property of the data and the current parameters.

    The gradient is taken with respect to the *encoder* alone.  That is the quantity
    of interest: the projectors are free to fit their targets, and the question is how
    much of the trunk's learning each term is responsible for.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    _, terms = compute_losses(model, heads, store, batch, cfg, grl_lambda,
                              concept_gallery=concept_gallery)
    budget = D.GradientBudget()
    for name, value in terms.items():
        weight = getattr(cfg.weights, name, 0.0)
        if weight == 0.0:
            continue
        grads = torch.autograd.grad(value * weight, params, retain_graph=True,
                                    allow_unused=True)
        norm = math.sqrt(sum(float(g.pow(2).sum()) for g in grads if g is not None))
        budget.record(name, norm)
    return budget.shares()


def build_train_loader(ds: eeg_mod.TrainEEGDataset, cfg: AlignConfig) -> DataLoader:
    """DataLoader over the training subjects, with the configured batch sampling.

    With `sampler="grouped"` the sampler yields whole same-image groups and the
    DataLoader must be given `batch_sampler=`.  Passing `batch_size` and `shuffle`
    *as well* is the mistake to avoid: `batch_sampler` is mutually exclusive with
    them, and the combination fails at runtime with an indexing error that names the
    wrong thing (the sampler is handed a list of indices and an int where it expects
    a single index).
    """
    if cfg.sampler == "grouped":
        # One group per image slot: the trials that were evoked by the same
        # photograph.  `TrainEEGDataset.trials` is (subject, trial, slot) per row, and
        # with repetitions already averaged in the release every training subject
        # contributes exactly one row per slot, so a group is the N-1 training
        # subjects' responses to one image.
        groups = [slot for _, _, slot in ds.trials]
        sampler = GroupedBatchSampler(groups, cfg.batch_size, shuffle=True,
                                      seed=cfg.seed)
        print(f"[align] grouped sampler: {sampler.groups_per_batch} images/batch "
              f"x group_size {sampler.group_size} = "
              f"{sampler.groups_per_batch * sampler.group_size} rows, "
              f"{len(sampler)} batches/epoch", flush=True)
        return DataLoader(ds, batch_sampler=sampler, num_workers=cfg.num_workers,
                          collate_fn=eeg_mod.collate, pin_memory=True,
                          persistent_workers=cfg.num_workers > 0)
    if cfg.sampler != "shuffled":
        raise ValueError(f"unknown sampler {cfg.sampler!r}; "
                         f"expected 'grouped' or 'shuffled'")
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                      num_workers=cfg.num_workers, collate_fn=eeg_mod.collate,
                      pin_memory=True, persistent_workers=cfg.num_workers > 0)


@torch.inference_mode()
def _retrieval(model: EEGEncoder, heads: AlignmentHeads, store: TargetStore,
               test_ds: eeg_mod.TestEEGDataset, cfg: AlignConfig,
               device: torch.device, avg_reps: int, limit: int = 0
               ) -> tuple[dict[str, float], torch.Tensor, float]:
    """Retrieval accuracy for one test protocol.

    The gallery is the full test concept set, so a query's correct match is its own
    concept index and the ranking is the "200-way" of the design's protocol.

    Also returns the batch's `z_inv` (for collapse diagnostics) and the loss of a
    constant-encoder baseline on this batch (see `loso.diagnostics`).
    """
    loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=eeg_mod.collate,
                        pin_memory=True)
    # Centred, because the training objective is: `terms["img"]` now compares centred
    # queries against centred keys, so scoring an uncentred gallery here would measure a
    # different objective than the one being optimised.  The gallery's mean is taken
    # over the whole 200-row concept set, so it is a fixed direction rather than a
    # per-batch one, while each query batch is centred on its own statistics -- the same
    # empirical estimate of the shared direction that training removes.
    gallery = store.arrays["clip_image"]
    gallery = L.center_normalize(torch.from_numpy(np.asarray(gallery, dtype=np.float32))
                                 .to(device))
    teacher = store.as_normalized(
        torch.arange(gallery.shape[0], device=device), device=device)["clip_image"]

    correct1 = correct5 = total = 0
    per_concept_correct: dict[int, int] = {}
    per_concept_total: dict[int, int] = {}
    z_chunks: list[torch.Tensor] = []
    baseline: list[float] = []
    for step, batch in enumerate(loader):
        if limit and step * cfg.batch_size >= limit:
            break
        x = batch["x"].to(device, non_blocking=True)
        slots = batch["target_slot"].to(device)
        # The held-out subject has no adapter of its own, so the population-mean
        # adapter is used.  Selecting an arbitrary training subject here, or
        # bypassing the adapter, would respectively inject an unrelated subject's
        # statistics or change the function the encoder computes at train time.
        z_inv = model(x, None, adapter_mode="mean")["z_inv"]
        z_chunks.append(z_inv.float().cpu())
        query = L.center_normalize(heads.img(z_inv))
        sim = query @ gallery.t()

        rank = sim.argsort(dim=-1, descending=True)
        hit1 = rank[:, 0] == slots
        hit5 = (rank[:, :5] == slots[:, None]).any(dim=-1)
        correct1 += int(hit1.sum())
        correct5 += int(hit5.sum())
        total += slots.numel()
        for s, h in zip(slots.tolist(), hit1.tolist()):
            per_concept_total[s] = per_concept_total.get(s, 0) + 1
            per_concept_correct[s] = per_concept_correct.get(s, 0) + int(h)

        # The constant baseline is evaluated through the same objective on this exact
        # batch, so the comparison is against this batch's concept sample rather than
        # a published number: the failed run's 31.90-vs-32.57 was a 2.3% margin, and
        # that margin only means something against the same concept set.
        baseline.append(D.constant_encoder_loss(
            query, teacher[slots], heads.scaled_logit_scale(), center=True))

    if total == 0:
        empty = torch.zeros(0, 0)
        return {"n": 0.0}, empty, float("nan")
    accs = [per_concept_correct.get(c, 0) / per_concept_total[c]
            for c in per_concept_total]
    return {
        "n": float(total),
        "top1": correct1 / total,
        "top5": correct5 / total,
        "mean_concept_top1": float(np.mean(accs)),
        "worst_concept_top1": float(np.min(accs)),
        "acc": float(np.mean(accs)),
    }, torch.cat(z_chunks, dim=0), float(np.mean(baseline))


def evaluate(model: EEGEncoder, heads: AlignmentHeads, store: TargetStore,
             test_ds_by_reps: dict[int, eeg_mod.TestEEGDataset], cfg: AlignConfig,
             device: torch.device, limit: int = 0,
             initial: dict[str, float] | None = None) -> dict[str, object]:
    """Retrieval over every configured protocol, plus representation diagnostics.

    `initial` is the diagnostics measured on the untrained model.  Passing it turns
    the collapse check into a drift check, which is the only form that works here:
    at initialisation this encoder already reports `mean_offdiag_cosine = +0.80`, so
    an absolute threshold would flag every run from epoch 0.  See
    `loso.diagnostics.collapse_verdict`.
    """
    model.eval()
    metrics: dict[str, object] = {}
    z_all: list[torch.Tensor] = []
    for reps in cfg.eval_avg_reps:
        ds = test_ds_by_reps[reps]
        got, z_inv, baseline = _retrieval(model, heads, store, ds, cfg, device,
                                         reps, limit)
        suffix = "" if reps == 1 else f"_avg{reps}"
        for key, value in got.items():
            metrics[key if key == "n" else key + suffix] = value
        metrics[f"constant_baseline{suffix}"] = baseline
        z_all.append(z_inv)
    model.train()

    # Diagnostics on the single-trial z_inv, which is the representation every
    # downstream stage consumes.
    stats = D.representation_diagnostics(z_all[0])
    ok, verdict = D.collapse_verdict(stats, initial=initial)
    metrics.update(stats)
    metrics["collapse_ok"] = ok
    metrics["collapse_verdict"] = verdict
    return metrics


def run(cfg: AlignConfig) -> Path:
    """Train Stage 2. Returns the path to the best checkpoint."""
    paths.ensure_dirs()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = paths.ALIGN_DIR / cfg.test_subject
    if cfg.run_tag:
        # A preview must not be able to overwrite the canonical checkpoint: the path
        # would otherwise be identical, and a 20-step diagnostic run would leave a
        # `best.pt` that Stage 3 would happily load.
        run_dir = run_dir / cfg.run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    train_subjects, test_subject = things.loso_split(cfg.test_subject)
    n_subjects = len(train_subjects)
    print(f"[align] test={test_subject} train={n_subjects} subjects "
          f"{train_subjects}", flush=True)

    # --- data -----------------------------------------------------------------
    print("[align] fitting EEG normalizers ...", flush=True)
    per_subject, global_stats = eeg_mod.build_normalizers(
        train_subjects, cfg.norm_source,
        cache_path=paths.DATA_ROOT / f"norm_{cfg.norm_source}_{n_subjects}.json",
    )
    if cfg.norm_source == "train_subjects":
        # The held-out subject receives the training subjects' affine map; nothing
        # about it enters the statistics.  See `loso.data.eeg` for the inherited
        # MVNN caveat that this cannot undo.
        norm = {s: global_stats for s in list(train_subjects) + [test_subject]}
        norm_keys = {s: s for s in norm}
    else:
        norm = dict(per_subject)
        norm_keys = {s: s for s in per_subject}
        test_arr = eeg_mod.load_eeg(test_subject, "train", mmap=False)
        norm[test_subject] = eeg_mod.fit_channel_stats([test_arr])
        norm_keys[test_subject] = test_subject

    train_ds = eeg_mod.TrainEEGDataset(
        train_subjects, norm, norm_keys,
        augment_cfg=eeg_mod.AugmentConfig(), seed=cfg.seed,
        avg_trials=cfg.avg_trials,
    )
    # One dataset per protocol.  `avg_reps` controls how many of the 80 repetitions
    # of each test concept are averaged, so `avg_reps=1` is the single-trial headline
    # and `avg_reps=80` the high-SNR number.  Both are built because they are not
    # interchangeable, and a checkpoint scored on one can lose on the other.
    test_ds_by_reps = {
        reps: eeg_mod.TestEEGDataset(test_subject, norm[test_subject],
                                     avg_trials=reps > 1, avg_reps=reps)
        for reps in cfg.eval_avg_reps
    }
    train_loader = build_train_loader(train_ds, cfg)
    print(f"[align] {len(train_ds)} train trials"
          f"{' (avg-reps)' if cfg.avg_trials else ''}; test protocols "
          + ", ".join(f"{r}-trial avg ({len(d)} queries)"
                      for r, d in test_ds_by_reps.items()), flush=True)

    store = TargetStore("train", names=("clip_image", "clip_text_caption", "dino",
                                        "vae_latent"))
    test_store = TargetStore("test", names=("clip_image", "clip_text_caption", "dino",
                                            "vae_latent"))
    # UCK G_img: mean CLIP-image embedding per train concept.  Built once; every
    # training step only reads it.  Disjoint from the 200 test concepts by the
    # THINGS-EEG2 split.
    concept_gallery = build_concept_gallery(store, device)
    print(f"[align] UCK concept gallery {tuple(concept_gallery.shape)} "
          f"(gallery={cfg.weights.gallery} mem={cfg.weights.mem})", flush=True)

    # --- model ----------------------------------------------------------------
    cfg.enc.pretrained_subjects = n_subjects
    cfg.head.n_subjects = n_subjects
    cfg.head.d_model = cfg.enc.d_model
    cfg.head.d_inv = cfg.enc.d_inv
    cfg.head.d_sub = cfg.enc.d_sub
    cfg.head.n_time_patches = cfg.enc.n_tokens
    model = EEGEncoder(cfg.enc).to(device)
    heads = AlignmentHeads(cfg.head).to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    n_head = sum(p.numel() for p in heads.parameters())
    print(f"[align] encoder {n_params / 1e6:.2f}M + heads {n_head / 1e6:.2f}M params",
          flush=True)
    # The previous run's projectors were 2.11x the trunk, which is what let them fit
    # the frozen targets without the encoder contributing anything.  Printed as a
    # ratio rather than two absolute numbers because the ratio is the thing that
    # matters, and asserted in `scripts/smoke_align.py`.
    print(f"[align] heads/trunk parameter ratio {n_head / n_params:.2f}x "
          f"(must be < 1.0)", flush=True)

    opt = build_optimizer(model, heads, cfg)
    total_steps = min(len(train_loader), cfg.max_steps_per_epoch or len(train_loader)) * cfg.epochs
    warmup = min(total_steps // 20, cfg.warmup_epochs *
                 min(len(train_loader), cfg.max_steps_per_epoch or len(train_loader)))
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[cfg.amp_dtype]

    # --- the state the run must improve on ------------------------------------
    # Measured once, before any update, and used as the baseline for the collapse
    # check.  It has to be the *untrained* model on the same data: at initialisation
    # `mean_offdiag_cosine` is already +0.80 (80.4% of the sphere's energy in the
    # shared direction), so a fixed threshold would report every run as collapsed from
    # epoch 0 and the check would be ignored.  `limit` keeps this affordable -- it is
    # a scale reference, not a result.
    print("[align] measuring initialisation diagnostics ...", flush=True)
    init_metrics = evaluate(model, heads, test_store, test_ds_by_reps, cfg, device,
                            limit=2048)
    initial = {k: float(init_metrics[k]) for k in
               ("top1_sv_ratio", "eff_rank", "mean_offdiag_cosine")}
    print(f"[align] init: top1_sv_ratio={initial['top1_sv_ratio']:.4f} "
          f"eff_rank={initial['eff_rank']:.2f} "
          f"mean_cosine={initial['mean_offdiag_cosine']:+.4f} | "
          f"top1={init_metrics['top1']:.4f} "
          f"constant_baseline={init_metrics['constant_baseline']:.4f}", flush=True)

    (run_dir / "config.json").write_text(json.dumps(
        {"align": {k: v for k, v in asdict(cfg).items()},
         "train_subjects": train_subjects, "n_train_trials": len(train_ds)}, indent=1,
        default=str))

    best = -1.0
    step = 0
    history: list[dict] = []
    budget_done = False
    for epoch in range(cfg.epochs):
        train_ds.set_epoch(epoch)
        if isinstance(train_loader.batch_sampler, GroupedBatchSampler):
            # Reseeding per epoch keeps the group order and the within-group shuffle
            # from repeating across epochs.
            train_loader.batch_sampler.set_epoch(epoch)
        # Ramp the adversarial strength: applying a full-strength domain classifier
        # from step 0 pushes the encoder off the task-relevant manifold before
        # alignment has established it, and the run then never recovers.
        grl = 0.0 if epoch < cfg.adv_warmup_epochs else min(
            1.0, (epoch - cfg.adv_warmup_epochs + 1) / 5.0)
        t0 = time.time()
        agg: dict[str, float] = {}
        n_batch = 0

        for batch in train_loader:
            for g in opt.param_groups:
                base = cfg.lr * (cfg.head_lr_mult if g is opt.param_groups[2] else 1.0)
                g["lr"] = lr_at(step, total_steps, warmup, base)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            if (cfg.gradient_budget_steps and not budget_done
                    and step < cfg.gradient_budget_steps):
                shares = measure_gradient_budget(
                    model, heads, store, batch, cfg, grl,
                    concept_gallery=concept_gallery)
                ordered = sorted(shares.items(), key=lambda kv: -kv[1])
                print("[align] gradient budget (share of encoder gradient): "
                      + " ".join(f"{k}={v * 100:.0f}%" for k, v in ordered),
                      flush=True)
                budget_done = step + 1 >= cfg.gradient_budget_steps

            with torch.autocast("cuda", dtype=amp_dtype,
                                enabled=amp_dtype is not None and device.type == "cuda"):
                loss, terms = compute_losses(
                    model, heads, store, batch, cfg, grl,
                    concept_gallery=concept_gallery)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(heads.parameters()), cfg.grad_clip)
            opt.step()
            step += 1

            for k, v in terms.items():
                agg[k] = agg.get(k, 0.0) + float(v.detach())
            agg["total"] = agg.get("total", 0.0) + float(loss.detach())
            n_batch += 1

            if cfg.log_every and step % cfg.log_every == 0:
                dt = time.time() - t0
                print(f"[align] ep{epoch} step{step} "
                      + " ".join(f"{k}={v / n_batch:.4f}" for k, v in sorted(agg.items()))
                      + f" lr={opt.param_groups[0]['lr']:.2e} "
                        f"grl={grl:.2f} {dt / max(1, n_batch) * 1000:.0f}ms/it",
                      flush=True)
                n_batch = 0
                agg = {}
            if cfg.max_steps_per_epoch and step % cfg.max_steps_per_epoch == 0:
                break

        row: dict = {"epoch": epoch, "step": step, "grl": grl}
        if device.type == "cuda":
            row["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9

        if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            metrics = evaluate(model, heads, test_store, test_ds_by_reps, cfg, device,
                               limit=cfg.eval_trials, initial=initial)
            row.update(metrics)
            print(f"[align] == epoch {epoch} eval:\n"
                  f"    top1={metrics['top1']:.4f} top5={metrics['top5']:.4f} "
                  f"mean_concept={metrics['mean_concept_top1']:.4f} "
                  f"worst={metrics['worst_concept_top1']:.4f} "
                  f"n={metrics['n']:.0f}\n"
                  f"    avg80: top1={metrics.get('top1_avg80', float('nan')):.4f} "
                  f"top5={metrics.get('top5_avg80', float('nan')):.4f}\n"
                  f"    constant_encoder_baseline: "
                  f"single={metrics['constant_baseline']:.4f} "
                  f"avg80={metrics.get('constant_baseline_avg80', float('nan')):.4f}\n"
                  f"    collapse: {metrics['collapse_verdict']}",
                  flush=True)
            if not metrics["collapse_ok"]:
                # Not fatal: a partially collapsed representation can still retrieve,
                # and stopping would hide whether it recovers.  But it is the one
                # warning that has to be impossible to miss, because the previous
                # failure was silent for 14 epochs.
                print("[align] *** COLLAPSE WARNING: the invariant representation is "
                      "degenerating; if top1 is not improving, this run is not "
                      "learning ***", flush=True)
            if metrics["top1"] > best:
                best = metrics["top1"]
                torch.save({
                    "encoder": model.state_dict(),
                    "heads": heads.state_dict(),
                    "cfg_enc": asdict(cfg.enc), "cfg_head": asdict(cfg.head),
                    "weights": cfg.weights.to_dict(),
                    "epoch": epoch, "top1": best, "test_subject": test_subject,
                    "norm_source": cfg.norm_source, "train_subjects": train_subjects,
                }, run_dir / "best.pt")
                print(f"[align] saved best (top1={best:.4f})", flush=True)

        row["epoch_minutes"] = (time.time() - t0) / 60
        history.append(row)
        (run_dir / "history.json").write_text(json.dumps(history, indent=1))
        if step >= total_steps:
            break

    print(f"[align] done. best top1={best:.4f} -> {run_dir / 'best.pt'}", flush=True)
    return run_dir / "best.pt"
