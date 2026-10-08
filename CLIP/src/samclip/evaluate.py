"""Retrieval metrics and feature extraction.

This is the *training-side* evaluator: it scores the raw cosine similarity of the
frozen encoder, with no geometric calibration. That is deliberate. It is the number
that tells you whether the encoder is learning at all, and it must not be confused with
the deployable score -- SAW whitening, adaptive CSLS and coordinate recovery live in
``samclip.calibration`` and are applied by ``scripts/run_eval.py``, where they are
reported as their own axis.

`retrieval_report` follows SAMGA's own implementation (cosine similarity on the 200x200
matrix, `rank <= k`), so our numbers are directly comparable to the published table. Do
not "improve" the tie-breaking or normalisation -- a different convention would make the
comparison meaningless.

`extract_features` no longer takes a support set or a subject vector. It used to, so the
evaluation could condition a held-out subject on unlabelled calibration trials. Nothing
conditions any more; a subject-adaptive result is produced by whitening the EEG
embeddings after the fact, which is both simpler and (per the plan doc §7) stronger.

On a v4 model the forward pass itself contains one subject-adaptive step
(``models/smn.py``), so the evaluator's job is to give it the DEPLOYMENT statistic: the
raw embeddings are collected over the whole loader and centred once, not per minibatch.
That is why the extraction is two-pass -- see the function docstring.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def rank_vector(sim: np.ndarray) -> np.ndarray:
    """1-based rank of the diagonal entry of each row."""
    order = np.argsort(-sim, axis=1)
    return np.diag(np.argsort(order, axis=1)) + 1


def retrieval_report(query: np.ndarray, gallery: np.ndarray) -> dict:
    """Top-1 / Top-5 / mean rank for a square (200-way) retrieval task.

    Both sides are L2-normalised before the inner product, matching `retrieve_all`.
    """
    q = query / np.maximum(np.linalg.norm(query, axis=-1, keepdims=True), 1e-8)
    g = gallery / np.maximum(np.linalg.norm(gallery, axis=-1, keepdims=True), 1e-8)
    sim = q @ g.T
    rk = rank_vector(sim)
    n = sim.shape[0]
    return {
        "top1": float((rk <= 1).mean() * 100.0),
        "top5": float((rk <= 5).mean() * 100.0),
        "mean_rank": float(rk.mean()),
        "n": int(n),
    }


@torch.no_grad()
def extract_features(model, loader, device: torch.device, transduce: bool = True) -> dict:
    """Run the model over a loader, returning EEG/image features and concept ids.

    The EEG and image features are extracted from the SAME model in the same pass, so
    the retrieval matrix is always self-consistent -- there is no path here that scores
    one checkpoint's EEG against another's image head.

    TWO-PASS, NOT PER-MINIBATCH. The EEG embedding is collected RAW (pre-SMN,
    pre-normalisation) over the whole loader and the SMN is applied ONCE to the
    concatenated result. This is not an optimisation -- it is the definition of the
    operation. Deployment sees one held-out subject's full query set, so its statistic is
    the mean over all 200 trials; applying the SMN per minibatch instead would make the
    score depend on `eval_batch`, which the v3 audit already identified as a live knob
    (batch 8 vs 24 vs 128 changed Top-1 by more than a point for reasons that had nothing
    to do with the batch's effect on training). One pass also means the result cannot
    differ between `eval_batch=200` and four batches of 50.

    ``subject_ids=None`` is the deployed setting: the batch IS one subject's query set.
    ``eeg_raw`` is returned as well because the D1 offset diagnostic -- the number the
    redesign's first falsifiable prediction is stated in terms of -- has to be read
    BEFORE the SMN removes the thing it measures.
    """
    model.eval()
    eeg_raw, img_feats, concepts = [], [], []
    for batch in loader:
        eeg = batch["eeg"].to(device)
        target = batch["target"].to(device)
        # `subject_ids` is not passed to the router: evaluation is the deployed setting,
        # where the subject identity is unknown to the target branch. For `mean`/`routed`
        # fusion the argument is unused anyway, and for `routed_sr` this is exactly the
        # "drop the residual" inference path the router documents.
        eeg_raw.append(model.embed_eeg(eeg).cpu())
        img_feats.append(model.encode_target(target, training=False).cpu())
        if "concept" in batch:
            concepts.extend(batch["concept"].tolist())
    raw = torch.cat(eeg_raw)
    z_e = model.apply_smn(raw.to(device)) if transduce else raw.to(device)
    out = {
        # Both sides are normalised here rather than inside `retrieval_report` alone, so
        # that anything reading `feats["eeg"]` (the calibration ladder, the probes) gets
        # the same object the metric uses.
        "eeg": F.normalize(z_e, dim=-1).cpu().numpy(),
        "img": torch.cat(img_feats).numpy(),
        "eeg_raw": raw.numpy(),
    }
    if concepts:
        out["concept"] = np.asarray(concepts)
    return out


@torch.no_grad()
def extract_route_features(model, loader, device: torch.device) -> dict[str, dict]:
    """Per-route features for a multi-route (`objective: v6`) model.

    ONE trunk pass is what makes this worth its own function: `extract_features` calls
    `model.embed_eeg` once per batch, and doing that N times -- one call per route -- is
    what the shared trunk exists to prevent. The route-agnostic parts (the two-pass SMN,
    the deployment statistic over the whole loader) are reproduced exactly as in
    `extract_features`, because a per-route statistic that was estimated differently from
    the single-route path would make "route r alone" unreproducible as a standalone run.

    Returns ``{route_name: {"eeg", "img", "eeg_raw"}}``.
    """
    model.eval()
    names = list(getattr(model, "route_names", []))
    if not names:
        raise ValueError("extract_route_features needs a MultiRouteSAMCLIP; the model "
                         "exposes no `route_names`")
    raws: dict[str, list[torch.Tensor]] = {n: [] for n in names}
    imgs: dict[str, list[torch.Tensor]] = {n: [] for n in names}
    concepts: list[int] = []
    primary = str(getattr(model, "primary", names[0]))
    for batch in loader:
        eeg = batch["eeg"].to(device)
        h = model.trunk(eeg)                      # the ONE shared pass
        for n in names:
            head = model.routes[n]
            key = f"target__{n}"
            # The primary route's stack travels under the plain `target` key; see the
            # matching fallback in `Trainer._assemble_v6`.
            t = batch.get(key)
            if t is None and n == primary:
                t = batch.get("target")
            if t is None:
                raise KeyError(f"multi-route eval needs a target stack for route {n!r} "
                               f"(key {key!r}, or 'target' for the primary); got "
                               f"{sorted(batch)}")
            raws[n].append(head.embed_from_h(h).cpu())
            imgs[n].append(head.encode_target(t.to(device), training=False).cpu())
        if "concept" in batch:
            concepts.extend(batch["concept"].tolist())

    out: dict[str, dict] = {}
    for n in names:
        raw = torch.cat(raws[n])
        z = model.routes[n].apply_smn(raw.to(device))
        out[n] = {
            "eeg": F.normalize(z, dim=-1).cpu().numpy(),
            "img": torch.cat(imgs[n]).numpy(),
            "eeg_raw": raw.numpy(),
        }
    if concepts:
        for n in names:
            out[n]["concept"] = np.asarray(concepts)
    return out


@torch.no_grad()
def embed_reps(model, reps: np.ndarray, device: torch.device, batch: int = 8) -> np.ndarray:
    """Embed un-averaged repetition trials ``(C, R, Ch, T)`` -> ``(C, R, d)``.

    Used only by the T2 test-time refinement. The repetitions must be embedded with the
    SAME deployment statistic as the averaged query, so this collects the raw embeddings
    of every repetition and applies the SMN once over the whole ``(C*R, d)`` cloud --
    applying it per repetition would give each repetition its own centring, which is a
    different (and much weaker) operator than the one deployment applies.

    Repetitions are a shared-trunk pass too, so on a v6 model this returns a dict keyed
    by ROUTE when the model is multi-route; the caller fuses the routes' refined scores
    exactly as it fuses their base scores.
    """
    model.eval()
    flat = np.asarray(reps).reshape(-1, *np.asarray(reps).shape[2:])
    names = list(getattr(model, "route_names", []))
    accum: dict[str, list[torch.Tensor]] = {n: [] for n in names} if names else {"": []}
    with torch.no_grad():
        for i in range(0, flat.shape[0], batch):
            x = torch.as_tensor(flat[i:i + batch], dtype=torch.float32, device=device)
            if names:
                h = model.trunk(x)
                for n in names:
                    accum[n].append(model.routes[n].embed_from_h(h).cpu())
            else:
                accum[""].append(model.embed_eeg(x).cpu())
    out: dict[str, np.ndarray] = {}
    for n, chunks in accum.items():
        raw = torch.cat(chunks).to(device)
        z = model.apply_smn(raw, None) if not names else model.routes[n].apply_smn(raw, None)
        z = F.normalize(z, dim=-1).cpu().numpy()
        out[n] = z.reshape(*np.asarray(reps).shape[:2], -1)
    return out if names else out[""]
