"""v6 multi-route model: ONE shared EEG trunk, N frozen visual geometries.

WHY A WRAPPER RATHER THAN A NEW ARCHITECTURE
--------------------------------------------
`docs/eeg2image_v6_architecture.md` §2.2 asks for three routes -- alpha semantic
(InternViT multi-layer), beta low-level (pixel/VAE latent), gamma self-supervised
(DINOv2) -- each with its own EEG projection, its own image projection and its own shared
head, fused at the SCORE level. Every one of those pieces already exists in `SAMCLIP`
with `arch: v4`, so this class does not reimplement any of it:

    shared EEG trunk  ->  [ route.embed_from_h ] x N   (one trunk pass)
    each route's frozen target -> [ route.encode_target ]
    per-route cosine/CSLS scores -> summed (score fusion)

What is genuinely new is only the SHARING of the trunk: N independent `SAMCLIP` heads are
constructed (one per route, sized to that route's target dimension) and each one's
`trunk` attribute is REPLACED by a single shared module. The orphaned trunks are dropped
from `_modules`, so `state_dict()` and `parameters()` see exactly one trunk --

    `parameters()` still yields the trunk N times (it appears in N submodules), which is
    why `dedup_parameters` exists. Handing the optimiser a duplicated parameter means N
    identical updates per step and effectively N x the learning rate on the encoder, a
    failure that trains happily and just looks like a badly tuned LR.

WHY SCORE FUSION AND NOT EMBEDDING CONCATENATION
------------------------------------------------
Axiom A5 of the plan: merging embeddings before the contrast imposes ONE similarity
geometry and discards exactly the disagreement between the routes that is the reason for
having more than one. So the routes are fused after their scores exist, as
``S = sum_r S_r`` over cosine (or CSLS) score matrices -- CORTIVA's late fusion, its
sum-rule bolted onto our retrieval metric rather than its image-generation head.

WEIGHTS ARE UNIFORM BY DEFAULT, and that is a measured choice, not laziness: CORTIVA's
four weight controls all landed within interval-crossing-zero of each other with UNIFORM
(74.22) on top, and the Kittler sum rule is first-order insensitive to the weights' spread.
Fitting weights on the held-out fold would also be a protocol violation.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .samclip import ARCHITECTURES, SAMCLIP, _head


def dedup_parameters(module: nn.Module) -> list[nn.Parameter]:
    """Every `requires_grad` parameter exactly once, in first-seen order.

    Needed because a shared submodule is reachable from several parents, so
    `module.parameters()` yields it once per parent. An optimiser given duplicates applies
    the update N times, i.e. an effective LR of N x on exactly the parameters that are
    shared -- here, the whole EEG encoder. That is not a crash; it is a silent
    hyper-parameter change, which is worse.
    """
    seen: set[int] = set()
    out: list[nn.Parameter] = []
    for p in module.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            out.append(p)
    return out


class MultiRouteSAMCLIP(nn.Module):
    """`N` route heads over one shared EEG trunk. See the module docstring."""

    def __init__(self, cfg: dict, routes: list[dict], ctor_cfg: dict) -> None:
        super().__init__()
        if not routes:
            raise ValueError("MultiRouteSAMCLIP needs at least one route")
        self.route_names = [str(r["name"]) for r in routes]
        if len(set(self.route_names)) != len(self.route_names):
            raise ValueError(f"route names must be unique, got {self.route_names}")
        self.arch = "v6"
        #: Route 0 is the "monitor" route: the in-training evaluation, the offset
        #: diagnostic and `smn_gate()` all read it, so `train_stage_a`'s existing
        #: `evaluate_fold` works unchanged and logs a real number every epoch. That is a
        #: deliberate choice -- monitoring the sum of routes would hide a route that has
        #: collapsed, which is the first failure a multi-route model shows.
        self.primary = self.route_names[0]

        # ONE trunk, built once and then installed into every route. `ctor_cfg` carries the
        # trunk geometry; the per-route construction below discards the trunks it builds.
        base = _build_route(cfg, routes[0], ctor_cfg)
        self.trunk = base.trunk
        self.routes = nn.ModuleDict()
        for i, spec in enumerate(routes):
            head = base if i == 0 else _build_route(cfg, spec, ctor_cfg)
            head.trunk = self.trunk          # replace; the orphan leaves `_modules`
            self.routes[head.route_tag] = head
        self.d_align = int(getattr(base, "d_align", 64))
        self.d_latent = int(getattr(base, "d_latent", 64))
        self.share_head_hidden = getattr(base, "share_head_hidden", None)
        #: Fusion weights: uniform unless a config says otherwise. Held as a buffer so it
        #: travels with the checkpoint and cannot silently differ between train and eval.
        self.register_buffer("fusion_weights",
                             torch.full((len(routes),), 1.0 / len(routes)))

    # ------------------------------------------------------------- trivially shared
    @property
    def route_list(self) -> list[SAMCLIP]:
        return [self.routes[n] for n in self.route_names]

    @property
    def smn(self):
        """The PRIMARY route's SMN module, so the run record tells the truth.

        `train_stage_a` logs `smn=off` when `getattr(model, "smn", None) is None`, and a
        bare wrapper would report `off` for a model whose every route has one -- a record
        that understates its own mechanism, which is the class of defect this project
        keeps paying for. The routes' SMNs are independent modules (one per route, since
        each has its own pre-SMN space), so there is no single object to point at; the
        primary's is the one the in-training evaluator also reads.

        A class-level property cannot collide with `nn.Module`'s attribute handling as
        long as nothing assigns `self.smn` on the wrapper, and nothing does.
        """
        return self.routes[self.primary].smn

    def smn_gates(self) -> dict[str, float]:
        """Per-route SMN gate values, for a record that shows a collapsed route."""
        return {n: float(self.routes[n].smn.gate()) for n in self.route_names
                if self.routes[n].smn is not None}

    # ------------------------------------------------- primary-route delegation
    # These four make the multi-route model look like a single-route SAMCLIP to the code
    # that does not need to know about routes (the in-training evaluator, the run record,
    # the smoke test). They all read route 0, documented above.
    def embed_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.routes[self.primary].embed_from_h(self.trunk(eeg))

    def apply_smn(self, z: torch.Tensor, subject_ids: torch.Tensor | None = None):
        return self.routes[self.primary].apply_smn(z, subject_ids)

    def encode_eeg(self, eeg, normalize: bool = True, subject_ids=None):
        return self.routes[self.primary].encode_eeg(eeg, normalize, subject_ids)

    def encode_target(self, target, subject_ids=None, training=False,
                      normalize=True, require_subject_ids=True):
        return self.routes[self.primary].encode_target(
            target, subject_ids, training, normalize, require_subject_ids)

    def target_layer_weights(self):
        return self.routes[self.primary].target_layer_weights()

    def router_entropy(self):
        return self.routes[self.primary].router_entropy()

    def smn_gate(self):
        return self.routes[self.primary].smn_gate()

    def subject_offset_ratio(self, z, subject_ids=None):
        return self.routes[self.primary].subject_offset_ratio(z, subject_ids)

    # -------------------------------------------------------------------- forward
    def forward(self, eeg: torch.Tensor, targets: dict[str, torch.Tensor],
                subject_ids: torch.Tensor | None = None,
                training: bool = True) -> dict[str, Any]:
        """``targets`` maps route name -> that route's ``(B, K_r, D_r)`` target stack."""
        missing = [n for n in self.route_names if n not in targets]
        if missing:
            raise KeyError(
                f"no target supplied for route(s) {missing}. A missing route would be "
                f"silently skipped from the loss while the config still claimed it, so "
                f"this raises. Got {sorted(targets)}.")
        h = self.trunk(eeg)                       # ONE trunk pass for every route
        out: dict[str, Any] = {"routes": {}}
        for name in self.route_names:
            head = self.routes[name]
            raw = head.embed_from_h(h)
            z_e = F.normalize(head.apply_smn(raw, subject_ids), dim=-1)
            z_i = head.encode_target(targets[name], subject_ids, training)
            out["routes"][name] = {"z_eeg": z_e, "z_img": z_i, "z_eeg_raw": raw}
        # Legacy keys, pointing at the primary route: a reader that only knows the
        # single-route contract still gets a valid step.
        out["z_eeg"] = out["routes"][self.primary]["z_eeg"]
        out["z_img"] = out["routes"][self.primary]["z_img"]
        out["z_eeg_raw"] = out["routes"][self.primary]["z_eeg_raw"]
        return out


def _build_route(cfg: dict, spec: dict, ctor_cfg: dict) -> SAMCLIP:
    """One route's `SAMCLIP` head, sized to that route's target geometry."""
    head = SAMCLIP(
        n_channels=ctor_cfg.get("n_channels", 63),
        n_timepoints=ctor_cfg.get("n_timepoints", 250),
        n_target_layers=int(spec["n_layers"]),
        image_dim=int(spec["image_dim"]),
        n_subjects=ctor_cfg.get("n_subjects", 9),
        d_model=cfg.get("d_model", 200),
        d_embed=cfg.get("d_embed", 512),
        n_heads=cfg.get("n_heads", 4),
        n_blocks=cfg.get("n_blocks", 2),
        dim_ff=cfg.get("dim_ff", 512),
        dropout=cfg.get("dropout", 0.1),
        head_dropout=cfg.get("head_dropout", 0.1),
        target_fusion=cfg.get("target_fusion", "mean"),
        target_subject_dropout=cfg.get("target_subject_dropout", 0.3),
        router_temperature=cfg.get("router_temperature", 1.0),
        router_layer_dropout=cfg.get("router_layer_dropout", 0.0),
        img_head_mode=cfg.get("img_head", "mlp"),
        eeg_head_mode=cfg.get("eeg_head", "mlp"),
        agg_width=cfg.get("agg_width", 16),
        agg_pool=cfg.get("agg_pool", 8),
        front_end=cfg.get("front_end", "linear"),
        front_pool=cfg.get("front_pool", 5),
        arch="v4",                       # every v6 route is a shared-head v4 route
        d_latent=cfg.get("d_latent"),
        d_align=cfg.get("d_align"),
        share_head_hidden=cfg.get("share_head_hidden"),
        smn_enabled=(cfg.get("smn", {}) or {}).get("enabled", True),
        smn_gate_scale=(cfg.get("smn", {}) or {}).get("gate_scale", True),
        smn_init_gate=(cfg.get("smn", {}) or {}).get("init_gate", 0.0),
        smn_min_rows=(cfg.get("smn", {}) or {}).get("min_rows", 4),
    )
    head.route_tag = str(spec["name"])            # type: ignore[attr-defined]
    return head


def resolve_routes(cfg: dict) -> list[dict]:
    """Config -> concrete route specs, checking the caches the routes need exist.

    A route whose feature cache is missing must fail HERE, with the cache path, rather than
    inside `load_target_stack` after the fold's EEG is already resident. The error also
    names the builder, because "which script makes this file" is the only question a
    missing cache raises.
    """
    from .. import config as _cfg
    from ..data.targets import target_dir

    raw = cfg.get("routes")
    if raw in (None, "default"):
        raw = _cfg.DEFAULT_ROUTES
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"`routes` must be a non-empty list or 'default', got {raw!r}")
    out: list[dict] = []
    for r in raw:
        fs = str(r["feature_set"])
        if fs not in _cfg.IMAGE_FEATURE_SETS:
            raise KeyError(f"route {r.get('name')!r} names unknown feature_set {fs!r}; "
                           f"known: {sorted(_cfg.IMAGE_FEATURE_SETS)}")
        spec = dict(r)
        spec["layers"] = list(r.get("layers") or _cfg.IMAGE_FEATURE_SETS[fs]["layers"])
        spec["image_dim"] = int(_cfg.IMAGE_FEATURE_SETS[fs]["dim"])
        spec["n_layers"] = len(spec["layers"])
        d = target_dir(fs)
        probe = d / _cfg.IMAGE_FEATURE_SETS[fs]["pattern"].format(
            split="test", layer=spec["layers"][0])
        if not probe.is_file():
            raise FileNotFoundError(
                f"route {spec.get('name')!r}: missing feature cache {probe}. Build it "
                f"with `python scripts/build_route_cache.py --only {fs}`, or drop the "
                f"route from `routes:`.")
        out.append(spec)
    return out
