#!/usr/bin/env python3
"""Layer-wise condition injection decode -- the NeuroWeave "layered injection" test.

THE QUESTION THIS ANSWERS
-------------------------
Every condition we currently use (the semantic CLIP embedding, the depth CLIP
embedding, the edge CLIP embedding) is injected through IP-Adapter at EVERY UNet
attention layer, i.e. all three modalities share one conditioning channel at
every depth.  The measured consequence is a hard Pareto front between pixel
fidelity and semantic identifiability:

    arm            SSIM      cycle_disc_top1 (CLIP space)
    atm_aligned    0.2300    0.565    semantic-only conditioning
    mb_p1_i1       0.3500    0.130    1 IP branch
    single         0.3696    0.080    3 IP branches + depth CN + LL init

Buying SSIM with structural conditioning costs semantic identifiability almost
one-for-one.  NeuroWeave's last untested claim is that this is a PLACEMENT
problem, not a CAPACITY problem: inject the semantic condition where semantic
content is written (late / up blocks), and the structural condition where layout
is written (early / down blocks), so they stop competing for one channel.

The arm definitions, the early/late split, the mass arithmetic and the
pre-registered verdicts all live in `layered_arms.py`; this file only renders.

WHY THE SCALES GO THROUGH `set_ip_adapter_scale` AND THEN GET READ BACK
----------------------------------------------------------------------
The documented dict form is
`pipe.set_ip_adapter_scale([{...per-adapter dict...}, ...])`, and it does work in
diffusers 0.31: the dict is expanded by `_maybe_expand_lora_scales`, whose keys
(`down.block_1.0`) are translated by `_translate_into_actual_layer_name` into real
layer names (`down_blocks.1.attentions.0`) before being prefix-matched against
`attn_processors`.

What it does NOT do is tell you whether the assignment landed.  A fully wrong
spec can leave every scale at zero and produce a CLEAN-LOOKING null result -- the
most dangerous possible outcome for this experiment, because it would be read as
evidence against layered injection when in fact nothing was injected at all.

So every spec is verified against `attn_processor.scale` read back off the live
UNet (`readback_level_map` -> `verify_assignment`), and a mismatch is fatal.
`--layer-sanity N` adds a second, empirical guard: it renders N rows under this
arm's spec and under `--sanity-ref-spec` on ONE loaded pipeline and reports the
mean absolute pixel difference.  A near-zero diff means the assignment changed
nothing and the arm may not be reported.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

BRAINIT = Path("/project/peilab/why/eeg-brainit")
sys.path.insert(0, str(BRAINIT / "scripts"))
sys.path.insert(0, str(BRAINIT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_atm_pipeline import resolve_ip_adapter_dir, resolve_sdxl_model_path  # type: ignore

from layered_arms import (  # noqa: E402
    LEVEL_ORDER, SIDES, level_key, level_side, parse_spec, plan_counts,
    read_unet_layout, scale_dict, verify_assignment,
)


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


# ------------------------------------------------------------------- pipelines
def resolve_turbo_path(hub: Path) -> str | None:
    root = hub / "models--stabilityai--sdxl-turbo" / "snapshots"
    if root.is_dir():
        for snap in sorted(root.iterdir(), reverse=True):
            if (snap / "model_index.json").is_file():
                return str(snap)
    return None


def resolve_controlnet_path(hub: Path) -> str:
    root = hub / "models--diffusers--controlnet-depth-sdxl-1.0" / "snapshots"
    if root.is_dir():
        for snap in sorted(root.iterdir(), reverse=True):
            if (snap / "config.json").is_file():
                return str(snap)
    return "diffusers/controlnet-depth-sdxl-1.0"


def load_pipeline(kind: str, cache: Path, n_branches: int, want_cn: bool, device, dtype,
                  want_init: bool | None = None):
    """Pipeline with `n_branches` registered IP-Adapters. Scales are NOT set here.

    `want_cn` (ControlNet-depth) and `want_init` (img2img init) are INDEPENDENT.
    They used to be welded together and turbo forced both off, which made two
    configurations inexpressible: init-without-CN (the band-decoupled anchor) and
    turbo-with-CN.  Both matter now:

      * CogCapPro generates with **no** pixel constraint at all (pure IP-Adapter).
        Reproducing its mechanism therefore needs want_cn=False AND want_init=False.
      * The band-decoupled anchor keeps the init but drops ControlNet, so it needs
        want_cn=False, want_init=True.

    Passing want_init=None keeps the legacy behaviour (init iff CN).
    """
    want_init = want_cn if want_init is None else want_init
    if kind == "turbo":
        sdxl = resolve_turbo_path(cache) or "stabilityai/sdxl-turbo"
        print(f"[INFO] sdxl-turbo = {sdxl}")
    elif kind == "base":
        sdxl = resolve_sdxl_model_path(cache)
        print(f"[INFO] sdxl = {sdxl}")
    else:
        raise SystemExit(f"[FATAL] unknown --pipeline {kind}")

    variant = "fp16" if dtype == torch.float16 else None
    local = Path(str(sdxl)).is_dir()
    if want_cn:
        from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
        cn_path = resolve_controlnet_path(cache)
        print(f"[INFO] controlnet = {cn_path}")
        controlnet = ControlNetModel.from_pretrained(
            cn_path, torch_dtype=dtype, local_files_only=Path(str(cn_path)).is_dir())
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            sdxl, controlnet=controlnet, torch_dtype=dtype, variant=variant,
            use_safetensors=True, local_files_only=local).to(device)
        has_cn = True
    elif want_init:
        from diffusers import StableDiffusionXLImg2ImgPipeline
        print("[INFO] init-only img2img (no ControlNet)")
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            sdxl, torch_dtype=dtype, variant=variant,
            use_safetensors=True, local_files_only=local).to(device)
        has_cn = False
    else:
        from diffusers import StableDiffusionXLPipeline
        print("[INFO] pure text2img (no ControlNet, no init)")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            sdxl, torch_dtype=dtype, variant=variant,
            use_safetensors=True, local_files_only=local).to(device)
        has_cn = False

    ip_root = resolve_ip_adapter_dir(cache)
    kwargs = {"subfolder": "sdxl_models", "image_encoder_folder": None, "local_files_only": True}
    loaded = None
    for weight_name in ("ip-adapter_sdxl_vit-h.bin", "ip-adapter_sdxl.bin"):
        try:
            pipe.load_ip_adapter(str(ip_root), weight_name=[weight_name] * n_branches, **kwargs)
            loaded = weight_name
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] {weight_name} x{n_branches} failed: {exc}")
    if loaded is None:
        raise SystemExit("[FATAL] could not load IP-Adapter weights")
    n_layers = len(pipe.unet.encoder_hid_proj.image_projection_layers)
    print(f"[INFO] ip-adapter = {loaded} x{n_branches} (registered layers = {n_layers})")
    if n_layers != n_branches:
        raise SystemExit(f"[FATAL] registered {n_layers} adapters != {n_branches} branches")
    return pipe, has_cn


def readback_level_map(unet) -> dict[str, list[float]]:
    """The scales ACTUALLY stored on the live UNet, one entry per attention level."""
    from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
    out: dict[str, list[float]] = {}
    for name, proc in unet.attn_processors.items():
        if not isinstance(proc, IPAdapterAttnProcessor2_0):
            continue
        out.setdefault(level_key(name), [float(s) for s in proc.scale])
    return out


def apply_spec(pipe, spec, n_levels) -> dict:
    """Set the per-branch, per-level scales and prove they landed."""
    pipe.set_ip_adapter_scale([scale_dict(m, s) for m, s in spec])
    return verify_assignment(readback_level_map(pipe.unet), spec, n_levels)


def load_raw_scales(path: str) -> list:
    """Read a list of diffusers-native IP-Adapter scale configs, one per branch.

    WHY THIS EXISTS -- the {down,mid,up}x single-scalar form above CANNOT express the
    published CogCapPro layout.  Its generator uses (verbatim, from diffusers' own
    `set_ip_adapter_scale` docstring -- CogCapPro copied the example):

        image = {"down": {"block_2": [1.0, 1.0]}, "up": {"block_0": [1.0, 1.0, 1.0]}}
        depth = {"down": {"block_2": [0.0, 0.5]}, "up": {"block_0": [0.0, 0.0, 0.0]}}
        edge  = {"down": {"block_2": [0.0, 0.5]}, "up": {"block_0": [0.0, 0.0, 0.0]}}

    i.e. semantic travels on down_blocks.2 + up_blocks.0 at full strength, while
    structure is allowed in exactly ONE level (down_blocks.2.attentions.1) at 0.5 and
    is switched off everywhere else.  `scale_dict("early", 0.5)` would instead touch
    all four down levels, so the mechanism is not reproducible with our modes.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise SystemExit(f"[FATAL] {path}: expected a non-empty JSON list of scale configs")
    _validate_raw_scales(data, path)
    return data


def _validate_raw_scales(scales: list, path: str = "") -> None:
    """Reject shapes diffusers accepts structurally but then interprets wrongly.

    `_maybe_expand_lora_scales` expands `down`/`up` deeply but only special-cases `mid`
    by `isinstance(scales["mid"], list)`.  A dict under `mid` is therefore passed through
    UNEXPANDED, ends up stored verbatim on the attention processor, and only fails much
    later -- in the readback, as `float() argument must be ... not 'dict'`, after the
    pipeline has already been loaded.  Catching it here keeps the error local and
    explains the cause.  (This is a real failure mode: it cost the A3 arm of the first
    NW5 run, and the offline layout checker had missed it because it re-implemented the
    expansion instead of consulting diffusers' rules.)
    """
    where = f" ({path})" if path else ""
    for i, cfg in enumerate(scales):
        if not isinstance(cfg, dict):
            continue
        mid = cfg.get("mid")
        if isinstance(mid, dict):
            raise SystemExit(
                f"[FATAL] ip-scale config {i}{where}: 'mid' must be a scalar or a "
                f"1-element list, not a dict -- diffusers does not expand 'mid', so "
                f"{mid!r} would be stored unexpanded. Use e.g. \"mid\": 1.0")
        for side in ("down", "up"):
            blocks = cfg.get(side)
            if blocks is not None and not isinstance(blocks, dict):
                raise SystemExit(
                    f"[FATAL] ip-scale config {i}{where}: '{side}' must be a dict of "
                    f"block_N -> [per-layer scales], got {blocks!r}")


def apply_raw_scales(pipe, scales: list) -> dict:
    """Set scales verbatim, then read back the live UNet to prove where they landed.

    This readback is the only real evidence that the nested dict survived diffusers'
    `_maybe_expand_lora_scales` and the `attn_name.startswith(k)` matching.  A silent
    no-op here would leave every scale at its pre-existing value and make all the arms
    uninterpretable, so the level map is printed rather than assumed.
    """
    pipe.set_ip_adapter_scale(scales)
    lvl = readback_level_map(pipe.unet)
    active = {k: [round(float(x), 4) for x in v] for k, v in sorted(lvl.items())}
    print("[raw-scales] per-level map on the live UNet:")
    for k in sorted(active):
        print(f"    {k:<28} {active[k]}")
    n_branch = max((len(v) for v in active.values()), default=0)
    per_branch = [max((v[i] if i < len(v) else 0.0) for v in active.values())
                  for i in range(n_branch)]
    if not any(s > 0 for s in per_branch):
        raise SystemExit("[FATAL] every scale is zero - nothing would be injected")
    print(f"[raw-scales] per-branch max scale: {per_branch}")
    return {"mode": "raw-json", "level_map": active, "per_branch_max": per_branch,
            "n_active_levels": sum(1 for v in active.values() if max(v) > 0)}



def build_ip_embeds(branches: list[np.ndarray], i: int, device, dtype,
                    do_cfg: bool) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for arr in branches:
        v = torch.from_numpy(arr[i:i + 1]).to(device=device, dtype=dtype).unsqueeze(0)
        if do_cfg:
            v = torch.cat([torch.zeros_like(v), v], dim=0)
        out.append(v)
    return out


def _render(pipe, args, branches, i, prompt, device):
    do_cfg = float(args.gen_guidance) > 1.0
    g = torch.Generator(device=device).manual_seed(args.seed)
    return np.asarray(pipe(
        prompt=prompt, negative_prompt="",
        ip_adapter_image_embeds=build_ip_embeds(branches, i, device, pipe.dtype, do_cfg),
        num_inference_steps=int(args.gen_steps),
        guidance_scale=float(args.gen_guidance),
        generator=g, height=args.gen_size, width=args.gen_size,
    ).images[0].convert("RGB"), dtype=np.float32)


def run_layer_sanity(args, spec_a, spec_b, branches, prompts, cache, device, dtype,
                     n_levels) -> dict:
    """Render N rows under two specs on ONE pipeline; a no-op spec shows up as ~0 diff."""
    n = min(int(args.layer_sanity), len(branches[0]))
    pipe, _ = load_pipeline(args.pipeline, cache, len(branches), False, device, dtype)
    ver_a = apply_spec(pipe, spec_a, n_levels)
    diffs: list[float] = []
    ver_b = None
    for i in range(n):
        a = _render(pipe, args, branches, i, prompts[i] if prompts[i] else "", device)
        ver_b = apply_spec(pipe, spec_b, n_levels)
        b = _render(pipe, args, branches, i, prompts[i] if prompts[i] else "", device)
        diffs.append(float(np.abs(a - b).mean()))
        apply_spec(pipe, spec_a, n_levels)
    mean_diff = float(np.mean(diffs))
    del pipe
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "rows_checked": n,
        "spec_a_mass": ver_a["total_mass"], "spec_b_mass": ver_b["total_mass"],
        "pixel_diff_mean": mean_diff, "pixel_diff_per_row": diffs,
        "verdict": ("LAYER_SPEC_ACTIVE" if mean_diff > 0.5 else
                    "SUSPECT_NOOP: the two specs render nearly identical images"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond-npys", type=str, required=True,
                    help="comma-separated per-branch condition npy, in order "
                         "semantic,depth,edge (row order must match)")
    ap.add_argument("--branch-spec", type=str, default="",
                    help="comma-separated MODE:SCALE per branch, MODE in {all,early,late,none}; "
                         "e.g. 'late:1.0,early:1.0,early:1.0'. Not required when "
                         "--ip-scale-json supplies native per-block dicts.")
    ap.add_argument("--ip-scale-json", type=str, default="",
                    help="path to a JSON list of diffusers-native IP-Adapter scale configs "
                         "(one per branch, each a float or {'down'/'mid'/'up': {'block_N': "
                         "[per-layer scales]}}). Overrides --branch-spec and reproduces the "
                         "published CogCapPro layout exactly. See load_raw_scales().")
    ap.add_argument("--prompts-json", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--tag", type=str, default="layered")
    ap.add_argument("--pipeline", type=str, default="base", choices=["base", "turbo"])
    ap.add_argument("--depth-rgb-dir", type=str, default="")
    ap.add_argument("--lowlevel-rgb-dir", type=str, default="")
    ap.add_argument("--cn-scale", type=float, default=0.40)
    ap.add_argument("--strength", type=float, default=0.82)
    ap.add_argument("--use-cn", type=int, default=-1,
                    help="0/1: force ControlNet-depth on/off instead of inferring it from "
                         "--depth-rgb-dir. -1 = legacy auto.")
    ap.add_argument("--use-init", type=int, default=-1,
                    help="0/1: force the img2img init on/off instead of tying it to ControlNet. "
                         "-1 = legacy auto. 0/1 combination with --use-cn 0 gives init-only, "
                         "which is what the band-decoupled anchor needs.")
    ap.add_argument("--init-blur-sigma", type=float, default=0.0,
                    help="gaussian sigma applied to the init image before img2img. The "
                         "low-frequency anchor: PixCorr/SSIM are dominated by the low band, "
                         "so blurring the init releases high-frequency freedom while keeping "
                         "the layout constraint. 0 = off (plain init).")

    ap.add_argument("--gen-steps", type=int, default=28)
    ap.add_argument("--gen-guidance", type=float, default=5.0)
    ap.add_argument("--gen-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-cpu", type=int, default=0,
                    help="opt out of the GPU fail-fast guard. Diffusion on CPU is "
                         "orders of magnitude slower; only for debugging.")
    ap.add_argument("--device", type=str, default="cuda:0",
                    help="preferred device; falls back to cpu if cuda is unavailable")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--negative-prompt", type=str, default="blurry, low quality, distorted, watermark")
    ap.add_argument("--plan-only", action="store_true",
                    help="resolve the spec against the UNet config on disk, write the plan, "
                         "load no weights, generate nothing")
    ap.add_argument("--layer-sanity", type=int, default=0,
                    help="if >0, render this many rows under two specs and report pixel diff")
    ap.add_argument("--sanity-ref-spec", type=str, default="all:1.0,all:1.0,all:1.0",
                    help="the spec to compare against in --layer-sanity")
    ap.add_argument("--sanity-out", type=str, default="")
    ap.add_argument("--layer-report", type=str, default="",
                    help="write the verified layer assignment JSON here")
    args = ap.parse_args()

    cache = Path(os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
    os.environ.setdefault("HF_HOME", str(cache.parent))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache.parent / "open_clip"))
    os.environ.setdefault("TORCH_HOME", str(cache.parent / "torch"))

    paths = [p for p in args.cond_npys.split(",") if p.strip()]
    raw = load_raw_scales(args.ip_scale_json) if args.ip_scale_json else None
    if raw is not None and len(raw) != len(paths):
        raise SystemExit(f"[FATAL] {args.ip_scale_json} has {len(raw)} configs but "
                         f"--cond-npys lists {len(paths)} branches")
    spec = None
    if raw is None:
        try:
            spec = parse_spec(args.branch_spec, len(paths))
        except ValueError as exc:
            raise SystemExit(f"[FATAL] {exc}")

    # ---- the plan, from the real config, before any weight is touched --------
    if args.pipeline == "base":
        layout = read_unet_layout(resolve_sdxl_model_path(cache))
    else:
        n_early = sum(1 for n in LEVEL_ORDER if n.startswith("down_blocks"))
        layout = {"layers_per_block": 2, "down_blocks_with_attn": [1, 2],
                  "up_blocks_with_attn": [0, 1],
                  "n_levels": {"early": n_early, "late": len(LEVEL_ORDER) - n_early}}
    n_levels = layout["n_levels"]
    if spec is not None:
        plan = plan_counts(spec, n_levels)
        plan["spec"] = [{"branch": i, "mode": m, "scale": s} for i, (m, s) in enumerate(spec)]
        print(f"[plan] tag={args.tag} spec={args.branch_spec}")
    else:
        plan = {"mode": "raw-json", "source": args.ip_scale_json, "scales": raw}
        print(f"[plan] tag={args.tag} raw-scale-json={args.ip_scale_json}")
    print(f"[plan] levels: early={n_levels['early']} late={n_levels['late']} "
          f"total={sum(n_levels.values())}")
    print(f"[plan] {json.dumps(plan, indent=2)}")

    if args.plan_only:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "plan.json").write_text(
            json.dumps({"tag": args.tag, "layout": layout, "plan": plan}, indent=2),
            encoding="utf-8")
        print(f"[plan] wrote {out / 'plan.json'}")
        return

    from dev_guard import pick_device  # noqa: E402
    device = pick_device(args.device, allow_cpu=bool(args.allow_cpu))
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    branches = [l2(np.load(p).astype(np.float32)) for p in paths]
    n_rows = len(branches[0])
    for p, b in zip(paths, branches):
        if len(b) != n_rows:
            raise SystemExit(f"[FATAL] {p} rows {len(b)} != {n_rows}")

    prompts = json.loads(Path(args.prompts_json).read_text(encoding="utf-8"))
    if len(prompts) < n_rows:
        raise SystemExit(f"[FATAL] prompts {len(prompts)} < rows {n_rows}")

    want_cn = bool(args.depth_rgb_dir) and bool(args.lowlevel_rgb_dir)
    if args.pipeline == "turbo" and want_cn and args.use_cn < 0 and args.use_init < 0:
        print("[WARN] turbo pipeline ignores ControlNet/init; falling back to text2img "
              "(pass --use-cn/--use-init explicitly to override)")
        want_cn = False
    want_init = want_cn
    if args.use_cn >= 0:
        want_cn = bool(args.use_cn)
    if args.use_init >= 0:
        want_init = bool(args.use_init)
    else:
        want_init = want_cn
    has_pixels = want_cn or want_init
    if want_cn and not args.depth_rgb_dir:
        raise SystemExit("[FATAL] ControlNet requested but --depth-rgb-dir is empty")
    if has_pixels and not args.lowlevel_rgb_dir:
        raise SystemExit("[FATAL] --use-cn/--use-init requested but --lowlevel-rgb-dir is empty")

    out_dir = Path(args.output_dir)
    gen_dir = out_dir / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    n = n_rows if args.max_images <= 0 else min(n_rows, args.max_images)
    do_cfg = float(args.gen_guidance) > 1.0

    print(f"[INFO] pipeline={args.pipeline} branches={len(branches)} "
          f"{'spec=' + args.branch_spec if spec is not None else 'raw-scales=' + args.ip_scale_json} "
          f"rows={n} steps={args.gen_steps} guidance={args.gen_guidance} cfg={do_cfg} "
          f"cn={want_cn} init={want_init} init_blur_sigma={args.init_blur_sigma}")

    verified = None
    if all((gen_dir / f"{i:03d}.png").is_file() for i in range(n)):
        print(f"[INFO] skip gen ({n} images exist)")
    else:
        pipe, has_cn = load_pipeline(args.pipeline, cache, len(branches), want_cn, device, dtype,
                                     want_init=want_init)
        if spec is not None:
            verified = apply_spec(pipe, spec, n_levels)
            print(f"[verify] {json.dumps(verified, indent=2)}")
            if verified["semantic_structure_overlap"]:
                print(f"[WARN] semantic and structure branches share levels: "
                      f"{verified['semantic_structure_overlap']}")
        else:
            verified = apply_raw_scales(pipe, raw)
        if args.layer_report:
            Path(args.layer_report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.layer_report).write_text(json.dumps(
                {"tag": args.tag, "layout": layout, "plan": plan, "verified": verified},
                indent=2), encoding="utf-8")

        g = torch.Generator(device=device).manual_seed(args.seed)
        for i in tqdm(range(n), desc=f"layered[{args.tag}]"):
            path = gen_dir / f"{i:03d}.png"
            if path.is_file():
                continue
            prompt = prompts[i] if prompts[i] else ""
            neg = args.negative_prompt if prompt else ""
            kw: dict = dict(
                prompt=prompt,
                negative_prompt=neg,
                ip_adapter_image_embeds=build_ip_embeds(branches, i, device, pipe.dtype, do_cfg),
                num_inference_steps=int(args.gen_steps),
                guidance_scale=float(args.gen_guidance),
                generator=g,
                height=int(args.gen_size),
                width=int(args.gen_size),
            )
            if has_pixels:
                lp = Path(args.lowlevel_rgb_dir) / f"{i:03d}.png"
                if not lp.is_file():
                    raise FileNotFoundError(lp)
                init = Image.open(lp).convert("RGB").resize(
                    (args.gen_size, args.gen_size), Image.Resampling.BICUBIC)
                if args.init_blur_sigma > 0:
                    from PIL import ImageFilter
                    init = init.filter(ImageFilter.GaussianBlur(float(args.init_blur_sigma)))
                kw["image"] = init
                kw["strength"] = float(args.strength)
                if want_cn:
                    dp = Path(args.depth_rgb_dir) / f"{i:03d}.png"
                    if not dp.is_file():
                        raise FileNotFoundError(dp)
                    kw["control_image"] = Image.open(dp).convert("RGB").resize(
                        (args.gen_size, args.gen_size), Image.Resampling.BICUBIC)
                    kw["controlnet_conditioning_scale"] = float(args.cn_scale)
            pipe(**kw).images[0].save(path)
        del pipe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    sanity = None
    if args.layer_sanity > 0:
        if spec is None:
            print("[WARN] --layer-sanity needs --branch-spec; skipping under raw scales")
        else:
            try:
                spec_b = parse_spec(args.sanity_ref_spec, len(branches))
            except ValueError as exc:
                raise SystemExit(f"[FATAL] --sanity-ref-spec: {exc}")
            sanity = run_layer_sanity(args, spec, spec_b, branches, prompts, cache,
                                      device, dtype, n_levels)
            if args.sanity_out:
                Path(args.sanity_out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.sanity_out).write_text(json.dumps(sanity, indent=2), encoding="utf-8")
            print("[sanity]", json.dumps(sanity, indent=2))

    report = {
        "tag": args.tag,
        "pipeline": f"layer-wise IP ({args.pipeline})",
        "method_note": ("per-branch UNet level assignment via set_ip_adapter_scale dicts, "
                        "verified against attn_processor.scale read back off the live UNet"),
        "cond_npys": paths,
        "branch_spec": args.branch_spec if spec is not None else None,
        "ip_scale_json": args.ip_scale_json or None,
        "layout": layout,
        "plan": plan,
        "verified": verified,
        "prompts_json": args.prompts_json,
        "gen_steps": args.gen_steps,
        "gen_guidance": args.gen_guidance,
        "use_cn": want_cn,
        "use_init": want_init,
        "init_blur_sigma": args.init_blur_sigma,
        "cn_scale": args.cn_scale if want_cn else None,
        "strength": args.strength if has_pixels else None,
        "n_gen": n,
        "layer_sanity": sanity,
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "plan"}, indent=2)[:900])


if __name__ == "__main__":
    main()
