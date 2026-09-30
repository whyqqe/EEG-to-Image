#!/usr/bin/env python3
"""GVM extension -- DEPTH and EDGE targets, the two modalities the four-branch
EEG-to-image literature injects alongside the image embedding.

WHY THESE TWO AND NOT SOMETHING ELSE
------------------------------------
The 2026 four-branch result (arXiv 2605.23996) maps one EEG signal through four
parallel heads to four embedding spaces -- image, text, DEPTH, EDGE -- and reports
that they "play complementary but not identical roles": the image head carries
category-level appearance and global shape, while the depth and edge heads add the
geometry that the single image embedding compresses away.  Their alignment stage
then feeds image/depth/edge to the generator and explicitly reports that the TEXT
branch "contributes indirectly through auxiliary semantic supervision" rather than
being a generation condition.  That is the same division of labour this pipeline
already uses between `s_code` (auxiliary) and the IP-Adapter condition (primary);
this script adds the two geometric branches.

WHY IT RENDERS RATHER THAN LOOKING FOR GROUND TRUTH
---------------------------------------------------
THINGS-EEG2 ships no depth or edge ground truth -- the stimuli are photographs.  So
both maps are DERIVED from the stimulus image, exactly as the reference work
derives them:
  * depth: a frozen monocular estimator (`depth-anything/Depth-Anything-V2-Small-hf`,
    already in the local HF cache, so this runs offline),
  * edges: a fixed Sobel magnitude.
Both are deterministic functions of the image, so a row's depth/edge target
contains nothing beyond that row's own stimulus, and the leak argument is identical
to the one the CLIP tower already relies on.

NO IS-THIS-A-NEW-MODALITY ASSUMPTION.  Rendering a depth map and re-encoding it
with the SAME CLIP image encoder is only useful if the result occupies a different
direction than the RGB embedding; a depth map can come back as a near-copy of the
RGB feature.  The script therefore measures `cos(depth, image)`,
`cos(edge, image)` and `cos(depth, edge)` row-wise and writes them to the report
BEFORE any training run can treat the two new heads as independent evidence.  A
cosine near 1.0 is the signal that a head is a relabelling, and it is reported as
that rather than discovered later as a null result.

ROW ORDER
---------
Identical to `gem_clip_img.py`: `list_split_images()` order, verified row by row
against the `path` column of `captions_<split>.jsonl`.  The check is repeated here
rather than imported because a silent reordering would misalign every per-row array
downstream, and this script is the one writing fresh caches.

DISK DISCIPLINE.  The rendered depth and edge maps are NEVER written to disk -- they
exist only long enough to be encoded.  Only the four feature arrays per split are
kept.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

NB_ROOT = Path(__file__).resolve().parents[2]

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def l2(a: np.ndarray) -> np.ndarray:
    return a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8, None)


def list_split_images(images_root: Path, split: str) -> list[Path]:
    """Identical to `g2_build_targets.list_split_images` -- the project-wide order."""
    root = images_root / ("training_images" if split == "train" else "test_images")
    if not root.is_dir():
        raise SystemExit(f"[FATAL] missing image root {root}")
    paths: list[Path] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        paths.extend(sorted(list(d.glob("*.jpg")) + list(d.glob("*.JPEG"))
                            + list(d.glob("*.png"))))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default=str(NB_ROOT / "outputs/gem/cond_cache"))
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--captions-dir", type=str, default=str(NB_ROOT / "outputs/g2/captions"))
    ap.add_argument("--clip-model", type=str, default="ViT-H-14")
    ap.add_argument("--clip-pretrained", type=str, default="laion2b_s32b_b79k")
    ap.add_argument("--depth-model", type=str, default=DEPTH_MODEL)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--depth-invert", type=int, default=1,
                    help="render NEAR as bright (the standard depth-map convention, and"
                         " the one the SDXL depth ControlNet was trained on).")
    ap.add_argument("--edge-blur", type=int, default=3,
                    help="odd BOX-average window applied before Sobel, in pixels.  A raw"
                         " Sobel on a photograph fires on sensor noise and JPEG blocking,"
                         " so the edge map would encode COMPRESSION artefacts rather than"
                         " object boundaries.  A box average is used rather than a"
                         " Gaussian because it is exactly reproducible with `avg_pool2d`"
                         " and needs no further constant.")
    ap.add_argument("--max-images", type=int, default=0,
                    help="cap rows per split. Non-zero is for SMOKE TESTS only: the"
                         " cache is then a PREFIX of the split.")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OPENCLIP_CACHE_DIR",
                          "/project/peilab/why/cache/eeg-brainit/open_clip")
    # the depth estimator is read from the local hub cache; refusing to reach the
    # network turns a silently different model into a hard error
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import torch
    import torch.nn.functional as F
    import open_clip
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained, device=dev)
    model.eval()

    # open_clip keeps the projection as a bare `nn.Parameter` (`visual.proj`) applied
    # inside `forward` as `pooled @ proj`, so the penultimate 1280-d vector is
    # `ln_post(...)[:, 0]`.  The identity is checked rather than assumed, exactly as
    # `gem_clip_img.py` does: if a future open_clip reorders pooling, this fails
    # instead of writing a cache whose two columns mean different things.
    pen: dict[str, torch.Tensor] = {}

    def hook(_m, _inp, o):
        pen["x"] = o.detach()

    if not hasattr(model.visual, "proj") or model.visual.proj is None:
        raise SystemExit("[FATAL] visual.proj absent; the penultimate activation and the"
                         " projected feature cannot both be recovered")
    h = model.visual.ln_post.register_forward_hook(hook)
    with torch.no_grad():
        _p = model.encode_image(torch.zeros(1, 3, 224, 224, device=dev))
        _q = pen["x"][:, 0] @ model.visual.proj
        _d = float((_p.float() - _q.float()).abs().max())
    if _d > 1e-3:
        raise SystemExit(f"[FATAL] `ln_post[:,0] @ proj` differs from `encode_image` by "
                         f"{_d:.3g}; refusing to write a mislabelled cache")
    print(f"[gem-geo] proj identity verified (max abs diff {_d:.2e})")

    dproc = AutoImageProcessor.from_pretrained(args.depth_model)
    dmodel = AutoModelForDepthEstimation.from_pretrained(args.depth_model).to(dev).eval()
    print(f"[gem-geo] depth model {args.depth_model} loaded offline "
          f"({sum(p.numel() for p in dmodel.parameters()) / 1e6:.1f}M params)")

    # fixed Sobel kernels -- a constant, not a learned operator, so the "edge" target
    # is reproducible from the stimulus alone
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                      device=dev).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2).contiguous()

    def to_pil_u8(x: torch.Tensor) -> Image.Image:
        """(h,w) float -> 8-bit RGB PIL.  Quantising is required because the CLIP
        preprocessing path takes images, and it is also what "render" means."""
        x = x.detach().float()
        lo, hi = x.min(), x.max()
        x = (x - lo) / (hi - lo).clamp_min(1e-6)
        a = (x.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
        # `Image.fromarray` infers mode "L" from the (h,w) uint8 shape; passing
        # `mode=` explicitly is deprecated in current Pillow and warns on load
        return Image.fromarray(a).convert("RGB")

    def render_depth(img: Image.Image) -> tuple[Image.Image, torch.Tensor]:
        inp = dproc(images=img, return_tensors="pt").to(dev)
        with torch.no_grad():
            pred = dmodel(**inp).predicted_depth            # (1,h,w) at model res
        pred = F.interpolate(pred.unsqueeze(1), size=(img.height, img.width),
                            mode="bicubic", align_corners=False)[0, 0]
        if args.depth_invert:
            pred = -pred                                    # near (small depth) -> bright
        return to_pil_u8(pred), pred

    def render_edge(img: Image.Image) -> tuple[Image.Image, torch.Tensor]:
        g = torch.from_numpy(np.asarray(img.convert("L"), dtype=np.float32) / 255.0)
        g = g.to(dev)[None, None]
        if args.edge_blur > 0:
            r = int(args.edge_blur) | 1
            g = F.avg_pool2d(F.pad(g, (r // 2,) * 4, mode="reflect"), r, 1)
        gx = F.conv2d(F.pad(g, (1, 1, 1, 1), mode="reflect"), kx)
        gy = F.conv2d(F.pad(g, (1, 1, 1, 1), mode="reflect"), ky)
        mag = torch.sqrt(gx * gx + gy * gy)[0, 0]
        # a square root compresses the dynamic range the way the usual log-scaled
        # gradient displays do, so thin object boundaries are not crushed by a few
        # strong specular highlights when the map is quantised to 8 bits
        return to_pil_u8(mag), mag

    report: dict = {"clip_model": args.clip_model, "pretrained": args.clip_pretrained,
                    "depth_model": args.depth_model, "depth_invert": bool(args.depth_invert),
                    "edge_blur": int(args.edge_blur), "splits": {}}

    for split in args.splits:
        paths = list_split_images(Path(args.images_root), split)
        cap = Path(args.captions_dir) / f"captions_{split}.jsonl"
        if not cap.is_file():
            raise SystemExit(f"[FATAL] missing {cap}")
        jpaths = [json.loads(l)["path"] for l in
                  cap.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(jpaths) != len(paths):
            raise SystemExit(f"[FATAL] {split}: jsonl {len(jpaths)} rows vs images "
                             f"{len(paths)}; the caches cannot be aligned")
        bad = [i for i, (a, b) in enumerate(zip(jpaths, paths))
               if Path(a).name != b.name or Path(a).parent.name != b.parent.name]
        if bad:
            raise SystemExit(f"[FATAL] {split}: row order differs from captions_{split}"
                             f".jsonl at {len(bad)} positions, first {bad[:5]}. Writing "
                             f"this cache would misalign every per-row array downstream.")
        print(f"[gem-geo] {split}: {len(paths)} images, row order verified against "
              f"captions_{split}.jsonl")
        if args.max_images:
            paths = paths[: args.max_images]
            print(f"[gem-geo] {split}: CAPPED to {len(paths)} (--max-images); this cache "
                  f"is a PREFIX of the split and is only valid for a run limited to "
                  f"those rows")

        acc = {m: {"1024": [], "1280": []} for m in ("depth", "edge")}
        acc_img1024: list[np.ndarray] = []
        acc_img1280: list[np.ndarray] = []
        bs = args.batch_size
        for s in range(0, len(paths), bs):
            chunk = paths[s:s + bs]
            dpils, epils, dts, ets = [], [], [], []
            for p in chunk:
                im = Image.open(p).convert("RGB")
                dpl, dt = render_depth(im)
                epl, et = render_edge(im)
                dpils.append(dpl)
                epils.append(epl)
                dts.append(dt)
                ets.append(et)
            if s == 0:
                # the two derived maps must actually be DIFFERENT images; if a
                # rendering bug returned the same map twice, every downstream
                # conclusion about "depth and edge" would be about one modality
                a = dts[0].flatten()
                b = ets[0].flatten()
                dts_c = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
                report.setdefault("render_sanity", {})
                report["render_sanity"][split] = {
                    "depth_edge_pixel_corr_first_row": dts_c,
                    "depth_range": [float(dts[0].min()), float(dts[0].max())],
                    "edge_range": [float(ets[0].min()), float(ets[0].max())]}
                print(f"  [render] first row depth-vs-edge pixel corr {dts_c:+.4f} "
                      f"(near 1.0 would mean the two maps are the same image)")
            with torch.no_grad():
                for m, pils in (("depth", dpils), ("edge", epils)):
                    x = torch.stack([preprocess(im) for im in pils]).to(dev)
                    acc[m]["1024"].append(model.encode_image(x).float().cpu().numpy())
                    acc[m]["1280"].append(pen["x"][:, 0].float().cpu().numpy())
                xr = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in chunk]).to(dev)
                acc_img1024.append(model.encode_image(xr).float().cpu().numpy())
                acc_img1280.append(pen["x"][:, 0].float().cpu().numpy())
            if s % (bs * 20) == 0:
                print(f"  [gem-geo] {split} {s + len(chunk)}/{len(paths)}")

        feats = {m: {d: l2(np.concatenate(acc[m][d])) if d == "1024"
                     else np.concatenate(acc[m][d]) for d in ("1024", "1280")}
                 for m in ("depth", "edge")}
        fimg = l2(np.concatenate(acc_img1024))
        fimg1280 = l2(np.concatenate(acc_img1280))

        # ---- the distinctness measurement.  Only takes a few dot products, and
        # without it a "depth tower" that is a relabelling of the RGB tower would be
        # trained, reported as a gain, and only caught by an ablation much later.
        sims = {}
        for m in ("depth", "edge"):
            sims[f"cos_{m}_vs_image_1024"] = float((feats[m]["1024"] * fimg).sum(1).mean())
            sims[f"cos_{m}_vs_image_1280"] = float(
                (l2(feats[m]["1280"]) * fimg1280).sum(1).mean())
        sims["cos_depth_vs_edge_1024"] = float(
            (feats["depth"]["1024"] * feats["edge"]["1024"]).sum(1).mean())
        sims["cos_depth_vs_edge_1280"] = float(
            (l2(feats["depth"]["1280"]) * l2(feats["edge"]["1280"])).sum(1).mean())
        # both spaces, because the NVOL selector may hand the generation condition the
        # 1280-d branch rather than the projected one, and the two do not agree
        rep: dict = {"n": len(paths), "dim1024": int(feats["depth"]["1024"].shape[1]),
                     "dim1280": int(feats["depth"]["1280"].shape[1]), **sims}
        print(f"  [distinctness] {split}: cos(depth,image) "
              f"{sims['cos_depth_vs_image_1024']:+.4f}/{sims['cos_depth_vs_image_1280']:+.4f}"
              f"  cos(edge,image) "
              f"{sims['cos_edge_vs_image_1024']:+.4f}/{sims['cos_edge_vs_image_1280']:+.4f}"
              f"  cos(depth,edge) "
              f"{sims['cos_depth_vs_edge_1024']:+.4f}/{sims['cos_depth_vs_edge_1280']:+.4f}"
              f"  (1024/1280)")
        for _k, _v in sims.items():
            if _k.endswith("1024") and _v > 0.95:
                # NOT fatal: it is a real measurement about these stimuli and it must be
                # visible in the report, because it changes how a later "depth/edge head
                # helps" result has to be read
                print(f"  [WARN] {_k} = {_v:.4f}: a derived modality is close to the RGB "
                      f"embedding in CLIP space.  A head for it is then closer to a "
                      f"re-parameterisation of the image head than to new information, "
                      f"and any gain attributed to it must be read with that in mind.")
        rep["render_sanity"] = report.get("render_sanity", {}).get(split, {})

        for m in ("depth", "edge"):
            np.save(out / f"clip_{m}1024_{split}.npy",
                    feats[m]["1024"].astype(np.float32))
            np.save(out / f"clip_{m}1280_{split}.npy",
                    feats[m]["1280"].astype(np.float16))
        report["splits"][split] = rep

    h.remove()
    # MERGE INTO ANY EXISTING REPORT, do not overwrite it.  The orchestrator extracts
    # the two splits in SEPARATE invocations (`run_gem_geo.sh`, so a failure on one
    # does not discard the other), and each invocation writes this file.  Overwriting
    # meant the second call erased the first split's measurements -- which is exactly
    # what happened on the first full run, where `gem_geo_report.json` ended up
    # documenting only `test` and the train distinctness numbers had to be recomputed
    # from the arrays afterwards.  The caches were never wrong; only the record of
    # them was lost, and a lost record is how a later reader concludes a check was
    # never run.
    rp = out / "gem_geo_report.json"
    if rp.is_file():
        try:
            prev = json.loads(rp.read_text(encoding="utf-8"))
        except Exception:                                       # noqa: BLE001
            prev = {}
        # a previous report for a DIFFERENT configuration must not be merged into
        # this one: the distinctness numbers would then mix two definitions
        same_cfg = all(prev.get(k) == report[k] for k in
                       ("clip_model", "pretrained", "depth_model", "depth_invert",
                        "edge_blur"))
        if same_cfg:
            merged_splits = {**prev.get("splits", {}), **report["splits"]}
            merged_sanity = {**prev.get("render_sanity", {}), **report.get("render_sanity", {})}
            report["splits"] = merged_splits
            report["render_sanity"] = merged_sanity
            report["merged_with_previous"] = True
        else:
            print("[gem-geo] NOTE the existing report was written with a DIFFERENT "
                  "configuration (depth model / blur / invert changed), so it is "
                  "replaced rather than merged: mixing the two would make the "
                  "distinctness numbers incomparable.")
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "splits"}, indent=2))
    print(f"[gem-geo] wrote {out}/clip_depth*_{{train,test}}.npy and "
          f"clip_edge*_{{train,test}}.npy")


if __name__ == "__main__":
    main()
