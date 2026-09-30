#!/usr/bin/env python3
"""G2 stage 0: generate multi-granularity DESCRIPTIONS for every image with a VLM.

Why this stage exists
---------------------
The semantic tower needs supervision at four granularities -- overall, subject,
background, detail -- and the user's design specifies these as *descriptions*
(i.e. text produced by an image-to-text model from the image), not as pooled
image features.  This script produces those descriptions.

Design decisions and their reasons
----------------------------------
ONE STRUCTURED CALL PER IMAGE, NOT FOUR CALLS.
    Asking for all four fields in a single JSON response cuts generation cost by
    ~4x (16740 images, not 66960) and -- more importantly -- forces the model to
    keep the four fields *distinct*, because it sees them side by side. Four
    independent calls tend to converge on near-identical sentences, which would
    make the granularity axis vacuous. The redundancy is still measured
    downstream (`g2_build_targets.py` reports mean pairwise cosine between the
    four text embeddings) so the claim is auditable rather than assumed.

GREEDY DECODING (do_sample=False).
    The descriptions are supervision, so they must be reproducible: re-running
    the pipeline must not silently change the training targets.

RESUMABLE AND SHARDABLE.
    16740 generations do not fit comfortably in one interactive session, and a
    preempted job must not lose its work. Progress is appended to a JSONL file
    keyed by row index, and `--shard i --num-shards n` lets several GPU jobs
    cover disjoint slices in parallel.

ROW ORDER IS THE PIPELINE'S ROW ORDER.
    Rows come from `list_split_images`, which is the same listing used when the
    VAE / depth / DINOv2 caches were built (sorted concept dirs, sorted files).
    Alignment with those caches is therefore positional and is asserted at the
    end against an expected row count if one is supplied.

No EEG is read here: descriptions are image-side supervision only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_CACHE", "/home/sbaiae/.cache/huggingface/hub")
os.environ.setdefault("HF_HUB_OFFLINE", "1")   # read-only cache; never try to write

PROMPT = (
    "You are writing annotation text for an image, to be used as supervision. "
    "Look carefully at the image and return ONLY a JSON object with exactly "
    "these four keys, no markdown, no extra text:\n"
    '{"overall": "...", "subject": "...", "background": "...", "detail": "..."}\n'
    "Definition of each key:\n"
    "- overall: one sentence describing the whole image, i.e. what kind of scene "
    "or object this is at a glance.\n"
    "- subject: the main object only: its identity, its pose or orientation, and "
    "where it sits in the frame.\n"
    "- background: everything that is not the main object: surface, backdrop, "
    "colour, lighting and depth behind the object.\n"
    "- detail: FINE visual detail of the main object: material, texture, small "
    "parts, markings, edge quality, subtle colour variation.\n"
    "Rules: each value must be one short phrase or clause of 8 to 20 words. "
    "Do not reuse the same wording across keys. Describe only what is visible."
)

FIELDS = ("overall", "subject", "background", "detail")


def resolve_model_dir(p: str) -> str:
    """Resolve a HF cache repo dir to the snapshot dir that actually holds config.json.

    `--model` may be given as a repo id, as the `models--Org--Name` cache dir, or
    directly as a snapshot dir. The cache dir has only blobs/refs/snapshots at its
    top level, so passing it straight to from_pretrained fails with
    "Unrecognized model ... Should have a `model_type` key in its config.json".
    """
    q = Path(p)
    if (q / "config.json").is_file():
        return str(q)
    snaps = q / "snapshots"
    if snaps.is_dir():
        ref = q / "refs" / "main"
        if ref.is_file():
            h = ref.read_text().strip()
            if (snaps / h / "config.json").is_file():
                return str(snaps / h)
        cands = sorted([d for d in snaps.iterdir() if (d / "config.json").is_file()],
                       key=lambda d: d.stat().st_mtime)
        if cands:
            return str(cands[-1])
    return p


def list_split_images(images_root: Path, split: str) -> list[Path]:
    """Identical ordering to build_gt_depth_cache / the VAE caches."""
    root = images_root / ("training_images" if split == "train" else "test_images")
    paths: list[Path] = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.JPEG")) + list(d.glob("*.png")))
        paths.extend(imgs)
    return paths


def parse_fields(raw: str) -> dict[str, str] | None:
    """Extract the four fields from a model response. Returns None if unusable."""
    s = raw.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.M).strip()
    obj = None
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        try:
            obj = json.loads(s[i : j + 1])
        except Exception:
            obj = None
    if obj is None:
        # Fallback: scrape "key": "value" pairs individually. A single malformed
        # field must not throw away a whole image's supervision.
        obj = {}
        for k in FIELDS:
            m = re.search(rf'"{k}"\s*:\s*"(.*?)"\s*(?:,|\}})', s, flags=re.S)
            if m:
                obj[k] = m.group(1).strip()
    out = {}
    for k in FIELDS:
        v = obj.get(k)
        if isinstance(v, (list, tuple)):
            v = " ".join(str(x) for x in v)
        if not isinstance(v, str) or not v.strip():
            return None
        out[k] = " ".join(v.split())[:400]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", default="/project/peilab/why/data/images_set")
    ap.add_argument("--out", required=True, help="JSONL path (appended, resumable)")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--model", default="/home/sbaiae/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=260)
    ap.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="stop after N images (smoke test)")
    ap.add_argument("--expect-rows", type=int, default=0,
                    help="assert the split has exactly this many images")
    ap.add_argument("--dump-raw", action="store_true", help="also store the raw response")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    paths = list_split_images(Path(args.images_root), args.split)
    n_all = len(paths)
    if args.expect_rows and n_all != args.expect_rows:
        raise SystemExit(f"row mismatch: images={n_all} expected={args.expect_rows}")
    model_dir = resolve_model_dir(args.model)
    if not (Path(model_dir) / "config.json").is_file():
        raise SystemExit(f"no config.json under {args.model} (resolved: {model_dir}); "
                         f"the HF cache entry may be incomplete")
    print(f"[cap] model_dir={model_dir}", flush=True)
    # shard BEFORE limit so shards stay disjoint and complete
    paths = [p for i, p in enumerate(paths) if i % args.num_shards == args.shard]
    if args.limit:
        paths = paths[: args.limit]
    print(f"[cap] split={args.split} images={n_all} this_shard={len(paths)} "
          f"shard={args.shard}/{args.num_shards}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if out.is_file():
        with out.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["path"])
                except Exception:
                    continue
    todo = [p for p in paths if str(p) not in done]
    print(f"[cap] already done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return

    processor = AutoProcessor.from_pretrained(
        model_dir, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map=args.device).eval()

    lock = out.with_suffix(out.suffix + ".lock")
    n_ok, n_bad, t0 = 0, 0, time.time()

    def run_batch(batch_paths: list[Path]) -> list[str | None]:
        """One processor call for the whole batch; returns one raw string per image."""
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        texts, images_in = [], []
        for im in imgs:
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": im},
                {"type": "text", "text": PROMPT}]}]
            texts.append(processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
            images_in.append(im)
        inputs = processor(text=texts, images=images_in, padding=True,
                           return_tensors="pt").to(args.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False)
        cut = inputs["input_ids"].shape[1]
        return [processor.decode(g[cut:], skip_special_tokens=True) for g in gen]

    with out.open("a", encoding="utf-8") as fo, lock.open("w") as fl:
        fl.write(f"{os.getpid()}\n")
        for s in range(0, len(todo), args.batch_size):
            batch = todo[s : s + args.batch_size]
            try:
                raws = run_batch(batch)
            except Exception as e:                       # OOM or shape issue
                print(f"[cap] batch failed ({type(e).__name__}: {e}); "
                      f"falling back to per-image", flush=True)
                torch.cuda.empty_cache()
                raws = []
                for p in batch:
                    try:
                        raws.append(run_batch([p])[0])
                    except Exception as e2:
                        print(f"[cap] single failed {p.name}: {e2}", flush=True)
                        raws.append(None)

            for p, raw in zip(batch, raws):
                rec = {"path": str(p), "split": args.split, "model": Path(args.model).name}
                if raw is None:
                    rec["error"] = "generation_failed"
                else:
                    f = parse_fields(raw)
                    if f is None:
                        rec["error"] = "parse_failed"
                        if args.dump_raw:
                            rec["raw"] = raw[:1000]
                    else:
                        rec.update(f)
                if args.dump_raw and raw is not None and "raw" not in rec:
                    rec["raw"] = raw[:1000]
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_ok += ("error" not in rec)
                n_bad += ("error" in rec)
            fo.flush()

            if (s // args.batch_size) % 20 == 0:
                el = time.time() - t0
                rate = (s + len(batch)) / max(el, 1e-6)
                eta = (len(todo) - s - len(batch)) / max(rate, 1e-6)
                print(f"[cap] {s + len(batch)}/{len(todo)} ok={n_ok} bad={n_bad} "
                      f"{rate:.2f} img/s eta={eta/60:.1f} min", flush=True)

    print(f"[cap] done ok={n_ok} bad={n_bad} in {(time.time()-t0)/60:.1f} min -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
