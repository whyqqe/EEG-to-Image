#!/usr/bin/env python
"""Generate per-image BLIP2 captions for the THINGS-EEG2 stimulus set.

This replaces the earlier attempt in the workspace, whose caption files turned out
to hold the *prompt* handed to a VLM ("Write a detailed Stable Diffusion XL prompt
for this image...") rather than any generated text, and only at concept
granularity (1,654 rows keyed 0, 10, 20, ...).  A text target that is really a
prompt is worse than no text target, because it looks populated while carrying no
per-stimulus information.

Following the SOTA recipe this pipeline aligns against (CognitionCapturerPro,
which reads `weights/texts/eeg/texts_BLIP2_{mode}.npy`), the captions are
generated per image, not per concept, using BLIP2.  A vision-language QA prompt is
used instead of bare captioning because OPT under an empty prefix collapses to
short generic strings ("a dog"), whereas the question form elicits the appearance,
colour and background detail that makes the text branch a useful auxiliary signal.

Output is JSONL keyed by `image_key`, appended incrementally so the job is
resumable and shardable across GPUs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

from loso import paths
from loso.data import captions as captions_mod
from loso.data import things

DEFAULT_PROMPT = "a photography of"
"""BLIP2's canonical captioning prefix.

An earlier draft used a long VQA-style question ("Question: Describe this image in
one detailed sentence. Mention the main object... Answer:") to elicit more
descriptive text.  That backfired: with ~30 tokens of prompt, greedy decoding on
BLIP2-OPT emitted EOS immediately on 5.9% of images, returning the prompt and
nothing else.  The short prefix is the form BLIP2 was actually validated on and
produces naturalistic captions reliably; `--prompt` still allows the longer form.
"""


def load_done(path: Path) -> set[str]:
    """image_keys already present in a partially written output file."""
    done: set[str] = set()
    if not path.is_file():
        return done
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["image_key"])
            except (json.JSONDecodeError, KeyError):
                # A torn last line from a killed job; the key is simply unknown and
                # will be regenerated, which is safe.
                continue
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("train", "test"), required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSONL (default: data/captions/blip2_<split>[_shardN].jsonl)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--num-beams", type=int, default=1,
                    help="1 = greedy; higher is slower but less repetitive")
    ap.add_argument("--no-fallback", action="store_true",
                    help="keep empty completions instead of substituting the concept "
                         "template (diagnostics only; the targets are unusable then)")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap images processed")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not (0 <= args.shard < args.num_shards):
        raise SystemExit(f"invalid shard {args.shard}/{args.num_shards}")

    paths.ensure_dirs()
    if args.out is None:
        suffix = f"_shard{args.shard}" if args.num_shards > 1 else ""
        args.out = paths.CAPTION_DIR / f"blip2_{args.split}{suffix}.jsonl"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    records = things.build_index(args.split)
    # Shard by contiguous stride rather than contiguous blocks: the index is sorted
    # by concept, so blocks would give each shard a disjoint concept range and make
    # a partial run impossible to compare across shards.
    mine = [r for i, r in enumerate(records) if i % args.num_shards == args.shard]
    done = load_done(args.out)
    todo = [r for r in mine if r.image_key not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(f"[blip2] split={args.split} shard={args.shard}/{args.num_shards} "
          f"assigned={len(mine)} done={len(done & {r.image_key for r in mine})} "
          f"todo={len(todo)}", flush=True)
    if not todo:
        print("[blip2] nothing to do", flush=True)
        return 0

    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    model_id = paths.BLIP2_ID
    print(f"[blip2] loading {model_id}", flush=True)
    processor = Blip2Processor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map={"": args.device},
    )
    model.eval()

    t0 = time.time()
    written = 0
    with args.out.open("a") as fh:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]
            images = []
            kept = []
            for rec in batch:
                try:
                    images.append(Image.open(rec.path).convert("RGB"))
                    kept.append(rec)
                except Exception as exc:  # noqa: BLE001 - one bad JPEG must not kill the run
                    print(f"[warn] unreadable {rec.path}: {exc}", file=sys.stderr, flush=True)
            if not images:
                continue

            inputs = processor(
                images=images, text=[args.prompt] * len(images), return_tensors="pt",
            ).to(args.device, torch.float16)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    do_sample=False,
                )
            texts = processor.batch_decode(generated, skip_special_tokens=True)

            for rec, text in zip(kept, texts):
                # BLIP2's LM is decoder-only, so `generate` returns the prompt
                # followed by the completion.  Strip it, substitute the concept
                # template when the completion is empty/degenerate, and record which
                # path was taken so the fallback rate stays measurable.
                raw = " ".join(text.split())
                caption, source = captions_mod.finalize_caption(
                    raw, args.prompt, rec.concept, allow_fallback=not args.no_fallback,
                )
                fh.write(json.dumps({
                    "image_key": rec.image_key,
                    "index": rec.index,
                    "concept_index": rec.concept_index,
                    "concept": rec.concept,
                    "caption": caption,
                    "source": source,
                    "raw_caption": raw,
                }) + "\n")
                written += 1
            fh.flush()

            if start % (args.batch_size * 20) == 0:
                rate = written / max(time.time() - t0, 1e-6)
                eta = (len(todo) - written) / max(rate, 1e-6)
                print(f"[blip2] {written}/{len(todo)}  {rate:.1f} img/s  "
                      f"eta {eta / 60:.1f} min", flush=True)

    print(f"[blip2] done: wrote {written} captions -> {args.out} "
          f"in {(time.time() - t0) / 60:.1f} min", flush=True)

    # Verify the store is actually informative before the feature stage trusts it.
    records = captions_mod.load_caption_records(args.split)
    ok, msg = captions_mod.caption_quality_report(records, args.split)
    print(f"[blip2] quality {'OK' if ok else 'FAIL'} -- {msg}", flush=True)
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
