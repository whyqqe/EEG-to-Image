#!/usr/bin/env python
"""Compare captioning prompts on a small sample before committing to a full run.

Run as a short GPU job.  The full captioning pass costs ~6 min of GPU per prompt
variant, so measuring the empty-completion rate on a few hundred images first is
cheaper than discovering the rate again after each full pass.

Reports, per prompt: empty rate, degenerate rate, distinct fraction, mean length.
"""
from __future__ import annotations

import argparse
import time

import torch
from PIL import Image

from loso import paths
from loso.data import captions as captions_mod
from loso.data import things

PROMPTS = {
    "short_canonical": "a photography of",
    "vqa_detailed": (
        "Question: Describe this image in one detailed sentence. Mention the main "
        "object, its appearance and colour, and the background. Answer:"
    ),
    "vqa_short": "Question: What is in this image? Answer:",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    records = things.build_index("train")
    # Stride across the whole index so the sample is not one contiguous concept
    # block, which would make per-concept difficulty look like prompt quality.
    stride = max(1, len(records) // args.n)
    sample = records[::stride][: args.n]
    print(f"[diag] {len(sample)} images, stride {stride}", flush=True)

    processor = Blip2Processor.from_pretrained(paths.BLIP2_ID)
    model = Blip2ForConditionalGeneration.from_pretrained(
        paths.BLIP2_ID, torch_dtype=torch.float16, device_map={"": args.device},
    ).eval()

    images = []
    for rec in sample:
        with Image.open(rec.path) as im:
            images.append(im.convert("RGB"))

    results: dict[str, dict] = {}
    for name, prompt in PROMPTS.items():
        t0 = time.time()
        outputs: list[str] = []
        for start in range(0, len(images), args.batch_size):
            chunk = images[start:start + args.batch_size]
            inputs = processor(images=chunk, text=[prompt] * len(chunk),
                               return_tensors="pt").to(args.device, torch.float16)
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     num_beams=1, do_sample=False)
            outputs.extend(processor.batch_decode(gen, skip_special_tokens=True))

        finalized = [captions_mod.finalize_caption(o, prompt, sample[i].concept,
                                                   allow_fallback=False)[0]
                     for i, o in enumerate(outputs)]
        n = len(finalized)
        empty = sum(1 for c in finalized if not c.strip())
        degen = sum(1 for c in finalized if captions_mod.is_degenerate(c))
        distinct = len(set(finalized)) / n
        mean_len = sum(len(c.split()) for c in finalized) / n
        results[name] = {"empty": empty / n, "degenerate": degen / n,
                         "distinct": distinct, "mean_words": mean_len}
        print(f"[diag] {name:18s} empty={empty / n:6.2%} degenerate={degen / n:6.2%} "
              f"distinct={distinct:.3f} mean_words={mean_len:5.1f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        print(f"        examples: " + " | ".join(
            repr(c[:55]) for c in finalized[:3]), flush=True)

    best = min(results, key=lambda k: (results[k]["empty"], -results[k]["distinct"]))
    print(f"\n[diag] best prompt by empty-rate then diversity: {best}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
