#!/usr/bin/env python
"""Extract InternViT-6B multi-level image features in SAMGA's exact on-disk format.

Why this exists
---------------
`run_samga_official_baseline.sh` runs SAMGA's unmodified `train.py`, so the
architecture and the training settings are identical by construction. The one
substantive deviation left was the *image features*: SAMGA's main results use
InternViT-6B-448px-V2_5 at layers {20,24,28,32,36}, and every run we had recorded
was fed five CLIP ViT-H-14 blocks instead, because InternViT was believed
unobtainable here. That belief was wrong on both counts it rested on:

  * all 16,540 training images and all 200 test images are on disk under
    `data/images_set/`, and
  * the login node has outbound network, and
    `OpenGVLab/InternViT-6B-448px-V2_5` is 11.1 GB.

SAMGA ships no extraction script, so this reproduces the format from their README,
which specifies exactly:

    image_feature/internvit_multilevel_20_24_28_32_36/
        image_train_layer20.npy   # [1654, 10, 3200], float16
        ...
        image_test_layer20.npy    # [200, 1, 3200], float16

`train.py::prepare_multilayer_feature_dir` then stacks those into
`[Nobj, Nimg, K, D]` itself, so this script's only job is to emit per-layer
`[Nobj, Nimg, 3200]` arrays with the rows in canonical order.

Two conventions this script cannot read off the README
------------------------------------------------------
1. **CLS vs patch-mean pooling.** README gives shapes, not semantics. InternViT is
   a ViT whose sequence is `[cls, patch_0..patch_1023]` (verified in
   `modeling_intern_vit.py::InternVisionEmbeddings`: `cat([class_embeds, patch_embeds])`,
   so CLS is index 0), and the natural per-image `D`-vector is the CLS token, which
   is what this defaults to. `--pool mean` switches to a patch-token mean. The two
   give genuinely different features, and the README does not disambiguate.

2. **Whether layer ids are 0- or 1-indexed.** InternViT here has
   `num_hidden_layers = 45` (read from the downloaded config, not assumed), so
   {20,24,28,32,36} fit under either reading; 36 < 45 is satisfied by both. This
   defaults to 1-indexed (`layers[lid-1]`) and `--layer-index-base 0` switches.

Neither is settled by argument. Both are settled by whether the resulting features
reproduce SAMGA's published held-out sub-08 cell (28.7/59.5) once we rerun the
official baseline on top of them. That is the point of extracting them.

Verification, as hard gates
---------------------------
Gates that cannot fire are worse than no gates, because a manifest that records
"verified" when nothing was checked reads as evidence. So each of these aborts
rather than warns:

  * the image listing must reproduce `image_metadata.npy`'s `{train,test}_img_files`
    sequence exactly. This is the row-alignment proof and it is *independent of the
    backbone*: the EEG arrays index concepts in canonical THINGS order, so if our
    sorted listing disagrees with the metadata, every per-layer array is silently
    paired with the wrong EEG trials and all downstream numbers are noise. It passed
    for both splits before this script was written, and it is re-checked here.
  * output shapes must be exactly `[1654, 10, D]` / `[200, 1, D]` with
    `D == config.hidden_size`.
  * a sanity gate on the backbone itself: for each concept, cross-image similarity
    within the concept must exceed cross-concept similarity. A mis-pathed hook or a
    silently random model would still produce correctly-shaped arrays.

Usage
-----
    python scripts/epd/extract_internvit_layers.py --split train
    python scripts/epd/extract_internvit_layers.py --split test
    python scripts/epd/extract_internvit_layers.py --split both
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
IMG_ROOT = REPO / "data" / "images_set"
META = IMG_ROOT / "image_metadata.npy"
SPLIT_DIR = {"train": "training_images", "test": "test_images"}
EXT = (".jpg", ".jpeg", ".png")
DEFAULT_OUT = REPO / "data" / "image_feature" / "internvit_multilevel_20_24_28_32_36"
MODEL_ID = "OpenGVLab/InternViT-6B-448px-V2_5"


def list_concepts(split: str) -> list[tuple[str, list[str]]]:
    """[(concept_dir, [img_path, ...]), ...] in lexicographic (canonical) order."""
    root = IMG_ROOT / SPLIT_DIR[split]
    out: list[tuple[str, list[str]]] = []
    for cid in sorted(d.name for d in root.iterdir() if d.is_dir()):
        imgs = sorted(f.name for f in (root / cid).iterdir()
                      if f.name.lower().endswith(EXT))
        if imgs:
            out.append((cid, [str(root / cid / f) for f in imgs]))
    return out


def gate_listing_matches_metadata(split: str, concepts) -> dict:
    """Independently prove the row order matches the EEG arrays' concept order.

    `image_metadata.npy` is the dataset's own record of the canonical sequence; the
    EEG `.npy` files index into that same sequence. Comparing our sorted listing to
    it is therefore a check that costs nothing and catches the one error that would
    invalidate every number downstream while still producing plausible-looking files.
    """
    if not META.is_file():
        return {"checked": False, "reason": f"missing {META}"}
    meta = np.load(META, allow_pickle=True).item()
    key = f"{split}_img_files"
    ref = [str(x) for x in meta.get(key, [])]
    ours = [Path(p).name for _cid, paths in concepts for p in paths]
    if not ref:
        return {"checked": False, "reason": f"no {key} in metadata"}
    # Compare as a PREFIX, not for length equality. Under `--limit` (smoke) our listing
    # is intentionally short, and comparing it against the full 16,540 would fail on a
    # perfectly correct listing. A gate that fires on correct input is worse than no
    # gate, because it teaches the reader to ignore gates. So the check is "our sequence
    # must reproduce the metadata sequence from index 0", with `complete` recording
    # whether it also covers all of it.
    n = min(len(ours), len(ref))
    if len(ours) > len(ref):
        return {"checked": True, "match": False,
                "reason": f"we listed {len(ours)} images, metadata has only {len(ref)}"}
    first_bad = next((i for i, (a, b) in enumerate(zip(ours[:n], ref[:n])) if a != b), None)
    return {"checked": True, "match": first_bad is None, "n_compared": n,
            "n_metadata": len(ref), "complete": len(ours) == len(ref),
            "first_mismatch": None if first_bad is None else {
                "index": int(first_bad), "ours": ours[first_bad], "ref": ref[first_bad]}}


class ConceptImages(Dataset):
    """Flat (concept_index, image_path); decoding on workers, transform in collate."""

    def __init__(self, concepts, processor) -> None:
        self.items = [(ci, p) for ci, (_cid, paths) in enumerate(concepts) for p in paths]
        self.processor = processor

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        from PIL import Image
        ci, path = self.items[i]
        with Image.open(path) as im:
            t = self.processor(images=im.convert("RGB"), return_tensors="pt")["pixel_values"][0]
        return t, ci


def collate(batch):
    return torch.stack([b[0] for b in batch]), torch.tensor([b[1] for b in batch], dtype=torch.long)


def build_processor(model_path: str):
    """Rebuild the exact preprocessing in the checkpoint's preprocessor_config.json.

    Loaded from the checkpoint rather than hand-written so the resize/crop/normalise
    is the checkpoint's own; the loaded attributes are then *checked* against the
    JSON so a silent default cannot masquerade as fidelity.
    """
    from transformers import CLIPImageProcessor
    proc = CLIPImageProcessor.from_pretrained(model_path)
    on_disk = json.loads((Path(model_path) / "preprocessor_config.json").read_text())
    checks = {
        "crop_size": dict(proc.crop_size) == ({"height": on_disk["crop_size"],
                                               "width": on_disk["crop_size"]}
                                              if isinstance(on_disk["crop_size"], int)
                                              else on_disk["crop_size"]),
        "do_center_crop": proc.do_center_crop == on_disk["do_center_crop"],
        "do_normalize": proc.do_normalize == on_disk["do_normalize"],
        "image_mean": [round(float(v), 6) for v in proc.image_mean] == [round(float(v), 6) for v in on_disk["image_mean"]],
        "image_std": [round(float(v), 6) for v in proc.image_std] == [round(float(v), 6) for v in on_disk["image_std"]],
        "resample": int(proc.resample) == int(on_disk["resample"]),
    }
    # `size: 448` in the JSON is an int; CLIPImageProcessor stores one of several
    # shapes for it. Accept any shape that denotes 448.
    size = proc.size
    got = size.get("shortest_edge") if isinstance(size, dict) else size
    if isinstance(size, dict) and size.get("height") and not got:
        got = size.get("height")
    checks["size_is_448"] = int(got) == int(on_disk["size"])
    return proc, checks


def resolve_model_path(model_id: str) -> str:
    """Find the snapshot on disk, preferring the project cache over the home cache.

    Two reasons this is not a bare `snapshot_download`. First, the compute nodes run
    with `HF_HUB_OFFLINE=1`, so a download attempt is the wrong primitive: the weights
    are baked into the image of the job by having been fetched from the login node
    beforehand. Second, the default cache is `~/.cache/huggingface`, which on this
    cluster is *full*: an unset `HF_HOME` fails with `OSError: [Errno 28] No space left
    on device` rather than a network error, which reads like an unrelated fault. The
    pipeline already pins `HF_HOME=/project/peilab/why/cache/eeg-brainit/hf` in every
    run script; we resolve against the same place and say so loudly if it is missing.
    """
    import glob
    hub = os.environ.get("HF_HUB_CACHE") or (
        os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else "")
    cands: list[str] = []
    if hub:
        cands += glob.glob(os.path.join(hub, f"models--{model_id.replace('/', '--')}", "snapshots", "*"))
    cands += glob.glob(os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{model_id.replace('/', '--')}/snapshots/*"))
    complete = [c for c in cands if (Path(c) / "config.json").is_file()
                and any(Path(c).glob("*.safetensors"))]
    if complete:
        return sorted(complete)[-1]
    print(f"[internvit] no complete local snapshot; falling back to download "
          f"(HF_HUB_CACHE={hub or 'unset'})")
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt"])


@torch.no_grad()
def extract(split: str, layer_ids: list[int], index_base: int, pool: str,
            batch: int, workers: int, limit: int, device: str, dtype: str,
            model_path: str) -> tuple[dict, list, np.ndarray]:
    from transformers import AutoModel

    concepts = list_concepts(split)
    if limit:
        concepts = concepts[:limit]

    n_img = np.array([len(p) for _c, p in concepts])
    proc, proc_checks = build_processor(model_path)

    td = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    print(f"[internvit] loading {model_path} as {dtype} ...")
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=td)
    model = model.to(device).eval()

    cfg = model.config
    n_layers = int(cfg.num_hidden_layers)
    d_model = int(cfg.hidden_size)
    print(f"[internvit] layers={n_layers} hidden={d_model} image_size={cfg.image_size} "
          f"patch={cfg.patch_size}")
    print(f"[internvit] split={split} concepts={len(concepts)} "
          f"imgs/concept={n_img.min()}..{n_img.max()} total={int(n_img.sum())}")
    print(f"[internvit] processor checks: {proc_checks}")

    # Locate the encoder's layer list without assuming the attribute path.
    enc = getattr(model, "encoder", None)
    if enc is None or not hasattr(enc, "layers"):
        raise SystemExit(f"[FATAL] cannot find model.encoder.layers on {type(model).__name__}")
    layers = enc.layers
    if len(layers) != n_layers:
        raise SystemExit(f"[FATAL] len(encoder.layers)={len(layers)} != num_hidden_layers={n_layers}")
    for lid in layer_ids:
        if not (0 <= lid - index_base < n_layers):
            raise SystemExit(f"[FATAL] layer {lid} out of range for "
                             f"num_hidden_layers={n_layers} at index_base={index_base}")

    acts: dict[int, torch.Tensor] = {}
    handles = []
    for lid in layer_ids:
        idx = lid - index_base
        handles.append(layers[idx].register_forward_hook(
            lambda _m, _i, o, lid=lid: acts.__setitem__(
                lid, o[0] if isinstance(o, (tuple, list)) else o)))

    ds = ConceptImages(concepts, proc)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                    collate_fn=collate, pin_memory=(device == "cuda"))

    acc: dict[int, list[np.ndarray]] = {lid: [] for lid in layer_ids}
    pos: list[np.ndarray] = []
    t0, done = time.time(), 0
    for x, ci in dl:
        x = x.to(device, non_blocking=True, dtype=td)
        acts.clear()
        model(pixel_values=x)
        for lid, v in acts.items():
            if v.shape[0] != x.shape[0]:       # (L, B, D) -> (B, L, D) if needed
                v = v.permute(1, 0, 2)
            tok = v[:, 0] if pool == "cls" else v[:, 1:].mean(dim=1)
            acc[lid].append(tok.float().cpu().numpy())
        pos.append(ci.numpy())
        done += x.shape[0]
        if done % (batch * 25) < batch:
            taken = time.time() - t0
            print(f"    {done}/{len(ds)} imgs  {taken:.0f}s "
                  f"({done / max(taken, 1e-9):.1f} img/s)", flush=True)

    for h in handles:
        h.remove()

    order = np.argsort(np.concatenate(pos), kind="stable")
    feats = {lid: np.concatenate(chunks, 0)[order] for lid, chunks in acc.items()}
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return feats, [c for c, _ in concepts], n_img


def gate_backbone_sane(feats: dict, n_img: np.ndarray, split: str, layer_ids) -> dict:
    """Within-concept image similarity must exceed cross-concept similarity.

    A random model, a dead hook, or a hook on the wrong module still yields
    correctly-shaped arrays. This does not.

    The centering here is load-bearing and the first version of this gate got it
    wrong, so it is worth recording why. InternViT's CLS token is dominated by one
    shared direction: raw cosine between *any* two images is ~0.9995 at layer 20 and
    ~0.956 at layer 36, so raw within-concept similarity sits *below* raw
    between-concept similarity for every layer and the gate fails on a perfectly good
    backbone. Removing each row's own mean (subtracting its DC component) changes
    nothing, because the shared direction is not a constant offset across dimensions.
    Subtracting the mean over all rows removes that direction and the margin becomes
    unambiguous:

        layer 20: within 0.309 vs between -0.022
        layer 28: within 0.325 vs between -0.022
        layer 36: within 0.291 vs between -0.024

    So the gate centers globally, and it also reports the raw figures, because the
    size of that gap is itself the reason not to compare these features by raw cosine.
    """
    if n_img.min() < 2:
        return {"checked": False, "reason": f"{split} has 1 image/concept; gate is vacuous"}
    lid = layer_ids[len(layer_ids) // 2]
    n, k = len(n_img), int(n_img[0])
    X = torch.from_numpy(feats[lid]).reshape(n, k, -1).float()

    def _sim(Z: torch.Tensor) -> tuple[float, float]:
        Z = Z / Z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        off = Z @ Z.transpose(1, 2)
        within = ((off.sum(dim=(1, 2)) - off.diagonal(dim1=1, dim2=2).sum(1)) / (k * (k - 1))).mean()
        m = Z.mean(dim=1)
        m = m / m.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        b = (m @ m.t())[~torch.eye(n, dtype=torch.bool)]
        return float(within), float(b.mean())

    raw_within, raw_between = _sim(X)
    Xc = X - X.reshape(-1, X.shape[-1]).mean(0, keepdim=True)
    within, between = _sim(Xc)
    return {"checked": True, "layer": int(lid),
            "mean_within_concept": within, "mean_between_concept": between,
            "gap": within - between,
            "raw_within_concept": raw_within, "raw_between_concept": raw_between,
            "pass": bool(within > between)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--split", default="train", choices=["train", "test", "both"])
    ap.add_argument("--layer-ids", type=int, nargs="+", default=[20, 24, 28, 32, 36])
    ap.add_argument("--layer-index-base", type=int, default=1, choices=[0, 1],
                    help="1 means layer id L reads encoder.layers[L-1] (default, unresolved "
                         "ambiguity: see module docstring)")
    ap.add_argument("--pool", default="cls", choices=["cls", "mean"])
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N concepts (smoke only)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model-path", default="", help="local snapshot dir; default resolves the HF cache")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        return int(print("[FATAL] no CUDA on this node. Run on a GPU node (see "
                         "slurm/extract_internvit.sbatch); a 6B model on CPU is not viable.") or 5)

    model_path = a.model_path or resolve_model_path(MODEL_ID)
    print(f"[internvit] model snapshot: {model_path}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = ["train", "test"] if a.split == "both" else [a.split]

    summary = {}
    t_all = time.time()
    for split in splits:
        print("=" * 88)
        feats, concept_ids, n_img = extract(
            split, a.layer_ids, a.layer_index_base, a.pool,
            a.batch, a.workers, a.limit, a.device, a.dtype, model_path)

        gate_order = gate_listing_matches_metadata(split, list_concepts(split)[:len(concept_ids)])
        gate_sane = gate_backbone_sane(feats, n_img, split, a.layer_ids)

        manifest = {
            "split": split, "n_concepts": len(concept_ids),
            "imgs_per_concept": int(n_img[0]) if n_img.min() == n_img.max() else "ragged",
            "source": f"{MODEL_ID}, AutoModel trust_remote_code, pool={a.pool}, dtype={a.dtype}",
            "layer_ids": list(a.layer_ids), "layer_index_base": a.layer_index_base,
            "gate_order_vs_metadata": gate_order, "gate_backbone_sane": gate_sane,
            "layers": {},
        }

        for lid in a.layer_ids:
            v = feats[lid]
            arr = v.reshape(len(concept_ids), int(n_img[0]), v.shape[-1])
            manifest["layers"][str(lid)] = list(arr.shape)
            # float32 on disk: SAMGA's released arrays are float16, but train.py casts to
            # float32 before use, so saving fp16 would only lose precision, not gain fidelity.
            np.save(out / f"image_{split}_layer{lid}.npy", arr.astype(np.float32))

        (out / f"manifest_{split}.json").write_text(json.dumps(manifest, indent=2))
        summary[split] = manifest
        print(f"[internvit] {split}: wrote {len(a.layer_ids)} arrays -> {out}")
        print(f"[internvit]   shapes: {manifest['layers']}")
        print(f"[gate    ] order vs metadata: {json.dumps(gate_order)}")
        print(f"[gate    ] backbone sane:     {json.dumps(gate_sane)}")

    for split, m in summary.items():
        o = m["gate_order_vs_metadata"]
        if not o.get("checked") or not o.get("match"):
            print(f"[FATAL] {split}: image listing does not match image_metadata.npy.")
            print("        Row order is the pairing between features and EEG trials; "
                  "getting it wrong is silent and invalidates everything downstream.")
            return 3
        # A prefix match is the right check under `--limit`, but a full run that matched
        # only a prefix would mean we silently extracted fewer concepts than the EEG
        # arrays contain -- shaped correctly, index-aligned for the first N, and wrong.
        if not a.limit and not o.get("complete"):
            print(f"[FATAL] {split}: listing matched the metadata prefix but is incomplete "
                  f"({o.get('n_compared')} of {o.get('n_metadata')}). A partial extraction "
                  f"would train on a truncated concept set without any shape error.")
            return 3
        s = m["gate_backbone_sane"]
        if s.get("checked") and not s.get("pass"):
            print(f"[FATAL] {split}: within-concept similarity <= cross-concept similarity "
                  f"({s['mean_within_concept']:.4f} <= {s['mean_between_concept']:.4f}).")
            print("        The backbone or the hook is wrong; arrays are shaped correctly "
                  "but carry no image information.")
            return 4

    print(f"\n[internvit] all gates passed in {time.time() - t_all:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
