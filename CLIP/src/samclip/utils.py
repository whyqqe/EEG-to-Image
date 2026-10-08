"""Small shared helpers: seeding, logging, config IO, checkpointing."""
from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import config


# ------------------------------------------------------------------- logging
def get_logger(name: str = "samclip", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "[%(asctime)s %(levelname)s %(name)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------------------- config
def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    """Load a YAML config, resolving an optional ``base: <file>`` inheritance.

    A single flat schema would force every fold config to duplicate the whole model
    section, and duplicated model sections are how a Stage-2 run silently loads a
    Stage-1 checkpoint with a different width. Inheritance keeps exactly one copy of
    the architecture.
    """
    import yaml
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"config {path} did not parse to a mapping")
    base = cfg.pop("base", None)
    if base:
        base_cfg = load_config(path.parent / base)
        cfg = _deep_merge(base_cfg, cfg)
    return cfg


def merge_cli_overrides(cfg: dict, overrides: dict) -> dict:
    """Shallow-merge CLI overrides (values that are not None) into `cfg`."""
    out = dict(cfg)
    for k, v in overrides.items():
        if v is not None:
            out[k] = v
    return out


def dump_config(cfg: dict, out_dir: Path, name: str = "config.json") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(json.dumps(cfg, indent=2, default=str))


# --------------------------------------------------------------- checkpoints
def save_checkpoint(state: dict, out_dir: Path, name: str, keep_best: bool = True) -> Path:
    """Save `state` to `out_dir/name` and (optionally) mirror it to `best.pt`.

    NOTE ON RETENTION. This function does NOT cap anything -- an earlier version of this
    docstring claimed it did ("retention is capped deliberately"), which was false and
    cost real time: `train_stage_a` saves `epoch{epoch:03d}.pt` every epoch WITHOUT going
    through here at all, so the cap that was documented did not exist on the only path
    that writes per-epoch files. The actual cap now lives in `prune_epoch_checkpoints`,
    which the training loop calls after each save. Keep the two together.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    torch.save(state, path)
    if keep_best:
        torch.save(state, out_dir / "best.pt")
    return path


def prune_epoch_checkpoints(out_dir: Path, keep_last: int) -> list[Path]:
    """Delete all but the newest `keep_last` ``epochNNN.pt`` files; return those kept.

    WHY THIS EXISTS. The training loop wrote one checkpoint per epoch unconditionally,
    so a 50-epoch run left 50 files (~190 MB) and a few ablation arms filled
    `outputs/stage1/` with 16 GB. `/project` is not the constraint (measured 2.7 PB
    free); the constraint is that an unbounded pile of artefacts makes a PARTIAL run
    indistinguishable from a finished one -- after a `scancel` at epoch 36 the directory
    contained 37 checkpoints including a `last.pt`, and whether that `last.pt` was a
    finished artefact had to be established by hand before it was safe to resume.

    `keep_last=0` deletes every epoch file (keeping only `last.pt`/`best.pt`, which are
    written separately and are what the eval job reads).
    """
    if keep_last < 0:
        raise ValueError(f"keep_last must be >= 0, got {keep_last}")
    existing = sorted(out_dir.glob("epoch[0-9][0-9][0-9].pt"))
    doomed = existing if keep_last == 0 else existing[:-keep_last]
    for path in doomed:
        path.unlink(missing_ok=True)
    return existing[len(doomed):]


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters()
               if (p.requires_grad or not trainable_only))


def human_int(n: int) -> str:
    return f"{n/1e6:.2f}M" if n >= 1e6 else f"{n/1e3:.1f}k" if n >= 1e3 else str(n)


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
