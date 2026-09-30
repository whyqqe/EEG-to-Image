"""Shared path helpers for per-subject LOSO train/eval."""

from __future__ import annotations

from pathlib import Path


def subject_dir_suffix(subject: str) -> str:
    """Legacy folder: sub-08 -> sub08; others keep sub-01 style."""
    return "sub08" if subject == "sub-08" else subject


def distill_root(root: Path, subject: str, stage: int) -> Path:
    return root / f"outputs/atm_distill_s{stage}_{subject_dir_suffix(subject)}"


def resolve_stage_ckpt(root: Path, subject: str, stage: int) -> Path | None:
    ckpt_dir = distill_root(root, subject, stage) / "checkpoints"
    best = ckpt_dir / f"atm_stage{stage}_best.pt"
    last = ckpt_dir / f"atm_stage{stage}_last.pt"
    if best.is_file():
        return best
    if last.is_file():
        return last
    return None


def ensure_stage_best(root: Path, subject: str, stage: int) -> Path | None:
    """Copy last->best when training finished without meeting save gates."""
    ckpt_dir = distill_root(root, subject, stage) / "checkpoints"
    best = ckpt_dir / f"atm_stage{stage}_best.pt"
    last = ckpt_dir / f"atm_stage{stage}_last.pt"
    if best.is_file():
        return best
    if last.is_file():
        import shutil

        shutil.copy2(last, best)
        return best
    return None
