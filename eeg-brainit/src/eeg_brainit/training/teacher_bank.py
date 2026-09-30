"""Precomputed OpenCLIP teacher embedding bank for large-negative InfoNCE."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class TeacherEmbeddingBank:
    """Holds L2-normalized CLIP image teachers; samples negatives each step."""

    def __init__(self, teacher_dir: str | Path, device: torch.device | None = None) -> None:
        root = Path(teacher_dir)
        emb = np.load(root / "embeddings.npy")
        if emb.ndim != 2:
            raise ValueError(f"Expected (N,D) embeddings, got {emb.shape}")
        self.embeddings = F.normalize(
            torch.from_numpy(np.asarray(emb, dtype=np.float32)), dim=-1
        )
        self.device = device or torch.device("cpu")
        self.n = int(self.embeddings.shape[0])
        self.dim = int(self.embeddings.shape[1])
        print(f"[INFO] TeacherEmbeddingBank loaded n={self.n} dim={self.dim} from {root}")

    def to(self, device: torch.device) -> "TeacherEmbeddingBank":
        self.device = device
        # Keep full bank on CPU; move sampled slices to GPU to save VRAM.
        return self

    @torch.no_grad()
    def sample(
        self,
        k: int,
        exclude: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample k unique teacher rows, optionally excluding positive indices."""
        k = min(int(k), self.n)
        if exclude is None or exclude.numel() == 0:
            idx = torch.randperm(self.n, generator=generator)[:k]
        else:
            excl = set(int(x) for x in exclude.detach().cpu().tolist())
            # Rejection sampling is fine for small exclude sets.
            need = k
            chosen: list[int] = []
            # Cap attempts to avoid infinite loops on tiny banks.
            for _ in range(8):
                cand = torch.randperm(self.n, generator=generator)[: need + len(excl) + 8]
                for i in cand.tolist():
                    if i in excl:
                        continue
                    chosen.append(i)
                    if len(chosen) >= k:
                        break
                if len(chosen) >= k:
                    break
                need = k - len(chosen)
            if len(chosen) < k:
                # Fall back to allowing replacements if bank is tiny.
                extra = torch.randperm(self.n, generator=generator)[: k - len(chosen)]
                chosen.extend(extra.tolist())
            idx = torch.tensor(chosen[:k], dtype=torch.long)
        return self.embeddings[idx].to(self.device, non_blocking=True)
